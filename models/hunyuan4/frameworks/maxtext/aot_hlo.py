"""CPU 上为 TPU 拓扑编一次训练步（只编不跑，0 张卡），存优化后 HLO 与编译期显存分析。给 04-tpu-self 跑前的离线筛选用。

和 TPU 上 train_hy4.py 编的是同一个程序：同一套配置（mt_common.make_config：对齐 yml + 与模型 yml 冲突的键 + 阶段 + 覆盖），
同一个 jit（train_utils.jit_train_and_eval_step(train.train_step)），在同一个 mesh 与 logical_axis_rules 下 lower。
区别只有两处：设备换成 MaxText train_compile 的拓扑描述（compile_topology，不要真卡），训练状态与 batch 换成同形状、同切分的
抽象值（train_compile.get_shaped_inputs），不载检查点。没有照抄 train_compile 的 main：它会改 PRNG 实现、往 LIBTPU_INIT_ARGS 里
加开关，TPU 上的训练不这样做。

在 MaxText 环境里跑（要 jax、libtpu、maxtext；libtpu 只用它的编译器）。开关照 TPU 运行那样用环境变量给：

  JAX_PLATFORMS=cpu ALLOW_MULTIPLE_LIBTPU_LOAD=1 TPU_SKIP_MDS_QUERY=1 HY4_YML=<对齐 yml> \
  XLA_FLAGS="<被测的 XLA 开关> --xla_dump_to=<目录>/xla --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train_step.*" \
  LIBTPU_INIT_ARGS="<被测的 libtpu 开关>" \
    python aot_hlo.py --stage mini --topology v5p-16 --steps 1000 --out <目录> [MaxText key=value …]

输出 <目录>/aot.json（版本、配置、编译秒数、memory_analysis、进程峰值内存）；HLO 在 <目录>/xla/（XLA 自己 dump 的）。
编译器拒绝（例如 HBM 放不下）时 aot.json 记 error，退出码 2。
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FIELDS = (
    "argument_size_in_bytes",
    "output_size_in_bytes",
    "alias_size_in_bytes",
    "temp_size_in_bytes",
    "generated_code_size_in_bytes",
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True)
    p.add_argument(
        "--topology",
        required=True,
        help="MaxText compile_topology，如 v5p-16（8 片 v5p = 8 个设备）、v5p-32（16 个设备）",
    )
    p.add_argument("--num-slices", type=int, default=1)
    p.add_argument("--steps", type=int, default=1000, help="与 TPU 运行相同（学习率表按它建）")
    p.add_argument("--out", required=True)
    p.add_argument("maxtext_args", nargs="*")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec: dict = {
        "stage": a.stage,
        "topology": a.topology,
        "num_slices": a.num_slices,
        "steps": a.steps,
        "maxtext_args": a.maxtext_args,
        "yml": os.environ.get("HY4_YML"),
        "env": {k: os.environ.get(k) for k in ("XLA_FLAGS", "LIBTPU_INIT_ARGS", "JAX_PLATFORMS")},
    }
    t0 = time.time()
    try:
        import jax
        import mt_common
        from maxtext.trainers.pre_train import train, train_compile
        from maxtext.utils import sharding, train_utils

        try:
            from importlib.metadata import version

            rec["versions"] = {k: version(k) for k in ("jax", "jaxlib", "libtpu", "flax", "optax")}
        except Exception as e:  # noqa: BLE001 — 只是记录
            rec["versions"] = {"error": repr(e)}
        overrides = [
            f"run_name=aot-{a.stage}",
            "base_output_directory=/tmp/aot-hlo",
            f"steps={a.steps}",
            f"learning_rate_schedule_steps={a.steps}",
            f"compile_topology={a.topology}",
            f"compile_topology_num_slices={a.num_slices}",
        ]
        config = mt_common.make_config(a.stage, overrides + list(a.maxtext_args))
        rec["config_mismatch_vs_yml"] = [list(map(str, x)) for x in mt_common.effective_mismatch(config)]
        mesh = train_compile.get_topology_mesh(config)
        rec["devices"] = int(mesh.devices.size)
        rec["device_kind"] = mesh.devices.flat[0].device_kind
        rec["mesh"] = dict(zip(mesh.axis_names, map(int, mesh.devices.shape)))
        shaped_args, _, state_mesh_shardings, _, graphdef = train_compile.get_shaped_inputs(mesh, config)
        abstract_state, shaped_batch = shaped_args[0], shaped_args[1]
        rec["batch"] = {k: [list(v.shape), str(v.dtype)] for k, v in sorted(shaped_batch.items())}
        params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(
            config, state_mesh_shardings
        )
        p_train, _ = train_utils.jit_train_and_eval_step(
            config,
            graphdef,
            mesh,
            abstract_state,
            state_mesh_shardings,
            train.train_step,
            params_shardings=params_shardings,
        )
        with jax.set_mesh(mesh), train.logical_axis_rules(config.logical_axis_rules):
            lowered = p_train.lower(abstract_state, shaped_batch)
        t1 = time.time()
        compiled = lowered.compile()
        rec["lower_seconds"] = round(t1 - t0, 1)
        rec["compile_seconds"] = round(time.time() - t1, 1)
        ma = compiled.memory_analysis()
        rec["memory_analysis"] = {k: getattr(ma, k, None) for k in FIELDS}
        rc = 0
    except Exception as e:  # noqa: BLE001 — 编不出来也要留下原因，由调用方判「无法离线判断」
        rec["error"] = f"{type(e).__name__}: {e}"[:4000]
        rec["traceback"] = traceback.format_exc()[-6000:]
        rc = 2
    rec["wall_seconds"] = round(time.time() - t0, 1)
    rec["max_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    (out / "aot.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(
        "[aot_hlo] "
        + json.dumps(
            {k: rec.get(k) for k in ("devices", "compile_seconds", "memory_analysis", "max_rss_gib", "error")},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
