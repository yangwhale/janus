"""CPU 上为 TPU 拓扑编一次混元 3 训练步（只编不跑，0 张卡），HLO 由 XLA 自己 dump。

做法同 models/hunyuan4/frameworks/maxtext/aot_hlo.py，配置换成混元 3 的 mt_common.make_config
（对齐 yml + mtp_num_layers=0 + 命令行覆盖）。在 MaxText 镜像里跑：

  JAX_PLATFORMS=cpu ALLOW_MULTIPLE_LIBTPU_LOAD=1 TPU_SKIP_MDS_QUERY=1 HY3_YML=<对齐 yml> PYTHONPATH=<janus>:<maxtext/src> \
  XLA_FLAGS="--xla_dump_to=<目录>/xla --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train_step.*" \
    python aot_hlo.py <compile_topology，如 v5p-16、tpu7x-8> <steps> <目录> [MaxText key=value …]

输出 <目录>/aot.json（版本、mesh、batch 形状、编译秒数、进程峰值内存；编不出来时记 error，退出码 2）。
"""

import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    topo, steps, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    extra = sys.argv[4:]
    out.mkdir(parents=True, exist_ok=True)
    rec = {"topology": topo, "steps": steps, "extra": extra, "yml": os.environ.get("HY3_YML")}
    t0 = time.time()
    try:
        from importlib.metadata import version

        import jax
        import mt_common
        from maxtext.trainers.pre_train import train, train_compile
        from maxtext.utils import sharding, train_utils

        rec["versions"] = {k: version(k) for k in ("jax", "jaxlib", "libtpu", "flax", "optax")}
        overrides = [
            "run_name=aot-hy3",
            "base_output_directory=/tmp/aot-hlo",
            f"steps={steps}",
            f"learning_rate_schedule_steps={steps}",
            f"compile_topology={topo}",
            "compile_topology_num_slices=1",
        ]
        config = mt_common.make_config(overrides + extra, yml=os.environ.get("HY3_YML") or mt_common.HY3_YML)
        mesh = train_compile.get_topology_mesh(config)
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
        lowered.compile()
        rec["compile_seconds"] = round(time.time() - t1, 1)
        rc = 0
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:4000]
        rec["traceback"] = traceback.format_exc()[-6000:]
        rc = 2
    rec["wall_seconds"] = round(time.time() - t0, 1)
    rec["max_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    (out / "aot.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(
        "[aot_hy3]",
        json.dumps({k: rec.get(k) for k in ("mesh", "compile_seconds", "max_rss_gib", "error")}),
        flush=True,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
