"""HF（截层的混元 4）→ MaxText 参数检查点，按阶段（A/B/C/D/4L）只取这个阶段用到的层。

用法（CPU 即可；不占 TPU）：
    JAX_PLATFORMS=cpu python convert_hf_to_maxtext.py --hf /data/hy4-4l --stage A --out /data/hy4-a-maxtext
    # 产物：/data/hy4-a-maxtext/0/items（训练时 --load 指向它）
    JAX_PLATFORMS=cpu python verify_ckpt.py maxtext --ckpt /data/hy4-a-maxtext/0/items --hf /data/hy4-4l --stage A

做了什么：
1. 用 frameworks/maxtext/hy4-4layers.yml + 阶段覆盖建 MaxText 抽象模型，列出全部变量（params 与 MoEBiasVar）及形状，
   参数量对 hy4_map.STAGE_PARAMS（HF 头里数出来的值）。
2. HF config.json 截到这个阶段的层数，取分支的映射与变换（hy4_map.mapping_and_hooks，见那里的说明），
   与抽象模型逐个对：MaxText 有、映射没有的叶子直接报错（防止漏搬、静默留在初始值）；映射里 HF 张量的集合必须
   正好等于这个阶段该有的 HF 张量（hy4_map.stage_tensor_names），多一个少一个都报错。
3. 每个叶子用分支的 _get_hf_loading_function 读、变换、按层堆叠（与 MaxText 自己的 to_maxtext.py 同一段代码），
   存 FP32（BF16 → FP32 精确）。RoPE 列去交错由 hy4-preview.yml 的 checkpoint_rope_interleave: true 打开。
4. 用 MaxText 的 save_weights_to_checkpoint 存成 Orbax 检查点。

只有转置、改形状、拆分、置换，没有算术；verify_ckpt.py maxtext 反向取回后逐张量对 tensor_sha256.json。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "frameworks" / "maxtext"))   # mt_common（MaxText 配置）
import mt_common  # noqa: E402
from hy4_map import STAGE_PARAMS, STAGES, iter_hf_sources, mapping_and_hooks, stage_hf_config, stage_tensor_names  # noqa: E402

from janus.store import read_safetensors_tensor  # noqa: E402


def build(hf_dir: Path, stage: str, extra: list[str], strict_params: bool = True):
    """建配置、抽象树、映射；核对覆盖与参数量。返回 (config, abstract{dash_key: sds}, mapping, hooks, hf_cfg, slash_of)。"""
    hf_cfg_full = json.loads((hf_dir / "config.json").read_text())
    layers, mtp = STAGES[stage]
    hf_cfg = stage_hf_config(hf_cfg_full, layers)
    config = mt_common.make_config(stage, ["hardware=cpu", "skip_jax_distributed_system=True", "run_name=convert",
                                           "attention=dot_product", *extra])
    abstract_slash = mt_common.abstract_param_tree(config)
    slash_of = {mt_common.dash_key(k): k for k in abstract_slash}
    abstract = {mt_common.dash_key(k): v for k, v in abstract_slash.items()}
    total = sum(int(np.prod(v.shape)) for v in abstract.values())
    want_total = STAGE_PARAMS.get(stage)
    print(f"阶段 {stage}：{layers} 层、MTP {mtp} 层；MaxText 变量 {len(abstract)} 个，参数 {total:,}（HF 头：{want_total:,}）",
          flush=True)
    if strict_params and want_total is not None and total != want_total:
        raise SystemExit(f"参数量 {total:,} ≠ {want_total:,}")
    mapping, hooks = mapping_and_hooks(hf_cfg, config)
    only_mt = sorted(set(abstract) - set(mapping))
    only_map = sorted(set(mapping) - set(abstract))
    if only_mt or only_map:
        raise SystemExit(f"MaxText 抽象模型与映射不一致：只在 MaxText {only_mt[:8]}；只在映射 {only_map[:8]}")
    return config, abstract, mapping, hooks, hf_cfg, slash_of


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf", required=True, help="HF 目录（含 config.json、model.safetensors.index.json、tensor_sha256.json）")
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--out", required=True, help="输出目录（本地或 gs://）；检查点写在 <out>/0/items")
    p.add_argument("--no-save", action="store_true", help="只搬运、核对覆盖，不存（试跑）")
    p.add_argument("maxtext_args", nargs="*", help="追加给 MaxText 配置的 key=value（与训练时一致，通常不用给）")
    a = p.parse_args(argv)

    hf_dir = Path(a.hf)
    wm = json.loads((hf_dir / "model.safetensors.index.json").read_text())["weight_map"]
    config, abstract, mapping, hooks, _, slash_of = build(hf_dir, a.stage, a.maxtext_args,
                                                          strict_params=not a.maxtext_args)
    layers, mtp = STAGES[a.stage]
    used = sorted({h for _, _, h in iter_hf_sources(mapping)})
    want = stage_tensor_names(wm, layers, mtp)
    if used != want:
        raise SystemExit(f"映射用到的 HF 张量与阶段 {a.stage} 该有的不一致：多 {sorted(set(used) - set(want))[:8]}；"
                         f"少 {sorted(set(want) - set(used))[:8]}")
    print(f"映射覆盖阶段 {a.stage} 的 HF 张量 {len(want)} 个（HF 目录共 {len(wm)} 个）", flush=True)

    from maxtext.checkpoint_conversion.to_maxtext import _get_hf_loading_function

    def getter(name):
        x, _ = read_safetensors_tensor(hf_dir / wm[name], name)
        return x.astype(np.float32)                              # BF16 / FP32 → FP32，精确

    leaves: dict[str, np.ndarray] = {}
    t0 = time.time()
    for k, sds in abstract.items():
        if np.dtype(sds.dtype) != np.float32:
            raise SystemExit(f"{k} 的 dtype 是 {sds.dtype}，期望 float32（weight_dtype=float32）")
        fn = _get_hf_loading_function(mapping[k], getter, hooks.get(k), tuple(sds.shape), config, k)
        y = np.asarray(fn(), dtype=np.float32)
        if tuple(y.shape) != tuple(sds.shape):
            raise SystemExit(f"{k}：搬出来的形状 {y.shape} ≠ 抽象模型 {sds.shape}")
        leaves[slash_of[k]] = y
    print(f"搬运完 {len(leaves)} 个叶子（{time.time() - t0:.0f} s）", flush=True)
    if a.no_save:
        print("--no-save：不存")
        return
    from maxtext.checkpoint_conversion.utils.utils import save_weights_to_checkpoint
    tree = mt_common.nest(leaves)                              # {'params': {...}, 'MoEBiasVar': {...}}
    t0 = time.time()
    save_weights_to_checkpoint(a.out, tree, device_count=1, use_ocdbt=config.checkpoint_storage_use_ocdbt,
                               use_zarr3=config.checkpoint_storage_use_zarr3)
    print(f"已存 {a.out}/0/items（{time.time() - t0:.0f} s）")


if __name__ == "__main__":
    main()
