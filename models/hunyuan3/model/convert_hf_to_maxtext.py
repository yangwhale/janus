"""HF（截 4 层的混元 3）→ MaxText 参数检查点。按 models/hunyuan3/model/name_map.yaml 的 maxtext 一栏逐张量搬运。

用法（CPU 即可，约 60 GB 内存；不占 TPU）：
    JAX_PLATFORMS=cpu python convert_hf_to_maxtext.py --hf /data/hy3-4l --out /data/hy3-4l-maxtext
    # 产物：/data/hy3-4l-maxtext/0/items（训练时 load_parameters_path 指向它）
    python verify_ckpt.py maxtext --ckpt /data/hy3-4l-maxtext/0/items --hf /data/hy3-4l

做了什么：
1. 用 hy3-4layers.yml 建 MaxText 抽象模型，列出全部变量（params 与 MoEBiasVar 两个集合）及形状；
   和 name_map 推出的叶子形状逐个对：MaxText 有、映射表没有的叶子直接报错（防止漏搬、静默留在初始值）。
2. 每个叶子分配 FP32 数组，按映射表把每个 HF 张量转置 / 改形状后写进对应的一片（层号落在栈轴、专家号落在专家轴）；
   BF16 → FP32 是精确的。写完核对每个叶子的每个元素都被写过恰好一次。
3. 用 MaxText 自己的 save_weights_to_checkpoint 存成 Orbax 检查点（与上游 to_maxtext.py 同一个存法）。

只有转置、改形状、拼接，没有算术；verify_ckpt.py maxtext 反向取回后逐张量对 tensor_sha256.json。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "frameworks" / "maxtext"))   # mt_common（MaxText 配置）
import mt_common  # noqa: E402

from janus.dump.name_map import apply_ops, leaf_index  # noqa: E402
from janus.store import read_safetensors_tensor  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf", required=True, help="HF 目录（含 model.safetensors.index.json、tensor_sha256.json）")
    p.add_argument("--out", required=True, help="输出目录（本地或 gs://）；检查点写在 <out>/0/items")
    p.add_argument("maxtext_args", nargs="*", help="追加给 MaxText 配置的 key=value（与训练时一致，通常不用给）")
    a = p.parse_args(argv)

    import json
    hf_dir = Path(a.hf)
    wm = json.loads((hf_dir / "model.safetensors.index.json").read_text())["weight_map"]
    config = mt_common.make_config(["hardware=cpu", "skip_jax_distributed_system=True", "run_name=convert",
                                    "attention=dot_product", *a.maxtext_args])
    # 维度取 MaxText 实际生效的配置；正式转换（不带额外参数）要求与 name_map.yaml 完全一致
    nm = mt_common.name_map_for(config, strict=not a.maxtext_args)
    abstract = mt_common.abstract_param_tree(config)
    want = nm.leaf_shapes()
    only_mt = sorted(set(abstract) - set(want))
    only_map = sorted(set(want) - set(abstract))
    bad = [(k, tuple(abstract[k].shape), want[k]) for k in sorted(set(abstract) & set(want)) if tuple(abstract[k].shape) != want[k]]
    if only_mt or only_map or bad:
        raise SystemExit(f"MaxText 抽象模型与 name_map 不一致：只在 MaxText {only_mt}；只在 name_map {only_map}；形状不符 {bad}")
    print(f"MaxText 变量 {len(abstract)} 个，与 name_map 推出的叶子逐个一致", flush=True)

    leaves: dict[str, np.ndarray] = {}
    written: dict[str, np.ndarray] = {}
    for path, sds in abstract.items():
        if np.dtype(sds.dtype) != np.float32:
            raise SystemExit(f"{path} 的 dtype 是 {sds.dtype}，期望 float32（weight_dtype=float32）")
        leaves[path] = np.zeros(sds.shape, np.float32)
        written[path] = np.zeros(sds.shape, np.uint8)

    t0, n = time.time(), 0
    for ent in nm.entries:
        for pc in ent.pieces:
            x, _ = read_safetensors_tensor(hf_dir / wm[pc.hf], pc.hf)
            y = apply_ops(x.astype(np.float32), pc.ops)        # BF16 / FP32 → FP32，精确
            idx = leaf_index(pc.index, leaves[pc.path].ndim)
            if leaves[pc.path][idx].shape != y.shape:
                raise SystemExit(f"{pc.hf} → {pc.path}{pc.index}：形状 {y.shape} 对不上 {leaves[pc.path][idx].shape}")
            leaves[pc.path][idx] = y
            written[pc.path][idx] += 1
            n += 1
    for path, w in written.items():
        if not np.all(w == 1):
            raise SystemExit(f"{path}：有元素没写或写了不止一次（min {w.min()}，max {w.max()}）")
    print(f"搬运 {n} 个 HF 张量 → {len(leaves)} 个叶子，每个元素恰好写一次（{time.time() - t0:.0f} s）", flush=True)
    del written

    from maxtext.checkpoint_conversion.utils.utils import save_weights_to_checkpoint
    tree = mt_common.nest(leaves)                              # {'params': {...}, 'MoEBiasVar': {...}}
    t0 = time.time()
    save_weights_to_checkpoint(a.out, tree, device_count=1, use_ocdbt=config.checkpoint_storage_use_ocdbt,
                               use_zarr3=config.checkpoint_storage_use_zarr3)
    print(f"已存 {a.out}/0/items（{time.time() - t0:.0f} s）")


if __name__ == "__main__":
    main()
