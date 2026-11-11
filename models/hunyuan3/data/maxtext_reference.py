"""MaxText 侧参照：与 megatron_reference.py 一一对应，用 MaxText mmap_npy 管线里同样的数据源和变换取样本。

只用 CPU，在装了 MaxText 的 venv 里跑。数据源与变换就是训练管线用的那两个
（_mmap_datasource.create_mmap_npy_source → MegatronNpyDataSource；input_pipeline_utils.MegatronSplitInputsTargets），
只是不经 Grain 批次和设备分片——那部分已由 maxtext_dump.py 走完整训练入口对拍过。

  epochs     按同样的样本数现建索引（缓存未命中时 MaxText 运行时自己建），存索引 sha256，取 Megatron 参照里同一组流下标的样本
  synthetic  读 megatron_reference.py 写的合成 .bin/.idx，在 MaxText 能表达的开关组合下取全部样本

用法：
  python maxtext_reference.py epochs --ref <megatron_reference 的输出目录> --data <前缀> --npy-dir <目录> --out <目录>
  python maxtext_reference.py synthetic --ref <megatron_reference synthetic 的输出目录> --npy-dir <目录> --out <目录>
产出 <out>/ref.json + samples.npz（字段已换成 GPU 的四个字段 + 注意力 mask，见 to_megatron_fields）。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def to_megatron_fields(x: dict) -> dict:
    """MaxText 六个字段 → Megatron 四个字段 + 注意力 mask（True = 屏蔽，与 Megatron create_attention_mask 一致）。
    MaxText 的注意力：因果且同一 inputs_segmentation 段内可见。"""
    seg = x["inputs_segmentation"]
    L = seg.shape[0]
    causal = np.tril(np.ones((L, L), dtype=bool))
    visible = causal & (seg[:, None] == seg[None, :])
    return {"tokens": x["inputs"].astype(np.int64), "labels": x["targets"].astype(np.int64),
            "loss_mask": (x["targets_segmentation"] != 0).astype(np.float32),
            "position_ids": x["inputs_position"].astype(np.int64), "attention_mask": ~visible}


def source(spec, eod, seq, n, seed, split):
    from maxtext.input_pipeline._mmap_datasource import create_mmap_npy_source
    return create_mmap_npy_source(spec, eod, seq, False, num_samples=n, seed=seed, split=split, split_index=0)


def transform(eod, ram, eml):
    from maxtext.input_pipeline.input_pipeline_utils import MegatronSplitInputsTargets
    return MegatronSplitInputsTargets(eod_id=eod, reset_attention_mask=ram, eod_mask_loss=eml, min_segment_length=0)


def index_files(npy_dir):
    out = {}
    for f in sorted(os.listdir(npy_dir)):
        if f.endswith(".npy"):
            p = os.path.join(npy_dir, f)
            a = np.load(p, mmap_mode="r")
            out[f] = {"sha256_file": sha_file(p), "dtype": str(a.dtype), "shape": list(a.shape)}
    return out


def cmd_epochs(a):
    ref = json.load(open(Path(a.ref) / "ref.json"))
    d = ref["description"]
    n, seq, seed, split = ref["num_samples"], d["sequence_length"], d["random_seed"], d["split"]
    npy = os.path.join(a.npy_dir, f"n{n}")
    os.makedirs(npy, exist_ok=True)
    ds = source(f"{npy}|{a.data}", ref["eod"], seq, n, seed, split)
    tf = transform(ref["eod"], False, False)
    arrs = {}
    for i in ref["picks"]:
        x = tf.map(dict(ds[i]))
        for k, v in to_megatron_fields(x).items():
            if k != "attention_mask":
                arrs[f"{i}.{k}"] = v
    np.savez(Path(a.out) / "samples.npz", **arrs)
    out = {"mode": "epochs", "num_samples": n, "len": len(ds), "index": index_files(npy), "picks": ref["picks"]}
    json.dump(out, open(Path(a.out) / "ref.json", "w"), indent=1)
    print(json.dumps({"num_samples": n, "len": len(ds), "index": sorted(out["index"])}))


def cmd_synthetic(a):
    ref = json.load(open(Path(a.ref) / "ref.json"))
    seq, eod, n = ref["seq"], ref["eod"], ref["num_samples"]
    npy = os.path.join(a.npy_dir, "syn")
    os.makedirs(npy, exist_ok=True)
    ds = source(f"{npy}|{ref['prefix']}", eod, seq, n, 1234, "1,0,0")
    arrs, combos = {}, []
    # MaxText 只有两个开关：reset_attention_mask 同时管注意力与位置重置，eod_mask_loss 管 loss
    for ram in (False, True):
        for eml in (False, True):
            name = f"ram{int(ram)}_eml{int(eml)}"
            tf = transform(eod, ram, eml)
            for i in range(len(ds)):
                for k, v in to_megatron_fields(tf.map(dict(ds[i]))).items():
                    arrs[f"{name}.{i}.{k}"] = v
            combos.append({"name": name, "reset_attention_mask": ram, "eod_mask_loss": eml, "len": len(ds)})
    np.savez(Path(a.out) / "samples.npz", **arrs)
    json.dump({"mode": "synthetic", "combos": combos, "index": index_files(npy)}, open(Path(a.out) / "ref.json", "w"), indent=1)
    print(json.dumps({"combos": len(combos), "len": len(ds)}))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("epochs", "synthetic"):
        q = sub.add_parser(name)
        q.add_argument("--ref", required=True)
        q.add_argument("--npy-dir", required=True)
        q.add_argument("--out", required=True)
        if name == "epochs":
            q.add_argument("--data", required=True)
    a = p.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    {"epochs": cmd_epochs, "synthetic": cmd_synthetic}[a.cmd](a)


if __name__ == "__main__":
    main()
