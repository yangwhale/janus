"""Megatron 侧参照：用 Megatron-Bridge / Megatron-LM 自己的 GPTDataset 现建索引、取样本，给原生加载器对拍补 GPU 训练没跑到的情况。

只用 CPU（CUDA_VISIBLE_DEVICES= ），在装了 Megatron-Bridge 的环境里跑（GPU Pod 的 /opt/venv）。

两个子命令：
  epochs     对同一份数据按给定样本数建索引（跨 epoch、最后一个 epoch 单独打乱），存三个索引的 sha256，
             并取一组流下标上的样本（含跨 epoch 边界的那个样本）
  synthetic  写一个小的合成 .bin/.idx（短文档、恰好 seq+1 长的文档、含 pad id 的文档），
             在 reset_attention_mask / reset_position_ids / eod_mask_loss 的 8 种组合下取全部样本（含注意力 mask）

用法：
  python megatron_reference.py epochs --data <前缀> --hf <分词器目录> --num-samples 400000 --cache /tmp/g62/meg-cache --out <目录>
  python megatron_reference.py synthetic --hf <分词器目录> --work <目录> --out <目录>
产出 <out>/ref.json + samples.npz。
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np

FIELDS = ("tokens", "labels", "loss_mask", "position_ids")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def make_config(data, hf, seq, seed, split, cache, ram=False, rpi=False, eml=False, attn=False):
    """照 models/hunyuan3/frameworks/megatron/train_hy3.py 的写法建数据配置（_pretrain_common 的默认 + 同样的覆盖）。"""
    from megatron.bridge.recipes.common import _pretrain_common
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
    cfg = _pretrain_common()
    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = hf
    tok = build_tokenizer(cfg.tokenizer)
    d = cfg.dataset
    d.seq_length = seq
    d.blend = ([data], None)
    d.split = split
    d.random_seed = seed
    d.path_to_cache = cache
    d.reset_attention_mask, d.reset_position_ids, d.eod_mask_loss = ram, rpi, eml
    d.create_attention_mask = attn
    d.tokenizer = tok
    d.finalize()
    return d, tok


def build(cfg, n):
    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
    from megatron.core.datasets.gpt_dataset import GPTDataset
    train, _, _ = BlendedMegatronDatasetBuilder(GPTDataset, [n, 0, 0], lambda: True, cfg).build()
    return train


def sample(ds, i, attn=False):
    s = ds[i]
    out = {k: s[k].numpy() for k in FIELDS}
    if attn and s.get("attention_mask") is not None:
        out["attention_mask"] = s["attention_mask"].numpy()[0]   # [L, L]，True = 屏蔽
    return out


def cmd_epochs(a):
    cfg, tok = make_config(a.data, a.hf, a.seq, a.seed, a.split, a.cache)
    ds = build(cfg, a.num_samples)
    # 索引文件：GPTDataset 把三件套写进 path_to_cache，文件名前缀是 unique_description_hash
    h = ds.unique_description_hash
    files = {k: os.path.join(a.cache, f"{h}-GPTDataset-train-{k}.npy") for k in ("document_index", "sample_index", "shuffle_index")}
    idx = {k: {"sha256_file": sha_file(p), "dtype": str(np.load(p, mmap_mode="r").dtype),
               "shape": list(np.load(p, mmap_mode="r").shape)} for k, p in files.items()}
    doc, smp, shf = (np.load(files[k], mmap_mode="r") for k in ("document_index", "sample_index", "shuffle_index"))
    n_docs_epoch = len(ds.indices)
    n_epochs = len(doc) // n_docs_epoch
    # 跨 epoch 边界的样本：sample_index 的文档位置跨过 n_docs_epoch × e
    cross = []
    for e in range(1, n_epochs):
        j = int(np.searchsorted(smp[:, 0], e * n_docs_epoch, side="right")) - 1
        cross.append({"epoch_boundary": e, "sample_id": j, "stream_index": int(np.nonzero(np.asarray(shf) == j)[0][0])})
    # 最后一个 epoch 是否单独打乱：shuffle_index 的前 num_samples_sans_final 个只在 [0, num_samples_sans_final) 里取
    tokens_per_epoch = int(np.asarray(ds.dataset.sequence_lengths)[ds.indices].sum())
    sans_final = ((n_epochs - 1) * tokens_per_epoch - 1) // a.seq if n_epochs > 1 else None
    separate = bool(sans_final is not None and sans_final < len(shf) and int(np.max(shf[:sans_final])) < sans_final)
    n = len(ds)
    picks = sorted({0, 1, 2, n // 2, n - 2, n - 1, *[c["stream_index"] for c in cross],
                    *[x for c in cross for x in (c["stream_index"] - 1, c["stream_index"] + 1) if 0 <= x < n],
                    *([int(a.num_samples) - 1, int(a.num_samples)] if a.num_samples < n else [])})
    arrs = {}
    for i in picks:
        for k, v in sample(ds, i).items():
            arrs[f"{i}.{k}"] = v
    np.savez(Path(a.out) / "samples.npz", **arrs)
    ref = {"mode": "epochs", "num_samples": a.num_samples, "len": n, "num_epochs": n_epochs, "docs_per_epoch": n_docs_epoch,
           "unique_description_hash": h, "description": json.loads(ds.unique_description), "index": idx,
           "cross_epoch": cross, "picks": picks, "eod": tok.eod, "pad": tok.pad, "tokens_per_epoch": tokens_per_epoch,
           "num_samples_sans_final": sans_final, "separate_final_epoch": separate}
    json.dump(ref, open(Path(a.out) / "ref.json", "w"), indent=1, default=str)
    print(json.dumps({k: ref[k] for k in ("num_samples", "len", "num_epochs", "cross_epoch", "separate_final_epoch")}))


def write_synthetic(prefix, eod, pad, seq):
    """几种边界：很短的文档、恰好 seq+1 长、恰好 seq 长、只有一个 EOD 的空文档、文档中间夹 pad id。"""
    import torch
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
    rng = np.random.RandomState(0)
    docs = []
    lens = [3, 1, seq + 1, seq, 2, 7, 1, 5, seq - 1, 4, 9, 2 * seq + 3, 6, 1, 3]
    for i, n in enumerate(lens):
        body = rng.randint(10, 1000, size=max(n - 1, 0)).astype(np.int32)
        if i == 5 and n > 3:
            body[1] = pad                                                # 文档中间夹一个 pad id
        docs.append(np.concatenate([body, [eod]]).astype(np.int32))    # 预处理 --append-eod 的样子
    b = IndexedDatasetBuilder(prefix + ".bin", dtype=np.int32)
    for d in docs:
        b.add_item(torch.from_numpy(d))
        b.end_document()
    b.finalize(prefix + ".idx")
    return [len(d) for d in docs]


def cmd_synthetic(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    _, tok = make_config(a.data or "x", a.hf, a.seq, a.seed, "1,0,0", str(work / "cache-tok"))
    prefix = str(work / "syn_text_document")
    lens = write_synthetic(prefix, tok.eod, tok.pad, a.seq)
    arrs, combos = {}, []
    for ram, rpi, eml in itertools.product([False, True], repeat=3):
        name = f"ram{int(ram)}_rpi{int(rpi)}_eml{int(eml)}"
        cfg, _ = make_config(prefix, a.hf, a.seq, a.seed, "1,0,0", str(work / f"cache-{name}"), ram, rpi, eml, attn=True)
        ds = build(cfg, a.num_samples)
        for i in range(len(ds)):
            for k, v in sample(ds, i, attn=True).items():
                arrs[f"{name}.{i}.{k}"] = v
        combos.append({"name": name, "reset_attention_mask": ram, "reset_position_ids": rpi, "eod_mask_loss": eml,
                       "len": len(ds), "hash": ds.unique_description_hash})
    np.savez(Path(a.out) / "samples.npz", **arrs)
    json.dump({"mode": "synthetic", "prefix": prefix, "doc_lens": lens, "seq": a.seq, "num_samples": a.num_samples,
               "eod": tok.eod, "pad": tok.pad, "combos": combos, "idx_sha256": sha_file(prefix + ".idx"),
               "bin_sha256": sha_file(prefix + ".bin")},
              open(Path(a.out) / "ref.json", "w"), indent=1)
    print(json.dumps({"combos": len(combos), "samples": combos[0]["len"], "eod": tok.eod, "pad": tok.pad}))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("epochs", "synthetic"):
        q = sub.add_parser(name)
        q.add_argument("--hf", required=True)
        q.add_argument("--out", required=True)
        q.add_argument("--seed", type=int, default=1234)
        q.add_argument("--data", default=None)
        if name == "epochs":
            q.add_argument("--seq", type=int, default=4096)
            q.add_argument("--split", default="9999,1,0")
            q.add_argument("--num-samples", type=int, required=True)
            q.add_argument("--cache", required=True)
        else:
            q.add_argument("--seq", type=int, default=16)
            q.add_argument("--num-samples", type=int, default=8)
            q.add_argument("--work", required=True)
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    {"epochs": cmd_epochs, "synthetic": cmd_synthetic}[a.cmd](a)


if __name__ == "__main__":
    main()
