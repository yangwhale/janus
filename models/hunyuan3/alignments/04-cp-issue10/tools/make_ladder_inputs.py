#!/usr/bin/env python3
"""实践 04 阶梯实验的两份输入：所有臂共用同一份初始权重、同一份数据，按同一顺序喂。

  hf    16 层随机初始化混元 3 的 HF 目录（ladder/name_map_smoke16.yaml 展开的全部张量）。矩阵 N(0, 0.006²)（HF 原版 config 的
        initializer_range）舍成 BF16，RMSNorm 权重 1，路由偏置 0（FP32）；写 model.safetensors.index.json、tensor_sha256.json、
        config.json（真实混元 3 的 config 只改尺寸）。之后照常 convert_hf_to_maxtext.py 转成 MaxText 检查点；CPU FP64 参照读这份 HF。
  data  回放目录（golden 同格式：data/batches/rank<r>.{bin,jsonl}、data/rank<r>.jsonl 指纹、meta.json），train_hy3.py 按
        replay.py 原样喂。FineWeb-Edu parquet 按行号顺序取文档，HF 分词器切词，文档之间插 EOS，首尾相接成一条长流，
        每 4097 个词元切一条：tokens = 前 4096，labels = 后 4096；position_ids 0…4095，loss_mask 全 1（整条一段，不按文档切注意力，
        同 golden 约定）。第 s 步 rank r 拿第 (s−1)·R + r 条。

  python make_ladder_inputs.py hf --name-map ladder/name_map_smoke16.yaml --config-from <真实混元 3 config.json> --out <HF 目录> [--seed 0]
  python make_ladder_inputs.py data --parquet <fineweb parquet> --tokenizer <tokenizer.json 所在目录> --steps 3000 --ranks 8 --out <回放目录>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4]))
from janus.dump.name_map import NameMap  # noqa: E402
from janus.store import write_safetensors  # noqa: E402

SEQ = 4096


def make_hf(a) -> int:
    import ml_dtypes
    nm = NameMap.load(a.name_map)
    dims = nm.dims
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest, wmap, shard, n_shard, size = {}, {}, {}, 1, 0

    def flush():
        nonlocal shard, n_shard, size
        if shard:
            name = f"model-{n_shard:05d}.safetensors"
            write_safetensors(out / name, shard)
            for k in shard:
                wmap[k] = name
            n_shard, shard, size = n_shard + 1, {}, 0

    n_params = 0
    for ent in nm.entries:
        for h, shp in zip(ent.hf, nm.hf_shapes(ent)):
            if h.endswith("expert_bias"):
                x = np.zeros(shp, np.float32)
            elif "norm" in h:
                x = np.ones(shp, ml_dtypes.bfloat16)
            else:
                x = (rng.standard_normal(shp, dtype=np.float32) * a.std).astype(ml_dtypes.bfloat16)
            manifest[h] = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
            shard[h] = x
            size += x.nbytes
            n_params += x.size
            if size > 2 ** 31:
                flush()
    flush()
    (out / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict(sorted(wmap.items()))}, indent=1))
    (out / "tensor_sha256.json").write_text(json.dumps(dict(sorted(manifest.items())), indent=1))
    cfg = json.loads(Path(a.config_from).read_text())
    cfg.update({"num_hidden_layers": int(dims["layers"]), "hidden_size": int(dims["hidden"]), "intermediate_size": int(dims["ffn"]),
                "num_attention_heads": int(dims["heads"]), "num_key_value_heads": int(dims["kv_heads"]), "head_dim": int(dims["head_dim"]),
                "moe_intermediate_size": int(dims["moe_ffn"]), "expert_hidden_dim": int(dims["moe_ffn"]),
                "num_experts": int(dims["experts"]), "num_experts_per_tok": int(dims["topk"]), "vocab_size": int(dims["vocab"]),
                "first_k_dense_replace": int(dims["first_dense"]), "initializer_range": a.std})
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "random_init.json").write_text(json.dumps({"seed": a.seed, "std": a.std, "norms": 1.0, "expert_bias": 0.0,
                                                      "params": n_params, "name_map": str(a.name_map)}, indent=1))
    print(f"{len(manifest)} 个张量，{n_params:.3e} 个参数 → {out}")
    return 0


def make_data(a) -> int:
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(a.tokenizer) / "tokenizer.json"))
    eos = json.loads((Path(a.tokenizer) / "tokenizer_config.json").read_text()).get("eos_token")
    eos_id = tok.token_to_id(eos if isinstance(eos, str) else eos["content"])
    need = a.steps * a.ranks * (SEQ + 1)
    pf = pq.ParquetFile(a.parquet)
    stream: list[int] = []
    docs = 0
    for rg in range(pf.num_row_groups):
        texts = pf.read_row_group(rg, columns=[a.column]).column(0).to_pylist()
        for enc in tok.encode_batch(texts, add_special_tokens=False):
            stream.extend(enc.ids)
            stream.append(eos_id)
            docs += 1
        if len(stream) >= need:
            break
    if len(stream) < need:
        raise SystemExit(f"分片只有 {len(stream)} 个词元，不够 {need}")
    arr = np.asarray(stream[:need], np.int64).reshape(a.steps * a.ranks, SEQ + 1)
    d = Path(a.out)
    (d / "data" / "batches").mkdir(parents=True, exist_ok=True)
    pos = np.arange(SEQ, dtype=np.int64).reshape(1, SEQ)
    ones = np.ones((1, SEQ), np.float32)
    for r in range(a.ranks):
        off = 0
        with open(d / "data" / "batches" / f"rank{r}.bin", "wb") as fb, \
                open(d / "data" / "batches" / f"rank{r}.jsonl", "w") as fi, open(d / "data" / f"rank{r}.jsonl", "w") as ff:
            for s in range(1, a.steps + 1):
                row = arr[(s - 1) * a.ranks + r]
                fields = {"tokens": row[:-1].reshape(1, SEQ), "labels": row[1:].reshape(1, SEQ), "loss_mask": ones, "position_ids": pos}
                rec, fp = {"step": s, "micro": 0, "fields": {}}, {"step": s, "micro": 0}
                for k, x in fields.items():
                    raw = np.ascontiguousarray(x).tobytes()
                    h = hashlib.sha256(raw).hexdigest()
                    rec["fields"][k] = {"offset": off, "dtype": str(x.dtype), "shape": list(x.shape), "sha256": h}
                    fp[k] = h
                    fb.write(raw)
                    off += len(raw)
                fp["nonpad"] = SEQ
                fi.write(json.dumps(rec) + "\n")
                ff.write(json.dumps(fp) + "\n")
    man = hashlib.sha256(b"".join((d / "data" / "batches" / f"rank{r}.jsonl").read_bytes() for r in range(a.ranks))).hexdigest()
    (d / "meta.json").write_text(json.dumps({
        "run_name": d.name, "kind": "replay-data（实践 04 阶梯实验，不是 GPU golden）",
        "data": {"manifest": "data/batches", "manifest_sha256": man,
                 "source": {"parquet": Path(a.parquet).name, "column": a.column, "docs_used": docs, "tokenizer": "混元 3 HF 分词器",
                            "eos_id": eos_id, "rule": "按行号顺序取文档、文档间插 EOS、首尾相接、每 4097 个词元切一条"},
                 "steps": a.steps, "ranks": a.ranks, "micro": 1, "seq": SEQ}}, ensure_ascii=False, indent=1))
    print(f"{a.steps} 步 × {a.ranks} 条 × {SEQ}，用了 {docs} 篇文档 → {d}（manifest {man[:12]}）")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hf")
    h.add_argument("--name-map", required=True)
    h.add_argument("--config-from", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--std", type=float, default=0.006)
    d = sub.add_parser("data")
    d.add_argument("--parquet", required=True)
    d.add_argument("--column", default="text")
    d.add_argument("--tokenizer", required=True)
    d.add_argument("--steps", type=int, default=3000)
    d.add_argument("--ranks", type=int, default=8)
    d.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return make_hf(a) if a.cmd == "hf" else make_data(a)


if __name__ == "__main__":
    sys.exit(main())
