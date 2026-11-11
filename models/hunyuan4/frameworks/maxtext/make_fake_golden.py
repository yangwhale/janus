"""造一个「合成 golden」：只有 data/batches 与 data/rank*.jsonl（格式与 GPU 抓数相同）和最小的 meta.json，token 随机。

    python make_fake_golden.py --out /tmp/hy4-fake-golden --steps 3 [--seq 4096 --vocab 120832 --ranks 8 --micro 2]

**不是 GPU golden**：只给 CPU 冒烟与 TPU 上「跑得通 / 装得下 / 实测显存峰值」用，结果不和任何 GPU 运行比
（meta.json 的 synthetic: true 标明）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def make_golden(d: Path, steps: int, seq: int, vocab: int, ranks: int = 8, micro: int = 2, seed: int = 1,
                fields=("tokens", "labels", "loss_mask", "position_ids")):
    rng = np.random.default_rng(seed)
    keep = tuple(fields)
    (d / "data" / "batches").mkdir(parents=True, exist_ok=True)
    for r in range(ranks):
        off = 0
        with open(d / "data" / "batches" / f"rank{r}.bin", "wb") as fb, \
                open(d / "data" / "batches" / f"rank{r}.jsonl", "w") as fi, open(d / "data" / f"rank{r}.jsonl", "w") as ff:
            for s in range(1, steps + 1):
                for m in range(micro):
                    sample = rng.integers(0, vocab, seq + 1)
                    fields = {"tokens": sample[:-1].reshape(1, seq).astype(np.int64),
                              "labels": sample[1:].reshape(1, seq).astype(np.int64),
                              "loss_mask": np.ones((1, seq), np.float32),
                              "position_ids": np.arange(seq, dtype=np.int64).reshape(1, seq)}
                    fields = {k: v for k, v in fields.items() if k in keep}
                    rec, fp = {"step": s, "micro": m, "fields": {}}, {"step": s, "micro": m}
                    for k, a in fields.items():
                        raw = a.tobytes()
                        h = hashlib.sha256(raw).hexdigest()
                        rec["fields"][k] = {"offset": off, "dtype": str(a.dtype), "shape": list(a.shape), "sha256": h}
                        fp[k] = h
                        fb.write(raw)
                        off += len(raw)
                    fp["nonpad"] = seq
                    fi.write(json.dumps(rec) + "\n")
                    ff.write(json.dumps(fp) + "\n")
    (d / "meta.json").write_text(json.dumps({"run_name": d.name, "synthetic": True,
                                             "data": {"manifest": "synthetic random tokens", "manifest_sha256": None,
                                                      "seed": seed}}))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--vocab", type=int, default=120832)
    p.add_argument("--ranks", type=int, default=8)
    p.add_argument("--micro", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--fields", default="tokens,labels,loss_mask,position_ids",
                   help="存哪些字段（混元 4 的 GPU golden 只存 tokens,labels）")
    a = p.parse_args(argv)
    make_golden(Path(a.out), a.steps, a.seq, a.vocab, a.ranks, a.micro, a.seed, a.fields.split(","))
    print(f"写了合成 golden {a.out}（{a.ranks} rank × {a.micro} micro × {a.steps} 步，seq {a.seq}）")


if __name__ == "__main__":
    main()
