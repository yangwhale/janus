"""第 1 步词嵌入梯度按行分组，各组离 CPU FP64 参照的 rel_L2（04-tpu-self/b1-grad-reduction.md 第四节的分组表）。

行的分组用这一步 16 条序列的 token（主输入是每行前 4096 个，MTP 输入是后移一位的 4096 个，两者合计出现次数）；
只取 FP64 参照里梯度不为 0 的行。「每个微批只在 1 张卡上」按纯数据并行 8 的分配：第 d 张卡第 m 个微批是第 d + 8m 条序列。

  A  每个微批里最多 1 张卡出现（BF16 all-reduce 加的是 0）
  B  其余行里合计出现 ≤ 300 次
  C  其余行里合计出现 ≥ 301 次（高频）

每个运行的梯度从第 1 步抓数的 Adam 一阶动量反推（同 tpu_self_compare.py）。

    python models/hunyuan4/analysis/emb_grad_row_groups.py --runs-root <运行目录的上一层> \\
        --fp64 <CPU FP64 第 1 步参照目录> --tokens <tokens.npy> --out <json> ts-g-a ts-dp8 ts-ebc-dp8 …
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tpu_self_compare import Dump, train_grad  # noqa: E402

NAME = "model.embed_tokens.weight"
SEQ, N_SEQ, N_CARD, HIGH = 4096, 16, 8, 300


def groups(tokens: np.ndarray, nonzero: np.ndarray) -> dict[str, np.ndarray]:
    vocab = nonzero.shape[0]
    per_seq = [
        np.bincount(tokens[s, :SEQ], minlength=vocab) + np.bincount(tokens[s, 1 : SEQ + 1], minlength=vocab)
        for s in range(N_SEQ)
    ]
    count = np.sum(per_seq, axis=0)
    cards = np.stack(
        [np.sum([per_seq[d + N_CARD * m] > 0 for d in range(N_CARD)], axis=0) for m in range(N_SEQ // N_CARD)]
    )
    one_card = nonzero & (cards.max(axis=0) <= 1)
    return {
        "A": one_card,
        "B": nonzero & ~one_card & (count <= HIGH),
        "C": nonzero & ~one_card & (count > HIGH),
        "all": nonzero,
    }


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs-root", required=True)
    p.add_argument("--fp64", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("runs", nargs="+")
    a = p.parse_args()
    from safetensors import safe_open

    fp64_file = os.path.join(a.fp64, "grads_hf.safetensors")
    ref = safe_open(fp64_file, "pt").get_tensor(NAME).double().numpy()
    tokens = np.asarray(np.load(a.tokens, mmap_mode="r")[:N_SEQ])
    g = groups(tokens, np.abs(ref).sum(axis=1) > 0)
    ref_sq = float((ref**2).sum())
    out = {
        "what": "第 1 步词嵌入梯度按行分组离 CPU FP64 的 rel_L2",
        "fp64_sha256": sha256(fp64_file),
        "tokens_sha256": sha256(a.tokens),
        "groups": {
            k: {"rows": int(v.sum()), "ref_norm_sq_share": float((ref[v] ** 2).sum()) / ref_sq} for k, v in g.items()
        },
        "runs": {},
    }
    for run in a.runs:
        d = os.path.join(a.runs_root, run)
        with open(os.path.join(d, "metrics.jsonl")) as f:
            raw_gn = json.loads(f.readline())["raw_grad_norm"]
        grad = train_grad(Dump(d), NAME, raw_gn)
        out["runs"][run] = {k: float(np.linalg.norm(grad[v] - ref[v]) / np.linalg.norm(ref[v])) for k, v in g.items()}
        print(run, {k: f"{x:.3e}" for k, x in out["runs"][run].items()})
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
