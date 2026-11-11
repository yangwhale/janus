"""第 1 步训练梯度（从动量反推）的平方和按参数类别汇总，对 CPU FP64 参照的相对差；看「整体变大 / 变小」落在哪些参数上。

类别：把层号、专家号换成 N / E 后的参数名。另报每层注意力三个投影与 MLP 的合计。不判档。

    python models/hunyuan4/analysis/step1_grad_sq_by_class.py --root <运行目录根> --fp64 <FP64 参照目录> --out <json> <运行>...
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import tpu_self_compare as tsc  # noqa: E402

ATTN = ("self_attn.o_proj", "kv_a_proj", "kv_b_proj")


def cls(n):
    return re.sub(r"experts\.\d+\.", "experts.E.", re.sub(r"layers\.\d+\.", "layers.N.", n))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--fp64", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("runs", nargs="+")
    a = p.parse_args(argv)
    from safetensors import safe_open

    ref_f = safe_open(os.path.join(a.fp64, "grads_hf.safetensors"), "pt")
    ref = None
    out = {"what": "第 1 步训练梯度平方和对 CPU FP64 的相对差（run / fp64 − 1），按类别与按层；不判档", "runs": {}}
    for r in a.runs:
        d = tsc.Dump(os.path.join(a.root, r))
        m = json.loads(open(os.path.join(a.root, r, "metrics.jsonl")).readline())
        names = sorted(n for (k, n) in d.idx if k == "exp_avg" and n in ref_f.keys())
        if ref is None:
            ref = {n: float((ref_f.get_tensor(n).double().numpy() ** 2).sum()) for n in names}
            out["fp64_grad_norm"] = float(np.sqrt(sum(ref.values())))
        sq = {n: float((tsc.train_grad(d, n, m["raw_grad_norm"]) ** 2).sum()) for n in names}
        by = collections.defaultdict(lambda: [0.0, 0.0])
        for n in names:
            by[cls(n)][0] += ref[n]
            by[cls(n)][1] += sq[n]
        layers = {}
        for L in range(64):
            ks = [n for n in names if n.startswith(f"model.layers.{L}.")]
            if not ks:
                break
            at = [n for n in ks if any(s in n for s in ATTN)]
            ml = [n for n in ks if ".mlp." in n]
            layers[str(L)] = {
                "attn": sum(sq[n] for n in at) / sum(ref[n] for n in at) - 1,
                "mlp": sum(sq[n] for n in ml) / sum(ref[n] for n in ml) - 1,
            }
        out["runs"][r] = {
            "raw_grad_norm": m["raw_grad_norm"],
            "sqrt_sum_sq": float(np.sqrt(sum(sq.values()))),
            "by_class": {k: v[1] / v[0] - 1 for k, v in sorted(by.items(), key=lambda kv: -kv[1][0])},
            "by_layer": layers,
        }
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
