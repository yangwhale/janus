"""残差流探针逐层比：带 resid_mean.* / probe_logit_mean / probe_lse_mean 的运行（TPU `train_hy3.py --resid-probe-every`，
GPU `frameworks/megatron/resid_probe.py`，同一条固定序列）逐步比，找哪一层、哪一步最先分开。

  python resid_probe_compare.py --test <被测 metrics.jsonl> --ref <参照> [--pair <另一参照>] [--dims 2007,3881,2146,1567,528,14]
      [--out json]

每个 resid_mean.<名> 是该点输出在 token 上的均值向量。报：相对差 ‖t − r‖/‖r‖、均值向量范数之比、指定维度上的相对差；
参照对（--pair 对 --ref）给出同一量的噪声。

proj：均值向量在 w = γ ⊙ û 上的投影之相对差 (t·w − r·w)/|r·w|。γ 是参照该步的 final_norm_scale，û 是参照该步
最后归一化输出均值的单位方向。最后归一化输出 = γ ⊙ x/rms(x)，所以它在 û 上的分量 = w · mean(x/rms)：各层输出均值在 w 上的
投影就是它们对 h̄·û 的贡献（忽略逐 token RMS 加权；拆那部分要 tok_rms.*）。看哪一层最先在这个方向上分开。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(p):
    out = {}
    for line in open(p):
        r = json.loads(line)
        if any(k.startswith("resid_mean.") for k in r):
            out[r["step"]] = r
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--pair", default=None)
    p.add_argument("--dims", default="2007,3881,2146,1567,528,14")
    p.add_argument("--out")
    a = p.parse_args(argv)
    T, R = load(a.test), load(a.ref)
    P = load(a.pair) if a.pair else {}
    dims = [int(x) for x in a.dims.split(",") if x]
    steps = sorted(set(T) & set(R))
    keys = sorted(k for k in T[steps[0]] if k.startswith("resid_mean.") and k in R[steps[0]])
    res = {"steps": steps, "keys": keys, "rows": []}
    fin = "resid_mean.decoder.decoder_norm.out"
    for s in steps:
        row = {"step": s}
        w = None
        if fin in R[s] and "final_norm_scale" in R[s]:
            h = np.asarray(R[s][fin], dtype=np.float64)
            w = np.asarray(R[s]["final_norm_scale"], dtype=np.float64) * h / np.linalg.norm(h)
        for sc in ("probe_logit_mean", "probe_lse_mean"):
            if sc in T[s] and sc in R[s]:
                row[sc] = {"test": T[s][sc], "ref": R[s][sc], "pair": P[s][sc] if s in P and sc in P[s] else None}
        for k in keys:
            t, r = np.asarray(T[s][k]), np.asarray(R[s][k])
            ent = {"rel": float(np.linalg.norm(t - r) / np.linalg.norm(r)), "norm_ratio": float(np.linalg.norm(t) / np.linalg.norm(r)),
                   "dims_rel": [float((t[d] - r[d]) / abs(r[d])) if r[d] != 0 else None for d in dims]}
            if w is not None:
                rw = float(r @ w)
                ent["proj"] = float((t @ w - rw) / abs(rw))
            if s in P and k in P[s]:
                q = np.asarray(P[s][k])
                ent["pair_rel"] = float(np.linalg.norm(q - r) / np.linalg.norm(r))
                ent["pair_norm_ratio"] = float(np.linalg.norm(q) / np.linalg.norm(r))
                if w is not None:
                    ent["pair_proj"] = float((q @ w - rw) / abs(rw))
            row[k] = ent
        res["rows"].append(row)
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for row in res["rows"]:
        s = row["step"]
        line = f"step {s:4d}"
        if "probe_lse_mean" in row:
            v = row["probe_lse_mean"]
            line += f"  lse T−R {v['test'] - v['ref']:+.4f}" + (f" (pair {v['pair'] - v['ref']:+.4f})" if v["pair"] is not None else "")
        print(line)
        for k in keys:
            e = row[k]
            print(f"    {k[len('resid_mean.'):]:34s} rel {e['rel']:.3e}" + (f" (pair {e['pair_rel']:.3e})" if "pair_rel" in e else "")
                  + f"  |m| ratio {e['norm_ratio'] - 1:+.3e}" + (f" (pair {e['pair_norm_ratio'] - 1:+.3e})" if "pair_norm_ratio" in e else "")
                  + (f"  proj {e['proj']:+.3e}" if "proj" in e else "") + (f" (pair {e['pair_proj']:+.3e})" if "pair_proj" in e else ""))


if __name__ == "__main__":
    main()
