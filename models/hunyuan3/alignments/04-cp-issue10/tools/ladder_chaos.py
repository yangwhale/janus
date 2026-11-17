#!/usr/bin/env python3
"""实践 04 阶梯实验：只改编译开关的那一族（基线 + 4 份尺子）随阶梯怎么变——检验「三段形态 = 轨迹混沌」是不是被 BF16 误差放大了。

每级、每对（尺子对基线，CP 臂对基线），逐步相对差 dₜ = (aₜ − bₜ) / bₜ：
- 分叉步：过去 25 步 |dₜ| 的滑动平均首次超过 2.5e-4（同 10-29 评论的起跳步，这里用相对差）；
- 各段平均 |dₜ|：2–50、50–100、100–300、300–1000、1000–2000、2000–3000；
- 后段（2000–3000）带符号均值；基线族 5 份两两配对，后段带符号均值的标准差（只改编译开关能把后段推开多远）。

  python ladder_chaos.py --series results/series/ladder --prefix lbd --out results/ladder/chaos-lbd.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

LEVELS = ["L0", "L0p", "L1", "L2", "L3", "L4"]
CP = ["ag2", "ag4", "rg2", "rg4", "ul2", "ul4", "usp4"]
REFS = ["v64", "v56", "v40", "v24"]
SAME = {("L1", "rg2"): "L0", ("L1", "rg4"): "L0", ("L1", "usp4"): "L0"}
SEGS = [(1, 50), (50, 100), (100, 300), (300, 1000), (1000, 2000), (2000, 3000)]   # 0 起的下标，对应第 2–50 步 …


def load(p: Path):
    return np.array(json.loads(p.read_text())["loss"]) if p.exists() else None


def pair(a, b):
    d = (a - b) / b
    r = np.convolve(np.abs(d), np.ones(25) / 25, "valid")
    hit = np.where(r > 2.5e-4)[0]
    return {"onset": int(hit[0]) + 25 if len(hit) else None,
            "mean_abs": {f"{lo + 1}-{hi}": float(np.abs(d[lo:hi]).mean()) for lo, hi in SEGS},
            "late_signed": float(d[2000:3000].mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", required=True)
    ap.add_argument("--prefix", default="lbd")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    S = Path(a.series)
    res = {}
    for L in LEVELS:
        b = load(S / f"{a.prefix}-{L}-dp8.json")
        fam = {"base": b, **{r: load(S / f"{a.prefix}-{L}-dp8-{r}.json") for r in REFS}}
        if b is None or any(v is None for v in fam.values()):
            continue
        rec = {"refs": {r: pair(fam[r], b) for r in REFS}, "arms": {}}
        for arm in CP:
            x = load(S / f"{a.prefix}-{SAME.get((L, arm), L)}-{arm}.json")
            if x is not None:
                rec["arms"][arm] = pair(x, b)
        late = [pair(fam[i], fam[j])["late_signed"] for i, j in itertools.combinations(fam, 2)]
        rec["family_late_signed_std"] = float(np.std(late))
        rec["family_late_signed_maxabs"] = float(np.max(np.abs(late)))
        rec["ref_onsets"] = [rec["refs"][r]["onset"] for r in REFS]
        res[L] = rec
        ma = {k: np.mean([rec["refs"][r]["mean_abs"][k] for r in REFS]) for k in rec["refs"][REFS[0]]["mean_abs"]}
        print(f"{L:4s} 尺子分叉步 {rec['ref_onsets']}  尺子平均|d| " + " ".join(f"{k}:{v:.1e}" for k, v in ma.items())
              + f"  族后段带符号 std {rec['family_late_signed_std']:.1e} max {rec['family_late_signed_maxabs']:.1e}")
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
