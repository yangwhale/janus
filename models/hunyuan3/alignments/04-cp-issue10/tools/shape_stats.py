"""三段形态的分段统计（第 1 节），口径同 一次实际排查的评论：
逐步差 = 被测 loss − 基线 loss（绝对值，完整精度）；起跳步 = 过去 25 步 |差| 的滑动平均首次超过 2.5e-4；
分段报平均绝对差、带符号均值（固定偏移）和抖动（逐步差减去带符号均值后的标准差）。

  python shape_stats.py --base results/series/E1B.json --arm 名=results/series/E1RG2.json … --out results/shape-E1.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

SEGS = [(0, 100), (100, 300), (300, 1000), (1000, 2000), (2000, 3000)]


def load(p):
    x = json.loads(Path(p).read_text())
    return {x["first_step"] + i: float(v) for i, v in enumerate(x["loss"])}


def stats(a, b):
    steps = sorted(set(a) & set(b))
    d = {s: a[s] - b[s] for s in steps}
    jump, w = None, []
    for s in steps:
        w = (w + [abs(d[s])])[-25:]
        if len(w) == 25 and sum(w) / 25 > 2.5e-4:
            jump = s
            break
    seg = {}
    for lo, hi in SEGS:
        v = np.array([d[s] for s in steps if lo <= s < hi])
        if len(v):
            seg[f"{lo}-{hi}"] = {"mean_abs": float(np.abs(v).mean()), "signed_mean": float(v.mean()), "jitter_sd": float(v.std()), "n": len(v)}
    return {"steps": len(steps), "bitwise_equal_steps": sum(1 for s in steps if d[s] == 0.0), "jump_step": jump, "segments": seg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    b = load(a.base)
    res = {"base": Path(a.base).stem, "base_loss": {str(s): b[s] for s in (0, 300, 1000, 2000, max(b))}, "arms": {}}
    for x in a.arm:
        k, p = x.split("=", 1)
        res["arms"][k] = dict(run=Path(p).stem, **stats(load(p), b))
        r = res["arms"][k]
        print(k, r["jump_step"], {s: f"{v['mean_abs']:.1e}/{v['signed_mean']:+.1e}" for s, v in r["segments"].items()})
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
