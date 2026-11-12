"""后段偏移的分布（第 6 节）：每条运行对同一个基线，在窗口里算逐步相对差 dₜ = (aₜ − bₜ)/|bₜ| 的带符号均值，
按「族」（只改编译开关、Ring、基线关 base2……）列出来，报每族的均值与标准差，并画成散点图。

  python offset_spread.py --base results/series/E1B.json --family '只改编译=E1N1,E1N56,…' --family 'Ring=E1RG2,…' \
      --windows 100:3000,2000:3000 --out results/e1-offset-spread.json --fig figures/e1-offset-spread.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(d, n):
    return np.array(json.loads((Path(d) / f"{n}.json").read_text())["loss"], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--family", action="append", required=True)
    ap.add_argument("--windows", default="100:3000,2000:3000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig")
    a = ap.parse_args()
    sdir = Path(a.base).parent
    b = load(sdir, Path(a.base).stem)
    wins = [tuple(int(x) for x in w.split(":")) for w in a.windows.split(",")]
    res = {"base": Path(a.base).stem, "families": {}}
    for f in a.family:
        name, runs = f.split("=", 1)
        fam = {}
        for w in wins:
            vals = {}
            for r in runs.split(","):
                x = load(sdir, r)
                lo, hi = w
                vals[r] = float(np.mean((x[lo:hi] - b[lo:hi]) / np.abs(b[lo:hi])))
            v = np.array(list(vals.values()))
            fam[f"{w[0]}:{w[1]}"] = {"runs": vals, "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None}
        res["families"][name] = fam
        print(name, {k: (f"{x['mean']:+.2e}", x["sd"]) for k, x in fam.items()})
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    if a.fig:
        fig, axes = plt.subplots(1, len(wins), figsize=(5.5 * len(wins), 3.8), dpi=110)
        axes = np.atleast_1d(axes)
        for ax, w in zip(axes, wins):
            wk = f"{w[0]}:{w[1]}"
            for i, fam in enumerate(res["families"].values()):
                ys = list(fam[wk]["runs"].values())
                ax.scatter([i] * len(ys), ys, s=28, alpha=0.8)
                ax.hlines(np.mean(ys), i - 0.25, i + 0.25, color="#202124", lw=1.2)
            ax.axhline(0, color="#9aa0a6", lw=0.6)
            ax.set_xticks(range(len(res["families"])))
            ax.set_xticklabels(list(res["families"]), fontsize=8, rotation=15)
            ax.set_title(f"signed mean of relative loss diff, steps {wk}", fontsize=9)
            ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(a.fig)


if __name__ == "__main__":
    main()
