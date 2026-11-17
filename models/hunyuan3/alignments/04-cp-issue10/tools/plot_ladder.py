#!/usr/bin/env python3
"""实践 04 阶梯实验的逐步差图：每级一格，横轴步数，纵轴相对差 (被测 − 基线) / 基线（50 步滑动平均）；
灰线 = 只改片上内存上限的 4 份尺子，彩线 = 7 个 CP 臂。

  python plot_ladder.py --series results/series/ladder --prefix lbd --levels L0,L0p,L1,L2,L3,L4 --out figures/ladder-lbd.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CP = ["ag2", "ag4", "rg2", "rg4", "ul2", "ul4", "usp4"]
REFS = ["v64", "v56", "v40", "v24"]
SAME = {("L1", "rg2"): "L0", ("L1", "rg4"): "L0", ("L1", "usp4"): "L0"}
COLORS = {"ag2": "#1f77b4", "ag4": "#aec7e8", "rg2": "#d62728", "rg4": "#ff9896", "ul2": "#2ca02c", "ul4": "#98df8a", "usp4": "#9467bd"}
TITLE = {"L0": "L0 原始默认", "L0p": "L0′ 关 excess precision", "L1": "L1 + base2 统一关", "L2": "L2 + 词嵌入切开",
         "L3": "L3 + q/k/v 权重梯度 FP32", "L4": "L4 + 其余对齐开关"}


def load(p: Path):
    return np.array(json.loads(p.read_text())["loss"]) if p.exists() else None


def smooth(x, w=50):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", required=True)
    ap.add_argument("--prefix", default="lbd")
    ap.add_argument("--levels", default="L0,L0p,L1,L2,L3,L4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    plt.rcParams["font.family"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
    levels = a.levels.split(",")
    S = Path(a.series)
    fig, axes = plt.subplots(len(levels), 1, figsize=(10, 2.6 * len(levels)), sharex=True, sharey=True)
    for ax, L in zip(np.atleast_1d(axes), levels):
        b = load(S / f"{a.prefix}-{L}-dp8.json")
        if b is None:
            ax.set_title(f"{TITLE.get(L, L)}（未跑）")
            continue
        for r in REFS:
            x = load(S / f"{a.prefix}-{L}-dp8-{r}.json")
            if x is not None:
                ax.plot(np.arange(50, len(b) + 1), smooth((x - b) / b), color="0.6", lw=0.8, label="只改编译开关" if r == REFS[0] else None)
        for arm in CP:
            x = load(S / f"{a.prefix}-{SAME.get((L, arm), L)}-{arm}.json")
            if x is not None:
                ax.plot(np.arange(50, len(b) + 1), smooth((x - b) / b), color=COLORS[arm], lw=1.0, label=arm)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(TITLE.get(L, L), fontsize=10, loc="left")
        ax.set_ylabel("相对差（50 步平均）")
    np.atleast_1d(axes)[0].legend(ncol=8, fontsize=7, loc="upper left")
    np.atleast_1d(axes)[-1].set_xlabel("步")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
