"""逐步 loss 差图：上半是每步「被测 − 基线」，下半是 100 步滑动平均（看固定偏移）。名字以 ctl 开头的画成灰色（只改编译开关的对照）。

  python plot_diff.py <输出 png> <标题> results/series/<基线>.json <名>=results/series/<运行>.json …
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(p):
    x = json.loads(Path(p).read_text())
    return np.array(x["loss"], dtype=np.float64)


def main():
    out, title, base, *arms = sys.argv[1:]
    b = load(base)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 7.4), dpi=110, sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    pal = ["#1a73e8", "#e8710a", "#1e8e3e", "#9334e6", "#d93025", "#12b5cb", "#f9ab00"]
    greys = ["#5f6368", "#9aa0a6", "#bdc1c6", "#80868b"]
    items = [(lab, load(p)) for lab, p in (a.split("=", 1) for a in arms)]
    ctl = [x for x in items if x[0].startswith("ctl")]
    cp = [x for x in items if not x[0].startswith("ctl")]
    for i, (lab, a) in enumerate(ctl + cp):
        n = min(len(a), len(b))
        y = a[:n] - b[:n]
        isc = lab.startswith("ctl")
        col = greys[i % len(greys)] if isc else pal[(i - len(ctl)) % len(pal)]
        ax.plot(np.arange(n), y, lw=0.5, color=col, alpha=0.8, label=lab)
        r = np.convolve(y, np.ones(100) / 100, mode="full")[:n]
        r[:99] = np.cumsum(y[:99]) / np.arange(1, 100)
        ax2.plot(np.arange(n), r, lw=1.1 if isc else 1.3, color=col, ls="--" if isc else "-", label=lab)
    for a_ in (ax, ax2):
        a_.axhline(0, color="#202124", lw=0.5)
        a_.grid(alpha=0.25)
    ax.set_ylim(-0.02, 0.02)
    ax.set_ylabel("loss(arm) - loss(base)")
    ax2.set_ylabel("100-step rolling mean")
    ax2.set_xlabel("step")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, ncol=6, loc="upper right")
    ax2.legend(fontsize=7, ncol=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    main()
