"""对齐实践大表配的图（只读 results.json / details，不重算指标）。要 matplotlib（pip install -e '.[report]'）。

- L2：逐层柱状图，柱是实测（最差指标），横线是该层的噪声上限 N 与 2N；
- L1 / L3：ULP 分桶（ulp_distribution 的 buckets）；
- 路由：翻转 token 的 topk 分差直方图（details 里带 margins 时才画；没有翻转就没有图）。
L4 逐步曲线在 janus/analysis/curves.py（要读逐步原值，不在 results.json 里）。
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from janus.report.tables import measured


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from janus.report.html import setup_fonts
    setup_fonts()
    return plt


LAYER = re.compile(r"^L2\.layer(\d+)\.(fwd|bwd|grads)$")
KIND_ZH = {"fwd": "前向输出", "bwd": "反向（激活梯度）", "grads": "参数梯度"}


def l2_layers(checks: list[dict], out: Path, title: str) -> Path | None:
    """checks：同一个 results.json 里的检查。按 L2.layer{i}.{fwd,bwd,grads} 画三栏。"""
    by = {}
    for c in checks:
        m = LAYER.match(c["id"])
        th = c.get("threshold") or {}
        if not m or th.get("kind") != "calibrated":
            continue
        ups = (th.get("noise") or {}).get("upper") or {}
        key = "rel_l2" if "rel_l2" in ups else next(iter(ups), None)
        if key is None:
            continue
        by.setdefault(m.group(2), []).append((int(m.group(1)), measured(c.get("metrics") or {}, key), ups[key], key))
    if not by:
        return None
    plt = _plt()
    kinds = [k for k in ("fwd", "bwd", "grads") if k in by]
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.3 * len(kinds), 3.3), squeeze=False)
    for ax, k in zip(axes[0], kinds):
        rows = sorted(by[k])
        xs = [r[0] for r in rows]
        vals = [r[1] or 0.0 for r in rows]
        ns = [r[2] for r in rows]
        pos = [v for v in vals + ns + [2 * n for n in ns] if v and v > 0]
        lo = min(pos) / 10 if pos else 1e-12
        ax.bar(xs, [v if v > 0 else lo for v in vals], color=["tab:blue" if v > 0 else "0.85" for v in vals], width=0.6, label="实测")
        for x, n, v in zip(xs, ns, vals):
            ax.hlines(n if n > 0 else lo, x - 0.4, x + 0.4, color="k", lw=1.6)
            ax.hlines(2 * n if n > 0 else lo, x - 0.4, x + 0.4, color="tab:red", lw=1.2, ls="--")
            if v == 0:
                ax.text(x, lo * 1.3, "0", ha="center", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylim(lo, max(pos) * 3 if pos else 1.0)
        ax.set_xticks(xs)
        ax.set_xlabel("层")
        ax.set_title(f"{KIND_ZH[k]}：{rows[0][3]}", fontsize=10)
    axes[0][0].plot([], [], color="k", lw=1.6, label="N（噪声上限）")
    axes[0][0].plot([], [], color="tab:red", lw=1.2, ls="--", label="2N")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def ulp_buckets(check: dict, out: Path, title: str) -> Path | None:
    b = (check.get("metrics") or {}).get("buckets")
    if not b:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(list(b), [float(v) for v in b.values()], color="tab:blue")
    for i, v in enumerate(b.values()):
        ax.text(i, float(v), f"{float(v):.1%}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("|被测 − 参照|（ULP）")
    ax.set_ylabel("元素占比")
    ax.set_title(title, fontsize=10)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def router_margin(details: Path, out: Path, title: str) -> Path | None:
    """details 每行若带 margins_flipped（翻转 token 上参照自己第 k 与第 k+1 名的分差）就画直方图；没有就返回 None。"""
    if not details.exists():
        return None
    flipped, allm = [], []
    for line in details.read_text().splitlines():
        r = json.loads(line)
        flipped += [float(x) for x in r.get("margins_flipped", [])]
        allm += [float(x) for x in r.get("margins_all", [])]
    if not flipped:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3))
    pos = [x for x in flipped + allm if x > 0]
    bins = [10 ** e for e in range(int(math.floor(math.log10(min(pos)))), int(math.ceil(math.log10(max(pos)))) + 1)] if pos else 20
    if allm:
        ax.hist(allm, bins=bins, alpha=0.4, color="0.5", label=f"全部 token（{len(allm)}）", density=True)
    ax.hist(flipped, bins=bins, alpha=0.8, color="tab:red", label=f"翻转的 token（{len(flipped)}）", density=True)
    ax.set_xscale("log")
    ax.set_xlabel("topk 分差（第 k 名 − 第 k+1 名）")
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out
