"""画「每种改动的曲线最差 r」：按 arms.yaml 与 results/，每个有结果的臂一个点（横轴 r，对数），按档上色；r = 1、2 两条竖线。

    python models/hunyuan4/alignments/04-tpu-self/figures/plot_r.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
HERE = Path(__file__).resolve().parent.parent
COLOR = {"逐位相同": "#2a78d6", "在噪声内": "#1baf7a", "略超": "#eda100", "超": "#e34948"}
FLOOR = 0.05   # 逐位相同（r = 0）画在这里


def main():
    spec = yaml.safe_load((HERE / "arms.yaml").read_text())
    pts = []
    for a in spec["arms"]:
        if not a.get("run"):
            continue
        vs = a.get("vs", "ts-g-a")
        p = HERE / "results" / f"{a['run']}-vs-{'golden' if vs == 'ts-g-a' else vs}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        if d["curves"].get("skipped"):
            continue
        v = d["verdict"]
        pts.append((spec["classes"][a["id"][0]], a["plain"], {"在刻度内": "在噪声内"}.get(v.get("band") or v["label"], v.get("band") or v["label"]), v.get("curves_r_max") or 0.0))
    pts.sort(key=lambda t: (list(spec["classes"].values()).index(t[0]), t[3]))
    fig, ax = plt.subplots(figsize=(8, 0.26 * len(pts) + 1.2), dpi=150)
    for i, (_cls, _name, band, r) in enumerate(pts):
        ax.scatter(max(r, FLOOR), i, s=36, color=COLOR.get(band, "#52514e"), edgecolors="#fcfcfb", linewidths=1.5, zorder=3)
    ax.set_yticks(range(len(pts)), [f"{n}（{c}）" for c, n, _, _ in pts], fontsize=7)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(FLOOR * 0.8, max(3.0, max(t[3] for t in pts) * 1.3))
    for x in (1, 2):
        ax.axvline(x, color="#52514e", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlabel("300 步曲线最差 r（差 ÷ GPU 重跑波动上限；逐位相同画在最左）", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#ebeae6", linewidth=0.6)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=b) for b, c in COLOR.items()]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "r-by-change.png")


if __name__ == "__main__":
    main()
