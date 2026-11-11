"""hy4-mini 训练曲线（L4）的三张图：主 loss 与 GPU 波动带、七项三个量各自的比值（三个并排小图）、其余六个量的逐步相对差。

数据：GPU g3-a … g3-p（16 份同设置，g3-a 是基线）与 TPU 现行一代 tz-off-rep（2026-11-11 起，base2 关）的 metrics.jsonl；
判定数（图 2）读 alignments/01-mini/results/t1zo-l4.results.json，不另算。
逐步相对差 = (TPU − GPU 基线) / GPU 基线（分母一律是 GPU 基线，与判定同口径，见 catalogs/t1zo-l4.yaml；2026-11-10 前梯度范数用两边均值）；
GPU 波动带 = 每一步 GPU 16 份两两 120 对的相对差绝对值的最大值，画成 ±包络。

用法：
    python models/hunyuan4/analysis/plot_l4_figures.py --root /tmp/hy4w/t1root
运行目录的 meta.json 先对 results.json 里登记的 sha256 核一遍，不一致就停。
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MINI = HERE.parent / "alignments" / "01-mini"
RESULTS = MINI / "results" / "t1zo-l4.results.json"
OUT = MINI / "figures"
GPU = [f"g3-{c}" for c in "abcdefghijklmnop"]
TPU = "tz-off-rep"
SUFFIX = "-t1zo"  # 现行比较换代时换后缀，已提交的旧图（上一代 t1zg-a 无后缀）不覆盖

# 图 3 的六个小图：(标题, GPU 字段, TPU 字段, 分母, 灰带取哪些字段)。
# 路由偏置按层判、各层取最差（janus/checks/extra.py 的 router_stats），所以红线画最差的一层、灰带取各层的最大值。
LAYERS = ["1", "2", "3", "4", "5", "6", "mtp0"]
SMALL = [
    ("MTP loss", "mtp_loss_mt", "mtp_loss_mt", "golden", None),
    ("索引器蒸馏 loss", "indexer_loss", "indexer_loss", "golden", None),
    ("梯度范数", "grad_norm", "grad_norm", "golden", None),
    ("参数范数", "param_norm_fp64", "param_norm_exact", "golden", None),
    (
        "路由偏置均值（最差一层：多 token 预测层）",
        "router.bias_mean.mtp0",
        "router.bias_mean.mtp0",
        "golden",
        [f"router.bias_mean.{x}" for x in LAYERS],
    ),
    (
        "路由偏置标准差（最差一层：第 3 层）",
        "router.bias_std.3",
        "router.bias_std.3",
        "golden",
        [f"router.bias_std.{x}" for x in LAYERS],
    ),
]

# 图 2 的七项（全程窗口）：(名称, 检查 id, 是否已接受)
ITEMS = [
    ("主 loss", "L4.loss", False),
    ("MTP loss", "L4.mtp_loss", False),
    ("索引器蒸馏 loss", "L4.indexer_loss", False),
    ("梯度范数", "L4.grad_norm", True),
    ("参数范数", "L4.param_norm", False),
    ("路由偏置均值", "L4.router.bias_mean", False),
    ("路由偏置标准差", "L4.router.bias_std", True),
]
METRIC_SAY = {"signed_mean": "带符号均值", "mare": "平均绝对相对误差", "max_step": "单步最大"}


def load(root: Path, run: str) -> dict[str, np.ndarray]:
    rows = [json.loads(line) for line in (root / run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: r["step"])
    keys = set().union(*rows)
    return {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in keys}


def check_sha(root: Path, res: dict):
    for run, info in res["runs"].items():
        got = hashlib.sha256((root / run / "meta.json").read_bytes()).hexdigest()
        if got != info["meta_sha256"]:
            raise SystemExit(f"{run}/meta.json 的 sha256 与 {RESULTS.name} 登记的不同：{got} ≠ {info['meta_sha256']}")


def rel(t, g, den):
    d = np.abs(g) if den == "golden" else np.abs(t + g) / 2
    return (t - g) / d


def band(gpu: list[np.ndarray], den: str) -> np.ndarray:
    return np.max([np.abs(rel(a, b, den)) for a, b in itertools.combinations(gpu, 2)], axis=0)


def fig_loss(plt, gpu, tpu, step):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    # TPU 画粗一点的半透明红线垫在下面，GPU 16 份的细灰线画在上面，两边都看得见
    ax1.plot(step, tpu["lm_loss"], color="tab:red", lw=2.2, alpha=0.45, zorder=1)
    for g in gpu:
        ax1.plot(step, g["lm_loss"], color="0.35", lw=0.5, zorder=2)
    ax1.set_ylabel("主 loss")
    ax1.plot([], [], color="0.35", lw=0.5, label="GPU 16 份")
    ax1.plot([], [], color="tab:red", lw=2.2, alpha=0.45, label="TPU")
    ax1.legend(loc="upper right", frameon=False)
    b = band([g["lm_loss"] for g in gpu], "golden")
    ax2.fill_between(step, -b, b, color="0.85", lw=0, label="GPU 16 份两两之差的范围")
    ax2.plot(step, rel(tpu["lm_loss"], gpu[0]["lm_loss"], "golden"), color="tab:red", lw=0.6, label="TPU 对 GPU 基线")
    ax2.set_ylabel("逐步相对差")
    ax2.set_xlabel("步数")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    fig.suptitle("1000 步主 loss：TPU 对 GPU 的逐步差与 GPU 自己重跑同量级（判定按整段统计，不逐步判）")
    fig.tight_layout()
    return fig


def signed_router(gpu0, tpu, prefix):
    """路由偏置按层判：带符号均值取绝对值最大的那一层，返回带符号的值（results.json 里存的是绝对值）。
    用判定同一个函数算（janus.checks.series.window_stats，第 1 步单独报、不进窗口统计）。"""
    from janus.checks.series import parse_window, window_stats

    best = 0.0
    for x in LAYERS:
        k = prefix + x
        t = dict(zip(tpu["step"].astype(int), tpu[k]))
        g = dict(zip(gpu0["step"].astype(int), gpu0[k]))
        v = window_stats(t, g, parse_window("1:1001"), "golden", 1)[1]["signed_mean"]  # [1] 是带符号的原值
        best = v if abs(v) > abs(best) else best
    return best


def fig_items(plt, res, gpu0, tpu):
    """三个并排小图：每个量各自除以 GPU 同一量的波动上限，条旁写这个量的 TPU 值与上限。"""
    by_id = {c["id"]: c for c in res["checks"]}
    rows = []
    for name, cid, _acc in ITEMS:
        c = by_id[cid]
        m, up = c["metrics"], c["threshold"]["noise"]["upper"]
        val = {k: m.get(k, m.get(f"{k}_max")) for k in METRIC_SAY}
        if cid.startswith("L4.router"):
            sv = signed_router(gpu0, tpu, "router.bias_mean." if "mean" in cid else "router.bias_std.")
            if not np.isclose(abs(sv), val["signed_mean"], rtol=1e-6):
                raise SystemExit(f"{cid}：重算的带符号均值 {sv} 与 results.json 的 {val['signed_mean']} 不一致")
            val["signed_mean"] = sv
        rows.append((name, val, up))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, k in zip(axes, METRIC_SAY):
        for yi, (_name, val, up) in zip(y, rows):
            r = abs(val[k]) / up[k]
            color = "tab:green" if r <= 1 else ("tab:orange" if r <= 2 else "tab:red")
            ax.barh(yi, min(r, 2.6), color=color, height=0.55)
            ax.text(
                min(r, 2.6) + 0.04,
                yi,
                f"{val[k]:.2e} / {up[k]:.2e}",
                va="center",
                fontsize=7,
                zorder=3,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
            )
        ax.axvline(1, color="0.3", lw=1, ls="--", zorder=1)
        ax.axvline(2, color="tab:red", lw=1, ls="--", zorder=1)
        ax.set_xlim(0, 3.4)
        ax.set_title(METRIC_SAY[k], fontsize=10)
        ax.set_xlabel("比值（≤ 1 对上，> 2 没对上）")
    axes[0].set_yticks(y, [r[0] for r in rows])
    fig.suptitle("训练曲线 1000 步：每个量各自除以 GPU 自己同一量的波动上限（条旁是 TPU 值 / 上限）")
    fig.tight_layout()
    return fig


def fig_small(plt, gpu, tpu, step):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for ax, (title, gf, tf, den, fields) in zip(axes.flat, SMALL):
        b = np.max([band([g[f] for g in gpu], den) for f in (fields or [gf])], axis=0)
        ax.fill_between(step, -b, b, color="0.85", lw=0)
        t = rel(tpu[tf], gpu[0][gf], den)
        ax.plot(step, t, color="tab:red", lw=0.5)
        # 分母过零的个别步会让相对差爆到几百倍，纵轴按 TPU 线与灰带的 99% 分位截断
        lim = 1.15 * max(np.nanmax(np.abs(t)), np.nanpercentile(b, 99))
        ax.set_ylim(-lim, lim)
        ax.set_title(title, fontsize=10)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for ax in axes[1]:
        ax.set_xlabel("步数")
    for ax in axes[:, 0]:
        ax.set_ylabel("逐步相对差")
    fig.suptitle("其余各量：TPU 对 GPU 基线的逐步相对差（灰带 = GPU 16 份两两之差的范围；路由偏置取各层最大）")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True, help="运行目录的上一级（含 g3-a … g3-p 与 tz-off-rep）")
    a = ap.parse_args()
    from janus.report.html import setup_fonts

    setup_fonts()
    import matplotlib.pyplot as plt

    res = json.loads(RESULTS.read_text())
    check_sha(a.root, res)
    gpu = [load(a.root, r) for r in GPU]
    tpu = load(a.root, TPU)
    step = gpu[0]["step"]
    for g in gpu + [tpu]:
        if not np.array_equal(g["step"], step):
            raise SystemExit("各运行的步号不一致")
    OUT.mkdir(exist_ok=True)
    for name, fig in (
        (f"l4-main-loss{SUFFIX}.png", fig_loss(plt, gpu, tpu, step)),
        (f"l4-items{SUFFIX}.png", fig_items(plt, res, gpu[0], tpu)),
        (f"l4-other-series{SUFFIX}.png", fig_small(plt, gpu, tpu, step)),
    ):
        fig.savefig(OUT / name, dpi=130)
        plt.close(fig)
        print(OUT / name)


if __name__ == "__main__":
    main()
