"""混元 4 案例 17（后段为什么慢慢偏开）的六张图，只读 alignments/01-mini/results/ 里已提交的结果文件，不读运行目录。

图 1 case17-drift.png：C 口径之前的对齐配置，TPU 8 份主 loss 对 GPU 16 份逐步均值的相对差（25 步分块）；
      右边是延长到 2000 步的 1 份（对 GPU 2 份）。数据 mini-main-loss-drift.json、mini-ihc-ablation.json 的 x2000。
图 2 case17-fix-curves.png：同一种分块曲线，对照（同图 1）、补上流梯度舍入（只开 M58，5 份）、两边都去掉 iHC（TPU 2 份对 GPU 3 份）。
      数据 mini-main-loss-drift.json、mini-ihc-ablation.json 的 ihc、noihc。
图 3 case17-windows.png：前段（第 51–100 步）与后段（第 551–1000 步）每份运行一个点。左 TPU 四种配置对 GPU 16 份均值
      （mini-factorial-tpu-6.json），右 GPU 诊断开关对 GPU 16 份留一法（mini-gpu-tpulike.json、mini-gpu-headcot.json）。
图 4 case17-cancel.png：「两个错误抵消」示意。纵轴是格均值，取自图 3 的同两份结果文件；箭头只表示方向，不是加法分解。
图 5 case17-scan-classes.png：C 口径整步扫描出的被省掉的舍入，按位置类（mini-c-elided-rounding-scan.json）。
图 6 case17-gpu-candidates.png：清单 ①、② 两类拆成三处在 GPU 上照 TPU 改，后段逐份的点（mini-gpu-tpulike-fanin-wgrad.json）；
      STREAM 一格（主因在 GPU 上的大小）取自 mini-gpu-tpulike.json，作参照。

用法：
    python models/hunyuan4/analysis/plot_case17_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "alignments" / "01-mini" / "results"
OUT = HERE.parent / "alignments" / "01-mini" / "figures"

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

E6 = 1e6  # 纵轴统一写成「× 1e-6」
C_CTL = "#c0392b"  # 对照（只开 M36）
C_M58 = "#2471a3"  # 补流梯度舍入
C_M52 = "#8e44ad"
C_C = "#d68910"  # C 口径
C_GRAY = "0.45"


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())


def block_mid(steps: str) -> float:
    a, b = (int(x) for x in steps.split("-"))
    return (a + b) / 2


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="0.3", lw=0.8, zorder=1)
    ax.grid(axis="y", color="0.92", lw=0.8, zorder=0)


def fig_drift():
    drift = load("mini-main-loss-drift.json")["lm_loss"]["blocks"]
    x2000 = load("mini-ihc-ablation.json")["configs"]["x2000"]["lm_loss"]["blocks_tpu_mean"]
    x = [block_mid(b["steps"]) for b in drift]
    y = [b["tpu_mean"] * E6 for b in drift]
    sd = [2 * b["gpu_loo_sd"] * E6 for b in drift]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 3.9), dpi=150, gridspec_kw={"width_ratios": [2.2, 1]})
    style(ax)
    ax.fill_between(
        x,
        [-s for s in sd],
        sd,
        color="0.88",
        lw=0,
        label="GPU 自己重跑的波动（单份对其余 15 份，±2 倍标准差；对 8 份平均来说偏宽）",
    )
    ax.plot(x, y, color=C_CTL, lw=2, marker="o", ms=3, label="TPU 8 份的平均（C 口径之前的对齐配置）")
    ax.annotate(
        "前段小鼓包\n（第 51–100 步）",
        xy=(63, 5.1),
        xytext=(150, 12),
        arrowprops={"arrowstyle": "->", "color": "0.3"},
        fontsize=9,
    )
    ax.annotate(
        "后段一路变大\n（第 550 步起）",
        xy=(850, 14.5),
        xytext=(560, 16),
        arrowprops={"arrowstyle": "->", "color": "0.3"},
        fontsize=9,
    )
    ax.set_xlim(0, 1000)
    ax.set_xlabel("训练步（每 25 步一个点）")
    ax.set_ylabel("主 loss 相对差（× 1e-6）\n(TPU − GPU) ÷ GPU")
    ax.set_title("1000 步：TPU 比 GPU 高多少", fontsize=11)
    ax.legend(loc="lower left", fontsize=8, frameon=False)

    style(bx)
    xb = [25 * i + 12.5 for i in range(len(x2000))]
    bx.plot(xb, [v * E6 for v in x2000], color=C_CTL, lw=1.2)
    bx.axvline(1000, color="0.6", lw=0.8, ls="--")
    bx.set_xlim(0, 2000)
    bx.set_xlabel("训练步")
    bx.set_title("延长到 2000 步（TPU 1 份对 GPU 2 份）", fontsize=10)
    bx.text(1300, 32, "第 1000 步前后到顶，\n之后回落", fontsize=8.5, color="0.25")
    fig.tight_layout()
    fig.savefig(OUT / "case17-drift.png")
    plt.close(fig)


def fig_fix_curves():
    drift = load("mini-main-loss-drift.json")["lm_loss"]["blocks"]
    cfg = load("mini-ihc-ablation.json")["configs"]
    x = [block_mid(b["steps"]) for b in drift]
    sd = [2 * b["gpu_loo_sd"] * E6 for b in drift]
    fig, ax = plt.subplots(figsize=(9, 3.9), dpi=150)
    style(ax)
    ax.fill_between(x, [-s for s in sd], sd, color="0.88", lw=0, label="GPU 自己重跑的波动（同图 1）")
    ax.plot(x, [b["tpu_mean"] * E6 for b in drift], color=C_CTL, lw=2, label="对照：C 口径之前的对齐配置（8 份平均）")
    m58 = cfg["ihc"]["lm_loss"]
    ax.plot(
        x,
        [v * E6 for v in m58["blocks_tpu_mean"]],
        color=C_M58,
        lw=2,
        label=f"补上流梯度的舍入（只开 M58，{len(m58['tpu'])} 份平均）",
    )
    no = cfg["noihc"]["lm_loss"]
    ax.plot(
        x,
        [v * E6 for v in no["blocks_tpu_mean"]],
        color="0.2",
        lw=1.4,
        ls="--",
        label=f"两边都去掉 iHC（TPU {len(no['tpu'])} 份对 GPU {len(no['gpu'])} 份）",
    )
    ax.set_xlim(0, 1000)
    ax.set_xlabel("训练步（每 25 步一个点）")
    ax.set_ylabel("主 loss 相对差（× 1e-6）")
    ax.set_title("去掉 iHC，或者照 GPU 补上流梯度的舍入，后段的尾巴都下去了", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "case17-fix-curves.png")
    plt.close(fig)


def _dots(ax, cells, labels, colors, key):
    for i, (c, col) in enumerate(zip(cells, colors)):
        vals = [r[key] * E6 for r in c["runs"].values()]
        n = len(vals)
        xs = [i + (j - (n - 1) / 2) * min(0.08, 0.5 / max(n, 1)) for j in range(n)]
        ax.scatter(xs, vals, s=22, color=col, alpha=0.8, zorder=3, edgecolor="none")
        m = c[key]["mean"] * E6
        ax.plot([i - 0.28, i + 0.28], [m, m], color=col, lw=2.4, zorder=4)
        ax.text(i + 0.31, m, f"{m:+.1f}", va="center", fontsize=8, color=col)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"{lab}\n{len(c['runs'])} 份" for lab, c in zip(labels, cells)], fontsize=8.5)
    ax.set_xlim(-0.5, len(cells) - 0.3)


def fig_windows():
    tpu = load("mini-factorial-tpu-6.json")["cells"]
    gl = load("mini-gpu-tpulike.json")["cells"]
    gh = load("mini-gpu-headcot.json")["cells"]
    t_cells = [tpu["base"], tpu["M52"], tpu["M58"], tpu["C"]]
    t_lab = ["对照", "只开 M52\n（hc_head 强制舍）", "只开 M58\n（流梯度照 GPU 舍）", "C 口径\n（三处全开）"]
    t_col = [C_CTL, C_M52, C_M58, C_C]
    g_cells = [gl["gpu-loo"], gl["gpu-tpulike-stream"], gl["gpu-tpulike-head"], gh["headcot"]]
    g_lab = ["GPU 原样\n（留一法）", "STREAM\n流梯度照 TPU", "HEAD\n收拢照 TPU", "HEADCOT\n收拢的梯度照 TPU"]
    g_col = [C_GRAY, C_M58, C_M52, C_C]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.6), dpi=150, sharey="row")
    for row, (key, name) in enumerate((("lm_51_100", "前段 第 51–100 步"), ("lm_551_1000", "后段 第 551–1000 步"))):
        for col, (cells, lab, colr, side) in enumerate(
            (
                (t_cells, t_lab, t_col, "TPU：对 GPU 16 份均值"),
                (g_cells, g_lab, g_col, "GPU 诊断开关：对 GPU 16 份留一法"),
            )
        ):
            ax = axes[row, col]
            style(ax)
            _dots(ax, cells, lab, colr, key)
            ax.set_title(f"{side} ｜ {name}", fontsize=10)
            if col == 0:
                ax.set_ylabel("主 loss 相对差（× 1e-6）")
    fig.text(0.5, 0.005, "每个点是一份 1000 步运行，横线是平均。", ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT / "case17-windows.png")
    plt.close(fig)


def fig_cancel():
    tpu = load("mini-factorial-tpu-6.json")["cells"]
    gh = load("mini-gpu-headcot.json")["cells"]["headcot"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), dpi=150, sharey=True)
    for ax, key, name in ((axes[0], "lm_51_100", "前段 第 51–100 步"), (axes[1], "lm_551_1000", "后段 第 551–1000 步")):
        style(ax)
        c = tpu["C"][key]["mean"] * E6
        m = tpu["M58"][key]["mean"] * E6
        g = gh[key]["mean"] * E6
        # TPU 一列：C 口径（hc_head 余切照 GPU 舍）→ 只开 M58（这次不舍）
        ax.scatter([0], [c], s=70, color=C_C, zorder=3)
        ax.scatter([0.9], [m], s=70, color=C_M58, zorder=3)
        ax.annotate("", xy=(0.86, m), xytext=(0.04, c), arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "0.25"})
        ax.text(
            0, c, f"  C 口径 {c:+.1f}\n  （这次照 GPU 舍）", fontsize=8.5, va="bottom" if c >= m else "top", color=C_C
        )
        ax.text(
            0.9, m, f"  只开 M58 {m:+.1f}\n  （这次不舍）", fontsize=8.5, va="top" if c >= m else "bottom", color=C_M58
        )
        # GPU 一列：原样 0 → 拿掉这次舍入
        ax.scatter([2.0], [0], s=70, color=C_GRAY, zorder=3)
        ax.scatter([2.9], [g], s=70, color=C_C, marker="s", zorder=3)
        ax.annotate("", xy=(2.86, g), xytext=(2.04, 0), arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "0.25"})
        ax.text(2.0, 0, "  GPU 原样 0", fontsize=8.5, va="bottom" if g < 0 else "top", color="0.3")
        ax.text(2.9, g, f"  GPU 拿掉这次舍入 {g:+.1f}", fontsize=8.5, va="top" if g < 0 else "bottom", color=C_C)
        ax.set_xticks([0.45, 2.45])
        ax.set_xticklabels(["TPU（C 口径 → 只开 M58）", "GPU（原样 → HEADCOT）"], fontsize=9)
        ax.set_xlim(-0.3, 4.1)
        ax.set_title(name, fontsize=10.5)
    axes[0].set_ylabel("主 loss 相对差（× 1e-6）")
    fig.suptitle("同一处舍入（hc_head 反向那次梯度舍入）拿掉以后：两边都是前段往上、后段往下", fontsize=11)
    fig.text(
        0.5,
        0.01,
        "点是格均值（TPU C 口径 6 份、只开 M58 5 份；GPU HEADCOT 4 份）。箭头只表示方向，不是加法分解。TPU 两格的配置只差 M52，编出来在 hc_head 这一段只差这一次舍入。",
        ha="center",
        fontsize=8,
        color="0.3",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(OUT / "case17-cancel.png")
    plt.close(fig)


CLASS_SAY = {
    "weight_grad": ("② 各线性层每个微批的 dW", "反向"),
    "branch_input_dgrad_sum": ("① 分支输入的梯度合流", "反向"),
    "dense_out_fwd": ("③ 前向线性层输出（fc1 等）", "前向"),
    "attn_gate_fwd": ("③ 注意力输出门", "前向"),
    "routed_expert_actup_fwd": ("③ 路由专家 act×up", "前向"),
    "routed_expert_actup_dgrad": ("路由专家 act×up 的反向", "反向"),
    "indexer_mask_exact": ("索引器掩码（值在 BF16 上精确，无害）", "前向"),
    "norm_input_dgrad_sum": ("norm 输入梯度", "反向"),
    "norm_output_fwd": ("norm 输出", "前向"),
}


def fig_scan_classes():
    d = load("mini-c-elided-rounding-scan.json")["elided_uses_by_class"]
    rows = sorted(d.items(), key=lambda kv: sum(kv[1].values()))
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    ax.spines[["top", "right"]].set_visible(False)
    parts = (
        ("scan", "层 scan 里", "#2e86c1"),
        ("outside_scan", "scan 外（MTP 块、头尾）", "#85c1e9"),
        ("other", "XLA 自己插的", "#d5d8dc"),
    )
    total = 0
    for i, (k, v) in enumerate(rows):
        left = 0
        for key, lab, col in parts:
            n = v.get(key, 0)
            ax.barh(i, n, left=left, color=col, height=0.62, label=lab if i == len(rows) - 1 else None)
            left += n
        total += left
        name, dirn = CLASS_SAY[k]
        ax.text(left + 1, i, f"{left}", va="center", fontsize=8.5)
        ax.text(-2, i, f"{name}（{dirn}）", va="center", ha="right", fontsize=8.5)
    ax.set_yticks([])
    ax.set_xlabel("被省掉的舍入（处；同一个 convert 每被转回一次算一处）")
    ax.set_title(f"C 口径整步扫出来的 {total} 处，按位置归类", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "case17-scan-classes.png")
    plt.close(fig)


def fig_gpu_candidates():
    fw = load("mini-gpu-tpulike-fanin-wgrad.json")["cells"]
    st = load("mini-gpu-tpulike.json")["cells"]["gpu-tpulike-stream"]
    cells = [fw["gpu-loo"], st, fw["attnin"], fw["ffnin"], fw["attnin_ffnin"], fw["wgrad"]]
    labels = [
        "GPU 原样\n（留一法）",
        "STREAM\n（主因，参照）",
        "ATTNIN\n注意力输入合流",
        "FFNIN\nMoE 输入合流",
        "ATTNIN\n＋FFNIN",
        "WGRAD\n每微批 dW",
    ]
    colors = [C_GRAY, C_M58, C_C, C_M52, "#7f8c8d", C_CTL]
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    style(ax)
    _dots(ax, cells, labels, colors, "lm_551_1000")
    ax.set_ylabel("后段主 loss 相对差（× 1e-6）")
    ax.set_title("GPU 上把 ①、② 两类照 TPU 改：dW 不动；两处合流单开有一点，合起来反而没了", fontsize=11)
    fig.text(
        0.5,
        0.01,
        "每个点是一份 1000 步 GPU 运行，对 GPU 16 份留一法，第 551–1000 步；横线是平均。",
        ha="center",
        fontsize=8.5,
        color="0.3",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "case17-gpu-candidates.png")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    fig_drift()
    fig_fix_curves()
    fig_windows()
    fig_cancel()
    fig_scan_classes()
    fig_gpu_candidates()


if __name__ == "__main__":
    main()
