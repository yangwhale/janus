"""画 TPU 自比的图：被测减 golden 的主 loss（前 300 步），灰带是 GPU 同设置 8 份两两 28 对每步差的绝对值上限（尺子）。

    python models/hunyuan4/alignments/04-tpu-self/figures/plot.py --root <运行目录根> --out <png>
"""
import argparse
import itertools
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]


def series(root, run, field="lm_loss", n=300):
    out = {}
    for line in open(os.path.join(root, run, "metrics.jsonl")):
        r = json.loads(line)
        if r["step"] <= n:
            out[r["step"]] = r[field]
    return [out[s] for s in range(1, n + 1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    steps = list(range(1, 301))
    g = series(a.root, "ts-g-a")
    gpu = [series(a.root, f"g2-{c}") for c in "abcdefgh"]
    env = [max(abs(x[i] - y[i]) for x, y in itertools.combinations(gpu, 2)) for i in range(300)]
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    ax.fill_between(steps, [-e for e in env], env, color="#d9d8d3", linewidth=0, label="GPU 重跑两两之差的上限（尺子）")
    for run, color, label in (("ts-pc-logits", "#eb6834", "正对照：输出头矩阵乘降回默认精度"),
                              ("ts-pc-zm", "#1baf7a", "正对照：路由偏置改零均值更新"),
                              ("ts-g-b", "#2a78d6", "负对照：golden 换切片重跑（与 golden 逐位相同）")):
        t = series(a.root, run)
        ax.plot(steps, [x - y for x, y in zip(t, g)], color=color, linewidth=1.2 if run != "ts-g-b" else 2, label=label)
    ax.axhline(0, color="#52514e", linewidth=0.6)
    ax.set_xlabel("步")
    ax.set_ylabel("主 loss：被测 − golden")
    ax.set_xlim(1, 300)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#ebeae6", linewidth=0.6)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(a.out)


if __name__ == "__main__":
    main()
