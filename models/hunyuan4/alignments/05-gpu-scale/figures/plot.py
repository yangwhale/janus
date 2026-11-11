"""画 GPU 扩展的图：被测对单机 golden g3-a 的逐步相对差（前 300 步），左主 loss、右 param_norm；
灰带是单机同设置 8 份（g3-a … g3-h）两两 28 对每步相对差的绝对值上限（尺子），灰线是单机重跑 g3-b … g3-h 本身。正对照 mn-lr 出了尺子几百倍，不画。

    python models/hunyuan4/alignments/05-gpu-scale/figures/plot.py --root <运行目录根> --out <png>
"""

import argparse
import itertools
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
N = 300


def series(root, run, field):
    out = {}
    for line in open(os.path.join(root, run, "metrics.jsonl")):
        r = json.loads(line)
        if r["step"] <= N:
            out[r["step"]] = r[field]
    return [out[s] for s in range(1, N + 1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--multi", default="mn-a,mn-b,mn-c,mn-d,mn-e,mn-f,mn-g,mn-h")
    a = p.parse_args()
    steps = list(range(1, N + 1))
    multi = [r for r in a.multi.split(",") if os.path.isdir(os.path.join(a.root, r))]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150)
    for ax, field, title in ((axes[0], "lm_loss", "主 loss"), (axes[1], "param_norm_fp64", "param_norm")):
        g = series(a.root, "g3-a", field)
        single = [series(a.root, f"g3-{c}", field) for c in "abcdefgh"]
        env = [max(abs(x[i] - y[i]) for x, y in itertools.combinations(single, 2)) / g[i] for i in range(N)]
        ax.fill_between(
            steps, [-e for e in env], env, color="#d9d8d3", linewidth=0, label="单机重跑两两之差的上限（尺子）"
        )
        for k, t in enumerate(single[1:]):
            ax.plot(
                steps,
                [(x - y) / y for x, y in zip(t, g)],
                color="#8a8984",
                linewidth=0.6,
                label="单机重跑 g3-b … g3-h" if k == 0 else None,
            )
        for k, run in enumerate(multi):
            t = series(a.root, run, field)
            ax.plot(
                steps,
                [(x - y) / y for x, y in zip(t, g)],
                color="#2a78d6",
                linewidth=0.8,
                alpha=0.7,
                label=f"2 台 DP 16 + EP 4（{len(multi)} 份重跑）" if k == 0 else None,
            )
        for run, color, label in (
            ("mn-pp2", "#eb6834", "2 台 PP 2 + DP 8 + EP 4"),
            ("mn-ep16", "#1baf7a", "2 台 DP 16 + EP 16"),
        ):
            t = series(a.root, run, field)
            ax.plot(steps, [(x - y) / y for x, y in zip(t, g)], color=color, linewidth=1.2, label=label)
        ax.axhline(0, color="#52514e", linewidth=0.6)
        ax.set_xlabel("步")
        ax.set_ylabel(f"{title}：(被测 − g3-a) ÷ g3-a")
        ax.set_xlim(1, N)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#ebeae6", linewidth=0.6)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(a.out)


if __name__ == "__main__":
    main()
