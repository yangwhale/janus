"""TPU 自比：纯数据并行与只切词嵌入表两臂的梯度范数逐步相对差，摊开到每一步与每 50 步。

`results/ts-dp8-vs-golden.*`、`ts-esd-dp8-vs-golden.*` 只有窗口统计（带符号均值、MARE、最大单步）；
这里从原始逐步记录把同一口径的逐步序列算出来，给「偏差是一直偏，还是只在前一段」这类问题看：

- 被测：`ts-dp8`（纯数据并行 8）、`ts-esd-dp8`（只切词嵌入表、其余纯数据并行），对 TPU 基线 `ts-g-a`（FSDP 8），
  d_t = (被测 − 基线) / |基线|；
- 噪声：GPU 同设置 `g2-a … g2-h`（就是这两臂判定时那把 28 对尺子），另 7 份对 `g2-a` 的逐步差，
  以及 28 对 d_t = (前一份 − 后一份) / |后一份| 的逐步最大 |d_t|；
- 窗口统计：第 2–300 步（第 1 步单独报 d0，同目录 `1:301` 窗口），N 是 28 对各量的最大值，r = 量 / N；
- 每 50 步一块的带符号均值：被测两臂，与 GPU 另 7 份的最小、最大。只作诊断，不进门槛。

用法（仓库根目录）：
  python models/hunyuan4/analysis/tpu_self_dp_grad_norm_steps.py --root <运行根> \\
      --json models/hunyuan4/alignments/04-tpu-self/results/ts-dp8-esd-grad-norm-steps.json
运行根下每个运行一个目录，只读 <运行名>/metrics.jsonl（原始记录在 POC 存储
`janus/hunyuan4/tpu/<运行名>/`、`janus/hunyuan4/golden-t1/g2-*/`）。
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))  # 仓库根：没 pip install 也能 import janus

from janus import metrics as M  # noqa: E402
from janus.analysis.curves import load_file, rel  # noqa: E402

FIELD = "grad_norm"
BASE = "ts-g-a"
TESTS = {"ts-dp8": "纯数据并行 8", "ts-esd-dp8": "只切词嵌入表"}
GPU = [f"g2-{c}" for c in "abcdefgh"]
STEPS = (1, 301)
WINDOW = (2, 301)
CHUNK = 50
KEYS = ("signed_mean", "mare", "max_step")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="运行根：<运行名>/metrics.jsonl")
    p.add_argument("--json", required=True, help="结果文件")
    a = p.parse_args(argv)
    S = {n: load_file(Path(a.root) / n / "metrics.jsonl", FIELD, STEPS) for n in [BASE, *TESTS, *GPU]}
    steps = list(range(*STEPS))

    def series(x, y):
        st, d = rel(S[x], S[y])
        assert list(st) == steps, (x, y)
        return d

    env = None
    pair_stats = []
    for x, y in itertools.combinations(GPU, 2):
        d = series(x, y)
        env = abs(d) if env is None else (abs(d)).clip(min=env)
        s = M.series_stats(S[x], S[y], WINDOW, "golden")
        pair_stats.append({"signed_mean": abs(s.signed_mean), "mare": s.mare, "max_step": s.max_step})
    N = {k: max(r[k] for r in pair_stats) for k in KEYS}
    gpu7 = {n: series(n, GPU[0]) for n in GPU[1:]}

    def chunks(d):
        return [float(d[i : i + CHUNK].mean()) for i in range(0, len(d), CHUNK)]

    out = {
        "schema_version": 1,
        "what": "TPU 自比两臂梯度范数逐步相对差（下标 0 = 第 1 步，共 300 步）；GPU 同设置另 7 份对 g2-a、28 对逐步最大 |d_t|；诊断，不进门槛",
        "field": FIELD,
        "baseline": BASE,
        "denominator": "golden（参照一侧）",
        "steps": [steps[0], steps[-1]],
        "window": "2:301（第 1 步单独报 d0）",
        "noise": {"runs": GPU, "pairs": len(pair_stats), "N": N},
        "chunk": CHUNK,
        "chunks_steps": [[s, min(s + CHUNK - 1, steps[-1])] for s in steps[::CHUNK]],
        "tests": {},
        "gpu_vs_g2a": {n: [float(v) for v in d] for n, d in gpu7.items()},
        "gpu_pairs_envelope": [float(v) for v in env],
        "gpu_vs_g2a_chunks_min": [min(chunks(d)[i] for d in gpu7.values()) for i in range(len(steps[::CHUNK]))],
        "gpu_vs_g2a_chunks_max": [max(chunks(d)[i] for d in gpu7.values()) for i in range(len(steps[::CHUNK]))],
    }
    for n, label in TESTS.items():
        d = series(n, BASE)
        s = M.series_stats(S[n], S[BASE], WINDOW, "golden")
        win = {"signed_mean": s.signed_mean, "mare": s.mare, "max_step": s.max_step, "max_step_at": s.max_step_at}
        out["tests"][n] = {
            "label": label,
            "d0": float(d[0]),
            "window_stats": win,
            "r": {k: abs(win[k]) / N[k] for k in KEYS},
            "chunks": chunks(d),
            "per_step": [float(v) for v in d],
        }
    Path(a.json).write_text(json.dumps(out, ensure_ascii=False) + "\n")
    print("N", {k: f"{v:.4g}" for k, v in N.items()})
    for n, t in out["tests"].items():
        w = t["window_stats"]
        rr = {k: round(v, 2) for k, v in t["r"].items()}
        print(f"{n:<11} sm {w['signed_mean']:.4g} mare {w['mare']:.4g} max {w['max_step']:.4g}@{w['max_step_at']}  r {rr}")
        print("   50 步块", [f"{c:.3g}" for c in t["chunks"]])
    lo_hi = zip(out["gpu_vs_g2a_chunks_min"], out["gpu_vs_g2a_chunks_max"])
    print("GPU 7 份 50 步块范围", [f"{lo:.3g}..{hi:.3g}" for lo, hi in lo_hi])
    return 0


if __name__ == "__main__":
    sys.exit(main())
