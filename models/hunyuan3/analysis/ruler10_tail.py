"""GPU 默认设置 10 份 1000 步运行两两 45 对的主 loss 噪声上限 N，与 TPU 三臂对 g1k-a 的倍数。

案例 01 第 29 项「噪声带」只记了 10 份各自对 g1k-a 的后段带符号均值（`case01-arms-1000.json`）和均值、
标准差；这里按正式判法（`docs/reference/judging.md`：噪声参照两两配对、每对算判定量、取最大值当 N）
从原始逐步记录重算：

- 噪声：10 份 GPU 默认设置运行（g1k-a/b/c、g-ctl、g-ctl-b、g-ctl-c、g-gn-a/b、g-rs-a/b）两两 45 对，
  每对 d_t = (前一份 − 后一份) / |后一份|（同 `janus.checks.series.noise_from_pairs` 的配对顺序），
  窗口内带符号均值取绝对值、MARE、最大单步，各取 45 对的最大值 = N；
- 被测：TPU 三臂 `t-a`（原样）、`t-noremat-1000`（不重算）、`t-vmem24-1000`（改片上内存上限）对 g1k-a，
  同窗口三个量，r = 量 / N；
- 窗口：后段 301–1000 与全程 2–1000（第 1 步单独报 d0）。只作诊断，不进门槛。

用法（仓库根目录）：
  python models/hunyuan3/analysis/ruler10_tail.py --root <运行根> \\
      --json models/hunyuan3/alignments/01-v5p-acceptance/results/case01-ruler10-tail.json
运行根下每个运行一个目录，只读 <运行名>/metrics.jsonl（原始记录在 POC 存储
`janus/hunyuan3-4l/{golden,tpu}/<运行名>/metrics.jsonl`）。
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
from janus.analysis.curves import load_file  # noqa: E402

GPU = ["g1k-a", "g1k-b", "g1k-c", "g-ctl", "g-ctl-b", "g-ctl-c", "g-gn-a", "g-gn-b", "g-rs-a", "g-rs-b"]
BASE = "g1k-a"
TPU = {"t-a": "原样", "t-noremat-1000": "不重算（remat_policy=save_everything）", "t-vmem24-1000": "改片上内存上限"}
WINDOWS = {"301:1001": (301, 1001), "2:1001": (2, 1001)}
FIELD = "lm_loss"
KEYS = ("signed_mean", "mare", "max_step")


def stats(t, r, w):
    s = M.series_stats(t, r, w, "golden")
    return {"signed_mean": s.signed_mean, "mare": s.mare, "max_step": s.max_step, "max_step_at": s.max_step_at}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="运行根：<运行名>/metrics.jsonl")
    p.add_argument("--json", required=True, help="结果文件")
    a = p.parse_args(argv)
    S = {n: load_file(Path(a.root) / n / "metrics.jsonl", FIELD) for n in GPU + list(TPU)}
    out = {
        "schema_version": 1,
        "what": "混元 3 截 4 层 1000 步主 loss：GPU 默认设置 10 份两两 45 对的噪声上限 N，TPU 三臂对 g1k-a 的三个量与 r；诊断，不进门槛",
        "field": FIELD,
        "denominator": "golden（每对的后一份；被测对 g1k-a）",
        "baseline": BASE,
        "gpu_runs": GPU,
        "pairs": len(list(itertools.combinations(GPU, 2))),
        "gpu_vs_baseline_signed_mean_301_1000": {n: stats(S[n], S[BASE], WINDOWS["301:1001"])["signed_mean"] for n in GPU},
        "windows": {},
    }
    for wn, w in WINDOWS.items():
        pair_rows = []
        for x, y in itertools.combinations(GPU, 2):
            st = stats(S[x], S[y], w)
            pair_rows.append({"pair": f"{x}~{y}", **{k: st[k] for k in KEYS}})
        N = {}
        for k in KEYS:
            top = max(pair_rows, key=lambda r: abs(r[k]))
            N[k] = {"value": abs(top[k]), "pair": top["pair"]}
        tests = {}
        for n, label in TPU.items():
            st = stats(S[n], S[BASE], w)
            tests[n] = {"label": label, **st, "r": {k: abs(st[k]) / N[k]["value"] for k in KEYS}}
        out["windows"][wn] = {"N": N, "tpu": tests}
    Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    for wn, v in out["windows"].items():
        print(wn, "N", {k: "{:.4g} ({})".format(x["value"], x["pair"]) for k, x in v["N"].items()})
        for n, t in v["tpu"].items():
            print(
                f"  {n:<15} sm {t['signed_mean']:.4g} mare {t['mare']:.4g} max {t['max_step']:.4g}@{t['max_step_at']}"
                f"  r_sm {t['r']['signed_mean']:.3f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
