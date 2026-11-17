"""mini 1000 步：TPU 运行主 loss 对 GPU `g3-a` 与对 GPU 8 份均值的带符号均值。

口径同 `alignments/01-mini/results/t1-1000step-loss-tpu-variants.json`：逐步 d_t = (TPU − 参照) / 参照，
参照是 `g3-a`，或 `g3-a … g3-h` 8 份同一步的算术平均；第 1–1000 步取平均。那份文件没有收 `t1zr-a`
（打开 `ihc_force_bf16_rounding`，台账 M36）；这里补算，并把 `t1zk-a` 一起算一遍，和那份文件核对口径（相对差记在 variants_file_rel_diff）。

用法（仓库根目录）：
  python models/hunyuan4/analysis/mini_loss_vs_gpu_mean.py --root <运行根> --runs t1zk-a t1zr-a \\
      --json models/hunyuan4/alignments/01-mini/results/t1zr-1000step-loss-vs-gpu-mean.json
运行根下每个运行一个目录，只读 <运行名>/metrics.jsonl（原始记录在 POC 存储
`janus/hunyuan4/tpu/<运行名>/`、`janus/hunyuan4/golden-t1/g3-*/`）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))  # 仓库根：没 pip install 也能 import janus

from janus.analysis.curves import load_file  # noqa: E402

GPU = [f"g3-{c}" for c in "abcdefgh"]
STEPS = range(1, 1001)
VARIANTS = ROOT / "models/hunyuan4/alignments/01-mini/results/t1-1000step-loss-tpu-variants.json"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="运行根：<运行名>/metrics.jsonl")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--json", required=True, help="结果文件")
    a = p.parse_args(argv)
    L = lambda n: load_file(Path(a.root) / n / "metrics.jsonl", "lm_loss")  # noqa: E731
    G = {n: L(n) for n in GPU}
    gm = {s: float(np.mean([G[n][s] for n in GPU])) for s in STEPS}
    known = json.loads(VARIANTS.read_text())["runs"]
    out = {
        "schema_version": 1,
        "what": "mini 1000 步主 loss 带符号均值：TPU 对 g3-a、对 GPU g3-a…h 8 份均值（第 1–1000 步，口径同 t1-1000step-loss-tpu-variants.json）",
        "steps": [STEPS[0], STEPS[-1]],
        "gpu_runs": GPU,
        "runs": {},
    }
    for n in a.runs:
        T = L(n)
        va = float(np.mean([(T[s] - G[GPU[0]][s]) / G[GPU[0]][s] for s in STEPS]))
        vm = float(np.mean([(T[s] - gm[s]) / gm[s] for s in STEPS]))
        row = {"signed_vs_g3a": va, "signed_vs_gpu_mean": vm}
        if n in known:
            # 8 份均值的求和顺序与原文件不同，末几位会差，只记相对差
            row["variants_file_rel_diff"] = max(
                abs(va / known[n]["signed_vs_g3a"] - 1), abs(vm / known[n]["signed_vs_gpu_mean"] - 1)
            )
        out["runs"][n] = row
        print(n, {k: (f"{v:.4e}" if isinstance(v, float) else v) for k, v in row.items()})
    Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
