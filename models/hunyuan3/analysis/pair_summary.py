"""几组运行之间两两比的汇总：每一组「对比」给出各逐步量在后段的相对差范围、第 N 步 param_norm 的绝对差范围。

薄包装：算法在 janus/analysis/drift.py（`janus drift pairs`），这里只保留旧入口与混元 3 的默认值，命令与输出不变。

用法：
  python models/hunyuan3/analysis/pair_summary.py --series <名>=<metrics.jsonl> ... \
      --pairs "<标签>:<组A>-<组B>" ... --groups <组>=<名>,<名>,... ... [--json <输出.json>]

  组内两两：--pairs "rc 两两:rc-rc"（组A 与组B 相同时取组内组合）；
  跨组：--pairs "rc − g1k:rc-g1k"（组A 每个对组B 每个，差 = A − B，分母 B）。
03 的 results/remat-pair-summary.json 就是这样算的（在仓库根目录跑）：
  D=models/hunyuan3/alignments; R=$D/03-gpu-remat-golden/results/series; S=$D/00-gpu-noise-and-drills/results/series
  python models/hunyuan3/analysis/pair_summary.py \
      --series rc-a=$R/rc-a.metrics.jsonl rc-b=$R/rc-b.metrics.jsonl rc-c=$R/rc-c.metrics.jsonl rc-d=$R/rc-d.metrics.jsonl \
               rc-e=$R/rc-e.metrics.jsonl g1k-a=$S/g1k-a.metrics.jsonl g1k-b=$S/g1k-b.metrics.jsonl g1k-c=$S/g1k-c.metrics.jsonl \
               t-a=$D/01-v5p-acceptance/results/series/t-a.metrics.jsonl \
      --groups rc=rc-a,rc-b,rc-c,rc-d,rc-e g1k=g1k-a,g1k-b,g1k-c tpu=t-a \
      --pairs "rc 两两:rc-rc" "g1k 两两:g1k-g1k" "rc − g1k:rc-g1k" "t-a − g1k:tpu-g1k" "t-a − rc:tpu-rc" \
      --json $D/03-gpu-remat-golden/results/remat-pair-summary.json
相对差 d_t =（A_t − B_t）/ |B_t|，与 offset_attribution.py、janus 的 loss_series 一致；分母为 0 的步跳过。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # 仓库根：没 pip install 也能 import janus

from janus.analysis.drift import main_pairs  # noqa: E402

FIELDS = [
    "lm_loss",
    "param_norm",
    "grad_norm",
    "router.bias_std.1",
    "router.bias_std.2",
    "router.bias_std.3",
    "router.bias_mean.1",
    "router.bias_mean.2",
    "router.bias_mean.3",
]  # 混元 3：第 1–3 层是 MoE 层

if __name__ == "__main__":
    sys.exit(main_pairs(["--fields", *FIELDS, *sys.argv[1:]]))
