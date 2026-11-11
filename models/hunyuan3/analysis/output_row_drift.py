"""输出层（与 embedding）逐行漂移：带 output_row_sq / output_row_u / embedding_row_sq 的运行逐行比，按词在数据里出现的次数分桶。

薄包装：算法在 janus/analysis/drift.py（`janus drift rows`），这里只保留旧入口与混元 3 的默认值，命令与输出不变。
回答「输出层多涨的平方和来自哪些词的那一行」以及「这些行的 Adam 更新量是不是更大」。

  python output_row_drift.py --test <被测 metrics.jsonl> --ref <参照> [--pair <另一参照>] --counts <label_counts.npz> [--key c300]
      [--steps 100,200,300] [--out json]

label_counts.npz：每个词在 golden 前 N 步 labels 里出现的次数（c300 = 前 300 步，c1000 = 前 1000 步）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # 仓库根：没 pip install 也能 import janus

from janus.analysis.drift import main_rows  # noqa: E402

if __name__ == "__main__":
    sys.exit(main_rows(sys.argv[1:]))
