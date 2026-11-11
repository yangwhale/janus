"""混元 3 · loss 偏移归因的薄包装。算法与参数全在 janus/analysis/offsets.py（`janus offsets`），这里原样转过去。

用法（与搬家前相同）：
  python models/hunyuan3/analysis/offset_attribution.py --test <名>=<metrics.jsonl> \
      --golden <名>=<metrics.jsonl> [--golden ...] [--json <输出.json>]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # 仓库根：没 pip install 也能 import janus

from janus.analysis.offsets import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
