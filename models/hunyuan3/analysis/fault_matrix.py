"""混元 3 · 第一阶段故障矩阵的薄包装。算法在 janus/analysis/fault_matrix.py（`janus fault-matrix`），
每个故障埋了什么、期望哪些检查报警写在同目录的 phase1-faults.yaml。

用法（与搬家前相同）：python models/hunyuan3/analysis/fault_matrix.py <faults 结果目录> > fault-matrix.md
（目录下每个故障运行一个子目录，里面是 janus run --test <运行> 的 results.json）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # 仓库根：没 pip install 也能 import janus

from janus.analysis.fault_matrix import main as _main

SPEC = Path(__file__).resolve().with_name("phase1-faults.yaml")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(_main([sys.argv[1], "--spec", str(SPEC)]))
