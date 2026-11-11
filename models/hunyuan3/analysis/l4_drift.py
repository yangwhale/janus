"""多步趋势统计：被测运行对每份 golden、golden 两两之间，逐步相对差 (test − ref) / |ref| 的

薄包装：算法在 janus/analysis/drift.py（`janus drift trend`），这里只保留旧入口与混元 3 的默认值，命令与输出不变。
带符号均值（±标准误）、平均绝对相对差、最大单步、分段 95 分位包络、累积均值、100 步滑动均值；
另比路由偏置统计 router.bias_{mean,std}.<层> 的最大差与末步差。

  python l4_drift.py --test <被测运行目录> --golden <g1k-a> <g1k-b> <g1k-c> --out <json> [--steps 1000]

判读：包络走平 = 不累积；累积 / 滑动均值一直往一边走 = 系统偏差。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # 仓库根：没 pip install 也能 import janus

from janus.analysis.drift import main_trend  # noqa: E402

if __name__ == "__main__":
    sys.exit(main_trend(sys.argv[1:]))
