"""分组参数范数漂移：两个（或多个）带 group_sq.* 的运行目录逐步比，回答「param_norm 多涨的部分来自哪一组」。

薄包装：算法在 janus/analysis/drift.py（`janus drift groups`），这里只保留旧入口与混元 3 的默认值，命令与输出不变。

  python group_norm_drift.py --test <被测 metrics.jsonl> --ref <参照 metrics.jsonl> [--pair <另一参照>] [--steps 100,300,1000] [--out json]

group_sq.<组>[.<层>] 是该组 FP32 主权重的平方和（TPU：models/hunyuan3/frameworks/maxtext/train_hy3.py --group-norms；
GPU：janus/dump/megatron.py --group-norms，组名两侧一致）。把 param_norm 的差拆成各组的贡献：
‖p‖_test − ‖p‖_ref ≈ Σ_组 (sq_test − sq_ref) / (2‖p‖_ref)。expert_sq.<层> 是每个专家（fc1 + fc2）的平方和，另给
「多涨的部分」在专家之间的分布（是不是集中在少数专家）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # 仓库根：没 pip install 也能 import janus

from janus.analysis.drift import main_groups  # noqa: E402

if __name__ == "__main__":
    sys.exit(main_groups(sys.argv[1:]))
