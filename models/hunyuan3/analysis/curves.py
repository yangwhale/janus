"""混元 3 · 逐步曲线的薄包装。算法在 janus/analysis/curves.py（`janus curves`），这里只放混元 3 的运行分组。

用法（与搬家前相同）：
  python models/hunyuan3/analysis/curves.py <series 目录> <输出目录>
      第一阶段 9 次 GPU 运行按设置分组互比 + 按设置分组的持续偏移；分组在同目录的 phase1-curves.yaml。
      等价于 janus curves groups <series 目录> <输出目录> --config models/hunyuan3/analysis/phase1-curves.yaml
  python models/hunyuan3/analysis/curves.py compare --test <名>=<metrics.jsonl> --golden <名>=<metrics.jsonl> ... \
      --out <输出目录> [--prefix L4] [--fields lm_loss grad_norm] [--steps 1:31]
      等价于 janus curves compare …
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # 仓库根：没 pip install 也能 import janus

from janus.analysis.curves import compare, load_file, rel, rolling  # noqa: F401  旧调用方按这些名字用
from janus.analysis.curves import main as _main

CONFIG = Path(__file__).resolve().with_name("phase1-curves.yaml")


def main(argv):
    if argv[:1] == ["compare"]:
        return _main(argv)
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    return _main(["groups", argv[0], argv[1], "--config", str(CONFIG)])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
