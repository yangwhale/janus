"""pytest 共用设置：仓库根与 tests/ 进 sys.path。

- 仓库根：没 pip install 也能 import janus（CI 是装好的，本地直接跑也行）；
- tests/：测试里 `from helpers import …` 取共用的造数据函数。

测试目录照包的结构：tests/<子包>/test_<模块>.py；模型层脚本的测试在 tests/models/<模型>/；
文档与仓库卫生在 tests/docs/。pytest 用 importlib 模式（pyproject.toml），子目录不需要 __init__.py。
"""

import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
for p in (TESTS.parent, TESTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
