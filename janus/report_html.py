"""旧模块名，已改为 `janus.report.html`（见 docs/dev/CHANGES-PATHS.md）。0.2 删除这个转发。"""

import importlib
import sys
import warnings

warnings.warn("janus.report_html 已改名 janus.report.html，旧名 0.2 删除", DeprecationWarning, stacklevel=2)
sys.modules[__name__] = importlib.import_module("janus.report.html")
