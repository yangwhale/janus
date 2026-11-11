"""检查类型。每种类型一个模块，注册到 REGISTRY；规格见 docs/reference/checks/<type>.md。"""
from janus.checks import discrete, extra, meta, ops, series, tensor  # noqa: F401  注册各类型
from janus.checks.base import REGISTRY, CheckResult, Threshold, register  # noqa: F401
