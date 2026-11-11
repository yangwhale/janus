"""report.md：从 results.json 渲染的 Markdown 报告（janus run 每次都写；--render-only 只重画它）。

数字只从 results.json 来，不重算、不手写。HTML 报告在 janus/report/html.py。
"""

from __future__ import annotations

from janus.catalog import HASH_ALGO, display_path, recorded_hash
from janus.checks.base import INCONCLUSIVE


def fmt(v) -> str:
    """报告表格里的数：浮点三位有效数字；字典（如 ULP 分桶）写成 键:值 的紧凑形式。"""
    if isinstance(v, float):
        return f"{v:.3e}"
    if isinstance(v, dict):
        return "{" + " ".join(f"{k}:{x:.3g}" if isinstance(x, float) else f"{k}:{x}" for k, x in v.items()) + "}"
    return str(v)


def _catalog_hash(entry: dict) -> str:
    """首屏的目录指纹：content-v1 写「内容指纹 …」；旧结果（schema_version 1）写原文件的「sha256 …」。"""
    algo, h = recorded_hash(entry)
    return f"内容指纹 {h[:12]}…" if algo == HASH_ALGO else f"sha256 {h[:12]}…"


def render(cat, results: dict) -> str:
    """report.md：首屏一句话结论、证据不足的单列、每条检查一行。数字全部来自 results.json。"""
    s = results["summary"]
    lines = [f"# janus 报告：{cat.data['model']} / {cat.data['stage']}", "",
             f"- 被测：`{cat.runs['test']}`" + (f"；LKG：`{cat.runs['lkg']}`" if cat.runs.get("lkg") else "")
             + (f"；golden：{', '.join(f'`{g}`' for g in cat.runs.get('golden', []))}" if cat.runs.get("golden") else ""),
             f"- janus {results['janus']['version']}；目录 `{display_path(cat.path)}`（{_catalog_hash(results['catalog'])}）",
             f"- 结果：pass {s['pass']}、fail {s['fail']}、inconclusive {s['inconclusive']}、error {s['error']}"
             + ("；**有硬检查失败**" if s["hard_fail"] else ""), ""]
    inc = [c for c in results["checks"] if c["status"] == INCONCLUSIVE]
    if inc:
        lines += ["**证据不足（inconclusive）**：" + "、".join(f"`{c['id']}`" for c in inc), ""]
    lines += ["| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |", "|---|---|---|---|---|---|"]
    for c in results["checks"]:
        m = c.get("metrics", {})
        key = ", ".join(f"{k}={fmt(v)}" for k, v in list(m.items())[:4])
        lines.append(f"| `{c['id']}` | {c['type']} | {c['level']} | {c['status']} | {key} | {c.get('message', '')} |")
    lines += ["", "数字全部来自 `results.json`，本报告不手写数字。"]
    return "\n".join(lines) + "\n"
