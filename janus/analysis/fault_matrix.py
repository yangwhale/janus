"""故障矩阵：读每个故障运行的 results.json，出「哪条检查对哪种故障报警」的两张 Markdown 表。

工具先自证：在确定性基线上逐个埋错（改权重、错位数据、缩放梯度……），每个故障运行跑同一份目录，
再用这张矩阵核对「期望报警的检查都报了」。期望写在一份 YAML 里（每个模型、每组演练一份）::

    faults:
      f-weight:
        what: 第 2 层 linear_proj 的 FP32 主权重 ×(1+1e-3)
        expect: [L0.main_params.loaded, L3.step1.updated]

输入目录下每个故障一个子目录，里面是 ``janus run --test <故障运行>`` 的输出（``results.json``）。
第一张表每个故障一行：报警的检查数、期望抓到而没抓到的；第二张表检查 × 故障：
**F** = fail（报警），· = pass，? = inconclusive，E = error，✗ = 期望报警但没报。
用法见 ``janus fault-matrix --help``；混元 3 第一阶段的期望是 models/hunyuan3/analysis/phase1-faults.yaml。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MARK = {"pass": "·", "fail": "**F**", "inconclusive": "?", "error": "E"}


def load_spec(path) -> dict[str, dict]:
    """读期望：{故障运行名: {what, expect}}。"""
    import yaml

    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    faults = d.get("faults") if isinstance(d, dict) else None
    if not isinstance(faults, dict) or not faults:
        raise ValueError(f"{path}：要有非空的 faults: {{运行名: {{what, expect}}}}")
    for name, f in faults.items():
        if not isinstance(f, dict) or "what" not in f:
            raise ValueError(f"{path}：故障 {name} 缺 what")
        f.setdefault("expect", [])
    return faults


def render(results: dict[str, dict], faults: dict[str, dict]) -> str:
    """results = {故障运行名: results.json 的内容}（按 faults 的顺序取）。返回两张表的 Markdown。"""
    res = {f: results[f] for f in faults if f in results}
    if not res:
        raise ValueError("没有任何故障运行的结果")
    ids = [c["id"] for c in next(iter(res.values()))["checks"]]
    st = {f: {c["id"]: c["status"] for c in r["checks"]} for f, r in res.items()}
    lines = ["| 故障 | 埋了什么 | 报警的检查数 | 期望抓到而没抓到 |", "|---|---|---|---|"]
    for f in res:
        miss = [e for e in faults[f]["expect"] if st[f].get(e) != "fail"]
        n = sum(1 for i in ids if st[f].get(i) == "fail")
        lines.append(f"| `{f}` | {faults[f]['what']} | {n} / {len(ids)} | {'、'.join(miss) or '无'} |")
    lines += ["", "| 检查 | " + " | ".join(f"`{f}`" for f in res) + " |", "|---|" + "---|" * len(res)]
    for i in ids:
        cells = []
        for f in res:
            m = MARK.get(st[f].get(i), "")
            if i in faults[f]["expect"] and st[f].get(i) != "fail":
                m = "✗"
            cells.append(m)
        lines.append(f"| `{i}` | " + " | ".join(cells) + " |")
    lines += ["", "**F** = fail（报警），· = pass，E = error，✗ = 期望报警但没报。"]
    return "\n".join(lines)


def missed(results: dict[str, dict], faults: dict[str, dict]) -> dict[str, list[str]]:
    """每个故障期望报警而没报的检查 id（全部抓到时为空 dict）。"""
    out = {}
    for f, spec in faults.items():
        if f in results:
            st = {c["id"]: c["status"] for c in results[f]["checks"]}
            m = [e for e in spec["expect"] if st.get(e) != "fail"]
            if m:
                out[f] = m
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="janus fault-matrix", description="故障演练矩阵：哪条检查对哪种故障报警。")
    ap.add_argument("root", type=Path, help="结果目录：每个故障运行一个子目录，里面有 results.json")
    ap.add_argument("--spec", required=True, type=Path, help="期望 YAML：faults: {运行名: {what, expect}}")
    ap.add_argument("--strict", action="store_true", help="有期望报警而没报的故障时退出码 1")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    faults = load_spec(a.spec)
    results = {
        f: json.loads((a.root / f / "results.json").read_text())
        for f in faults
        if (a.root / f / "results.json").exists()
    }
    if not results:
        print(f"{a.root} 下没有任何故障运行的 results.json", file=sys.stderr)
        return 2
    print(render(results, faults))
    return 1 if a.strict and missed(results, faults) else 0


if __name__ == "__main__":
    sys.exit(main())
