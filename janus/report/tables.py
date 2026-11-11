"""对齐实践的逐级大表：从 results.json 生成，不手写数字。

一行一条检查（L4 逐步量按指标展开），列：级、检查项、严格档、参考档、说明（实测｜严格档｜参考档）、证据。
哪些检查进表由调用方给（模型的 dashboard/items.yaml 里本实践那一列）；数字全部从结果文件里读。
严格档判定只来自结果里的 status；参考档由 janus/report/gates.py 按 reference_gates.yaml 从同一份 metrics 现算。
严格档符号：✅ pass ｜ 🟡 inconclusive（介于 N 与 2N，或参照不足）｜ ❌ fail / error ｜ ⬜ 还没跑 ｜ — 不适用。
"""
from __future__ import annotations

import fnmatch
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from janus.checks.base import MIN_NOISE_REFS
from janus.report import gates as G
from janus.report.gates import METRIC_ZH, SERIES_TYPES, fmt, measured, ratio  # noqa: F401  （measured / ratio 给别的模块按旧位置导入）

SYM = {"pass": "✅", "inconclusive": "🟡", "fail": "❌", "error": "❌", "todo": "⬜", "wip": "🟡", "na": "—"}
LEVEL_ORDER = ["L0", "L1", "L2", "L3", "L4"]
BEGIN = "<!-- janus:table:begin（以下由脚本生成，不要手改） -->"
END = "<!-- janus:table:end -->"
SERIES_KEYS = ("signed_mean", "mare", "max_step")


@dataclass
class Row:
    level: str
    item: str
    metric: str = "—"
    noise: str = "—"
    threshold: str = "—"
    value: str = "—"
    ratio: str = "—"
    status: str = "todo"
    evidence: list = field(default_factory=list)
    check_id: str = ""
    ref: G.RefVerdict | None = None      # 参考档；None = 不适用（还没跑、演练）
    note: str = ""                       # 说明栏：实测｜严格档｜参考档

    def md(self) -> str:
        ev = "<br>".join(self.evidence) if self.evidence else "—"
        name = f"{self.item}<br>`{self.check_id}`" if self.check_id else self.item
        ref = self.ref.sym if self.ref is not None else "—"
        return f"| {self.level} | {name} | {SYM[self.status]} | {ref} | {self.note or self.value} | {ev} |"


def exact_summary(metrics: dict) -> str:
    keys = [k for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    pref = [k for k in ("differ", "only_one_side", "wrong_dtype", "unmatched", "bad", "bad_steps", "problems", "fake", "same",
                        "inconsistent", "must_equal_differ",
                        "params", "tensors", "units", "entries", "fields", "steps", "objects", "rules", "cases") if k in keys]
    return "<br>".join(f"{k}={fmt(metrics[k])}" for k in (pref or keys)[:4]) or "见报告"


def _finish(row: Row, c: dict, metric: str | None = None, only: str | None = None, per_metric: bool = False) -> Row:
    """填参考档与说明栏。per_metric：本行只代表 metric 这一个指标（判定词用本行的逐指标判定）。"""
    row.ref = G.reference_verdict(c, only=only)
    mtext, stext = G.strict_parts(c, metric, row.status if per_metric else None)
    row.note = G.note(mtext, stext, row.ref)
    return row


def rows_for_check(c: dict, item: str, evidence: list) -> list[Row]:
    th = c.get("threshold") or {}
    kind = th.get("kind")
    m = c.get("metrics") or {}
    base = dict(level=c.get("level", "—"), item=item, status=c.get("status", "error"), evidence=evidence, check_id=c["id"])
    if kind == "calibrated":
        refs = (th.get("noise") or {}).get("refs")
        cand = G.calibrated_metrics(c)
        if not cand:
            return [_finish(Row(**base, threshold="≤ N 过，> 2N 不过", value="参照不足"), c)]
        series = c.get("type") in SERIES_TYPES
        keep = ([t for t in cand if t[0] in SERIES_KEYS] or cand) if series else [G.worst_calibrated(c)]
        out = []
        for k, n, x, r in keep:
            row = Row(**base, metric=METRIC_ZH.get(k, k), noise=f"N={fmt(n)}<br>{refs} 对",
                      threshold=f"≤ {fmt(n)} 过，> {fmt(2 * n if n is not None else None)} 不过",
                      value=fmt(x), ratio=(f"{r:.2f}×N" if r is not None and math.isfinite(r) else fmt(r)))
            many = len(keep) > 1 and r is not None and c.get("status") not in ("error",)
            if many:
                # 逐个指标按同一规则判（janus 对每个指标各判一次再取最差，整条检查的状态见报告）
                row.status = "fail" if r > 2 else ("pass" if r <= 1 and (refs or 0) >= MIN_NOISE_REFS else "inconclusive")
            out.append(_finish(row, c, k if series else None, only=k if series else None, per_metric=many))
        if series:
            d0 = m.get("d0_max", m.get("d0"))
            if isinstance(d0, (int, float)):
                out.insert(0, _finish(Row(**{**base, "status": "na"}, metric=METRIC_ZH["d0"], threshold="只报，不判",
                                          value=f"{d0:+.2e}" if d0 else "0"), c, "d0", only="d0", per_metric=True))
        return out
    if kind == "numeric":
        crit = G.numeric_criteria(c)
        known = [q for q in crit if q[2] is not None]
        if crit and known:
            k, t, x, r = max(known, key=lambda q: -1 if q[3] is None else q[3])
            return [_finish(Row(**base, metric=METRIC_ZH.get(k, k), threshold=t, value=fmt(x),
                                ratio=f"{r:.2f}" if r is not None else "—"), c)]
        return [_finish(Row(**base, metric="逐项", threshold="按规则（见报告）", value=exact_summary(m)), c)]
    if kind == "exact":
        return [_finish(Row(**base, metric="逐位 / 逐项", threshold="逐位相同", value=exact_summary(m)), c)]
    return [_finish(Row(**base, metric="只诊断", threshold="不判（只报）", value=exact_summary(m)), c)]


class Results:
    """一个对齐实践文件夹里的结果文件：results/**/<名>.results.json。"""

    def __init__(self, align_dir: Path, shared: list[Path] = ()):
        """shared：别的实践文件夹。本实践没有、但别的实践有的结果文件（如同一份 TPU 检查点的转换核对）也能引用，本实践优先。"""
        self.dir = Path(align_dir)
        self.files = {}
        for d in [*shared, self.dir]:
            for f in sorted((Path(d) / "results").rglob("*.results.json")):
                self.files[f.name[: -len(".results.json")]] = f
        self.own = {n for n, f in self.files.items() if self.dir in f.parents}

    def load(self, name):
        f = self.files.get(name)
        if f is None:
            return None, None
        d = json.loads(f.read_text())
        return (d if "checks" in d else None), f

    def checks(self, src: str):
        name, _, pat = src.partition(":")
        d, f = self.load(name)
        if d is None:
            return None, []
        return f, [c for c in d["checks"] if fnmatch.fnmatchcase(c["id"], pat or "*")]


def evidence_links(result_file: Path, readme_dir: Path, check_id: str, figures: dict) -> list:
    rep = result_file.with_name(result_file.name.replace(".results.json", ".report.md"))
    target = rep if rep.exists() else result_file
    rel = Path(os.path.relpath(target, readme_dir)).as_posix()
    out = [f"[{result_file.name.replace('.results.json', '')}]({rel})"]
    for pat, figs in figures.items():
        if fnmatch.fnmatchcase(check_id, pat):
            # 链接文字写短（图名末段：overlay / pairdiff / mean …），长文件名会把证据栏撑得很宽
            out += [f"[图·{Path(g).stem.rsplit('-', 1)[-1]}]({Path(g).as_posix()})" for g in figs]
    return out


def drill_note(c: dict, fired: bool) -> str:
    mtext, _ = G.strict_parts(c)
    word = G.STRICT_WORD.get(c.get("status"), c.get("status"))
    return f"演练（期望报警）：{'按期望报警' if fired else '该报警没报'}｜本条原判定 {word}：{mtext}"


STRICT_ORDER = ("error", "fail", "inconclusive", "pass")


def _strict_key(c: dict):
    st = c.get("status", "error")
    w = G.worst_calibrated(c) if (c.get("threshold") or {}).get("kind") == "calibrated" else None
    if w is None and (c.get("threshold") or {}).get("kind") == "numeric":
        rs = [q[3] for q in G.numeric_criteria(c) if q[3] is not None]
        w = (None, None, None, max(rs)) if rs else None
    r = w[3] if w and w[3] is not None and math.isfinite(w[3]) else (math.inf if w and w[3] is not None else -1)
    if "drill_judged" in (c.get("metrics") or {}):
        r = -2          # 演练用例（故意改坏、超线才算过）：倍数是反着的，不拿来当「最差」
    return (STRICT_ORDER.index(st) if st in STRICT_ORDER else 0, -r)


def summarize_checks(cs: list[dict], expect: str | None = None) -> tuple[str, G.RefVerdict | None, str]:
    """总览一格（一项检查 × 一个实践）：→ (严格档状态, 参考档, 说明)。
    选中多条检查时严格档取最差（同级按离门槛的倍数），参考档取最差（不过 > 未记录 > 部分 > 过 > 无门槛），说明写最差那条。
    expect = "fail"：演练，有一条报警才算过，参考档不适用。"""
    if expect == "fail":
        fired = any(c.get("status") == "fail" for c in cs)
        pick = next((c for c in cs if c.get("status") == "fail"), cs[0])
        return ("pass" if fired else "fail"), None, drill_note(pick, fired)
    worst = min(cs, key=_strict_key)
    st = worst.get("status", "error")
    refs = [(G.reference_verdict(c), c) for c in cs]
    ref, rc = min(refs, key=lambda t: (G.REF_ORDER.index(t[0].status), t[1] is not worst))   # 同级时用严格档最差的那条
    st_ref = G.combine_status([v.status for v, _ in refs])
    if st_ref != ref.status:
        # 例：几个算子里有的判过、有的没有同输入的对照 → 格子给 ☑️，说明栏写没记录的那条并注明其余判过
        n_ok = sum(1 for v, _ in refs if v.status in ("pass", "partial"))
        ref = G.RefVerdict(st_ref, f"{ref.text}；另 {n_ok} 条判过")
    mtext, stext = G.strict_parts(worst)
    many = f"{len(cs)} 条取最差 `{worst['id']}`：" if len(cs) > 1 else ""
    rtext = ref.text if (rc is worst or len(cs) == 1 or ref.status == "none") else f"`{rc['id']}` {ref.text}"
    note = f"{many}实测 {mtext}｜严格档 {stext}｜参考档 {rtext}"
    return st, ref, note


def overview_cell(cs: list[dict], expect: str | None = None) -> dict:
    """总览一格的全部材料：summarize_checks 的 (status, ref, note)，加最差那条检查的短写法（G.cell_measure）、
    参考档门槛短写法、选中几条检查。"""
    st, ref, note = summarize_checks(cs, expect)
    worst = min(cs, key=_strict_key)
    out = {"status": st, "ref": ref, "note": note, "n": len(cs), "check_id": worst["id"],
           "ref_thr": G.ref_threshold_text(worst["id"]), **G.cell_measure(worst)}
    if expect == "fail":
        fired = st == "pass"
        out.update(metric="演练", value="按期望报警" if fired else "该报警没报", how="", thr="期望报警", short_thr="", ref_thr="—")
    return out


def build_rows(items: list[dict], results: Results, figures: dict | None = None) -> list[Row]:
    """items：[{group, name, src, expect?}]，src 是「结果文件名:检查 id 通配」或 todo / wip / na。
    expect = "fail"：故障演练，选中的检查里有一条 fail 才算这一项通过（判定翻过来，原判定写进说明）。"""
    figures = figures or {}
    rows = []
    for it in items:
        src = it["src"]
        level = it.get("level", "—")
        if src.startswith("wip:"):
            rows.append(Row(level=level, item=it["name"], status="wip", value="进行中", evidence=[src[4:].strip()]))
            continue
        if src.startswith("todo:"):  # 「todo: 说明」：还没跑，说明写在证据栏
            rows.append(Row(level=level, item=it["name"], status="todo", value="还没跑", evidence=[src[5:].strip()]))
            continue
        if src in ("todo", "wip", "na"):
            rows.append(Row(level=level, item=it["name"], status=src,
                            value={"todo": "还没跑", "wip": "进行中", "na": "不适用"}[src]))
            continue
        f, cs = results.checks(src)
        if f is None:
            rows.append(Row(level=level, item=it["name"], status="todo", value="无结果文件"))
            continue
        if not cs:
            rows.append(Row(level=level, item=it["name"], status="todo", value="结果里没有这条"))
            continue
        drill = it.get("expect") == "fail"
        fired = any(c.get("status") == "fail" for c in cs)
        for c in cs:
            got = rows_for_check(c, it["name"], evidence_links(f, results.dir, c["id"], figures))
            if drill:
                # 演练：埋了错，期望这些检查报警。有一条报了（fail）才算演练通过；原判定写进说明，参考档不适用
                for r in got:
                    r.note = drill_note(c, fired)
                    r.status, r.ref = ("pass" if fired else "fail"), None
            rows.extend(got)
    rows.sort(key=lambda r: LEVEL_ORDER.index(r.level) if r.level in LEVEL_ORDER else len(LEVEL_ORDER))
    return rows


def render(rows: list[Row], title: str = "逐级大表") -> str:
    cnt, rcnt = {}, {}
    for r in rows:
        cnt[r.status] = cnt.get(r.status, 0) + 1
        if r.ref is not None and r.ref.status != "none":
            rcnt[r.ref.status] = rcnt.get(r.ref.status, 0) + 1
    summary = "　".join(f"{SYM[s]} {cnt[s]}" for s in ("pass", "inconclusive", "wip", "fail", "error", "todo", "na") if s in cnt)
    rsum = "　".join(f"{G.REF_SYM[s]} {rcnt[s]}" for s in ("pass", "partial", "fail", "missing") if s in rcnt) or "无"
    md = [f"## {title}", "",
          "### 来源", "",
          "- 由 `models/<模型>/dashboard/build.py` 调 `janus/report/tables.py` 生成。",
          "- 数据来自本文件夹 `results/` 下的 `results.json`，数字不手写。", "",
          "### 两档门槛", "",
          "- 含义见 `docs/reference/judging.md`「两档门槛」。",
          "- **严格档**是验收判定：",
          "  - calibrated 检查对噪声上限 N 判：≤ N 过，> 2N 不过，中间待定；",
          "  - 参照不足 3 对最多待定；",
          "  - 逐位类必须逐位相同；",
          "  - 数值门槛（如 FP64 R 比）照目录里写的线。",
          "  - 符号：✅ 过 ｜ 🟡 待定 ｜ ❌ 不过或出错 ｜ ⬜ 还没跑 ｜ — 不适用。",
          "- **参考档**是同类 GPU→TPU 迁移项目的实践门槛：",
          "  - 只并列报告、不决定验收。",
          "  - 符号：✅ 过 ｜ ☑️ 记录了的都过，但有门槛项没记录 ｜ ❌ 不过 ｜ ❔ 所需指标未记录 ｜ — 无对应门槛。", "",
          "### 说明栏怎么读", "",
          "- 格式一律是「实测（指标）｜严格档 门槛 → 离门槛多远，判定｜参考档 门槛：实测（实测 / 门槛的倍数），判定」。",
          "- calibrated 检查先写主判指标 rel_L2：",
          "  - 没有 rel_L2 就写最差的指标；",
          "  - 最差的不是它时，补一句最差的那个。",
          "- L4 逐步量三个指标各占一行：",
          "  - 各自按同一规则判；",
          "  - 整条检查取最差，见报告。",
          "- 首步差：严格档只报不判，参考档要判（MARE＝平均绝对相对差）。",
          "- 进行中的项在证据栏写证据位置。",
          "- 严格档里「N=…（k 对）」的 k 是算 N 用的噪声参照对数：",
          "  - 一对是两份运行的一次比较，k 对不是 k 份运行；",
          "  - 用了哪几份运行、怎么配对见该检查项目录的 `runs.noise`。", "",
          "### 逐行结果", "",
          f"- 共 {len(rows)} 行。",
          f"- 严格档：{summary}",
          f"- 参考档（有对应门槛的 {sum(rcnt.values())} 行）：{rsum}", "",
          "| 级 | 检查项 | 严格档 | 参考档 | 说明 | 证据 |",
          "|---|---|---|---|---|---|"]
    md += [r.md() for r in rows]
    return "\n".join(md) + "\n"


def write_section(readme: Path, body: str) -> None:
    """把生成的内容放进 README 的标记之间；没有标记就插在第一个二级标题之前（没有二级标题就追加在末尾）。"""
    s = readme.read_text() if readme.exists() else ""
    block = f"{BEGIN}\n\n{body}\n{END}"
    if BEGIN in s and END in s:
        a, b = s.index(BEGIN), s.index(END) + len(END)
        s = s[:a] + block + s[b:]
    else:
        i = s.find("\n## ")
        s = (s[: i + 1] + block + "\n\n" + s[i + 1:]) if i >= 0 else (s.rstrip("\n") + "\n\n" + block + "\n")
    readme.write_text(s)
