"""检查清单（CHECKS.md）：每一级比了什么、对象是哪些、怎么判、各平台的结果，一页列全。

清单的骨架（每级的目的、每一项比什么、对象是哪些模块 / 张量、原因说明）由模型自己的 ``checks.yaml`` 写；
判法、实测、判定、比较单元个数全部从结果文件现读（``janus/report/tables.py`` 与 ``gates.py`` 同一套写法），不手写数字。

``checks.yaml`` 的格式：

- ``columns``：平台列，``{键: 列名}``；行里 ``src`` 用同样的键。
- ``levels``：每级一节，``{id, title, lead, tables, modules?}``。
  - ``lead``：这一级的说明（Markdown 行），可以用 ``{rows}`` ``{checks}`` ``{units}`` ``{modules}`` 引用本级的计数，``{cols}`` 是这几个计数对应的列名。
  - ``tables``：``{title?, lead?, columns?, rows}``；``columns`` 不写就用全局的平台列。
  - 每行 ``{name, what, objects, src: {列: "结果文件名:检查 id 通配"}, why?}``。
    ``why`` 是非 ✅ 格子的原因（一句话 + 去哪看）；有非 ✅ 格子却没写 ``why`` 时生成失败，免得出现没人解释的红格。
    ``info: true``：不在验收路径上的行（对照、可选配置、只记事实、定责用的比较），名字后标「（只报）」，不进判定计数。
    ``expect: fail``：演练行，选中的检查报警才算过。
    ``unit: 单位``：覆盖按结果计数键推出的比较单元单位（键名与实际对象不符时用）。
  - ``modules``：只给 L1 这类「按模块列」的级：``{lead?, columns, rows: [{name, does, checks: {列: [src, …]}}]}``，
    生成「模型的每个算子由哪条检查覆盖」的对照表；``same_as_above: true`` 时与上一行同列相同的格子写「同 #k」。
- ``unlisted_reasons``：``{结果文件名或通配: 原因}``；``unlisted_group: true`` 时原因相同的文件并成一行（不列判定计数）。
- ``gpu``：可选，GPU 一侧的检查（参照本身是否可信、工具能不能抓到错），格式同一级。

比较单元：一条检查比了多少个张量 / 用例 / 步，从结果里的计数取（``unit_count``）；
同一批张量被几条检查各比一次时分别计数，所以是「比较次数」，不是去重后的张量数。
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
from pathlib import Path

from janus.report import gates as G
from janus.report import tables as AL

SYM = {"pass": "✅", "inconclusive": "🟡", "fail": "❌", "error": "❌", "todo": "⬜", "na": "—"}
# 结果里的计数键 → 单位（按这个顺序取第一个有的）
UNIT_KEYS = [
    ("params", "个参数张量"),
    ("tensors", "个张量"),
    ("cases", "个用例"),
    ("units", "个数据单元"),
    ("entries", "项 dtype"),
    ("objects", "个对象"),
    ("rules", "条规则"),
    ("fields", "个字段"),
    ("steps", "步"),
    ("overlap_steps", "步"),
]
SERIES_TYPES = {"loss_series", "multi_golden", "router_stats"}


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def unit_count(c: dict) -> tuple[int, str] | None:
    """一条检查比了多少个单元：→ (个数, 单位)；结果里没有可数的 → None。"""
    cov = c.get("coverage") or {}
    if _num(cov.get("compared")):
        return int(cov["compared"]), "个名字"
    m = c.get("metrics") or {}
    if c.get("type") in SERIES_TYPES and isinstance(m.get("window"), str):
        a, _, b = m["window"].partition(":")
        if a.isdigit() and b.isdigit():
            return int(b) - int(a), "步"
    for k, unit in UNIT_KEYS:
        if _num(m.get(k)) and m.get(k):
            return int(m[k]), unit
    return None


def add_units(acc: dict, cs: list[dict], unit: str | None = None) -> None:
    """unit：行里写的 ``unit``，覆盖按计数键推出的单位（结果里的键名与实际对象不符时用，如 entries 实为参数张量）。"""
    for c in cs:
        u = unit_count(c)
        if u:
            k = unit or u[1]
            acc[k] = acc.get(k, 0) + u[0]


TENSOR_UNITS = ("个参数张量", "个张量", "个用例")  # 一个用例就是一个张量（一个算子输出、一个观测点）


def tensor_total(acc: dict) -> int:
    """张量级比较次数：参数张量、张量、用例三种单位相加。"""
    return sum(v for u, v in acc.items() if u in TENSOR_UNITS)


def other_units(acc: dict) -> str:
    return units_text({u: v for u, v in acc.items() if u not in TENSOR_UNITS})


def units_text(acc: dict) -> str:
    order = [u for _, u in UNIT_KEYS] + ["个名字"]
    keys = sorted(acc, key=lambda u: order.index(u) if u in order else len(order))
    return "、".join(f"{acc[u]:,} {u}" for u in keys) or "—"


class Sources:
    """一个模型的全部结果文件（alignments/*/results/**/<名>.results.json），按文件名找；文件名全模型唯一。

    ``virtual``：``{名: (原始文件, [检查, …])}``，模型的 build.py 从别的结果文件现算出来的检查（见 ``janus/report/overview.py``），
    和结果文件一样按名字引用，证据链接指向原始文件。结果名可以带 ``*``：取排序后最后一个匹配的。"""

    def __init__(self, model_dir: Path, virtual: dict | None = None):
        self.files = {}
        for f in sorted(Path(model_dir).glob("alignments/*/results/**/*.results.json")):
            name = f.name[: -len(".results.json")]
            if name in self.files:
                raise SystemExit(f"结果文件重名：{self.files[name]} 与 {f}")
            self.files[name] = f
        self._cache = {}
        self.virtual = set()
        for name, (f, cs) in (virtual or {}).items():
            if name in self.files:
                raise SystemExit(f"现算的检查与结果文件重名：{name}")
            self.files[name] = Path(f)
            self._cache[Path(f), name] = list(cs)
            self.virtual.add(name)
        self.evidence: list[tuple[str, str]] = []  # (结果文件名通配, 证据来源)，render() 从 checks.yaml 填

    def resolve(self, name: str) -> str | None:
        if "*" not in name:
            return name if name in self.files else None
        hits = sorted(n for n in self.files if fnmatch.fnmatchcase(n, name))
        return hits[-1] if hits else None

    def checks(self, src: str) -> tuple[Path | None, list[dict]]:
        name, _, pat = src.partition(":")
        name = self.resolve(name) or name
        f = self.files.get(name)
        if f is not None and name in self.virtual:
            return f, [c for c in self._cache[f, name] if fnmatch.fnmatchcase(c["id"], pat or "*")]
        if f is None:
            return None, []
        if f not in self._cache:
            d = json.loads(f.read_text())
            self._cache[f] = d.get("checks") or []
        return f, [c for c in self._cache[f] if fnmatch.fnmatchcase(c["id"], pat or "*")]


def _link(f: Path, out_dir: Path) -> str:
    rep = f.with_name(f.name.replace(".results.json", ".report.md")) if f.name.endswith(".results.json") else f
    return Path(os.path.relpath(rep if rep.exists() else f, out_dir)).as_posix()


def _srcs(v) -> list[str]:
    return [v] if isinstance(v, str) else list(v or [])


PENDING = {"todo": "⬜ 未开始", "wip": "🟡 进行中"}


def _pending(s: str) -> tuple[str, str] | None:
    """「todo: 说明」「wip: 说明」：还没有结果的格子 → (状态, 说明)；别的写法 → None。"""
    for k in PENDING:
        if s == k or s.startswith(k + ":"):
            return k, s[len(k) + 1 :].strip()
    return None


def _result_name(f: Path, name_hint: str) -> str:
    return f.name[: -len(".results.json")] if f.name.endswith(".results.json") else name_hint


class Cell:
    """一行 × 一个平台：选中的检查、判定、短写法、证据链接。"""

    def __init__(self, srcs: list[str], sources: Sources, out_dir: Path, expect: str | None = None):
        self.checks, self.links, self.pairs, self.pending = [], [], [], []
        for s in srcs:
            p = _pending(s)
            if p:
                self.pending.append(p)
                continue
            f, cs = sources.checks(s)
            if f is None or not cs:
                raise SystemExit(f"检查清单引用的结果不存在：{s}")
            name = sources.resolve(s.partition(":")[0]) or s.partition(":")[0]
            self.checks += cs
            self.pairs += [(name, c["id"]) for c in cs]
            self.links.append((_result_name(f, name), _link(f, out_dir)))
        self.info = AL.overview_cell(self.checks, expect) if self.checks else None

    @property
    def status(self) -> str:
        if self.info:
            return self.info["status"]
        return self.pending[0][0] if self.pending else "na"

    def text(self, canon: dict | None) -> str:
        if not self.info:
            if self.pending:
                return "<br>".join(PENDING[k] + (f"（{n}）" if n else "") for k, n in self.pending)
            return "—"
        i = self.info
        q = []
        if (canon and i["metric"] != canon["metric"]) or i.get("how", "").endswith("×N"):
            q.append(i["metric"])  # 对噪声判的格子一律写出是哪个判定量（三个判定量各有自己的 N）
        if i["how"]:
            q.append(i["how"])
        if canon and i["thr"] != canon["thr"] and i["short_thr"]:
            q.append(i["short_thr"])
        worst = self._worst_metric()
        if worst:
            q.append(worst)
        value = re.sub(r"（[a-z_]+）$", "", str(i["value"]))
        counts = [G.exact_counts(c) for c in self.checks]
        if len(self.checks) > 1 and all(counts) and len({u for _, _, u in counts}) == 1 and i["status"] == "pass":
            # 几条逐位检查合在一格：写合计，不写其中一条（「0 / 1 不同」看不出是哪条）
            value = f"{sum(b for b, _, _ in counts):,} / {sum(n for _, n, _ in counts):,} 不同"  # 「逐位全中（bit_match_frac）」只留结论
        if value.endswith("…"):  # 总览格子里截短的说明，清单里写全一些
            c = next((c for c in self.checks if c["id"] == i.get("check_id")), None)
            value = G.exact_measured(c) if c else value
        txt = value + (f"（{'，'.join(q)}）" if q else "") + " " + SYM.get(i["status"], i["status"])
        ref = i["ref"]
        if ref is not None and ref.status != "none":
            txt += f"｜参{ref.sym}"
        name, href = self.links[0]
        return f"[{txt}]({href})"

    def _worst_metric(self) -> str:
        """格子写的是 rel_L2、而最差的指标是别的（如 1−cos）且离 N 更远时，补一句最差的那个，免得读者看见 2.08×N 却不知道还有 4.45×N。"""
        i = self.info
        if not i or not i.get("how", "").endswith("×N"):
            return ""
        c = next((c for c in self.checks if c["id"] == i.get("check_id")), None)
        w = G.worst_calibrated(c) if c else None
        if not w or w[3] is None or G.METRIC_ZH.get(w[0], w[0]) == i["metric"]:
            return ""
        shown = float(i["how"][:-2]) if i["how"][:-2].replace(".", "", 1).isdigit() else None
        if shown is not None and w[3] <= shown:
            return ""
        return f"最差 {G.METRIC_ZH.get(w[0], w[0])} {G.fmt_ratio(w[3])}×N"


# 每种检查类型「一个数是怎么聚合出来的」。行里写了 agg 时用行里的。
AGG = {
    "tensor_tolerance": "逐张量算（每个张量拉直成一个向量），每个指标取这批里最差的张量，各自对 N 判",
    "op_replay": "逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例",
    "optimizer_step": "逐张量算更新量的 rel_L2，取最差的张量",
    "discrete_agreement": "逐 token 比选中的 top-8 集合，算集合相同的比例，取最差的层",
    "multi_golden": "每步一个标量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步，各自对 N 判",
    "loss_series": "每步一个标量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步，各自对 N 判",
    "router_stats": "每步每层一个统计量，算逐步相对差，再在窗口内算带符号均值、MARE、最大单步",
    "scalar_series": "逐步比，任一步超出容差就不过",
    "health_signals": "逐步看 NaN / Inf 与突跳",
    "bitwise_equal": "逐张量按字节比，数不同的个数",
    "checkpoint_equal": "逐张量比 sha256，数不同的个数",
    "data_fingerprint": "逐单元比 sha256，数不同的个数",
    "dtype_manifest": "逐项核 dtype，数不合规的个数",
    "config_diff": "逐字段比，数不同的个数",
    "artifact_check": "逐条规则核，数不满足的条数",
    "coverage": "比名字清单，数缺的个数",
    "fake_fp32": "逐张量看低 16 位全 0 的比例，取最高的张量",
    "determinism": "逐步、逐张量比两次运行，数不可复现的项",
    "overlap_replay": "逐步比重叠的步，数不同的步",
    "replica_consistency": "逐对象比各副本，数不一致的个数",
    "poison_selftest": "逐用例看埋的错有没有被抓到",
}


def agg_text(row: dict, cells: dict) -> str:
    if row.get("agg"):
        return str(row["agg"])
    kinds = []
    for c in cells.values():
        for ch in c.checks:
            t = agg_for_check(ch)
            if t and t not in kinds:
                kinds.append(t)
    return "；".join(kinds)


SERIES_ALL = ("multi_golden", "loss_series", "router_stats")


def agg_for_check(c: dict) -> str:
    """一条检查的聚合方式：按检查类型与门槛种类（exact / numeric / calibrated / 只诊断）写，指标名取自门槛。"""
    t = c.get("type", "")
    th = c.get("threshold") or {}
    kind = th.get("kind")
    vals = th.get("values") if isinstance(th.get("values"), dict) else {}
    names = "、".join(G.METRIC_ZH.get(k, k) for k in vals) or "指标"
    if t == "op_replay":
        if kind == "numeric" and any(k.startswith("r_ratio") for k in vals):
            return "逐用例算 R 比（一个用例是一个算子输出或观测点，整个张量拉直），取最差的用例"
        if kind == "numeric":
            return f"逐用例算 {names}，取最差的用例"
        if kind == "exact":
            return "逐用例比，数不一致的用例"
        return "逐用例算，只报不判"
    if t in SERIES_ALL and kind == "numeric":
        return f"每步一个标量，算逐步相对差，{names} 对固定门槛判"
    if t in ("tensor_tolerance", "optimizer_step") and kind == "numeric":
        return f"逐张量算 {names}，取最差的张量，对固定门槛判"
    if kind == "exact" and t in ("loss_series", "tensor_tolerance"):
        return "逐步 / 逐张量按位比，数不同的个数"
    return AGG.get(t, "")


def evidence_source(row: dict, cells: dict, sources: Sources) -> str:
    """证据来源：行里写了 source 用行里的，否则按引用的结果文件名在 checks.yaml 的 evidence_sources 里找（通配，按顺序取第一个）。
    找不到就失败：每一行都要说清证据来自生产程序、回放还是离线计算。"""
    if row.get("source"):
        return str(row["source"])
    out = []
    for c in cells.values():
        for name, _ in c.links:
            hit = next((v for pat, v in sources.evidence if fnmatch.fnmatchcase(name, pat)), None)
            if hit is None and name.endswith(".json"):
                hit = next((v for pat, v in sources.evidence if fnmatch.fnmatchcase(name[:-5], pat)), None)
            if hit is None:
                raise SystemExit(f"检查清单「{row['name']}」引用的 {name} 在 evidence_sources 里没有对应的证据来源")
            if hit not in out:
                out.append(hit)
    return "<br>".join(out) or "—"


def _rule_text(canon: dict, cells: dict) -> str:
    """判法栏：严格档写明 N 是哪个判定量的（三个判定量各有一个 N）；参考档栏没有线、而某一格有参考档判定时，不写成「无」。"""
    thr = canon["thr"]
    m = re.match(r"^≤ N=(\S+) 过，> 2N 不过$", thr)
    if m:
        thr = f"≤ N 过，> 2N 不过（{canon['metric']} 的 N = {m.group(1)}）"
    ref = canon["ref_thr"]
    if ref == "无" and any(
        c.info and c.info["ref"] is not None and c.info["ref"].status != "none" for c in cells.values()
    ):
        ref = "部分检查有，见格子"
    return f"{thr}｜参考：{ref}"


def _canonical(cells: dict) -> dict | None:
    got = [c.info for c in cells.values() if c.info and c.info.get("metric")]
    if not got:
        return None
    count = {}
    for i in got:
        count.setdefault((i["metric"], i["thr"]), []).append(i)
    return max(count.values(), key=len)[0]


def _ids(cells: dict) -> str:
    ids = []
    for c in cells.values():
        for ch in c.checks:
            if ch["id"] not in ids:
                ids.append(ch["id"])
    return "<br>".join(f"`{i}`" for i in ids[:4]) + ("<br>…" if len(ids) > 4 else "")


def render_table(tab: dict, columns: dict, sources: Sources, out_dir: Path, stats: dict) -> list[str]:
    cols = tab.get("columns") or list(columns)
    md = []
    if tab.get("title"):
        md += [f"### {tab['title']}", ""]
    if tab.get("lead"):
        md += list(tab["lead"]) + [""]
    md += [
        "| 检查 | 比什么 | 对象 | 判法（严格｜参考） | 证据来源 | " + " | ".join(columns[c] for c in cols) + " |",
        "|---|---|---|---|---|" + "---|" * len(cols),
    ]
    notes = []
    for row in tab["rows"]:
        cells = {c: Cell(_srcs(row["src"].get(c)), sources, out_dir, row.get("expect")) for c in cols}
        canon = _canonical(cells)
        rule = _rule_text(canon, cells) if canon else "—"
        units = {}
        for c in cols:
            u = {}
            add_units(u, cells[c].checks, row.get("unit"))
            units[c] = units_text(u) if u else ""
            st = stats.setdefault(c, {"checks": set(), "units": {}, "status": {}})
            st["checks"].update(cells[c].pairs)
            add_units(st["units"], cells[c].checks, row.get("unit"))
            if cells[c].info:
                key = "info" if row.get("info") else cells[c].status
                st["status"][key] = st["status"].get(key, 0) + 1
        distinct = {v for v in units.values() if v}
        if len(distinct) == 1:
            utext = f"比较单元：{distinct.pop()}"
        else:
            utext = "比较单元：" + "；".join(f"{columns[c]} {v}" for c, v in units.items() if v) if distinct else ""
        objects = row.get("objects", "")
        agg = agg_text(row, cells)
        obj = objects + (f"<br>{utext}" if utext else "") + (f"<br>聚合：{agg}" if agg else "")
        name = row["name"] + ("（只报）" if row.get("info") else "")
        md.append(
            f"| {name}<br>{_ids(cells)} | {row['what']} | {obj} | {rule} | {evidence_source(row, cells, sources)} | "
            + " | ".join(cells[c].text(canon) for c in cols)
            + " |"
        )
        bad = [c for c in cols if cells[c].info and cells[c].status != "pass"]
        if bad:
            if not row.get("why"):
                raise SystemExit(f"检查清单「{row['name']}」有非 ✅ 格子（{', '.join(bad)}），要在 checks.yaml 写 why")
            marks = "、".join(f"{columns[c]} {SYM[cells[c].status]}" for c in bad)
            why = re.sub(r"([；。，：）」])\s+", r"\1", " ".join(str(row["why"]).split()))
            notes.append(f"- **{row['name']}**（{marks}）：{why}")
        stats.setdefault("_rows", 0)
        stats["_rows"] += 1
    md.append("")
    if notes:
        md += ["非 ✅ 格子的原因：", ""] + notes + [""]
    return md


STATUS_ORDER = ("error", "fail", "inconclusive", "pass")


def render_modules(mod: dict, columns: dict, sources: Sources, out_dir: Path) -> tuple[list[str], int, dict]:
    """「模型的每个算子由哪条检查覆盖」：一行一个算子，格子里是覆盖它的检查与判定。
    → (Markdown 行, 有 ``count`` 列检查覆盖的算子数, 按判定的算子计数)。

    ``status: <列名>``（可选）：表末加一列，写这一行 ``count`` 列里全部检查的最差判定；没有就写「尚未覆盖」。
    行里写 ``attributed: 说明`` 时，非 ✅ 的判定后加「已定责」（说明写在「说明」一类的列里）。
    计数 ``{"pass", "open", "none", "attributed"}``：全过 / 有待定或不过 / 尚未覆盖 / 其中已定责。
    表前的说明（``lead``）由 render_section 写，可以用 ``{modules_total}`` 等计数。"""
    cols = mod["columns"]
    md = []
    status_col = mod.get("status")
    heads = list(cols.values()) + ([status_col] if status_col else [])
    md += ["| # | 算子 | 做什么 | " + " | ".join(heads) + " |", "|---|---|---|" + "---|" * len(heads)]
    covered = 0
    tally = {"pass": 0, "open": 0, "none": 0, "attributed": 0}
    # same_as_above：格子与上一行同一列完全相同时写「同 #k」（k 是第一次出现的行号），省掉重复的链接与读数
    same = bool(mod.get("same_as_above"))
    prev: dict = {}
    for k, row in enumerate(mod["rows"], 1):
        cells = []
        hit = False
        worst = []
        for c in cols:
            parts = []
            for s in _srcs((row.get("checks") or {}).get(c)):
                if sources.resolve(s.partition(":")[0]) is None:  # 不是「结果文件名:检查 id」：原样写的说明
                    parts.append(s)
                    continue
                f, cs = sources.checks(s)
                if f is None or not cs:
                    raise SystemExit(f"算子对照表引用的结果不存在：{s}")
                st = AL.summarize_checks(cs)[0]
                ids = "、".join(dict.fromkeys(ch["id"] for ch in cs))
                # R 比类检查带了两边对参照的 rel_L2 时，判定后并排写「GPU a、TPU b → R」（取 R 最大的那条）
                rc = [
                    ch
                    for ch in cs
                    if {"r_ratio", "rel_l2_gpu_vs_ref", "rel_l2_tpu_vs_ref"} <= set(ch.get("metrics") or {})
                ]
                meas = f"（{G.cell_measure(max(rc, key=lambda ch: ch['metrics']['r_ratio']))['value']}）" if rc else ""
                parts.append(f"[`{ids}`]({_link(f, out_dir)}) {SYM[st]}{meas}")
                if c in mod.get("count", []):
                    hit = True
                    worst.append(st)
            text = "<br>".join(parts) or "—"
            if same and text != "—" and prev.get(c, (None,))[0] == text:
                cells.append(f"同 #{prev[c][1]}")
            else:
                cells.append(text)
                prev[c] = (text, k)
            if same and text == "—":
                prev.pop(c, None)
        covered += hit
        if status_col:
            if worst:
                w = min(worst, key=lambda x: STATUS_ORDER.index(x) if x in STATUS_ORDER else 0)
                tally["pass" if w == "pass" else "open"] += 1
                att = w != "pass" and row.get("attributed")
                tally["attributed"] += bool(att)
                cells.append(SYM[w] + (" 已定责" if att else ""))
            else:
                tally["none"] += 1
                cells.append("尚未覆盖")
        md.append(f"| {k} | {row['name']} | {row['does']} | " + " | ".join(cells) + " |")
    return md + [""], covered, tally


def _fmt_status(st: dict) -> str:
    """判定计数：只数验收路径上的行；标了 info 的行（只报、对照、可选配置）另计「只报 N」。"""
    s = st.get("status", {})
    out = " ".join(f"{SYM[k]} {s[k]}" for k in ("pass", "inconclusive", "fail", "error") if s.get(k)) or "—"
    return out + (f"；只报 {s['info']}" if s.get("info") else "")


def render_section(sec: dict, columns: dict, sources: Sources, out_dir: Path) -> tuple[list[str], dict]:
    stats: dict = {}
    body = []
    n_mod = None
    tally = {}
    if sec.get("modules"):
        mlines, n_mod, tally = render_modules(sec["modules"], columns, sources, out_dir)
    for tab in sec.get("tables", []):
        body += render_table(tab, columns, sources, out_dir, stats)
    cols = [c for c in columns if c in stats]  # 按全局列序，不按各表第一次出现的顺序
    fill = {
        "rows": stats.get("_rows", 0),
        "cols": " / ".join(columns[c] for c in cols),
        "checks": " / ".join(str(len(stats[c]["checks"])) for c in cols),
        "tensors": " / ".join(f"{tensor_total(stats[c]['units']):,}" for c in cols),
        "units": " / ".join(units_text(stats[c]["units"]) for c in cols),
        "modules": n_mod if n_mod is not None else "",
        "modules_total": len(sec["modules"]["rows"]) if sec.get("modules") else "",
        "modules_pass": tally.get("pass", ""),
        "modules_open": tally.get("open", ""),
        "modules_none": tally.get("none", ""),
        "modules_attributed": tally.get("attributed", ""),
    }
    md = [f"## {sec['title']}", ""]
    md += [line.format(**fill) for line in sec.get("lead", [])] + [""]
    if sec.get("modules"):
        if sec["modules"].get("title"):
            md += [f"### {sec['modules']['title']}", ""]
        lead = [line.format(**fill) for line in sec["modules"].get("lead") or []]
        md += lead + ([""] if lead else []) + mlines
    md += body
    stats["_modules"] = n_mod
    return md, stats


def render(spec: dict, sources: Sources, out_dir: Path) -> str:
    sources.evidence = [(e["match"], e["source"]) for e in spec.get("evidence_sources") or []]
    columns = spec["columns"]
    md = list(spec.get("header") or [])
    summary, sections = [], []
    total = {"rows": 0, "checks": {c: 0 for c in columns}, "tensors": {c: 0 for c in columns}}
    for sec in spec["levels"]:
        lines, st = render_section(sec, columns, sources, out_dir)
        sections += lines
        total["rows"] += st.get("_rows", 0)
        for c in columns:
            if c in st:
                total["checks"][c] += len(st[c]["checks"])
                total["tensors"][c] += tensor_total(st[c]["units"])
        summary.append(
            f"| [{sec['id']}](#{_anchor(sec['title'])}) | {sec['purpose']} | {st.get('_rows', 0)} | "
            + " / ".join(str(len(st[c]["checks"])) if c in st else "—" for c in columns)
            + " | "
            + (
                " / ".join(f"{tensor_total(st[c]['units']):,}" if c in st else "—" for c in columns)
                if any(c in st and tensor_total(st[c]["units"]) for c in columns)
                else "—"
            )
            + " | "
            + " / ".join(other_units(st[c]["units"]) if c in st else "—" for c in list(columns)[:1])
            + " | "
            + " | ".join(_fmt_status(st.get(c, {})) for c in columns)
            + " |"
        )
    first = list(columns.values())[0]
    md += list(spec.get("summary_lead") or [])
    md += [
        "| 级 | 目的 | 检查项 | 检查条数 | 张量级比较 | 其他比较单元（"
        + first
        + "） | "
        + " | ".join(f"{v} 判定" for v in columns.values())
        + " |",
        "|---|---|---|---|---|---|" + "---|" * len(columns),
    ]
    md += summary
    cells = [
        "合计",
        "",
        str(total["rows"]),
        " / ".join(str(v) for v in total["checks"].values()),
        " / ".join(f"{v:,}" for v in total["tensors"].values()),
        "",
    ] + [""] * len(columns)
    md += ["| " + " | ".join(cells) + " |", ""]
    md += list(spec.get("how_to_read") or []) + [""]
    md += sections
    if spec.get("gpu"):
        g = spec["gpu"]
        lines, _ = render_section(g, g["columns"], sources, out_dir)
        md += lines
    md += list(spec.get("footer") or [])
    return re.sub(r"\n{3,}", "\n\n", "\n".join(md)).rstrip("\n") + "\n"


def _anchor(title: str) -> str:
    """GitHub 标题锚点：小写、去掉标点（保留中文、字母数字、空格、连字符），空格换连字符。"""
    s = title.strip().lower()
    s = re.sub(r"[^\w一-鿿\- ]", "", s)
    return s.replace(" ", "-")


def _all_srcs(spec: dict):
    """清单引用到的全部 (列, src)：各级与 GPU 一节的表格行，加算子对照表里的结果引用。"""
    secs = list(spec["levels"]) + ([spec["gpu"]] if spec.get("gpu") else [])
    for sec in secs:
        for tab in sec.get("tables", []):
            for row in tab["rows"]:
                for c, v in (row.get("src") or {}).items():
                    for s in _srcs(v):
                        yield c, s
        for row in (sec.get("modules") or {}).get("rows", []):
            for c, v in (row.get("checks") or {}).items():
                for s in _srcs(v):
                    yield c, s


def claimed(spec: dict, sources: Sources) -> dict:
    """清单里每个平台列引用到的 (结果文件名, 检查 id)。"""
    out: dict = {}
    for c, s in _all_srcs(spec):
        if _pending(s):
            continue
        f, cs = sources.checks(s)
        name = s.partition(":")[0]
        out.setdefault(c, set()).update((sources.resolve(name) or name, ch["id"]) for ch in cs)
    return out


def unlisted(spec: dict, sources: Sources, model_dir: Path, out_dir: Path, dirs: list[str]) -> list[str]:
    """dirs 这几个对齐实践里，清单一条也没引用的结果文件（旧配置、旧判法、诊断）。→ Markdown 行。
    每个都要在 checks.yaml 的 unlisted_reasons 里写一句为什么不在清单里，缺了就失败：新结果不能悄悄落在清单外。"""
    used = {sources.resolve(s.partition(":")[0]) or s.partition(":")[0] for _, s in _all_srcs(spec)}
    reasons = spec.get("unlisted_reasons") or {}
    pats = [k for k in reasons if any(ch in k for ch in "*?[")]
    group = bool(spec.get("unlisted_group"))  # 原因相同的文件并成一行（只列文件，不列判定计数）
    rows, missing, grouped = [], [], {}
    for name, f in sorted(sources.files.items()):
        adir = f.relative_to(model_dir).parts[1]
        if adir not in dirs or name in used:
            continue
        why = reasons.get(name)
        if why is None:
            why = next((reasons[k] for k in pats if fnmatch.fnmatchcase(name, k)), None)
        if why is None:
            missing.append(name)
            continue
        if group:
            grouped.setdefault(why, []).append(f"[`{name}`]({_link(f, out_dir)})")
            continue
        d = json.loads(f.read_text())
        s = d.get("summary") or {}
        cnt = "、".join(f"{k} {s[k]}" for k in ("pass", "fail", "inconclusive", "error") if s.get(k))
        rows.append(f"| [`{name}`]({_link(f, out_dir)}) | {why} | {cnt} |")
    if group:
        rows = [f"| {'、'.join(fs)} | {why} |" for why, fs in grouped.items()]
    if missing:
        raise SystemExit(
            "这些结果文件不在检查清单里，要在 checks.yaml 的 unlisted_reasons 写原因：" + "、".join(missing)
        )
    if not rows:
        return []
    if group:
        return ["| 结果 | 是什么、为什么不在清单里 |", "|---|---|", *rows, ""]
    return ["| 结果 | 是什么、为什么不在清单里 | 判定 |", "|---|---|---|", *rows, ""]
