"""跨实践总览（dashboard.md / dashboard.html）与检查清单（CHECKS.md）的生成，各模型共用。

每个模型的 ``models/<模型>/dashboard/build.py`` 只写本模型的说明文字与配置（标题、读法、参照说明、故障演练、配图），
再调 :class:`Overview` 的 ``main()``。骨架在模型自己的 ``dashboard/items.yaml``（总览：每行一项检查、每列一种对比）
与 ``dashboard/checks.yaml``（清单，格式见 ``janus/report/inventory.py``）。数字和判定全部从结果文件现读。

结果的来源有两种：

- ``alignments/*/results/<名>.results.json``（janus 的结果文件，文件名全模型唯一）；
- ``virtual``：模型的 build.py 从别的结果文件（如单算子回放、第 1 步诊断的原始 JSON）现算出来的检查，
  ``{名: (原始文件路径, [检查, …])}``。格式与 results.json 里的检查相同，判定规则由 build.py 写明出处。

items.yaml 里 ``<列>_attributed: 说明``：这一格不过或待定的原因已定责（写明谁接受、证据在哪）。判定照结果文件不变，
格子后加「已定责」，说明里写这句话。

src 的写法：``结果名:检查 id 通配``。结果名可以带 ``*``（取排序后最后一个匹配的文件，用于还没提交、名字只知道前缀的结果）；
``todo`` / ``wip: 说明`` / ``na`` 直接写状态。

状态符号（严格档）：✅ 通过 / 对齐；🟡 证据不足或进行中；❌ 不对齐或有错误；⬜ 未开始；— 不适用。
参考档（同类 GPU→TPU 迁移项目的实践门槛，janus/report/reference_gates.yaml）：格子里「参」+ 符号，只并列报告，不决定验收。
故障演练列：✅ = 这一项至少抓到一种埋的错，且期望它抓的都抓到了；· = 没有哪种故障要求它报警。
"""

from __future__ import annotations

import fnmatch
import html
import json
import re
from pathlib import Path

import yaml

SYM = {
    "pass": "✅",
    "inconclusive": "🟡",
    "fail": "❌",
    "error": "❌",
    "todo": "⬜",
    "wip": "🟡",
    "na": "—",
    "none": "·",
}
LABEL = {"todo": "未开始", "wip": "进行中", "na": "不适用"}
PREFER = ("v5p", "v7x", "gpu", "remat")  # 门槛栏取哪一列的门槛：多数列相同的那个；一样多时按这个顺序
FIELD_OF = {"L4.loss": "lm_loss", "L4.grad_norm": "grad_norm", "L4.param_norm": "param_norm"}


def level_of(group_name: str) -> str:
    m = re.match(r"(L\d)", group_name)
    return m.group(1) if m else "—"


def parse_fault_matrix(path: Path) -> dict:
    """解析故障矩阵 Markdown 的「检查 × 故障」表：{检查 id: {故障: 符号}}。"""
    rows, faults = {}, []
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == "检查":
            faults = [c.strip("`") for c in cells[1:]]
        elif faults and len(cells) == len(faults) + 1 and cells[0].startswith("`"):
            rows[cells[0].strip("`")] = dict(zip(faults, cells[1:]))
    return rows


class Overview:
    """一个模型的总览与检查清单。

    参数：
    - ``model_dir``：``models/<模型>``；``title``：页面标题里的模型名（如「混元 3」）。
    - ``how_to_read`` / ``golden_note``：总览开头的读法与参照说明（Markdown 行）；``golden_note`` 第一行是小标题。
      以「  - 」开头的是上一条的子条目（HTML 里同样缩一级）。
    - ``structure``：「本页结构」一行。``faults_file`` / ``fault_codes``：故障矩阵与代号释义，没有演练就不给。
    - ``legacy_figs`` / ``series_compare``：配图（见混元 3 的 build.py）。
    - ``checks_dirs``：检查清单末尾「未列入本页的结果文件」核对哪几个实践。
    - ``virtual``：现算的检查（见模块说明）。``prefer``：门槛栏一样多时优先取哪一列。
    - ``out_dir`` / ``checks_md``：总览（items.yaml、checks.yaml 与生成的 dashboard.md / .html）所在的文件夹与检查清单的路径，
      默认 ``<模型>/dashboard/`` 与 ``<模型>/CHECKS.md``；一个模型按尺寸分成几次独立对齐时（如混元 4），每个尺寸指到自己的实践文件夹。
      ``stem``：骨架与产出的文件名（``<stem>`` 的骨架是 ``items.yaml`` 或 ``<stem>.items.yaml``，产出 ``<stem>.md`` / ``.html``），
      默认 ``dashboard``；一个模型分主表与支线表时（如混元 4 的其他尺寸）用另一个 stem。
      ``skip_all_pass``：表里不列各列都是 ✅ 或 — 的行（照样计入汇总），用于只看差别的支线表。
      ``checks_md=False``：不生成检查清单（总览跨几个尺寸、各尺寸的清单另外生成时）。
      ``builder``：页面开头「由谁生成」写的脚本路径，默认 ``<out_dir>/build.py``。
    - ``detail_stem`` / ``plain_legend``：给人看的简版。给了 ``detail_stem`` 时，上面这份带推导的总览改写到
      ``<detail_stem>.md`` / ``.html``，``<stem>.md`` 改成简版：开头是 ``plain_legend``（几行图例），每组一张表
      「检查项｜各列判定符号｜一行说明」，末尾汇总。检查项与组名取 items.yaml 的 ``say``（没有就用 ``name``），
      列名取 ``column_say``；组的 ``plain_after``（几行 Markdown，如一张缩略图）接在这组的表后；非 ✅、非 — 的格子必须有 ``<列>_say``（一行人话，可带链接），缺了报错；✅ 的格子写了也照列。

    items.yaml 里 ``<列>_why: 说明``：这一格非 ✅ 时，说明里在推导后面接一句结论与出处（如根源一览第几行）；不改判定。
    """

    def __init__(
        self,
        model_dir: Path,
        title: str,
        how_to_read: list[str],
        golden_note: list[str],
        structure: str,
        faults_file: Path | None = None,
        fault_codes: str = "",
        legacy_figs: dict | None = None,
        series_compare: dict | None = None,
        checks_dirs: list[str] = (),
        virtual: dict | None = None,
        prefer: tuple = PREFER,
        out_dir: Path | None = None,
        checks_md: Path | None | bool = None,
        builder: Path | None = None,
        stem: str = "dashboard",
        skip_all_pass: bool = False,
        detail_stem: str | None = None,
        plain_legend: list[str] | None = None,
    ):
        self.root = Path(model_dir)
        self.align = self.root / "alignments"
        self.out = Path(out_dir) if out_dir else self.root / "dashboard"
        self.checks_md = None if checks_md is False else Path(checks_md) if checks_md else self.root / "CHECKS.md"
        self.builder = Path(builder) if builder else self.out / "build.py"
        self.stem = stem
        self.skip_all_pass = skip_all_pass
        self.detail_stem = detail_stem
        self.plain_legend = plain_legend or []
        self.title = title
        self.how_to_read = how_to_read
        self.golden_note = golden_note
        self.structure = structure
        self.faults_file = faults_file
        self.fault_codes = fault_codes
        self.legacy_figs = legacy_figs or {}
        self.series_compare = series_compare or {}
        self.checks_dirs = list(checks_dirs)
        self.virtual = virtual or {}
        self.prefer = prefer

    # ------------------------------------------------------------ 取结果

    def resolve(self, name: str) -> str | None:
        """结果名 → 实际的文件名（不带后缀）。带 ``*`` 时取排序后最后一个匹配的；找不到 → None。"""
        if "*" not in name:
            return name
        hits = sorted(
            {f.name[: -len(".results.json")] for f in self.align.glob(f"*/results/{name}.results.json")}
            | {n for n in self.virtual if fnmatch.fnmatchcase(n, name)}
        )
        return hits[-1] if hits else None

    def load_results(self, name: str):
        name = self.resolve(name)
        if name is None:
            return None
        if name in self.virtual:
            return {"checks": self.virtual[name][1]}
        hits = sorted(self.align.glob(f"*/results/{name}.results.json"))
        if len(hits) > 1:
            raise SystemExit(f"结果文件重名：{[str(h) for h in hits]}")
        return json.loads(hits[0].read_text()) if hits else None

    def fault_matrix(self) -> dict:
        return parse_fault_matrix(self.faults_file) if self.faults_file else {}

    # ------------------------------------------------------------ 一格

    def judge(self, src, expect=None, fm=None):
        """一格的判定：→ (严格档状态, 参考档 RefVerdict 或 None（不适用）, 说明)。"""
        from janus.report import tables as AL

        if src.startswith("wip:"):  # 进行中：有诊断证据、还没接成正式检查；说明里写证据在哪
            return "wip", None, src[4:].strip()
        if src.startswith("todo:"):
            return "todo", None, src[5:].strip()
        if src in SYM and src not in ("pass", "fail"):
            return src, None, LABEL.get(src, "")
        name, _, pat = src.partition(":")
        if name == "faults":
            hits = {k: v for k, v in (fm or {}).items() if fnmatch.fnmatchcase(k, pat)}
            if not hits:
                return "todo", None, "无结果"
            caught = sorted({f for v in hits.values() for f, m in v.items() if "F" in m})
            missed = sorted({f for v in hits.values() for f, m in v.items() if "✗" in m})
            if missed:
                return "fail", None, "期望报警没报：" + "、".join(missed)
            if caught:
                return "pass", None, "抓到：" + "、".join(caught)
            return "none", None, ""
        r = self.load_results(name)
        if r is None:
            return "todo", None, "无结果文件"
        cs = [c for c in r.get("checks", []) if fnmatch.fnmatchcase(c["id"], pat)]
        if not cs:
            return "todo", None, "结果里没有这条"
        return AL.summarize_checks(cs, expect)

    def cell_info(self, src, expect=None, fm=None):
        """总览一格：{status, ref, note, metric, value, how, thr, short_thr, ref_thr, n}。没有结果文件的格子只有 status 与 note。"""
        from janus.report import tables as AL

        if isinstance(src, list):  # 几个结果文件合成一格（如同一项检查分几个阶段各一份结果）
            cs = []
            for s in src:
                name, _, pat = s.partition(":")
                cs += [
                    c for c in (self.load_results(name) or {}).get("checks", []) if fnmatch.fnmatchcase(c["id"], pat)
                ]
            if cs:
                return AL.overview_cell(cs, expect)
            src = "todo:无结果文件"
        name, _, pat = src.partition(":")
        if not (src.startswith(("wip:", "todo:")) or src in SYM or name == "faults"):
            r = self.load_results(name)
            cs = [c for c in (r or {}).get("checks", []) if fnmatch.fnmatchcase(c["id"], pat)]
            if cs:
                return AL.overview_cell(cs, expect)
        st, ref, note = self.judge(src, expect, fm)
        return {
            "status": st,
            "ref": ref,
            "note": note,
            "metric": "",
            "value": "",
            "how": "",
            "thr": "",
            "short_thr": "",
            "ref_thr": "",
        }

    def canonical(self, infos):
        """门槛栏与指标栏：各列 (指标, 严格门槛) 里出现最多的那个（一样多时按 prefer）。"""
        got = [(c, i) for c, i in infos.items() if i.get("metric")]
        if not got:
            return None
        count = {}
        for c, i in got:
            count.setdefault((i["metric"], i["thr"]), []).append(c)
        rank = {c: k for k, c in enumerate(self.prefer)}
        key = max(count, key=lambda k: (len(count[k]), -min(rank.get(c, 99) for c in count[k])))
        return next(i for c, i in got if (i["metric"], i["thr"]) == key)

    @staticmethod
    def cell_text(i, canon):
        """格子：实测（离门槛多远；指标或门槛和门槛栏不同时写本列自己的）+ 严格档符号 + ｜参 + 参考档符号。"""
        st, ref = i["status"], i["ref"]
        if not i.get("value"):
            return {"todo": "⬜", "na": "—", "wip": "🟡 进行中", "none": "·"}.get(st, SYM[st])
        q = []
        if canon and i["metric"] != canon["metric"]:
            q.append(i["metric"])
        if i["how"]:
            q.append(i["how"])
        if canon and i["thr"] != canon["thr"] and i["short_thr"]:
            q.append(i["short_thr"])
        txt = (
            i["value"]
            + (f"（{'，'.join(q)}）" if q else "")
            + " "
            + SYM[st]
            + (" 已定责" if i.get("attributed") else "")
        )
        return txt + (f"｜参{ref.sym}" if ref is not None and ref.status != "none" else "")

    # ------------------------------------------------------------ 各实践的逐级大表

    def series_file(self, name):
        hits = sorted(self.align.glob(f"*/results/series/{name}.metrics.jsonl"))
        return hits[0] if hits else None

    def make_figures(self, adir, res, used):
        """生成本实践的配图，返回 {检查 id 通配: [相对 README 的图路径]}。"""
        from janus.report import figures as F

        figdir = adir / "figures"
        figs = dict(self.legacy_figs.get(adir.name, {}))
        own = sorted(res.own & used)
        if not own and adir.name not in self.series_compare and not figs:
            return figs
        figdir.mkdir(exist_ok=True)
        for name in own:
            f = res.files[name]
            d, _ = res.load(name)
            if d is None:
                continue
            cs = d["checks"]
            if any(c["id"].startswith("L2.layer") for c in cs):
                out = F.l2_layers(
                    cs,
                    figdir / f"L2-step1-layers.{name}.png",
                    f"{name}：第 1 步逐层（柱 = 实测，黑线 = N，红虚线 = 2N）",
                )
                if out:
                    figs.setdefault("L2.layer*", []).append(f"figures/{out.name}")
            for c in cs:
                if c.get("type") == "ulp_distribution":
                    out = F.ulp_buckets(c, figdir / f"{c['id']}.{name}.png", f"{name} · {c['id']}：ULP 分桶")
                    if out:
                        figs.setdefault(c["id"], []).append(f"figures/{out.name}")
                if c.get("type") == "discrete_agreement" and c.get("details"):
                    out = F.router_margin(
                        f.parent / c["details"],
                        figdir / f"{c['id']}-margin.{name}.png",
                        f"{name} · {c['id']}：翻转 token 的 topk 分差",
                    )
                    if out:
                        figs.setdefault(c["id"], []).append(f"figures/{out.name}")
        from janus.analysis import curves

        for rname, test, goldens, steps in self.series_compare.get(adir.name, []):
            if rname not in res.files:
                continue
            made = curves.compare(
                (test, self.series_file(test)),
                [(g, self.series_file(g)) for g in goldens],
                figdir,
                fields=("lm_loss", "grad_norm", "param_norm"),
                prefix=f"L4.{test}",
                steps=steps,
            )
            for cid, field in FIELD_OF.items():
                if field in made:
                    figs.setdefault(cid, []).extend(f"figures/{p.name}" for p in made[field])
        return figs

    def build_alignment(self, adir, cfg, spec, fm):
        from janus.report import tables as AL

        res = AL.Results(adir, [d for d in sorted(self.align.iterdir()) if d.is_dir() and d != adir])
        col = cfg["column"]
        items = [
            {
                "name": it["name"],
                "src": it.get(col, "na"),
                "level": level_of(g["name"]),
                "expect": it.get(f"{col}_expect"),
            }
            for g in spec["groups"]
            for it in g["items"]
            if it.get(col, "na") != "na"
        ]
        for it in items:  # 带 * 的结果名换成实际文件名；还没有文件的记成未开始
            name, sep, pat = it["src"].partition(":")
            if "*" in name and name != "faults" and not it["src"].startswith(("wip:", "todo:")):
                real = self.resolve(name)
                it["src"] = f"{real}{sep}{pat}" if real else "todo"
        used = {it["src"].partition(":")[0] for it in items}
        figs = self.make_figures(adir, res, used)
        rows = AL.build_rows(items, res, figs)
        body = AL.render(rows)
        if cfg.get("drill"):
            dc = cfg["drill"]
            lines = [
                "",
                "### 故障演练（确定性基线上真埋错）",
                "",
                "由故障矩阵 `faults/phase1-faults.md` 与注入类结果文件判定：✅ 期望报警的都报了；❌ 该报没报；· 没有哪种故障要求它报警。",
                "",
                self.fault_codes,
                "",
                "| 级 | 检查项 | 判定 | 说明 |",
                "|---|---|---|---|",
            ]
            for g in spec["groups"]:
                for it in g["items"]:
                    src = it.get(dc)
                    if not src or src == "na":
                        continue
                    st, _, note = self.judge(src, it.get("drill_expect"), fm)
                    lines.append(f"| {level_of(g['name'])} | {it['name']} | {SYM[st]} | {note} |")
            body += "\n".join(lines) + "\n"
        readme = adir / "README.md"
        AL.write_section(readme, body)
        return {
            s: sum(1 for r in rows if r.status == s) for s in ("pass", "inconclusive", "fail", "error", "todo", "wip")
        }

    # ------------------------------------------------------------ 检查清单

    def sources(self):
        from janus.report import inventory as INV

        return INV.Sources(self.root, virtual={k: v for k, v in self.virtual.items()})

    def build_checks(self, spec_items, spec=None):
        """检查清单 CHECKS.md。总览里 checks.yaml 各平台列引用的每条检查，清单里都要有一行引用，否则失败。"""
        from janus.report import inventory as INV

        spec = spec or yaml.safe_load((self.out / "checks.yaml").read_text())
        sources = self.sources()
        self._fill_pending(spec, sources)
        md = INV.render(spec, sources, self.checks_md.parent)
        got = INV.claimed(spec, sources)
        missing = []
        for g in spec_items["groups"]:
            for it in g["items"]:
                for col in spec["columns"]:
                    srcs = it.get(col)
                    for src in srcs if isinstance(srcs, list) else [srcs]:
                        if not src or ":" not in src or src.startswith(("faults:", "wip:", "todo:")):
                            continue
                        _, cs = sources.checks(src)
                        miss = {(sources.resolve(src.partition(":")[0]), c["id"]) for c in cs} - got.get(col, set())
                        missing += [f"{col} · {it['name']} · {n}:{i}" for n, i in sorted(miss)]
        if missing:
            raise SystemExit("总览引用的这些检查在 checks.yaml 里没有对应的行：\n" + "\n".join(missing))
        rows = INV.unlisted(spec, sources, self.root, self.checks_md.parent, self.checks_dirs)
        if rows:  # unlisted_lead：有清单外的结果时才写的小标题与说明（混元 3 把它写在 footer 末尾）
            rows = list(spec.get("unlisted_lead") or []) + rows
        md = md.rstrip("\n") + "\n\n" + "\n".join(rows).rstrip("\n") + "\n"
        self.checks_md.write_text(md)

    @staticmethod
    def _fill_pending(spec, sources):
        """清单里结果名带 * 却还没有匹配文件的格子，换成「todo: 待 <名> 的结果」。"""

        def fix(v):
            if isinstance(v, list):
                return [fix(x) for x in v]
            if isinstance(v, str) and ":" in v and "*" in v.partition(":")[0] and not v.startswith(("todo", "wip")):
                name = v.partition(":")[0]
                return v if sources.resolve(name) else f"todo: 待 {name.rstrip('*')} 的结果"
            return v

        for sec in list(spec["levels"]) + ([spec["gpu"]] if spec.get("gpu") else []):
            for tab in sec.get("tables", []):
                for row in tab["rows"]:
                    row["src"] = {c: fix(v) for c, v in (row.get("src") or {}).items()}

    # ------------------------------------------------------------ 总览

    def main(self, checks_spec=None):
        items = self.out / ("items.yaml" if self.stem == "dashboard" else f"{self.stem}.items.yaml")
        spec = yaml.safe_load(items.read_text())
        if self.checks_md is not None:
            self.build_checks(spec, checks_spec)
        fm = self.fault_matrix()
        cols = list(spec["columns"])
        drill_cols = [cfg["drill"] for cfg in (spec.get("alignments") or {}).values() if cfg.get("drill")]
        main_cols = [c for c in cols if c not in drill_cols]
        rel = self.builder.relative_to(self.root.parents[1]).as_posix()
        gn = self.golden_note
        md = [
            f"# {self.title} · 对齐总览" + ("细节" if self.detail_stem else ""),
            "",
            f"由 `{rel}` 从 `alignments/*/results/` 生成，数字和判定都不手写。",
            "",
            self.structure,
            "",
            *self.how_to_read,
            gn[0],
            "",
            *gn[1:],
            "",
            "## 各列是什么",
            "",
        ]
        md += ["| 列 | 含义 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in spec["columns"].items()] + [""]
        count = {c: {} for c in cols}
        rcount = {c: {} for c in cols}
        body, drills, plain = [], [], []
        for g in spec["groups"]:
            prow = []
            plain.append((g.get("say", g["name"]), prow, g.get("plain_after") or []))
            md += [
                f"## {g['name']}",
                "",
                "| 检查项 | 指标 | 门槛（严格｜参考） | " + " | ".join(main_cols) + " |",
                "|---|---|---|" + "---|" * len(main_cols),
            ]
            hrows, notes = [], []
            for it in g["items"]:
                infos = {}
                for c in cols:
                    info = self.cell_info(it.get(c, "na"), it.get(f"{c}_expect"), fm)
                    st, ref = info["status"], info["ref"]
                    count[c][st] = count[c].get(st, 0) + 1
                    if ref is not None and ref.status != "none":
                        rcount[c][ref.status] = rcount[c].get(ref.status, 0) + 1
                    infos[c] = info
                for c in drill_cols:
                    if it.get(c, "na") != "na":
                        drills.append((level_of(g["name"]), it["name"], infos[c]))
                canon = self.canonical({c: infos[c] for c in main_cols})
                metric = canon["metric"] if canon else "—"
                thr = f"严格：{canon['thr']}｜参考：{canon['ref_thr']}" if canon else "—"
                att = {c: it.get(f"{c}_attributed") for c in main_cols if it.get(f"{c}_attributed")}
                for c in att:  # 已定责：判定照结果文件不变，格子后加「已定责」，说明里写定责的结论与证据
                    infos[c] = {**infos[c], "attributed": att[c]}
                cells = [self.cell_text(infos[c], canon) for c in main_cols]
                prow.append((it, {c: infos[c] for c in main_cols}))
                if self.skip_all_pass and all(infos[c]["status"] in ("pass", "na") for c in main_cols):
                    continue
                md.append(f"| {it['name']} | {metric} | {thr} | " + " | ".join(cells) + " |")
                for c in main_cols:
                    i = infos[c]
                    bad_ref = i["ref"] is not None and i["ref"].status in ("fail", "missing")
                    if i["note"] and i.get("value") and (i["status"] != "pass" or bad_ref):
                        why = it.get(f"{c}_why")
                        notes.append(f"- **{it['name']} · {c}**：{i['note']}" + (f"｜{why}" if why else ""))
                    if i.get("attributed"):
                        notes.append(f"- **{it['name']} · {c}**（已定责）：{i['attributed']}")
                    elif (
                        i["note"]
                        and i["status"] in ("wip", "todo")
                        and str(it.get(c, "na")).startswith(("wip:", "todo:"))
                    ):
                        notes.append(f"- **{it['name']} · {c}**（{LABEL[i['status']]}）：{i['note']}")
                hrows.append(
                    (it["name"], metric, thr, [(infos[c], self.cell_text(infos[c], canon)) for c in main_cols])
                )
            md.append("")
            if notes:
                md += (
                    ["<details><summary>说明（非 ✅ 格子，以及参考档不过或无法判的格子）</summary>", ""]
                    + notes
                    + ["", "</details>", ""]
                )
            body.append((g["name"], hrows, notes))
        if drill_cols:
            faults_rel = self.faults_file.relative_to(self.root).as_posix()
            md += [
                "## 故障演练",
                "",
                f"确定性基线上真埋错，看期望报警的检查有没有报（故障矩阵 `{faults_rel}`）。"
                "✅ 期望报警的都报了；❌ 该报没报；· 没有哪种故障要求它报警。",
                "",
                self.fault_codes,
                "",
                "| 级 | 检查项 | 判定 | 抓到的故障 / 说明 |",
                "|---|---|---|---|",
            ]
            md += [f"| {lv} | {name} | {SYM[i['status']]} | {i['note']} |" for lv, name, i in drills]
            md += [""]
        md += [
            "## 汇总",
            "",
            "严格档数全部格子；参考档只数有对应门槛的格子（演练不适用）。",
            "",
            "| 列 | 严格档 ✅ | 🟡 | ❌ | ⬜ | 参考档 ✅ | ☑️ | ❌ | ❔ |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for c in cols:
            d, rd = count[c], rcount[c]
            r = (
                " | ".join(str(rd.get(s, 0)) for s in ("pass", "partial", "fail", "missing"))
                if c not in drill_cols
                else "— | — | — | —"
            )
            md.append(
                f"| {c} | {d.get('pass', 0)} | {d.get('inconclusive', 0) + d.get('wip', 0)} | "
                f"{d.get('fail', 0) + d.get('error', 0)} | {d.get('todo', 0)} | {r} |"
            )
        detail = self.detail_stem or self.stem
        (self.out / f"{detail}.md").write_text("\n".join(md) + "\n")
        self.write_html(spec, main_cols, body, drills, detail)
        if self.detail_stem:
            self.write_plain(spec, main_cols, plain, count)
        print({c: count[c] for c in cols})
        for name, cfg in (spec.get("alignments") or {}).items():
            print(name, self.build_alignment(self.align / name, cfg, spec, fm))

    def write_plain(self, spec, main_cols, plain, count):
        """给人看的简版：图例 + 每组一张表（判定符号 + 一行说明）+ 汇总；推导都在带推导的那份。"""
        say = spec.get("column_say") or {}
        md = [f"# {self.title} · 对齐总览", "", *self.plain_legend, ""]
        for gname, rows, after in plain:
            md += [
                f"## {gname}",
                "",
                "| 检查项 | " + " | ".join(say.get(c, c) for c in main_cols) + " | 说明 |",
                "|---|" + "---|" * len(main_cols) + "---|",
            ]
            for it, infos in rows:
                notes = []
                for c in main_cols:
                    st = infos[c]["status"]
                    note = it.get(f"{c}_say")
                    if st == "na" or (st == "pass" and not note):
                        continue
                    if not note:
                        raise SystemExit(f"简版总览：「{it['name']}」的 {c} 列是 {st}，items.yaml 要写 {c}_say")
                    notes.append(note if len(main_cols) == 1 else f"{say.get(c, c)}：{note}")
                sym = " | ".join(SYM[infos[c]["status"]] for c in main_cols)
                md.append(f"| {it.get('say', it['name'])} | {sym} | {'；'.join(notes)} |")
            md.append("")
            if after:
                md += [*after, ""]
        md += ["## 汇总", "", "| 列 | ✅ | 🟡 | ❌ | ⬜ |", "|---|---|---|---|---|"]
        for c in main_cols:
            d = count[c]
            md.append(
                f"| {say.get(c, c)} | {d.get('pass', 0)} | {d.get('inconclusive', 0) + d.get('wip', 0)} | "
                f"{d.get('fail', 0) + d.get('error', 0)} | {d.get('todo', 0)} |"
            )
        (self.out / f"{self.stem}.md").write_text("\n".join(md) + "\n")

    def write_html(self, spec, main_cols, body, drills, stem=None):
        def _links(t):  # 说明里的 Markdown 链接 [文字](地址) 换成可点的 <a>（t 已转义，地址里没有要转义的字符）
            return re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', t)

        color = {
            "pass": "#dafbe1",
            "inconclusive": "#fff8c5",
            "wip": "#fff8c5",
            "fail": "#ffebe9",
            "error": "#ffebe9",
            "todo": "#f6f8fa",
            "na": "#ffffff",
            "none": "#ffffff",
        }
        esc = html.escape
        gn = self.golden_note
        h = [
            f"<!doctype html><html><head><meta charset=utf-8><title>{self.title} · 对齐总览</title><style>"
            "body{font-family:-apple-system,'Noto Sans CJK SC',sans-serif;max-width:1400px;margin:auto;padding:1em}"
            "table{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:.6em}td,th{border:1px solid #d0d7de;padding:4px 8px;vertical-align:top}"
            "th{background:#f6f8fa}td.s{text-align:center;white-space:nowrap}td.t{color:#57606a;font-size:12px;min-width:16em}"
            "details{font-size:12px;color:#57606a;margin-bottom:1.4em}</style></head><body>",
            f"<h1>{self.title} · 对齐总览</h1>",
            "<p>由结果文件自动生成。平台格子：实测值（离严格门槛多远；本列指标或 N 与门槛栏不同时另写）＋严格档符号＋「｜参」参考档符号。"
            "严格档是验收判定（✅ 过　🟡 待定或进行中　❌ 不过或出错　⬜ 未开始　— 不适用）；参考档是同类 GPU→TPU 迁移项目的实践门槛，只并列报告"
            "（✅ 过　☑️ 记录了的都过、有门槛项没记录　❌ 不过　❔ 所需指标未记录；没有「参」＝无对应门槛）。两档含义见 docs/reference/judging.md「两档门槛」。</p>",
            "<p>"
            + esc(gn[0].lstrip("# "))
            + "</p><ul>"
            + "".join(
                (
                    f"<li style='margin-left:1.5em'>{_links(esc(n[4:].replace('**', '')))}</li>"
                    if n.startswith("  - ")
                    else f"<li>{_links(esc(n[2:].replace('**', '')))}</li>"
                )
                for n in gn[1:]
            )
            + "</ul>",
            "<table><tr><th>列</th><th>含义</th></tr>"
            + "".join(f"<tr><td>{k}</td><td>{esc(v)}</td></tr>" for k, v in spec["columns"].items())
            + "</table>",
        ]
        for gname, rows, notes in body:
            h.append(
                f"<h2>{esc(gname)}</h2><table><tr><th>检查项</th><th>指标</th><th>门槛（严格｜参考）</th>"
                + "".join(f"<th>{c}</th>" for c in main_cols)
                + "</tr>"
            )
            for name, metric, thr, cells in rows:
                h.append(
                    f"<tr><td>{esc(name)}</td><td>{esc(metric)}</td><td class=t>{esc(thr).replace('｜', '<br>')}</td>"
                    + "".join(
                        f'<td class=s style="background:{color[i["status"]]}">{esc(t).replace("｜", "<br>")}</td>'
                        for i, t in cells
                    )
                    + "</tr>"
                )
            h.append("</table>")
            if notes:
                h.append(
                    "<details><summary>说明</summary><ul>"
                    + "".join(f"<li>{_links(esc(n[2:].replace('**', '')))}</li>" for n in notes)
                    + "</ul></details>"
                )
        if drills:
            h.append(
                "<h2>故障演练</h2><p>"
                + esc(self.fault_codes.replace("`", ""))
                + "</p><table><tr><th>级</th><th>检查项</th><th>判定</th><th>抓到的故障 / 说明</th></tr>"
                + "".join(
                    f'<tr><td>{lv}</td><td>{esc(n)}</td><td class=s style="background:{color[i["status"]]}">{SYM[i["status"]]}</td>'
                    f"<td class=t>{esc(i['note'])}</td></tr>"
                    for lv, n, i in drills
                )
                + "</table>"
            )
        h[-1] += "</body></html>"
        (self.out / f"{stem or self.stem}.html").write_text("\n".join(h))
