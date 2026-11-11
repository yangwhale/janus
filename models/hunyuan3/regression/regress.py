"""混元 3 · 日常回归的辅助脚本：读档位配置、按 PR 声明改判、写回归报告、写 LKG 登记条目。run_regression.sh 调它，也可以单独用。

  python regress.py args <T0|T1>                                  # 打印档位配置（shell 可 eval）
  python regress.py report --tier T0 --janus <janus 结果目录> --root <运行根> --test <被测> --lkg <LKG> --out <目录>
         [--declare 数学不变|执行级|数学级|升级依赖] [--scope R08,R12] [--lkg-results <LKG 自己的回归结果目录>] [--first-lkg]
  python regress.py lkg-entry --tier T0 --root <运行根> --run <运行> [--pair <双跑的另一份>] --platform v5p --location <占位路径>
         --results <仓库里的结果路径> --reason <换代理由> [--approved-by <批准人>]

报告首屏照标准第十一节：过没过、哪一类、第一个不同的量、下一步。数字全部来自 janus 的 results.json 与运行目录，不手写。
判法照标准第三节的表：janus 跑的是「数学不变」的目录（逐位为主），这里按 PR 声明改判声明范围内的行；
role: tolerance 的检查（门槛待标定）只在「会改数值 · 执行级」与「升级依赖」不逐位时采用。
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shlex
import statistics
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from janus.catalog import HASH_ALGO, recorded_hash  # noqa: E402
from janus.catalog import content_sha256 as catalog_content_sha256  # noqa: E402
from janus.catalog import load as load_catalog  # noqa: E402
from janus.store import Run  # noqa: E402

TIERS = {"T0": HERE / "tiers" / "t0.yaml", "T1": HERE / "tiers" / "t1.yaml"}
# 每档要出结论的行（标准第六节）。R22（步时）与 R23（覆盖）在这里算，其余来自 janus 的检查
ROWS = {
    "T0": ["R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09", "R14", "R15", "R17", "R18", "R20", "R21",
           "R22", "R23"],
}
ROWS["T1"] = sorted(ROWS["T0"] + ["R12", "R13", "R16", "R25"], key=lambda r: int(r[1:]))
NUMERIC = {"R08", "R09", "R10", "R12", "R13", "R14", "R15", "R16"}     # 标准第三节「数值结果」（R14 只有声明改调度时才放）
CONFIG = {"R01", "R02", "R03", "R04", "R05", "R06", "R07"}             # 「输入与环境」：声明改的字段除外
ARTIFACT = {"R18", "R19", "R20"}                                        # 「编译产物」：声明的变化以外相等
ALWAYS_HARD = {"R11", "R17", "R23", "R24", "R27"}
REPORT_ONLY = {"R21", "R22", "R28"}
DECLARATIONS = ("数学不变", "执行级", "数学级", "升级依赖")
# 第一个不同的量：按「越靠前越可能是根因」的顺序找（标准第八节第二步）
ORDER = ["R01", "R02", "R04", "R05", "R06", "R07", "R18", "R20", "R03", "R08", "R09", "R14", "R12", "R13", "R16", "R15",
         "R17"]
STEP_TIME_SKIP = 2      # 步时中位数去掉前两步（第 1 步含编译，第 2 步常有异步收尾）


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tier_cfg(tier: str) -> dict:
    if tier not in TIERS:
        raise SystemExit(f"档位只能是 {sorted(TIERS)}：{tier}")
    cfg = yaml.safe_load(TIERS[tier].read_text())
    cfg["_path"] = TIERS[tier]
    cfg["_sha256"] = sha_file(TIERS[tier])
    return cfg


# ------------------------------------------------------------------------------------------------ args
def cmd_args(a) -> int:
    c = tier_cfg(a.tier)
    cat = HERE / c["catalog"]
    print(f"TIER_STEPS={c['steps']}")
    print(f"TIER_CATALOG={shlex.quote(str(cat))}")
    print(f"TIER_SHA256={c['_sha256']}")
    print("TIER_TRAIN_ARGS=(" + " ".join(shlex.quote(x) for x in c["train_args"] + c["overrides"]) + ")")
    print("TIER_GOLDEN=(" + " ".join(shlex.quote(x) for x in c.get("golden") or []) + ")")
    return 0


# ------------------------------------------------------------------------------------------------ report
def _median_step_time(run: Run) -> tuple[float | None, int]:
    st = run.series("step_time")
    vals = [v for s, v in sorted(st.items()) if s > STEP_TIME_SKIP]
    return (statistics.median(vals) if vals else None), len(vals)


def judge_rows(tier: str, catalog, results: dict, declare: str, scope: set) -> dict:
    """按行汇总 janus 的检查结果，再按 PR 声明改判。返回 {行: {status, checks, note}}。
    status：pass / fail / inconclusive / error / report（只报）。"""
    by_id = {c["id"]: c for c in results["checks"]}
    rows: dict[str, dict] = {}
    for c in catalog.checks:
        row = c.get("row")
        if not row:
            continue
        r = by_id.get(c["id"])
        ent = rows.setdefault(row, {"strict": [], "tolerance": []})
        ent["tolerance" if c.get("role") == "tolerance" else "strict"].append((c, r))
    out = {}
    for row in ROWS[tier]:
        if row in ("R22", "R23"):
            continue
        ent = rows.get(row)
        if not ent or not ent["strict"]:
            out[row] = {"status": "error", "checks": [], "note": "目录里没有这一行的检查"}
            continue
        strict = [(c, r) for c, r in ent["strict"]]
        sts = [(r or {}).get("status", "error") for _, r in strict]
        hard_sts = [s for (c, _), s in zip(strict, sts) if c.get("hard", True)]
        if "error" in sts:
            st = "error"
        elif row in REPORT_ONLY or not hard_sts:
            st = "report"
        else:
            st = "fail" if "fail" in hard_sts else ("inconclusive" if "inconclusive" in hard_sts else "pass")
        note = ""
        in_scope = row in scope
        if st == "fail" and row not in ALWAYS_HARD:
            if declare == "执行级" and in_scope and row in NUMERIC:
                tol = [(r or {}).get("status", "error") for _, r in ent["tolerance"]]
                if tol and all(t in ("pass", "inconclusive", "fail", "error") for t in tol):
                    st = "fail" if "fail" in tol else ("error" if "error" in tol else
                                                         ("inconclusive" if "inconclusive" in tol else "pass"))
                    note = "声明「执行级」且在范围内：不逐位，改按容差门判（门槛待标定时为 inconclusive）"
                else:
                    st, note = "inconclusive", "声明「执行级」且在范围内：不逐位；这一行没有容差门，待 TPU 执行噪声标定后才能判"
            elif declare == "数学级" and in_scope and row in NUMERIC:
                st, note = "report", "声明「数学级」且在范围内：只报，由对齐门（R25）拦"
            elif declare in ("执行级", "数学级", "升级依赖") and in_scope and (row in CONFIG or row in ARTIFACT):
                st, note = "report", f"声明「{declare}」改了这一行：变化在声明范围内，只报（看 details 核对确实只变了声明的字段）"
            elif declare == "升级依赖" and row in NUMERIC:
                tol = [(r or {}).get("status", "error") for _, r in ent["tolerance"]]
                st = "inconclusive" if not tol or "fail" not in tol else "fail"
                note = "声明「升级依赖」且不逐位：按执行级判（门槛待标定时为 inconclusive），并按标准第九节做归因"
            elif declare != "数学不变" and not in_scope and row in NUMERIC:
                note = "声明范围以外的行不逐位：PR 退回（标准第三节「声明与实测不符就拦」）"
        out[row] = {"status": st, "note": note,
                    "checks": [{"id": c["id"], "status": (r or {}).get("status", "error"), "hard": c.get("hard", True),
                                "message": (r or {}).get("message", "没有结果"),
                                "metrics": (r or {}).get("metrics", {})} for c, r in strict + ent["tolerance"]]}
    return out


def registry_check(rows: dict, lkg_name: str, cfg: dict, results: dict, declare: str, scope: set) -> None:
    """R02 的另一半：档位配置文件的 sha256、检查项目录的内容指纹对 lkg.yaml 里这份 LKG 登记的值。变了就是改了档位定义，旧 LKG 作废。"""
    p = HERE / "lkg.yaml"
    reg = (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}
    # 同一个运行可以登记多次（检查项目录换代后，在新目录下重判逐位、重新登记）：取最新的一条
    ent = next((e for e in reversed(reg.get("entries") or []) if e.get("run") == lkg_name), None)
    row = rows.setdefault("R02", {"status": "pass", "checks": [], "note": ""})
    if ent is None:
        row["checks"].append({"id": "R02.registry", "status": "inconclusive", "hard": False, "metrics": {},
                              "message": f"LKG {lkg_name} 不在 lkg.yaml 里（建首个 LKG 或手动指定时）：档位配置与目录的 sha256 没核"})
        return
    # 目录比内容指纹（content-v1，改注释与说明文字不算改目录）：结果与登记都有内容指纹时比它；
    # 旧结果（schema_version 1）或没有 catalog_content_sha256 的旧登记，退回比原文件 sha256
    entries = [results["catalog"], *results["catalog"].get("includes", [])]
    use_content = {recorded_hash(e)[0] for e in entries} == {HASH_ALGO} and bool(ent.get("catalog_content_sha256"))
    if use_content:
        now = {Path(e["path"]).name: e["content_sha256"] for e in entries}
        reg_hashes = ent["catalog_content_sha256"]
    else:
        now = {Path(e["path"]).name: e.get("raw_sha256") or e.get("sha256") for e in entries}
        reg_hashes = ent.get("catalog_sha256") or {}
    bad = []
    if ent.get("tier_config_sha256") != cfg["_sha256"]:
        bad.append("档位配置文件")
    for k, v in reg_hashes.items():
        if now.get(k) != v:
            bad.append(k)
    row["checks"].append({"id": "R02.registry", "status": "fail" if bad else "pass", "hard": True,
                          "metrics": {"tier_config_sha256": cfg["_sha256"][:12], "catalog_files": len(now),
                                      "catalog_hash": HASH_ALGO if use_content else "raw"},
                          "message": (f"与 lkg.yaml 登记的不同：{bad}（改档位定义要重建 LKG）" if bad
                                      else "档位配置的 sha256、目录的内容指纹与 lkg.yaml 登记的相同" if use_content
                                      else "档位配置与目录的 sha256 与 lkg.yaml 登记的相同")})
    if bad and not (declare != "数学不变" and "R02" in scope) and row["status"] in ("pass", "inconclusive"):
        row["status"] = "fail"


def load_calibration(platform: str, tier: str) -> dict | None:
    """calibration.yaml 里这个平台、这一档最新的一条标定（TPU 执行噪声、R22 步时阈值）。没有就返回 None。"""
    p = HERE / "calibration.yaml"
    if not p.exists():
        return None
    d = yaml.safe_load(p.read_text()) or {}
    hits = [e for e in d.get("entries") or [] if e["key"].get("platform") == platform and e["key"].get("tier") == tier]
    return hits[-1] if hits else None


def calibration_valid(cal: dict | None, test: Run, cfg: dict) -> tuple[bool, str]:
    """标准第五节：一份噪声参照绑定平台、档位配置（sha256）、训练配置（hy3-4layers.yml 的 sha256）、依赖集合（镜像与版本）。
    任何一样变了，参照作废。训练配置改了（例如打开一个会改数值的开关）基线就变了，旧参照量的不是新配置的噪声。
    绑定里没写训练配置 sha256 的老条目核不了这一项，按作废处理（宁可降为 inconclusive，不拿核不了的尺子判 pass）。"""
    if not cal:
        return False, "这一档在这个平台上还没标定"
    b = cal.get("binding") or {}
    bad = []
    if b.get("tier_config_sha256") != cfg["_sha256"]:
        bad.append("档位配置")
    yml = (test.meta.get("config") or {}).get("sha256")
    if "config_yml_sha256" not in b:
        bad.append("训练配置（标定条目没记 hy3-4layers.yml 的 sha256）")
    elif b["config_yml_sha256"] != yml:
        bad.append("训练配置（hy3-4layers.yml）")
    if b.get("image") != test.meta.get("image"):
        bad.append("镜像")
    if b.get("versions") != test.meta.get("versions"):
        bad.append("依赖版本")
    return (not bad), ("" if not bad else f"噪声参照作废：{'、'.join(bad)}和标定时不同，要重标（标准第五节）")


def r22_step_time(test: Run, lkg: Run, cal: dict | None = None) -> dict:
    t, nt = _median_step_time(test)
    r, nr = _median_step_time(lkg)
    rel = (t - r) / r if (t is not None and r) else None
    m = {"test_median_s": t, "lkg_median_s": r, "rel_diff": rel, "steps_used": nt, "skip_first": STEP_TIME_SKIP}
    thr = (cal or {}).get("step_time_threshold")
    if thr is None:
        note = "只报。阈值待标定（标准 R22：同一提交 ≥ 5 次运行的步时极差）；峰值 HBM、编译期临时区还没记"
    else:
        m["threshold"] = thr
        over = rel is not None and abs(rel) > thr
        note = (f"只报。偏离 LKG {rel:+.2%}，{'超过' if over else '在'}阈值 {thr:.2%}{'，连续两次超过就开性能 issue' if over else '内'}"
                f"（阈值出自 calibration.yaml）；峰值 HBM、编译期临时区还没记")
    return {"status": "report", "checks": [], "metrics": m, "note": note}


def r23_coverage(tier: str, rows: dict, catalog, results: dict) -> dict:
    by_id = {c["id"]: c for c in results["checks"]}
    problems = []
    for row in ROWS[tier]:
        if row in ("R22", "R23"):
            continue
        if row not in rows or rows[row]["status"] == "error":
            problems.append(f"{row} 没出结论")
    for c in catalog.checks:
        for k, want in (c.get("expect") or {}).items():
            got = (by_id.get(c["id"]) or {}).get("metrics", {}).get(k)
            if got is None or int(got) != int(want):
                problems.append(f"{c['id']} 的 {k} = {got}，期望 {want}")
    return {"status": "pass" if not problems else "error", "checks": [],
            "metrics": {"rows": len(ROWS[tier]), "problems": len(problems)},
            "note": "每一行都出了结论、单元数对得上" if not problems else "；".join(problems)}


def r25_not_worse(rows: dict, lkg_results: Path | None, upper: dict | None = None) -> None:
    """R25「不变差」：回归门逐位通过时继承 LKG；否则新值 − LKG 值 ≤ TPU 执行噪声上限，噪声待标定时 inconclusive。"""
    if "R25" not in rows:
        return
    bitwise = all(rows.get(r, {}).get("status") == "pass" for r in ("R12", "R13", "R15", "R16"))
    ent = rows["R25"]
    test_m = next((c["metrics"] for c in ent["checks"]), {})
    keys = ("signed_mean_max", "mare_max", "max_step_max")
    ent["metrics"] = {f"test_{k}": test_m.get(k) for k in keys}
    if bitwise:
        ent["status"], ent["note"] = "pass", "回归门逐位通过：数值与 LKG 相同，对齐门结论继承 LKG（标准第二节）"
        return
    lkg_m = None
    if lkg_results and (lkg_results / "janus" / "results.json").is_file():
        lr = json.loads((lkg_results / "janus" / "results.json").read_text())
        lkg_m = next((c.get("metrics", {}) for c in lr["checks"] if c["id"] == "R25.loss_vs_golden"), None)
    if lkg_m:
        ent["metrics"].update({f"lkg_{k}": lkg_m.get(k) for k in keys})
        ent["metrics"].update({f"delta_{k}": (test_m.get(k) - lkg_m.get(k)) if test_m.get(k) is not None
                               and lkg_m.get(k) is not None else None for k in keys})
    if lkg_m and upper:
        worse = {k: ent["metrics"][f"delta_{k}"] for k in keys
                 if ent["metrics"].get(f"delta_{k}") is not None and ent["metrics"][f"delta_{k}"] > upper[k.removesuffix("_max")]}
        ent["metrics"]["tpu_noise_upper"] = {k: upper[k.removesuffix("_max")] for k in keys}
        ent["status"] = "fail" if worse else "pass"
        ent["note"] = (f"变差：{sorted(worse)} 的增量超过 TPU 执行噪声上限" if worse
                       else "不变差：对 GPU golden 的三个量，新值 − LKG 值都不大于 TPU 执行噪声上限（标准第三节）")
        return
    ent["status"] = "inconclusive"
    ent["note"] = ("数值变了：「不变差」要 TPU 执行噪声当尺子" + ("，还没标定或已作废" if not upper else "")
                   + "，只报新值与 LKG 值的差（标准第三节）"
                   + ("" if lkg_m else "；没给 LKG 的回归结果（--lkg-results），只报被测值"))


def classify(rows: dict) -> tuple[str, str | None]:
    """报告首行的分类（标准第八节）与第一个不同的量。"""
    if rows.get("R01", {}).get("status") in ("fail", "error"):
        return "基础设施或代码（跑不完）", "R01"
    errs = [r for r, e in rows.items() if e["status"] == "error"]
    if errs:
        return "工具", errs[0]
    fails = [r for r, e in rows.items() if e["status"] == "fail"]
    first = next((r for r in ORDER if rows.get(r, {}).get("status") == "fail"), fails[0] if fails else None)
    if fails:
        return "数值", first
    if any(e["status"] == "inconclusive" for e in rows.values()):
        return "无结论", next(r for r in ORDER + sorted(rows) if rows.get(r, {}).get("status") == "inconclusive")
    return "通过", None


def first_differ_detail(rows: dict, row: str | None) -> str:
    if not row:
        return "无"
    for c in rows[row]["checks"]:
        if c["status"] in ("fail", "error"):
            m = c["metrics"]
            extra = ""
            if "first_differ_step" in m:
                extra = f"，第一个不同的步 {m['first_differ_step']}"
            elif m.get("first_differ"):
                extra = f"，{m['first_differ']}"
            return f"{row} {c['id']}：{c['message'][:160]}{extra}"
    return f"{row}：{rows[row].get('note', '')}"


NEXT_STEP = {
    "通过": "无。数值与 LKG 逐位相同，LKG 指针可前移到被测提交（数值没变，不用批准）",
    "数值": "先跑对照臂（LKG 的提交在今天的环境上重跑一次）分清环境还是代码；对照臂逐位就二分，判据取第一个不同的量（标准第八节）",
    "工具": "看报 error 的行：运行目录缺文件、目录写错或 janus 版本变了；修好后重判，不动 LKG",
    "基础设施或代码（跑不完）": "看作业日志：起不来、节点被平台回收（基础设施中断）、超时 → 重跑；编译失败、OOM、程序报错且重跑复现 → 按「能不能跑完」二分",
    "无结论": "看报告里是哪类：门槛还没标定（TPU 执行噪声）的，标完才能判；没有容差门的行（如执行级下的 R09、R15）不逐位的，只能人工看报告里的数",
}


def cmd_report(a) -> int:
    cfg = tier_cfg(a.tier)
    janus_dir = Path(a.janus)
    results = json.loads((janus_dir / "results.json").read_text())
    catalog = load_catalog(results["catalog"]["path"])          # janus 实际跑的那份目录（行号、role、expect 从这里读）
    scope = {s.strip() for s in (a.scope or "").split(",") if s.strip()}
    if a.declare not in DECLARATIONS:
        raise SystemExit(f"--declare 只能是 {DECLARATIONS}")
    root = Path(a.root)
    test, lkg = Run.open(root / a.test), Run.open(root / a.lkg)
    rows = judge_rows(a.tier, catalog, results, a.declare, scope)
    registry_check(rows, a.lkg, cfg, results, a.declare, scope)
    platform0 = a.platform or "v5p"
    cal = load_calibration(platform0, a.tier)
    cal_ok, cal_note = calibration_valid(cal, test, cfg)
    if not cal_ok:            # 噪声参照没有或作废：用到它的容差门不能判 pass（标准第五节「参照作废」）
        for e in rows.values():
            if "容差门" in (e.get("note") or "") and e["status"] == "pass":
                e["status"], e["note"] = "inconclusive", e["note"] + "；" + cal_note
    rows["R22"] = r22_step_time(test, lkg, cal if cal_ok else None)
    upper = None
    if cal_ok and cal.get("noise_upper", {}).get("lm_loss"):
        upper = cal["noise_upper"]["lm_loss"]
    r25_not_worse(rows, Path(a.lkg_results) if a.lkg_results else None, upper)
    rows["R23"] = r23_coverage(a.tier, rows, catalog, results)
    cls, first = classify(rows)
    counts = {k: sum(1 for r in rows.values() if r["status"] == k) for k in ("pass", "fail", "inconclusive", "error", "report")}
    commits = {c["repo"]: c["commit"] for c in test.meta.get("code", [])}
    lkg_commits = {c["repo"]: c["commit"] for c in lkg.meta.get("code", [])}
    deps_same = rows.get("R04", {}).get("status") == "pass" and rows.get("R05", {}).get("status") == "pass"
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    verdict = {"通过": "PASS", "数值": "数值 FAIL", "工具": "ERROR（工具）", "基础设施或代码（跑不完）": "无结论（跑不完）",
               "无结论": "INCONCLUSIVE"}[cls]
    platform = a.platform or test.meta.get("hardware", {}).get("topology", "?")
    cand = (cls == "通过")
    out = {
        "tier": a.tier, "platform": platform, "test": a.test, "lkg": a.lkg, "declare": a.declare, "scope": sorted(scope),
        "class": cls, "verdict": verdict, "first_differ": first, "counts": counts,
        "tier_config": {"path": str(cfg["_path"].relative_to(REPO)), "sha256": cfg["_sha256"]},
        "catalog": _rel_catalog(results["catalog"]), "janus": results["janus"],
        "commits": {"test": commits, "lkg": lkg_commits},
        "rows": rows, "first_lkg": bool(a.first_lkg),
        "lkg_candidate": cand,
        "created_at": now.isoformat(timespec="seconds"),
    }
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    (od / "regression.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    (od / "report.md").write_text(render(out, deps_same))
    print((od / "report.md").read_text().split("\n\n")[0])
    return {"通过": 0, "数值": 1, "无结论": 3}.get(cls, 2)


def _rel(path: str) -> str:
    """仓库里的文件写成相对仓库根的路径（结果要进仓库，不带本机路径）。"""
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return path


def _rel_catalog(c: dict) -> dict:
    return {**c, "path": _rel(c["path"]), **({"includes": [{**i, "path": _rel(i["path"])} for i in c["includes"]]}
                                             if c.get("includes") else {})}


def _catalog_hash(entry: dict) -> str:
    """报告首屏的目录指纹：content-v1 写「内容指纹 …」；旧结果写原文件的「sha256 …」。"""
    algo, h = recorded_hash(entry)
    return f"内容指纹 {h[:12]}…" if algo == HASH_ALGO else f"sha256 {h[:12]}…"


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_fmt(x)}" for k, x in list(v.items())[:6]) + "}"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v[:5]) + ("…" if len(v) > 5 else "") + "]"
    return str(v)


def render(o: dict, deps_same: bool) -> str:
    c = o["counts"]
    tm = o["commits"]["test"].get("maxtext", "?")[:10]
    lm = o["commits"]["lkg"].get("maxtext", "?")[:10]
    head = [
        f"[{o['tier']} · {o['platform']} · {o['created_at'][:16].replace('T', ' ')}] {o['verdict']}",
        f"窗口：LKG {o['lkg']}@{lm} → {o['test']}@{tm}（声明：{o['declare']}"
        + (f"，范围 {','.join(o['scope'])}" if o["scope"] else "") + "）",
        f"依赖与开关：{'无变化（R04、R05 逐位）' if deps_same else '有变化或没判出（看 R04、R05）'}      "
        f"对照臂：{'本次就是同一提交双跑（建首个 LKG）' if o['first_lkg'] else '本次没跑（手动阶段）'}",
        f"第一个不同：{first_differ_detail(o['rows'], o['first_differ'])}",
        f"各行：pass {c['pass']} / fail {c['fail']} / inconclusive {c['inconclusive']} / error {c['error']}      "
        f"只报：{', '.join(r for r, e in o['rows'].items() if e['status'] == 'report') or '无'}",
        f"下一步：{NEXT_STEP[o['class']]}",
    ]
    lines = ["```", *head, "```", "",
             f"- 档位配置 `{o['tier_config']['path']}`（sha256 {o['tier_config']['sha256'][:12]}…）；"
             f"目录 `{Path(o['catalog']['path']).name}`（{_catalog_hash(o['catalog'])}）；janus {o['janus']['version']}",
             f"- 被测提交：{_fmt(o['commits']['test'])}；LKG 提交：{_fmt(o['commits']['lkg'])}",
             f"- LKG 候选：{'是' if o['lkg_candidate'] else '否'}"
             + ("（同一提交双跑逐位：这一档在这个平台上「同程序重跑逐位」的实测，见标准第七节）" if o["first_lkg"] and o["lkg_candidate"] else ""),
             "", "| 行 | 状态 | 检查 | 关键数 | 说明 |", "|---|---|---|---|---|"]
    for row in sorted(o["rows"], key=lambda r: int(r[1:])):
        e = o["rows"][row]
        if e["checks"]:
            for i, ch in enumerate(e["checks"]):
                key = ", ".join(f"{k}={_fmt(v)}" for k, v in list(ch["metrics"].items())[:5])
                msg = (ch["message"] or "").replace("|", "/").replace("\n", " ")[:140]
                note = e["note"] if i == 0 and e.get("note") else ""
                lines.append(f"| {row if i == 0 else ''} | {e['status'] if i == 0 else ''} | `{ch['id']}`（{ch['status']}"
                             f"{'' if ch['hard'] else '，只报'}） | {key} | {msg}{('；' + note) if note else ''} |")
        else:
            key = ", ".join(f"{k}={_fmt(v)}" for k, v in (e.get("metrics") or {}).items())
            lines.append(f"| {row} | {e['status']} | （regress.py） | {key} | {e.get('note', '')} |")
    lines += ["", "数字全部来自 janus 的 `results.json` 与运行目录（`regression.json` 是同一份数据），本报告不手写数字。"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------ lkg-entry
def cmd_lkg_entry(a) -> int:
    cfg = tier_cfg(a.tier)
    run = Run.open(Path(a.root) / a.run)
    m = run.meta
    cat = load_catalog(HERE / cfg["catalog"])
    ent = {
        "key": {"platform": a.platform, "tier": a.tier, "variant": a.variant},
        "run": a.run,
        **({"pair": a.pair} if a.pair else {}),
        "date": m.get("created_at", "")[:10],
        "code": {c["repo"]: c["commit"] for c in m.get("code", [])},
        "image": m.get("image"),
        "versions": m.get("versions"),
        "config": {"overrides": m.get("config", {}).get("overrides"), "yml_sha256": m.get("config", {}).get("sha256")},
        "tier_config_sha256": cfg["_sha256"],
        "catalog_sha256": {Path(cat.path).name: sha_file(cat.path),             # 原文件（留作历史）
                           **{Path(i["path"]).name: i["raw_sha256"] for i in cat.data.get("_includes", [])}},
        "catalog_content_sha256": {Path(cat.path).name: catalog_content_sha256(Path(cat.path)),   # R02 比这个
                                   **{Path(i["path"]).name: i["content_sha256"] for i in cat.data.get("_includes", [])}},
        "reason": a.reason,
        "approved_by": a.approved_by,
        "location": a.location,
        "results": a.results,
    }
    print(yaml.safe_dump([ent], allow_unicode=True, sort_keys=False, width=200), end="")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("args")
    s.add_argument("tier")
    s = sub.add_parser("report")
    s.add_argument("--tier", required=True)
    s.add_argument("--janus", required=True, help="janus run 的输出目录（有 results.json）")
    s.add_argument("--root", required=True)
    s.add_argument("--test", required=True)
    s.add_argument("--lkg", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--platform", default=None)
    s.add_argument("--declare", default="数学不变")
    s.add_argument("--scope", default="")
    s.add_argument("--lkg-results", default=None)
    s.add_argument("--first-lkg", action="store_true")
    s = sub.add_parser("lkg-entry")
    s.add_argument("--tier", required=True)
    s.add_argument("--root", required=True)
    s.add_argument("--run", required=True)
    s.add_argument("--pair", default=None)
    s.add_argument("--platform", default="v5p")
    s.add_argument("--variant", default="main")
    s.add_argument("--location", required=True)
    s.add_argument("--results", required=True)
    s.add_argument("--reason", required=True)
    s.add_argument("--approved-by", default=None)
    a = p.parse_args(argv)
    return {"args": cmd_args, "report": cmd_report, "lkg-entry": cmd_lkg_entry}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
