"""TPU 自比第 1 步：给 iHC 豁免加倍数上限（R ≤ IHC_R64_CAP 才单列、不计入未定责），用入库的数重判全部结果文件。

规则（PROTOCOL 变更记录 2026-11-12）：超出 GPU 尺子、R > 1.2 的 iHC 张量，R ≤ 上限的照旧单列为 ihc_accepted、不计入未定责；
R > 上限的记 unattributed，进判档。上限取 01-mini D11 实际接受过的最大 R：base2 开那一代（与自比 golden 同一代）GPU↔TPU
第 1 步登记接受的 6 个 iHC 张量（`t1zg-step1-train-grad-vs-fp64.json` 的 judge_fp64 里 R > 1.2 的那 6 个）R 1.57–5.218，取 5.218
向上到两位 5.22。脚本先从那份文件重算这个最大值，对不上就停。

每个结果文件（04-tpu-self/results/*-vs-*.json，带 step1_rows 与 r64 的）：
  1. 按旧规则（iHC 无上限）复算每个张量的判定与判档，和文件里存的逐张量、逐项对上，对不上就停；
  2. 按新规则重判：step1_rows 的 verdict、step1_grad 的 n_unattributed / unattributed / n_ihc_accepted / ihc_accepted / within，
     verdict 的 tier / band / why（曲线不重算，用文件里存的 curves）；
  3. --apply 时写回文件（规则写在 step1_grad.ihc_rule；有张量改判的文件，旧判档另存在 verdict.before_ihc_cap），否则只出汇总。

    python models/hunyuan4/analysis/tpu_self_ihc_cap_rejudge.py \
        --out models/hunyuan4/alignments/04-tpu-self/results/ihc-cap-rejudge.json [--apply]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "alignments/04-tpu-self/results"
D11 = ROOT / "alignments/01-mini/results/t1zg-step1-train-grad-vs-fp64.json"
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tpu_self_compare import IHC, IHC_R64_CAP, R64_LIMIT  # noqa: E402  同一套规则
from tpu_self_compare import IHC_RULE as RULE

# 早期结果文件用旧叫法（2026-11-10 改名，只改叫法不改判定）
OLD_NAMES = {"在刻度内": "在噪声内", "超出刻度": "超出噪声上限"}
KEYS = ("tensor", "d", "scale", "over_scale", "test_vs_fp64", "golden_vs_fp64", "gpu_vs_fp64", "r64")


def d11_max_r():
    d = json.loads(D11.read_text())
    j = d["judge_fp64"]
    acc = set(j["unattributed_over_2N"] + j["unattributed_1_2N"])
    rows = [r for r in d["rows"] if r["tensor"] in acc]
    assert len(rows) == 6 and all(IHC.search(r["tensor"]) for r in rows), "D11 接受的应是 6 个 iHC 张量"
    return max(r["R_vs_ref"] for r in rows), min(r["R_vs_ref"] for r in rows)


def verdict_of(r, cap):
    if r["over_scale"] <= 1:
        return "scale"
    if r["r64"] <= R64_LIMIT:
        return "attributed"
    if IHC.search(r["tensor"]) and r["r64"] <= cap:
        return "ihc_accepted"
    return "unattributed"


def band_of(d, n_un):
    """tpu_self_compare.py 的判档，第 1 步部分换成给定的未定责个数。"""
    bw = d["bitwise"]
    tier1 = bw["metrics"]["equal"] and bw["step1_dump"].get("equal", True)
    rmax = d["curves"]["r_max"] or 0.0
    s1_ok = n_un == 0
    if tier1:
        return 1, "逐位相同"
    if s1_ok and d["curves"]["within"]:
        return 2, "在噪声内"
    return 3, ("超" if (rmax > 2 or not s1_ok) else "略超")


def why_of(d, un):
    tier, band = band_of(d, len(un))
    if tier == 1:
        return "逐位相同"
    if tier == 2:
        return "不逐位；第 1 步梯度没有未定责的张量，曲线 r ≤ 1"
    bad = []
    if un:
        bad.append(
            f"第 1 步梯度 {len(un)} 个张量超出噪声上限且离 FP64 远 1.2 倍以上"
            f"（最远 {un[0]['tensor']}，{un[0]['r64']:.3g} 倍）"
        )
    bad += [f"{c['id']} r {c['r']:.3g}" for c in d["curves"]["checks"] if c["r"] > 1]
    return "超出噪声上限：" + "；".join(bad)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--apply", action="store_true", help="把新判定写回各结果文件")
    a = p.parse_args(argv)
    mx, mn = d11_max_r()
    assert math.ceil(mx * 100) / 100 == IHC_R64_CAP, f"D11 接受过的最大 R {mx} 与上限 {IHC_R64_CAP} 对不上"
    runs, n_files, n_already = [], 0, 0
    for f in sorted(RESULTS.glob("*-vs-*.json")):
        d = json.loads(f.read_text())
        rows = d.get("step1_rows") if isinstance(d, dict) else None
        if not rows or "r64" not in rows[0]:
            continue
        n_files += 1
        s1 = d["step1_grad"]
        if s1.get("ihc_rule"):  # 已按新规则写回过：用 before_ihc_cap 复核旧判档
            n_already += 1
        old_v = {r["tensor"]: verdict_of(r, math.inf) for r in rows}
        if not s1.get("ihc_rule"):
            for r in rows:
                if old_v[r["tensor"]] != r["verdict"]:
                    raise SystemExit(
                        f"{f.name} {r['tensor']}：旧规则复算 {old_v[r['tensor']]} 与文件 {r['verdict']} 对不上"
                    )
        old_un = sum(v == "unattributed" for v in old_v.values())
        old_tier, old_band = band_of(d, old_un)
        stored = d["verdict"].get("before_ihc_cap") or d["verdict"]
        if (old_tier, old_band) != (stored["tier"], OLD_NAMES.get(stored["band"], stored["band"])):
            raise SystemExit(f"{f.name}：旧规则复算判档 {old_band} 与文件 {stored['band']} 对不上")
        new_v = {r["tensor"]: verdict_of(r, IHC_R64_CAP) for r in rows}
        moved = [r for r in rows if new_v[r["tensor"]] != old_v[r["tensor"]]]
        un = sorted((r for r in rows if new_v[r["tensor"]] == "unattributed"), key=lambda r: -r["r64"])
        ih = sorted((r for r in rows if new_v[r["tensor"]] == "ihc_accepted"), key=lambda r: -r["r64"])
        tier, band = band_of(d, len(un))
        rec = {
            "file": f.name,
            "band": [old_band, band],
            "curves_r_max": d["verdict"].get("curves_r_max"),
            "n_unattributed": [old_un, len(un)],
            "n_ihc_accepted": [sum(v == "ihc_accepted" for v in old_v.values()), len(ih)],
            "moved_to_unattributed": [
                {k: r[k] for k in ("tensor", "r64", "over_scale", "test_vs_fp64", "golden_vs_fp64", "gpu_vs_fp64")}
                for r in sorted(moved, key=lambda r: -r["r64"])
            ],
        }
        if moved:
            runs.append(rec)
        if a.apply and not s1.get("ihc_rule"):
            for r in rows:
                r["verdict"] = new_v[r["tensor"]]
            s1.update(
                {
                    "n_unattributed": len(un),
                    "unattributed": [{k: r[k] for k in KEYS} for r in un[:10]],
                    "n_ihc_accepted": len(ih),
                    "ihc_accepted": [{k: r[k] for k in KEYS} for r in ih[:10]],
                    "within": not un,
                    "ihc_rule": RULE,
                }
            )
            if moved:  # 判定有变的才改 verdict，旧判档另存一份
                v = d["verdict"]
                before = {k: v[k] for k in ("tier", "label", "band", "why")}
                v.update(
                    {
                        "tier": tier,
                        "label": {1: "逐位相同", 2: "在噪声内", 3: "超出噪声上限"}[tier],
                        "band": band,
                        "why": why_of(d, un),
                        "before_ihc_cap": before,
                    }
                )
            f.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    out = {
        "what": "TPU 自比第 1 步给 iHC 豁免加倍数上限后重判（只用入库的数；曲线不重算，判档按 tpu_self_compare.py 同一判法）",
        "how": "python models/hunyuan4/analysis/tpu_self_ihc_cap_rejudge.py --out <本文件> [--apply]",
        "rule": RULE,
        "cap_source": {
            "file": str(D11.relative_to(ROOT)),
            "d11_accepted_ihc_R_min": mn,
            "d11_accepted_ihc_R_max": mx,
            "cap": IHC_R64_CAP,
        },
        "n_files": n_files,
        "old_rule_recheck": "旧规则（iHC 无上限）复算与各结果文件存的逐张量判定、判档都相同",
        "band_changed": [x for x in runs if x["band"][0] != x["band"][1]],
        "runs_with_tensors_moved": runs,
    }
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(
        f"{n_files} 个结果文件，{len(runs)} 个有张量从单列改记未定责，判档变了的 {len(out['band_changed'])} 个"
        + (f"（{n_already} 个已写回过）" if n_already else "")
    )
    for x in out["band_changed"]:
        print(
            " ",
            x["file"],
            " → ".join(x["band"]),
            x["n_unattributed"],
            [(m["tensor"].split(".")[-1], round(m["r64"], 1)) for m in x["moved_to_unattributed"]],
        )


if __name__ == "__main__":
    main()
