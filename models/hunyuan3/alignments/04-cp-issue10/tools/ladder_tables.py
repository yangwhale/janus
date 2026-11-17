#!/usr/bin/env python3
"""从 results/ladder/ 的 step1-*.json、ruler-*.json 印出 README 用的几张 markdown 表（数字只从结果文件来，不手写）。

  python ladder_tables.py --res results/ladder --prefix lbd
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LEVELS = ["L0", "L0p", "L1", "L2", "L3", "L4"]
CP = ["ag2", "ag4", "rg2", "rg4", "ul2", "ul4", "usp4"]
NAME = {"L0p": "L0′"}


def f(x, fmt=".2e"):
    return "—" if x is None else format(x, fmt)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--prefix", default="lbd")
    a = ap.parse_args(argv)
    R = Path(a.res)
    s1 = {L: json.loads(p.read_text()) for L in LEVELS if (p := R / f"step1-{a.prefix}-{L}.json").exists()}
    ru = {L: json.loads(p.read_text()) for L in LEVELS if (p := R / f"ruler-{a.prefix}-{L}.json").exists()}

    print("### 第 1 步：词嵌入梯度离 FP64（rel_l2）\n")
    print("| 级 | " + " | ".join(["dp8", "fsdp8", *CP]) + " |")
    print("|---|" + "---|" * (2 + len(CP)))
    for L, d in s1.items():
        print(f"| {NAME.get(L, L)} | " + " | ".join(f(d["runs"].get(k, {}).get("groups", {}).get("embedding")) for k in ["dp8", "fsdp8", *CP]) + " |")

    print("\n### 第 1 步：整网（按 FP64 梯度能量合起来）离 FP64，括号里是对同级 FSDP 参照的倍数\n")
    print("| 级 | " + " | ".join(["dp8", "fsdp8", *CP]) + " |")
    print("|---|" + "---|" * (2 + len(CP)))
    for L, d in s1.items():
        fs = d["runs"].get("fsdp8", {}).get("pooled")
        cells = []
        for k in ["dp8", "fsdp8", *CP]:
            p = d["runs"].get(k, {}).get("pooled")
            cells.append("—" if p is None else (f"{p:.2e}" + (f"（{p / fs:.2f}）" if fs and k != "fsdp8" else "")))
        print(f"| {NAME.get(L, L)} | " + " | ".join(cells) + " |")

    print("\n### 第 1 步：比同级 FSDP 参照远 1.2 倍以上的张量（路由专家另列）\n")
    print("| 级 | 臂 | 张量（倍数） |")
    print("|---|---|---|")
    for L, d in s1.items():
        for k, p in d["pairs"].items():
            over = p.get("over_ref", [])
            if over:
                txt = "、".join(f"`{z[0][len('grad.'):]}`（{z[1]:.2f}）" for z in over[:6]) + ("…" if len(over) > 6 else "")
                print(f"| {NAME.get(L, L)} | {k} | {txt} |")

    print("\n### 第 1 步 loss − FP64\n")
    print("| 级 | " + " | ".join(["dp8", "fsdp8", *CP]) + " |")
    print("|---|" + "---|" * (2 + len(CP)))
    for L, d in s1.items():
        print(f"| {NAME.get(L, L)} | " + " | ".join(f(d["runs"].get(k, {}).get("loss1_minus_fp64"), "+.1e") for k in ["dp8", "fsdp8", *CP]) + " |")

    for wk in ["2:3001", "2000:3001"]:
        print(f"\n### 长跑 {wk} 步：尺子 N（带符号均值 / MARE / 最大单步）与各 CP 臂的带符号均值、r（三个统计量里最差的）\n")
        print("| 级 | N | " + " | ".join(CP) + " |")
        print("|---|---|" + "---|" * len(CP))
        for L, d in ru.items():
            w = d["windows"][wk]
            N = w["N"]
            cells = []
            for k in CP:
                t = d["tests"].get(k, {}).get(wk)
                cells.append("—" if t is None else f"{t['signed_mean']:+.1e} / {t['r_worst']:.2f}")
            print(f"| {NAME.get(L, L)} | {N['signed_mean']:.1e} / {N['mare']:.1e} / {N['max_step']:.1e} | " + " | ".join(cells) + " |")
    print("\n### 尺子自己：4 份只改编译开关对基线，2000–3000 步带符号均值\n")
    print("| 级 | v64 | v56 | v40 | v24 | 留一法最大 r |")
    print("|---|---|---|---|---|---|")
    for L, d in ru.items():
        w = d["windows"]["2000:3001"]
        cells = [f(w["pairs"].get(f"{r}-base", {}).get("signed_mean"), "+.1e") for r in ["v64", "v56", "v40", "v24"]]
        loo = max([x["r"] for x in w.get("loo", [])] or [float("nan")])
        print(f"| {NAME.get(L, L)} | " + " | ".join(cells) + f" | {loo:.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
