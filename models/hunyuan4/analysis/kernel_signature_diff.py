"""两份优化后 HLO 里 Pallas kernel（tpu_custom_call）的输出签名比对：补「编一次、扫一遍」看不到的 kernel 内部精度变化。

elided_rounding_scan.py 只看融合里的 convert 与 reduce_precision。kernel 内部的累加精度（例如 splash 反向的 dq 部分和
从 FP32 缓冲换成 BF16 缓冲）不在融合里，那份扫描读不到；但缓冲是 kernel 的输出，dtype 写在 custom-call 的结果类型里。
本工具按 kernel 名（去掉编号后缀）把两边每颗 custom-call 的结果类型列出来比：

- 个数变了：多了或少了某种 kernel（换了实现、分块数变了）；
- 同名 kernel 的结果类型变了：形状变、dtype 变。某个结果从 f32 变成 bf16 / f16 记「变窄」，是数值相关的变化，要逐处判；
  只有形状变（例如分块数）记「形状」，一般是执行方式变化，也要看一眼。

实例（变异测试）：tokamax 0.0.17 去掉静态 dK/dV 网格包装后，splash dkv kernel 的第一个结果
从 f32[按 kv 块数, …] 变成 bf16[3, …]（dq 三缓冲），融合舍入扫描新增 0 处、个数也不变，只有这里看得出来。

    python kernel_signature_diff.py --base <基线 after_optimizations.txt> --test <被测 after_optimizations.txt> [--out <json>]

退出码：0 两边签名相同；1 有不同（有「变窄」时报告首行写明）；2 读文件出错。
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys

CALL = re.compile(r"^\s*(?:ROOT )?%([\w.\-]+) = (.+?) custom-call\(")
TARGET = 'custom_call_target="tpu_custom_call"'
SUFFIX = re.compile(r"[._]\d+$")
TYPE = re.compile(r"\b(pred|s8|s16|s32|s64|u8|u16|u32|u64|f8e4m3fn|f8e5m2|bf16|f16|f32|f64)\[([\d,]*)\]")
WIDTH = {"f64": 64, "f32": 32, "bf16": 16, "f16": 16, "f8e4m3fn": 8, "f8e5m2": 8}


def kernel_name(inst: str) -> str:
    """指令名去掉 XLA 加的编号后缀：splash_mha_dkv_segmented_no_residuals.37 → splash_mha_dkv_segmented_no_residuals。"""
    name = inst
    while SUFFIX.search(name):
        name = SUFFIX.sub("", name)
    return name


def result_types(text: str) -> tuple[tuple[str, str], ...]:
    """结果类型（单个或元组）里的 (dtype, 形状) 序列，布局与 tiling 注记去掉。"""
    return tuple(TYPE.findall(re.sub(r"\{[^}]*\}", "", text)))


def signatures(path: str) -> collections.Counter:
    """(kernel 名, 结果类型) → 出现次数。"""
    out: collections.Counter = collections.Counter()
    with open(path) as f:
        for line in f:
            if TARGET not in line:
                continue
            m = CALL.match(line)
            if m:
                out[(kernel_name(m.group(1)), result_types(m.group(2)))] += 1
    return out


def fmt(sig) -> str:
    return "(" + ", ".join(f"{d}[{s}]" for d, s in sig) + ")"


def narrowed(a, b) -> list[str]:
    """同名 kernel、结果个数相同时，逐个结果看浮点 dtype 有没有变窄。"""
    out = []
    if len(a) != len(b):
        return out
    for i, ((da, _), (db, _)) in enumerate(zip(a, b)):
        if da in WIDTH and db in WIDTH and WIDTH[db] < WIDTH[da]:
            out.append(f"结果 {i}: {da} → {db}")
    return out


def compare(base: collections.Counter, test: collections.Counter) -> dict:
    names = sorted({k for k, _ in base} | {k for k, _ in test})
    rows = []
    for n in names:
        sa = {s: c for (k, s), c in base.items() if k == n}
        sb = {s: c for (k, s), c in test.items() if k == n}
        if sa == sb:
            continue
        only_a = {s: c for s, c in sa.items() if sb.get(s) != c}
        only_b = {s: c for s, c in sb.items() if sa.get(s) != c}
        narrow = []
        for x in only_a:
            for y in only_b:
                narrow += narrowed(x, y)
        rows.append(
            {
                "kernel": n,
                "base": {fmt(s): c for s, c in sorted(only_a.items())},
                "test": {fmt(s): c for s, c in sorted(only_b.items())},
                "count_base": sum(sa.values()),
                "count_test": sum(sb.values()),
                "narrowed": sorted(set(narrow)),
            }
        )
    return {
        "kernels_base": sum(base.values()),
        "kernels_test": sum(test.values()),
        "changed": rows,
        "narrowed": any(r["narrowed"] for r in rows),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    try:
        res = compare(signatures(a.base), signatures(a.test))
    except OSError as e:
        print(f"读文件出错：{e}", file=sys.stderr)
        return 2
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    if not res["changed"]:
        print(f"kernel 签名相同（{res['kernels_base']} 颗）")
        return 0
    head = "有 kernel 结果 dtype 变窄（数值相关，逐处判）" if res["narrowed"] else "kernel 签名有变化（只有个数或形状）"
    print(f"{head}：基线 {res['kernels_base']} 颗，被测 {res['kernels_test']} 颗")
    for r in res["changed"]:
        print(f"- {r['kernel']}（{r['count_base']} → {r['count_test']}）")
        for s, c in r["base"].items():
            print(f"    基线 ×{c} {s}")
        for s, c in r["test"].items():
            print(f"    被测 ×{c} {s}")
        for n in r["narrowed"]:
            print(f"    变窄：{n}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
