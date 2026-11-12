"""训练步优化后 HLO 里的集合通信清单：每条指令的类型、形状、每步执行几次、每步每设备字节数。

用法：
  python models/hunyuan4/analysis/hlo_comm_bytes.py <名字>=<train_step.after_optimizations.txt> … [--vocab 120832] [--out <json>]

口径（都是按 HLO 里的形状算，不是线上实际传输的字节；后者还取决于通信算法，例如 ring all-reduce 每设备约收发 2(n−1)/n 倍）：
- 字节 = 指令结果形状的字节数（all-reduce 是被求和的那块，all-gather 是收齐后的结果，all-to-all 是交换的那块；
  异步 collective-permute 的结果是 (发送, 接收, 上下文) 元组，只算一块）。
- 每步次数 = 外层各个 while 的循环次数之积（梯度累积、层扫描、窗口化矩阵乘）。TPU 后端优化后的 HLO 不带 known_trip_count（这里读的都是上卡存下的 TPU 程序），
  循环次数从条件计算里「计数器 < 常数」的常数读出；读不出的按 1 算并在 warnings 里列出。
- 按（类型, 形状）合并成行，每行带一个 op_name 样本（JAX 的源头算子名，用来认出是哪个参数的通信）。
  哪几行属于词嵌入要看数据流认定（例如 token id 的 all-gather 喂给排序再进 scatter），这个脚本不替你判断。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

DT = {"bf16": 2, "f32": 4, "s32": 4, "u32": 4, "f16": 2, "s8": 1, "u8": 1, "pred": 1, "f64": 8, "s64": 8}
OPS = ("all-gather", "all-reduce", "reduce-scatter", "all-to-all", "collective-permute", "ragged-all-to-all")
ARR = re.compile(r"\b(bf16|f32|s32|u32|f16|s8|u8|pred|f64|s64)\[([0-9,]*)\]")
INSTR = re.compile(r"^\s+(?:ROOT )?%([\w.\-]+) = (.*)$")
COLL = re.compile(r"^(.*?) (" + "|".join(OPS) + r")(?:-start)?\((.*)$")


def _arrays(s):
    out = []
    for dt, dims in ARR.findall(s):
        shape = tuple(int(d) for d in dims.split(",") if d)
        n = 1
        for d in shape:
            n *= d
        out.append((dt, shape, n * DT[dt]))
    return out


def parse(path):
    comp = None
    entry = None
    edges = defaultdict(list)
    consts = defaultdict(dict)
    lt_root = {}
    whiles = []
    colls = []
    for line in open(path):
        if not line.startswith("  "):
            m = re.match(r"^(ENTRY )?%([\w.\-]+) .*\{\s*$", line)
            if m:
                comp = m.group(2)
                entry = comp if m.group(1) else entry
            continue
        mi = INSTR.match(line)
        if not mi or comp is None:
            continue
        name, rhs = mi.groups()
        rhs = re.sub(r",?\s*metadata=\{[^{}]*\}", "", rhs)
        cm = re.match(r"[su]32\[\]\S* constant\((-?\d+)\)", rhs)
        if cm:
            consts[comp][name] = int(cm.group(1))
        rm = re.match(r"pred\[\]\S* compare\(%[\w.#\-]+, %([\w.\-]+)\), direction=LT", rhs)
        if rm and line.lstrip().startswith("ROOT"):
            lt_root[comp] = rm.group(1)
        if " while(" in rhs:
            b = re.search(r"body=%([\w.\-]+)", rhs)
            c = re.search(r"condition=%([\w.\-]+)", rhs)
            whiles.append((comp, b.group(1), c.group(1)))
        for key in ("calls", "to_apply"):
            for c in re.findall(key + r"=%([\w.\-]+)", rhs):
                edges[comp].append((c, 1))
        bc = re.search(r"branch_computations=\{([^}]*)\}", rhs)
        if bc:
            for c in re.findall(r"%([\w.\-]+)", bc.group(1)):
                edges[comp].append((c, 1))
        mc = COLL.match(rhs)
        if mc and " while(" not in rhs:
            shp, op, _ = mc.groups()
            arrs = _arrays(shp)
            if op == "collective-permute" and shp.startswith("("):
                arrs = arrs[:1]
            opn = re.search(r'op_name="([^"]*)"', line)
            colls.append({
                "name": name, "op": op, "comp": comp,
                "shape": " ".join(f"{dt}[{','.join(map(str, s))}]" for dt, s, _ in arrs),
                "dtypes": sorted({dt for dt, _, _ in arrs}),
                "bytes": sum(b for _, _, b in arrs),
                "op_name": opn.group(1) if opn else "",
            })
    warnings = []
    for parent, body, cond in whiles:
        k = lt_root.get(cond)
        trip = consts[cond].get(k) if k else None
        if trip is None:
            warnings.append(f"循环次数读不出：{body}，按 1 算")
            trip = 1
        edges[parent].append((body, trip))
    mult = defaultdict(int)
    mult[entry] = 1
    order, seen = [], set()

    def topo(c):
        if c in seen:
            return
        seen.add(c)
        for ch, _ in edges.get(c, []):
            topo(ch)
        order.append(c)

    topo(entry)
    for c in reversed(order):
        for ch, t in edges.get(c, []):
            mult[ch] += mult[c] * t

    for x in colls:
        x["times"] = mult.get(x["comp"], 0)
        x["bytes_per_step"] = x["bytes"] * x["times"]
    return colls, warnings


def summarize(colls):
    agg = {}
    for x in colls:
        k = (x["op"], x["shape"])
        a = agg.setdefault(k, {"op": x["op"], "shape": x["shape"], "instructions": 0, "times_per_step": 0,
                               "bytes_per_step": 0, "op_name_sample": x["op_name"]})
        a["instructions"] += 1
        a["times_per_step"] += x["times"]
        a["bytes_per_step"] += x["bytes_per_step"]
    rows = sorted(agg.values(), key=lambda r: -r["bytes_per_step"])
    return {"bytes_per_step_total": sum(x["bytes_per_step"] for x in colls), "instructions": len(colls), "rows": rows}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hlo", nargs="+", help="<名字>=<train_step.after_optimizations.txt>")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    res = {"runs": {}}
    for spec in a.hlo:
        name, path = spec.split("=", 1)
        colls, warnings = parse(path)
        s = summarize(colls)
        s["warnings"] = warnings
        res["runs"][name] = s
        print(f"{name}: 每步每设备 {s['bytes_per_step_total']:.3e} 字节（{s['instructions']} 条集合通信）")
        for r in s["rows"]:
            if r["bytes_per_step"] >= 1e5:
                print(f"    {r['op']:18s} {r['shape'][:60]:60s} 每步 {r['times_per_step']:4d} 次  {r['bytes_per_step']:.3e}  {r['op_name_sample'][-60:]}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
