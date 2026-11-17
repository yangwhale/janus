#!/usr/bin/env python3
"""实践 04 阶梯实验：离线编译的训练步（aot_hlo.py 的产物）逐个读优化后 HLO，回答三件事：

1. 词嵌入反向那次 scatter-add（输出 [词表, 列]）：每卡一次收多少个下标、带不带 indices_are_sorted（排序 = 片上 FP32 累加、只舍一次；
   不排序 = 加一次舍一次 BF16，混元 4 案例 14）、输出 dtype；
2. 集合通信：all-reduce / reduce-scatter / all-gather / all-to-all 按「结果 dtype × 形状」计数（BF16 的梯度跨卡求和是候选的额外舍入）；
3. 程序指纹（去掉元数据与编号后的规范文本 sha256，同 models/hunyuan4/alignments/04-tpu-self/hlo_screen.py）：阶梯上相邻两级同一臂
   指纹相同 = 编出同一个程序，上卡逐位相同，不必重跑。

  python hlo_audit.py <aot 根目录> [--vocab 120832] [--out x.json]       # 根目录下每个子目录是一次 aot_hlo.py 的输出（含 aot.json、xla/）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "hunyuan4" / "alignments" / "04-tpu-self"))
import hlo_screen as hs  # noqa: E402

SCATTER = re.compile(r"%(\S+) = (bf16|f32)\[(\d+),(\d+)\]\S* scatter\(([^)]*)\)(.*)$")
COLL = re.compile(r"= \(?((?:bf16|f32|s32|u32|pred)\[[\d,]*\][^ ]*(?:, (?:bf16|f32|s32|u32|pred)\[[\d,]*\][^ ]*)*)\)? "
                  r"(all-reduce|reduce-scatter|all-gather|all-to-all)(?:-start)?\(")
SHAPE = re.compile(r"(bf16|f32|s32|u32|pred)\[([\d,]*)\]")


def defs(text: str) -> dict[str, str]:
    """%名字 → 类型串（只取第一段形状）。"""
    out = {}
    for m in re.finditer(r"%(\S+) = (\(?[a-z0-9]+\[[\d,]*\])", text):
        out.setdefault(m.group(1), m.group(2).lstrip("("))
    return out


def audit(text: str, vocab: int) -> dict:
    d = defs(text)
    emb = []
    for ln in text.splitlines():
        m = SCATTER.search(ln)
        if not m or int(m.group(3)) != vocab:
            continue
        ops = [o.strip().lstrip("%") for o in m.group(5).split(",")]
        idx_t = d.get(ops[1], "?") if len(ops) > 1 else "?"
        n = SHAPE.search(idx_t)
        dims = [int(x) for x in n.group(2).split(",") if x] if n else []
        cnt = 1
        for x in dims:
            cnt *= x
        emb.append({"dtype": m.group(2), "shape": [int(m.group(3)), int(m.group(4))], "indices": idx_t,
                    "n_indices": cnt if dims else None, "sorted": "indices_are_sorted=true" in m.group(6)})
    coll: Counter = Counter()
    for ln in text.splitlines():
        m = COLL.search(ln)
        if not m:
            continue
        for s in SHAPE.finditer(m.group(1)):
            coll[f"{m.group(2)} {s.group(1)}[{s.group(2)}]"] += 1
    vocab_ar = {k: v for k, v in coll.items() if k.startswith(("all-reduce", "reduce-scatter")) and f"[{vocab}," in k}
    return {"emb_scatter": emb, "vocab_table_reductions": vocab_ar,
            "bf16_reductions": {k: v for k, v in coll.items() if k.startswith(("all-reduce", "reduce-scatter")) and " bf16[" in k},
            "collectives": dict(sorted(coll.items()))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--vocab", type=int, default=120832)
    ap.add_argument("--only", default="", help="子目录名的正则")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    res = {}
    for sub in sorted(Path(a.root).iterdir()):
        if not sub.is_dir() or (a.only and not re.search(a.only, sub.name)):
            continue
        aot = json.loads((sub / "aot.json").read_text()) if (sub / "aot.json").exists() else {"error": "没编"}
        files = sorted((sub / "xla").glob("*jit_train_step*after_optimizations.txt"))
        if aot.get("error") or not files:
            res[sub.name] = {"error": str(aot.get("error"))[:300]}
            continue
        t = files[-1].read_text()
        mem = None
        rep = sorted((sub / "xla").glob("*jit_train_step*memory-usage-report.txt"))
        if rep:
            mm = re.search(r"Total bytes: (\d+)", rep[-1].read_text())
            mem = int(mm.group(1)) if mm else None
        res[sub.name] = {"fp": hs.program_fp(hs.canon_lines(t)), "extra": aot.get("extra"), "mesh": aot.get("mesh"),
                         "hbm_bytes": mem, "versions": aot.get("versions"), **audit(t, a.vocab)}
    js = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out:
        Path(a.out).write_text(js + "\n")
    for k, v in res.items():
        if "error" in v:
            print(k, "ERROR", v["error"][:120])
            continue
        e = "; ".join(f"{x['dtype']}{x['shape']} n={x['n_indices']} {'sorted' if x['sorted'] else 'UNSORTED'}" for x in v["emb_scatter"])
        print(f"{k:12s} {v['fp'][:10]} hbm={v['hbm_bytes'] / 2**30 if v['hbm_bytes'] else 0:.1f}G emb[{e}] "
              f"vocabAR={v['vocab_table_reductions']} bf16AR={sum(v['bf16_reductions'].values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
