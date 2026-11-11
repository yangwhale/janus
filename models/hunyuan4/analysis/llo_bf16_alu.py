"""整步 LLO 里哪些 kernel 用 BF16 的向量算术指令（vadd/vsub/vmul/vmax/vmin/vmac… .bf16），按源文件归到 RMSNorm、iHC。

v5p 的向量单元没有 BF16 算术，BF16 运算都解包成 FP32 算（vadd.f32 等）；v7x 有 BF16 算术，后端可以让一部分 BF16
逐元素运算直接在 BF16 里算，每步舍一次。这个脚本数整步编出来的每颗 kernel 里这类指令的静态条数。

kernel 归类：按优化后 HLO 的栈帧表（FileNames / FileLocations / StackFrames），融合指令本身与它调用的计算里任一条指令
来自 normalizations.py 记 RMSNorm、来自 mhc.py 记 iHC，其余记 other。一颗 kernel 可以同时属于两类。

LLO dump 的取法（CPU 上整步 AOT，0 张卡）：用 hlo_screen.compile_spec 编 golden 的规格，libtpu_args 里加
  --xla_jf_dump_to=/out/jf --xla_jf_dump_llo_text=true
dump 落在缓存目录的 jf/ 下（整步约 40 GB，数完就删）。

  python models/hunyuan4/analysis/llo_bf16_alu.py <缓存目录/aot/<规格哈希>> [<另一份> …] --out <json>
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

ARITH = re.compile(r"\bv(add|sub|mul|max|min|mac|fma)(\.[a-z]+)*\.bf16\b")
F32 = re.compile(r"\bv(add|sub|mul|max|min|mac|fma)\.f32\b")
TAGS = (("rmsnorm", "normalizations.py"), ("ihc", "mhc.py"))


def _table(lines, name):
    out, i = {}, lines.index(name) + 1
    while i < len(lines) and lines[i] and lines[i][0].isdigit():
        k, _, v = lines[i].partition(" ")
        out[int(k)] = v
        i += 1
    return out


def kernel_files(hlo_text: str):
    """kernel 名 → 它（及其调用的计算）各指令栈帧里出现过的源文件名集合。"""
    lines = hlo_text.splitlines()
    files = {k: v.strip('"').split("/")[-1] for k, v in _table(lines, "FileNames").items()}
    locs = {k: int(re.search(r"file_name_id=(\d+)", v).group(1)) for k, v in _table(lines, "FileLocations").items()}
    frames = {
        k: (int(re.search(r"file_location_id=(\d+)", v).group(1)), int(re.search(r"parent_frame_id=(\d+)", v).group(1)))
        for k, v in _table(lines, "StackFrames").items()
    }
    memo: dict[int, set] = {}

    def files_of(fid):
        if fid not in memo:
            out, seen, f = set(), set(), fid
            while f in frames and f not in seen:
                seen.add(f)
                loc, f2 = frames[f]
                out.add(files[locs[loc]])
                f = f2
            memo[fid] = out
        return memo[fid]

    comp, calls, own = collections.defaultdict(set), {}, {}
    cur = None
    for ln in lines:
        m = re.match(r"^(?:ENTRY )?%([\w.\-]+) ", ln)
        if m and ln.rstrip().endswith("{"):
            cur = m.group(1)
            continue
        m = re.match(r"^\s*(?:ROOT )?%([\w.\-]+) = ", ln)
        if not m:
            continue
        s = re.search(r"stack_frame_id=(\d+)", ln)
        if s:
            comp[cur].add(int(s.group(1)))
            own[m.group(1)] = int(s.group(1))
        c = re.search(r"calls=%([\w.\-]+)", ln)
        if c:
            calls[m.group(1)] = c.group(1)

    def get(k):
        fs = set(files_of(own[k])) if k in own else set()
        for f in comp.get(calls.get(k), ()):
            fs |= files_of(f)
        return fs

    return get


def count(d: Path) -> dict:
    spec = json.loads((d / "spec.json").read_text()) if (d / "spec.json").exists() else {}
    get = kernel_files((d / "train_step.after_optimizations.txt").read_text())
    agg = collections.defaultdict(
        lambda: {
            "kernels": 0,
            "kernels_with_bf16_arith": 0,
            "bf16_arith": collections.Counter(),
            "f32_arith": 0,
            "top": [],
        }
    )
    for f in sorted((d / "jf").glob("*-final_bundles.txt")):
        if "schedule-analysis" in f.name:
            continue
        m = re.match(r"\d+-(.+)-\d+-final_bundles\.txt", f.name)
        k = m.group(1) if m else f.name
        t = f.read_text(errors="ignore")
        c = collections.Counter(x.group(0) for x in ARITH.finditer(t))
        fs = get(k)
        for tag in [t_ for t_, src in TAGS if src in fs] or ["other"]:
            g = agg[tag]
            g["kernels"] += 1
            g["kernels_with_bf16_arith"] += bool(c)
            g["bf16_arith"].update(c)
            g["f32_arith"] += len(F32.findall(t))
            if c:
                g["top"].append([k, dict(c)])
    for g in agg.values():
        g["bf16_arith"] = dict(g["bf16_arith"])
        g["top"] = sorted(g["top"], key=lambda e: -sum(e[1].values()))[:10]
    return {"platform": spec.get("platform"), "libtpu": spec.get("libtpu"), "by_tag": dict(agg)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    res = [count(d) for d in a.dirs]
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    for r in res:
        print(
            r["platform"],
            r["libtpu"],
            {k: (v["kernels"], v["kernels_with_bf16_arith"]) for k, v in r["by_tag"].items()},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
