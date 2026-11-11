"""hc_head 第 1 步梯度偏差的 HLO 级判据：主输出头反向（dgrad）那颗卷积 kernel 里，有没有并进末层 RMSNorm 反向的每 token 行和 S。

背景（04-tpu-self MATRIX「hc_head」）：v5p 上「D3 只改末层」的程序里，偏差定位到输出头 dgrad 卷积 kernel 的融合切分：
偏的程序把 S = Σ_d x·s·dy 并进卷积 kernel，S 用 kernel 里的 dy，主项 r·s·dy 读写进显存的 BF16 dy；不偏的程序（golden）里
S 不在卷积 kernel 里，两处读的都是显存里同一份 BF16 dy。HLO 里卷积 kernel 内的 S 仍写成 convert(bf16)→convert(f32)，
生成 LLO 时这对 convert 可以被省掉（excess precision），所以 HLO 语义相同、结果不同；这里只看融合切分，省没省要看 LLO。

做法（只读优化后 HLO，CPU 即可）：
  1. 找输出头 dgrad 卷积：fusion 计算里的 convolution，操作数里有词表维（--vocab），输出里没有；
  2. 分主头与 MTP 头：fusion 在外层的直接输入或直接消费者里有来自 mhc.py 的 kernel（hc_head 的输出 / 余切）记主头；
  3. 主头那颗 fusion 里，找输出是「每 token」形状（把特征维 --hidden 规约掉，元素数 = 卷积输出元素数 ÷ hidden）、且依赖卷积结果的 reduce：有就是「S 并进卷积 kernel」。

  python models/hunyuan4/analysis/hc_head_dgrad_fusion.py --root <运行目录根> --out <json> <运行>…
  python models/hunyuan4/analysis/hc_head_dgrad_fusion.py --hlo <after_optimizations.txt> …   # 直接给文件（如 CPU 编的）

--llo <jf 目录>（与 --hlo 一一对应，CPU 上整步 AOT 时 LIBTPU_INIT_ARGS 加 --xla_jf_dump_to=<目录> --xla_jf_dump_llo_text=true
得到）：再数每颗输出头 dgrad 卷积 kernel 的 final_bundles 里各条向量指令的静态条数。HLO 里卷积结果到 S 之间写着
convert(bf16)→convert(f32)，LLO 里有没有对应的打包 / 解包，要和同一融合切分、结果不偏的程序（如关 excess precision）对着看。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hlo_mof_dus_scan as s  # noqa: E402
import llo_bf16_alu as lb  # noqa: E402

DIMS = re.compile(r"\[([\d,]*)\]")
MNEMONIC = re.compile(r"(?:^|[\s;{])(v[a-z][\w.]*)")


def llo_counts(jf: Path, kernel: str) -> dict | None:
    """jf 目录里某颗 kernel 的 final_bundles 里各条向量指令（v 开头的助记符）的静态条数；找不到或不止一份时 None。
    v5p 与 v7x 的助记符写法不同（v5p vpack.i.bf16 / vunpack.c.h.bf16，v7x vpack.c.f32.bf16 / vunpack.c.u.bf16.f32），全数，不挑。"""
    pat = re.compile(r"^[0-9]+-" + re.escape(kernel) + r"-(?:[0-9]+-)?final_bundles\.txt$")
    fs = [f for f in jf.iterdir() if pat.match(f.name)]
    if len(fs) != 1:
        return None
    c: dict[str, int] = {}
    for m in MNEMONIC.finditer(fs[0].read_text(errors="ignore")):
        c[m.group(1)] = c.get(m.group(1), 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def _dims(shape: str) -> list[int]:
    m = DIMS.search(shape or "")
    return [int(x) for x in m.group(1).split(",") if x] if m else []


def analyse(text: str, vocab: int = 120832, hidden: int = 1024) -> dict:
    comps = s.parse_computations(text)
    files = lb.kernel_files(text)
    # fusion 计算 → (外层计算, fusion 指令名)
    caller = {}
    for cname, lines in comps.items():
        for line in lines:
            m = s._INS.match(line)
            c = re.search(r"calls=%([\w.\-]+)", line)
            if m and c and m.group(4) == "fusion":
                caller.setdefault(c.group(1), (cname, m.group(2)))
    heads = []
    for fname, lines in comps.items():
        if fname not in caller:
            continue
        parsed = {}
        for line in lines:
            m = s._INS.match(line)
            if m:
                parsed[m.group(2)] = (m.group(4), s._operands(m.group(5)), m.group(3).split("{")[0])
        convs = [
            n
            for n, (op, ops, ty) in parsed.items()
            if op == "convolution"
            and vocab not in _dims(ty)
            and hidden in _dims(ty)
            and any(vocab in _dims(parsed.get(o, ("", [], ""))[2]) for o in ops)
        ]
        if not convs:
            continue
        conv = convs[0]
        # fusion 内依赖卷积结果的指令（传递闭包）
        dep, changed = {conv}, True
        while changed:
            changed = False
            for n, (_op, ops, _ty) in parsed.items():
                if n not in dep and any(o in dep for o in ops):
                    dep.add(n)
                    changed = True
        n_tok = 1
        for x in _dims(parsed[conv][2]):
            n_tok *= x
        n_tok //= hidden

        def numel(ty):
            n = 1
            for x in _dims(ty):
                n *= x
            return n

        per_token = [
            n
            for n, (op, ops, ty) in parsed.items()
            if op == "reduce" and n in dep and hidden not in _dims(ty) and numel(ty) == n_tok
        ]
        per_feature = [
            n for n, (op, ops, ty) in parsed.items() if op == "reduce" and n in dep and _dims(ty) == [hidden]
        ]
        # 外层：直接输入与直接消费者用到的 kernel 的源文件
        outer, inst = caller[fname]
        o_parsed = {}
        for line in comps[outer]:
            m = s._INS.match(line)
            if m:
                o_parsed[m.group(2)] = (m.group(4), s._operands(m.group(5)))

        def through(n, k=0):
            op, ops = o_parsed.get(n, ("?", []))
            if op in ("get-tuple-element", "bitcast", "copy", "copy-start", "copy-done") and ops and k < 6:
                return through(ops[0], k + 1)
            return n

        ins = [through(o) for o in o_parsed[inst][1]]
        gtes = [n for n, (op, ops) in o_parsed.items() if inst in ops]
        cons = [n for n, (op, ops) in o_parsed.items() if any(g in ops for g in gtes + [inst])]
        nb = {
            "inputs": sorted({f for n in ins for f in files(n)}),
            "consumers": sorted({f for n in cons for f in files(n)}),
        }
        heads.append(
            {
                "fusion": inst,
                "computation": fname,
                "main_head": "mhc.py" in nb["inputs"] or "mhc.py" in nb["consumers"],
                "n_instructions": len(lines),
                "conv_out": parsed[conv][2],
                "per_token_reduce_on_conv": per_token,
                "per_feature_reduce_on_conv": per_feature,
                # 卷积结果到 S 之间有没有先转 BF16 再转回 FP32（HLO 里写着舍一次；LLO 可能省掉）、有没有 reduce-precision
                "conv_bf16_roundtrip": any(
                    op == "convert" and "bf16" in ty and n in dep for n, (op, ops, ty) in parsed.items()
                ),
                "n_reduce_precision": sum(op == "reduce-precision" for op, _, _ in parsed.values()),
                "neighbour_files": nb,
            }
        )
    main = [h for h in heads if h["main_head"]]
    return {
        "dgrad_conv_fusions": heads,
        "main_head_found": len(main),
        "S_in_main_head_conv_kernel": (bool(main[0]["per_token_reduce_on_conv"]) if len(main) == 1 else None),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path)
    p.add_argument("--hlo", nargs="*", default=[])
    p.add_argument("--llo", nargs="*", default=[], help="与 --hlo 一一对应的 jf dump 目录")
    p.add_argument("--vocab", type=int, default=120832)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--out", type=Path)
    p.add_argument("runs", nargs="*")
    a = p.parse_args(argv)
    res = {}
    items = [(r, sorted((a.root / r / "hlo").glob("*after_optimizations.txt"))) for r in a.runs] + [
        (h, [Path(h)]) for h in a.hlo
    ]
    for name, fs in items:
        if len(fs) != 1:
            res[name] = {"error": f"{len(fs)} 份优化后 HLO"}
            continue
        r = analyse(fs[0].read_text(), a.vocab, a.hidden)
        if name in a.hlo and a.hlo.index(name) < len(a.llo):
            jf = Path(a.llo[a.hlo.index(name)])
            for h in r["dgrad_conv_fusions"]:
                h["llo_final_bundles"] = llo_counts(jf, h["fusion"])
        res[name] = r
        print(f"{name}: 主头 dgrad 卷积 kernel 里有 S = {r['S_in_main_head_conv_kernel']}", flush=True)
    if a.out:
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
