"""整个训练步里被省掉的 BF16 舍入：优化后 HLO 里 f32→bf16 的 convert，在同一颗融合里又被 convert 回 f32 使用
（融合内部的来回 convert），按源码位置归类；再拿 LLO（`final_bundles`）核这些融合有没有打包指令。

依据（实测，见 04-tpu-self impacts/05 与 01-mini 知识库「同一段在 v5p 上编出来」）：融合内部的来回 convert 在后端不产生
舍入指令，拿 FP32 值直接用；跨融合边界写回的 BF16 值、喂给 BF16 矩阵乘的值是真舍了的。所以一处 convert(f32→bf16)：
- 它在同一个计算里的使用者里有 convert(bf16→f32) → 这一路使用记「省掉」（elided_uses）；
  转回 f32 后紧跟舍到 BF16 网格的 reduce_precision 的，记「已补」（covered_uses，比如 M58 的流梯度求和），不算省掉；
- 其余使用者（融合输出、BF16 矩阵乘 / 卷积、其它 BF16 运算）记「舍了」（kept_uses）。
LLO 核对：只含「省掉」、不含「舍了」的融合，期望那颗 kernel 一条打包指令（vpack.*.bf16）都没有。

源码位置取 HLO 栈帧表里最内层的 MaxText 文件与行号。栈帧表是截短的，看不全调用链，所以位置先按 op_name 里
`while/body` 的层数分区：2 层 = 外层微批循环里的解码层 scan（iHC 各层）、1 层 = scan 之外（MTP 块、hc_head、嵌入、loss）、
0 层 = 优化器；栈里直接出现 multi_token_prediction.py 的记 MTP。op_name 里有 `transpose(` 记反向。

    python elided_rounding_scan.py --hlo <train_step.after_optimizations.txt> --jf <LLO dump 目录> --out <json> [--positions]

--jf 可省（只做 HLO 匹配，不做 LLO 核对）；--positions 另给逐处清单 `positions`：每一处往返（省掉或已补）所在的
计算、kernel、区域 / 模块、源码位置、前向还是反向、省掉 / 已补 / 舍了各几路使用、那颗 kernel 的打包指令条数、
是不是「只含往返」的 kernel。HLO 没有栈帧表（别的框架、或 dump 时去掉了）时源码位置记「?」，其余照常。
离线编译的做法与读法见案例 17（cases/17-late-drift-elided-rounding.md）与 docs/guide/06-debugging.md「扫全图找被省掉的舍入」。
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

PACK = re.compile(r"\bvpack\.[a-z]+\.bf16\b")
DEF = re.compile(r"^\s*(?:ROOT )?%([\w.\-]+) = (\w+)\[")


def _table(lines, name):
    if name not in lines:
        return {}
    out, i = {}, lines.index(name) + 1
    while i < len(lines) and lines[i] and lines[i][0].isdigit():
        k, _, v = lines[i].partition(" ")
        out[int(k)] = v
        i += 1
    return out


def stack_resolver(lines):
    files = {k: v.strip('"').split("/")[-1] for k, v in _table(lines, "FileNames").items()}
    locs = {
        k: (int(re.search(r"file_name_id=(\d+)", v).group(1)), int(re.search(r"line=(\d+)", v).group(1)))
        for k, v in _table(lines, "FileLocations").items()
    }
    frames = {
        k: (int(re.search(r"file_location_id=(\d+)", v).group(1)), int(re.search(r"parent_frame_id=(\d+)", v).group(1)))
        for k, v in _table(lines, "StackFrames").items()
    }
    memo = {}

    def chain(fid):
        if fid not in memo:
            out, seen, f = [], set(), fid
            while f in frames and f not in seen:
                seen.add(f)
                loc, f = frames[f]
                fn, ln = locs[loc]
                out.append(f"{files[fn]}:{ln}")
            memo[fid] = out
        return memo[fid]

    return chain


MODULE_OF = (
    ("multi_token_prediction.py", "mtp"),
    ("mhc.py", "ihc"),
    ("moe.py", "moe"),
    ("attention_mla.py", "attention"),
    ("attention_op.py", "attention"),
    ("attentions.py", "attention"),
    ("splash_attention", "attention"),
    ("linears.py", "mlp"),
    ("normalizations.py", "norm"),
    ("embeddings.py", "embed"),
    ("indexer", "indexer"),
    ("muon", "optimizer"),
    ("optimizers.py", "optimizer"),
    ("train", "train_loop"),
)


def classify(ch):
    mtp = any(c.startswith("multi_token_prediction.py") for c in ch)
    site = next(
        (c for c in ch if not c.startswith(("api.py", "linear_util", "core.py", "traceback"))), ch[0] if ch else "?"
    )
    mod = "other"
    for c in ch:
        hit = next((m for key, m in MODULE_OF if key in c and m != "mtp"), None)
        if hit:
            mod = hit
            site = c
            break
    return ("mtp/" if mtp else "") + mod, site


_VIEW = re.compile(r"^\s*(?:ROOT )?%([\w.\-]+) = bf16\[[^ ]* (?:bitcast|reshape|copy|transpose|broadcast|slice)\(")


def _uses(name, users, roots, depth=0):
    """一个 BF16 值在同一个计算里的使用，按「省掉 / 舍了 / 已补」计数；穿过 bitcast、reshape 这类只搬数据的 BF16 运算。"""
    el = kp = cov = 0
    for u in users.get(name, ()):
        mb = re.match(r"^\s*(?:ROOT )?%([\w.\-]+) = f32\[[^ ]* convert\(%" + re.escape(name) + r"\)", u)
        mv = _VIEW.match(u)
        if mb:
            # 转回 f32 之后紧跟一个舍到 BF16 网格的 reduce_precision（例如 M58 的流梯度求和）：这一路舍入有人补了
            back = mb.group(1)
            if any(
                re.search(r"reduce-precision\(%" + re.escape(back) + r"\).*exponent_bits=8, mantissa_bits=7", w)
                for w in users.get(back, ())
            ):
                cov += 1
            else:
                el += 1
        elif mv and depth < 8:
            e2, k2, c2 = _uses(mv.group(1), users, roots, depth + 1)
            el, kp, cov = el + e2, kp + k2 + (mv.group(1) in roots), cov + c2
        else:
            kp += 1
    return el, kp, cov


def classify_position(site: str, bwd: bool, shape: str) -> str:
    """把一处往返归到「位置类」（经验规则，按 mini 的形状：序列 4096、路由专家 token 16384；读过各源码行后定的）。
    激活一类的形状里有序列维（4096 或 16384），权重梯度没有。"""
    dims = [int(d) for d in shape.split(",") if d]
    act = 4096 in dims or 16384 in dims
    if site in ("linears.py:203", "?") and bwd:
        return "branch_input_dgrad_sum" if act else "weight_grad"
    if site in ("moe.py:765", "linears.py:478") and bwd:
        return "branch_input_dgrad_sum"  # 路由 / DenseGeneral 输入那次 asarray 的转置：输入梯度
    if site == "?":
        return "branch_input_dgrad_sum" if act else "weight_grad"
    if site == "attention_mla.py:1531" or (site == "linears.py:211" and 8 in dims and 256 in dims):
        return "attn_gate_fwd"
    if site == "linears.py:211":
        return "dense_out_fwd"
    if site == "moe.py:1606":
        return "routed_expert_actup_fwd"
    if site == "moe.py:174":
        return "routed_expert_actup_dgrad" if bwd else "routed_expert_combine_fwd"
    if site.startswith("normalizations.py"):
        return "norm_input_dgrad_sum" if bwd else "norm_output_fwd"
    if site == "attention_mla.py:513":
        return "indexer_mask_exact"
    return "other"


def scan(hlo_path: Path, jf: Path | None, positions: bool = False):
    lines = hlo_path.read_text().splitlines()
    chain = stack_resolver(lines)
    comps, cur, order = collections.defaultdict(list), None, []
    for ln in lines:
        m = re.match(r"^(?:ENTRY )?%([\w.\-]+) ", ln)
        if m and ln.rstrip().endswith("{"):
            cur = m.group(1)
            order.append(cur)
            continue
        if ln.rstrip() == "}":
            cur = None
            continue
        if cur:
            comps[cur].append(ln)
    # 计算 → 调用它的融合（顶层 kernel 名）
    caller = {}
    for c in order:
        for ln in comps[c]:
            m = re.match(r"^\s*(?:ROOT )?%([\w.\-]+) = .*\bfusion\(.*calls=%([\w.\-]+)", ln)
            if m:
                caller[m.group(2)] = (c, m.group(1))

    def kernel_of(comp):
        seen = set()
        k = None
        while comp in caller and comp not in seen:
            seen.add(comp)
            parent, fus = caller[comp]
            k = fus
            comp = parent
        return k

    llo = {}
    if jf:
        for f in jf.glob("*-final_bundles.txt"):
            if "schedule-analysis" in f.name:
                continue
            m = re.match(r"\d+-(.+)-\d+-final_bundles\.txt", f.name)
            if m:
                llo[m.group(1)] = len(PACK.findall(f.read_text(errors="ignore")))
    sites = collections.defaultdict(
        lambda: {
            "elided_uses": 0,
            "kept_uses": 0,
            "covered_uses": 0,
            "converts": 0,
            "kernels": set(),
            "backward": 0,
            "forward": 0,
            "shapes": collections.Counter(),
            "ops": collections.Counter(),
        }
    )
    kern = collections.defaultdict(lambda: {"elided": 0, "kept": 0, "covered": 0, "rp": 0})
    exact = [0]
    classes = collections.defaultdict(collections.Counter)
    pos = []
    for c in order:
        body = comps[c]
        dtype, defs, users = {}, {}, collections.defaultdict(list)
        body_set_root = set()
        for ln in body:
            m = DEF.match(ln)
            if m:
                dtype[m.group(1)] = m.group(2)
                defs[m.group(1)] = ln.split(", metadata=")[0]
                if ln.lstrip().startswith("ROOT"):
                    body_set_root.add(m.group(1))
            # 使用者按所有指令登记，含多输出的 `ROOT %tuple = (bf16[..], f32[..]) tuple(...)`（DEF 不匹配它）
            if re.match(r"^\s*(?:ROOT )?%[\w.\-]+ = ", ln):
                rhs = ln.split(", metadata=")[0].split("=", 1)[-1]
                for o in set(re.findall(r"%([\w.\-]+)", rhs)):
                    users[o].append(ln)
        if any(re.search(r"reduce-precision\(.*exponent_bits=8, mantissa_bits=7", ln) for ln in body):
            kern[kernel_of(c) or c]["rp"] += 1
        for ln in body:
            m = re.match(r"^\s*(?:ROOT )?%([\w.\-]+) = bf16\[([^\]]*)\][^ ]* convert\(%([\w.\-]+)\)", ln)
            if not m or dtype.get(m.group(3)) != "f32":
                continue
            name = m.group(1)
            src = defs.get(m.group(3), "")
            if (
                re.search(r"reduce-precision\(.*exponent_bits=8, mantissa_bits=7", src)
                or re.search(r"= f32\[[^ ]* convert\(%([\w.\-]+)\)", src)
                and dtype.get(re.search(r"convert\(%([\w.\-]+)\)", src).group(1)) == "bf16"
            ):
                exact[0] += (
                    1  # 输入已经在 BF16 网格上（reduce_precision 舍过，或刚从 BF16 转上来）：来回 convert 是精确的
                )
                continue
            is_root = ln.lstrip().startswith("ROOT")
            el, kp, cov = _uses(name, users, body_set_root)
            if is_root:
                kp += 1
            s = re.search(r"stack_frame_id=(\d+)", ln)
            mod, site = classify(chain(int(s.group(1)))) if s else ("other", "?")
            op = re.search(r'op_name="([^"]*)"', ln)
            op = op.group(1) if op else ""
            depth = op.count("while/body")
            if not mod.startswith("mtp/"):
                region = (
                    {2: "decoder", 1: "outside_scan", 0: "optimizer"}.get(depth, f"while{depth}")
                    if op
                    else "no_metadata"
                )
                mod = f"{region}/{mod}"
            bwd = "transpose(" in op
            k = kernel_of(c) or c
            key = f"{mod} | {site}"
            e = sites[key]
            e["converts"] += 1
            e["elided_uses"] += el
            e["kept_uses"] += kp
            e["covered_uses"] += cov
            e["kernels"].add(k)
            if el:
                e.setdefault("el_kernels", set()).add(k)
            e["backward" if bwd else "forward"] += 1
            e["shapes"][m.group(2)] += 1
            if el:
                e["ops"][re.sub(r"\.\d+", "", "/".join(op.split("/")[-3:]))] += el
            kern[k]["elided"] += el
            kern[k]["kept"] += kp
            kern[k]["covered"] += cov
            if positions and (el or cov):
                pos.append(
                    {
                        "convert": name,
                        "computation": c,
                        "kernel": k,
                        "module": mod,
                        "site": site,
                        "direction": "backward" if bwd else "forward",
                        "shape": m.group(2),
                        "elided_uses": el,
                        "covered_uses": cov,
                        "kept_uses": kp,
                        "class": classify_position(site, bwd, m.group(2)) if el else None,
                    }
                )
            if el:
                cls = classify_position(site, bwd, m.group(2))
                region = (
                    "scan"
                    if mod.startswith("decoder/")
                    else "outside_scan"
                    if mod.startswith("outside_scan/")
                    else "other"
                )
                classes[cls][region] += el
    # LLO 核对：只有省掉、没有舍了的 kernel
    # 只含往返的 kernel：没有真舍入的使用，也没有 reduce_precision（它会产生打包指令）
    only_el = [
        k for k, v in kern.items() if v["elided"] and not v["kept"] and not v["covered"] and not v["rp"] and k in llo
    ]
    confirm = {
        "kernels_only_elided": len(only_el),
        "with_zero_packs": sum(llo[k] == 0 for k in only_el),
        "nonzero_examples": sorted(((k, llo[k]) for k in only_el if llo[k]), key=lambda t: -t[1])[:10],
    }
    only = set(only_el)
    for r in pos:
        r["kernel_packs"] = llo.get(r["kernel"])
        r["kernel_only_elided"] = r["kernel"] in only
    rows = []
    for key, e in sites.items():
        if not e["elided_uses"]:
            continue
        mod, site = key.split(" | ", 1)
        rows.append(
            {
                "module": mod,
                "site": site,
                "converts": e["converts"],
                "elided_uses": e["elided_uses"],
                "kept_uses": e["kept_uses"],
                "covered_uses": e["covered_uses"],
                "forward": e["forward"],
                "backward": e["backward"],
                "n_kernels": len(e["kernels"]),
                "shapes": dict(e["shapes"].most_common(3)),
                "ops": dict(e["ops"].most_common(4)),
                "elided_kernels": sorted(e.get("el_kernels", ())),
                "elided_kernel_packs": [llo.get(k) for k in sorted(e.get("el_kernels", ()))],
            }
        )
    rows.sort(key=lambda r: (r["module"], -r["elided_uses"]))
    by_mod = collections.Counter()
    for r in rows:
        by_mod[r["module"]] += r["elided_uses"]
    res = {
        "llo_check": confirm,
        "elided_uses_by_module": dict(by_mod),
        "elided_uses_by_class": {k: dict(v) for k, v in sorted(classes.items())},
        "sites": rows,
        "n_llo_kernels": len(llo),
        "skipped_exact_converts": exact[0],
    }
    if positions:
        res["positions"] = pos
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hlo", required=True, type=Path, help="优化后 HLO（*.after_optimizations.txt）")
    p.add_argument("--jf", type=Path, help="LLO dump 目录（--xla_jf_dump_to 的目录，里面是 *-final_bundles.txt）")
    p.add_argument("--out", type=Path, help="结果 JSON；不给就只打印")
    p.add_argument("--positions", action="store_true", help="JSON 里另给逐处清单 positions")
    a = p.parse_args()
    res = scan(a.hlo, a.jf, positions=a.positions)
    if a.out:
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(res["llo_check"], ensure_ascii=False))
    print(json.dumps(res["elided_uses_by_module"], ensure_ascii=False))
    for r in res["sites"][:80]:
        print(
            r["module"],
            r["site"],
            "el",
            r["elided_uses"],
            "kept",
            r["kept_uses"],
            "cov",
            r["covered_uses"],
            "F",
            r["forward"],
            "B",
            r["backward"],
            r["shapes"],
        )


if __name__ == "__main__":
    main()
