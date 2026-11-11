"""TPU 自比：一次 TPU 运行（被测，只改了一样东西）对 TPU golden 比，判三档。

- 档 1「逐位相同」：被测跑过的每一步、两边都记的全部逐步量（步时除外）逐位相同，且第 1 步全量抓数（更新前参数、
  两个动量、更新后主权重、路由偏置）每个张量的 sha256 相同。
- 档 2「在噪声内」：不逐位，但
  - 第 1 步训练梯度：每个张量 rel(被测, golden) ≤ 这个张量上 GPU 三份第 1 步两两 rel_L2 的最大值（尺子；
    不低于分辨率下限 2^-20，见 SCALE_FLOOR）；超出噪声上限的张量按 GPU↔TPU 第 1 步的办法对 CPU FP64 定责：
    被测离 FP64 的距离 ÷ max(golden 离 FP64, GPU 三份离 FP64 的中位数) ≤ 1.2 记「换了顺序」（已定责，不挡），否则记未定责；
    iHC 小参数定不了责的单列（同 01-mini D11 已接受），不挡；
  - 曲线：目录 `vs-golden.yaml` 的每一条曲线检查，三个判定量（带符号均值、MARE、最大单步）÷ GPU 同设置 8 份
    两两 28 对的上限都 ≤ 1。
  尺子 = GPU 自己重跑的波动上限，只当尺子，不是 TPU 的门槛。
- 档 3「超出噪声上限」：其余；曲线最差 r ≤ 2 且第 1 步没有未定责张量记「略超」，否则记「超」。

另报（不进判档）：第 1 步训练梯度两边各自离 CPU FP64 多远、谁更近；优化后 HLO 指纹是否相同（另有只看程序的指纹 program_fingerprint，
规范化同 04-tpu-self/hlo_screen.py）、各类指令个数之差；步时中位数（第 11 步起）、显存峰值；第 1 步梯度按层汇总。

第 1 步训练梯度从全量抓数的动量反推（同 l3_optim_step1.py）：Adam 参数 m / (1 − β1)、Muon 参数 buf / (1 − β)，
得到裁剪后的梯度，再乘 max(1, 第 1 步 raw_grad_norm) 还原成裁剪前（与 GPU 导出、FP64 参照同口径）。
哪个参数走 Adam 按抓数里有没有 exp_avg_sq 认。路由专家：尺子只在 GPU 抓数有的专家（0–3 号）上量，被测对 golden 两种都报。

    python models/hunyuan4/analysis/tpu_self_compare.py --root <运行目录根> --golden ts-g-a --test ts-g-b \\
        --catalog models/hunyuan4/alignments/04-tpu-self/catalogs/vs-golden.yaml \\
        --gpu-step1 g2s-a=<safetensors> g2s-b=<…> g2s-c=<…> --fp64 <step1_ref FP64 目录> \\
        --out models/hunyuan4/alignments/04-tpu-self/results/<被测>-vs-golden
  （--gpu-step1 每个文件旁边要有同名 .json，记路由专家抓了哪几个；--root 下要有 golden、被测与目录里的 GPU 噪声运行）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics as stt
import subprocess
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
B1, BETA = 0.9, 0.95
KEYS = ("signed_mean", "mare", "max_step")
# 第 1 步尺子的分辨率下限：训练梯度从动量反推（exp_avg = FP32 舍入((1−β)·裁剪系数·g)，再除回去），只要裁剪系数变了，
# 每个元素就带进几个 2^-24 量级的相对舍入；所以相对差在 2^-20 以下的不可分辨。GPU 三份逐位相同（尺子为 0）的张量
# （如各层索引器）按这个下限比。下限比 GPU 尺子的中位数（约 2.7e-3）小三个数量级以上
SCALE_FLOOR = 2.0**-20
R64_LIMIT = 1.2   # 定责上限：同 GPU↔TPU 第 1 步（PROTOCOL 第 5 节）的 R ≤ 1.2
# iHC 的小参数（hc_base / hc_fn / hc_scale、hc_head）病态，GPU↔TPU 第 1 步里已按 01-mini 对齐细节 D11 接受；
# 这里超出噪声上限又定不了责的 iHC 张量单列（ihc_accepted），不挡判档
IHC = re.compile(r"\.hc_(attn|mlp)_layer\.|^model\.hc_head\.")
SKIP_FIELDS = {"step", "step_time"}
RENAMES = ((".self_attn.gate_proj.", ".self_attn.linear_gate."), (".attn_hc.", ".hc_attn_layer.hc_pre.hc_"),
           (".ffn_hc.", ".hc_mlp_layer.hc_pre.hc_"), ("model.hc_head.hc_", "model.hc_head.hc_head_"),
           (".self_attn.sinks", ".self_attn.learnable_sink_param"))


def hf_name(n: str) -> str:
    n = n[3:] if n.startswith("hf.") else n
    for a, b in RENAMES:
        n = n.replace(a, b)
    return n


def metrics(run_dir):
    out = {}
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            out[int(r["step"])] = r
    return out


def same_bits(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return np.float64(a).tobytes() == np.float64(b).tobytes() or (math.isnan(a) and math.isnan(b))
    return a == b


def bitwise_metrics(mt, mg):
    steps = sorted(mt)
    missing = [s for s in steps if s not in mg]
    fields = sorted((set().union(*(mt[s].keys() for s in steps)) & set().union(*(mg[s].keys() for s in steps if s in mg)))
                    - SKIP_FIELDS)
    first = {}
    for k in fields:
        for s in steps:
            if s in mg and not same_bits(mt[s].get(k), mg[s].get(k)):
                first[k] = s
                break
    return {"steps": len(steps), "steps_missing_in_golden": missing[:5], "fields": len(fields),
            "fields_differ": len(first), "first_differ_step": min(first.values()) if first else None,
            "first_differ_fields": sorted(first, key=first.get)[:8], "equal": not first and not missing}


def step1_index(run_dir):
    p = os.path.join(run_dir, "step_1", "index.json")
    if not os.path.exists(p):
        return None
    return {e["name"]: e for e in json.load(open(p))["tensors"]}


def bitwise_step1(it, ig):
    if it is None or ig is None:
        return {"available": False}
    names = sorted(set(it) | set(ig))
    differ = [n for n in names if n not in it or n not in ig or it[n].get("sha256") != ig[n].get("sha256")
              or not it[n].get("sha256")]
    by_kind = {}
    for n in differ:
        by_kind[n.split(".", 1)[0]] = by_kind.get(n.split(".", 1)[0], 0) + 1
    return {"available": True, "tensors": len(names), "tensors_differ": len(differ), "differ_by_kind": by_kind,
            "equal": not differ}


class Dump:
    """全量抓数 step_1/ 的读取器：按 (种类, 检查点名) 取张量，读时核 sha256。"""

    def __init__(self, run_dir):
        from safetensors import safe_open
        self._open = safe_open
        self.dir = os.path.join(run_dir, "step_1")
        self.idx = {}
        for e in json.load(open(os.path.join(self.dir, "index.json")))["tensors"]:
            kind, rest = e["name"].split(".", 1)
            self.idx[(kind, hf_name(rest))] = e
        self.h = {}

    def has(self, kind, n):
        return (kind, n) in self.idx

    def get(self, kind, n):
        import torch
        e = self.idx[(kind, n)]
        f = self.h.setdefault(e["file"], self._open(os.path.join(self.dir, e["file"]), "pt"))
        x = f.get_tensor(e["name"])
        raw = x.view(torch.int16) if x.dtype == torch.bfloat16 else x
        assert hashlib.sha256(raw.contiguous().numpy().tobytes()).hexdigest() == e["sha256"], e["name"]
        return x.double().numpy()


def train_grad(d: Dump, n, raw_gn):
    beta = (1 - B1) if d.has("exp_avg_sq", n) else (1 - BETA)
    return d.get("exp_avg", n) / beta * max(1.0, raw_gn)


def rel(a, b, den):
    return float(np.linalg.norm(a - b) / den) if den else (0.0 if np.array_equal(a, b) else math.inf)


def step1_grads(test_dir, golden_dir, gn_t, gn_g, gpu_specs, fp64_dir):
    from safetensors import safe_open
    T, G = Dump(test_dir), Dump(golden_dir)
    gpus = []
    for spec in gpu_specs:
        nm, path = spec.split("=", 1)
        side = os.path.splitext(path)[0] + ".json"
        gpus.append((nm, safe_open(path, "pt"), json.load(open(side))["tensors"] if os.path.exists(side) else {}))
    F = safe_open(os.path.join(fp64_dir, "grads_hf.safetensors"), "pt") if fp64_dir else None
    names = sorted(n for (k, n) in G.idx if k == "exp_avg")
    rows = []
    for n in names:
        gt, gg = train_grad(T, n, gn_t), train_grad(G, n, gn_g)
        row = {"tensor": n, "numel": int(gg.size), "opt": "adam" if G.has("exp_avg_sq", n) else "muon",
               "d_full": rel(gt, gg, float(np.linalg.norm(gg)))}
        ex = gpus[0][2].get(n, {}).get("experts") if gpus else None
        sl = np.array(ex) if ex is not None and gg.ndim == 3 else None
        ts, gs = (gt[sl], gg[sl]) if sl is not None else (gt, gg)
        row["d"] = rel(ts, gs, float(np.linalg.norm(gs)))
        if gpus:
            ga = [g.get_tensor(n).double().numpy() for _, g, _ in gpus]
            row["scale"] = max(rel(x, y, float(np.linalg.norm(y))) for i, x in enumerate(ga) for y in ga[i + 1:])
            row["over_scale"] = row["d"] / max(row["scale"], SCALE_FLOOR)
        if F is not None:
            r = F.get_tensor(n).double().numpy()
            r = r[sl] if sl is not None and r.shape[0] != ts.shape[0] else r
            rn = float(np.linalg.norm(r))
            row["test_vs_fp64"], row["golden_vs_fp64"] = rel(ts, r, rn), rel(gs, r, rn)
            if gpus:
                row["gpu_vs_fp64"] = stt.median(rel(x, r, rn) for x in ga)
        rows.append(row)
    by_layer = {}
    for r in rows:
        m = re.search(r"(?:^model\.)?(layers\.\d+|mtp_layers\.\d+)", r["tensor"])
        k = m.group(1) if m else ("embed" if "embed" in r["tensor"] else "head")
        b = by_layer.setdefault(k, {"n": 0, "n_bitwise": 0, "d": []})
        b["n"] += 1
        b["n_bitwise"] += r["d_full"] == 0
        b["d"].append(r["d"])
    for b in by_layer.values():
        b["d_median"], b["d_max"] = stt.median(b["d"]), max(b["d"])
        del b["d"]
    out = {"n_tensors": len(rows), "n_bitwise": sum(r["d_full"] == 0 for r in rows), "by_layer": by_layer,
           "d_median": stt.median(r["d"] for r in rows), "d_max": max(r["d"] for r in rows),
           "worst": sorted(rows, key=lambda r: -r.get("over_scale", r["d"]))[:5]}
    if gpus:
        out.update({"scale_median": stt.median(r["scale"] for r in rows),
                    "n_over_scale": sum(r["over_scale"] > 1 for r in rows),
                    "over_scale_median": stt.median(r["over_scale"] for r in rows),
                    "over_scale_max": max(r["over_scale"] for r in rows),
                    "within": all(r["over_scale"] <= 1 for r in rows)})
    if gpus and F is not None:
        # 定责（同 GPU↔TPU 第 1 步的判法）：超出噪声上限的张量，被测离 FP64 的距离 ÷ max(golden 离 FP64, GPU 三份离 FP64 的中位数)
        # ≤ 1.2 → 记「换了顺序」（已定责）；否则记「丢了精度」（未定责；iHC 小参数单列，见 IHC）
        for r in rows:
            ref = max(r["golden_vs_fp64"], r["gpu_vs_fp64"])
            r["r64"] = r["test_vs_fp64"] / ref if ref else (1.0 if r["test_vs_fp64"] == 0 else math.inf)
            r["verdict"] = ("scale" if r["over_scale"] <= 1 else "attributed" if r["r64"] <= R64_LIMIT
                            else "ihc_accepted" if IHC.search(r["tensor"]) else "unattributed")
        keys = ("tensor", "d", "scale", "over_scale", "test_vs_fp64", "golden_vs_fp64", "gpu_vs_fp64", "r64")
        un = sorted((r for r in rows if r["verdict"] == "unattributed"), key=lambda r: -r["r64"])
        ih = sorted((r for r in rows if r["verdict"] == "ihc_accepted"), key=lambda r: -r["r64"])
        out.update({"n_attributed": sum(r["verdict"] == "attributed" for r in rows), "n_unattributed": len(un),
                    "unattributed": [{k: r[k] for k in keys} for r in un[:10]],
                    "n_ihc_accepted": len(ih), "ihc_accepted": [{k: r[k] for k in keys} for r in ih[:10]],
                    "within": not un})
    if F is not None:
        moved = [r for r in rows if r["d"] > 0]
        out["fp64"] = {"test_vs_fp64_median": stt.median(r["test_vs_fp64"] for r in rows),
                       "golden_vs_fp64_median": stt.median(r["golden_vs_fp64"] for r in rows),
                       "tensors_changed": len(moved),
                       "test_closer": sum(r["test_vs_fp64"] < r["golden_vs_fp64"] for r in moved),
                       "golden_closer": sum(r["test_vs_fp64"] > r["golden_vs_fp64"] for r in moved),
                       "ratio_median": stt.median(r["test_vs_fp64"] / r["golden_vs_fp64"] for r in moved) if moved else 1.0}
    return out, rows


def curves(catalog, root, test, work):
    """跑 janus 目录（被测换成 test），读回每条曲线检查的三个判定量与 GPU 噪声上限，算 r。"""
    cmd = [sys.executable, "-m", "janus", "run", catalog, "--test", test, "--root", root, "--out", work]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    res = json.load(open(os.path.join(work, "results.json")))
    for src, dst in (("results.json", ".curves.results.json"), ("report.md", ".curves.report.md")):
        shutil.copyfile(os.path.join(work, src), work.removesuffix(".janus") + dst)
    out = []
    for c in res["checks"]:
        if c.get("type") == "determinism":
            continue
        up, m = c["threshold"]["noise"]["upper"], c["metrics"]
        vals = {k: m.get(k, m.get(k + "_max")) for k in KEYS}
        r = {k: (abs(vals[k]) / up[k]) if up[k] else (0.0 if vals[k] == 0 else math.inf) for k in KEYS}
        k = max(r, key=r.get)
        out.append({"id": c["id"], "status": c["status"], "worst_key": k, "value": vals[k], "scale": up[k], "r": r[k],
                    "r_all": r, "first_differ_step": m.get("first_differ_step")})
    return {"janus_rc": p.returncode, "results": os.path.basename(work.removesuffix(".janus")) + ".curves.results.json", "checks": out,
            "within": all(c["r"] <= 1 for c in out), "r_max": max(c["r"] for c in out) if out else None}


def hlo_info(meta):
    h = meta.get("hlo") or {}
    fp = h.get("fingerprint") or {}
    files = {k: v for k, v in (h.get("files") or {}).items() if k.endswith("after_optimizations.txt")}
    one = len(fp) == 1 and len(files) == 1   # 训练步只编译出一个可执行文件才有唯一指纹
    return {"fingerprint": next(iter(fp.values())) if one else None, "sha256": next(iter(files.values())) if one else None,
            "compiled_modules": len(files)}


# 元组形状里有 /*index=5*/ 这样的注释（带等号），要放过，否则 6 元及以上的元组指令（合并后的 all-reduce 等）漏数
HLO_OP = re.compile(r"^\s*(?:ROOT )?%\S+ = (?:\((?:[^=]|/\*index=\d+\*/)*?\)|\S+) ([a-z][a-z0-9-]*)\(", re.M)
HLO_KEEP = ("all-reduce", "all-gather", "reduce-scatter", "all-to-all", "collective-permute", "fusion", "convert", "reduce", "dot",
            "custom-call")


def hlo_program_fingerprint(text: str) -> str:
    """只看程序本身的指纹：用跑前离线筛（04-tpu-self/hlo_screen.py）同一套规范化——去掉 metadata、stack_frame_id 与
    头部源码位置表，名字按出现顺序重新编号，replica_groups 的 mesh 写法展开，每个计算里的 parameter 按参数号排前——再算 sha256。
    不重编号时，同一个程序会因为追踪时中间量个数不同、名字序号不同而记成「不同」（C13、E11 两臂就是这样）。"""
    import sys as _sys
    from pathlib import Path as _P

    d = str(_P(__file__).resolve().parents[1] / "alignments" / "04-tpu-self")
    if d not in _sys.path:
        _sys.path.insert(0, d)
    import hlo_screen

    return hlo_screen.program_fp(hlo_screen.canon_lines(text))


def hlo_text(run_dir):
    d = os.path.join(run_dir, "hlo")
    fs = [f for f in os.listdir(d) if f.endswith("after_optimizations.txt")] if os.path.isdir(d) else []
    return open(os.path.join(d, fs[0])).read() if len(fs) == 1 else None


def hlo_ops(run_dir):
    """训练步优化后 HLO 里各类指令的个数（集合通信、融合、convert、reduce、dot、custom-call；含融合体内部）。"""
    d = os.path.join(run_dir, "hlo")
    fs = [f for f in os.listdir(d) if f.endswith("after_optimizations.txt")] if os.path.isdir(d) else []
    if len(fs) != 1:
        return None
    cnt = {}
    for m in HLO_OP.finditer(open(os.path.join(d, fs[0])).read()):
        op = m.group(1)
        if op.startswith(HLO_KEEP):
            cnt[op] = cnt.get(op, 0) + 1
    return dict(sorted(cnt.items()))


def peak(meta):
    mem = meta.get("memory") or {}
    xs = [v.get("peak_bytes_in_use") for v in mem.values() if isinstance(v, dict) and v.get("peak_bytes_in_use")]
    return max(xs) if xs else None


def step_time(m, lo=11):
    xs = [r["step_time"] for s, r in m.items() if s >= lo and "step_time" in r]
    return stt.median(xs) if xs else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--golden", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--catalog", default=None, help="曲线检查的 janus 目录；不给就不比曲线（只用于不足 300 步的诊断运行，判档时曲线一项记为缺）")
    p.add_argument("--gpu-step1", nargs="*", default=[])
    p.add_argument("--fp64", default=None)
    p.add_argument("--out", required=True, help="输出前缀：写 <前缀>.json，曲线的目录结果写 <前缀>.curves.results.json / .curves.report.md")
    a = p.parse_args(argv)
    td, gd = os.path.join(a.root, a.test), os.path.join(a.root, a.golden)
    mt, mg = metrics(td), metrics(gd)
    meta_t, meta_g = json.load(open(os.path.join(td, "meta.json"))), json.load(open(os.path.join(gd, "meta.json")))
    bm = bitwise_metrics(mt, mg)
    bs = bitwise_step1(step1_index(td), step1_index(gd))
    tier1 = bm["equal"] and bs.get("equal", True)   # 两边都没有第 1 步抓数时只凭逐步量（结果里 step1_dump.available = false）
    out = {"test": a.test, "golden": a.golden,
           "code": {"test": meta_t.get("code"), "golden": meta_g.get("code")},
           # 只列改了数值相关设置的覆盖参数；检查点与输出的路径随站点变（不同集群读同一份检查点的不同副本），不列
           "overrides_only_in_test": sorted(o for o in set(meta_t["config"]["overrides"]) - set(meta_g["config"]["overrides"])
                                            - {f"run_name={a.test}", f"steps={len(mt)}", f"learning_rate_schedule_steps={len(mt)}"}
                                            if not o.startswith(("load_parameters_path=", "base_output_directory="))),
           "bitwise": {"metrics": bm, "step1_dump": bs},
           "hlo": {"test": hlo_info(meta_t), "golden": hlo_info(meta_g)},
           "step_time_median_from_11": {"test": step_time(mt), "golden": step_time(mg)},
           "peak_hbm_bytes_max_local_device": {"test": peak(meta_t), "golden": peak(meta_g)}}
    h = out["hlo"]
    xt, xg = hlo_text(td), hlo_text(gd)
    if xt and xg:
        out["hlo"]["program_fingerprint"] = {"test": hlo_program_fingerprint(xt), "golden": hlo_program_fingerprint(xg)}
        out["hlo"]["program_equal"] = out["hlo"]["program_fingerprint"]["test"] == out["hlo"]["program_fingerprint"]["golden"]
    ot, og = hlo_ops(td), hlo_ops(gd)
    if ot and og:
        out["hlo"]["op_counts_diff"] = {k: {"test": ot.get(k, 0), "golden": og.get(k, 0)}
                                        for k in sorted(set(ot) | set(og)) if ot.get(k, 0) != og.get(k, 0)}
    out["hlo"]["equal"] = (h["test"]["fingerprint"] == h["golden"]["fingerprint"]) if h["test"]["fingerprint"] and h["golden"]["fingerprint"] else None
    if bs["available"]:
        s1, rows = step1_grads(td, gd, mt[1]["raw_grad_norm"], mg[1]["raw_grad_norm"], a.gpu_step1, a.fp64)
        out["step1_grad"] = s1
    else:
        s1, rows = None, []
    # 目录的原始输出放 <前缀>.janus/（不入库）
    out["curves"] = curves(a.catalog, a.root, a.test, a.out + ".janus") if a.catalog else {"within": True, "checks": [], "r_max": None, "skipped": True}
    rmax = out["curves"]["r_max"] or 0.0
    s1_ok = s1 is None or s1.get("within", False)
    if tier1:
        tier, band, why = 1, "逐位相同", "逐位相同"
    elif s1_ok and out["curves"]["within"]:
        tier, band, why = 2, "在噪声内", "不逐位；第 1 步梯度没有未定责的张量，曲线 r ≤ 1"
    else:
        bad = []
        if not s1_ok:
            bad.append(f"第 1 步梯度 {s1['n_unattributed']} 个张量超出噪声上限且离 FP64 远 1.2 倍以上"
                       f"（最远 {s1['unattributed'][0]['tensor']}，{s1['unattributed'][0]['r64']:.3g} 倍）")
        bad += [f"{c['id']} r {c['r']:.3g}" for c in out["curves"]["checks"] if c["r"] > 1]
        band = "超" if (rmax > 2 or not s1_ok) else "略超"
        tier, why = 3, "超出噪声上限：" + "；".join(bad)
    out["verdict"] = {"tier": tier, "label": {1: "逐位相同", 2: "在噪声内", 3: "超出噪声上限"}[tier], "band": band,
                      "curves_r_max": rmax, "why": why}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".json", "w") as f:
        json.dump({**out, "step1_rows": rows}, f, ensure_ascii=False, indent=1)
    print(json.dumps({"test": a.test, "verdict": out["verdict"], "bitwise_metrics": bm["equal"],
                      "first_differ_step": bm["first_differ_step"], "step1_dump_equal": bs.get("equal"),
                      "hlo_equal": out["hlo"]["equal"], "hlo_program_equal": out["hlo"].get("program_equal"), "curves_r_max": out["curves"]["r_max"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
