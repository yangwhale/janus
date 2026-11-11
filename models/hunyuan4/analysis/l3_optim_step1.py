"""L3 第 1 步优化器一步对 CPU FP64：更新后参数、Adam 两个动量、Muon 动量（TPU 现行配置 vs GPU 第 1 步三份 vs FP64）。

FP64 参照：从同一份初始权重（HF 目录，FP32 主权重 = 检查点值）与 CPU FP64 第 1 步梯度（frameworks/megatron/step1_ref.py，
与 GPU / TPU 不同源）出发，在 numpy FP64 里按 Megatron-Core 的公式走一步（同 frameworks/maxtext/optim_replay.py 的 Ref64）：
- 裁剪：g ← g / max(1, ‖g‖_FP64 / 1.0)；
- Adam（标量参数与 iHC / hc_head，GPU optimizer_partition.json 里 opt = adam 的）：m = (1−β1) g，v = (1−β2) g²，
  Δ = −lr · (m / (1−β1)) / (√(v / (1−β2)) + eps)，β1 0.9、β2 0.95、eps 1e-10；
- Muon（其余）：buf = (1−β) g（β 0.95），方向 u = g + β (buf − g)（Nesterov，不做偏差修正），X / max(‖X‖_F, 1e-7)，
  Newton–Schulz 5 步（3.4445, −4.7750, 2.0315），乘 √max(1, 输出维 / 输入维)，× −lr（lr 1e-5）；矩阵的取法同 MaxText 的
  Muon 维度表：输出门每头一个矩阵，稠密层与共享专家的 gate / up 拼成一个（muon_fuse_glu，对上 Megatron 的融合 fc1），
  路由专家每个专家一个，其余整个二维矩阵。
比较（HF 名字；路由专家只取 GPU 抓数里有的 0–3 号）：
- 更新后参数：Δ = 更新后 − 更新前（两边的 Δ 都含主权重加更新时的 FP32 舍入；FP64 的 Δ 不舍）；
- Adam 的 m、v：99 个 Adam 参数；
- Muon 动量：TPU 的 exp_avg（Muon 参数的动量记在同一个名字下）；GPU 抓数没有 Muon 动量，只报 TPU 对 FP64。
判定同 L3（PROTOCOL 第 5 节）：N = GPU 三份两两 rel_L2 的最大值（分母 ‖GPU‖）；TPU 对最近一份 GPU 的 rel_L2 ÷ N > 1
的张量要对 FP64 定责：R = rel_L2(TPU, FP64) ÷ 中位数 rel_L2(GPU_i, FP64) ≤ 1.2；N = 0 且两边离 FP64 都小于一次 BF16
舍入（2^-8）的按分辨率下限过。

  python l3_optim_step1.py --fp64 <step1_ref 目录> --hf <HF 目录> --tpu <TPU 全量抓数运行目录> \
      --gpu g2s-a=<前缀> g2s-b=<前缀> g2s-c=<前缀> --gpu-partition <optimizer_partition.json> --out <json> \
      [--gpu-grads g2s-a=<梯度.safetensors>=<grad_norm> …] [--tpu-gnorm <TPU 第 1 步 raw_grad_norm>] \
      [--tpu-grads <诊断程序梯度，只作旁证>] [--stage <结果名，默认 t1-l3-optim-step1>]
  （GPU 前缀下要有 .s1.main_param / main_param_after / exp_avg / exp_avg_sq .hf.safetensors，export_hf_grads.py --kind 导出）
  给了 --gpu-grads 才算「优化器一步本身」（optim_arith）；TPU 一侧的梯度总是由它自己的动量反推，--tpu-grads 只用来报
  诊断程序梯度与训练步梯度之差。现行 `t1zo-l3-optim-step1` 用 tz-off-rep 的第 1 步全量抓数（没有诊断运行，不给 --tpu-grads）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics as stt
import sys

import numpy as np
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frameworks", "megatron"))

R_LIMIT, RES_FLOOR = 1.2, 2.0**-8
NS = (3.4445, -4.7750, 2.0315)
LR, BETA, B1, B2, EPS = 1e-5, 0.95, 0.9, 0.95, 1e-10
RENAMES = ((".self_attn.gate_proj.", ".self_attn.linear_gate."), (".attn_hc.", ".hc_attn_layer.hc_pre.hc_"),
           (".ffn_hc.", ".hc_mlp_layer.hc_pre.hc_"), ("model.hc_head.hc_", "model.hc_head.hc_head_"),
           (".self_attn.sinks", ".self_attn.learnable_sink_param"))


def hf_name(gpu_dump_name: str) -> str:
    """TPU 抓数里的名字（hf.<transformers 模块名>）→ 检查点名。"""
    n = gpu_dump_name[3:] if gpu_dump_name.startswith("hf.") else gpu_dump_name
    for a, b in RENAMES:
        n = n.replace(a, b)
    return n


def ns64(x):
    t = x.shape[0] > x.shape[1]
    if t:
        x = x.T
    x = x / max(np.linalg.norm(x), 1e-7)
    a, b, c = NS
    for _ in range(5):
        A = x @ x.T
        x = a * x + (b * A + c * A @ A) @ x
    return x.T if t else x


def muon_delta(g_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """HF 布局的一个二维矩阵 [输出, 输入]：返回 (Δ, buf)。"""
    buf = (1 - BETA) * g_rows
    u = g_rows + BETA * (buf - g_rows)
    fo, fi = u.shape
    d = -LR * np.sqrt(max(1.0, fo / fi)) * ns64(u.T).T
    return d, buf


def adam(g):
    m = (1 - B1) * g
    v = (1 - B2) * g * g
    return -LR * (m / (1 - B1)) / (np.sqrt(v / (1 - B2)) + EPS), m, v


def rel(a, b, den):
    return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)) / den) if den else float("nan")


def npy(t):
    return t.double().numpy()


class St:
    def __init__(self, path, prefix=""):
        self.f = safe_open(path, "pt")
        self.prefix = prefix
        self.keys = {k[len(prefix):]: k for k in self.f.keys() if k.startswith(prefix)}

    def get(self, n):
        return npy(self.f.get_tensor(self.keys[n]))


def fp64_opt(get_grad, names, coef, opt_of, n_heads):
    """FP64 优化器一步（Megatron-Core 公式、MaxText 的矩阵取法）：{HF 名: {delta, m, v} 或 {delta, buf}}。"""
    ref, done = {}, set()
    for n in names:
        if n in done:
            continue
        g = get_grad(n) * coef
        if opt_of[n] == "adam":
            d, m, v = adam(g)
            ref[n] = {"delta": d, "m": m, "v": v}
            done.add(n)
            continue
        if re.search(r"\.mlp\.(shared_experts\.)?gate_proj\.weight$", n):  # 融合 GLU：gate | up 一个矩阵
            nu = n.replace("gate_proj", "up_proj")
            d, buf = muon_delta(np.concatenate([g, get_grad(nu) * coef], axis=0))
            h = g.shape[0]
            ref[n] = {"delta": d[:h], "buf": buf[:h]}
            ref[nu] = {"delta": d[h:], "buf": buf[h:]}
            done |= {n, nu}
            continue
        if g.ndim == 3:  # 路由专家：每个专家一个矩阵（gate_up_proj 本身就是融合的 fc1）
            ds, bs = zip(*[muon_delta(g[e]) for e in range(g.shape[0])])
            ref[n] = {"delta": np.stack(ds), "buf": np.stack(bs)}
        elif n.endswith(".self_attn.linear_gate.weight"):  # 输出门：每头一个矩阵
            blk = g.shape[0] // n_heads
            ds, bs = zip(*[muon_delta(g[i * blk:(i + 1) * blk]) for i in range(n_heads)])
            ref[n] = {"delta": np.concatenate(ds), "buf": np.concatenate(bs)}
        else:
            d, buf = muon_delta(g)
            ref[n] = {"delta": d, "buf": buf}
        done.add(n)
    return ref


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fp64", required=True)
    p.add_argument("--hf", required=True)
    p.add_argument("--tpu", required=True)
    p.add_argument("--gpu", nargs="+", required=True)
    p.add_argument("--gpu-partition", required=True)
    p.add_argument("--tpu-grads", default="", help="TPU 同一状态同一批的第 1 步梯度（diag_grads.safetensors，HF 名字）")
    p.add_argument("--tpu-gnorm", type=float, default=0.0, help="TPU 训练第 1 步记下的裁剪前 grad_norm")
    p.add_argument("--gpu-grads", nargs="*", default=[], help="名字=GPU 第 1 步梯度（export_hf_grads.py）=它记下的 grad_norm")
    p.add_argument("--stage", default="t1-l3-optim-step1", help="结果文件的 stage 名")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from hy4_weights import GLOBAL, layer_plan, mtp_names  # noqa: F401  (只为确认对照表可用)

    summ = json.load(open(os.path.join(a.fp64, "summary.json")))
    G64 = safe_open(os.path.join(a.fp64, "grads_hf.safetensors"), "pt")
    names = sorted(G64.keys())
    gnorm = float(np.sqrt(sum(float(np.square(npy(G64.get_tensor(n))).sum()) for n in names)))
    coef = 1.0 / max(1.0, gnorm / 1.0)
    # 初始权重（FP32 主权重 = 检查点值）
    HF = safe_open(os.path.join(a.hf, "model.safetensors"), "pt")
    # 分区：GPU 的 optimizer_partition.json（Megatron 名字）→ HF 名字
    part_m = json.load(open(a.gpu_partition))
    from types import SimpleNamespace

    from export_hf_grads import mcore_to_hf
    cfgj = json.load(open(os.path.join(a.hf, "config.json")))
    L = cfgj["num_hidden_layers"]
    table = mcore_to_hf(SimpleNamespace(num_layers=L, hy4_indexer_types=cfgj["indexer_types"][:L],
                                        moe_layer_freq=[int(t == "sparse") for t in cfgj["mlp_layer_types"][:L]],
                                        mtp_num_layers=int(cfgj.get("num_nextn_predict_layers", 0)),
                                        num_moe_experts=cfgj["n_routed_experts"]))
    opt_of = {}
    for m, ent in part_m.items():
        base = re.sub(r"\.weight\d+$", "", m)
        if base in table:
            h, tf = table[base]
        elif m in table:
            h, tf = table[m]
        else:
            continue
        for x in (h if isinstance(h, tuple) else (h,)):
            opt_of[x] = ent["opt"]
    missing = [n for n in names if n not in opt_of]
    assert not missing, missing[:5]
    # FP64 一步（从 CPU FP64 的梯度出发）
    ref = fp64_opt(lambda n: npy(G64.get_tensor(n)), names, coef, opt_of, int(cfgj["num_attention_heads"]))
    # TPU
    tidx = json.load(open(os.path.join(a.tpu, "step_1", "index.json")))["tensors"]
    tf = {}
    for e in tidx:
        kind, rest = e["name"].split(".", 1)
        tf[(kind, hf_name(rest))] = e
    hdl = {}

    def tget(kind, n):
        e = tf[(kind, n)]
        f = hdl.setdefault(e["file"], safe_open(os.path.join(a.tpu, "step_1", e["file"]), "pt"))
        x = f.get_tensor(e["name"])
        if e.get("sha256"):
            import torch
            raw = x.view(torch.int16) if x.dtype == torch.bfloat16 else x
            assert hashlib.sha256(raw.contiguous().numpy().tobytes()).hexdigest() == e["sha256"], e["name"]
        return npy(x)
    # GPU
    gpus = {}
    for spec in a.gpu:
        nm, pre = spec.split("=")
        gpus[nm] = {k: St(f"{pre}.s1.{k}.hf.safetensors") for k in ("main_param", "main_param_after", "exp_avg", "exp_avg_sq")}
        js = json.load(open(f"{pre}.s1.main_param.hf.json"))
        gpus[nm]["experts"] = {h: v.get("experts") for h, v in js["tensors"].items()}  # 路由专家：抓数里有的全局号
    gn = sorted(gpus)
    rows = []

    def compare(qty, n, t_arr, g_arrs, r_arr, sl):
        if sl is not None:
            t_arr, r_arr = t_arr[sl], r_arr[sl]
        rn = float(np.linalg.norm(r_arr))
        row = {"tensor": n, "quantity": qty, "ref_norm": rn, "tpu_vs_ref": rel(t_arr, r_arr, rn)}
        if g_arrs:
            gv = [rel(x, r_arr, rn) for x in g_arrs]
            noise = max(rel(x, y, float(np.linalg.norm(y))) for i, x in enumerate(g_arrs) for y in g_arrs[i + 1:])
            tvg = min(rel(t_arr, y, float(np.linalg.norm(y))) for y in g_arrs)
            med = stt.median(gv)
            row.update({"gpu_vs_ref": gv, "gpu_vs_ref_median": med, "gpu_noise_max": noise, "tpu_vs_gpu_min": tvg,
                        "over_N": (tvg / noise) if noise else (float("inf") if tvg > 0 else 0.0),
                        "R": row["tpu_vs_ref"] / med if med else None})
            if row["over_N"] <= 1:
                row["verdict"] = "pass_noise"
            elif noise == 0 and row["tpu_vs_ref"] < RES_FLOOR and max(gv) < RES_FLOOR:
                row["verdict"] = "pass_floor"
            elif row["R"] is not None and row["R"] <= R_LIMIT:
                row["verdict"] = "attributed"
            else:
                row["verdict"] = "unattributed_over_2N" if row["over_N"] > 2 else "unattributed_1_2N"
        rows.append(row)

    for n in names:
        p0 = npy(HF.get_tensor(n))
        ex = gpus[gn[0]]["experts"].get(n)
        sl = np.array(ex) if ex is not None and p0.ndim == 3 else None
        t_after = tget("main_param_after", n)
        g_d = []
        for g in gn:
            pa, pb = gpus[g]["main_param_after"].get(n), gpus[g]["main_param"].get(n)
            g_d.append(pa - pb)
        compare("delta", n, t_after - p0, g_d, ref[n]["delta"], sl)
        if opt_of[n] == "adam":
            compare("adam_m", n, tget("exp_avg", n), [gpus[g]["exp_avg"].get(n) for g in gn], ref[n]["m"], sl)
            compare("adam_v", n, tget("exp_avg_sq", n), [gpus[g]["exp_avg_sq"].get(n) for g in gn], ref[n]["v"], sl)
        else:
            compare("muon_momentum", n, tget("exp_avg", n), [], ref[n]["buf"], None)

    # 优化器算术本身：每一侧对「FP64 优化器作用在这一侧自己的梯度上」（裁剪系数用这一侧自己记下的 grad_norm），
    # 把第 1 步梯度带来的误差扣掉，只看优化器一步加的那部分（同 L1 优化器回放的思路，这里用的是训练里真实走的那一步）
    arith = []
    diag_vs_train = None
    if a.gpu_grads:
        H = int(cfgj["num_attention_heads"])
        # TPU 训练这一步实际用的（裁剪后）梯度从它自己的动量反推：Adam m = (1−β1)·g、Muon buf = (1−β)·g（第 1 步从零起）。
        # 不用诊断运行的梯度：诊断（--diag-step1）与训练步是分开编译的两个程序，个别病态小张量的梯度不逐位相同（diag_vs_train 里报）
        g_train = {n: tget("exp_avg", n) / ((1 - B1) if opt_of[n] == "adam" else (1 - BETA)) for n in names}
        tref = fp64_opt(lambda n: g_train[n], names, 1.0, opt_of, H)
        coef_t = 1.0 / max(1.0, a.tpu_gnorm)
        if a.tpu_grads:  # 诊断程序梯度只作旁证；没有诊断运行时不比
            TG = safe_open(a.tpu_grads, "pt")
            diag_vs_train = {n: rel(npy(TG.get_tensor(n)) * coef_t, g_train[n], float(np.linalg.norm(g_train[n]))) for n in names}
        grefs = {}
        for spec in a.gpu_grads:
            nm, path, gnv = spec.split("=")
            GG = safe_open(path, "pt")
            grefs[nm] = fp64_opt(lambda n, GG=GG: npy(GG.get_tensor(n)), names, 1.0 / max(1.0, float(gnv)), opt_of, H)
        for n in names:
            p0 = npy(HF.get_tensor(n))
            ex = gpus[gn[0]]["experts"].get(n)
            sl = np.array(ex) if ex is not None and p0.ndim == 3 else None
            qs = [("delta", tget("main_param_after", n) - p0, {g: gpus[g]["main_param_after"].get(n) - gpus[g]["main_param"].get(n) for g in gn}, "delta")]
            if opt_of[n] == "adam":
                qs += [("adam_m", tget("exp_avg", n), {g: gpus[g]["exp_avg"].get(n) for g in gn}, "m"),
                       ("adam_v", tget("exp_avg_sq", n), {g: gpus[g]["exp_avg_sq"].get(n) for g in gn}, "v")]
            else:
                qs += [("muon_momentum", tget("exp_avg", n), None, "buf")]
            for q, tv, gv, key in qs:
                tr = tref[n][key]
                tt = tv[sl] if sl is not None else tv
                trr = tr[sl] if sl is not None else tr
                et = rel(tt, trr, float(np.linalg.norm(trr)))
                row = {"tensor": n, "quantity": q, "tpu_vs_own_ref": et}
                if gv is not None:
                    eg = [rel(gv[g], grefs[g][n][key], float(np.linalg.norm(grefs[g][n][key]))) for g in gn]
                    med = stt.median(eg)
                    row.update({"gpu_vs_own_ref": eg, "R": et / med if med else (0.0 if et == 0 else float("inf"))})
                    row["verdict"] = ("pass_floor" if et < RES_FLOOR and max(eg) < RES_FLOOR else
                                      ("pass" if row["R"] <= R_LIMIT else "fail"))
                arith.append(row)

    def summary(qty):
        rs = [r for r in rows if r["quantity"] == qty]
        out = {"n": len(rs), "tpu_vs_ref_median": stt.median(r["tpu_vs_ref"] for r in rs)}
        if rs and "R" in rs[0]:
            out.update({"gpu_vs_ref_median": stt.median(r["gpu_vs_ref_median"] for r in rs),
                        "R_median": stt.median(r["R"] for r in rs if r["R"] is not None),
                        "R_max": max(r["R"] for r in rs if r["R"] is not None),
                        "n_over_N": sum(r["over_N"] > 1 for r in rs),
                        "unattributed": [r["tensor"] for r in rs if r["verdict"].startswith("unattributed")]})
            w = max(rs, key=lambda r: r["R"] or 0)
            out["worst"] = {k: w[k] for k in ("tensor", "R", "gpu_vs_ref_median", "tpu_vs_ref", "over_N")}
        else:
            w = max(rs, key=lambda r: r["tpu_vs_ref"])
            out["worst"] = {k: w[k] for k in ("tensor", "tpu_vs_ref")}
        return out
    summ_out = {q: summary(q) for q in ("delta", "adam_m", "adam_v", "muon_momentum")}
    checks = []
    for q, zh in (("delta", "更新后参数（Δ）"), ("adam_m", "Adam 一阶动量"), ("adam_v", "Adam 二阶动量")):
        s = summ_out[q]
        ok = not s["unattributed"]
        checks.append({"id": f"L3.step1.optim.{q}", "type": "optimizer_step", "level": "L3", "status": "pass" if ok else "fail",
                       "threshold": {"kind": "numeric", "values": {"r_ratio": R_LIMIT},
                                     "source": "PROTOCOL.md 第 5 节：超出 GPU 第 1 步噪声的张量对 CPU FP64 定责，R ≤ 1.2；N = 0 且两边都小于一次 BF16 舍入按分辨率下限过"},
                       "metrics": {"r_ratio": s["R_max"], "rel_l2_gpu_vs_ref": s["worst"]["gpu_vs_ref_median"],
                                   "rel_l2_tpu_vs_ref": s["worst"]["tpu_vs_ref"], "tensors": s["n"], "over_N": s["n_over_N"],
                                   "unattributed": len(s["unattributed"]), "R_median": s["R_median"]},
                       "message": f"{zh} {s['n']} 个张量：超出 GPU 噪声 {s['n_over_N']} 个，未定责 {len(s['unattributed'])} 个"
                                  f"{('：' + '、'.join(s['unattributed'][:6])) if s['unattributed'] else ''}；R 中位数 {s['R_median']:.3f}，"
                                  f"最大 {s['worst']['tensor']} GPU {s['worst']['gpu_vs_ref_median']:.3g}、TPU {s['worst']['tpu_vs_ref']:.3g} → R {s['R_max']:.3f}"})
    s = summ_out["muon_momentum"]
    checks.append({"id": "L3.step1.optim.muon_momentum", "type": "optimizer_step", "level": "L3", "status": "inconclusive",
                   "threshold": {}, "metrics": {"tensors": s["n"], "tpu_vs_ref_median": s["tpu_vs_ref_median"], "worst": s["worst"]},
                   "message": f"只报（GPU 抓数没有 Muon 动量）：TPU Muon 动量 {s['n']} 个张量对 FP64 的 rel_L2 中位数 {s['tpu_vs_ref_median']:.3g}，"
                              f"最大 {s['worst']['tensor']} {s['worst']['tpu_vs_ref']:.3g}（第 1 步动量 = (1−β)·裁剪后梯度，误差就是梯度的误差）"})
    if arith:
        for q, zh in (("delta", "更新后参数（Δ）"), ("adam_m", "Adam 一阶动量"), ("adam_v", "Adam 二阶动量")):
            rs = [r for r in arith if r["quantity"] == q]
            bad = [r for r in rs if r["verdict"] == "fail"]
            w = max(rs, key=lambda r: r["R"])
            checks.append({"id": f"L3.step1.optim_arith.{q}", "type": "optimizer_step", "level": "L3",
                           "status": "pass" if not bad else "fail",
                           "threshold": {"kind": "numeric", "values": {"r_ratio": R_LIMIT},
                                         "source": "PROTOCOL.md 第 5 节的 R ≤ 1.2；两边都小于一次 BF16 舍入按分辨率下限过"},
                           "metrics": {"r_ratio": w["R"], "rel_l2_gpu_vs_ref": stt.median(w["gpu_vs_own_ref"]),
                                       "rel_l2_tpu_vs_ref": w["tpu_vs_own_ref"], "tensors": len(rs), "fail": len(bad),
                                       "below_floor": sum(r["verdict"] == "pass_floor" for r in rs),
                                       "R_median": stt.median(r["R"] for r in rs)},
                           "message": f"{zh}，优化器一步本身（各侧对 FP64 优化器作用在自己的梯度上）{len(rs)} 个张量：不过 {len(bad)}，"
                                      f"低于分辨率下限 {sum(r['verdict'] == 'pass_floor' for r in rs)}；最大 R {w['tensor']} "
                                      f"GPU {stt.median(w['gpu_vs_own_ref']):.3g}、TPU {w['tpu_vs_own_ref']:.3g} → R {w['R']:.3f}"})
        rs = [r for r in arith if r["quantity"] == "muon_momentum"]
        w = max(rs, key=lambda r: r["tpu_vs_own_ref"])
        checks[-4]["message"] += f"；对 FP64 作用在 TPU 自己梯度上的动量：最大 {w['tensor']} {w['tpu_vs_own_ref']:.3g}（中位数 {stt.median(r['tpu_vs_own_ref'] for r in rs):.3g}）"
        checks[-4]["metrics"]["arith_tpu_vs_own_ref_max"] = w["tpu_vs_own_ref"]
    res = {"schema_version": 1, "stage": a.stage, "model": "hunyuan4-mini",
           "inputs": {"tpu": os.path.basename(a.tpu.rstrip("/")), "gpu": gn, "fp64": "CPU FP64 第 1 步梯度（step1_ref.py）",
                      "fp64_grad_norm": gnorm, "fp64_grad_norm_summary": summ.get("grad_norm"), "clip_coef": coef,
                      "hyper": {"lr": LR, "muon_beta": BETA, "adam_b1": B1, "adam_b2": B2, "adam_eps": EPS, "ns": NS}},
           "summary_by_quantity": summ_out, "checks": checks, "rows": rows, "arith_rows": arith,
           "tpu_diag_vs_train_grad": diag_vs_train}
    res["summary"] = {k: sum(c["status"] == k for c in checks) for k in ("pass", "fail", "inconclusive", "error")}
    with open(a.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    for c in checks:
        print(c["status"], c["id"], c["message"])


if __name__ == "__main__":
    main()
