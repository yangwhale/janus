"""逐步拆分（案例 01）：在一侧自己的状态上，把「这一步权重为什么这样变」拆成两半，各对一个精确参照。

(a) 更新计算：给定这一步实际的梯度与优化器状态（θ_{k-1}、m_{k-1}、v_{k-1}），按 AdamW 公式在 FP64 里重算 m_k、v_k、θ_k，
    对实际写回的值。梯度来源：GPU 用抓到的 main_grad（裁剪前，可信）；TPU 不加钩子，用一阶动量反推
    g_k = (m_k − β1·m_{k-1}) / (1 − β1)（这是裁剪后的梯度；除以裁剪系数 c_k = min(1, clip / grad_norm_k) 得裁剪前），
    再用二阶动量核对：(v_k − β2·v_{k-1}) / (1 − β2) 应等于 g_k²。优化器状态在步与步之间读出，零扰动。
(b) 梯度计算：同一个 θ_{k-1}、第 k 步的 batch，实际梯度对参照精度（全程 FP32）的梯度。TPU 的参照在同一个进程里另编程序算
    （train_hy3.py --grad-probe-at，gp_f32.*）；GPU 的参照是另一次 FP32 运行（train_hy3.py --fp32 --import-params --skip-steps）
    的第 1 步 grad.*。
方向：「所有 logits 一起动」的方向在参数空间里的样子 d = ∂ mean_{t,v}(logits) / ∂θ（TPU 参照程序算的 gp_lvl.*）。
梯度下降一步 logits 整体水平的一阶变化 ≈ −lr·⟨d, u⟩，u 是这一步的更新方向；所以 ⟨δg, d⟩ 与 ⟨g, d⟩ 同号就是「往下压得更多」。
报两种内积：直接内积 ⟨δg, d⟩，以及按 Adam 逐元素缩放后的 ⟨δg/(√v̂+ε), d⟩（更新量实际的方向）。

用法：
  python step_decomp.py --run <运行目录> --side tpu|gpu --steps 1,25,50,100,200,300 [--ref-run <GPU FP32 参照目录> …]
      [--lvl-run <有 gp_lvl.* 的运行目录>] --out <json>
GPU 的参照按「第 k 步 ↔ 参照运行目录」给：--ref k=<目录>（该目录第 1 步的 grad.* 就是第 k 步 batch 上的参照梯度）。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from janus.store import Run

B1, B2, EPS, LR, WD, CLIP = 0.9, 0.95, 1e-8, 1e-5, 0.033, 1.0


def decays(name: str, shape) -> bool:
    """Megatron 规则：二维、名字不含 norm 的参数才衰减（adamw_step_check.py 第 1 步核过）。"""
    return len(shape) >= 2 and "norm" not in name


def groups(name: str) -> list[str]:
    out = ["all"]
    m = re.search(r"\.layers\.(\d+)\.", name)
    if m:
        out.append(f"layer{m.group(1)}")
    if "output_layer" in name:
        out.append("output")
    if "linear_proj" in name:
        out.append("attn_proj")
    if name.startswith("decoder.layers.0.self_attention.linear_proj"):
        out.append("l0_proj")
    if "router" in name:
        out.append("router")
    return out


def f64(x):
    return np.asarray(x, np.float64)


class Side:
    def __init__(self, path: str):
        self.run = Run.open(path)
        self.metrics = self.run.metrics()

    def has(self, step, name):
        return name in set(self.run.names(step))

    def t(self, step, name):
        return f64(self.run.tensor(step, name))


def clip_coef(gn: float, side: str) -> float:
    if side == "gpu":   # Megatron：clip / (norm + 1e-6)，小于 1 才乘
        c = CLIP / (gn + 1e-6)
    else:               # optax clip_by_global_norm：norm < clip 不动，否则乘 clip / norm
        c = CLIP / gn
    return min(1.0, c)


def acc(d: dict, keys, field: str, val: float):
    for k in keys:
        d.setdefault(k, {}).setdefault(field, 0.0)
        d[k][field] += val


def load_drift(spec: str) -> dict:
    """--drift A@s,B@t：方向 = A 第 s 步更新后的主权重 − B 第 t 步更新后的主权重（例：TPU 与 GPU 第 100 步的第 0 层权重之差，
    即漂移已经累积的方向）。"""
    (ra, sa), (rb, sb) = [(x.rsplit("@", 1)[0], int(x.rsplit("@", 1)[1])) for x in spec.split(",")]
    A, B = Run.open(ra), Run.open(rb)
    out = {}
    for n in A.names(sa):
        if n.startswith("main_param_after.") and n in set(B.names(sb)):
            out[n[len("main_param_after."):]] = f64(A.tensor(sa, n)) - f64(B.tensor(sb, n))
    return out


def analyze_step(S: Side, side: str, k: int, ref: Side | None, ref_step: int, lvl: Side | None, lvl_step: int,
                 drift: dict | None = None):
    names = sorted(n[len("main_param_after."):] for n in S.run.names(k) if n.startswith("main_param_after."))
    gn = float(S.metrics[k]["grad_norm"])
    c = clip_coef(gn, side)
    res = {"step": k, "grad_norm": gn, "clip_coef": c, "tensors": {}, "groups": {}}
    for n in names:
        th0, th1 = S.t(k, f"main_param.{n}"), S.t(k, f"main_param_after.{n}")
        m1, v1 = S.t(k, f"exp_avg.{n}"), S.t(k, f"exp_avg_sq.{n}")
        if k == 1:
            m0 = v0 = np.zeros_like(m1)
        elif S.has(k - 1, f"exp_avg.{n}"):
            m0, v0 = S.t(k - 1, f"exp_avg.{n}"), S.t(k - 1, f"exp_avg_sq.{n}")
        else:
            continue
        g_rec = (m1 - B1 * m0) / (1 - B1)                     # 裁剪后的梯度（由一阶动量反推）
        v_from_g = B2 * v0 + (1 - B2) * g_rec ** 2
        r = {"numel": int(th0.size),
             "v_check_rel_l2": float(np.linalg.norm(v1 - v_from_g) / max(np.linalg.norm(v1 - B2 * v0), 1e-300))}
        g_dump = S.t(k, f"grad.{n}") if S.has(k, f"grad.{n}") else None
        g_probe = S.t(k, f"gp_bf16.{n}") if S.has(k, f"gp_bf16.{n}") else None
        if g_dump is not None:                               # GPU：抓到的裁剪前梯度 × c 应等于反推值
            r["rec_vs_dump_rel_l2"] = float(np.linalg.norm(g_rec - c * g_dump) / np.linalg.norm(c * g_dump))
            g_used = c * g_dump
        else:
            g_used = g_rec
        if g_probe is not None:                              # TPU：独立程序的生产精度梯度 × c 对反推值
            r["rec_vs_probe_rel_l2"] = float(np.linalg.norm(g_rec - c * g_probe) / np.linalg.norm(c * g_probe))
        # (a) 更新计算：用实际梯度在 FP64 里重算这一步
        m_ref = B1 * m0 + (1 - B1) * g_used
        v_ref = B2 * v0 + (1 - B2) * g_used ** 2
        mh, vh = m_ref / (1 - B1 ** k), v_ref / (1 - B2 ** k)
        u = mh / (np.sqrt(vh) + EPS)
        wd = WD if decays(n, th0.shape) else 0.0
        th_ref = th0 - LR * (u + wd * th0)
        dth = th1 - th0
        e_th = th1 - th_ref
        r["upd_rel_err"] = float(np.linalg.norm(e_th) / np.linalg.norm(dth))
        r["upd_err_along_upd"] = float(np.sum(e_th * dth) / np.sum(dth * dth))   # 系统性地多走 / 少走（正 = 多走）
        acc(res["groups"], groups(n), "upd_err_dot_upd", float(np.sum(e_th * dth)))
        acc(res["groups"], groups(n), "upd_sq", float(np.sum(dth * dth)))
        r["m_rel_err"] = float(np.linalg.norm(m1 - m_ref) / np.linalg.norm(m_ref))
        r["v_rel_err"] = float(np.linalg.norm(v1 - v_ref) / np.linalg.norm(v_ref))
        g_raw = g_used / c                                    # 裁剪前，和参照同一个尺度
        pre = 1.0 / (np.sqrt(v1 / (1 - B2 ** k)) + EPS)       # Adam 逐元素缩放（这一步实际的 v̂）
        if lvl is not None and lvl.has(lvl_step, f"gp_lvl.{n}"):
            d = lvl.t(lvl_step, f"gp_lvl.{n}")
            r["d_norm"] = float(np.linalg.norm(d))
            r["upd_err_dot_d"] = float(np.sum(e_th * d))       # 更新误差在 d 上：正 = 水平往上
            r["upd_dot_d"] = float(np.sum(dth * d))
            r["g_dot_d"] = float(np.sum(g_raw * d))
            r["gpre_dot_d"] = float(np.sum(g_raw * pre * d))
            acc(res["groups"], groups(n), "upd_err_dot_d", r["upd_err_dot_d"])
            acc(res["groups"], groups(n), "upd_dot_d", r["upd_dot_d"])
            acc(res["groups"], groups(n), "g_dot_d", r["g_dot_d"])
            acc(res["groups"], groups(n), "gpre_dot_d", r["gpre_dot_d"])
        if drift and n in drift:
            dd = drift[n]
            for f, v in (("upd_err_dot_drift", np.sum(e_th * dd)), ("upd_dot_drift", np.sum(dth * dd))):
                r[f] = float(v)
                acc(res["groups"], groups(n), f, float(v))
        if ref is not None:
            rn = f"gp_f32.{n}" if ref.has(ref_step, f"gp_f32.{n}") else f"grad.{n}"
            if ref.has(ref_step, rn):
                gr = ref.t(ref_step, rn)
                dg = g_raw - gr
                r["grad_rel_l2_vs_ref"] = float(np.linalg.norm(dg) / np.linalg.norm(gr))
                r["grad_cos_vs_ref"] = float(np.sum(g_raw * gr) / (np.linalg.norm(g_raw) * np.linalg.norm(gr)))
                if g_probe is not None:
                    r["probe_rel_l2_vs_ref"] = float(np.linalg.norm(g_probe - gr) / np.linalg.norm(gr))
                if "d_norm" in r:
                    r["dg_dot_d"] = float(np.sum(dg * d))
                    r["ref_dot_d"] = float(np.sum(gr * d))
                    r["dgpre_dot_d"] = float(np.sum(dg * pre * d))
                    r["refpre_dot_d"] = float(np.sum(gr * pre * d))
                    for f in ("dg_dot_d", "ref_dot_d", "dgpre_dot_d", "refpre_dot_d"):
                        acc(res["groups"], groups(n), f, r[f])
                if drift and n in drift:
                    for f, v in (("dg_dot_drift", np.sum(dg * drift[n])), ("ref_dot_drift", np.sum(gr * drift[n])),
                                 ("dgpre_dot_drift", np.sum(dg * pre * drift[n])),
                                 ("refpre_dot_drift", np.sum(gr * pre * drift[n]))):
                        r[f] = float(v)
                        acc(res["groups"], groups(n), f, float(v))
                acc(res["groups"], groups(n), "dg_sq", float(np.sum(dg * dg)))
                acc(res["groups"], groups(n), "ref_sq", float(np.sum(gr * gr)))
        res["tensors"][n] = r
    for gv in res["groups"].values():
        if "ref_dot_d" in gv and gv["ref_dot_d"]:
            gv["dg_dot_d_rel"] = gv["dg_dot_d"] / abs(gv["ref_dot_d"])
            gv["dgpre_dot_d_rel"] = gv["dgpre_dot_d"] / abs(gv["refpre_dot_d"])
        if "upd_dot_d" in gv and gv["upd_dot_d"]:
            gv["upd_err_dot_d_rel"] = gv["upd_err_dot_d"] / abs(gv["upd_dot_d"])
        if gv.get("ref_dot_drift"):
            gv["dg_dot_drift_rel"] = gv["dg_dot_drift"] / abs(gv["ref_dot_drift"])
            gv["dgpre_dot_drift_rel"] = gv["dgpre_dot_drift"] / abs(gv["refpre_dot_drift"])
        if gv.get("upd_dot_drift"):
            gv["upd_err_dot_drift_rel"] = gv["upd_err_dot_drift"] / abs(gv["upd_dot_drift"])
        if gv.get("upd_sq"):
            gv["upd_err_along_upd"] = gv["upd_err_dot_upd"] / gv["upd_sq"]
        if "ref_sq" in gv:
            gv["grad_rel_l2_vs_ref"] = (gv["dg_sq"] / gv["ref_sq"]) ** 0.5
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--side", required=True, choices=["tpu", "gpu"])
    p.add_argument("--steps", required=True)
    p.add_argument("--ref", action="append", default=[],
                   help="GPU：k=<FP32 参照运行目录>（参照目录第 1 步）；TPU 不给时用本运行的 gp_f32.*")
    p.add_argument("--lvl-run", default=None, help="方向 d 取自这个运行的 gp_lvl.*（同一步号）；不给用本运行的")
    p.add_argument("--drift", default=None, help="另一个方向：A@s,B@t = A 第 s 步更新后主权重 − B 第 t 步（只覆盖两边都抓了的参数）")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    S = Side(a.run)
    refs = {int(x.split("=", 1)[0]): Side(x.split("=", 1)[1]) for x in a.ref}
    L = Side(a.lvl_run) if a.lvl_run else S
    drift = load_drift(a.drift) if a.drift else None
    out = {"run": S.run.name, "side": a.side, "lvl_run": L.run.name, "steps": {}}
    for k in [int(x) for x in a.steps.split(",") if x]:
        if k in refs:
            ref, rs = refs[k], 1
        elif a.side == "tpu":
            ref, rs = S, k
        else:
            ref, rs = None, None
        res = analyze_step(S, a.side, k, ref, rs, L, k, drift)
        out["steps"][str(k)] = res
        g = res["groups"]
        print(f"== 第 {k} 步  grad_norm {res['grad_norm']:.4f}  c {res['clip_coef']:.4e}")
        for gname in ("all", "layer0", "l0_proj", "attn_proj", "output", "router"):
            if gname in g:
                v = g[gname]
                print("  " + gname.ljust(9) + "  ".join(f"{f}={v[f]:+.4e}" for f in
                      ("upd_err_along_upd", "upd_err_dot_d_rel", "dg_dot_d_rel", "dgpre_dot_d_rel", "upd_err_dot_drift_rel",
                       "dg_dot_drift_rel", "dgpre_dot_drift_rel", "grad_rel_l2_vs_ref")
                      if f in v))
        worst = {f: max((t.get(f, 0.0) for t in res["tensors"].values()), default=0.0)
                 for f in ("upd_rel_err", "m_rel_err", "v_rel_err", "v_check_rel_l2", "rec_vs_dump_rel_l2", "rec_vs_probe_rel_l2")}
        print("  最大：" + "  ".join(f"{f}={v:.3e}" for f, v in worst.items()))
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
