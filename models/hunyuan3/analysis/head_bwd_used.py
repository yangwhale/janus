"""输出头反向：拿 GPU 输出层矩阵乘反向「实际用到的」张量（rr 臂 `head-log-<步>` 抓的 head_used/step_<N>.safetensors），
在 FP64 里把 dh 沿 û（输出层 BF16 权重行均值的单位方向，所有 logits 一起动的方向）的误差拆成两段：

  (a) 矩阵乘：dh_used − dlogits_used @ W_used
  (b) 交叉熵反向：dlogits_used − 参照，参照 = (softmax_FP64(logits_BF16) − onehot) · c_t，c_t 是逐 token 的上游梯度
      （从 dlogits 的目标列反推，见 token_scale）；再拆成「行和」那份（s_t·|w̄|）和去行和后的其余部分，与 head_bwd_ref.py 同口径。
      另外按 Megatron 融合交叉熵的写法在 FP32 里重算一遍（减最大值、exp、除和、目标列减 1、乘 c_t），把 (b) 拆成
      「FP32 计算本身」和「最后舍到 BF16」两段，并报告重算结果与 GPU 实际值逐位相同的比例
  (c) janus dump 的 logits.grad / logits 与实际用到的是否同一份（逐位比）
  (d) janus dump 的 final_norm.out.grad 与 dh_used、dump 的 param.output_layer.weight 与 W_used 是否同一份

所有 Σ_t 都是对第一个 microbatch 的全部 token 求和；「相对」= 该项 / Σ_t |dh_ref·û|。
用法：python head_bwd_used.py <head_used/step_N.safetensors> [<janus dump step 目录>] [--json out.json --tag 名字]
"""

import argparse
import json
import os

import torch
from safetensors import safe_open

torch.set_grad_enabled(False)
torch.set_num_threads(min(32, os.cpu_count() or 1))


_FILES = {}


def load(path, key):
    # 句柄不关：get_tensor 返回的张量可能还映射着文件，关掉再访问会段错误
    if path not in _FILES:
        _FILES[path] = safe_open(path, "pt")
    return _FILES[path].get_tensor(key).clone()


def same(a, b):
    """逐位相同（BF16 / FP32 按位比），以及不同元素的个数与最大绝对差。"""
    if a.shape != b.shape:
        a, b = a.reshape(-1), b.reshape(-1)
        if a.numel() != b.numel():
            return {"bitwise": False, "note": f"shape {tuple(a.shape)} vs {tuple(b.shape)}"}
    if a.dtype != b.dtype:
        return {"bitwise": False, "dtype": [str(a.dtype), str(b.dtype)],
                "max_abs": float((a.double() - b.double()).abs().max())}
    iv = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}.get(a.dtype)
    ne = int((a.view(iv) != b.view(iv)).sum()) if iv else int((a != b).sum())
    return {"bitwise": ne == 0, "dtype": str(a.dtype), "n_diff": ne, "max_abs": float((a.double() - b.double()).abs().max())}


def token_scale(dl, lg, lab):
    """逐 token 的上游梯度 c_t：目标列 dl = (p − 1)·c → c = dl_target / (p_target − 1)（FP64 softmax）。"""
    T = lab.shape[0]
    mx = lg.max(-1, keepdim=True).values                 # 不用 torch.logsumexp：这台机器上对这个尺寸偶发段错误
    lse = (mx + torch.log(torch.exp(lg - mx).sum(-1, keepdim=True)))[:, 0]
    p_t = torch.exp(lg[torch.arange(T), lab] - lse)
    return dl[torch.arange(T), lab] / (p_t - 1.0)


def gpu_gemm(dl_b, W_b, dh_b, dlW, u, denom, lock=None):
    """在 GPU 上按几种写法重做 dh = dlogits @ W，与 FP64 精确值比。dl_b: [T, V] BF16，W_b: [V, D] BF16。
    lock：GPU 锁文件路径（与训练排队共用），只在用卡的那几秒里持有。"""
    import contextlib
    import fcntl

    dev = torch.device("cuda")
    m = torch.backends.cuda.matmul
    saved = (m.allow_tf32, m.allow_bf16_reduced_precision_reduction)
    ys = {}
    with open(lock, "a") if lock else contextlib.nullcontext() as lf:
        if lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            dlg, Wg = dl_b.to(dev)[:, None, :], W_b.to(dev)        # 与 Megatron backward 相同的 [s, b, V] @ [V, D]
            m.allow_bf16_reduced_precision_reduction = True
            ys["bf16_default"] = dlg.matmul(Wg)[:, 0].cpu()
            m.allow_bf16_reduced_precision_reduction = False
            ys["bf16_no_reduced_precision_reduction"] = dlg.matmul(Wg)[:, 0].cpu()
            m.allow_tf32 = True
            ys["fp32_tf32"] = dlg.float().matmul(Wg.float())[:, 0].cpu()
            m.allow_tf32 = False
            ys["fp32_no_tf32"] = dlg.float().matmul(Wg.float())[:, 0].cpu()
            ys["fp64"] = dlg.double().matmul(Wg.double())[:, 0].cpu()
            del dlg, Wg
            torch.cuda.empty_cache()
        finally:
            m.allow_tf32, m.allow_bf16_reduced_precision_reduction = saved
    res = {}
    for name, y in ys.items():
        e = y.double() - dlW
        k = float((y.double() * dlW).sum() / (dlW * dlW).sum()) - 1.0
        res[name] = {"dtype": str(y.dtype), "abs": float((e @ u).sum()), "rel": float((e @ u).sum()) / denom,
                     "scale_fit_minus_1": k, "rel_l2": float(e.norm() / dlW.norm()),
                     "toward_zero": float((e * dlW.sign()).sum() / dlW.abs().sum())}
        if y.dtype == torch.bfloat16:
            res[name]["bitwise_vs_dh_used"] = bool((y.view(torch.int16) == dh_b.view(torch.int16)).all())
        print(f"  gpu {name}: {res[name]}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("head_used")
    ap.add_argument("dump_step", nargs="?")
    ap.add_argument("--json", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--gpu-gemm", action="store_true",
                    help="另在一张 GPU 上用抓到的 dlogits_used、W 重做 dgrad 矩阵乘的几种写法（BF16 默认 / 关 reduced-precision "
                         "reduction / FP32+TF32 / FP32 无 TF32 / FP64），同口径比沿 û 的误差；要占一张卡")
    ap.add_argument("--gpu-lock", default=None, help="--gpu-gemm 用卡时持有的锁文件（与训练排队共用的 flock 文件）")
    a = ap.parse_args()
    H = a.head_used
    load(H, "head.h")
    meta = _FILES[H].metadata()

    def g(k):                                    # [s, b, ...] → 第 0 个序列
        t = load(H, k)
        return t[:, 0] if t.dim() == 3 else t

    lg_b, dl_in, dl_b, dh_b, h_b = g("head.logits"), g("head.dlogits_in"), g("head.dlogits_used"), g("head.dh"), g("head.h")
    W_b = load(H, "head.weight")
    out = {"tag": a.tag, "head_used": H, "dtypes": meta}
    print("dtypes:", meta)
    T, V = lg_b.shape
    out["dlogits_in_vs_used"] = same(dl_in.to(dl_b.dtype), dl_b) | {"in_dtype": str(dl_in.dtype)}

    lab = None
    if a.dump_step:
        D = a.dump_step
        lab = load(f"{D}/input.safetensors", "input.labels")[0]
        mask = load(f"{D}/input.safetensors", "input.loss_mask")[0]
        out["loss_mask_sum"] = float(mask.sum())
        c = {}
        c["logits_dump_vs_used"] = same(load(f"{D}/logits.safetensors", "logits")[:, 0], lg_b)
        try:
            c["logits_grad_dump_vs_used"] = same(load(f"{D}/logits.safetensors", "logits.grad")[:, 0], dl_b)
            c["logits_grad_dump_vs_in"] = same(load(f"{D}/logits.safetensors", "logits.grad")[:, 0], dl_in)
        except Exception as e:  # noqa: BLE001
            c["logits_grad_dump"] = f"missing: {e}"
        try:
            c["final_norm_out_grad_dump_vs_dh_used"] = same(load(f"{D}/final_norm.safetensors", "final_norm.out.grad")[:, 0], dh_b)
            c["final_norm_out_dump_vs_h_used"] = same(load(f"{D}/final_norm.safetensors", "final_norm.out")[:, 0], h_b)
        except Exception as e:  # noqa: BLE001
            c["final_norm_dump"] = f"missing: {e}"
        for name in ("param.other.safetensors", "main_param.other.safetensors"):
            p = f"{D}/{name}"
            if os.path.exists(p):
                k = name.split(".")[0] + ".output_layer.weight"
                Wd = load(p, k)
                c[f"{k}_vs_weight_used"] = same(Wd.to(W_b.dtype), W_b) | {"dump_dtype": str(Wd.dtype)}
        out["dump_vs_used"] = c
        for k, v in c.items():
            print(f"(c/d) {k}: {v}")
    if lab is None:
        raise SystemExit("需要 janus dump 目录拿 labels")

    W = W_b.double()
    wbar = W.mean(0)
    u = wbar / wbar.norm()
    Wu = W @ u                                   # (X @ W) @ û = X @ (W @ û)
    lg = lg_b.double()
    dl = dl_b.double()
    dh = dh_b.double()

    # 逐 token 上游梯度
    ct = token_scale(dl, lg, lab)
    masked = mask == 0
    c_nz = ct[~masked]
    out["token_scale"] = {"mean": float(c_nz.mean()), "min": float(c_nz.min()), "max": float(c_nz.max()),
                          "1/mean": float(1 / c_nz.mean()), "n_masked": int(masked.sum()),
                          "masked_absmax_dlogits": float(dl[masked].abs().max()) if masked.any() else 0.0}
    print("c_t:", out["token_scale"])
    # 参照用常数 c：c_t 是从舍过 BF16 的目标列反推的，带 2^-9 量级的噪声；Megatron 这里的上游梯度是 1/本 rank token 数
    # （2 个 microbatch × 4096 = 8192，2 的幂），所以离 2 的幂不到 1e-3 时取那个 2 的幂（否则 FP32 重算对不上逐位）。被 mask 的 token 参照为 0
    import math
    c0 = float(c_nz.median())
    c_pow2 = 2.0 ** round(math.log2(c0))
    out["token_scale"]["median"] = c0
    if abs(c0 / c_pow2 - 1) < 1e-3:
        c0 = c_pow2
    out["token_scale"]["c_used"] = c0
    cvec = torch.where(masked, torch.zeros_like(ct), torch.full_like(ct, c0))
    p = torch.softmax(lg, -1)
    d_ref = p.clone()
    d_ref[torch.arange(T), lab] -= 1.0
    d_ref *= cvec[:, None]
    del p
    dh_ref = d_ref @ W
    ref_u = dh_ref @ u
    denom = float(ref_u.abs().sum())

    def rel(x):
        return {"abs": float(x), "rel": float(x) / denom}

    r = {"c0": c0, "sum_abs_ref_u": denom, "sum_ref_u": float(ref_u.sum()), "sum_dh_used_u": float((dh @ u).sum())}
    r["dh_used_minus_ref"] = rel(((dh - dh_ref) @ u).sum())
    r["dh_rel_l2"] = float((dh - dh_ref).norm() / dh_ref.norm())
    # (a) 矩阵乘
    dlW = dl @ W
    e_mm = dh - dlW
    r["a_gemm"] = rel((e_mm @ u).sum()) | {"rel_l2": float(e_mm.norm() / dlW.norm())}
    # 矩阵乘误差里「整体缩放」那份：dh ≈ (1 + k)·dlW，k 按最小二乘拟合；减掉之后沿 û 还剩多少
    k = float((dh * dlW).sum() / (dlW * dlW).sum()) - 1.0
    r["a_gemm_scale_fit_minus_1"] = k
    r["a_gemm_after_scale_fit"] = rel(((dh - (1 + k) * dlW) @ u).sum())
    # 误差是否「朝 0」：逐元素 e·sign(dlW) / |dlW| 的加权均值（= Σ e·sign / Σ |dlW|），朝 0 截断时为负
    r["a_gemm_toward_zero"] = float((e_mm * dlW.sign()).sum() / dlW.abs().sum())
    # 只把精确乘积舍到 BF16 会带来多少（矩阵乘输出舍入的期望量级）
    e_round = dlW.bfloat16().double() - dlW
    r["a_gemm_if_exact_then_bf16_round"] = rel((e_round @ u).sum())
    del dlW, e_mm, e_round
    # (b) 交叉熵反向
    ed = dl - d_ref
    s = ed.sum(-1)
    r["b_ce_total"] = rel((ed @ Wu).sum())
    r["b_ce_rowsum"] = rel((s * wbar.norm()).sum())
    edc = ed - ed.mean(-1, keepdim=True)
    r["b_ce_rest"] = rel((edc @ Wu).sum())
    r["rowsum_mean"] = float(s.mean())
    del edc
    # (b') 按融合交叉熵在 FP32 里重算：exp(l − max) / Σ，目标列减 1，乘 c_t；再舍 BF16
    l32 = lg_b.float()
    e32 = torch.exp(l32 - l32.max(-1, keepdim=True).values)
    p32 = e32 / e32.sum(-1, keepdim=True)
    del e32
    p32[torch.arange(T), lab] -= 1.0
    c32 = torch.where(masked, torch.zeros_like(ct), torch.full_like(ct, c0)).float()
    d32 = p32 * c32[:, None]
    del p32
    d32b = d32.to(dl_b.dtype)
    r["b_recompute_bitwise_frac"] = float((d32b.view(torch.int16) == dl_b.view(torch.int16)).double().mean())
    d32d = d32.double()
    r["b_ce_fp32_compute"] = rel(((d32d - d_ref) @ Wu).sum())
    r["b_ce_bf16_cast_of_recompute"] = rel(((d32b.double() - d32d) @ Wu).sum())
    r["b_ce_actual_minus_recompute_bf16"] = rel(((dl - d32b.double()) @ Wu).sum())
    rc = d32b.double() - d32d
    r["b_cast_rowsum"] = rel((rc.sum(-1) * wbar.norm()).sum())
    r["b_cast_rest"] = rel(((rc - rc.mean(-1, keepdim=True)) @ Wu).sum())
    del rc, d32, d32d, d32b
    if a.gpu_gemm:
        r["gpu_gemm"] = gpu_gemm(dl_b, W_b, dh_b, dl @ W, u, denom, lock=a.gpu_lock)
    out["fp64"] = r
    print(json.dumps(r, indent=1))
    if a.json:
        allj = json.load(open(a.json)) if os.path.exists(a.json) else {}
        allj[a.tag or H] = out
        with open(a.json, "w") as f:
            json.dump(allj, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
