"""输出头 dgrad（dh = dlogits · W，收缩维 = 词表 120832）的对照消融：rcc 的 `dgrad-chunk` 去掉偏差，靠的是「分段」还是「FP32」。

输入是训练里实际用到的张量：rr 臂 `head-log-<步>` 抓的 head_used/step_<N>.safetensors 里的 head.dlogits_used（BF16）
与 head.weight（BF16，ctx 里存的那份）。所有写法都乘同一对 BF16 值，对同一个 FP64 精确积比：

  A    BF16 张量核一次乘完（生产路径，与 Megatron backward 同写法 [s, 1, V] @ [V, D]，输出 BF16）。顺带核对与 head.dh 逐位相同
  A32  同 A，但输出 FP32（torch.mm(..., out_dtype=float32)），把「输出舍 BF16」从张量核累加里分出来
  E    同 A，但 allow_bf16_reduced_precision_reduction = False（cuBLAS 若做 split-K，段间不许用 BF16 归约）
  B<c> BF16 张量核分段：词表按 c 一段，每段 BF16 输入、FP32 输出的张量核乘法，段间 FP32 累加（假设：只分段就够，还能保住张量核的速度）
  Bh<c> 同 B，但每段输出先舍 BF16 再 FP32 累加（不用 out_dtype 时最顺手的写法，对照用）
  C    升 FP32 一次乘完，TF32 真关（= `dgrad-exact`、`rce` 的写法）
  D<c> 升 FP32 分段，段间 FP32 累加（c = 1024 即 rcc 的 `dgrad-chunk`）
  floor  FP64 精确积直接舍 BF16：任何最后舍回 BF16 的写法的下限

FP32 输出的写法另报「舍回 BF16 之后」（`*->bf16`）：rcc 进主干的是舍回 BF16 的 dh。
指标（e = y − ref，ref = 同一对 BF16 值的 FP64 积，在 GPU 上用 FP64 乘）：
  rel_l2 = ‖e‖ / ‖ref‖；u_rel = Σ_t e_t·ŵ̄ / Σ_t |ref_t·ŵ̄|（ŵ̄ = W 行均值的单位方向，所有 logits 一起升降的方向，
  与 head_bwd_used.py、gemm_accum_bias.py 同口径）；max_abs = max |e|；toward_zero = Σ e·sign(ref) / Σ |ref|。
耗时：CUDA event，每种写法预热后跑 --reps 次取中位数，含 FP32 写法里的升精度（与 rr_arms.py 里每次调用的做法相同）。

TF32 要在进程外关：NVIDIA_TF32_OVERRIDE=0、不设 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE；脚本里再设 torch 开关，并用随机 FP32 乘法
对 FP64 的 rel_l2 做探针（真 FP32 约 1e-7 量级，TF32 约 3e-4），探针不过就退出。

训练臂自检（--check-arm <已提交的结果 JSON>）：用 rr_arms.py `dgrad-chunk-bf16` 的乘法函数在同一份抓数上重算，
要求与本脚本的 B1024 逐位相同，且沿 ŵ̄、rel_l2 与结果 JSON 里记的 B1024 读数一致；不过就非零退出（训练前跑）。

用法：NVIDIA_TF32_OVERRIDE=0 python dgrad_accum_ablation.py --capture step1=<head_used/step_1.safetensors> \\
          --capture step300=<...> --json out.json [--chunks 256,1024,4096,16384] [--reps 10]
"""

import argparse
import json
import os
import statistics
import time

import torch

torch.set_grad_enabled(False)


def probe_fp32(dev):
    """FP32 乘法是否真是 FP32（非 TF32）：K = 4096 的随机乘法对 FP64 的 rel_l2。"""
    g = torch.Generator(device="cpu").manual_seed(0)
    a = torch.randn(1024, 4096, generator=g).to(dev)
    b = torch.randn(4096, 1024, generator=g).to(dev)
    ref = a.double() @ b.double()
    return float(((a @ b).double() - ref).norm() / ref.norm())


def mm_f32out(a, b):
    """BF16 输入、FP32 输出的张量核乘法。"""
    return torch.mm(a, b, out_dtype=torch.float32)


def build_variants(dl, W, chunks):
    """dl: [T, V] BF16，W: [V, D] BF16，都在 GPU 上。返回 {名字: 无参函数}，函数返回结果张量。"""
    V = W.shape[0]
    m = torch.backends.cuda.matmul
    v = {}

    def a_prod():
        return dl[:, None, :].matmul(W)[:, 0]

    def e_norpr():
        m.allow_bf16_reduced_precision_reduction = False
        try:
            return dl[:, None, :].matmul(W)[:, 0]
        finally:
            m.allow_bf16_reduced_precision_reduction = True

    v["A_bf16tc_oneshot"] = a_prod
    v["A32_bf16tc_oneshot_f32out"] = lambda: mm_f32out(dl, W)
    v["E_bf16tc_oneshot_no_rpr"] = e_norpr

    def b_chunk(c, round_partial):
        def f():
            acc = torch.zeros(dl.shape[0], W.shape[1], dtype=torch.float32, device=dl.device)
            for k0 in range(0, V, c):
                if round_partial:
                    acc += dl[:, k0 : k0 + c] @ W[k0 : k0 + c]
                else:
                    acc += mm_f32out(dl[:, k0 : k0 + c], W[k0 : k0 + c])
            return acc

        return f

    def d_chunk(c):
        def f():  # 与 rr_arms.py dgrad-chunk 同写法
            dlf, wf = dl.float(), W.float()
            acc = torch.zeros(dlf.shape[0], wf.shape[1], dtype=torch.float32, device=dl.device)
            for k0 in range(0, V, c):
                acc += dlf[:, k0 : k0 + c] @ wf[k0 : k0 + c]
            return acc

        return f

    for c in chunks:
        v[f"B{c}_bf16tc_chunk"] = b_chunk(c, False)
    for c in chunks:
        v[f"Bh{c}_bf16tc_chunk_bf16partials"] = b_chunk(c, True)
    v["C_fp32_oneshot"] = lambda: dl.float() @ W.float()
    for c in chunks:
        v[f"D{c}_fp32_chunk"] = d_chunk(c)
    return v


def metrics(y, ref, u, denom):
    e = y.double() - ref
    return {
        "dtype": str(y.dtype).replace("torch.", ""),
        "rel_l2": float(e.norm() / ref.norm()),
        "u_rel": float((e @ u).sum()) / denom,
        "u_abs": float((e @ u).sum()),
        "max_abs": float(e.abs().max()),
        "max_abs_over_max_ref": float(e.abs().max() / ref.abs().max()),
        "toward_zero": float((e * ref.sign()).sum() / ref.abs().sum()),
    }


def timeit(f, reps):
    for _ in range(2):
        f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, t = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        f()
        t.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(t))
    return statistics.median(ts)


def run_capture(path, chunks, reps, dev):
    from safetensors import safe_open

    f = safe_open(path, "pt")
    keys = set(f.keys())
    dl = f.get_tensor("head.dlogits_used").clone()
    dl = dl[:, 0] if dl.dim() == 3 else dl
    W = f.get_tensor("head.weight").clone()
    dh_used = f.get_tensor("head.dh").clone() if "head.dh" in keys else None
    dh_used = dh_used[:, 0] if dh_used is not None and dh_used.dim() == 3 else dh_used
    assert dl.dtype == torch.bfloat16 and W.dtype == torch.bfloat16, (dl.dtype, W.dtype)
    T, V = dl.shape
    D = W.shape[1]
    out = {"path": os.path.basename(path), "T": T, "V": V, "D": D, "metadata": f.metadata()}
    dlg, Wg = dl.to(dev), W.to(dev)

    u = Wg.double().mean(0)
    u = u / u.norm()
    ref = dlg.double() @ Wg.double()
    denom = float((ref @ u).abs().sum())
    out["sum_abs_ref_u"] = denom
    out["sum_ref_u"] = float((ref @ u).sum())
    out["ref_norm"] = float(ref.norm())
    # FP64 自检：分两半乘再加，与一次乘完的差（FP64 GEMM 自身的误差量级）
    h = V // 2
    ref2 = dlg[:, :h].double() @ Wg[:h].double() + dlg[:, h:].double() @ Wg[h:].double()
    out["fp64_split_vs_oneshot_rel_l2"] = float((ref2 - ref).norm() / ref.norm())
    del ref2

    res = {"floor_exact_to_bf16": metrics(ref.bfloat16(), ref, u, denom)}
    timing = {}
    for name, fn in build_variants(dlg, Wg, chunks).items():
        y = fn()
        torch.cuda.synchronize()
        y2 = fn()
        rep = bool(torch.equal(y, y2))
        del y2
        r = metrics(y, ref, u, denom) | {"repeatable_bitwise": rep}
        if name.startswith("A_") and dh_used is not None:
            r["bitwise_vs_head_dh"] = bool(torch.equal(y.cpu().view(torch.int16), dh_used.view(torch.int16)))
        res[name] = r
        if y.dtype == torch.float32:
            res[name + "->bf16"] = metrics(y.bfloat16(), ref, u, denom)
        del y
        timing[name] = timeit(fn, reps)
        print(
            f"  {name:36s} u_rel {res[name]['u_rel']:+.3e}  rel_l2 {res[name]['rel_l2']:.3e}  "
            f"max_abs {res[name]['max_abs']:.3e}  {timing[name]:8.2f} ms",
            flush=True,
        )
    # 生产程序实际的 dh（训练里 backward 返回的那份）
    if dh_used is not None:
        res["train_head_dh"] = metrics(dh_used.to(dev), ref, u, denom)
    out["variants"] = res
    out["time_ms_median"] = timing
    out["gemm_tflop"] = 2.0 * T * V * D / 1e12
    del dlg, Wg, ref
    torch.cuda.empty_cache()
    return out


def check_arm(captures, ref_json, dev):
    """rr_arms.chunked_dgrad_bf16 与本脚本 B1024 逐位相同，且读数与已提交结果一致。返回每份抓数的核对结果。"""
    import sys

    from safetensors import safe_open

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frameworks", "megatron"))
    import rr_arms

    want = json.load(open(ref_json))["captures"]
    out, ok = {}, True
    for spec in captures:
        tag, path = spec.split("=", 1)
        f = safe_open(path, "pt")
        dl = f.get_tensor("head.dlogits_used").clone().to(dev)  # [T, 1, V]，与训练里钩子拿到的形状相同
        W = f.get_tensor("head.weight").clone().to(dev)
        y_arm = rr_arms.chunked_dgrad_bf16(dl, W, 1024)[:, 0]
        y_b = build_variants(dl[:, 0], W, [1024])["B1024_bf16tc_chunk"]()
        u = W.double().mean(0)
        u = u / u.norm()
        ref = dl[:, 0].double() @ W.double()
        denom = float((ref @ u).abs().sum())
        got = metrics(y_arm, ref, u, denom)
        w = want[tag]["variants"]["B1024_bf16tc_chunk"]
        r = {
            "bitwise_vs_B1024": bool(torch.equal(y_arm, y_b)),
            "u_rel": got["u_rel"],
            "u_rel_committed": w["u_rel"],
            "rel_l2": got["rel_l2"],
            "rel_l2_committed": w["rel_l2"],
        }
        r["match"] = (
            r["bitwise_vs_B1024"]
            and abs(got["u_rel"] - w["u_rel"]) <= 1e-9 * abs(w["u_rel"]) + 1e-15
            and abs(got["rel_l2"] - w["rel_l2"]) <= 1e-9 * w["rel_l2"]
        )
        ok &= r["match"]
        out[tag] = r
        print(f"  check-arm {tag}: {r}", flush=True)
        del dl, W, y_arm, y_b, ref
        torch.cuda.empty_cache()
    return out, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-arm", default=None, help="已提交的结果 JSON：只做训练臂 dgrad-chunk-bf16 的自检")
    ap.add_argument("--capture", action="append", required=True, help="名字=head_used/step_N.safetensors，可重复")
    ap.add_argument("--chunks", default="256,1024,4096,16384")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    m = torch.backends.cuda.matmul
    m.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    m.allow_bf16_reduced_precision_reduction = True  # torch 默认值，即生产路径
    env = {
        k: os.environ.get(k)
        for k in (
            "NVIDIA_TF32_OVERRIDE",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
            "CUBLAS_WORKSPACE_CONFIG",
            "CUBLAS_EMULATION_STRATEGY",
        )
    }
    probe = probe_fp32(dev)
    x = torch.ones(16, 16, dtype=torch.bfloat16, device=dev)
    assert mm_f32out(x, x).dtype == torch.float32, "torch.mm(out_dtype=float32) 不可用"
    info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(dev),
        "env": env,
        "allow_tf32": m.allow_tf32,
        "fp32_probe_rel_l2_K4096": probe,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(info, flush=True)
    if env["NVIDIA_TF32_OVERRIDE"] != "0" or probe > 1e-5:
        raise SystemExit(f"TF32 没有真关（NVIDIA_TF32_OVERRIDE={env['NVIDIA_TF32_OVERRIDE']}，探针 {probe:.2e}）")
    if a.check_arm:
        res, ok = check_arm(a.capture, a.check_arm, dev)
        with open(a.json, "w") as fh:
            json.dump({"env": info, "check_arm": res, "ok": ok}, fh, ensure_ascii=False, indent=1)
        raise SystemExit(0 if ok else "dgrad-chunk-bf16 与 B1024 对不上")
    chunks = [int(x) for x in a.chunks.split(",") if x]
    out = {"what": __doc__.split("\n")[0], "env": info, "chunks": chunks, "reps": a.reps, "captures": {}}
    for spec in a.capture:
        tag, path = spec.split("=", 1)
        print(f"== {tag}: {path}", flush=True)
        out["captures"][tag] = run_capture(path, chunks, a.reps, dev)
        with open(a.json, "w") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
