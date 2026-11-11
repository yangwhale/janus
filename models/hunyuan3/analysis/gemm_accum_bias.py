"""GPU 矩阵乘的累加偏差：看 BF16 / TF32 张量核 GEMM 的结果是否系统地「朝 0」偏，以及偏差怎么随收缩长度 K 变。

对照：FP32 关 TF32（非张量核 FFMA，就近舍入）与 FP64。三组输入：
  pos      A、B 都是正的均匀随机数（BF16 可表示）：没有抵消，结果全正，朝 0 截断 = 系统偏小
  neg      A 取负：结果全负，朝 0 截断 = 系统偏大（相对误差仍为负）
  head     抓到的输出头 dlogits（[T, V]）与输出层权重（[V, D]），只取前 K 个词表行：dh_K = dl[:, :K] @ W[:K]
指标：相对偏差 bias = Σ e·sign(ref) / Σ |ref|（负 = 朝 0），以及 Σ_t e·û / Σ_t |ref·û|（head 才有）。
用法：python gemm_accum_bias.py [--head <head_used/step_N.safetensors>] [--gpu-lock /data/gpu.lock] [--json out.json]
"""

import argparse
import contextlib
import fcntl
import json

import torch

torch.set_grad_enabled(False)


def metrics(y, ref, u=None):
    e = y.double() - ref
    out = {"bias_toward_zero": float((e * ref.sign()).sum() / ref.abs().sum()),
           "rel_l2": float(e.norm() / ref.norm())}
    if u is not None:
        ru = ref @ u
        out["u_rel"] = float((e @ u).sum() / ru.abs().sum())
    return out


def variants(a, b):
    """a: [M, K]，b: [K, N]，都在 GPU 上、BF16。返回各写法结果（CPU）与 FP64 参照。"""
    m = torch.backends.cuda.matmul
    saved = m.allow_tf32
    try:
        ys = {"bf16": (a @ b).cpu()}
        m.allow_tf32 = True
        ys["fp32_tf32"] = (a.float() @ b.float()).cpu()
        m.allow_tf32 = False
        ys["fp32_no_tf32"] = (a.float() @ b.float()).cpu()
        ref = (a.double() @ b.double()).cpu()
    finally:
        m.allow_tf32 = saved
    return ys, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=None)
    ap.add_argument("--gpu-lock", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(0)
    res = {"synthetic": {}, "head": {}}
    Ks = [256, 1024, 4096, 16384, 65536, 120832]
    with open(a.gpu_lock, "a") if a.gpu_lock else contextlib.nullcontext() as lf:
        if lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
        M = N = 512
        for kind in ("pos", "neg"):
            for K in Ks:
                A = torch.rand(M, K, generator=g).bfloat16()
                B = torch.rand(K, N, generator=g).bfloat16()
                if kind == "neg":
                    A = -A
                ys, ref = variants(A.to(dev), B.to(dev))
                res["synthetic"][f"{kind}_K{K}"] = {k: metrics(y, ref) for k, y in ys.items()}
                print(kind, K, {k: round(v["bias_toward_zero"], 9) for k, v in res["synthetic"][f"{kind}_K{K}"].items()}, flush=True)
        if a.head:
            from safetensors import safe_open
            f = safe_open(a.head, "pt")
            dl = f.get_tensor("head.dlogits_used")[:, 0].clone()
            W = f.get_tensor("head.weight").clone()
            u = W.double().mean(0)
            u = u / u.norm()
            # 词表行打乱一次再截前 K 行，避免词表顺序（高频词在前）混进 K 的效应；全长 K = V 与原顺序结果比较
            perm = torch.randperm(W.shape[0], generator=g)
            for K in Ks:
                idx = perm[:K] if K < W.shape[0] else torch.arange(W.shape[0])
                ys, ref = variants(dl[:, idx].contiguous().to(dev), W[idx].contiguous().to(dev))
                res["head"][f"K{K}"] = {k: metrics(y, ref, u) for k, y in ys.items()}
                print("head", K, {k: (round(v["bias_toward_zero"], 9), round(v["u_rel"], 9)) for k, v in res["head"][f"K{K}"].items()},
                      flush=True)
            torch.cuda.empty_cache()
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
