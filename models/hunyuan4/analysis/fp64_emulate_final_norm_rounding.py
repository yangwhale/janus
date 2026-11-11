"""最后一层 RMSNorm 舍一次还是舍两次，对主 loss 的影响：CPU FP64 里按 TPU 的实际舍入方式模拟。

输入：fp64_capture_final_norm.py 在 CPU FP64 参照上抓到的末层 RMSNorm 输入（hc_head 输出 outs，每行一个）、
同一份权重的输出头（HF 快照里的 lm_head.weight）、token（取 --step 那一批的这些行，标签 = 下一个 token）。

几种写法（x 先舍成 BF16，同 TPU 上 hc_head 的 BF16 输出；n = x·rsqrt(mean(x²)+eps) 在 FP64 里算；w = BF16 的 scale）：
  exact    h = n·w（FP64，不舍）
  single   h = bf16(n·w)                 rmsnorm_single_rounding（只舍一次）
  double   h = bf16(bf16(n)·w)           默认写法两次舍入都真的发生（libtpu 0.0.52 上最后一层的实际情况）
然后 logits = h·Wᵀ（FP64），对下一个 token 算交叉熵，报每种写法的平均 loss、double − single 的差（绝对值与相对值）。

把 double − single 的差 δh 拆成沿 h 的分量（每个 token 一个缩放系数 a = Σ δh·h ÷ Σ h²）和其余部分，
分别算 loss 的变化：scale_only 用 h_single·(1 + a)，看 loss 的增量是不是几乎全来自 h 被整体缩放。

另报一张「scale 取值 → 两次舍入让 |h| 平均变多少」的表（scale_sweep）：n 取 400 万个标准正态数，单个 BF16 scale w，
比较 Σ|bf16(bf16(n)·w)| 与 Σ|bf16(n·w)|。看两次舍入的偏差和 w 离 1 多远、在哪一侧的关系。

另报「前向噪声」的二阶估计：δz = (h_double − h_single)·Wᵀ，ΔL ≈ ½·E[Σ p δz² − (Σ p δz)²]（交叉熵对 logits 的
Hessian 是 diag(p) − ppᵀ，δz 和标签无关、均值近 0 时，一阶项平均掉，剩下的就是这一项，恒为正）。

    python fp64_emulate_final_norm_rounding.py --cap <cap.pt>:0,1,2,3 <cap2.pt>:4,5,…,15 --hf <HF 快照目录> \\
        --tokens <tokens.npy> --step 301 --out <json>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

F64, BF = torch.float64, torch.bfloat16


def rb(t):
    return t.to(BF).to(F64)


def load_lm_head(hf: str) -> torch.Tensor:
    from safetensors import safe_open

    idx = json.load(open(os.path.join(hf, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(hf, idx["lm_head.weight"]), "pt") as f:
        return f.get_tensor("lm_head.weight").to(F64)


def ce_stats(h, W, labels, chunk=512):
    """逐 token 交叉熵（FP64）与 softmax，按块算省内存；返回每个 token 的 loss 与 (p, logits) 的生成器不存。"""
    out = []
    for i in range(0, h.shape[0], chunk):
        z = h[i : i + chunk] @ W.T
        lse = torch.logsumexp(z, dim=-1)
        out.append(lse - z.gather(1, labels[i : i + chunk, None])[:, 0])
    return torch.cat(out)


def jensen(hd, hs, W, ref_h, chunk=512):
    """½·E[Σ p δz² − (Σ p δz)²]，p 取 ref_h 的 softmax，δz = (hd − hs)·Wᵀ。"""
    out = []
    for i in range(0, hd.shape[0], chunk):
        p = torch.softmax(ref_h[i : i + chunk] @ W.T, dim=-1)
        dz = (hd[i : i + chunk] - hs[i : i + chunk]) @ W.T
        out.append(0.5 * ((p * dz * dz).sum(-1) - (p * dz).sum(-1) ** 2))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap", nargs="+", required=True, help="一个或多个「抓数文件:行号,行号…」（行号同抓数时的 --rows）")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--gbs", type=int, default=16)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    pieces = []
    for spec in a.cap:
        path, rs = spec.rsplit(":", 1)
        rows_i = [int(r) for r in rs.split(",")]
        cap = torch.load(path, weights_only=False)
        # 主干的末层 norm：抓数里 outs 条数等于行数的那一个
        rec = [v for v in cap.values() if len(v.get("outs", [])) == len(rows_i)][0]
        pieces += [(r, rec["outs"][j], rec) for j, r in enumerate(rows_i)]
    rows = [r for r, _, _ in pieces]
    rec = pieces[0][2]
    w = rb(rec["norm_w"].to(F64))
    eps = float(rec["norm_eps"])
    W = load_lm_head(a.hf)
    tok = np.load(a.tokens, mmap_mode="r")
    res = {
        "step": a.step,
        "rows": rows,
        "norm_scale": {"mean": float(w.mean()), "std": float(w.std()), "min": float(w.min()), "max": float(w.max()),
                       "frac_eq_1": float((w == 1).double().mean())},
        "per_row": [],
    }
    acc = {k: [] for k in ("exact", "single", "double", "scale_only", "shrink", "jensen")}
    for r, outs, rec_r in pieces:
        assert torch.equal(rec_r["norm_w"], rec["norm_w"])
        g = (a.step - 1) * a.gbs + r
        t = torch.from_numpy(np.asarray(tok[g, : a.seq + 1], dtype=np.int64))
        labels = t[1:]
        x = rb(outs.to(F64).reshape(-1, w.shape[0]))
        n = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        he, hs, hd = n * w, rb(n * w), rb(rb(n) * w)
        le, ls, ld = (ce_stats(h, W, labels) for h in (he, hs, hd))
        shrink = ((hd - hs) * hs).sum(-1, keepdim=True) / (hs * hs).sum(-1, keepdim=True)
        lsc = ce_stats(hs * (1 + shrink), W, labels)
        jz = jensen(hd, hs, W, hs)
        for k, v in zip(("exact", "single", "double", "scale_only", "shrink", "jensen"), (le, ls, ld, lsc, shrink[:, 0], jz)):
            acc[k].append(v)
        res["per_row"].append(
            {
                "row": r,
                "loss_single": float(ls.mean()),
                "double_minus_single": float((ld - ls).mean()),
                "scale_only_minus_single": float((lsc - ls).mean()),
                "shrink_mean": float(shrink.mean()),
                "jensen": float(jz.mean()),
                "frac_h_changed": float((hd != hs).double().mean()),
            }
        )
        print(json.dumps(res["per_row"][-1]), flush=True)
    cat = {k: torch.cat(v) for k, v in acc.items()}
    d = cat["double"] - cat["single"]
    se = float(d.std() / d.numel() ** 0.5)
    res["summary"] = {
        "n_tokens": int(d.numel()),
        "loss_exact": float(cat["exact"].mean()),
        "loss_single": float(cat["single"].mean()),
        "loss_double": float(cat["double"].mean()),
        "double_minus_single": float(d.mean()),
        "double_minus_single_se": se,
        "double_minus_single_rel": float(d.mean() / cat["single"].mean()),
        "single_minus_exact_rel": float((cat["single"] - cat["exact"]).mean() / cat["exact"].mean()),
        "shrink_mean": float(cat["shrink"].mean()),
        "scale_only_minus_single": float((cat["scale_only"] - cat["single"]).mean()),
        "scale_only_share": float((cat["scale_only"] - cat["single"]).mean() / d.mean()),
        "jensen_estimate": float(cat["jensen"].mean()),
        "jensen_estimate_rel": float(cat["jensen"].mean() / cat["single"].mean()),
    }
    gen = torch.Generator().manual_seed(0)
    nz = torch.randn(4_000_000, dtype=F64, generator=gen)
    sweep = []
    for wv in (0.86, 0.90, 0.95, 0.98, 0.99, 0.996, 1.0, 1.004, 1.008, 1.02, 1.05, 1.10, 1.18):
        wb = rb(torch.tensor(wv, dtype=F64))
        hs1, hd1 = rb(nz * wb), rb(rb(nz) * wb)
        sweep.append({"w": float(wb), "rel_abs_change": float((hd1.abs() - hs1.abs()).sum() / hs1.abs().sum())})
    res["scale_sweep"] = sweep
    print(json.dumps(res["summary"], indent=1))
    print(json.dumps(sweep))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
