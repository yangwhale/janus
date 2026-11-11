"""iHC 参数梯度（hc_fn / hc_base / hc_scale，注意力前与 FFN 前两处）对 token 长求和的条件数 κ。

定义同 t1-ihc-post-grad-conditioning.json：参数梯度 = Σ_t c_t（c_t 是第 t 个 token 的贡献），κ = Σ_t |c_t| ÷ |Σ_t c_t|。
那份只量了第 1–3 层 FFN 一侧 post 门控的 dL/dpost_k；这里把同一定义直接用到参数梯度的每个元素上，另报整张量的
κ_tensor = ‖Σ_t |c_t|‖₂ ÷ ‖Σ_t c_t‖₂（R 用的是整张量相对 L2，这一项对应它）。

数据：GPU g2s-a 第 1 步全量抓数（rank 0 第 0 个 microbatch，一条 4096 token 的序列，同那份 κ）。每层在 FP64 里重建
两处 iHC：输入流取上一层 layers.{L-1}.out（第 0 层取 embed.out 复制 4 路），分支输出 attn.out / mlp.out 当常数，
余切取 layers.L.out.grad（流）、layers.L.in.grad（input_layernorm 输出）、layers.L.mlp_norm.out.grad（pre_mlp_layernorm
输出），参数取 main_param（第 1 步更新前）。自检：前向重建对抓数的相对 L2，反向重建的输入流梯度对 layers.{L-1}.out.grad，
FFN 一侧 post 门控 κ 复算对那份 JSON。

python models/hunyuan4/analysis/ihc_param_grad_conditioning.py --run <GPU g2s-a 运行目录>
    [--out models/hunyuan4/alignments/01-mini/results/t1-ihc-param-grad-conditioning.json]
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[3]
N, H = 4, 1024
HC_EPS, MAG, NORM_EPS = 1e-6, 2.0, 1e-5  # hf config.json：hc_eps、hc_magnitude、rms_norm_eps
# 01-mini ROOT-CAUSES u-mini-step1：base2 开那一代（mz-opt2）未定责的 6 个（t1zg-step1-train-grad-vs-fp64.json judge_fp64）
SIX = [
    "model.layers.2.hc_attn_layer.hc_pre.hc_scale",
    "model.layers.3.hc_mlp_layer.hc_pre.hc_base",
    "model.layers.3.hc_attn_layer.hc_pre.hc_scale",
    "model.layers.3.hc_mlp_layer.hc_pre.hc_fn",
    "model.layers.4.hc_attn_layer.hc_pre.hc_base",
    "model.layers.4.hc_attn_layer.hc_pre.hc_scale",
]
# 现行（tz-off-rep）未定责的 5 个（ROOT-CAUSES 同一行）
FIVE = [
    "model.layers.6.hc_mlp_layer.hc_pre.hc_base",
    "model.layers.0.hc_attn_layer.hc_pre.hc_base",
    "model.layers.2.hc_attn_layer.hc_pre.hc_scale",
    "model.layers.3.hc_attn_layer.hc_pre.hc_scale",
    "model.layers.6.hc_mlp_layer.hc_pre.hc_fn",
]


class Run:
    def __init__(self, d):
        self.d = Path(d) / "step_1"
        idx = json.loads((self.d / "index.json").read_text())
        self.file = {t["name"]: t["file"] for t in idx["tensors"]}

    def get(self, name):
        with safe_open(self.d / self.file[name], "pt") as h:
            return h.get_tensor(name).double()


def rel(a, b):
    return float((a - b).norm() / b.norm())


def rms(x, w):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + NORM_EPS) * w


def hc(x, fn, base, scale):
    """Hy4HyperConnection.forward，FP64，把 z（sigmoid 前）与 mixes 留出来。x [s, N·H]。"""
    s = x.shape[0]
    mixes = (x @ fn.T) * torch.rsqrt(x.square().mean(-1, keepdim=True) + NORM_EPS)
    sc = torch.cat([scale[0].expand(N), scale[1].expand(N)])
    z = mixes * sc + base
    z.retain_grad()
    sg = torch.sigmoid(z)
    pre = sg[:, :N] + HC_EPS
    post = MAG * sg[:, N:] + HC_EPS
    out = (pre.unsqueeze(-1) * x.view(s, N, H)).sum(1)
    return post, out, z, mixes, sc


def kappa(c):
    """c [s, ...] 每个 token 的贡献 → 逐元素 κ 与整张量 κ。"""
    tot = c.sum(0)
    ab = c.abs().sum(0)
    el = ab / tot.abs()
    return el, float(ab.norm() / tot.norm()), tot, ab


def fn_kappa(dz, sc, flat_n):
    """hc_fn 的贡献 c_t[j,i] = dz[t,j]·sc[j]·flat_n[t,i] 可分离：Σ_t|c| = |dz·sc|ᵀ|flat_n|，不必落 [s,8,4096]。"""
    a = dz * sc
    tot = a.T @ flat_n
    ab = a.abs().T @ flat_n.abs()
    el = (ab / tot.abs()).flatten()
    return el, float(ab.norm() / tot.norm()), tot


def q(v):
    v = v[torch.isfinite(v)]
    qs = torch.quantile(v, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=v.dtype))
    return dict(zip(["min", "q10", "median", "q90", "max"], [float(x) for x in qs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="GPU g2s-a 运行目录（含 step_1/）")
    ap.add_argument("--layers", type=int, default=7)
    ap.add_argument(
        "--out", default=str(ROOT / "models/hunyuan4/alignments/01-mini/results/t1-ihc-param-grad-conditioning.json")
    )
    a = ap.parse_args()
    torch.set_num_threads(16)
    r = Run(a.run)
    old = json.loads(
        (ROOT / "models/hunyuan4/alignments/01-mini/results/t1-ihc-post-grad-conditioning.json").read_text()
    )
    rows, checks, post_gate = {}, {}, {}
    for L in range(a.layers):
        P = f"main_param.decoder.layers.{L}"
        if L == 0:
            e = r.get("embed.out").squeeze(1)
            x0 = e.unsqueeze(1).expand(-1, N, -1).reshape(-1, N * H)
        else:
            x0 = r.get(f"layers.{L - 1}.out").squeeze(1)
        s = x0.shape[0]
        attn_out = r.get(f"layers.{L}.attn.out").squeeze(1)
        mlp_out = r.get(f"layers.{L}.mlp.out").squeeze(1)
        prm = {}
        for side in ("attn_hc", "ffn_hc"):
            for p in ("fn", "base", "scale"):
                prm[side, p] = r.get(f"{P}.{side}.{p}").requires_grad_(True)
        w1 = r.get(f"{P}.input_layernorm.weight")
        w2 = r.get(f"{P}.pre_mlp_layernorm.weight")
        x = x0.clone().requires_grad_(True)
        post1, agg1, z1, mix1, sc1 = hc(x, prm["attn_hc", "fn"], prm["attn_hc", "base"], prm["attn_hc", "scale"])
        n1 = rms(agg1, w1)
        mid = post1.unsqueeze(-1) * attn_out.unsqueeze(1) + x.view(s, N, H)
        mid.retain_grad()
        post2, agg2, z2, mix2, sc2 = hc(
            mid.reshape(s, N * H), prm["ffn_hc", "fn"], prm["ffn_hc", "base"], prm["ffn_hc", "scale"]
        )
        n2 = rms(agg2, w2)
        out = post2.unsqueeze(-1) * mlp_out.unsqueeze(1) + mid
        g_out = r.get(f"layers.{L}.out.grad").squeeze(1).view(s, N, H)
        g_n1 = r.get(f"layers.{L}.in.grad").squeeze(1)
        g_n2 = r.get(f"layers.{L}.mlp_norm.out.grad").squeeze(1)
        ((out * g_out).sum() + (n1 * g_n1).sum() + (n2 * g_n2).sum()).backward()
        chk = {
            "fwd_input_layernorm_vs_capture": rel(n1.detach(), r.get(f"layers.{L}.in").squeeze(1)),
            "fwd_pre_mlp_layernorm_vs_capture": rel(n2.detach(), r.get(f"layers.{L}.mlp_norm.out").squeeze(1)),
            "fwd_layer_out_vs_capture": rel(out.detach().reshape(s, -1), r.get(f"layers.{L}.out").squeeze(1)),
        }
        if L > 0:
            chk["bwd_input_stream_grad_vs_capture"] = rel(x.grad, r.get(f"layers.{L - 1}.out.grad").squeeze(1))
        checks[L] = chk
        # FFN 一侧 post 门控 dL/dpost_k = Σ_d h·g（旧定义）
        dpost = (mlp_out.unsqueeze(1) * g_out).sum(-1)
        el, _, tot, _ = kappa(dpost)
        post_gate[L] = {"sum": tot.tolist(), "kappa": el.tolist()}
        for side, z, mix, sc, xin, hfname in (
            ("attn_hc", z1, mix1, sc1, x, "hc_attn_layer"),
            ("ffn_hc", z2, mix2, sc2, mid.reshape(s, N * H), "hc_mlp_layer"),
        ):
            dz = z.grad
            xi = xin.detach()
            flat_n = xi * torch.rsqrt(xi.square().mean(-1, keepdim=True) + NORM_EPS)
            m = mix.detach()
            c_scale = torch.stack([(dz[:, :N] * m[:, :N]).sum(1), (dz[:, N:] * m[:, N:]).sum(1)], 1)
            base_el, base_k, base_tot, _ = kappa(dz)
            sc_el, sc_k, sc_tot, _ = kappa(c_scale)
            fn_el, fn_k, fn_tot = fn_kappa(dz, sc.detach(), flat_n)
            # 每个 token 的贡献加起来应等于本行的自动求导梯度（逐 token 拆分写对了）
            sums_ok = {
                "hc_base": rel(base_tot, prm[side, "base"].grad),
                "hc_scale": rel(sc_tot, prm[side, "scale"].grad),
                "hc_fn": rel(fn_tot, prm[side, "fn"].grad),
            }
            pre = f"model.layers.{L}.{hfname}.hc_pre"
            rows[f"{pre}.hc_base"] = {
                "kappa_tensor": base_k,
                "kappa_per_element": base_el.tolist(),
                "elements": "前 4 个 pre、后 4 个 post",
                "row_grad": base_tot.tolist(),
                "split_check": sums_ok["hc_base"],
            }
            rows[f"{pre}.hc_scale"] = {
                "kappa_tensor": sc_k,
                "kappa_per_element": sc_el.tolist(),
                "elements": "[pre_scale, post_scale]",
                "row_grad": sc_tot.tolist(),
                "split_check": sums_ok["hc_scale"],
            }
            fq = q(fn_el)
            rows[f"{pre}.hc_fn"] = {
                "kappa_tensor": fn_k,
                "kappa_per_element_quantiles": fq,
                "kappa_tensor_pre_rows": float(
                    (dz[:, :N] * sc.detach()[:N]).abs().T.matmul(flat_n.abs()).norm() / fn_tot[:N].norm()
                ),
                "kappa_tensor_post_rows": float(
                    (dz[:, N:] * sc.detach()[N:]).abs().T.matmul(flat_n.abs()).norm() / fn_tot[N:].norm()
                ),
                "split_check": sums_ok["hc_fn"],
            }
        print(L, json.dumps(chk))
    repro = {}
    for L, v in old["layers"].items():
        L = int(L)
        if L in post_gate:
            repro[L] = {
                "kappa_old": v["kappa"],
                "kappa_recomputed": post_gate[L]["kappa"],
                "max_rel_diff": max(abs(a_ - b_) / b_ for a_, b_ in zip(post_gate[L]["kappa"], v["kappa"])),
            }

    def pick(names):
        out = {}
        for nm in names:
            rr = rows[nm]
            out[nm] = {
                k: rr[k] for k in rr if k in ("kappa_tensor", "kappa_per_element", "kappa_per_element_quantiles")
            }
            out[nm]["rank_of_42"] = rank[nm]
        return out

    order = sorted(rows, key=lambda nm: -rows[nm]["kappa_tensor"])
    rank = {nm: i + 1 for i, nm in enumerate(order)}
    vs_r = {}
    for tag, fn in (
        ("mz_opt2", "t1zg-step1-train-grad-vs-fp64.json"),
        ("tz_off_rep", "t1zo-step1-train-grad-vs-fp64.json"),
    ):
        d = json.loads((ROOT / "models/hunyuan4/alignments/01-mini/results" / fn).read_text())
        R = {x["tensor"]: x["R_vs_ref"] for x in d["rows"] if x["tensor"] in rows}
        names = sorted(R)
        kr = torch.tensor([rank[nm] for nm in names], dtype=torch.float64)
        rr = torch.tensor([R[nm] for nm in names], dtype=torch.float64).argsort(descending=True).argsort() + 1.0
        rho = float(torch.corrcoef(torch.stack([kr, rr]))[0, 1])
        vs_r[tag] = {"file": fn, "n": len(names), "spearman_kappa_vs_R": rho, "R": R}

    res = {
        "what": "iHC 参数梯度对 token 长求和的条件数 κ（GPU g2s-a 第 1 步 rank 0 第 0 个 microbatch，一条 4096 token 序列）",
        "definition": {
            "kappa_per_element": "参数梯度第 i 个元素 = Σ_t c_t,i；κ_i = Σ_t |c_t,i| ÷ |Σ_t c_t,i|（同 t1-ihc-post-grad-conditioning.json 的定义，用到参数上）",
            "kappa_tensor": "‖Σ_t |c_t|‖₂ ÷ ‖Σ_t c_t‖₂（整张量；R 按整张量相对 L2 算，对应这一项）",
            "c_t": "hc_base：dL/dz_t（z = sigmoid 前的混合 logit）；hc_scale：Σ_k dL/dz_t,k · mixes_t,k（pre、post 各一个）；hc_fn：dL/dz_t,j · scale_j · rmsnorm(x_t)_i",
            "scope": "只有一条序列；整批梯度是 16 条序列（8 rank × 2 microbatch）之和，κ 在整批上会不同（推断：多条序列再相消时更大，同号时接近），没有量",
        },
        "source": {
            "run": Path(a.run).name,
            "index_sha256": hashlib.sha256((Path(a.run) / "step_1" / "index.json").read_bytes()).hexdigest(),
            "constants": {"hc_mult": N, "hidden": H, "hc_eps": HC_EPS, "hc_magnitude": MAG, "rms_norm_eps": NORM_EPS},
            "precision": "FP64 重建；抓数本身是 BF16 激活与梯度、FP32 参数",
        },
        "self_check": {
            "per_layer": checks,
            "ffn_post_gate_kappa_reproduced": repro,
            "split_check_max": max(v["split_check"] for v in rows.values()),
        },
        "six_mz_opt2": pick(SIX),
        "five_tz_off_rep": pick(FIVE),
        "rank_by_kappa_tensor": {"n": len(order), "order": [[nm, rows[nm]["kappa_tensor"]] for nm in order]},
        "kappa_vs_R": vs_r,
        "all_tensors": rows,
        "ffn_post_gate_kappa_all_layers": post_gate,
    }
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    for nm in SIX + FIVE:
        rr = rows[nm]
        print(
            nm, f"κ_tensor={rr['kappa_tensor']:.4g}", rr.get("kappa_per_element", rr.get("kappa_per_element_quantiles"))
        )
    print("repro", json.dumps(repro))
    print("spearman", {k: v["spearman_kappa_vs_R"] for k, v in vs_r.items()})
    print("split_check_max", res["self_check"]["split_check_max"])


if __name__ == "__main__":
    main()
