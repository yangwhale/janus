"""第 0 层（稠密层）离线重算：同一份输入（GPU dump 的 layers.0.in），分别配两组第 0 层权重，FP32 重算各观测点。
混元 3 第 0 层专用（形状、RoPE θ、QKV 按 KV 组交错的 Megatron 布局写死在下面）；两侧 dump 都是 Megatron 规范名，直接读。
用法：
  python l0_weight_swap.py --acts <GPU step 目录（layers.0 + input）> --wa <权重 A 文件> [--wb <权重 B 文件>] [--check]
      [--w <probe metrics.jsonl> --w-step N]   # 投影方向 w = γ ⊙ û（取自参照探针该步）
--check：只用 A，和 GPU dump 的各观测点比 rel_l2（验证实现）。
否则：报 B 对 A 的各观测点 token 均值差在 w 上的相对投影，并逐组参数替换（A 里只换一组成 B 的）看贡献。
"""
import argparse
import json

import torch
from safetensors import safe_open

torch.set_grad_enabled(False)
H, NQ, NKV, D, EPS, THETA = 4096, 64, 8, 128, 1e-5, 11158840.0
P = "decoder.layers.0."
NAMES = ["self_attention.linear_qkv.layer_norm_weight", "self_attention.linear_qkv.weight", "self_attention.q_layernorm.weight",
         "self_attention.k_layernorm.weight", "self_attention.linear_proj.weight", "mlp.linear_fc1.layer_norm_weight",
         "mlp.linear_fc1.weight", "mlp.linear_fc2.weight"]


def load_w(path):
    f = safe_open(path, "pt")
    out = {}
    for k in f.keys():
        for n in NAMES:
            if k.endswith(P + n):
                out[n] = f.get_tensor(k).float()
    assert len(out) == len(NAMES), (path, sorted(out))
    return out


def rms(x, g):
    x64 = x.float()
    return x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + EPS) * g


def rope(x, pos):  # x [T, h, D]，rotate_half
    inv = 1.0 / (THETA ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
    ang = pos.double()[:, None] * inv[None]
    emb = torch.cat([ang, ang], -1)
    cos, sin = emb.cos().float()[:, None], emb.sin().float()[:, None]
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


def forward(W, x, pos):
    T = x.shape[0]
    o = {}
    n1 = rms(x, W["self_attention.linear_qkv.layer_norm_weight"])
    qkv = n1 @ W["self_attention.linear_qkv.weight"].T
    o["attn.qkv.out"] = qkv
    g = qkv.view(T, NKV, (NQ // NKV + 2) * D)
    q = g[..., : NQ // NKV * D].reshape(T, NQ, D)
    k = g[..., NQ // NKV * D: (NQ // NKV + 1) * D].reshape(T, NKV, D)
    v = g[..., (NQ // NKV + 1) * D:].reshape(T, NKV, D)
    q = rms(q, W["self_attention.q_layernorm.weight"])
    k = rms(k, W["self_attention.k_layernorm.weight"])
    o["attn.q_norm.out"], o["attn.k_norm.out"] = q, k
    q, k = rope(q, pos), rope(k, pos)
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    core = torch.empty(T, NQ, D)
    for h in range(NQ):
        kv = h // (NQ // NKV)
        s = (q[:, h] @ k[:, kv].T) / D ** 0.5
        s = s.masked_fill(~mask, float("-inf"))
        core[:, h] = torch.softmax(s, -1) @ v[:, kv]
    core = core.reshape(T, NQ * D)
    o["attn.core.out"] = core
    a = core @ W["self_attention.linear_proj.weight"].T
    o["attn.proj.out"] = a
    h1 = x + a
    n2 = rms(h1, W["mlp.linear_fc1.layer_norm_weight"])
    o["mlp_norm.out"] = n2
    f1 = n2 @ W["mlp.linear_fc1.weight"].T
    o["mlp.fc1.out"] = f1
    gt, up = f1.chunk(2, -1)
    f2 = (torch.nn.functional.silu(gt) * up) @ W["mlp.linear_fc2.weight"].T
    o["mlp.fc2.out"] = f2
    o["out"] = h1 + f2
    return o


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--acts", required=True)
    p.add_argument("--wa", required=True)
    p.add_argument("--wb")
    p.add_argument("--check", action="store_true")
    p.add_argument("--w")
    p.add_argument("--w-step", type=int)
    p.add_argument("--tokens", type=int, default=0)
    p.add_argument("--out")
    a = p.parse_args()
    torch.set_num_threads(64)
    fa = safe_open(f"{a.acts}/layers.0.safetensors", "pt")
    x = fa.get_tensor("layers.0.in")[:, 0].float()
    pos = safe_open(f"{a.acts}/input.safetensors", "pt").get_tensor("input.position_ids")[0]
    if a.tokens:
        x, pos = x[: a.tokens], pos[: a.tokens]
    T = x.shape[0]
    WA = load_w(a.wa)
    oa = forward(WA, x, pos)
    res = {"acts": a.acts, "wa": a.wa, "wb": a.wb, "tokens": T}
    if a.check:
        res["check"] = {}
        for k, v in oa.items():
            g = fa.get_tensor(f"layers.0.{k}")[:T, 0].float().reshape(v.shape)
            res["check"][k] = float((v - g).norm() / g.norm())
            print(f"{k:18s} rel_l2 {res['check'][k]:.3e}")
    if a.wb:
        r = None
        for line in open(a.w):
            j = json.loads(line)
            if j.get("step") == a.w_step and "final_norm_scale" in j:
                r = j
        hb = torch.tensor(r["resid_mean.decoder.decoder_norm.out"], dtype=torch.float64)
        w = torch.tensor(r["final_norm_scale"], dtype=torch.float64) * hb / hb.norm()
        WB = load_w(a.wb)

        def proj(o1, o0, key):
            m1, m0 = o1[key].double().mean(0), o0[key].double().mean(0)
            return float((m1 - m0) @ w / abs(m0 @ w))

        keys = ["attn.proj.out", "mlp.fc2.out", "out"]
        ob = forward(WB, x, pos)
        res["all"] = {k: proj(ob, oa, k) for k in keys}
        # proj.out 与 fc2.out 的贡献按 out 的分母归一，便于相加
        den = abs(oa["out"].double().mean(0) @ w)
        res["all_abs"] = {k: float((ob[k].double().mean(0) - oa[k].double().mean(0)) @ w / den) for k in keys}
        res["w_rel"] = {n: float((WB[n] - WA[n]).norm() / WA[n].norm()) for n in NAMES}
        print("all-swap (rel to out·w):", res["all_abs"])
        res["swap_one"] = {}
        for n in NAMES:
            W1 = dict(WA)
            W1[n] = WB[n]
            o1 = forward(W1, x, pos)
            res["swap_one"][n] = float((o1["out"].double().mean(0) - oa["out"].double().mean(0)) @ w / den)
            print(f"  swap {n:46s} w_rel {res['w_rel'][n]:.3e}  out proj {res['swap_one'][n]:+.3e}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
