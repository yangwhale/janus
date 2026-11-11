"""混元 3（截 4 层）的 FP64 前向参照：从 BF16 权重与给定的层输入出发，全程 FP64、中间不舍入，算出与抓数同名同布局的观测点。

用途：前向激活没法用 GPU 自己重跑当尺子（GPU 默认设置下第 1 步前向逐位可复现，噪声上限是 0，docs/reference/regression.md 第五节）。
这里给每个观测点一个 FP64 真值，比「被测离 FP64 有多远」与「GPU 离 FP64 有多远」（R-ratio，docs/reference/checks/op_replay.md），
被测不比 GPU 远就算对齐。真值是「同一份 BF16 权重、同一份输入，在精确算术里」的结果；路由按 FP64 分数重新选专家。

模型语义按 HF 配置（tencent/Hy3 截 4 层，config.json）与 name_map.yaml：
  - 层：x → RMSNorm（input_layernorm）→ q / k / v 投影 → 每头 RMSNorm（q_norm / k_norm，RoPE 之前）→ RoPE（default，θ = rope_theta，
    前后两半配对 rotate_half）→ 因果注意力（缩放 1/√head_dim，GQA：q 头 h 用 kv 头 h // (heads / kv_heads)）→ o_proj；
    残差 h = x + attn.out；第 0 层稠密 MLP：RMSNorm（post_attention_layernorm）→ silu(gate)·up → down；
    MoE 层：RMSNorm → 路由 logits = h·Wr^T，scores = sigmoid，按 scores + expert_bias 取 top-k，权重 = 选中的 scores 归一化 × router_scaling_factor，
    路由专家加权和 + 共享专家；out = h + mlp.out
  - 观测点布局同 GPU 抓数（name_map.yaml activations）：qkv.out 按 KV 组交错 [S, 1, 10240]（每组 8 个 q 头、1 个 k 头、1 个 v 头），
    fc1.out = [gate | up]，第 0 层的 mlp_norm.out 是残差和（Megatron 第 0 层归一化融进 fc1）
  - 输出头：final_norm → logits = y·W_out^T → 逐 token 交叉熵（FP64 log-softmax）

自检（命令行）：用 golden 第 1 步的层输入算一层，每个观测点对 golden 的 rel_l2 应在 BF16 末位量级（1e-3）；布局或语义错了是 O(1)。
  python fp64_forward.py --hf <HF 目录> --golden <golden 运行目录> --layer 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from janus.store import read_safetensors_tensor  # noqa: E402


class HF:
    """按名字读 HF safetensors 里的张量，转 FP64。"""

    def __init__(self, hf_dir):
        self.dir = Path(hf_dir)
        self.cfg = json.loads((self.dir / "config.json").read_text())
        idx = self.dir / "model.safetensors.index.json"
        self.where = json.loads(idx.read_text())["weight_map"] if idx.exists() else None

    def get(self, name: str) -> np.ndarray:
        f = self.dir / self.where[name] if self.where else next(self.dir.glob("*.safetensors"))
        arr, _ = read_safetensors_tensor(f, name)
        return arr.astype(np.float64)


def rmsnorm(x, w, eps):
    x = np.asarray(x, np.float64)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rope(x, pos, theta, dim):
    """x [S, H, dim]；HF default RoPE：emb = cat(freqs, freqs)，x·cos + rotate_half(x)·sin。"""
    inv = theta ** (-np.arange(0, dim, 2, dtype=np.float64) / dim)
    ang = np.asarray(pos, np.float64)[:, None] * inv[None, :]
    cos = np.concatenate([np.cos(ang)] * 2, axis=-1)[:, None, :]
    sin = np.concatenate([np.sin(ang)] * 2, axis=-1)[:, None, :]
    h = dim // 2
    rot = np.concatenate([-x[..., h:], x[..., :h]], axis=-1)
    return x * cos + rot * sin


def attention(q, k, v, heads, kv_heads):
    """q [S, H, d]，k / v [S, KV, d]，因果；返回 [S, H·d]。逐头算，FP64。"""
    S, H, d = q.shape
    rep = heads // kv_heads
    out = np.empty((S, H, d))
    mask = np.triu(np.ones((S, S), dtype=bool), 1)
    for h in range(H):
        kk, vv = k[:, h // rep], v[:, h // rep]
        s = (q[:, h] @ kk.T) / np.sqrt(d)
        s[mask] = -np.inf
        s -= s.max(axis=-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(axis=-1, keepdims=True)
        out[:, h] = p @ vv
    return out.reshape(S, H * d)


def silu(x):
    return x / (1.0 + np.exp(-x))


def topk_sets(scores_biased, k):
    """按分数取 top-k，升序返回下标（同 GPU 抓数的 topk_idx：选中的 k 个专家升序）。并列取小下标。"""
    idx = np.argsort(-scores_biased, axis=-1, kind="stable")[:, :k]
    return np.sort(idx, axis=-1)


def moe_forward(hf: HF, i: int, hn, bias=None) -> dict:
    """MoE 层的 MLP 部分，hn [S, D] 是 MLP 前归一化的输出。返回路由 logits / topk_idx / probs、路由专家合并、共享专家（层内相对名）。"""
    c = hf.cfg
    p = f"model.layers.{i}."
    hn = np.asarray(hn, np.float64).reshape(-1, c["hidden_size"])
    S, D = hn.shape
    E, K = c["num_experts"], c["num_experts_per_tok"]
    out = {}
    logits = hn @ hf.get(p + "mlp.router.gate.weight").T
    out["mlp.router.logits"] = logits[:, None]
    scores = 1.0 / (1.0 + np.exp(-logits))
    b = hf.get(p + "mlp.expert_bias") if bias is None else np.asarray(bias, np.float64)
    sel = topk_sets(scores + b, K)
    out["mlp.router.topk_idx"] = sel.astype(np.int32)
    w = np.take_along_axis(scores, sel, axis=-1)
    w = w / w.sum(axis=-1, keepdims=True) * c["router_scaling_factor"]
    probs = np.zeros((S, E))
    np.put_along_axis(probs, sel, w, axis=-1)
    out["mlp.router.probs"] = probs
    routed = np.zeros((S, D))
    for e in range(E):
        rows = np.nonzero((sel == e).any(axis=-1))[0]
        if rows.size == 0:
            continue
        ep = p + f"mlp.experts.{e}."
        he = hn[rows]
        y = (silu(he @ hf.get(ep + "gate_proj.weight").T) * (he @ hf.get(ep + "up_proj.weight").T)) @ hf.get(ep + "down_proj.weight").T
        routed[rows] += probs[rows, e][:, None] * y
    sp = p + "mlp.shared_mlp."
    shared = (silu(hn @ hf.get(sp + "gate_proj.weight").T) * (hn @ hf.get(sp + "up_proj.weight").T)) @ hf.get(sp + "down_proj.weight").T
    out["mlp.routed.out"] = routed[:, None]
    out["mlp.shared.out"] = shared[:, None]
    out["mlp.out"] = (routed + shared)[:, None]
    return out


def layer_forward(hf: HF, i: int, x, pos, bias=None):
    """第 i 层，x [S, D]（任意 dtype，转 FP64）。返回 {观测点名: FP64 数组（与 GPU 抓数同形状）}。
    bias：MoE 层的 expert_bias（默认读 HF 的）。"""
    c = hf.cfg
    D, H, KV, d = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    eps, theta = c["rms_norm_eps"], c["rope_parameters"]["rope_theta"]
    p = f"model.layers.{i}."
    x = np.asarray(x, np.float64).reshape(-1, D)
    S = x.shape[0]
    out = {}
    h1 = rmsnorm(x, hf.get(p + "input_layernorm.weight"), eps)
    q = (h1 @ hf.get(p + "self_attn.q_proj.weight").T).reshape(S, H, d)
    k = (h1 @ hf.get(p + "self_attn.k_proj.weight").T).reshape(S, KV, d)
    v = (h1 @ hf.get(p + "self_attn.v_proj.weight").T).reshape(S, KV, d)
    rep = H // KV
    out[f"layers.{i}.attn.qkv.out"] = np.concatenate(
        [np.concatenate([q[:, g * rep:(g + 1) * rep].reshape(S, rep * d), k[:, g], v[:, g]], axis=-1) for g in range(KV)],
        axis=-1)[:, None]
    qn = rmsnorm(q, hf.get(p + "self_attn.q_norm.weight"), eps)
    kn = rmsnorm(k, hf.get(p + "self_attn.k_norm.weight"), eps)
    out[f"layers.{i}.attn.q_norm.out"] = qn[:, None]
    out[f"layers.{i}.attn.k_norm.out"] = kn[:, None]
    core = attention(rope(qn, pos, theta, d), rope(kn, pos, theta, d), v, H, KV)
    out[f"layers.{i}.attn.core.out"] = core[:, None]
    proj = core @ hf.get(p + "self_attn.o_proj.weight").T
    out[f"layers.{i}.attn.proj.out"] = proj[:, None]
    out[f"layers.{i}.attn.out"] = proj[:, None]
    h = x + proj
    hn = rmsnorm(h, hf.get(p + "post_attention_layernorm.weight"), eps)
    if i < c["first_k_dense_replace"]:
        out[f"layers.{i}.mlp_norm.out"] = h[:, None]                  # Megatron 第 0 层：残差和
        gate, up = hn @ hf.get(p + "mlp.gate_proj.weight").T, hn @ hf.get(p + "mlp.up_proj.weight").T
        out[f"layers.{i}.mlp.fc1.out"] = np.concatenate([gate, up], axis=-1)[:, None]
        mlp = (silu(gate) * up) @ hf.get(p + "mlp.down_proj.weight").T
        out[f"layers.{i}.mlp.fc2.out"] = mlp[:, None]
    else:
        out[f"layers.{i}.mlp_norm.out"] = hn[:, None]
        m = moe_forward(hf, i, hn, bias)
        out.update({f"layers.{i}.{k}": v for k, v in m.items()})
        routed, shared = m["mlp.routed.out"][:, 0], m["mlp.shared.out"][:, 0]
        mlp = routed + shared
    out[f"layers.{i}.mlp.out"] = mlp[:, None]
    out[f"layers.{i}.out"] = (h + mlp)[:, None]
    return out


def head_forward(hf: HF, y, labels):
    """y [S, D]：最后一层输出。返回 final_norm.out、logits、loss.per_token（FP64）。"""
    c = hf.cfg
    yn = rmsnorm(np.asarray(y, np.float64).reshape(-1, c["hidden_size"]), hf.get("model.norm.weight"), c["rms_norm_eps"])
    logits = yn @ hf.get("lm_head.weight").T
    m = logits.max(axis=-1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=-1))
    lab = np.asarray(labels).reshape(-1)
    loss = lse - logits[np.arange(lab.size), lab]
    return {"final_norm.out": yn[:, None], "logits": logits[:, None], "loss.per_token": loss[None]}


def main(argv=None) -> int:
    from janus import metrics as M
    from janus.store import Run
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--layer", type=int, default=1)
    a = ap.parse_args(argv)
    hf, g = HF(a.hf), Run.open(a.golden)
    pos = g.tensor(1, "input.position_ids")[0]
    ref = layer_forward(hf, a.layer, g.tensor(1, f"layers.{a.layer}.in")[:, 0], pos)
    for n, v in ref.items():
        if n in g.names(1) and not n.endswith("topk_idx"):
            gv = g.tensor(1, n)
            print(f"{n:<40} rel_l2(golden, FP64) = {M.rel_l2(gv.astype(np.float64).reshape(v.shape), v):.3e}")
        elif n.endswith("topk_idx") and n in g.names(1):
            print(f"{n:<40} top-k 集合一致 = {M.topk_agreement(g.tensor(1, n), v).set_frac:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
