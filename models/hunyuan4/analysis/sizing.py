"""混元 4（Hy4-preview）参数量与显存估算：逐模块参数量从公开 config 算，再对 checkpoint 头逐张量核对。

用法：
    python models/hunyuan4/analysis/sizing.py --config config.json [--shapes shapes.json] [--json out.json]

- config.json：Hugging Face `tencent/Hy4-preview` 的公开配置（revision 705d81ee）。
- shapes.json（可选）：checkpoint 全部 131 个分片 safetensors 头里的 {张量名: [dtype, shape]}，用 HTTP Range
  只读头部得到。给了就逐层核对公式与 checkpoint 逐个相等，不等就报错退出。
- 参数量是精确值；显存一栏是估算，假设写在 MEM_ASSUMPTIONS，要以实测峰值为准。
"""

from __future__ import annotations

import argparse
import json
import math

GB = 1e9

# 显存估算的假设（字节 / 参数；激活按「每 token 每层」的字节数，BF16，序列 4096，每设备 1 条序列，不重算）
MEM_ASSUMPTIONS = {
    "tpu_bytes_per_param": 20,  # MaxText weight_dtype=float32：参数 4 + Adam m 4 + v 4 + 梯度累积缓冲 4 + 当前 microbatch 梯度 4
    "gpu_fsdp_bytes_per_param": 16,  # FSDP2 全切分：FP32 主权重 4 + FP32 梯度 4 + Adam m/v 8
    "gpu_unit_transient_bytes_per_param": 10,  # 正在算的 FSDP 单元：BF16 参数（当前 + 预取）4 + BF16 梯度 2 + FP32 归约缓冲 4
    "act_bytes_per_token": {  # 估算：保存给反向的主要张量
        "moe_layer": 1.09e6,
        "dense_layer": 0.81e6,
        "head": 1.0e6,  # FP32 logits + 它的梯度（120832 × 4 B × 2）
    },
    "seq": 4096,
}


def module_params(c: dict) -> dict:
    H, V, nh = c["hidden_size"], c["vocab_size"], c["num_attention_heads"]
    ql, kl = c["q_lora_rank"], c["kv_lora_rank"]
    dn, dr, dv = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
    ih, idim = c["index_n_heads"], c["index_head_dim"]
    E, mi, di, hc = c["n_routed_experts"], c["moe_intermediate_size"], c["intermediate_size"], c["hc_mult"]
    p = {
        "mla": H * ql + ql + ql * nh * (dn + dr) + H * (kl + dr) + kl + kl * nh * (dn + dv) + nh * dv * H,
        "gate_proj": nh * dv * H,  # gated MLA，elementwise：每头每个 value 通道一个门
        "sink": nh,
        "indexer": ql * ih * idim + H * idim + H * ih + 2 * idim,  # wq_b + wk + weights_proj + k_norm(w, b)
        "ihc_per_layer": 2 * (2 * hc * hc * H + 2 * hc + 2),  # attn / ffn 两处：fn [2hc, hc·H] + base [2hc] + scale [2]
        "layer_norms": 2 * H,
        "dense_mlp": 3 * H * di,
        "one_expert": 3 * H * mi,
        "routed_experts": E * 3 * H * mi,
        "router": E * H + E,  # gate.weight + e_score_correction_bias
        "shared_expert": c["n_shared_experts"] * 3 * H * mi,
        "embed": V * H,
        "lm_head": V * H,
        "final_norm": H,
        "hc_head": hc * hc * H + hc + 1,
        "mtp_extra": 2 * H * H + 5 * H,  # eh_proj + enorm / hnorm / input_ln / post_attn_ln / final_ln
    }
    p["attention"] = p["mla"] + p["gate_proj"] + p["sink"]
    return p


def layer_params(c: dict, p: dict, i: int, active: bool = False) -> int:
    full = c["indexer_types"][i] == "full"
    if c["mlp_layer_types"][i] == "dense":
        mlp = p["dense_mlp"]
    else:
        experts = c["num_experts_per_tok"] * p["one_expert"] if active else p["routed_experts"]
        mlp = experts + p["router"] + p["shared_expert"]
    return p["attention"] + (p["indexer"] if full else 0) + p["ihc_per_layer"] + p["layer_norms"] + mlp


def mtp_params(c: dict, p: dict, active: bool = False) -> int:
    experts = c["num_experts_per_tok"] * p["one_expert"] if active else p["routed_experts"]
    return p["mtp_extra"] + p["attention"] + p["indexer"] + experts + p["router"] + p["shared_expert"]


def outer(p: dict) -> int:
    return p["embed"] + p["lm_head"] + p["final_norm"] + p["hc_head"]


def truncated(c: dict, p: dict, n_layers: int, mtp: bool) -> dict:
    layers = [layer_params(c, p, i) for i in range(n_layers)]
    total = outer(p) + sum(layers) + (mtp_params(c, p) if mtp else 0)
    moe = [i for i in range(n_layers) if c["mlp_layer_types"][i] != "dense"] + (["mtp"] if mtp else [])
    routed = len(moe) * p["routed_experts"]
    return {
        "layers": n_layers,
        "mtp": mtp,
        "params": total,
        "routed_expert_params": routed,
        "non_expert_params": total - routed,
        "moe_blocks": len(moe),
    }


def memory(c: dict, t: dict) -> dict:
    a = MEM_ASSUMPTIONS
    s, apt = a["seq"], a["act_bytes_per_token"]
    n_moe = t["moe_blocks"]
    n_dense = t["layers"] - (n_moe - (1 if t["mtp"] else 0))
    heads = 2 if t["mtp"] else 1
    act = s * (n_moe * apt["moe_layer"] + n_dense * apt["dense_layer"] + heads * apt["head"])
    largest_unit = 9_978_461_268 if n_moe else 615_201_620  # 一层 MoE（带 full indexer）或第 0 层
    out = {"activation_gb_est": act / GB}
    for dev in (8,):
        static = t["params"] * a["tpu_bytes_per_param"] / dev
        out[f"tpu_v5p_{dev}dev_static_gb"] = static / GB
        out[f"tpu_v5p_{dev}dev_total_gb_est"] = (static + act) / GB
        g_static = t["params"] * a["gpu_fsdp_bytes_per_param"] / dev
        g_trans = largest_unit * a["gpu_unit_transient_bytes_per_param"]
        out[f"gpu_b200_{dev}_fsdp_static_gb"] = g_static / GB
        out[f"gpu_b200_{dev}_fsdp_transient_gb"] = g_trans / GB
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shapes")
    ap.add_argument("--json")
    a = ap.parse_args()
    c = json.load(open(a.config))
    p = module_params(c)
    L = c["num_hidden_layers"]
    backbone = outer(p) + sum(layer_params(c, p, i) for i in range(L))
    mtp = mtp_params(c, p)
    act_backbone = outer(p) + sum(layer_params(c, p, i, active=True) for i in range(L))
    res = {
        "modules": p,
        "layer0": layer_params(c, p, 0),
        "moe_layer_full_indexer": layer_params(c, p, 1),
        "moe_layer_shared_indexer": layer_params(c, p, 2),
        "backbone": backbone,
        "mtp": mtp,
        "total_with_mtp": backbone + mtp,
        "active_backbone_incl_embed_and_head": act_backbone,
        "active_mtp": mtp_params(c, p, active=True),
        "anchors": {
            "backbone": "770B",
            "active": "49B",
            "mtp": "10B",
            "mtp_active": "0.7B",
            "source": "Hugging Face tencent/Hy4-preview 模型卡（backbone 不含 MTP）",
        },
        "stages": {},
        "memory_assumptions": MEM_ASSUMPTIONS,
    }
    for name, n, m in [
        ("A", 1, False),
        ("B", 2, False),
        ("C", 3, False),
        ("D_3L_mtp", 3, True),
        ("D_2L_mtp", 2, True),
        ("4L_mtp", 4, True),
    ]:
        t = truncated(c, p, n, m)
        t.update(memory(c, t))
        res["stages"][name] = t
    if a.shapes:
        s = json.load(open(a.shapes))
        ck = sum(math.prod(v[1]) for v in s.values())
        ck_mtp = sum(math.prod(v[1]) for k, v in s.items() if "mtp_layers" in k)
        for i in range(L):
            got = sum(math.prod(v[1]) for k, v in s.items() if k.startswith(f"model.layers.{i}."))
            assert got == layer_params(c, p, i), (i, got, layer_params(c, p, i))
        assert ck_mtp == mtp and ck - ck_mtp == backbone, (ck, ck_mtp)
        res["checkpoint_check"] = {"tensors": len(s), "params": ck, "per_layer_equal": True, "mtp_equal": True}
    txt = json.dumps(res, indent=1, ensure_ascii=False)
    if a.json:
        open(a.json, "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
