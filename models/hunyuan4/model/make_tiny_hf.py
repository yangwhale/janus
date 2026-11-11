"""造一个小维度混元 4 的 HF 目录（随机 BF16 权重）：hy4-tiny 给 CPU 冒烟，hy4-mini 给端到端对齐（两边从同一份权重起训）。

    python make_tiny_hf.py --out /tmp/hy4-tiny-hf [--config tiny] [--layers 4] [--mtp 1] [--seed 0]
    python make_tiny_hf.py --out /tmp/hy4-mini-hf --config mini --layers 7 --mtp 1 --seed 0
    python make_tiny_hf.py --out /tmp/hy4-mid-hf --config mid --layers 7 --mtp 1 --seed 0
    python make_tiny_hf.py --out /tmp/hy4-mini-noihc-hf --config mini --layers 7 --mtp 1 --seed 0 --no-ihc

维度取 MaxText 分支 hf_model_configs 的 hy4_tiny_dict（hidden 256、4 头、8 专家……）、hy4_mini_dict（hidden 1024、8 头、
16 专家、真实词表，每头维度同 hy4-preview）或 hy4_mid_dict（hy4-mini 的结构，宽 6144、64 头、q_lora 2048、稠密 FFN 18432、专家宽 512），层数截到 --layers；
张量名与形状按 HF 发布的命名（与 hf_shapes 一致，真实检查点的头会在 verify_ckpt 里逐个核）。
另写 tensor_sha256.json（逐张量原始字节的 sha256）与 model.safetensors.index.json，和 GPU 侧上传的目录同一种格式。
dtype 按 hy4_map.hf_dtype（与发布的检查点同一种分布：iHC、hc_head、主干路由偏置与 sink 存 FP32，其余 BF16）。

`--no-ihc`（结构消融：主干换成普通残差，只有一条流）：照常按同一个随机数顺序造出全部张量，再去掉 iHC 与 hc_head 的
参数，config 写 `hc_mult` 1、`enable_ihc` false。所以其余张量与不加这个参数时逐位相同。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import ml_dtypes
import numpy as np


def hf_shapes(c: dict, layers: int, mtp: int) -> dict[str, tuple]:
    """HF 张量名 → 形状（按 HF 发布的 Hy4 命名；hc 的 fn/base/scale 存 pre、post 两半）。"""
    D, V, n = c["hidden_size"], c["vocab_size"], c["hc_mult"]
    H, ql, kl = c["num_attention_heads"], c["q_lora_rank"], c["kv_lora_rank"]
    nope, rope, v = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
    nh, hd = c["index_n_heads"], c["index_head_dim"]
    F, I, E = c["intermediate_size"], c["moe_intermediate_size"], c["n_routed_experts"]
    Is = I * c.get("n_shared_experts", 1)
    s = {"model.embed_tokens.weight": (V, D), "lm_head.weight": (V, D), "model.norm.weight": (D,)}
    if n > 1:  # hc_mult 1：没有 iHC 与 hc_head（结构消融，--no-ihc）
        s.update({"model.hc_head.hc_head_fn": (n, n * D), "model.hc_head.hc_head_base": (n,),
                  "model.hc_head.hc_head_scale": (1,)})

    def layer(p, kind, with_indexer, with_hc):
        s[p + "input_layernorm.weight"] = (D,)
        s[p + "post_attention_layernorm.weight"] = (D,)
        a = p + "self_attn."
        s[a + "q_a_proj.weight"] = (ql, D)
        s[a + "q_a_layernorm.weight"] = (ql,)
        s[a + "q_b_proj.weight"] = (H * (nope + rope), ql)
        s[a + "kv_a_proj_with_mqa.weight"] = (kl + rope, D)
        s[a + "kv_a_layernorm.weight"] = (kl,)
        s[a + "kv_b_proj.weight"] = (H * (nope + v), kl)
        s[a + "o_proj.weight"] = (D, H * v)
        s[a + "linear_gate.weight"] = (H * v, D)
        s[a + "learnable_sink_param"] = (H,)
        if with_indexer:
            s[a + "indexer.wq_b.weight"] = (nh * hd, ql)
            s[a + "indexer.wk.weight"] = (hd, D)
            s[a + "indexer.k_norm.weight"] = (hd,)
            s[a + "indexer.k_norm.bias"] = (hd,)
            s[a + "indexer.weights_proj.weight"] = (nh, D)
        if with_hc and n > 1:
            for hc in ("hc_attn_layer", "hc_mlp_layer"):
                s[f"{p}{hc}.hc_pre.hc_fn"] = (2 * n, n * D)
                s[f"{p}{hc}.hc_pre.hc_base"] = (2 * n,)
                s[f"{p}{hc}.hc_pre.hc_scale"] = (2,)
        m = p + "mlp."
        if kind == "dense":
            s[m + "gate_proj.weight"] = (F, D)
            s[m + "up_proj.weight"] = (F, D)
            s[m + "down_proj.weight"] = (D, F)
        else:
            s[m + "gate.weight"] = (E, D)
            s[m + "gate.e_score_correction_bias"] = (E,)
            s[m + "experts.gate_up_proj"] = (E, 2 * I, D)
            s[m + "experts.down_proj"] = (E, D, I)
            s[m + "shared_experts.gate_proj.weight"] = (Is, D)
            s[m + "shared_experts.up_proj.weight"] = (Is, D)
            s[m + "shared_experts.down_proj.weight"] = (D, Is)

    for i in range(layers):
        layer(f"model.layers.{i}.", c["mlp_layer_types"][i], c["indexer_types"][i] == "full", True)
    for j in range(mtp):
        p = f"model.mtp_layers.{j}."
        s[p + "enorm.weight"] = (D,)
        s[p + "hnorm.weight"] = (D,)
        s[p + "eh_proj.weight"] = (D, 2 * D)
        s[p + "final_layernorm.weight"] = (D,)
        layer(p, "sparse", True, False)
    return s


# tencent/Hy4-preview 发布的 config.json 里与维度无关的键（取自截 4 层目录的 config.json，transformers 5.16.2 写出）。
# MaxText 的 hy4_*_dict 只有维度，HF HYV4Config 读缺键的文件会取默认值（如 rope_theta 1e4、gated_mla / learnable_sink /
# enable_ihc 为 None），所以写出的 config.json 补上这些键，让 HF、MaxText、Megatron 读同一份完整的配置。
RELEASE_KEYS = {
    "architectures": ["HYV4ForCausalLM"], "attention_bias": False, "attention_dropout": 0.0,
    "bitwise_backward_align": False, "bos_token_id": 120000, "dtype": "bfloat16", "enable_ihc": True,
    "enable_lm_head_fp32": True, "eos_token_id": 120025, "gated_mla": True, "gating_type": "elementwise",
    "hc_eps": 1e-06, "hc_magnitude": 2.0, "hidden_act": "silu", "initializer_range": 0.006, "learnable_sink": True,
    "learnable_sink_init": 0.0, "max_position_embeddings": 1048576, "model_type": "hy_v4", "mtp_loss_factor": 0.1,
    "n_group": 1, "norm_topk_prob": True, "pad_token_id": 120002, "rms_norm_eps": 1e-05,
    "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"}, "routed_scaling_factor": 2.827,
    "swiglu_limit": 10.0, "tie_word_embeddings": False, "topk_group": 1, "use_cache": True, "use_dsa": True,
    "use_mla": True, "torch_dtype": "bfloat16",
}


def full_config(c: dict) -> dict:
    """维度（MaxText hy4_*_dict 截层后）+ 发布配置里与维度无关的键 + 由维度推出的键。"""
    out = dict(RELEASE_KEYS)
    out.update(c)
    out["head_dim"] = c["qk_rope_head_dim"]
    out["qk_head_dim"] = c["qk_nope_head_dim"] + c["qk_rope_head_dim"]
    out["num_key_value_heads"] = min(8, c["num_attention_heads"])  # MLA 不用；发布的 64 头配 8，小模型取不超过头数
    out["layer_types"] = ["deepseek_sparse_attention"] * c["num_hidden_layers"]
    return out


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]):
    hdr, off, blobs = {}, 0, []
    for k in sorted(tensors):
        a = np.ascontiguousarray(tensors[k])
        b = a.tobytes()
        hdr[k] = {"dtype": {"bfloat16": "BF16", "float32": "F32"}[str(a.dtype)], "shape": list(a.shape),
                  "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(hdr).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--mtp", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vocab", type=int, default=None, help="改词表大小（默认取 config：tiny 1024、mini 120832）")
    p.add_argument("--config", choices=("tiny", "mini", "mid"), default="tiny")
    p.add_argument("--no-ihc", action="store_true", help="去掉 iHC 与 hc_head（其余张量与不加时逐位相同）")
    p.add_argument("--std", type=float, default=None,
                   help="权重矩阵的标准差（默认 tiny 0.02；mini、mid 取 HF initializer_range 0.006）")
    a = p.parse_args(argv)
    import sys

    from maxtext.checkpoint_conversion.utils import hf_model_configs
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hy4_map import hf_dtype, stage_hf_config
    base = {"tiny": hf_model_configs.hy4_tiny_dict, "mini": hf_model_configs.hy4_mini_dict,
            "mid": getattr(hf_model_configs, "hy4_mid_dict", None)}[a.config]
    if base is None:
        raise SystemExit("MaxText 分支里没有 hy4_mid_dict（要含 hy4-mid 的提交）")
    std = a.std if a.std is not None else (0.02 if a.config == "tiny" else 0.006)
    c = stage_hf_config(dict(base), a.layers)
    c["num_nextn_predict_layers"] = a.mtp
    if a.vocab:
        c["vocab_size"] = a.vocab
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    tensors = {}
    for name, shp in hf_shapes(c, a.layers, a.mtp).items():
        if name.endswith(("layernorm.weight", "norm.weight", "enorm.weight", "hnorm.weight", "k_norm.weight")):
            x = 1.0 + 0.05 * rng.standard_normal(shp)
        elif name.endswith("e_score_correction_bias"):
            x = 0.01 * rng.standard_normal(shp)
        else:
            x = std * rng.standard_normal(shp)
        tensors[name] = x.astype(np.float32 if hf_dtype(name) == "float32" else ml_dtypes.bfloat16)
    if a.no_ihc:
        tensors = {k: v for k, v in tensors.items() if ".hc_pre." not in k and not k.startswith("model.hc_head.")}
        c["hc_mult"] = 1
    write_safetensors(out / "model.safetensors", tensors)
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": {k: "model.safetensors" for k in sorted(tensors)}}, indent=1))
    (out / "tensor_sha256.json").write_text(json.dumps(
        {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in sorted(tensors.items())}, indent=1))
    cfg = full_config(c)
    if a.no_ihc:
        cfg["enable_ihc"] = False
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"写了 {len(tensors)} 个张量到 {out}")


if __name__ == "__main__":
    main()
