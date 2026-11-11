"""混元 4 checkpoint（HF 名字）→ Megatron-Core 参数名的对照表与载入。

同一张表给两处用：`load_hf_into_mcore`（等价性检查、训练脚本直接载入）与 Megatron-Bridge 的
`HYV4Bridge.mapping_registry`（hy4_bridge.py）。只支持 TP = 1；EP 按 Megatron 的切法（每个 EP rank 拿连续的
num_experts / EP 个专家）；PP 时只载入本段有的参数（本段的解码层按全局层号取，嵌入 / 输出头 / MTP 在哪段载哪段）。

checkpoint 里的名字是公开 checkpoint 的原名（`hc_attn_layer.hc_pre.hc_fn`、`learnable_sink_param`、
`linear_gate` 等），不是 HF 模块名；HF transformers 加载时按 `conversion_mapping.py` 的 "hy_v4" 改名，
这里直接对原名。
"""

from __future__ import annotations

import json
import os
import re
import struct

import torch

# (Megatron 名, checkpoint 名)。{i} 是解码层号。
GLOBAL = [
    ("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
    ("output_layer.weight", "lm_head.weight"),
    ("decoder.final_layernorm.norm.weight", "model.norm.weight"),
    ("decoder.final_layernorm.hc_fn", "model.hc_head.hc_head_fn"),
    ("decoder.final_layernorm.hc_base", "model.hc_head.hc_head_base"),
    ("decoder.final_layernorm.hc_scale", "model.hc_head.hc_head_scale"),
]
LAYER = [
    ("attn_hc.fn", "hc_attn_layer.hc_pre.hc_fn"),
    ("attn_hc.base", "hc_attn_layer.hc_pre.hc_base"),
    ("attn_hc.scale", "hc_attn_layer.hc_pre.hc_scale"),
    ("ffn_hc.fn", "hc_mlp_layer.hc_pre.hc_fn"),
    ("ffn_hc.base", "hc_mlp_layer.hc_pre.hc_base"),
    ("ffn_hc.scale", "hc_mlp_layer.hc_pre.hc_scale"),
    ("input_layernorm.weight", "input_layernorm.weight"),
    ("pre_mlp_layernorm.weight", "post_attention_layernorm.weight"),
]
ATTN = [  # 前缀：主干 decoder.layers.{i}.self_attention.，MTP 见 mtp_names
    ("linear_q_down_proj.weight", "self_attn.q_a_proj.weight"),
    ("q_layernorm.weight", "self_attn.q_a_layernorm.weight"),
    ("linear_q_up_proj.weight", "self_attn.q_b_proj.weight"),
    ("linear_kv_down_proj.weight", "self_attn.kv_a_proj_with_mqa.weight"),
    ("kv_layernorm.weight", "self_attn.kv_a_layernorm.weight"),
    ("linear_kv_up_proj.weight", "self_attn.kv_b_proj.weight"),
    ("linear_proj.weight", "self_attn.o_proj.weight"),
    ("linear_gate.weight", "self_attn.linear_gate.weight"),
    ("core_attention.sinks", "self_attn.learnable_sink_param"),
]
INDEXER = [
    ("core_attention.indexer.linear_wq_b.weight", "self_attn.indexer.wq_b.weight"),
    ("core_attention.indexer.linear_wk.weight", "self_attn.indexer.wk.weight"),
    ("core_attention.indexer.k_norm.weight", "self_attn.indexer.k_norm.weight"),
    ("core_attention.indexer.k_norm.bias", "self_attn.indexer.k_norm.bias"),
    ("core_attention.indexer.linear_weights_proj.weight", "self_attn.indexer.weights_proj.weight"),
]
MOE_FIXED = [
    ("router.weight", "mlp.gate.weight"),
    ("router.expert_bias", "mlp.gate.e_score_correction_bias"),
    ("shared_experts.linear_fc2.weight", "mlp.shared_experts.down_proj.weight"),
]
MTP_EXTRA = [
    ("enorm.weight", "enorm.weight"),
    ("hnorm.weight", "hnorm.weight"),
    ("eh_proj.weight", "eh_proj.weight"),
    ("final_layernorm.weight", "final_layernorm.weight"),
]


class Safetensors:
    """按名字读 safetensors 目录里的张量（只读需要的那些，不整个载入）。"""

    def __init__(self, d: str):
        self.d = d
        idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.where = idx
        self._hdr = {}

    def _header(self, f):
        if f not in self._hdr:
            with open(os.path.join(self.d, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self._hdr[f] = (n, json.loads(fh.read(n)))
        return self._hdr[f]

    def names(self):
        return list(self.where)

    def get(self, name: str) -> torch.Tensor:
        from safetensors import safe_open

        f = self.where[name]
        with safe_open(os.path.join(self.d, f), framework="pt") as fh:
            return fh.get_tensor(name)

    def get_slice(self, name: str, lo: int, hi: int) -> torch.Tensor:
        """第 0 维 [lo, hi) 的切片（专家张量按 EP 只读自己那段）。"""
        from safetensors import safe_open

        f = self.where[name]
        with safe_open(os.path.join(self.d, f), framework="pt") as fh:
            return fh.get_slice(name)[lo:hi]


def layer_plan(config, i: int):
    """第 i 层（0 起）的 (Megatron 名后缀, checkpoint 名后缀, 变换)。变换：None 原样；'cat' 两个张量沿第 0 维拼。"""
    out = [(m, h, None) for m, h in LAYER]
    out += [("self_attention." + m, h, None) for m, h in ATTN]
    if config.hy4_indexer_types[i] == "full":
        out += [("self_attention." + m, h, None) for m, h in INDEXER]
    if config.moe_layer_freq[i]:
        out += [("mlp." + m, h, None) for m, h in MOE_FIXED]
        out += [
            (
                "mlp.shared_experts.linear_fc1.weight",
                ("mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight"),
                "cat",
            )
        ]
    else:
        out += [("mlp.linear_fc1.weight", ("mlp.gate_proj.weight", "mlp.up_proj.weight"), "cat")]
        out += [("mlp.linear_fc2.weight", "mlp.down_proj.weight", None)]
    return out


def expert_param_name(prefix: str, which: str, local: int, grouped: bool) -> str:
    """路由专家在 Megatron 里的名字：TE 分组 GEMM 是 experts.linear_fc1.weight{e}，SequentialMLP 是 local_experts.{e}。"""
    if grouped:
        return f"{prefix}mlp.experts.{which}.weight{local}"
    return f"{prefix}mlp.experts.local_experts.{local}.{which}.weight"


@torch.no_grad()
def load_hf_into_mcore(
    model, config, hf_dir: str, ep_rank: int = 0, ep_size: int = 1, strict: bool = True, verify: bool = False
) -> dict:
    """把 checkpoint 载入只用 Megatron-Core 建的 GPTModel（TP = 1）。返回 {Megatron 名: checkpoint 名}。

    张量按目标参数的 dtype 拷贝（FP32 参数拿到 BF16 checkpoint 的精确上转）。`strict` 时模型里每个参数与 buffer
    （除 MoE 的计数 buffer）都必须被写到一次。

    `verify`：不写，按同一张映射逐个核「参数 == checkpoint 张量转成参数的 dtype」，并累加两边的 FP64 平方和
    （本 rank 的那一份）；返回 {"mismatch": [...], "sumsq_model": x, "sumsq_ckpt": y, "n": 个数}。开训前用。
    """
    st = Safetensors(hf_dir)
    params = dict(model.named_parameters())
    params.update({k: v for k, v in model.named_buffers() if k.endswith("expert_bias")})
    done = {}
    chk = {"mismatch": [], "sumsq_model": 0.0, "sumsq_ckpt": 0.0, "n": 0}

    def put(mname, t):
        dst = params[mname]
        assert tuple(dst.shape) == tuple(t.shape), (mname, tuple(dst.shape), tuple(t.shape))
        if verify:
            src = t.to(device=dst.device)
            if not torch.equal(dst.detach(), src.to(dst.dtype)) or not torch.equal(
                src.to(dst.dtype).to(src.dtype), src
            ):
                chk["mismatch"].append(mname)
            chk["sumsq_model"] += float(dst.detach().double().square().sum())
            chk["sumsq_ckpt"] += float(src.double().square().sum())
            chk["n"] += 1
        else:
            dst.copy_(t.to(dst.dtype))
        done[mname] = True

    for m, h in GLOBAL:
        if m in params:  # PP：嵌入在首段（与 MTP 段），输出头与 hc_head 在末段
            put(m, st.get(h))
    grouped = any(".experts.linear_fc1.weight" in k for k in params)
    from hy4_mcore import stage_layers

    rg = stage_layers(config)
    for i in rg:
        pre_m, pre_h = f"decoder.layers.{i - rg.start}.", f"model.layers.{i}."
        for m, h, tf in layer_plan(config, i):
            if tf == "cat":
                put(pre_m + m, torch.cat([st.get(pre_h + x) for x in h], dim=0))
            else:
                put(pre_m + m, st.get(pre_h + h))
        if config.moe_layer_freq[i]:
            E = config.num_moe_experts
            assert E % ep_size == 0
            n_local = E // ep_size
            lo = ep_rank * n_local
            gu = st.get_slice(pre_h + "mlp.experts.gate_up_proj", lo, lo + n_local)
            dn = st.get_slice(pre_h + "mlp.experts.down_proj", lo, lo + n_local)
            for e in range(n_local):
                put(expert_param_name(pre_m, "linear_fc1", e, grouped), gu[e])
                put(expert_param_name(pre_m, "linear_fc2", e, grouped), dn[e])
    if getattr(config, "mtp_num_layers", None) and any(k.startswith("mtp.") for k in params):
        for m, h, tf in mtp_names(config):
            if tf == "cat":
                put(m, torch.cat([st.get(x) for x in h], dim=0))
            elif tf == "experts":
                E = config.num_moe_experts
                n_local = E // ep_size
                lo = ep_rank * n_local
                gu = st.get_slice(h + "mlp.experts.gate_up_proj", lo, lo + n_local)
                dn = st.get_slice(h + "mlp.experts.down_proj", lo, lo + n_local)
                for e in range(n_local):
                    put(expert_param_name(m, "linear_fc1", e, grouped), gu[e])
                    put(expert_param_name(m, "linear_fc2", e, grouped), dn[e])
            else:
                put(m, st.get(h))
    if strict:
        missing = [k for k in params if k not in done]
        assert not missing, f"没有载入的参数：{missing[:10]}（共 {len(missing)}）"
    return chk if verify else done


def mtp_names(config):
    """MTP（一层）的 (Megatron 名, checkpoint 名, 变换)。checkpoint 里是 model.mtp_layers.0.*。"""
    out = []
    pm, ph = "mtp.layers.0.", "model.mtp_layers.0."
    for m, h in MTP_EXTRA:
        out.append((pm + m, ph + h, None))
    inner = pm + "mtp_model_layer."
    out.append((inner + "input_layernorm.weight", ph + "input_layernorm.weight", None))
    out.append((inner + "pre_mlp_layernorm.weight", ph + "post_attention_layernorm.weight", None))
    for m, h in ATTN + INDEXER:
        out.append((inner + "self_attention." + m, ph + h, None))
    for m, h in MOE_FIXED:
        out.append((inner + "mlp." + m, ph + h, None))
    out.append(
        (
            inner + "mlp.shared_experts.linear_fc1.weight",
            (ph + "mlp.shared_experts.gate_proj.weight", ph + "mlp.shared_experts.up_proj.weight"),
            "cat",
        )
    )
    out.append((inner, ph, "experts"))
    return out


def tensor_sha_ok(hf_dir: str, names=None) -> int:
    """按原始字节核 checkpoint 张量对 tensor_sha256.json（L0），返回核过的张量数。"""
    import hashlib

    man = json.load(open(os.path.join(hf_dir, "tensor_sha256.json")))
    st = Safetensors(hf_dir)
    n = 0
    for f in sorted({st.where[k] for k in (names or st.names())}):
        hl, hdr = st._header(f)
        with open(os.path.join(hf_dir, f), "rb") as fh:
            for name, meta in hdr.items():
                if name == "__metadata__" or (names is not None and name not in names):
                    continue
                if st.where.get(name) != f:  # 同名张量在别的文件里还有一份、index 指向那份时，只核 index 指向的那份
                    continue
                a, b = meta["data_offsets"]
                fh.seek(8 + hl + a)
                if hashlib.sha256(fh.read(b - a)).hexdigest() != man[name]:
                    raise ValueError(f"sha256 不对：{name}")
                n += 1
    return n


_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
