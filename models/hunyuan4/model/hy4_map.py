"""混元 4（Hy4-preview）截层的 HF ↔ MaxText 映射：分阶段截层、名字映射、逐张量的正反变换。

正向（HF → MaxText）不自己写：直接用 maxtext 分支里的
`HY4_MAXTEXT_TO_HF_PARAM_MAPPING` / `HY4_MAXTEXT_TO_HF_PARAM_HOOK_FN`（MaxText 自己的转换就是用它们），
这里只补一件事——分支的映射要求 MoE 层数至少一整组（indexer_share_interval，4 层），
而对齐用的截层（1～3 层 MoE）全是不足一组、整个落在「余数块」`moe_layers_remainder` 里的情形，
模型本身建得出来、分支的映射会抛 NotImplementedError。做法：把 HF 配置按 Hy4 的规律补到一整组，
取分支的映射与变换，再把 `moe_layers-layers_j`（每个只含 1 层）改名成 `moe_layers_remainder-layers_j`、
去掉多出来的层。键名和变换函数仍然全部来自分支。

反向（MaxText → HF，用于 L0 逐位核对）分支没有实现（HOOK_FN 的 saving_to_hf 直接抛错），这里独立写：
转置、拆分拼回、RoPE 列去交错的逆置换。反向对上原件只证明「两个方向互逆」，证明不了映射本身对
（两边错成同一种查不出）；映射本身靠后面的逐层前向比。

阶段（AdamW 时期的分阶段计划；参数量为 MaxText 抽象模型实测，与 HF 头里数出来的一致）：

| 阶段 | 层 | MTP | 参数量 |
|---|---|---|---|
| A | 第 0 层（稠密） | 无 | 2,100,089,689 |
| B | 0–1（+ 第 1 层 MoE，独立 indexer） | 无 | 12,078,550,957 |
| C | 0–2（+ 第 2 层 MoE，第一个共享 indexer 的层） | 无 | 22,047,640,321 |
| D | 0–1 | 1 层 | 22,132,134,893 |
| D1 | 0 | 1 层 | 12,153,673,625 |
| D3 | 0–2 | 1 层 | 32,101,224,257 |
| 4L | 0–3 | 1 层 | 42,070,313,621 |
"""
from __future__ import annotations

import copy
import re

import numpy as np

STAGES = {"A": (1, 0), "B": (2, 0), "C": (3, 0), "D": (2, 1), "D1": (1, 1), "D3": (3, 1), "4L": (4, 1), "mini": (7, 1), "mid": (7, 1)}
STAGE_PARAMS = {"A": 2_100_089_689, "B": 12_078_550_957, "C": 22_047_640_321, "D": 22_132_134_893,
                "D1": 12_153_673_625, "D3": 32_101_224_257, "4L": 42_070_313_621,
                "mini": 427_783_489,  # mini：hy4-mini（MaxText model_name=hy4-mini），随机初始化，不是截层
                "mid": 5_189_678_849}  # mid：hy4-mid（MaxText model_name=hy4-mid），随机初始化；HF 张量形状数出
PER_LAYER_LISTS = ("mlp_layer_types", "indexer_types", "layer_types")


# 发布的检查点里存 FP32 的张量（2026-11-07 对截 4 层目录 135 个张量的 safetensors 头逐个核过，verify_ckpt.py names）：
# 主干每层的 iHC 参数、hc_head、路由偏置、sink。MTP 层的路由偏置与 sink 在检查点里是 BF16（和主干不同）。
_FP32_RE = re.compile(r"^model\.(hc_head\.|layers\.\d+\.(hc_attn_layer\.|hc_mlp_layer\.|mlp\.gate\.e_score_correction_bias$"
                      r"|self_attn\.learnable_sink_param$))")


def hf_dtype(name: str) -> str:
    """HF 检查点里这个张量的 dtype（bfloat16 / float32）。"""
    return "float32" if _FP32_RE.match(name) else "bfloat16"


# GPU golden（transformers 加载后的模块名）里参数的名字：前缀 hf.，另有几处 transformers 加载时改名
#（GPU ga-b 的 params_sha 与 modules.txt 对过阶段 A 的 31 个张量；MoE 与 MTP 的改名等阶段 B / D 的 golden 再核）
_GPU_RENAMES = ((".hc_attn_layer.hc_pre.hc_", ".attn_hc."), (".hc_mlp_layer.hc_pre.hc_", ".ffn_hc."),
                ("model.hc_head.hc_head_", "model.hc_head.hc_"), (".self_attn.linear_gate.", ".self_attn.gate_proj."),
                (".self_attn.learnable_sink_param", ".self_attn.sinks"))


def gpu_name(hf_name: str) -> str:
    """safetensors 里的 HF 名字 → GPU golden 抓数里的参数名（hf.<transformers 模块名>）。"""
    for a, b in _GPU_RENAMES:
        hf_name = hf_name.replace(a, b)
    return "hf." + hf_name


# GPU golden 的参数清单 params_sha/rank<r>.<tag>.json 记的是每个 rank 本地的 FSDP2 分片（torch.chunk 沿第 0 维、每片
# ceil(N/8) 行）的 FP32 字节的 sha256，形状一栏写完整形状；没被切分的参数（GPU 上留 FP32 的 iHC、hc_head、sink、indexer 的
# weights_proj / k_norm）每个 rank 记完整张量。阶段 A 的 31 个张量 × 8 个 rank 全部按这个规则从 HF 原件复算对上（2026-11-07）
_GPU_REPLICATED = re.compile(r"(\.attn_hc\.|\.ffn_hc\.|^hf\.model\.hc_head\.|\.self_attn\.sinks$|"
                             r"\.self_attn\.indexer\.(weights_proj|k_norm)\.)")


def gpu_rank_pieces(gpu_param_name: str, x, world: int = 8):
    """GPU golden 清单里第 r 个 rank 记的那一片：不切分的参数是整个张量，其余是 FSDP2 沿第 0 维的第 r 片。"""
    if _GPU_REPLICATED.search(gpu_param_name):
        return [x] * world
    n = -(-x.shape[0] // world)
    return [x[r * n:(r + 1) * n] for r in range(world)]


def stage_hf_config(hf_cfg: dict, layers: int) -> dict:
    """HF config.json 截到前 `layers` 层：num_hidden_layers 与每层一项的列表一起截。"""
    if layers > hf_cfg["num_hidden_layers"]:
        raise ValueError(f"HF 只有 {hf_cfg['num_hidden_layers']} 层，要 {layers} 层")
    c = copy.deepcopy(hf_cfg)
    c["num_hidden_layers"] = layers
    for k in PER_LAYER_LISTS:
        if isinstance(c.get(k), list):
            c[k] = c[k][:layers]
    return c


def stage_tensor_names(names, layers: int, mtp: int) -> list[str]:
    """HF 张量名里属于这个阶段的：前 `layers` 层、前 `mtp` 个 MTP 层、以及非层张量。"""
    out = []
    for n in names:
        m = re.match(r"model\.layers\.(\d+)\.", n)
        if m:
            if int(m.group(1)) < layers:
                out.append(n)
            continue
        m = re.match(r"model\.mtp_layers\.(\d+)\.", n)
        if m:
            if int(m.group(1)) < mtp:
                out.append(n)
            continue
        out.append(n)
    return sorted(out)


def _pad_to_one_group(hf_cfg: dict, interval: int) -> dict:
    """MoE 层数不足一组时，按 Hy4 的规律（每组首层 full、其余 shared）补齐到一组。"""
    num_dense = sum(1 for t in hf_cfg["mlp_layer_types"] if t == "dense")
    c = copy.deepcopy(hf_cfg)
    n = num_dense + interval
    c["num_hidden_layers"] = n
    c["mlp_layer_types"] = ["dense"] * num_dense + ["sparse"] * interval
    c["indexer_types"] = ["full"] * num_dense + ["full"] + ["shared"] * (interval - 1)
    for k in PER_LAYER_LISTS:
        if k not in ("mlp_layer_types", "indexer_types") and isinstance(c.get(k), list):
            c[k] = (c[k] + [c[k][-1]] * n)[:n]
    return c


def mapping_and_hooks(hf_cfg: dict, mt_cfg) -> tuple[dict, dict]:
    """分支的 Hy4 映射与变换（scan_layers=True），补上「MoE 不足一组」的情形。"""
    from maxtext.checkpoint_conversion.utils import param_mapping as pm

    num_dense = sum(1 for t in hf_cfg["mlp_layer_types"] if t == "dense")
    num_moe = hf_cfg["num_hidden_layers"] - num_dense
    interval = getattr(mt_cfg, "indexer_share_interval", 1)
    index_cache = getattr(mt_cfg, "use_indexer", False) and interval > 1
    if not index_cache or num_moe >= interval:
        return (pm.HY4_MAXTEXT_TO_HF_PARAM_MAPPING(hf_cfg, mt_cfg, True),
                pm.HY4_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_cfg, mt_cfg, True))
    # 不足一组：真实层的 indexer 规律要和补齐后的一致（第一个 MoE 层 full，其余 shared）
    want = ["full"] * num_dense + (["full"] + ["shared"] * (interval - 1))[:num_moe]
    if list(hf_cfg["indexer_types"][: num_dense + num_moe]) != want:
        raise ValueError(f"indexer_types {hf_cfg['indexer_types']} 不是 Hy4 的分组规律 {want}")
    padded = _pad_to_one_group(hf_cfg, interval)
    mp = pm.HY4_MAXTEXT_TO_HF_PARAM_MAPPING(padded, mt_cfg, True)
    hk = pm.HY4_MAXTEXT_TO_HF_PARAM_HOOK_FN(padded, mt_cfg, True)
    pat = re.compile(r"^(params|MoEBiasVar)-decoder-moe_layers-layers_(\d+)-(.*)$")
    out_m, out_h = {}, {}
    for k, v in mp.items():
        m = pat.match(k)
        if not m:
            out_m[k] = v
            if k in hk:
                out_h[k] = hk[k]
            continue
        j = int(m.group(2))
        if j >= num_moe:
            continue
        assert isinstance(v, list) and len(v) == 1, (k, v)   # 补齐后正好一组：每个子层的栈只有 1 层
        nk = f"{m.group(1)}-decoder-moe_layers_remainder-layers_{j}-{m.group(3)}"
        out_m[nk] = v[0]
        if k in hk:
            out_h[nk] = hk[k]
    return out_m, out_h


# ------------------------------------------------------------------ 反向（MaxText 一层的切片 → HF 张量）

def hf_kind(mt_key: str) -> str:
    """按 MaxText 键名的后缀判断正向做了哪种变换（与分支 HOOK_FN 的 per_layer 表一一对应）。"""
    tail = mt_key.split("-", 1)[1]
    rope_keys = ("self_attention-wq_b-kernel", "self_attention-wkv_a-kernel",
                 "self_attention-indexer-wq_b-kernel", "self_attention-indexer-wk-kernel")
    if tail.endswith(("indexer-k_norm-scale", "indexer-k_norm-bias")):
        return "rope_only"
    if tail.endswith(rope_keys):
        return "kernel_rope"
    for part in ("pre_alpha_scale", "post_alpha_scale", "pre_alpha", "post_alpha", "pre_beta", "post_beta"):
        if tail.endswith("-" + part):
            return "hc_" + part
    if tail.endswith("MoeBlock_0-wi_0"):
        return "experts_in0"
    if tail.endswith("MoeBlock_0-wi_1"):
        return "experts_in1"
    if tail.endswith("MoeBlock_0-wo"):
        return "experts_out"
    if tail.endswith("-kernel") or tail == "decoder-hc_head-hc_fn":
        return "kernel"
    return "identity"


def deinterleave_perm(d: int, rope: int) -> np.ndarray:
    """正向：最后 rope 个位置从交错 (2i, 2i+1) 改成 [偶, 奇]；返回 index 置换（新位置取旧位置 perm[k]）。"""
    return np.concatenate([np.arange(d - rope), d - rope + np.arange(0, rope, 2), d - rope + np.arange(1, rope, 2)])


def kernel_in_dim(mt_key: str, slice_shape: tuple) -> int:
    """正向 x.T.reshape(target)：x 是 HF 的 [out, in]。in 是 target 的第一维；注意力输出投影例外（[H, v, D]，in = H·v）。"""
    if mt_key.endswith("self_attention-out-kernel"):
        return int(np.prod(slice_shape[:-1]))
    return int(slice_shape[0])


def undo_rope(y: np.ndarray, rope: int, deinterleave: bool) -> np.ndarray:
    """去交错的逆：最后一维的最后 rope 个位置从 [偶, 奇] 放回交错 (2i, 2i+1)。"""
    if not deinterleave:
        return y
    perm = deinterleave_perm(y.shape[-1], rope)
    return y[..., np.argsort(perm)]


def kernel_to_hf(mt_key: str, y: np.ndarray, rope: int, deinterleave: bool) -> np.ndarray:
    """MaxText 一层的 kernel 切片 → HF [out, in]。"""
    kind = hf_kind(mt_key)
    if kind == "kernel_rope":
        y = undo_rope(y, rope, deinterleave)
    k_in = kernel_in_dim(mt_key, y.shape)
    return y.reshape(k_in, -1).T


def piece_to_hf(mt_key: str, y: np.ndarray, rope: int, deinterleave: bool, hf_shape) -> np.ndarray:
    """单件变换的反向（不含 hc 拆分 / 专家 gate|up 拆分，那两类由 assemble_hf 拼回）。"""
    kind = hf_kind(mt_key)
    if kind in ("kernel", "kernel_rope"):
        x = kernel_to_hf(mt_key, y, rope, deinterleave)
    elif kind == "rope_only":
        x = undo_rope(y, rope, deinterleave)
    elif kind == "experts_out":
        x = np.transpose(y, (0, 2, 1))
    elif kind == "identity":
        x = y
    else:
        raise ValueError(f"{mt_key}：{kind} 要用 assemble_hf 拼")
    return x.reshape(hf_shape)


def assemble_hf(pieces: dict[str, np.ndarray], hf_shape, rope: int, deinterleave: bool) -> np.ndarray:
    """一个 HF 张量的全部 MaxText 切片（{mt_key: 一层的切片}）→ HF 张量。

    - 单件：直接反变换；
    - hc 的 fn / base / scale：pre / post 两半按行拼回（fn 两半各自转置回来）；
    - 专家 gate_up_proj：wi_0 / wi_1 各自 [E, D, I] → [E, I, D]，沿第 1 维拼成 [E, 2I, D]。
    """
    kinds = {hf_kind(k): (k, v) for k, v in pieces.items()}
    if len(pieces) == 1:
        (k, v), = pieces.items()
        return piece_to_hf(k, v, rope, deinterleave, hf_shape)
    if set(kinds) == {"experts_in0", "experts_in1"}:
        a = np.transpose(kinds["experts_in0"][1], (0, 2, 1))
        b = np.transpose(kinds["experts_in1"][1], (0, 2, 1))
        return np.concatenate([a, b], axis=1).reshape(hf_shape)
    for pre, post, transpose in (("hc_pre_alpha", "hc_post_alpha", True), ("hc_pre_beta", "hc_post_beta", False),
                                 ("hc_pre_alpha_scale", "hc_post_alpha_scale", False)):
        if set(kinds) == {pre, post}:
            rest = tuple(hf_shape[1:])
            halves = []
            for kk in (pre, post):
                y = kinds[kk][1]
                if transpose:          # 正向 x[lo:hi].T.reshape(target)：x[lo:hi] 是 [r, *rest]，r = target 的最后一维
                    r = y.shape[-1]
                    halves.append(y.reshape(-1, r).T.reshape((r, *rest)))
                else:                  # 正向 x[lo:hi].reshape(target)
                    halves.append(y.reshape((-1, *rest)))
            return np.concatenate(halves, axis=0).reshape(hf_shape)
    raise ValueError(f"不认识的拼法：{sorted(kinds)}")


def iter_hf_sources(mapping: dict):
    """展开映射：逐个产出 (mt_key, 栈里的层号或 None, hf 名)。scan 栈的值是列表，层号是它在列表里的位置。"""
    for mt_key, v in mapping.items():
        if isinstance(v, list):
            for i, h in enumerate(v):
                if isinstance(h, list):
                    raise NotImplementedError(f"{mt_key}：嵌套列表（专家分开存）Hy4 用不到")
                yield mt_key, i, h
        else:
            yield mt_key, None, v


def stack_axis(mt_key: str, ndim_full: int, param_scan_axis: int) -> int:
    """scan 栈的层轴：与分支 _build_single_axis_stacked_tensor 一致（MoEBiasVar 在第 0 维，其余在 param_scan_axis，1 维的在 0）。"""
    if mt_key.startswith("MoEBiasVar"):
        return 0
    return param_scan_axis if ndim_full > param_scan_axis else 0
