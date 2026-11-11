"""TPU 侧几个脚本共用的 MaxText 辅助：建配置、取抽象参数树。只在 MaxText 环境里用（要 jax、flax、maxtext）。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

HY3_YML = HERE / "hy3-4layers.yml"
NAME_MAP = HERE.parents[1] / "model" / "name_map.yaml"

# 模型 yml（hunyuan3-smoke.yml）里也有、必须在命令行覆盖才生效的键（MaxText 合并顺序：base → 模型 yml → 命令行）
FORCED = ["mtp_num_layers=0"]


def make_config(extra: list[str] | None = None, yml: str | os.PathLike = HY3_YML):
    """pyconfig.initialize(hy3-4layers.yml + FORCED + extra)。extra 是 key=value 列表，后写的覆盖先写的。"""
    from maxtext.configs import pyconfig
    argv = ["train", str(yml), *FORCED, *(extra or [])]
    return pyconfig.initialize(argv)


def make_mesh(config):
    import jax
    from maxtext.utils import maxtext_utils
    return jax.sharding.Mesh(maxtext_utils.create_device_mesh(config), config.mesh_axes)


def variable_paths(state) -> dict[str, object]:
    """NNX 状态 → {'<集合>/<模块路径>': 值}，集合名按变量类型（params、MoEBiasVar…），跳过 RNG。"""
    from flax import nnx
    from flax.nnx import variablelib
    out = {}
    for path, v in nnx.to_flat_state(state):
        if isinstance(v, nnx.RngState):
            continue
        col = variablelib.variable_name_from_type(type(v), allow_register=True)
        val = v.get_value() if hasattr(v, "get_value") else v
        out[col + "/" + "/".join(str(p) for p in path)] = val
    return out


def abstract_param_tree(config) -> dict[str, object]:
    """抽象模型（不分配内存）的全部非 RNG 变量：{'<集合>/<路径>': ShapeDtypeStruct}。"""
    from flax import nnx
    from maxtext.utils import model_creation_utils
    _, m = model_creation_utils.create_nnx_abstract_model(config, make_mesh(config))
    return variable_paths(nnx.state(m))


def nest(flat: dict[str, object]) -> dict:
    """{'a/b/c': x} → {'a': {'b': {'c': x}}}。"""
    out: dict = {}
    for k, v in flat.items():
        cur = out
        *parents, last = k.split("/")
        for p in parents:
            cur = cur.setdefault(p, {})
        cur[last] = v
    return out


def name_map_for(config, strict: bool = False):
    """name_map.yaml，维度换成 MaxText 配置里实际生效的值。strict=True 时维度必须与 yaml 一致（正式运行用）。"""
    from janus.dump.name_map import NameMap
    nm = NameMap.load(NAME_MAP)
    dims = {"layers": config.num_decoder_layers, "first_dense": config.first_num_dense_layers, "hidden": config.emb_dim,
            "heads": config.num_query_heads, "kv_heads": config.num_kv_heads, "head_dim": config.head_dim,
            "ffn": config.mlp_dim, "moe_ffn": config.moe_mlp_dim, "experts": config.num_experts,
            "topk": config.num_experts_per_tok, "vocab": config.vocab_size}
    diff = {k: (nm.dims.get(k), v) for k, v in dims.items() if nm.dims.get(k) != v}
    if not diff:
        return nm
    if strict:
        raise SystemExit(f"MaxText 配置与 name_map.yaml 的维度不一致：{diff}")
    ep = nm.ep
    return NameMap({**nm.spec, "dims": {**nm.dims, **dims},
                    "megatron": {**nm.spec.get("megatron", {}), "local_experts": max(1, config.num_experts // ep)}})
