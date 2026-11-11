"""混元 4 TPU 侧几个脚本共用的 MaxText 辅助：按阶段建配置、取抽象参数树。只在 MaxText 环境里用（要 jax、flax、maxtext）。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for p in (REPO, HERE.parents[1] / "model"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hy4_map import STAGE_PARAMS, STAGES  # noqa: E402,F401

HY4_YML = Path(os.environ.get("HY4_YML") or HERE / "hy4-4layers.yml")  # T1 用 hy4-mini-t1.yml（HY4_YML 指过去）


def stage_overrides(stage: str) -> list[str]:
    """阶段 → 命令行覆盖（层数、MTP 层数）。"""
    if stage not in STAGES:
        raise SystemExit(f"阶段只能是 {sorted(STAGES)}，给的是 {stage}")
    layers, mtp = STAGES[stage]
    return [f"base_num_decoder_layers={layers}", f"mtp_num_layers={mtp}"]


def forced_overrides(yml: str | os.PathLike = HY4_YML) -> list[str]:
    """hy4-4layers.yml 里和模型 yml（hy4-preview.yml）重复、取值不同的键，要在命令行再给一遍才生效
    （MaxText 的合并顺序：base → 对齐 yml → 模型 yml → 命令行；混元 3 的 FORCED 同理）。层数与 MTP 由阶段覆盖给。"""
    import yaml
    from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
    ours = yaml.safe_load(Path(yml).read_text())
    model = yaml.safe_load((Path(MAXTEXT_CONFIGS_DIR) / "models" / f"{ours['model_name']}.yml").read_text())
    skip = {"base_num_decoder_layers", "mtp_num_layers"}
    out = []
    for k in sorted(set(ours) & set(model) - skip):
        if ours[k] != model[k]:
            if not isinstance(ours[k], (int, float, str, bool)):
                raise SystemExit(f"{k} 在两个 yml 里都有且取值不同，且不是标量，要手工处理")
            out.append(f"{k}={str(ours[k]).lower() if isinstance(ours[k], bool) else ours[k]}")
    return out


def make_config(stage: str, extra: list[str] | None = None, yml: str | os.PathLike = HY4_YML):
    """pyconfig.initialize(hy4-4layers.yml + 与模型 yml 冲突的键 + 阶段覆盖 + extra)。extra 是 key=value 列表，后写的覆盖先写的。"""
    from maxtext.configs import pyconfig
    argv = ["train", str(yml), *forced_overrides(yml), *stage_overrides(stage), *(extra or [])]
    return pyconfig.initialize(argv)


def effective_mismatch(config, yml: str | os.PathLike = HY4_YML) -> list:
    """对齐 yml 里每个键在 MaxText 实际生效的配置里是不是同一个值（枚举按 .value 比）。返回 [(键, yml 值, 生效值)]。"""
    import yaml
    ours = yaml.safe_load(Path(yml).read_text())
    bad = []
    for k, v in ours.items():
        if k in ("base_config", "base_num_decoder_layers", "mtp_num_layers"):
            continue
        got = getattr(config, k, None)
        got = getattr(got, "value", got)
        if isinstance(v, list):
            same = list(got or []) == v
        elif isinstance(v, float):
            same = got is not None and float(got) == v
        else:
            same = str(got) == str(v)
        if not same:
            bad.append((k, v, got))
    return bad


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


def nest(flat: dict[str, object], sep: str = "/") -> dict:
    """{'a/b/c': x} → {'a': {'b': {'c': x}}}。"""
    out: dict = {}
    for k, v in flat.items():
        cur = out
        *parents, last = k.split(sep)
        for p in parents:
            cur = cur.setdefault(p, {})
        cur[last] = v
    return out


def dash_key(slash_key: str) -> str:
    """'params/decoder/dense_layers/mlp/wo/kernel' → 分支转换用的 'params-decoder-dense_layers-mlp-wo-kernel'。"""
    return slash_key.replace("/", "-")
