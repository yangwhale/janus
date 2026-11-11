"""TPU 侧抓数（MaxText）。不改 MaxText 源码：训练循环由 models/<模型>/frameworks/maxtext/train_*.py 自己写，调 MaxText 的
setup_train_loop / train_step；这里提供三样东西。

1. 状态类（零扰动）：参数（FP32 主权重）、优化器动量、路由偏置都在训练状态里，训练步与训练步之间从设备读出，
   不碰被编译的程序。逐步标量里的 loss、梯度范数、参数范数是 MaxText 训练步本来就算的输出；学习率、权重衰减按配置
   与调度函数在主机上算；每层路由偏置的均值 / 标准差在步与步之间从状态读出来算。
2. 图内抓数（加钩子的训练步，只在抓数步用）：
   - 按模块实例的路径给模块打标记（tag_model），给几个模块类的 __call__ 套一层壳：钩子打开时把输入 / 输出
     用 nnx 的 sow 存成 Intermediate，随 MaxText 原有的 Intermediate 通道（loss_fn 里 nnx.pop）带出来；
     钩子关着时壳只是原样转调，程序不变。
   - 只抓设备 0 的那几行（全局 batch 前 per_device_batch_size 行），先在图里切片再输出。
   - 参数梯度：把 train 模块里的 jax 换成一个只拦 value_and_grad 的代理，拿到 MaxText 自己算出的未裁剪梯度
     （与 GPU 的 main_grad 同义：已归约、裁剪之前）。
   - 这些额外输出放进训练步返回的 metrics["janus"]，训练状态的结构不变（out_shardings 不用改）。
   按 docs/reference/dump.md：生产图里只抓本来就要写回 HBM 的值；每种配置先过「开不开钩子逐位相同」——
   train_*.py 在每个抓数步把同一个输入状态分别喂给加钩子与不加钩子的程序，逐位比输出的状态与 loss（hooks.selfcheck）。
3. 写盘：按 docs/reference/dump-format.md 写运行目录。参数类张量换成 Megatron 名字与布局（models/<模型>/model/name_map.yaml），
   和 golden 同名可比；激活写规范名，布局转成 GPU 的 [S, B, …]。

需要 jax / flax / maxtext 的部分都在函数里 import，本模块本身可以在没有 jax 的环境里 import（单测用）。
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import threading
from pathlib import Path

import numpy as np

from janus.dump.name_map import NameMap
from janus.store import RunWriter

# ------------------------------------------------------------------------------------------------ 状态树里的路径
# name_map 里 MaxText 叶子路径写成 <集合>/<模块路径>（params/…、MoEBiasVar/…）；训练状态（TrainStateNNX）里多一层：
#   参数     params/model/<模块路径>          MoEBiasVar/model/<模块路径>
#   Adam     OptVariable/optimizer/opt_state/0/mu/<模块路径>   …/nu/…
# （CPU 上从 setup_train_loop 建出的状态读出，见 models/hunyuan3/frameworks/maxtext/README.md）
ADAM_PREFIX = "OptVariable/optimizer/opt_state/0/"


def state_key(path: str, kind: str = "param") -> str:
    col, rest = path.split("/", 1)
    if kind == "param":
        return f"{col}/model/{rest}"
    if kind in ("mu", "nu"):
        if col != "params":
            raise KeyError(f"{path} 不在优化器里")
        return f"{ADAM_PREFIX}{kind}/{rest}"
    raise ValueError(kind)


# ------------------------------------------------------------------------------------------------ 主机侧小工具
def sha256(a: np.ndarray) -> str:
    """原始字节的 sha256。按 uint8 视图喂给 hashlib，不另拷一份字节（大的专家叶子一个十几 GB）；结果与 tobytes() 相同。"""
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.reshape(-1).view(np.uint8)).hexdigest()


# 状态清单里每个叶子归哪一类（R07 / R15 / R18 按类报）。顺序即匹配顺序。
STATE_KINDS = (
    ("param", re.compile(r"^params/")),
    ("bias", re.compile(r"^MoEBiasVar/")),
    ("mu", re.compile(r"/opt_state/\d+/mu/")),
    ("nu", re.compile(r"/opt_state/\d+/nu/")),
    ("count", re.compile(r"(/count|/step)$")),
)


def state_kind(path: str) -> str:
    return next((k for k, pat in STATE_KINDS if pat.search(path)), "other")


def state_manifest(flat_state: dict) -> dict:
    """训练状态的全量清单 {状态路径: {sha256, shape, dtype, kind}}：每个数组叶子按**存储的 dtype** 取原始字节算 sha256，
    不转 BF16、不换名字和布局（FP32 主权重、两个 Adam 动量、路由偏置、步数计数器全在里面）。
    日常回归是 TPU 对 TPU，用 MaxText 原生路径就够；对 GPU 比的 BF16 清单见 write_param_sha。
    多进程时所有进程都要以同样顺序调用（to_host 是集合操作）。"""
    out = {}
    for k in sorted(flat_state):
        v = flat_state[k]
        if not hasattr(v, "shape") or not hasattr(v, "dtype"):
            continue
        a = to_host(v)
        out[k] = {"sha256": sha256(a), "shape": list(a.shape), "dtype": str(a.dtype), "kind": state_kind(k)}
        del a
    return out


def manifest_digest(manifest: dict, kind: str | None = None) -> str:
    """清单的总摘要：按路径排序拼「路径 + sha256」再算 sha256（kind 给定时只取这一类）。报告首屏用。"""
    h = hashlib.sha256()
    for k in sorted(manifest):
        if kind is None or manifest[k].get("kind") == kind:
            h.update(k.encode())
            h.update(manifest[k]["sha256"].encode())
    return h.hexdigest()


def to_bf16(a: np.ndarray) -> np.ndarray:
    import ml_dtypes
    return np.asarray(a).astype(ml_dtypes.bfloat16)


def to_host(x) -> np.ndarray:
    """全局 jax.Array → 主机 numpy（多进程时各进程都拿到完整数组；必须所有进程以同样顺序调用）。"""
    if isinstance(x, np.ndarray):
        return x
    import jax
    if getattr(x, "is_fully_addressable", True):
        return np.asarray(jax.device_get(x))
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


def seq_first(x: np.ndarray, row: int = 0) -> np.ndarray:
    """[B, S, …] 取第 row 行 → GPU 布局 [S, 1, …]。"""
    return np.ascontiguousarray(np.swapaxes(x[row: row + 1], 0, 1))


# ------------------------------------------------------------------------------------------------ 图内抓数
# 模块实例路径（相对 TrainState.model，'/' 连接）→ 标记。标记决定壳里抓什么（见 _CAPTURE_IO）。
TAG_RULES = [
    (r"^token_embedder$", "embed"),
    (r"^decoder/(dense_layers|moe_layers)$", "layer"),
    (r"^decoder/(dense_layers|moe_layers)/self_attention/query$", "q"),
    (r"^decoder/(dense_layers|moe_layers)/self_attention/key$", "k"),
    (r"^decoder/(dense_layers|moe_layers)/self_attention/value$", "v"),
    (r"^decoder/(dense_layers|moe_layers)/self_attention/query_norm$", "q_norm"),
    (r"^decoder/(dense_layers|moe_layers)/self_attention/key_norm$", "k_norm"),
    (r"^decoder/(dense_layers|moe_layers)/self_attention/out$", "o"),
    (r"^decoder/dense_layers/mlp$", "mlp"),
    (r"^decoder/dense_layers/mlp/wi_0$", "fc1_gate"),
    (r"^decoder/dense_layers/mlp/wi_1$", "fc1_up"),
    (r"^decoder/moe_layers/\w+MoeBlock_0$", "moe"),
    (r"^decoder/moe_layers/\w+MoeBlock_0/MoeBlock_0$", "routed"),
    (r"^decoder/moe_layers/\w+MoeBlock_0/shared_experts$", "shared"),
    (r"^decoder/moe_layers/\w+MoeBlock_0/MoeBlock_0/gate$", "gate"),
    (r"^decoder/decoder_norm$", "final_norm"),
    (r"^decoder/logits_dense$", "logits"),
]
# 标记 → (抓输入吗, 抓第几个输出：None 不抓 / 0 第一个 / "all" 整个)
_CAPTURE_IO = {
    "embed": (False, "all"), "layer": (True, 0), "q": (False, "all"), "k": (False, "all"), "v": (False, "all"),
    "q_norm": (False, "all"), "k_norm": (False, "all"), "o": (True, "all"), "mlp": (False, "all"),
    "fc1_gate": (False, "all"), "fc1_up": (False, "all"), "moe": (False, 0), "routed": (False, 0),
    "shared": (False, "all"), "gate": (False, "both"), "final_norm": (False, "all"), "logits": (False, "all"),
}
# 混元 4（deepseek 解码块 + MLA + iHC + hc_head）：模块路径与混元 3 不同，另一张表，tag_model(model, HY4_TAG_RULES) 时用。
# 层输出是 iHC 的 4 路流 [B, S, 4, D]；MLA 整个注意力模块的输出、门控投影、hc_head 各抓一处。混元 3 的表不动（它的抓数档位已逐档自证）。
HY4_TAG_RULES = [
    (r"^token_embedder$", "embed"),
    (r"^decoder/dense_layers$", "layer"),
    (r"^decoder/dense_layers/pre_self_attention_layer_norm$", "norm_io"),
    (r"^decoder/dense_layers/post_self_attention_layer_norm$", "norm_io"),
    (r"^decoder/dense_layers/mhc_(attention|mlp)$", "ihc"),
    (r"^decoder/dense_layers/self_attention$", "attn"),
    (r"^decoder/dense_layers/self_attention/gate_proj$", "norm_out"),
    (r"^decoder/dense_layers/self_attention/out$", "norm_out"),
    (r"^decoder/dense_layers/mlp$", "mlp"),
    (r"^decoder/hc_head$", "norm_out"),
    (r"^decoder/decoder_norm$", "final_norm"),
    (r"^decoder/logits_dense$", "logits"),
]
# norm_io：两个层内 RMSNorm 连输入一起抓（它们的输入就是 iHC 混合后的那一路，对 HF 的 attn_hc.out / ffn_hc.out）；
# ihc：不抓，只作注入点（iHC 的输入以 x= 传）；ihc_out：同样作注入点，另抓第一个输出（写回后的 4 路流）
_CAPTURE_IO.update({"norm_out": (False, "all"), "attn": (False, 0), "norm_io": (True, "all"), "ihc": (False, None),
                    "ihc_out": (False, 0)})
SOW_PREFIX = "janus__"

# 抓数档位：每档抓哪些标记（None = 全部）、要不要逐 token loss、要不要参数梯度。
# 「开不开钩子逐位相同」要逐档在 TPU 上证明（docs/reference/dump.md 第 2 条）；train_hy3.py --probe-tiers 一次测完各档。
TIERS = {
    "full": {"tags": None, "xent": True, "grads": True},
    "boundary": {"tags": {"embed", "layer", "final_norm", "logits"}, "xent": False, "grads": True},
    "router": {"tags": {"gate"}, "xent": False, "grads": False},
    "grads": {"tags": set(), "xent": False, "grads": True},
    "none": {"tags": set(), "xent": False, "grads": False},
}


class _CaptureState(threading.local):
    def __init__(self):
        self.active = False
        self.rows = 1
        self.tags = None          # None = 全部标记都抓
        self.xent = True
        self.current_gate = None
        self.box: dict = {}
        self.inject: dict = {}    # 逐算子隔离回放：{模块路径: 数组}，这个模块的输入换成给定的数组（只在 capturing(inject=…) 里）
        self.inject_out: dict = {}  # 同上，但换这个模块的输出（元组时换第一个）：只测下游的算术时用（如 iHC 写回吃参照的分支输出）


_STATE = _CaptureState()


def _inject_args(module, tag, args, kwargs):
    """capturing(inject=…) 时，把打了标记的模块的输入换成给定数组（按原输入的 dtype 与形状）；MLA 的 q / kv 两个输入一起换；
    输入以关键字 x= 传的（iHC）换 kwargs["x"]。返回 (args, kwargs)。"""
    path = getattr(module, "_janus_path", None)
    if not _STATE.inject or path not in _STATE.inject:
        return args, kwargs
    import jax.numpy as jnp
    if "x" in kwargs:
        x0 = kwargs["x"]
        return args, {**kwargs, "x": jnp.asarray(_STATE.inject[path], dtype=x0.dtype).reshape(x0.shape)}
    # 输入以关键字传的：iHC / mHC 包着的分支这样调（MLP：inputs=；注意力：inputs_q= 与 inputs_kv= 一起换）
    keys = [k for k in ("inputs", "inputs_q", "inputs_kv") if k in kwargs and hasattr(kwargs[k], "shape")]
    if keys and not args:
        x0 = kwargs[keys[0]]
        new = jnp.asarray(_STATE.inject[path], dtype=x0.dtype).reshape(x0.shape)
        return args, {**kwargs, **{k: new for k in keys}}
    if not args:
        raise ValueError(f"inject：{path} 的输入既不是位置参数也不是已知关键字，没法换（kwargs: {sorted(kwargs)}）")
    x0 = _first_array(args[0])
    new = jnp.asarray(_STATE.inject[path], dtype=x0.dtype).reshape(x0.shape)
    n = 2 if tag == "attn" and len(args) > 1 else 1
    return (new,) * n + tuple(args[n:]), kwargs


def _inject_output(module, out):
    """capturing(inject_out=…) 时，把打了标记的模块的输出换成给定数组（按原输出的 dtype 与形状；元组只换第一个）。"""
    path = getattr(module, "_janus_path", None)
    if not _STATE.inject_out or path not in _STATE.inject_out:
        return out
    import jax.numpy as jnp
    first = _first_array(out)
    new = jnp.asarray(_STATE.inject_out[path], dtype=first.dtype).reshape(first.shape)
    return (new, *out[1:]) if isinstance(out, tuple) else new


_INSTALLED = False


def _first_array(x):
    if isinstance(x, (tuple, list)):
        return x[0] if x else None
    return x


def _sow(module, name: str, x):
    from flax import nnx
    if x is None or not hasattr(x, "shape") or x.ndim == 0:
        return
    module.sow(nnx.Intermediate, SOW_PREFIX + name, x[: _STATE.rows])


def _wrap(cls, method: str = "__call__"):
    orig = getattr(cls, method)
    if getattr(orig, "_janus_wrapped", False):
        return

    @functools.wraps(orig)
    def shell(self, *args, **kwargs):
        tag = getattr(self, "_janus_tag", None)
        if not _STATE.active or tag is None or (_STATE.tags is not None and tag not in _STATE.tags):
            return orig(self, *args, **kwargs)
        args, kwargs = _inject_args(self, tag, args, kwargs)
        cap_in, cap_out = _CAPTURE_IO[tag]
        if cap_in and args:
            _sow(self, "in", _first_array(args[0]))
        if tag == "gate":
            prev, _STATE.current_gate = _STATE.current_gate, self
            try:
                out = orig(self, *args, **kwargs)
            finally:
                _STATE.current_gate = prev
            biased, pre_bias = out
            _sow(self, "biased", biased)
            _sow(self, "scores", pre_bias)
            return out
        out = _inject_output(self, orig(self, *args, **kwargs))
        if cap_out == "all":
            _sow(self, "out", _first_array(out) if isinstance(out, tuple) and tag not in ("q", "k", "v") else out)
        elif cap_out == 0:
            _sow(self, "out", _first_array(out))
        return out

    shell._janus_wrapped = True
    setattr(cls, method, shell)


def _wrap_attention_with_norm(cls):
    orig = cls.apply_attention_with_norm
    if getattr(orig, "_janus_wrapped", False):
        return

    @functools.wraps(orig)
    def shell(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        if (_STATE.active and getattr(self, "_janus_tag", None) == "layer"
                and (_STATE.tags is None or "attn_norm" in _STATE.tags)):
            hidden, resid = out[0], out[1]
            _sow(self, "post_attn_norm", hidden)     # post_self_attention_layer_norm 的输出
            _sow(self, "attn_resid", resid)          # 层输入 + 注意力输出
        return out

    shell._janus_wrapped = True
    cls.apply_attention_with_norm = shell


def install_capture():
    """给 MaxText 的几个模块类套壳（幂等）。必须在第一次 trace 训练步之前调用；钩子关着时壳原样转调。"""
    global _INSTALLED
    if _INSTALLED:
        return
    import importlib

    from maxtext.layers import embeddings, linears, moe, normalizations
    from maxtext.utils import max_utils
    for cls in (linears.DenseGeneral, linears.MlpBlock, normalizations.RMSNorm, embeddings.Embed, moe.RoutedAndSharedMoE,
                moe.RoutedMoE, moe.GateLogit):
        _wrap(cls)
    # 模型各自的层类：哪个分支有哪个就包哪个（混元 3 在 hunyuan 分支，混元 4 的 deepseek / MLA / hc_head 在 Hy4 分支）
    for mod, names in (("maxtext.models.hunyuan3", ("Hunyuan3DenseLayer", "Hunyuan3MoELayer")),
                       ("maxtext.models.deepseek", ("DeepSeekDenseLayer", "DeepSeekMoELayer")),
                       ("maxtext.layers.attention_mla", ("MLA",)),
                       ("maxtext.layers.mhc", ("DeepSeek4HyperHead", "IdentityHyperConnections"))):
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        for n in names:
            if hasattr(m, n):
                _wrap(getattr(m, n))
    try:
        from maxtext.models import qwen3
        _wrap_attention_with_norm(qwen3.AttentionWithNorm)
    except (ImportError, AttributeError):
        pass

    # 路由矩阵乘的原始输出（sigmoid 之前）：GateLogit 里是 linears._convert_to_activation_function(score_func)(output)，
    # 只在 GateLogit 的壳里（current_gate 非空）才在激活函数前多 sow 一次
    orig_conv = linears._convert_to_activation_function

    def conv(fn_or_string):
        f = orig_conv(fn_or_string)
        gate = _STATE.current_gate
        if gate is None:
            return f

        def g(x):
            _sow(gate, "logits", x)
            return f(x)
        return g
    linears._convert_to_activation_function = conv

    # 逐 token loss：train.loss_fn 调 max_utils.cross_entropy_with_logits，先存进 box，loss_fn 的壳再放进 aux
    orig_xent = max_utils.cross_entropy_with_logits

    def xent(*args, **kwargs):
        out = orig_xent(*args, **kwargs)
        if _STATE.active and _STATE.xent:
            _STATE.box["xent"] = out[0][: _STATE.rows]
        return out
    max_utils.cross_entropy_with_logits = xent
    _INSTALLED = True


class capturing:
    """上下文：在其中 trace 的前向会把打了标记的模块的输入 / 输出 sow 出来（rows：只留 batch 前几行）。
    加钩子的训练步与诊断用的单独前向（probe_forward.py）都用它。inject={模块路径: 数组} 时，这些模块的输入换成给定数组
    （逐算子隔离回放：每个算子吃参照一侧抓到的输入，误差不沿层累积）；inject_out 同样，换的是输出。"""

    def __init__(self, rows: int = 1, tags=None, xent: bool = True, inject=None, inject_out=None):
        self.rows, self.tags, self.xent, self.inject = rows, tags, xent, dict(inject or {})
        self.inject_out = dict(inject_out or {})

    def __enter__(self):
        install_capture()
        self.prev = (_STATE.active, _STATE.rows, _STATE.box, _STATE.tags, _STATE.xent, _STATE.inject, _STATE.inject_out)
        _STATE.active, _STATE.rows, _STATE.box, _STATE.tags, _STATE.xent = True, self.rows, {}, self.tags, self.xent
        _STATE.inject, _STATE.inject_out = self.inject, self.inject_out
        return _STATE.box

    def __exit__(self, *exc):
        (_STATE.active, _STATE.rows, _STATE.box, _STATE.tags, _STATE.xent, _STATE.inject,
         _STATE.inject_out) = self.prev
        return False


def collect_sown(intermediates) -> dict:
    """Intermediate 树 → {'<模块路径>/janus__<点>': 数组}。"""
    act = {}
    for path, val in _flat(intermediates):
        if path and str(path[-1]).startswith(SOW_PREFIX):
            act["/".join(str(p) for p in path)] = val[0] if isinstance(val, (tuple, list)) else val
    return act


def tag_model(model, rules=None) -> dict[str, str]:
    """按实例路径给模块打标记（_janus_tag，静态属性，进 graphdef）。返回 {路径: 标记}。rules 默认 TAG_RULES（混元 3），
    混元 4 传 HY4_TAG_RULES。"""
    from flax import nnx
    tagged = {}
    for path, node in nnx.iter_graph(model):
        if not isinstance(node, nnx.Module):
            continue
        p = "/".join(str(x) for x in path)
        for pat, tag in (TAG_RULES if rules is None else rules):
            if re.match(pat, p):
                object.__setattr__(node, "_janus_tag", tag)
                object.__setattr__(node, "_janus_path", p)
                tagged[p] = tag
                break
    return tagged


class _JaxProxy:
    """train 模块里的 jax：只拦 value_and_grad，记下它的输出；其余属性原样转给真 jax。"""

    def __init__(self, real, box):
        self._real, self._box = real, box

    def __getattr__(self, name):
        return getattr(self._real, name)

    def value_and_grad(self, fun, *a, **kw):
        g = self._real.value_and_grad(fun, *a, **kw)

        def wrapped(*args, **kwargs):
            out = g(*args, **kwargs)
            self._box["vg"] = out
            return out
        return wrapped


def make_hooked_train_step(train_module, grad_filter=None, rows: int = 1, tier: str = "full"):
    """包 MaxText 的 train_step：同签名，额外输出放进 metrics['janus'] = {'act': {...}, 'grad': {...}, 'xent': …}。
    grad_filter(状态路径) → None（不要）| () （整个）| ((轴, 起, 止), …)（在图里切片）。只支持 gradient_accumulation_steps=1。"""
    install_capture()
    from flax import nnx
    spec = TIERS[tier]
    if not spec["grads"]:
        grad_filter = None

    orig_loss_fn = train_module.loss_fn

    def loss_fn_shell(*args, **kwargs):
        loss, aux = orig_loss_fn(*args, **kwargs)
        if _STATE.active and "xent" in _STATE.box:
            aux = dict(aux)
            aux["janus_xent"] = _STATE.box.pop("xent")
        return loss, aux

    def hooked(model, config, state_mesh_shardings, params_shardings, state, data, dropout_rng=None):
        if config.gradient_accumulation_steps != 1:
            raise NotImplementedError("加钩子的训练步只支持 gradient_accumulation_steps=1")
        box = {}
        real_jax = train_module.jax
        with capturing(rows, tags=spec["tags"], xent=spec["xent"]):
            train_module.jax = _JaxProxy(real_jax, box)
            train_module.loss_fn = loss_fn_shell
            try:
                new_state, metrics = train_module.train_step(model, config, state_mesh_shardings, params_shardings,
                                                             state, data, dropout_rng)
            finally:
                train_module.jax = real_jax
                train_module.loss_fn = orig_loss_fn
        (_, (aux, _)), (raw_grads, _) = box["vg"]
        act = collect_sown(aux.get("intermediate_outputs", {}))
        grads = {}
        if grad_filter is not None:
            pure = nnx.to_pure_dict(raw_grads) if isinstance(raw_grads, nnx.State) else raw_grads
            for path, v in _flat(pure):
                key = "params/model/" + "/".join(str(p) for p in path)
                sel = grad_filter(key)
                if sel is None:
                    continue
                for ax, a, b in sel:
                    v = _slice_axis(v, ax, a, b)
                grads[key] = v
        extra = {"act": act, "grad": grads}
        if "janus_xent" in aux:
            extra["xent"] = aux["janus_xent"]
        metrics = dict(metrics)
        metrics["janus"] = extra
        return new_state, metrics

    return hooked


def _slice_axis(x, ax, a, b):
    import jax
    return jax.lax.slice_in_dim(x, a, b, axis=ax)


def _flat(tree, prefix=()):
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _flat(v, prefix + (k,))
    elif hasattr(tree, "items") and not hasattr(tree, "shape"):
        for k, v in tree.items():
            yield from _flat(v, prefix + (k,))
    else:
        yield prefix, tree


# ------------------------------------------------------------------------------------------------ 激活 → 规范名
def assemble_activations(act: dict[str, np.ndarray], dims: dict, rows: int) -> dict[str, np.ndarray]:
    """图内抓到的 {'<模块路径>/janus__<点>': 数组} → {规范名: GPU 布局数组}（只取第 0 行；路由计数用设备 0 的全部行）。
    扫描层里的值多一个前导的层轴（栈内下标 j），层号 = 栈的起始层 + j。"""
    fd = int(dims.get("first_dense", 0))
    g = int(dims["kv_heads"])
    qpg = int(dims["heads"]) // g
    out: dict[str, np.ndarray] = {}
    per_layer: dict[int, dict[str, np.ndarray]] = {}

    for key, v in act.items():
        v = np.asarray(v)
        parts = key.split("/")
        point = parts[-1][len(SOW_PREFIX):]
        mod = "/".join(parts[:-1])
        m = re.match(r"^decoder/(dense_layers|moe_layers)(/.*)?$", mod)
        if m:
            base = 0 if m.group(1) == "dense_layers" else fd
            sub = (m.group(2) or "").lstrip("/")
            for j in range(v.shape[0]):
                per_layer.setdefault(base + j, {})[f"{sub}:{point}"] = v[j]
            continue
        top = {"token_embedder:out": "embed.out", "decoder/decoder_norm:out": "final_norm.out",
               "decoder/logits_dense:out": "logits"}.get(f"{mod}:{point}")
        if top:
            out[top] = seq_first(v)

    for i, d in sorted(per_layer.items()):
        L = f"layers.{i}"

        def get(k):
            return d.get(k)
        if get(":in") is not None:
            out[f"{L}.in"] = seq_first(get(":in"))
        if get(":out") is not None:
            out[f"{L}.out"] = seq_first(get(":out"))
        q, k_, v_ = get("self_attention/query:out"), get("self_attention/key:out"), get("self_attention/value:out")
        if q is not None and k_ is not None and v_ is not None:
            qs, ks, vs = seq_first(q), seq_first(k_), seq_first(v_)           # [S, 1, H, hd] / [S, 1, G, hd]
            S = qs.shape[0]
            groups = [np.concatenate([qs[:, :, gi * qpg:(gi + 1) * qpg].reshape(S, 1, -1),
                                      ks[:, :, gi].reshape(S, 1, -1), vs[:, :, gi].reshape(S, 1, -1)], axis=-1)
                      for gi in range(g)]
            out[f"{L}.attn.qkv.out"] = np.ascontiguousarray(np.concatenate(groups, axis=-1))
        for pt, name in (("self_attention/query_norm:out", "attn.q_norm.out"), ("self_attention/key_norm:out", "attn.k_norm.out")):
            if get(pt) is not None:
                out[f"{L}.{name}"] = seq_first(get(pt))
        if get("self_attention/out:in") is not None:
            x = seq_first(get("self_attention/out:in"))
            out[f"{L}.attn.core.out"] = np.ascontiguousarray(x.reshape(x.shape[0], x.shape[1], -1))
        if get("self_attention/out:out") is not None:
            out[f"{L}.attn.proj.out"] = seq_first(get("self_attention/out:out"))
            out[f"{L}.attn.out"] = out[f"{L}.attn.proj.out"]
        # MLP 前归一化：稠密层 Megatron 的 pre_mlp_layernorm 是恒等（归一化融进 fc1），对应残差和；MoE 层是归一化输出
        if i < fd and get(":attn_resid") is not None:
            out[f"{L}.mlp_norm.out"] = seq_first(get(":attn_resid"))
        elif i >= fd and get(":post_attn_norm") is not None:
            out[f"{L}.mlp_norm.out"] = seq_first(get(":post_attn_norm"))
        if get("mlp/wi_0:out") is not None and get("mlp/wi_1:out") is not None:
            out[f"{L}.mlp.fc1.out"] = np.ascontiguousarray(
                np.concatenate([seq_first(get("mlp/wi_0:out")), seq_first(get("mlp/wi_1:out"))], axis=-1))
        if get("mlp:out") is not None:
            out[f"{L}.mlp.fc2.out"] = seq_first(get("mlp:out"))
            out[f"{L}.mlp.out"] = out[f"{L}.mlp.fc2.out"]
        moe_key = next((k for k in d if re.match(r"^\w+MoeBlock_0:out$", k)), None)
        if moe_key:
            out[f"{L}.mlp.out"] = seq_first(d[moe_key])
        for k in d:
            if re.match(r"^\w+MoeBlock_0/MoeBlock_0:out$", k):
                out[f"{L}.mlp.routed.out"] = seq_first(d[k])
            elif re.match(r"^\w+MoeBlock_0/shared_experts:out$", k):
                out[f"{L}.mlp.shared.out"] = seq_first(d[k])
        gk = next((k for k in d if re.match(r"^\w+MoeBlock_0/MoeBlock_0/gate:biased$", k)), None)
        if gk:
            base = gk[: -len(":biased")]
            biased, scores = d[gk], d.get(base + ":scores")
            if d.get(base + ":logits") is not None:
                out[f"{L}.mlp.router.logits"] = seq_first(d[base + ":logits"])
            if scores is not None:
                out[f"{L}.mlp.router.scores"] = seq_first(scores)
                idx, probs = route_from_scores(biased, scores, int(dims["topk"]), float(dims.get("routed_scaling", 1.0)))
                out[f"{L}.mlp.router.topk_idx"] = idx[0]
                out[f"{L}.mlp.router.probs"] = probs[0]
                counts = np.bincount(idx[:rows].reshape(-1), minlength=biased.shape[-1]).astype(np.int64)
                out[f"{L}.mlp.router.counts"] = counts
    return out


def route_from_scores(biased: np.ndarray, scores: np.ndarray, k: int, scale: float):
    """主机上重现 deepseek_routing + deepseek_scale_weights：
    选人用加了偏置的分数取 top-k（并列取小下标，同 jax.lax.top_k），权重用不加偏置的分数，除以和（+1e-20）再乘 scale。
    返回 (升序的 topk 下标 int32 [R, S, k]，散回 [R, S, E] 的 probs FP32)。"""
    b = np.asarray(biased, np.float32)
    s = np.asarray(scores, np.float32)
    order = np.argsort(-b, axis=-1, kind="stable")[..., :k]
    w = np.take_along_axis(s, order, axis=-1)
    w = w / (w.sum(-1, keepdims=True) + np.float32(1e-20))
    w = (w * np.float32(scale)).astype(np.float32)
    probs = np.zeros_like(s)
    np.put_along_axis(probs, order, w, axis=-1)
    return np.sort(order, axis=-1).astype(np.int32), probs


# ------------------------------------------------------------------------------------------------ 写盘
class MaxTextDump:
    """按 dump-format 写 TPU 运行目录。调用顺序：on_train_start → 每步 record_batch / on_step → on_train_end。
    多进程时所有进程都要调（读设备上的状态是集合操作），只有进程 0 写盘。"""

    # 全量抓数默认只留每层专家 0、1（对应 GPU rank 0 的 weight0 / weight1；其余太大），参数清单仍全量
    DEFAULT_EXPERTS = (0, 1)

    def __init__(self, out_dir, meta: dict, name_map: NameMap, dump_steps=(1,), full=False, full_steps=None,
                 experts=DEFAULT_EXPERTS, process_index: int = 0, micro: int = 2, routed_scaling: float = 1.0,
                 only: str | None = None, acts_steps=None):
        """acts_steps：只在这些抓数步写激活与逐 token loss（默认所有抓数步）；程序不变，只是不取回。"""
        self.out_dir = Path(out_dir)
        self.acts_steps = set(acts_steps) if acts_steps is not None else None
        self.meta, self.nm = meta, name_map
        self.dump_steps = set(dump_steps)
        self.full = full
        self.full_steps = set(full_steps) if full_steps else ({min(self.dump_steps)} if self.dump_steps else set())
        self.experts = set(experts)
        self.rank0 = process_index == 0
        self.micro = micro
        self.routed_scaling = routed_scaling
        # 全量抓数只抓 Megatron 名匹配的参数（buffer 照抓），与 GPU 侧 --full-only 同义；逗号分隔的几项按「或」合并
        #（TPU 的参数要经 JobSet 里的 bash 转一道，正则里不能有括号和竖线）
        self.only = re.compile("|".join(f"(?:{x})" for x in only.split(",") if x)) if only else None
        self.writer = RunWriter(self.out_dir, meta) if self.rank0 else None
        self._pending: dict[str, np.ndarray] = {}
        if self.rank0:
            (self.out_dir / "data").mkdir(parents=True, exist_ok=True)
            (self.out_dir / "params_sha").mkdir(parents=True, exist_ok=True)

    # ---- 状态 → Megatron 名字与布局
    def _iter_megatron(self, flat_state: dict, kind: str, keep_expert=lambda e: True, keep_name=None):
        """按 name_map 逐参数项把状态拼成 Megatron 布局，yield (项, 数组)。kind = param / mu / nu。
        一次只把当前项用到的叶子取到主机（同一叶子连续用时复用）。"""
        cache: dict[str, np.ndarray] = {}
        for ent in self.nm.entries:
            if ent.expert is not None and not keep_expert(ent.expert):
                continue
            if kind in ("mu", "nu") and ent.buffer:
                continue
            if keep_name is not None and not ent.buffer and not keep_name(self.nm.megatron_name(ent)):
                continue                                     # 不要的参数不从设备取（全量抓数时 embedding 一份就 2 GB）
            need = {pc.path for pc in ent.pieces}
            for p in list(cache):
                if p not in need:
                    del cache[p]
            shapes = self.nm.hf_shapes(ent)
            hf = []
            for pc in ent.pieces:
                if pc.path not in cache:
                    cache[pc.path] = to_host(flat_state[state_key(pc.path, kind)])
                hf.append(self.nm.hf_from_leaf(cache[pc.path], pc, shapes[ent.hf.index(pc.hf)]))
            order = [ent.hf.index(pc.hf) for pc in ent.pieces]
            hf = [x for _, x in sorted(zip(order, hf), key=lambda t: t[0])]
            yield ent, self.nm.to_megatron(ent, hf)

    def _mname(self, ent) -> str:
        return ("buffer." if ent.buffer else "") + self.nm.megatron_name(ent)

    def write_param_sha(self, flat_state: dict, tag: str):
        """params_sha/rank<r>.<tag>.json：按 GPU EP8 的持有关系分 rank（rank r：全部非专家参数 + 专家 24r…24r+23），
        参数存 BF16（主权重转 BF16，与 GPU 的 BF16 参数同义），expert_bias 存 FP32。全量不抽样。"""
        ranks = range(self.nm.ep)
        out = {r: {} for r in ranks}
        for ent, arr in self._iter_megatron(flat_state, "param"):
            a = arr if ent.buffer else to_bf16(arr)
            rec = {"sha256": sha256(a), "shape": list(a.shape), "dtype": str(a.dtype)}
            owner = self.nm.expert_owner(ent)
            for r in ranks:
                if owner is None or owner == r:
                    out[r][self._mname(ent)] = rec
        if self.rank0:
            for r in ranks:
                (self.out_dir / "params_sha" / f"rank{r}.{tag}.json").write_text(json.dumps(out[r], indent=0, sort_keys=True))

    def write_state_sha(self, flat_state: dict, tag: str) -> dict:
        """state_sha/<tag>.json：训练状态的 FP32 全量清单（见 state_manifest）。日常回归 R07（init）/ R15（final）用它，
        R18 的 dtype 清单也读它。和 params_sha 不同：不转 BF16——FP32 主权重上不到半个 BF16 格的差，转成 BF16 就看不见。"""
        man = state_manifest(flat_state)
        if self.rank0:
            d = self.out_dir / "state_sha"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{tag}.json").write_text(json.dumps(man, indent=0, sort_keys=True))
        return man

    def write_modules(self, tagged: dict[str, str], flat_state: dict):
        if not self.rank0:
            return
        with open(self.out_dir / "modules.txt", "w") as f:
            for p, t in sorted(tagged.items()):
                f.write(f"{p}\tjanus_tag={t}\n")
            for k, v in sorted(flat_state.items()):
                if hasattr(v, "shape"):
                    f.write(f"param\t{k}\t{tuple(v.shape)}\t{v.dtype}\n")

    # ---- 每步
    def record_batch(self, step: int, records: dict[int, list[dict]]):
        if not self.rank0:
            return
        for d, recs in records.items():
            with open(self.out_dir / "data" / f"rank{d}.jsonl", "a") as f:
                for rec in recs:
                    f.write(json.dumps(rec) + "\n")

    def capture_before(self, step: int, flat_state: dict):
        """抓数步、训练步之前：buffer（路由偏置）；全量抓数步另抓 BF16 参数与 FP32 主权重。"""
        if step not in self.dump_steps:
            return
        keep = lambda e: e in self.experts  # noqa: E731
        full = self.full and step in self.full_steps
        for ent, arr in self._iter_megatron(flat_state, "param", keep, self._only if full else (lambda n: False)):
            name = self.nm.megatron_name(ent)
            if ent.buffer:
                self._pending[f"buffer.{name}"] = arr
            elif full:
                self._pending[f"param.{name}"] = to_bf16(arr)
                self._pending[f"main_param.{name}"] = arr

    def capture_after(self, step: int, flat_state: dict, extras: dict | None):
        """抓数步、训练步之后：更新后的 buffer、主权重与 Adam 动量；图内额外输出（激活、梯度、逐 token loss）。"""
        if step not in self.dump_steps:
            return
        keep = lambda e: e in self.experts  # noqa: E731
        full = self.full and step in self.full_steps
        for ent, arr in self._iter_megatron(flat_state, "param", keep, self._only if full else (lambda n: False)):
            name = self.nm.megatron_name(ent)
            if ent.buffer:
                self._pending[f"buffer_after.{name}"] = arr
            elif full:
                self._pending[f"main_param_after.{name}"] = arr
        if full:
            for kind, prefix in (("mu", "exp_avg"), ("nu", "exp_avg_sq")):
                for ent, arr in self._iter_megatron(flat_state, kind, keep, self._only):
                    self._pending[f"{prefix}.{self.nm.megatron_name(ent)}"] = arr
        acts = bool(extras) and (self.acts_steps is None or step in self.acts_steps)
        if acts:
            dims = dict(self.nm.dims, routed_scaling=self.routed_scaling)
            act = {k: to_host(v) for k, v in sorted(extras.get("act", {}).items())}
            self._pending.update(assemble_activations(act, dims, rows=self.micro))
            if "xent" in extras:
                x = to_host(extras["xent"])
                self._pending["loss.per_token"] = np.ascontiguousarray(x[0:1].astype(np.float32))
        if extras and full and extras.get("grad"):
            g = {k: to_host(v) for k, v in sorted(extras["grad"].items())}
            self._pending.update({k: v for k, v in self._grads_megatron(g).items() if self._only(k[len("grad."):])})

    def _only(self, name: str) -> bool:
        return self.only is None or bool(self.only.search(name))

    def add_tensors(self, tensors: dict[str, np.ndarray]):
        """本步另外要写的张量（例如独立梯度探针 gp_*），随本步 flush 一起写出。"""
        if self.rank0:
            self._pending.update(tensors)

    def _grads_megatron(self, grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """图内取出的梯度（专家叶子已在图里切成专家 0…n−1）→ Megatron 名。"""
        out = {}
        n_exp = max(self.experts) + 1 if self.experts else 0
        for ent in self.nm.entries:
            if ent.buffer or (ent.expert is not None and ent.expert not in self.experts):
                continue
            shapes = self.nm.hf_shapes(ent)
            hf = [None] * len(ent.hf)
            ok = True
            for pc in ent.pieces:
                key = state_key(pc.path, "param")
                if key not in grads:
                    ok = False
                    break
                leaf = grads[key]
                if ent.expert is not None and leaf.shape[0] != n_exp:   # 专家轴已在图里切成 [0, n_exp)
                    raise ValueError(f"{key} 专家轴是 {leaf.shape[0]}，期望已切成 {n_exp}")
                hf[ent.hf.index(pc.hf)] = self.nm.hf_from_leaf(leaf, pc, shapes[ent.hf.index(pc.hf)])
            if ok:
                out[f"grad.{self.nm.megatron_name(ent)}"] = self.nm.to_megatron(ent, hf)
        return out

    def log_metrics(self, step: int, values: dict):
        if self.rank0:
            self.writer.log_metrics(step, **values)

    def flush(self, step: int):
        """把本步攒下的张量按 scope 分文件写出（与 GPU 侧同样的分组方式）。"""
        if not self.rank0 or not self._pending:
            self._pending = {}
            return
        by_scope: dict[str, dict] = {}
        for name, arr in self._pending.items():
            head = name.split(".")[0]
            if head in ("grad", "param", "main_param", "main_param_after", "exp_avg", "exp_avg_sq", "buffer", "buffer_after") \
                    or head.startswith("gp_"):
                m = re.search(r"\.layers\.(\d+)\.", name)
                scope = f"{head}.layers.{m.group(1)}" if m else f"{head}.other"
            elif name.startswith("layers."):
                scope = ".".join(name.split(".")[:2])
            else:
                scope = head
            by_scope.setdefault(scope, {})[name] = arr
        kinds = {"grad": ("grad", "backward"), "param": ("param", "before_update"), "main_param": ("param", "before_update"),
                 "main_param_after": ("param", "after_update"), "buffer": ("param", "before_update"),
                 "buffer_after": ("param", "after_update"), "exp_avg": ("opt_state", "after_update"),
                 "exp_avg_sq": ("opt_state", "after_update"), "input": ("tokens", "forward"), "loss": ("other", "forward")}
        for scope, tensors in sorted(by_scope.items()):
            h0 = scope.split(".")[0]
            kind, point = ("grad", "backward") if h0.startswith("gp_") else kinds.get(h0, ("activation", "forward"))
            self.writer.write_tensors(step, scope, tensors, kind=kind, point=point)
        self._pending = {}

    def on_train_end(self):
        if self.rank0:
            self.writer.done()


def bias_stats(bias_stack: np.ndarray, first_layer: int) -> dict[str, float]:
    """路由偏置 [L_moe, E] → 每层均值 / 标准差（与 GPU 回调同义：FP32 张量的 mean 与无偏 std）。主机上用 FP64 算。"""
    out = {}
    b = np.asarray(bias_stack, np.float64)
    for j in range(b.shape[0]):
        out[f"router.bias_mean.{first_layer + j}"] = float(b[j].mean())
        out[f"router.bias_std.{first_layer + j}"] = float(b[j].std(ddof=1))
    return out


def state_digest(flat_state: dict) -> str:
    """训练状态里全部数组的 sha256 摘要（逐叶子 sha256 再按路径排序拼起来）。用于「开不开钩子逐位相同」自检。"""
    h = hashlib.sha256()
    for k in sorted(flat_state):
        v = flat_state[k]
        if hasattr(v, "shape"):
            h.update(k.encode())
            h.update(sha256(to_host(v)).encode())
    return h.hexdigest()


def env_snapshot(keys=("LIBTPU_INIT_ARGS", "XLA_FLAGS", "JAX_PLATFORMS", "TPU_STDERR_LOG_LEVEL")) -> dict:
    return {k: os.environ.get(k) for k in keys}
