"""混元 3 小模型 TPU 训练入口（janus 第二阶段：TPU 对 GPU golden）。和 GPU 侧 models/hunyuan3/frameworks/megatron/train_hy3.py 对称。

不改 MaxText：用 MaxText 的 setup_train_loop 建模型、优化器、mesh，载入转换好的检查点；训练循环自己写，
每步把 golden 同一步的全局 batch 原样喂进去（replay.py），调 MaxText 自己的 train_step（jit 后）。

用法（每个 TPU 主机上都跑同一条命令；只有进程 0 写运行目录）：
  python train_hy3.py --load gs://…/hy3-4l-maxtext/0/items --golden /data/golden/g1k-a \\
      --hf-manifest /data/hy3-4l/tensor_sha256.json --steps 1000 \\
      --janus-run /data/janus-runs/t-a --side tpu --dump-steps 1,1000 --full [MaxText key=value …]

  最小可跑（先看 loss）：--no-hooks --dump-steps none --no-param-sha，只写 metrics.jsonl 与输入指纹
  --no-hooks         不加钩子：只写逐步标量、输入指纹、参数清单、状态类张量（零扰动），给「开不开钩子逐位相同」做对照
  --observe          观测构建：每一步都用同一个加钩子的程序往下训练（整条轨迹只有这一个编译好的程序），只在 --dump-steps
                     取额外输出；不做自检。它和生产构建（不加钩子）不逐位，先当一个执行级扰动臂过三关，过了才把它抓到的
                     激活 / 梯度当正式证据（frameworks/maxtext/README.md「观测构建」）
  --no-selfcheck     抓数步直接用加钩子程序的输出往下训练，不再跑不加钩子的程序做对照（默认做对照：往下训练用不加钩子的
                     程序，加钩子的程序在副本上跑、只取额外输出，两者逐位比，结果在 meta.hooks.selfcheck；要多两份训练状态的 HBM）
  --cpu-small        CPU 冒烟：缩小模型维度、用 8 个假 CPU 设备（见 README「CPU 冒烟」）

步号：本侧第 s 步吃 golden 第 s 步的 batch，metrics 也记在第 s 步（golden_step = step）。
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mt_common  # noqa: E402
import replay  # noqa: E402

from janus.dump import maxtext as jd  # noqa: E402

# CPU 冒烟用的小维度（结构与混元 3 相同：第 0 层稠密、1–3 层 MoE、sigmoid 路由 + 偏置、共享专家、GQA）
CPU_SMALL = ["hardware=cpu", "skip_jax_distributed_system=True", "base_emb_dim=256", "base_mlp_dim=512",
             "base_moe_mlp_dim=128", "num_experts=16", "num_experts_per_tok=4", "base_num_query_heads=8",
             "base_num_kv_heads=2", "head_dim=32", "vocab_size=1024", "attention=dot_product",
             "sparse_matmul=False", "megablox=False"]


# 分组参数平方和（--group-norms）：MaxText 状态路径 → 组名，和 GPU 侧 janus/dump/megatron.py 的 _GROUPS 一一对应
GROUP_RULES = [(r"token_embedder/", "embedding"), (r"decoder/logits_dense/", "output"), (r"decoder/decoder_norm/", "final_norm"),
               (r"self_attention/(query|key|value)/kernel", "attn_qkv"), (r"self_attention/out/kernel", "attn_proj"),
               (r"self_attention/(query|key)_norm/", "qk_norm"), (r"pre_self_attention_layer_norm/", "input_norm"),
               (r"post_self_attention_layer_norm/", "mlp_norm"), (r"shared_experts/wi_[01]/", "shared_fc1"),
               (r"shared_experts/wo/", "shared_fc2"), (r"MoeBlock_0/gate/kernel", "router"),
               (r"MoeBlock_0/wi_[01]$", "experts_fc1"), (r"MoeBlock_0/wo$", "experts_fc2"),
               (r"dense_layers/mlp/wi_[01]/", "dense_fc1"), (r"dense_layers/mlp/wo/", "dense_fc2")]


def group_of(path: str):
    import re
    return next((g for pat, g in GROUP_RULES if re.search(pat, path)), None)


def make_nan_train_step(train_module, target: str):
    """故障演练：包 MaxText 的 train_step，在 value_and_grad 输出的参数梯度里，路径含 target 的第一个浮点叶子的第一个元素
    换成 NaN（裁剪与优化器之前，和 GPU 演练 phase1-nan 埋的位置同义：第 0 层注意力输出投影的梯度）。只在演练步用，
    只支持 gradient_accumulation_steps=1（value_and_grad 直接在 train_step 里调）。"""
    import re

    import jax
    import jax.numpy as jnp
    real = train_module.jax

    class _NanProxy:
        def __getattr__(self, name):
            return getattr(real, name)

        def value_and_grad(self, fun, *a, **kw):
            g = real.value_and_grad(fun, *a, **kw)

            def wrapped(*args, **kwargs):
                val, grads = g(*args, **kwargs)
                head = grads[0] if isinstance(grads, tuple) else grads
                hit = []

                def put(path, x):
                    key = re.sub(r"[\[\]'.]+", "/", jax.tree_util.keystr(path)).strip("/")
                    if not hit and target in key and hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
                        hit.append(key)
                        return x.at[(0,) * x.ndim].set(jnp.nan)
                    return x
                head = jax.tree_util.tree_map_with_path(put, head)
                if not hit:
                    raise ValueError(f"--inject-nan-grad：梯度里没有路径含 {target!r} 的浮点叶子")
                print(f"[janus] 演练：在梯度 {hit[0]} 的第一个元素埋 NaN", flush=True)
                return val, ((head,) + tuple(grads[1:]) if isinstance(grads, tuple) else head)
            return wrapped

    def nan_step(model, config, state_mesh_shardings, params_shardings, state, data, dropout_rng=None):
        if config.gradient_accumulation_steps != 1:
            raise NotImplementedError("--inject-nan-grad 只支持 gradient_accumulation_steps=1")
        train_module.jax = _NanProxy()
        try:
            return train_module.train_step(model, config, state_mesh_shardings, params_shardings, state, data, dropout_rng)
        finally:
            train_module.jax = real
    return nan_step


def make_bf16w_train_step(train_module):
    """「BF16 权重、全程 FP32 计算」：包 MaxText 的 train_step，在 value_and_grad 求导的参数（第一个参数）上做直通舍入：
    前反向看到的是 reduce_precision(θ, bf16)，梯度对 θ 原样传回（θ + stop_gradient(round(θ) − θ)）。用 reduce_precision 而不是
    astype 往返：TPU 上 XLA 默认允许 excess precision，astype(bf16).astype(f32) 会被消掉。只支持 gradient_accumulation_steps=1。"""
    import jax
    import jax.numpy as jnp
    real = train_module.jax

    def st_round(x):
        if hasattr(x, "dtype") and x.dtype == jnp.float32:
            return x + jax.lax.stop_gradient(jax.lax.reduce_precision(x, exponent_bits=8, mantissa_bits=7) - x)
        return x

    class _Bf16wProxy:
        def __getattr__(self, name):
            return getattr(real, name)

        def value_and_grad(self, fun, *a, **kw):
            g = real.value_and_grad(fun, *a, **kw)

            def wrapped(params, *args, **kwargs):
                return g(jax.tree_util.tree_map(st_round, params), *args, **kwargs)
            return wrapped

    def bf16w_step(model, config, state_mesh_shardings, params_shardings, state, data, dropout_rng=None):
        if config.gradient_accumulation_steps != 1:
            raise NotImplementedError("--bf16-weights 只支持 gradient_accumulation_steps=1")
        train_module.jax = _Bf16wProxy()
        try:
            return train_module.train_step(model, config, state_mesh_shardings, params_shardings, state, data, dropout_rng)
        finally:
            train_module.jax = real
    return bf16w_step


def _prng(x) -> bool:
    import jax
    return hasattr(x, "dtype") and jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)


def save_train_state(state, path: str) -> dict:
    """整个训练状态（参数 / FP32 主权重、两个动量、路由偏置、步数计数器、随机数键）用 orbax 存到 path（本地或 gs://）。
    多进程时所有进程都要调。随机数键存它的原始数据（key_data），恢复时按原来的实现包回去。"""
    import jax
    import orbax.checkpoint as ocp
    from flax import nnx
    pure = nnx.to_pure_dict(state)
    keys = []

    def enc(p, x):
        if _prng(x):
            keys.append(jax.tree_util.keystr(p))
            return jax.random.key_data(x)
        return x
    pure = jax.tree_util.tree_map_with_path(enc, pure)
    ck = ocp.StandardCheckpointer()
    ck.save(path, pure)
    ck.wait_until_finished()
    return {"path": path, "leaves": len(jax.tree_util.tree_leaves(pure)), "prng_leaves": keys}


def load_train_state(state, path: str) -> dict:
    """save_train_state 的反向：按当前状态每个叶子的形状、dtype、sharding 读回，原地换掉 state 的值。"""
    import jax
    import orbax.checkpoint as ocp
    from flax import nnx
    pure = nnx.to_pure_dict(state)
    impls = {}

    def target(p, x):
        if _prng(x):
            impls[jax.tree_util.keystr(p)] = jax.random.key_impl(x)
            x = jax.random.key_data(x)
        return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding)
    tgt = jax.tree_util.tree_map_with_path(target, pure)
    got = ocp.StandardCheckpointer().restore(path, tgt)
    got = jax.tree_util.tree_map_with_path(
        lambda p, x: jax.random.wrap_key_data(x, impl=impls[jax.tree_util.keystr(p)]) if jax.tree_util.keystr(p) in impls else x, got)
    nnx.replace_by_pure_dict(state, got)
    return {"path": path, "leaves": len(jax.tree_util.tree_leaves(got)), "prng_leaves": sorted(impls)}


def sha_file(path) -> str | None:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--load", required=True, help="MaxText 参数检查点（convert_hf_to_maxtext.py 的 …/0/items）")
    p.add_argument("--golden", required=True, help="golden 运行目录（本地，至少要有 meta.json 与 data/batches/）")
    p.add_argument("--hf-manifest", default=None, help="HF 目录的 tensor_sha256.json（meta.checkpoint.manifest_sha256 用）")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--janus-run", required=True, help="运行目录（本地路径；跑完由 run.sh 传到 GCS）")
    p.add_argument("--side", default="tpu")
    p.add_argument("--dump-steps", default="1", help="逗号分隔，本侧步号（= golden 步号）；none = 不抓张量")
    p.add_argument("--full", action="store_true", help="全量抓数：另抓参数梯度、更新前后主权重与 Adam 动量")
    p.add_argument("--full-steps", default=None, help="做全量抓数的步，默认 dump 步里最小的一步")
    p.add_argument("--no-hooks", action="store_true", help="不加图内钩子（只抓状态类，零扰动）")
    p.add_argument("--row-stats-every", type=int, default=0,
                   help="每 N 步记输出层、embedding 每行（每个词）的主权重平方和，与输出层每行 Adam 更新量 |m̂/(√v̂+ε)| 的均值、√v̂ 的均值、w·u（范数变化的一阶项）"
                        "（output_row_sq / output_row_u / embedding_row_sq，长度 = 词表）；0 = 不记")
    p.add_argument("--resid-probe-every", type=int, default=0,
                   help="每 N 步用当前参数在固定一条序列（golden 第 1 步 rank 0 第一个 microbatch）上另跑一次前向（单独编译，"
                        "不进训练图、不改训练轨迹），记各层输出与最后归一化输出在 token 上的均值向量（resid_mean.<名>，长度 = 隐藏维）、"
                        "logits 均值与 logsumexp 均值，以及每个 token 的 RMS（tok_rms.<名>，长度 = 序列长）；0 = 不记")
    p.add_argument("--group-norms", action="store_true",
                   help="每步另记分组参数平方和（group_sq.<组>[.<层>]）与每个专家的平方和（expert_sq.<层>），"
                        "组名与 GPU 侧 janus/dump/megatron.py 的 _GROUPS 相同；只读状态，不改训练")
    p.add_argument("--observe", action="store_true",
                   help="观测构建：每一步都用同一个加钩子的程序（不在抓数步和非抓数步之间换程序），额外输出只在抓数步取回；"
                        "隐含 --no-selfcheck，要求 gradient_accumulation_steps=1")
    p.add_argument("--acts-steps", default=None,
                   help="逗号分隔：只在这些抓数步写激活、logits、逐 token loss（默认所有抓数步）；none = 都不写。程序不变，只是不取回")
    p.add_argument("--jax-default-matmul-precision", default=None, choices=["default", "high", "highest"],
                   help="（只作诊断）jax.config 的 jax_default_matmul_precision：没给 precision 的矩阵乘（如 MoE 的 ragged_dot）按它算")
    p.add_argument("--head-matmul-precision", default=None, choices=["default", "high", "highest"],
                   help="（只作诊断臂）只改输出头 logits 矩阵乘的 precision（其余仍按 matmul_precision）：nnx_decoders 里唯一的 DenseGeneral 就是输出头")
    p.add_argument("--head-round", default=None, choices=["input", "cotangent", "cot-dgrad", "cot-wgrad"],
                   help="（只作诊断臂，配全 FP32）输出头矩阵乘只舍一侧操作数到 BF16：input = 输入隐状态 h 直通舍入（前向 logits 与"
                        "权重梯度看到 BF16 的 h，dh 不舍）；cotangent = 流进输出头反向的 dlogits 舍 BF16（dgrad 与权重梯度都用它），前向不变；"
                        "cot-dgrad / cot-wgrad = 只有 dgrad（dh）/ 只有权重梯度用舍过的 dlogits（输出头改用自定义反向的 einsum，precision=highest）")
    p.add_argument("--router-jitter", default=None,
                   help="（只作诊断，敏感度臂）<σ>:<seed>：路由选专家时给分数加固定种子的高斯抖动，只改选谁、不改权重（router_jitter.py）")
    p.add_argument("--grad-fp32", default=None,
                   help="（只作诊断臂）逗号分隔的 norm / head / embed：这几类参数的梯度不经过 BF16、直接在 FP32 里求和，前向逐位不变"
                        "（grad_fp32.py；对齐报告台账 #20 的定责）")
    p.add_argument("--bf16-weights", action="store_true",
                   help="（只作诊断参照，配 dtype=float32 等全 FP32 覆盖）BF16 权重、全程 FP32 计算：每步前反向用 FP32 主权重按 RNE 舍到 BF16 的值"
                        "（直通：梯度原样回到主权重），对应 GPU 侧 train_hy3.py --fp32 --bf16-weights。只支持 gradient_accumulation_steps=1")
    p.add_argument("--inject-nan-grad", type=int, default=0,
                   help="故障演练：第 N 步在参数梯度里埋一个 NaN（--nan-target 指定的叶子，裁剪与优化器之前），其余步照常；0 = 不埋")
    p.add_argument("--nan-target", default="dense_layers/self_attention/out/kernel",
                   help="--inject-nan-grad 埋在哪个梯度叶子（MaxText 状态路径的子串；默认第 0 层注意力输出投影，同 GPU 演练）")
    p.add_argument("--no-selfcheck", action="store_true",
                   help="抓数步直接用加钩子程序的输出往下训练（不再跑一遍不加钩子的程序做对照）")
    p.add_argument("--capture", default="full", choices=sorted(jd.TIERS),
                   help="抓数档位（janus/dump/maxtext.py 的 TIERS）；默认 full")
    p.add_argument("--probe-tiers", default=None,
                   help="逗号分隔的档位：在第一个抓数步逐档编一个加钩子程序，各自和不加钩子的程序逐位比，结果写进 meta.hooks.probe")
    p.add_argument("--no-param-sha", action="store_true", help="不写参数清单（省掉训练前后各一次全量读状态，只想先看 loss 时用）")
    p.add_argument("--state-sha", action="store_true",
                   help="另写训练状态的 FP32 全量清单 state_sha/{init,final}.json（主权重、两个动量、路由偏置、计数器，"
                        "按存储 dtype 算 sha256，不转 BF16）。日常回归 R07 / R15 / R18 用")
    p.add_argument("--hlo-dir", default=None,
                   help="XLA 的 dump 目录（run.sh 的 XLA_DUMP=1 是 /data/xla）：训练结束后把生产训练步（jit_train_step）"
                        "优化后的 HLO 与编译器实际生效的开关（flagfile、debug_options）拷进运行目录的 hlo/。日常回归 R05 / R20 / R21 用")
    p.add_argument("--full-only", default=None,
                   help="全量抓数只抓 Megatron 名匹配的参数（逗号分隔的几个正则按「或」合并；路由偏置等 buffer 照抓）。"
                        "JobSet 里经 bash 转一道，不要用括号、竖线")
    p.add_argument("--grad-probe-at", default=None,
                   help="逗号分隔的步号 s：第 s 步训练之前，用当前状态 θ_{s-1} 与第 s 步的 batch 另编三个独立的 jit 程序（不进训练图、"
                        "不捐出状态、不改轨迹）：① 生产精度的梯度（与训练步同一套配置，gp_bf16.*）；② 参照精度的梯度"
                        "（--probe-ref 覆盖，默认全程 FP32，gp_f32.*）；③ 参照精度下 logits 在词表与 token 上的均值对参数的梯度"
                        "（「所有 logits 一起动」的方向，gp_lvl.*）。只写 --full-only 选中的参数（专家只留 0、1），"
                        "输出层另写沿词表求和 [D] 与逐行平方和 [V]")
    p.add_argument("--probe-ref", default="dtype=float32,matmul_precision=highest,attention=dot_product,megablox=False,"
                                          "float32_qk_product=True,float32_logits=True,logits_dot_in_fp32=True",
                   help="--grad-probe-at 参照程序在训练配置之上追加的 MaxText 覆盖（逗号分隔）")
    p.add_argument("--probe-ref-weights", default="bf16", choices=["bf16", "master"],
                   help="参照程序（gp_f32 / gp_lvl）用的权重：bf16 = 主权重先舍到 BF16（生产程序实际参与计算的那份，只比算术误差，默认）；"
                        "master = FP32 主权重原样")
    p.add_argument("--export-params-at", default=None,
                   help="逗号分隔的步号 N：第 N 步更新之后把整套 FP32 主权重按 GPU 的 rank 划分导出到 <运行目录>/export/step_N/"
                        "（与 GPU 侧 param_swap.py 同一种格式，见 param_io.py）")
    p.add_argument("--params-from", default=None,
                   help="开训前把整套参数（含路由偏置）换成这个目录里的（param_io.py / param_swap.py 的格式，本地或 gs://），"
                        "放回后读出逐张量核 sha256。优化器状态不动")
    p.add_argument("--first-step", type=int, default=1,
                   help="从第几步开始（吃 golden 同一步的 batch）；配 --params-from 在别处来的状态上重算某一步的梯度")
    p.add_argument("--save-state-at", type=int, default=0,
                   help="第 N 步更新之后用 orbax 把整个训练状态存到 --state-dir（存档 / 续训演练），同时写 state_sha/step<N>.json")
    p.add_argument("--state-dir", default=None, help="--save-state-at 的存档目录（本地或 gs://，不能已存在）")
    p.add_argument("--resume-state", default=None,
                   help="开训前从这个 orbax 存档恢复整个训练状态（save_train_state 写的），配 --first-step N+1 从下一步接着训；"
                        "恢复后写 state_sha/step<N>.json（N = --first-step − 1），可和存档那次逐项比")
    p.add_argument("--cpu-small", action="store_true")
    p.add_argument("--maxtext-commit", default=os.environ.get("MAXTEXT_COMMIT", "unknown"))
    p.add_argument("--janus-commit", default=os.environ.get("JANUS_COMMIT", "unknown"))
    p.add_argument("--image", default=os.environ.get("JANUS_IMAGE", "unknown"))
    p.add_argument("--topology", default=os.environ.get("TPU_TOPOLOGY", "unknown"))
    p.add_argument("maxtext_args", nargs="*", help="追加的 MaxText key=value（会写进 meta.config.overrides）")
    return p.parse_args(argv)


def _int_steps(s: str | None) -> list[int]:
    if not s or s.strip().lower() == "none":
        return []
    return [int(x) for x in s.split(",") if x]


FLAG_ENV_PREFIXES = ("XLA_", "LIBTPU_", "JAX_", "TPU_", "MEGASCALE_", "TF_CPP_")


def flag_snapshot() -> dict:
    """运行进程里实际的开关（R05）：相关环境变量全量 + jax.config 的全部标量取值。不从调用方参数推断。"""
    import jax
    env = {k: v for k, v in sorted(os.environ.items()) if k.startswith(FLAG_ENV_PREFIXES)}
    try:
        values = dict(jax.config.values)
    except Exception:  # noqa: BLE001  老版本 jax 没有 values
        values = {}
    jcfg = {k: v for k, v in sorted(values.items()) if isinstance(v, (str, int, float, bool, type(None)))}
    return {"env": env, "jax_config": jcfg}


def maxtext_full_config(config) -> dict:
    """MaxText 展开后的完整配置（pyconfig 的全部键），值不是 JSON 标量或列表的转成字符串。"""
    try:
        keys = dict(config.get_keys())
    except Exception:  # noqa: BLE001
        keys = {k: getattr(config, k) for k in dir(config) if not k.startswith("_")}
    out = {}
    for k, v in sorted(keys.items()):
        if isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float, str, bool, type(None))) for x in v):
            out[k] = list(v)
        else:
            out[k] = str(v)
    return out


def collect_hlo(hlo_dir, run_dir) -> dict:
    """XLA dump 目录里生产训练步的产物 → 运行目录 hlo/：优化后的 HLO（实际执行的那个可执行文件）、编译器生效的开关。
    文件名去掉 module 号与编译 id，固定成 train_step.<后缀>。只认名字恰好是 jit_train_step 的模块（加钩子的程序叫别的名字）；
    同名模块编了不止一次（重新编译）时全部按出现顺序编号保留，并在 meta 里记下，回归门判 error（不知道执行的是哪一个）。"""
    import re
    import shutil
    src, dst = Path(hlo_dir), Path(run_dir) / "hlo"
    # TPU 上文件名带编译 id（module_0366.jit_train_step.cl_963683785.<后缀>），CPU 上没有、优化后的 HLO 有时叫
    # cpu_after_optimizations.txt；按 module 号分组，后缀统一成 after_optimizations.txt
    pat = re.compile(r"^module_(\d+)\.jit_train_step(?:\.cl_\d+)?\.((?:cpu_)?after_optimizations\.txt|flagfile|debug_options)$")
    found: dict[str, list[Path]] = {}
    for f in sorted(src.glob("module_*.jit_train_step.*")) if src.is_dir() else []:
        m = pat.match(f.name)
        if m:
            found.setdefault(m.group(1), []).append(f)
    info = {"source": str(src), "modules": sorted(found), "files": {}}
    if not found:
        info["error"] = "dump 目录里没有 jit_train_step 的产物"
        return info
    dst.mkdir(parents=True, exist_ok=True)
    for i, cl in enumerate(sorted(found)):
        for f in found[cl]:
            suffix = pat.match(f.name).group(2).removeprefix("cpu_")
            name = f"train_step.{suffix}" if len(found) == 1 else f"train_step.{i}.{suffix}"
            shutil.copyfile(f, dst / name)
            info["files"][name] = hashlib.sha256((dst / name).read_bytes()).hexdigest()
    if len(found) > 1:
        info["error"] = f"jit_train_step 编译了 {len(found)} 次"
    return info


def build_meta(a, config, mesh, golden_meta: dict, shard_devices: dict, batches_sha: str) -> dict:
    import flax
    import jax
    import optax
    import orbax.checkpoint as ocp

    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:  # noqa: BLE001
            return None
    try:
        from jax.extend.backend import get_backend
        platform_version = get_backend().platform_version
    except Exception:  # noqa: BLE001
        platform_version = None
    c = config
    mesh_shape = dict(mesh.shape)
    model_config = {   # 与 GPU meta.model_config 同名同义（值来自 MaxText 实际生效的配置）
        "num_layers": c.num_decoder_layers, "hidden_size": c.emb_dim, "num_attention_heads": c.num_query_heads,
        "num_query_groups": c.num_kv_heads, "kv_channels": c.head_dim, "ffn_hidden_size": c.mlp_dim,
        "num_moe_experts": c.num_experts, "moe_router_topk": c.num_experts_per_tok, "moe_ffn_hidden_size": c.moe_mlp_dim,
        "moe_shared_expert_intermediate_size": c.shared_experts * c.moe_mlp_dim,
        "moe_router_dtype": "fp32" if c.float32_gate_logits else "bf16",
        "moe_router_score_function": c.routed_score_func, "moe_router_enable_expert_bias": bool(c.routed_bias),
        "moe_router_bias_update_rate": c.routed_bias_update_rate, "moe_router_topk_scaling_factor": c.routed_scaling_factor,
        "moe_aux_loss_coeff": c.load_balance_loss_weight,
        "moe_z_loss_coeff": None if not c.z_loss_multiplier else c.z_loss_multiplier,   # GPU 不开 z-loss 记 None
        "hidden_dropout": c.dropout_rate if c.enable_dropout else 0.0,
        "attention_dropout": c.dropout_rate if c.enable_dropout else 0.0,
        "normalization": "RMSNorm", "layernorm_epsilon": c.normalization_layer_epsilon, "qk_layernorm": bool(c.use_qk_norm),
        "rotary_base": float(c.rope_max_timescale), "position_embedding_type": "rope",
        "bf16": str(c.dtype) in ("bfloat16", "jnp.bfloat16", "<class 'jax.numpy.bfloat16'>") or "bfloat16" in str(c.dtype),
        "mtp_num_layers": c.mtp_num_layers, "use_random_routing": bool(c.use_random_routing),
        "n_routing_groups": c.n_routing_groups,
    }
    lr_constant = c.lr_schedule_type == "wsd" and c.warmup_steps_fraction == 0 and c.wsd_decay_steps_fraction == 0
    optimizer_config = {
        "optimizer": c.opt_type, "lr": c.learning_rate, "weight_decay": c.adam_weight_decay,
        "adam_beta1": c.adam_b1, "adam_beta2": c.adam_b2, "adam_eps": c.adam_eps, "adam_eps_root": c.adam_eps_root,
        "clip_grad": c.gradient_clipping_threshold, "adamw_mask": list(c.adamw_mask or []),
        "main_params_dtype": str(c.weight_dtype), "exp_avg_dtype": str(c.mu_dtype), "exp_avg_sq_dtype": "float32",
    }
    scheduler_config = {
        "lr_decay_style": "constant" if lr_constant else str(c.lr_schedule_type),
        "lr_warmup_iters": int(c.learning_rate_schedule_steps * c.warmup_steps_fraction),
        "start_weight_decay": c.adam_weight_decay, "end_weight_decay": c.adam_weight_decay,
        "weight_decay_incr_style": "constant",
    }
    keys = ["remat_policy", "moe_routing_maps", "attention", "matmul_precision", "float32_gate_logits", "float32_weight_sum",
            "float32_qk_product", "float32_logits", "cast_logits_to_fp32", "logits_dot_in_fp32", "megablox", "sparse_matmul",
            "capacity_factor", "use_iota_embed", "scan_layers", "per_device_batch_size", "gradient_accumulation_steps",
            "max_target_length", "global_batch_size_to_train_on", "ici_fsdp_parallelism", "ici_data_parallelism",
            "ici_expert_parallelism", "shard_mode", "grad_dtype", "weight_dtype", "dtype", "mu_dtype", "sa_block_q",
            "sa_block_kv", "use_tokamax_splash", "mlp_activations_limit", "routed_scaling_factor", "init_weights_seed",
            "data_shuffle_seed", "learning_rate_schedule_steps", "lr_schedule_type", "model_name", "decoder_block"]
    maxtext_config = {k: (v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v))
                      for k in keys if (v := getattr(c, k, None)) is not None or True}
    return {
        "schema_version": 1, "run_name": Path(a.janus_run).name, "side": a.side, "framework": "maxtext",
        "model": "hunyuan3-4l",
        "code": [{"repo": "maxtext", "commit": a.maxtext_commit}, {"repo": "janus", "commit": a.janus_commit}],
        "image": a.image,
        "versions": {"jax": jax.__version__, "jaxlib": ver("jaxlib"), "libtpu": platform_version, "flax": flax.__version__,
                     "optax": optax.__version__, "orbax_checkpoint": getattr(ocp, "__version__", None)},
        "hardware": {"accelerator": jax.devices()[0].device_kind, "count": jax.device_count(), "topology": a.topology,
                     "processes": jax.process_count()},
        "parallelism": {"dp": mesh_shape.get("data", 1), "fsdp": mesh_shape.get("fsdp", 1) * mesh_shape.get("fsdp_transpose", 1),
                        "tp": mesh_shape.get("tensor", 1), "ep": mesh_shape.get("expert", 1),
                        "cp": mesh_shape.get("context", 1), "pp": mesh_shape.get("stage", 1), "mesh": mesh_shape},
        "config": {"path": str(mt_common.HY3_YML), "sha256": sha_file(mt_common.HY3_YML),
                   "args": {k: v for k, v in vars(a).items() if k != "maxtext_args"},
                   "overrides": list(mt_common.FORCED) + list(a.maxtext_args),
                   # 与 GPU meta.config.args 同名的几项，config_diff 直接对
                   "gbs": c.global_batch_size_to_train_on, "seq": c.max_target_length, "lr": c.learning_rate,
                   "seed": c.init_weights_seed},
        "checkpoint": {"path": a.load, "manifest_sha256": sha_file(a.hf_manifest) if a.hf_manifest else None,
                       "manifest": a.hf_manifest},
        "data": {"manifest": str(Path(a.golden) / "data" / "batches"), "manifest_sha256": batches_sha,
                 "seed": c.data_shuffle_seed,
                 "replay_of": {"run": golden_meta.get("run_name"), "manifest": golden_meta.get("data", {}).get("manifest"),
                               "manifest_sha256": golden_meta.get("data", {}).get("manifest_sha256")},
                 "device_of_shard": shard_devices,
                 "rows": "全局第 r·M+m 行 = GPU rank r 的 micro m；batch 维第 d 个分片 = GPU rank d"},
        "steps": {"first": a.first_step, "last": a.steps, "convention": "golden_step = step"},
        "determinism": {**jd.env_snapshot(), "remat_policy": c.remat_policy, "moe_routing_maps": c.moe_routing_maps,
                        "attention": c.attention, "matmul_precision": c.matmul_precision},
        "dtypes": {"compute": str(c.dtype), "weight": (f"{c.weight_dtype} main params（前反向用其 BF16 舍入，--bf16-weights）" if a.bf16_weights
                                                      else f"{c.weight_dtype} main params（计算时转 {c.dtype}）"),
                   "optimizer_states": str(c.mu_dtype), "grad_reduce": str(c.grad_dtype)},
        "hooks": {"janus_dump": "janus.dump.maxtext", "dump_steps": a.dump_steps, "full": a.full,
                  "capture": "off" if a.no_hooks else "in-graph sow（激活）+ value_and_grad 输出（梯度）",
                  "selfcheck": None,
                  "build": "observe（每一步同一个加钩子的程序）" if a.observe else "production（往下训练用不加钩子的程序）"},
        "fault": ({"kind": "grad_nan", "step": a.inject_nan_grad, "target": a.nan_target} if a.inject_nan_grad else None),
        "model_config": model_config, "optimizer_config": optimizer_config, "scheduler_config": scheduler_config,
        "maxtext_config": maxtext_config,
        # 日常回归 R05 / R06：展开后的完整 MaxText 配置；运行进程里的开关（进程启动时一份，第一步编译执行完再取一份）
        "maxtext_config_full": maxtext_full_config(c),
        "flags": {"at_start": FLAG_AT_START, "after_compile": None},
        "restarts": 0,
        "created_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds"),
    }


FLAG_AT_START: dict = {}


def main(argv=None):
    global FLAG_AT_START
    a = parse_args(argv)
    if "://" not in a.load:
        a.load = str(Path(a.load).resolve())               # Orbax 要绝对路径
    if a.cpu_small:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        flags = os.environ.get("XLA_FLAGS", "")
        if "xla_force_host_platform_device_count" not in flags:
            os.environ["XLA_FLAGS"] = (flags + " --xla_force_host_platform_device_count=8").strip()
    import jax
    import jax.numpy as jnp
    from flax import nnx
    if a.jax_default_matmul_precision:
        jax.config.update("jax_default_matmul_precision", a.jax_default_matmul_precision)
    router_jitter_info = None
    if a.router_jitter:
        import router_jitter
        router_jitter_info = router_jitter.install(a.router_jitter)
    grad_fp32_info = None
    if a.grad_fp32:
        import grad_fp32

        from janus.dump.name_map import NameMap
        grad_fp32_info = grad_fp32.install(a.grad_fp32, int(NameMap.load(mt_common.NAME_MAP).dims["vocab"]))
    FLAG_AT_START = flag_snapshot()
    from maxtext.trainers.pre_train import train
    if a.head_matmul_precision or a.head_round:
        import types

        from maxtext.layers import linears as _lin
        from maxtext.layers import nnx_decoders as _nd

        class _HeadDense(_lin.DenseGeneral):
            def __init__(self, *args, **kw):
                if a.head_matmul_precision:
                    kw["matmul_precision"] = a.head_matmul_precision
                super().__init__(*args, **kw)

            def __call__(self, inputs, *args, **kw):
                rp = lambda x: jax.lax.reduce_precision(x, exponent_bits=8, mantissa_bits=7)  # noqa: E731
                if a.head_round == "input":
                    inputs = inputs + jax.lax.stop_gradient(rp(inputs) - inputs)
                if a.head_round in ("cot-dgrad", "cot-wgrad"):
                    return _head_dot(inputs, jnp.asarray(self.kernel[...], inputs.dtype))
                out = super().__call__(inputs, *args, **kw)
                if a.head_round == "cotangent":
                    out = _round_cotangent(out)
                return out

        @jax.custom_vjp
        def _round_cotangent(x):
            return x

        _round_cotangent.defvjp(lambda x: (x, None),
                                lambda _, g: (jax.lax.reduce_precision(g, exponent_bits=8, mantissa_bits=7),))

        _hp = jax.lax.Precision.HIGHEST

        @jax.custom_vjp
        def _head_dot(x, w):
            return jnp.einsum("...d,dv->...v", x, w, precision=_hp)

        def _head_dot_bwd(res, g):
            x, w = res
            gr = jax.lax.reduce_precision(g, exponent_bits=8, mantissa_bits=7)
            gd, gw = (gr, g) if a.head_round == "cot-dgrad" else (g, gr)
            dx = jnp.einsum("...v,dv->...d", gd, w, precision=_hp)
            dw = jnp.einsum("...d,...v->dv", x, gw, precision=_hp)
            return dx, dw

        _head_dot.defvjp(lambda x, w: (_head_dot(x, w), (x, w)), _head_dot_bwd)
        _proxy = types.ModuleType("linears_head_precision")
        _proxy.__dict__.update(_lin.__dict__)
        _proxy.DenseGeneral = _HeadDense
        _nd.linears = _proxy
    from maxtext.utils import sharding, train_utils

    # MaxText 在 TPU 上只有 enable_checkpointing=True 时才调 jax.distributed.initialize()（max_utils.
    # maybe_initialize_jax_distributed_system）；hy3-4layers.yml 关了存档，多主机时就没有分布式客户端，
    # 而 load_parameters_path 走 orbax v1 的 ocp.load，它的跨进程同步要求这个客户端（否则报
    # ThreadSafeKeyValueSignalingClient should only be used in a single controller setup）。
    # 所以在建配置（以及任何访问设备的调用）之前自己初始化；MaxText 随后看到已初始化会跳过。
    if not a.cpu_small and not jax.distributed.is_initialized():
        jax.distributed.initialize()

    dump_steps = _int_steps(a.dump_steps)
    gb = replay.GoldenBatches(a.golden)
    golden_meta = json.loads((Path(a.golden) / "meta.json").read_text())
    overrides = [f"run_name={Path(a.janus_run).name}", "base_output_directory=/tmp/hy3out", f"steps={a.steps}",
                 f"learning_rate_schedule_steps={a.steps}", f"load_parameters_path={a.load}"]
    if a.cpu_small:
        overrides += CPU_SMALL + [f"max_target_length={gb.global_batch(gb.steps[0], verify=False)['tokens'].shape[1]}"]
    config = mt_common.make_config(overrides + list(a.maxtext_args))
    if config.global_batch_size_to_train_on != gb.global_rows:
        raise SystemExit(f"MaxText 全局 batch {config.global_batch_size_to_train_on} ≠ golden 每步 {gb.global_rows} 行")
    pdbs = config.micro_batch_size_to_train_on // max(1, jax.device_count())
    micro = gb.micro

    nm = mt_common.name_map_for(config, strict=not a.cpu_small)

    jd.install_capture()                                     # 壳在钩子关着时原样转调，先装上
    (_, _, state_mesh_shardings, _, mesh, lr_schedule, _, _, _, _, state) = train_utils.setup_train_loop(config, None)
    tagged = jd.tag_model(state.model)
    jit_model, state = nnx.split(state)
    params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)
    p_train, _ = train_utils.jit_train_and_eval_step(config, jit_model, mesh, state, state_mesh_shardings,
                                                    make_bf16w_train_step(train) if a.bf16_weights else train.train_step,
                                                    params_shardings=params_shardings)
    n_exp = max(jd.MaxTextDump.DEFAULT_EXPERTS) + 1
    expert_paths = {jd.state_key(pc.path) for e in nm.entries if e.expert is not None for pc in e.pieces}

    def grad_filter(key):
        if key in expert_paths:
            return ((0, 0, n_exp),)                          # 专家叶子 [E, L, …]：只取专家 0…n_exp−1
        return ()
    def build_hooked(tier):
        hooked = jd.make_hooked_train_step(train, grad_filter=grad_filter if a.full else None, rows=pdbs, tier=tier)
        return train_utils.jit_train_and_eval_step(config, jit_model, mesh, state, state_mesh_shardings, hooked,
                                                  params_shardings=params_shardings)[0]
    if a.observe:
        if a.no_hooks:
            raise SystemExit("--observe 与 --no-hooks 矛盾：观测构建就是每一步都加钩子")
        if config.gradient_accumulation_steps != 1:
            raise SystemExit("--observe 要求 gradient_accumulation_steps=1（加钩子的训练步不支持梯度累积）")
        a.no_selfcheck = True
    if config.gradient_accumulation_steps != 1 and not a.no_hooks:
        # make_hooked_train_step 只支持不累积；GA>1 时只抓状态类张量（零扰动），激活另用 layer_replay.py
        print("[janus] gradient_accumulation_steps>1：加钩子的训练步不支持，按 --no-hooks 跑", flush=True)
        a.no_hooks = True
    p_hooked = build_hooked(a.capture) if (not a.no_hooks and (dump_steps or a.observe)) else None
    probe_tiers = [t for t in (a.probe_tiers or "").split(",") if t]
    p_nan = None
    if a.inject_nan_grad:
        p_nan = train_utils.jit_train_and_eval_step(config, jit_model, mesh, state, state_mesh_shardings,
                                                   make_nan_train_step(train, a.nan_target), params_shardings=params_shardings)[0]

    data_sharding = sharding.get_input_data_sharding(config, mesh)
    shape = (gb.global_rows, config.max_target_length)
    shard_devices = {}
    dmap = data_sharding.devices_indices_map(shape)
    starts = sorted({idx[0].start or 0 for idx in dmap.values()})
    per_shard = gb.global_rows // len(starts)              # 每个 batch 分片的行数（= 每 rank 的 microbatch 数时才对得上）
    if per_shard != micro:
        raise SystemExit(f"每个 batch 分片 {per_shard} 行，与 golden 每 rank {micro} 个 microbatch 对不上")
    for dev, idx in dmap.items():
        shard_devices.setdefault(str((idx[0].start or 0) // per_shard), []).append(
            {"id": dev.id, "process": dev.process_index, "coords": list(getattr(dev, "coords", []) or [])})

    def put(arrs: dict[str, np.ndarray]):
        return {k: jax.make_array_from_callback(v.shape, data_sharding, lambda i, v=v: v[i]) for k, v in arrs.items()}

    batches_sha = hashlib.sha256(b"".join((gb.dir / f"rank{r}.jsonl").read_bytes() for r in gb.ranks)).hexdigest()
    meta = build_meta(a, config, mesh, golden_meta, shard_devices, batches_sha)
    if router_jitter_info:
        meta["router_jitter"] = router_jitter_info
    if grad_fp32_info:
        meta["grad_fp32"] = grad_fp32_info
    probe_steps = _int_steps(a.grad_probe_at)
    meta["grad_probe"] = {"steps": probe_steps, "ref_overrides": a.probe_ref.split(",") if probe_steps else None,
                          "ref_weights": a.probe_ref_weights,
                          "state": "第 s 步训练之前的状态 θ_{s-1}，第 s 步的 batch（= 生产训练步第 s 步用的梯度的输入）"}
    dump = jd.MaxTextDump(a.janus_run, meta, nm, dump_steps=sorted(set(dump_steps) | set(probe_steps)), full=a.full,
                          full_steps=_int_steps(a.full_steps) or None, process_index=jax.process_index(), micro=micro,
                          routed_scaling=float(config.routed_scaling_factor), only=a.full_only,
                          acts_steps=None if a.acts_steps is None else _int_steps(a.acts_steps))
    if a.params_from:
        import param_io
        t0_ = time.time()
        leaves, info = param_io.load_leaves(a.params_from, nm, nm.leaf_shapes())
        nset = param_io.set_state(state, leaves)
        del leaves
        ver = param_io.verify(dump, nm, mt_common.variable_paths(state), info.pop("manifest"))
        meta["params_from"] = {"path": a.params_from, **info, "leaves_set": nset, "verify": ver,
                               "seconds": time.time() - t0_}
        if jax.process_index() == 0:
            print(f"[janus] --params-from：{json.dumps(meta['params_from'], ensure_ascii=False, default=str)}", flush=True)
        if ver["mismatch"]:
            raise SystemExit("--params-from：放回后读出与 manifest 不一致")
    if a.resume_state:
        t0_ = time.time()
        meta["resume"] = {**load_train_state(state, a.resume_state), "first_step": a.first_step, "seconds": time.time() - t0_}
        dump.write_state_sha(mt_common.variable_paths(state), f"step{a.first_step - 1}")
        if jax.process_index() == 0:
            print(f"[janus] 从存档恢复：{json.dumps(meta['resume'], ensure_ascii=False)}", flush=True)
    export_steps = _int_steps(a.export_params_at)
    flat = mt_common.variable_paths(state)
    dump.write_modules(tagged, flat)
    if not a.no_param_sha:
        dump.write_param_sha(flat, "init")
    if a.state_sha:
        dump.write_state_sha(flat, "init")
    bias_key = next(k for k in flat if k.startswith("MoEBiasVar/") and k.endswith("/gate/bias"))
    fd = int(nm.dims.get("first_dense", 0))

    def leaves_equal(s1, s2):
        """两份训练状态逐叶子逐位比（在设备上比，只把布尔值取回主机）。"""
        def eq(x, y):
            if jnp.issubdtype(x.dtype, jnp.floating):
                bits = {2: jnp.uint16, 4: jnp.uint32, 8: jnp.uint64}[x.dtype.itemsize]
                return jnp.all(jax.lax.bitcast_convert_type(x, bits) == jax.lax.bitcast_convert_type(y, bits))
            if jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
                return jnp.array(True)
            return jnp.all(x == y)
        f1, f2 = mt_common.variable_paths(s1), mt_common.variable_paths(s2)
        out = {}
        for k in sorted(f1):
            if hasattr(f1[k], "dtype"):
                out[k] = bool(jd.to_host(jax.jit(eq)(f1[k], f2[k])))
        return out

    group_sq_fn, group_keys = None, []
    if a.group_norms:
        flat0 = mt_common.variable_paths(state)
        group_keys = sorted(k for k in flat0 if k.startswith("params/") and group_of(k))
        unmapped = sorted(k for k in flat0 if k.startswith("params/") and not group_of(k))
        if unmapped:
            raise SystemExit(f"--group-norms：这些参数没有分组规则：{unmapped}")
        n_scan = {"dense_layers": fd, "moe_layers": config.num_decoder_layers - fd}

        @jax.jit
        def group_sq_fn(tree):
            out = {}
            for k, v in tree.items():
                v = v.astype(jnp.float32)
                stack = next((n for n in n_scan if f"/{n}/" in k), None)
                if stack is not None:                    # 扫描层：第 1 维是层（param_scan_axis = 1），按层求和
                    out[k] = jnp.sum(jnp.square(v), axis=tuple(i for i in range(v.ndim) if i != 1))
                    if group_of(k).startswith("experts_"):  # [专家, 层, …] → 每个专家每层
                        out[k + "#expert"] = jnp.sum(jnp.square(v), axis=tuple(range(2, v.ndim)))
                else:
                    out[k] = jnp.sum(jnp.square(v))
            return out

        def group_sq_values(res):
            vals_ = {}
            for k, v in res.items():
                v = np.asarray(jd.to_host(v), np.float64)
                base = k.split("#")[0]
                grp = group_of(base)
                stack = next((n for n in n_scan if f"/{n}/" in base), None)
                off = 0 if stack == "dense_layers" else fd
                if k.endswith("#expert"):
                    for li in range(v.shape[1]):
                        key = f"expert_sq.{li + off}"
                        vals_[key] = list(np.asarray(vals_.get(key, np.zeros(v.shape[0]))) + v[:, li])
                elif stack is None:
                    vals_[f"group_sq.{grp}"] = vals_.get(f"group_sq.{grp}", 0.0) + float(v)
                else:
                    for li in range(v.shape[0]):
                        key = f"group_sq.{grp}.{li + off}"
                        vals_[key] = vals_.get(key, 0.0) + float(v[li])
            return {k: ([float(x) for x in v] if isinstance(v, list) else v) for k, v in vals_.items()}

    @jax.jit
    def row_stats_fn(w, mu, nu, emb, t):
        """输出层 kernel [D, V]（MaxText 的方向），embedding [V, D]。Adam 更新量带偏差修正，ε 加在根号外（与 optax 相同）。"""
        w, mu, nu, emb = (x.astype(jnp.float32) for x in (w, mu, nu, emb))
        u = (mu / (1 - config.adam_b1 ** t)) / (jnp.sqrt(nu / (1 - config.adam_b2 ** t)) + config.adam_eps)
        g = jnp.sqrt(nu / (1 - config.adam_b2 ** t))
        return {"output_row_sq": jnp.sum(w * w, axis=0), "output_row_u": jnp.mean(jnp.abs(u), axis=0),
                "output_row_g": jnp.mean(g, axis=0), "output_row_wu": jnp.sum(w * u, axis=0),
                "embedding_row_sq": jnp.sum(emb * emb, axis=1)}

    probe_fn, probe_batch = None, None
    if a.resid_probe_every:
        g1 = replay.to_maxtext(gb.global_batch(gb.steps[0], verify=False))
        probe_batch = put({k: np.ascontiguousarray(np.broadcast_to(v[:1], v.shape)) for k, v in g1.items()})

        def probe_fwd(st, batch):
            m = nnx.merge(jit_model, st).model
            with jd.capturing(rows=1, tags={"layer", "final_norm"}, xent=False):
                logits = m(decoder_input_tokens=batch["inputs"], decoder_positions=batch["inputs_position"],
                           decoder_segment_ids=batch["inputs_segmentation"], enable_dropout=False,
                           decoder_target_tokens=batch["targets"], decoder_target_mask=batch["targets_segmentation"])
                inter = nnx.pop(m, nnx.Intermediate)
            sown = jd.collect_sown(inter.to_pure_dict())
            # 只取第 0 行（各行相同）在 token 上求均值：扫描层的值是 [层, 行, S, D] 或 [行, S, D]
            means = {k: jnp.mean(v.astype(jnp.float32)[..., 0:1, :, :], axis=-2)[..., 0, :] for k, v in sown.items()}
            # 每个 token 的 RMS（第 0 行）：[层, S] 或 [S]
            means.update({k + "#rms": jnp.sqrt(jnp.mean(jnp.square(v.astype(jnp.float32)[..., 0, :, :]), axis=-1))
                          for k, v in sown.items()})
            l0 = logits[0].astype(jnp.float32)
            lse = jax.scipy.special.logsumexp(l0, axis=-1)
            return means, jnp.mean(l0), jnp.mean(lse)

        probe_fn = jax.jit(probe_fwd)

    def probe_values(res):
        means, lmean, lsemean = res
        out = {"probe_logit_mean": float(jd.to_host(lmean)), "probe_lse_mean": float(jd.to_host(lsemean))}
        for k, v in means.items():
            v = np.asarray(jd.to_host(v), np.float64)
            rms = k.endswith("#rms")
            name = k.replace("#rms", "").replace("/janus__", ".").replace("/", ".")
            pre = "tok_rms." if rms else "resid_mean."
            if v.ndim == 2:                                  # 扫描层：[层, D]（RMS 是 [层, S]）
                for li in range(v.shape[0]):
                    out[f"{pre}{name}.{li}"] = [float(x) for x in v[li]]
            else:
                out[f"{pre}{name}"] = [float(x) for x in v]
        return out

    probe_fns = None
    if probe_steps:
        from flax.nnx import variablelib as _vl
        from maxtext.utils import model_creation_utils
        owg = _vl.variable_type_from_name("_overwrite_with_gradient", allow_register=True)
        cfilter = nnx.Any(owg)
        ref_config = mt_common.make_config(overrides + list(a.maxtext_args) + [x for x in a.probe_ref.split(",") if x])
        _, ref_abs = model_creation_utils.create_nnx_abstract_model(ref_config, mesh)
        ref_gd = nnx.split(ref_abs, nnx.Param, cfilter, ...)[0]
        out_key = next(k for k in mt_common.variable_paths(state) if k.endswith("decoder/logits_dense/kernel"))
        out_key = "params/model/" + out_key.split("/model/", 1)[1]

        def _select(g):
            """梯度树 → {状态键: 数组}，只留要写的：--full-only 选中的非专家参数（Megatron 名），专家叶子在图里切成专家 0、1。"""
            pure = nnx.to_pure_dict(g) if isinstance(g, nnx.State) else g
            out = {}
            for path, v in jd._flat(pure):
                key = "params/model/" + "/".join(str(x) for x in path)
                if key == out_key:                           # 输出层 kernel [D, V]：沿词表求和、逐行（每个词）平方和
                    vf = v.astype(jnp.float32)
                    out["#out_colsum"] = jnp.sum(vf, axis=1)
                    out["#out_rowsq"] = jnp.sum(vf * vf, axis=0)
                if key not in probe_keep:
                    continue
                if key in expert_paths:
                    v = jax.lax.slice_in_dim(v, 0, n_exp, axis=0)
                out[key] = v
            return out

        def _make(cfg_, gd_, level=False, round_w=False):
            def f(params, custom, rest, data):
                if round_w:                                   # 主权重舍到 BF16：与生产程序计算时用的那份权重相同
                    # 用 reduce_precision，不用 astype(bf16).astype(f32)：TPU 上 XLA 默认允许 excess precision，
                    # 会把一对来回的 convert 消掉（t-sd-300b / t-sw-g99 实测参照损失等于 FP32 主权重的值）
                    params = jax.tree.map(lambda x: jax.lax.reduce_precision(x, exponent_bits=8, mantissa_bits=7)
                                          if x.dtype == jnp.float32 else x, params)
                def lossf(p_, c_):
                    m_ = nnx.merge(gd_, p_, c_, rest, copy=True)
                    if level:                                 # logits 在 token 与词表上的均值
                        lg = m_(decoder_input_tokens=data["inputs"], decoder_positions=data["inputs_position"],
                                decoder_segment_ids=data["inputs_segmentation"], enable_dropout=False,
                                decoder_target_tokens=data["targets"], decoder_target_mask=data["targets_segmentation"])
                        return jnp.mean(lg.astype(jnp.float32))
                    loss_, _aux = train.loss_fn(m_, cfg_, dict(data), None, None, is_train=True)
                    return loss_
                val, (g, _) = jax.value_and_grad(lossf, argnums=(0, 1))(params, custom)
                g = jax.tree.map(lambda x: x.astype(cfg_.grad_dtype) if x.dtype == jnp.float32 else x, g)
                return val, _select(g)
            return jax.jit(f)

        def _split_model(st):
            mdl = nnx.merge(jit_model, st).model
            nnx.pop(mdl, nnx.Intermediate)
            return nnx.split(mdl, nnx.Param, cfilter, ...)

        prod_gd = _split_model(state)[0]
        rw = a.probe_ref_weights == "bf16"
        probe_fns = {"bf16": _make(config, prod_gd), "f32": _make(ref_config, ref_gd, round_w=rw),
                     "lvl": _make(ref_config, ref_gd, level=True, round_w=rw)}
        # 要写的非专家参数：按 Megatron 名用 --full-only 过滤（与全量抓数同一个过滤）；专家只留 0、1
        probe_keep = set()
        for ent in nm.entries:
            if ent.buffer or (ent.expert is not None and ent.expert >= n_exp):
                continue
            if not dump._only(nm.megatron_name(ent)):
                continue
            probe_keep |= {jd.state_key(pc.path) for pc in ent.pieces}

    def run_probes(st, batch):
        _, params_, custom_, rest_ = _split_model(st)
        vals_, tens_ = {}, {}
        for kind, fn in probe_fns.items():
            t0_ = time.time()
            with jax.set_mesh(mesh), train.logical_axis_rules(config.logical_axis_rules):
                v_, g_ = fn(params_, custom_, rest_, batch)
            vals_[f"gp_{'lvl' if kind == 'lvl' else 'loss_' + kind}"] = float(jd.to_host(v_))
            host = {k: jd.to_host(x) for k, x in sorted(g_.items())}
            tens_[f"gp_{kind}.output_layer.colsum"] = np.asarray(host.pop("#out_colsum"), np.float32)
            tens_[f"gp_{kind}.output_layer.rowsq"] = np.asarray(host.pop("#out_rowsq"), np.float32)
            for k, x in dump._grads_megatron(host).items():
                tens_[f"gp_{kind}." + k[len("grad."):]] = x
            vals_[f"gp_seconds_{kind}"] = time.time() - t0_
            del g_, host
        return vals_, tens_

    selfcheck = []
    t_start = time.time()
    for s in range(a.first_step, a.steps + 1):
        g = gb.global_batch(s)                               # 每个字段核对 sha256
        batch = put(replay.to_maxtext(g))
        fed = replay.from_maxtext({k: jd.to_host(v) for k, v in batch.items()})   # 从设备读回来再算指纹
        dump.record_batch(s, replay.fingerprints(fed, micro, s))
        if s in dump_steps:
            dump.capture_before(s, mt_common.variable_paths(state))
        probe_vals = {}
        if s in probe_steps:
            probe_vals, probe_tens = run_probes(state, batch)
            dump.add_tensors(probe_tens)
            if jax.process_index() == 0:
                print(f"[janus] 第 {s} 步梯度探针：" + json.dumps(probe_vals), flush=True)
        t0 = time.time()
        with jax.set_mesh(mesh), train.logical_axis_rules(config.logical_axis_rules):
            if p_nan is not None and s == a.inject_nan_grad:
                state, metrics = p_nan(state, batch)                 # 演练步：梯度里埋了 NaN
            elif p_hooked is not None and a.observe:
                # 观测构建：每一步同一个程序；非抓数步的额外输出不取回（capture_after 只在抓数步读它）
                state, metrics = p_hooked(state, batch)
            elif s in dump_steps and p_hooked is not None and a.no_selfcheck:
                state, metrics = p_hooked(state, batch)
            elif s in dump_steps and p_hooked is not None:
                # 往下训练一律用不加钩子的程序（生产图）的输出；加钩子的程序在同一输入状态的副本上跑，只取它的额外输出。
                # 两者输出的训练状态与标量逐位相同时，额外输出才算「生产图里的值」；不同时照写，但 meta 标为诊断值。
                tiers = [(a.capture, p_hooked)]
                if s == min(dump_steps):
                    tiers += [(t, None) for t in probe_tiers if t != a.capture]
                def copy_state(st):
                    return jax.tree.map(lambda x: x.copy() if hasattr(x, "copy") else x, st)
                base = copy_state(state)                       # 输入状态的原样副本（训练步会捐出输入缓冲）
                state, metrics = p_train(state, batch)
                extras = None
                for tier, prog in tiers:                       # 一次只多一份副本，控制 HBM
                    twin = copy_state(base)
                    prog = prog or build_hooked(tier)
                    h_state, h_metrics = prog(twin, batch)
                    eq = leaves_equal(state, h_state)
                    sm = {k: (float(jd.to_host(metrics["scalar"][k])), float(jd.to_host(h_metrics["scalar"][k])))
                          for k in ("learning/loss", "learning/raw_grad_norm", "learning/grad_norm", "learning/param_norm")}
                    rec = {"step": s, "tier": tier, "state_leaves": len(eq), "state_differ": [k for k, v in eq.items() if not v],
                           "scalars": {k: {"prod": x, "hooked": y} for k, (x, y) in sm.items()},
                           "scalars_differ": [k for k, (x, y) in sm.items() if np.float32(x).tobytes() != np.float32(y).tobytes()]}
                    rec["equal"] = not rec["state_differ"] and not rec["scalars_differ"]
                    selfcheck.append(rec)
                    print(f"[janus] 第 {s} 步钩子自检（{tier}）：" + ("逐位相同" if rec["equal"] else
                          f"不同：状态 {len(rec['state_differ'])}/{len(eq)} 个叶子，标量 {rec['scalars_differ']}，"
                          f"loss {sm['learning/loss']}"), flush=True)
                    if tier == a.capture:
                        extras = h_metrics.get("janus")
                    del h_state, h_metrics, twin
                del base
                metrics = dict(metrics)
                metrics["janus"] = extras
            else:
                state, metrics = p_train(state, batch)
        sc = {k: float(jd.to_host(v)) for k, v in metrics["scalar"].items()}
        step_time = time.time() - t0
        if s == 1:
            meta["flags"]["after_compile"] = flag_snapshot()
        flat = mt_common.variable_paths(state)
        vals = {"lm_loss": sc["learning/lm_loss"], "loss": sc["learning/loss"],
                # GPU golden 的 grad_norm 是 Megatron 裁剪之前的总范数（golden 第 1 步 17.5，clip 1.0），这里同义
                "grad_norm": sc["learning/raw_grad_norm"], "raw_grad_norm": sc["learning/raw_grad_norm"],
                "grad_norm_clipped": sc["learning/grad_norm"], "param_norm": sc["learning/param_norm"],
                "lr": float(lr_schedule(s - 1)), "wd": float(config.adam_weight_decay), "skipped_iter": 0.0,
                "total_weights": sc["learning/total_weights"], "step_time": step_time}
        vals.update(probe_vals)
        vals.update(jd.bias_stats(jd.to_host(flat[bias_key]), fd))
        if group_sq_fn is not None:
            vals.update(group_sq_values(group_sq_fn({k: flat[k] for k in group_keys})))
        if probe_fn is not None and (s % a.resid_probe_every == 0 or s == 1):
            with jax.set_mesh(mesh), train.logical_axis_rules(config.logical_axis_rules):
                vals.update(probe_values(probe_fn(state, probe_batch)))
            nk = next(k for k in flat if k.startswith("params/") and k.endswith("decoder/decoder_norm/scale"))
            vals["final_norm_scale"] = [float(x) for x in np.asarray(jd.to_host(flat[nk]), np.float64)]
        if a.row_stats_every and s % a.row_stats_every == 0:
            wk = next(k for k in flat if k.startswith("params/") and k.endswith("decoder/logits_dense/kernel"))
            ek = next(k for k in flat if k.startswith("params/") and k.endswith("token_embedder/embedding"))
            mk = next(k for k in flat if "/mu/" in k and k.endswith("decoder/logits_dense/kernel"))
            nk = next(k for k in flat if "/nu/" in k and k.endswith("decoder/logits_dense/kernel"))
            rs = row_stats_fn(flat[wk], flat[mk], flat[nk], flat[ek], jnp.float32(s))
            vals.update({k: [float(x) for x in np.asarray(jd.to_host(v), np.float64)] for k, v in rs.items()})
        dump.log_metrics(s, vals)
        if s in dump.dump_steps:
            dump.capture_after(s, flat, metrics.get("janus"))
            dump.flush(s)
        if a.save_state_at and s == a.save_state_at:
            if not a.state_dir:
                raise SystemExit("--save-state-at 要给 --state-dir")
            t0_ = time.time()
            meta["saved_state"] = {**save_train_state(state, a.state_dir), "step": s, "seconds": time.time() - t0_}
            dump.write_state_sha(flat, f"step{s}")
            if jax.process_index() == 0:
                print(f"[janus] 第 {s} 步存档：{json.dumps(meta['saved_state'], ensure_ascii=False)}", flush=True)
        if s in export_steps:
            import param_io
            param_io.export(dump, nm, flat, Path(a.janus_run) / "export" / f"step_{s}", s,
                            {"framework": "maxtext", "run": Path(a.janus_run).name}, jax.process_index())
        if jax.process_index() == 0:
            print(f"[janus] step {s} lm_loss {vals['lm_loss']!r} grad_norm {vals['grad_norm']!r} "
                  f"({step_time:.2f} s)", flush=True)
    if not a.no_param_sha:
        dump.write_param_sha(mt_common.variable_paths(state), "final")
    if a.state_sha:
        dump.write_state_sha(mt_common.variable_paths(state), "final")
    if jax.process_index() == 0:
        if a.hlo_dir:
            meta["hlo"] = collect_hlo(a.hlo_dir, a.janus_run)
        meta["hooks"]["selfcheck"] = selfcheck
        meta["hooks"]["capture"] = a.capture
        meta["hooks"]["extras_are"] = ("观测构建的值：整条轨迹就是这个加钩子的程序，额外输出是它自己的中间值" if a.observe
                                       else "生产图里的值（自检逐位相同）" if selfcheck and all(r["equal"] for r in selfcheck if r["tier"] == a.capture)
                                       else "诊断值：加钩子的程序与生产图输出不逐位相同，激活与梯度来自加钩子的程序；"
                                            "训练状态、逐步标量、状态类张量仍来自生产图")
        meta["hooks"]["tagged_modules"] = len(tagged)
        meta["wall_seconds"] = time.time() - t_start
        try:                                                 # 显存：本进程每个设备的峰值（步时 / 显存代价对比用）
            meta["memory"] = {str(d.id): {k: v for k, v in (d.memory_stats() or {}).items()
                                          if k in ("peak_bytes_in_use", "bytes_limit", "bytes_in_use")} for d in jax.local_devices()}
        except Exception as e:  # noqa: BLE001  有的后端没有 memory_stats
            meta["memory"] = {"error": str(e)}
        (Path(a.janus_run) / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    dump.on_train_end()
    bad = [r for r in selfcheck if not r["equal"] and r["tier"] == a.capture]
    if bad:
        print(f"[janus] 钩子自检不通过（{a.capture}）：抓到的激活 / 梯度是诊断值；训练轨迹不受影响（用的是生产图）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
