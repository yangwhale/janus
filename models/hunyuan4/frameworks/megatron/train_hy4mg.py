"""混元 4 Megatron 版训练入口：Megatron-Core 0.20.0，B200 × 卡数（单机或多机），DP × EP × PP，写 janus 运行目录。

用法（torchrun，NeMo 26.10 镜像；单机 / 多机的环境见 pod_run.sh）：
    torchrun --nproc_per_node 8 [--nnodes N --node_rank R --master_addr A --master_port P] train_hy4mg.py \
        --hf <截好的 HF 目录> --tokens tokens.npy --layers 1 --steps 10 --name <运行名> --out <运行根目录> \
        [--mtp] [--recompute] [--ep 4] [--pp 2 --pp-first 5] [--deterministic] [--save-at 10] [--resume <存档目录>]

并行：DP = 卡数 ÷ PP，全局批固定 16，GA = 16 ÷ DP（DP 要整除 16）；EP 在 DP 里切；PP 的切分点要落在 full 层上
（hy4_mcore.check_pp_split）。存档：--save-at 的步做完后每个 rank 存一份（模型、优化器含 Muon 动量与 FP32 主权重、
RNG、下一步的步号即数据游标），--resume 从存档的下一步接着跑，要求并行设置与存档相同。

设置（T1 的取值见 models/hunyuan4/alignments/01-mini/config-alignment.md 与 PROTOCOL.md 第 1.2 节；下面写的是命令行默认值）：
- 数据：tokens.npy [样本, seq+1]，第 s 步第 m 个 microbatch、DP rank r 取第 (s-1)·16 + m·N + r 条（N = 卡数），
  tokens = 前 seq 个，labels = 后 seq 个；每步每 rank 每个 microbatch 记 sha256（data/rank<r>.jsonl）。
- 优化器：Megatron 分布式优化器（Adam，AdamW 解耦衰减），lr 常数、β 0.9 / 0.95、ε 1e-8、权重衰减 0.1
  **对全部参数**（不按「一维参数与偏置不衰减」分组，HF golden 同）；梯度裁剪 1.0（grad_norm 记裁剪前）。
  --indexer-loss 0 时索引器参数 requires_grad=False（AdamW 时期）；默认 0.1：Megatron-Core 的 DSA 索引器 KD loss
  训练索引器，口径同 MaxText（产出层取平均再乘系数，见 hy4_mcore.indexer_loss_coeff_per_layer）。
- 开训前核权重（LOAD_CHECK，不过就退出）：同一张映射逐个核参数 == checkpoint 张量（转成参数 dtype），两边 FP64 平方和相等；
  checkpoint 张量本身在 Pod 里先按 tensor_sha256.json 核过（L0）。
- 精度：主权重 FP32、计算 BF16、梯度累加与 DP 归约 FP32；iHC、hc_head、sinks、输出头留 FP32（hy4_mcore.py）。
- 路由偏置：--bias-rate（默认 1e-3）由 Megatron 的 finalize_model_grads 按全局批更新（bias += rate·sign(平均负载 − 负载)）。
- QK-Clip（--qk-clip）：Megatron-Core 的 clip_qk 遍历主干；--qk-clip-mtp on（默认，同 MaxText）时 MTP 层照同一流程裁。
- 生产配方开关（默认都关，T1 不用；台账 G35）：--lr-schedule cosine（照 MaxText 的 cosine 调度逐步设学习率）、
  --yarn-factor（RoPE 频率做 YaRN 插值，不乘 mscale）、--bias-zero-mean on（路由偏置更新量减去对专家的均值）。

指标（metrics.jsonl）：loss、mtp_loss、indexer_loss（MaxText 口径）、grad_norm、param_norm（FP32 主权重）、lr、
step_time_s、peak_alloc_gib、router.bias_mean / bias_std（每个 MoE 层）、max_attention_logit。
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GBS = 16  # 全局批固定 16；DP = 卡数，GA = 16 / DP（hy4-mini 在 1–4 卡上跑时累积步数随之变）


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def tf32_probe(dev) -> dict:
    x = torch.randn(4096, 4096, device=dev, dtype=torch.float64)
    y = torch.randn(4096, 4096, device=dev, dtype=torch.float64)
    ref = x @ y
    got = (x.float() @ y.float()).double()
    return {
        "fp32_gemm_rel_err_vs_fp64": float((got - ref).norm() / ref.norm()),
        "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "torch_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def maxtext_lr(count: int, peak: float, warmup: int, final_frac: float, length: int) -> float:
    """MaxText create_learning_rate_schedule（lr_schedule_type cosine）第 count 次更新（从 0 起）的学习率，FP32 照写：
    warmup 段 optax.linear_schedule(0 → peak, W)：(0 − peak)·(1 − c / W) + peak；余弦段 c' = c − W，
    pct = c' / (L − W − 1)，a = 0.5·(cos(π·pct) + 1)，lr = peak·a + final·(1 − a)。"""
    f = np.float32
    pk, fin = f(peak), f(peak) * f(final_frac)
    if warmup > 0 and count < warmup:
        frac = f(1) - f(count) / f(warmup)
        return float((f(0) - pk) * frac + pk)
    n = length - warmup
    c = count - warmup
    pct = f(c) / f(n - 1) if n > 1 else f(1)
    a_ = f(0.5) * (np.cos(f(np.pi) * pct, dtype=np.float32) + f(1))
    return float(pk * a_ + fin * (f(1) - a_))


def apply_yarn(model, base: float, factor: float, orig_max: int, beta_fast: float, beta_slow: float) -> dict:
    """把模型里每个 RotaryEmbedding（MLA 与 MTP 层各一份；索引器用所在层 MLA 算出的那份）的 inv_freq 换成
    YaRN 插值后的频率，写法逐项照 MaxText YarnRotaryEmbedding.freqs_cis（truncate：下界 floor、上界 ceil）。
    只换频率：Megatron 的 rope_type 仍是 rope，不走 YarnRotaryEmbedding（那条路会把 mscale 乘到 cos / sin 上）。"""
    import math

    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

    n, info = 0, {}
    for mod in model.modules():
        if type(mod) is not RotaryEmbedding:
            continue
        inv = mod.inv_freq
        dim = 2 * inv.numel()

        def corr(r):
            return dim * math.log(orig_max / (r * 2 * math.pi)) / (2 * math.log(base))

        low, high = max(math.floor(corr(beta_fast)), 0), min(math.ceil(corr(beta_slow)), dim - 1)
        hi = high + 0.001 if low == high else high
        ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32, device=inv.device) - low) / (hi - low), 0, 1)
        smooth = 1 - ramp
        mod.inv_freq = inv / factor * (1 - smooth) + inv * smooth
        n += 1
        info = {
            "dim": dim,
            "base": base,
            "low": low,
            "high": high,
            "factor": factor,
            "original_max": orig_max,
            "beta_fast": beta_fast,
            "beta_slow": beta_slow,
            "mscale": "none",
        }
    assert n > 0, "模型里没有 RotaryEmbedding"
    RotaryEmbedding.forward.cache_clear()
    # 自检：前向真的用上了新频率（emb 前一半 = 位置 × 新频率）
    for mod in model.modules():
        if type(mod) is RotaryEmbedding:
            emb = mod(16)
            emb = emb[0] if isinstance(emb, tuple) else emb
            want = torch.outer(torch.arange(16, dtype=torch.float32, device=mod.inv_freq.device), mod.inv_freq)
            assert torch.equal(emb.reshape(16, -1)[:, : dim // 2].float(), want), "YaRN 频率没有用上"
            RotaryEmbedding.forward.cache_clear()
            info["forward_check"] = "emb == outer(pos, yarn_inv_freq)"
            break
    info["modules"] = n
    return info


def install_bias_zero_mean():
    """Megatron-Core 的路由偏置更新改成零均值：更新量 rate·sign(…) 先减去对专家（最后一维）的均值，再加到偏置上
    （MaxText _maybe_zero_mean_bias_update 的写法）。finalize_model_grads 按模块名引用 get_updated_expert_bias，换它。"""
    import importlib

    # 包 megatron.core.distributed 把同名函数 finalize_model_grads 导出到包上，「import … as」拿到的是函数不是模块
    fmg = importlib.import_module("megatron.core.distributed.finalize_model_grads")
    assert hasattr(fmg, "_update_router_expert_bias"), fmg

    def upd(tokens_per_expert, expert_bias, rate, tp_dp_cp_group=None):
        from megatron.core import parallel_state

        with torch.no_grad():
            if tp_dp_cp_group is None:
                tp_dp_cp_group = parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True)
            dist.all_reduce(tokens_per_expert, group=tp_dp_cp_group)
            ne = tokens_per_expert.shape[-1]
            total = tokens_per_expert.sum(dim=-1, keepdim=True)
            u = torch.sign(total - tokens_per_expert * ne).to(dtype=expert_bias.dtype) * rate
            u = u - u.mean(dim=-1, keepdim=True)
            if not upd.logged and dist.get_rank() == 0:
                upd.logged = True
                print(
                    "BIAS_ZERO_MEAN",
                    json.dumps({"shape": list(u.shape), "update_mean_absmax": float(u.mean(-1).abs().max())}),
                    flush=True,
                )
            return expert_bias + u

    upd.logged = False

    fmg.get_updated_expert_bias = upd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--mtp-factor", type=float, default=None, help="默认取 config 的 mtp_loss_factor")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--head-chunk", type=int, default=1024, help="输出头 dgrad 分段长度；0 = 一次乘完")
    ap.add_argument("--bias-rate", type=float, default=1e-3, help="路由偏置更新率（DeepSeek-V3 公开值，项目负责人定）")
    ap.add_argument(
        "--bias-zero-mean",
        default="off",
        choices=["on", "off"],
        help="路由偏置零均值更新：每层的更新量先减去对专家取的均值再加到偏置上（同 MaxText routed_bias_zero_mean_update；"
        "原版 Megatron-Core 没有，T1 关）",
    )
    ap.add_argument(
        "--lr-schedule",
        default="constant",
        choices=["constant", "cosine"],
        help="constant：每步都是 --lr（T1）；cosine：照 MaxText create_learning_rate_schedule（lr_schedule_type cosine）"
        "——第 c 次更新（c 从 0 起）先线性 warmup lr·c / W，再从 lr 余弦衰减到 lr·final，长度 = --steps",
    )
    ap.add_argument("--warmup-steps", type=int, default=0, help="--lr-schedule cosine 的 warmup 步数 W")
    ap.add_argument("--lr-final-fraction", type=float, default=0.1, help="--lr-schedule cosine 的终点 = lr × 它")
    ap.add_argument(
        "--yarn-factor",
        type=float,
        default=1.0,
        help="YaRN 频率插值因子（MLA 与索引器共用的 RoPE）；1 = 普通 RoPE（T1）。只改频率，照 MaxText YarnRotaryEmbedding："
        "freq / factor · (1 − smooth) + freq · smooth；不乘 mscale（MaxText 在 max_position == original_max 时不改 softmax 缩放）",
    )
    ap.add_argument("--yarn-orig-max", type=int, default=1048576, help="YaRN 的 original_max_position_embeddings")
    ap.add_argument("--yarn-beta-fast", type=float, default=32.0)
    ap.add_argument("--yarn-beta-slow", type=float, default=1.0)
    ap.add_argument(
        "--indexer-loss",
        type=float,
        default=0.1,
        help="索引器 KD 系数，MaxText 口径（各产出层取平均再乘它）；逐层系数 = 它 ÷ 产出层数。0 = 冻结索引器",
    )
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--pp", type=int, default=1, help="流水线并行段数；DP = 卡数 ÷ PP")
    ap.add_argument(
        "--pp-first", type=int, default=0, help="首段层数（Megatron num_layers_in_first_pipeline_stage）；0 = 均分"
    )
    ap.add_argument("--pp-last", type=int, default=0, help="末段层数（num_layers_in_last_pipeline_stage）；0 = 均分")
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Megatron-Core 确定性模式（同 Megatron-Bridge：deterministic_mode、torch 确定性算法、固定 Triton 调优；"
        "NCCL_ALGO 等环境变量缺省时补上）：同配置重跑逐位相同，存档续跑的逐位检查用",
    )
    ap.add_argument("--save-at", default="", help="逗号或 + 分隔的步：这一步做完后存档（每个 rank 一个文件）")
    ap.add_argument("--save-dir", default="", help="存档根目录；空 = <运行目录>/ckpt，下面是 step_<N>/rank<r>.pt")
    ap.add_argument("--resume", default="", help="存档目录（…/step_<N>）：载入后从第 N+1 步跑到 --steps")
    ap.add_argument(
        "--state-sha",
        action="store_true",
        help="最后一步做完后每个 rank 写 state_sha/rank<r>.json：模型与优化器全部状态（含 FP32 主权重、Muon 动量、Adam 矩）"
        "逐张量 sha256，存档续跑的逐位检查用",
    )
    ap.add_argument("--recompute", action="store_true", help="每个解码层整层重算（Megatron full / uniform / 1）")
    ap.add_argument("--optimizer", default="adam", choices=["adam", "muon"])
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--adam-beta2", type=float, default=0.95)
    ap.add_argument(
        "--opt-kv",
        default="",
        help="其余 OptimizerConfig 字段，逗号分隔的 k=v（如 muon_momentum=0.95,muon_nesterov=true），值按 Python 字面量解析",
    )
    ap.add_argument(
        "--dist-opt",
        default="auto",
        choices=["auto", "on", "off"],
        help="Megatron 分布式优化器：auto = adam 开、muon 关（Muon 走整参数的 main_param，QK-Clip 才能就地改）",
    )
    ap.add_argument(
        "--muon-exclude-ihc",
        default="ihc+head",
        choices=["none", "ihc", "ihc+head"],
        help="不走 Muon 的 iHC 参数：项目负责人定 iHC 的 pre/post 混合与 hc_head 的 hc_fn 都给 Adam（ihc+head）",
    )
    ap.add_argument(
        "--muon-gate",
        default="per_head",
        choices=["whole", "per_head"],
        help="输出门在 Muon 里每头一个（登记设置：项目负责人 13:40 定生产语义不明的照 MaxText 现状，MaxText 默认每头；"
        "用 Megatron-Core 的 QKV 拆分）或整矩阵",
    )
    ap.add_argument("--qk-clip", action="store_true")
    ap.add_argument("--qk-clip-alpha", type=float, default=0.5)
    ap.add_argument("--qk-clip-threshold", type=float, default=100.0)
    ap.add_argument(
        "--qk-clip-mtp",
        default="on",
        choices=["on", "off"],
        help="MTP 层的注意力也做 QK-Clip（MaxText 会裁 MTP，hy4-tiny 上实测；Megatron-Core 的 clip_qk 只遍历主干）",
    )
    ap.add_argument("--log-max-logit", action="store_true", help="只记最大注意力 logit，不裁剪")
    ap.add_argument(
        "--eod-mask",
        default="on",
        choices=["on", "off"],
        help="eod 位置不算 loss（Megatron eod_mask_loss：输入 token == eod 的位置 loss_mask = 0）；eod = config 的 eos_token_id",
    )
    ap.add_argument(
        "--per-token-loss",
        default="on",
        choices=["on", "off"],
        help="Megatron calculate_per_token_loss：主 loss 按全局批的有效 token 数平均（同 MaxText 手动梯度累积的 xent_sum / 全局 token 数）",
    )
    ap.add_argument(
        "--save-batches",
        action="store_true",
        help="每个 rank 写 data/batches/rank<r>.{bin,jsonl}（janus 格式，含 loss_mask）",
    )
    ap.add_argument(
        "--dump-steps", default="", help="逗号分隔的步（从 1 起）：rank 0 抓前向各模块输出（janus.dump.megatron）"
    )
    ap.add_argument(
        "--full", action="store_true", help="dump 的第一步全量抓数（梯度、参数、优化器状态）；第 1 步参照用，不开重算"
    )
    ap.add_argument(
        "--full-steps",
        default="",
        help="逗号分隔：做全量抓数（梯度、参数、主权重前后、优化器状态）的步；空 = dump 步里最小的一步（janus 默认）",
    )
    ap.add_argument(
        "--gather-experts",
        action="store_true",
        help="全量抓数时把 rank 0 所在专家并行组里其他 rank 的本地专家也收到 rank 0（按全局专家号命名，janus gather_experts）",
    )
    ap.add_argument(
        "--no-dump-acts",
        action="store_true",
        help="抓数步不抓前向激活（janus forward_acts=False），只要参数 / 梯度 / 优化器状态时用",
    )
    ap.add_argument(
        "--full-kinds",
        default="",
        help="全量抓数只写这些种类（逗号或 + 分隔：grad、param、main_param、main_param_after、exp_avg、exp_avg_sq、momentum_buffer）；空 = 全写",
    )
    ap.add_argument(
        "--dump-all-experts",
        action="store_true",
        help="全量抓数也抓全部路由专家（janus 默认每层只留 weight0 / weight1；hy4-mini 小，可以全抓）",
    )
    ap.add_argument(
        "--attn-scores-fp32",
        default="on",
        choices=["on", "off"],
        help="稀疏注意力的 q·k 在 FP32 里算（hy4_attn_scores_fp32）；关时在 BF16 里算再进 FP32 softmax（g1-* 用的是关）",
    )
    ap.add_argument(
        "--qk-dump-steps",
        default="",
        help="逗号或 + 分隔的步：rank 0 存各层（含 MTP）q / kv 上投影在 QK-Clip 前后的 FP32 主权重与 BF16 权重（qk_clip/step_<N>.safetensors）",
    )
    ap.add_argument(
        "--choices-capture-step",
        default="",
        help="这些步（逗号或 + 分隔）每个 rank、每个 microbatch 抓 MoE 路由与 DSA top-k（hy4_capture.Hy4ChoiceCapture，写 choices/）；开重算时只记首次前向",
    )
    ap.add_argument(
        "--l1-capture-step",
        type=int,
        default=0,
        help="这一步 rank 0 第 0 个 microbatch 逐算子抓输入 / 输出与每层输入输出的梯度（hy4_capture.py，写 l1/）；不能开 --recompute",
    )
    ap.add_argument(
        "--l1-grad-micro0",
        action="store_true",
        help="和 --l1-capture-step 一起用：另存第 0 个 microbatch 反向做完时 rank 0 各参数的 main_grad（l1/step_<N>_grad_micro0.safetensors）",
    )
    ap.add_argument(
        "--diag-kd-dump",
        action="store_true",
        help="诊断：第 1 步 rank 0 第 0 个 microbatch 的每个产出 KD 的层（full 层与 MTP 层），存 KD 的原始输入"
        "（教师 q / k、索引器 q / k / w、top-k）与记下的 KD 值；第 0 层的文件名不变",
    )
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--side", default="gpu-perturb-megatron")
    ap.add_argument("--image", default="")
    ap.add_argument("--code", default="")
    a = ap.parse_args()

    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
    rank, world = dist.get_rank(), dist.get_world_size()
    global DP, GA
    assert world % a.pp == 0, (world, a.pp)
    DP = world // a.pp
    assert GBS % DP == 0, f"全局批 {GBS} 不能被 DP {DP} 整除"
    GA = GBS // DP
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    det_env = {}
    # 确定性模式：照 Megatron-Bridge 的 _validate_and_apply_deterministic_mode 与 initialize，在任何 kernel 之前
    if a.deterministic:
        from types import SimpleNamespace as _NS

        from megatron.core import tuning
        from megatron.training.determinism import apply_determinism_to_args

        os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(a.out, ".triton_cache"))
        apply_determinism_to_args(
            _NS(
                cross_entropy_loss_fusion=False,
                tp_comm_overlap=False,
                moe_router_fusion=False,
                moe_router_aux_loss_fusion=None,
            )
        )
        tuning.install(deterministic=True)
        det_env = {
            k: os.environ.get(k) for k in ("NCCL_ALGO", "NVTE_ALLOW_NONDETERMINISTIC_ALGO", "CUBLAS_WORKSPACE_CONFIG")
        }
    probe = tf32_probe(dev)

    import hy4_mcore as H
    from hy4_weights import load_hf_into_mcore
    from megatron.core import parallel_state as ps
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig, finalize_model_grads
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.module import Float16Module

    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=a.pp, expert_model_parallel_size=a.ep
    )
    assert ps.get_data_parallel_world_size() == DP, (ps.get_data_parallel_world_size(), DP)
    is_first, is_last = ps.is_pipeline_first_stage(), ps.is_pipeline_last_stage()
    pp_grp = ps.get_pipeline_model_parallel_group() if a.pp > 1 else None
    if a.pp > 1:
        assert not a.l1_capture_step and not a.diag_kd_dump, "--l1-capture-step / --diag-kd-dump 只支持 PP = 1"
    model_parallel_cuda_manual_seed(1234)
    hf, hf_filled = H.complete_hf_config(json.load(open(os.path.join(a.hf, "config.json"))))
    kw = H.hy4_config_kwargs(hf, num_layers=a.layers)
    mtp_factor = a.mtp_factor if a.mtp_factor is not None else float(hf.get("mtp_loss_factor", 0.1))
    n_idx = sum(t == "full" for t in hf["indexer_types"][: a.layers]) + int(a.mtp)
    extra = dict(
        hy4_train_indexer=a.indexer_loss > 0,
        dsa_indexer_loss_coeff=H.indexer_loss_coeff_per_layer(
            a.indexer_loss, hf["indexer_types"][: a.layers], int(a.mtp)
        ),
        dsa_indexer_use_sparse_loss=True,  # hy4-preview：indexer_sparse_training
        calculate_per_token_loss=a.per_token_loss == "on",
        hy4_muon_exclude_ihc=False,  # 由下面 --muon-exclude-ihc 按名字标 use_muon=False
        hy4_muon_gate=a.muon_gate,
        hy4_attn_scores_fp32=a.attn_scores_fp32 == "on",
        params_dtype=torch.bfloat16,
        bf16=True,
        transformer_impl="transformer_engine",
        moe_grouped_gemm=True,
        expert_model_parallel_size=a.ep,
        perform_initialization=False,
        hy4_head_chunk=a.head_chunk,
        moe_router_bias_update_rate=a.bias_rate,
        cross_entropy_loss_fusion=False,
        gradient_accumulation_fusion=False,
    )
    if a.mtp:
        extra.update(mtp_num_layers=1, mtp_loss_scaling_factor=mtp_factor)
    if a.pp > 1:
        # 层间是 n 路流 [s, b, n·H]，不是 Megatron 按 hidden_size 算的 P2P 形状：让 P2P 先交换形状（hy4_mcore 开头）
        extra.update(pipeline_model_parallel_size=a.pp, pipeline_dtype=torch.bfloat16, variable_seq_lengths=True)
        if a.pp_first:
            extra["num_layers_in_first_pipeline_stage"] = a.pp_first
        if a.pp_last:
            extra["num_layers_in_last_pipeline_stage"] = a.pp_last
    if a.deterministic:
        extra["deterministic_mode"] = True
    if a.recompute:
        extra.update(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)
    if a.qk_clip or a.log_max_logit:
        extra.update(
            qk_clip=a.qk_clip,
            qk_clip_alpha=a.qk_clip_alpha,
            qk_clip_threshold=a.qk_clip_threshold,
            log_max_attention_logit=True,
        )
    dist_opt = a.dist_opt == "on" or (a.dist_opt == "auto" and a.optimizer == "adam")
    cfg = H.Hy4Config(**{**kw, **extra})  # extra 覆盖 hy4_config_kwargs 的默认（如 dsa_indexer_loss_coeff）
    pp_stages = H.check_pp_split(cfg)  # 每段的全局层号范围；切分点不在 full 层上就报错
    rg = H.stage_layers(cfg)  # 本段的全局层号
    model = H.build_model(cfg, use_te=True, vocab_size=hf["vocab_size"], max_seq=a.seq, mtp=a.mtp).to(dev)
    yarn_info = None
    if a.yarn_factor != 1.0:
        yarn_info = apply_yarn(
            model, float(cfg.rotary_base), a.yarn_factor, a.yarn_orig_max, a.yarn_beta_fast, a.yarn_beta_slow
        )
        if rank == 0:
            print("YARN", json.dumps(yarn_info), flush=True)
    if a.bias_zero_mean == "on":
        install_bias_zero_mean()
    model = Float16Module(
        cfg, model
    )  # 先转精度再载入：BF16 参数拿到 checkpoint 的原值，FP32 参数（iHC、偏置等）不经 BF16
    t0 = time.time()
    load_hf_into_mcore(model.module, cfg, a.hf, ep_rank=ps.get_expert_model_parallel_rank(), ep_size=a.ep)
    load_s = time.time() - t0
    # 开训前核：同一张映射逐个核「参数 == checkpoint 张量转成参数 dtype」（本 rank 的专家分片也核），
    # 两边 FP64 平方和对齐（非专家参数每个 rank 一份，按 EP 组只数一次）。不过就退出（同 GLM-5.2 一侧）
    chk = load_hf_into_mcore(
        model.module, cfg, a.hf, ep_rank=ps.get_expert_model_parallel_rank(), ep_size=a.ep, verify=True
    )
    bad = torch.tensor([len(chk["mismatch"])], device=dev)
    dist.all_reduce(bad)
    load_check = {
        "n_checked_rank0": chk["n"],
        "n_mismatch_all_ranks": int(bad[0]),
        "mismatch_rank0": chk["mismatch"][:20],
    }
    load_check["sumsq_rank0"] = {"model": chk["sumsq_model"], "ckpt": chk["sumsq_ckpt"]}
    load_check["pass"] = bool(int(bad[0]) == 0 and chk["sumsq_model"] == chk["sumsq_ckpt"])
    if rank == 0:
        print("LOAD_CHECK", json.dumps(load_check), flush=True)
    if not load_check["pass"]:
        raise SystemExit("LOAD_CHECK_FAIL")
    ddp_cfg = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        use_distributed_optimizer=dist_opt,
        check_for_nan_in_grad=True,
    )
    model = DistributedDataParallel(config=cfg, ddp_config=ddp_cfg, module=model)
    import ast

    okv = {}
    for kv in filter(None, a.opt_kv.replace(";", ",").split(",")):
        k, v = kv.split("=", 1)
        try:
            okv[k] = ast.literal_eval({"true": "True", "false": "False"}.get(v, v))
        except (ValueError, SyntaxError):
            okv[k] = v
    if a.optimizer == "muon":  # 项目负责人定：NS 系数 simple、Nesterov 开（Megatron-Core 默认 quintic、不开）
        okv.setdefault("muon_coefficient_type", "simple")
        okv.setdefault("muon_nesterov", True)
    ocfg = OptimizerConfig(
        optimizer=a.optimizer,
        lr=a.lr,
        min_lr=a.lr,
        weight_decay=a.wd,
        adam_beta1=0.9,
        adam_beta2=a.adam_beta2,
        adam_eps=a.adam_eps,
        clip_grad=a.clip,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=dist_opt,
        **okv,
    )
    if a.optimizer == "muon" and a.muon_exclude_ihc != "none":  # use_muon=False → Megatron 交给标量优化器（Adam）
        pats = (".attn_hc.", ".ffn_hc.") + ((".final_layernorm.hc_",) if a.muon_exclude_ihc == "ihc+head" else ())
        for n, p in model.named_parameters():
            if any(t in n for t in pats):
                p.use_muon = False
    opt = get_megatron_optimizer(ocfg, [model], config_overrides={})  # {}：不按一维参数 / 偏置分组免衰减
    if rank == 0 and a.optimizer == "muon":  # 每个参数实际走哪个优化器（Megatron-Core 的判法）
        from megatron.core.optimizer.emerging_optimizers import _is_muon_excluded

        part = {
            n.replace("module.", ""): {"shape": list(p.shape), "opt": "adam" if _is_muon_excluded(p) else "muon"}
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        os.makedirs(os.path.join(a.out, a.name), exist_ok=True)
        with open(os.path.join(a.out, a.name, "optimizer_partition.json"), "w") as f:
            json.dump(part, f, indent=0)
        print(
            "MUON_PARTITION",
            json.dumps(
                {
                    "muon": sum(v["opt"] == "muon" for v in part.values()),
                    "adam": sum(v["opt"] == "adam" for v in part.values()),
                }
            ),
            flush=True,
        )
    for g in opt.param_groups:
        g["lr"] = a.lr
        assert g["weight_decay"] == a.wd, g["weight_decay"]
    cfg.finalize_model_grads_func = finalize_model_grads
    cfg.no_sync_func = model.no_sync
    fb = get_forward_backward_func()

    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokens = np.load(a.tokens, mmap_mode="r")
    need = a.steps * GBS
    assert tokens.shape[0] >= need and tokens.shape[1] >= a.seq + 1, (tokens.shape, need)
    dp_rank = ps.get_data_parallel_rank()
    rundir = os.path.join(a.out, a.name)
    os.makedirs(rundir, exist_ok=True)
    import megatron.core

    if True:  # meta 各 rank 都建（JanusDump 只在 rank 0 写）
        meta = {
            "schema_version": 1,
            "run_name": a.name,
            "side": a.side,
            "framework": "megatron-core",
            "model": {"name": "hy4-preview", "layers": a.layers, "mtp": a.mtp},
            "code": {
                "janus": a.code,
                "hy4_mcore_sha256": hashlib.sha256(open(H.__file__, "rb").read()).hexdigest(),
                "train_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
            },
            "image": a.image,
            "versions": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "megatron_core": megatron.core.__version__,
                "nccl": ".".join(map(str, torch.cuda.nccl.version())),
            },
            "hardware": {"gpu": torch.cuda.get_device_name(0), "count": world},
            "parallelism": {
                "dp": DP,
                "ep": a.ep,
                "tp": 1,
                "pp": a.pp,
                "pp_stage_layers": pp_stages,
                "mbs": 1,
                "ga": GA,
                "gbs": GBS,
                "nodes": world // int(os.environ.get("LOCAL_WORLD_SIZE", world)),
            },
            "config": {
                "lr": a.lr,
                "lr_schedule": a.lr_schedule
                if a.lr_schedule == "constant"
                else {
                    "type": "cosine (MaxText create_learning_rate_schedule)",
                    "peak": a.lr,
                    "warmup_steps": a.warmup_steps,
                    "final_fraction": a.lr_final_fraction,
                    "length": a.steps,
                },
                "rope_yarn": yarn_info,
                "routing_bias_zero_mean": a.bias_zero_mean == "on",
                "betas": [0.9, a.adam_beta2],
                "eps": a.adam_eps,  # 原来写死 1e-8，与实际的 adam_eps 不符（台账 G24）
                "weight_decay_effective": a.wd,
                "weight_decay_groups": "all parameters",
                "optimizer": a.optimizer,
                "adam_eps": a.adam_eps,
                "adam_beta2": a.adam_beta2,
                "optimizer_extra": {k: str(v) for k, v in okv.items()},
                "distributed_optimizer": dist_opt,
                "muon_exclude_ihc": a.muon_exclude_ihc if a.optimizer == "muon" else None,
                "qk_clip": {"on": a.qk_clip, "alpha": a.qk_clip_alpha, "threshold": a.qk_clip_threshold},
                "clip": a.clip,
                "seq": a.seq,
                "head_dgrad_chunk": a.head_chunk,
                "recompute": a.recompute,
                "rope": "interleaved (MLA: mla_rotary_interleaved; indexer: dsa_indexer_rope_interleaved)",
                "mla_latent_norm_eps": cfg.attention_latent_norm_epsilon,
                "indexer_k_norm_eps": cfg.dsa_indexer_k_norm_epsilon,
                "mtp_loss_factor": mtp_factor if a.mtp else None,
                "routing_bias_update_rate": a.bias_rate,
                "indexer_trained": a.indexer_loss > 0,
                "indexer_loss": {
                    "maxtext_coeff": a.indexer_loss,
                    "producers": n_idx,
                    "coeff_per_layer": cfg.dsa_indexer_loss_coeff,
                    "sparse": cfg.dsa_indexer_use_sparse_loss,
                    "impl": "megatron.core compute_dsa_indexer_loss + DSAIndexerLossAutoScaler",
                },
                "qk_clip_mtp": a.qk_clip_mtp if a.mtp else None,
                "loss_mask": {
                    "eod_mask": a.eod_mask,
                    "eod_token_id": int(hf["eos_token_id"]),
                    "rule": "input token == eod -> 0",
                },
                "per_token_loss": a.per_token_loss,
                "attn_scores_fp32": bool(cfg.hy4_attn_scores_fp32),
                "muon_gate": a.muon_gate if a.optimizer == "muon" else None,
                "hf_config_filled_from_release": {k: hf[k] for k in hf_filled},
                "load_check": load_check,
                "attention": "dense masked softmax with sink (sparse_sink_attention); q·k dtype per attn_scores_fp32",
            },
            "params": {
                "trainable": n_train,
                "frozen_indexer": n_frozen,
                "note": "rank 0 本地（专家只算本 EP rank 的）",
            },
            "checkpoint": {
                "hf_dir": a.hf,
                "tensor_sha256_manifest": hashlib.sha256(
                    open(os.path.join(a.hf, "tensor_sha256.json"), "rb").read()
                ).hexdigest(),
            },
            "data": {
                "tokens": a.tokens,
                "tokens_sha256_first_rows": sha(np.asarray(tokens[:need])),
                "layout": f"sample=(step-1)*16+micro*{DP}+rank",
            },
            "steps": {"first": 1, "last": a.steps, "convention": "golden_step = step"},
            "dtypes": {
                "master": "float32",
                "compute": "bfloat16",
                "fp32_modules": "lm_head, iHC, hc_head, sinks, indexer k_norm",
            },
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "tf32": probe,
            "load_seconds": load_s,
            "determinism": {
                "deterministic_mode": bool(a.deterministic),
                "env": det_env,
                "note": "默认设置下 GPU 同设置重跑不逐位相同（hy4-mini EP4 实测），噪声取重跑各对",
            },
            "resume": {"from": a.resume or None, "save_at": a.save_at or None},
            "metrics_fields": {
                "lm_loss": "主 loss，全局批有效 token 平均（per-token），按 loss_mask",
                "param_norm": "FP32 主权重全局 L2（Megatron-Bridge calc_params_l2_norm，经 janus.dump.megatron）",
                "param_norm_fp64": "同一量的本程序 FP64 重算（专家按 EP 复制份数去重），核对用",
                "grad_norm": "裁剪前全局梯度范数（Megatron 优化器返回值）",
                "mtp_loss": "MTP loss，各 microbatch 内按有效 token 平均、再对 microbatch 与 DP 取平均（未乘系数）",
                "indexer_loss": "索引器 KD，MaxText 口径：各产出层逐行平均 KL × (系数 ÷ 产出层数) 求和，对 microbatch 与 DP 取平均",
                "max_attention_logit": "本步全局最大缩放 q·k（含 MTP 层），QK-Clip 用",
                "indexer_loss.L": "产出层 L（主干层号或 mtp0）的 KD：逐行平均 KL × 逐层系数；各 L 之和 = indexer_loss",
                "grad_norm_fp64": "本程序 FP64 重算的裁剪前梯度范数（main_grad，参数，不含 buffer），核对 grad_norm 用",
                "grad_sq.G": "裁剪前梯度平方和按组：indexer / mtp / experts / hc / embed_head / rest",
                "qk.max_logit.L": "层 L（主干层号或 mtp0）每头的最大缩放 q·k（DP 上取 MAX，QK-Clip 之前），列表长 = 头数",
                "qk.eta.L": "每头的缩放 η = min(1, τ ÷ 最大 logit)；q_nope × η^α、q_rope × η、k_nope × η^(1−α)（Megatron-Core MLA.clip_qk）",
                "qk.clipped.L": "层 L 被裁的头数（最大 logit > τ）；qk.clipped_total = 各层之和",
                "mtp_loss_mt": "MaxText 口径（M22 开）：每个 microbatch 0.1 × Σ_rank(MTP 均值_r × 主 loss 有效 token_r) ÷ Σ_rank 有效 token_r，再对 microbatch 平均",
                "router.bias_mean.L": "L = 主干层号（0 起，与 HF 层号相同）；mtp0 = MTP 层",
            },
        }
        if rank == 0:
            print("META", json.dumps(meta["params"]), flush=True)
    from janus.dump.megatron import JanusDump

    # 步列表用 , 或 + 分隔（pod_run.sh 把附加参数里的逗号换成空格，Pod 里用 +）
    dump_steps = {int(x) for x in a.dump_steps.replace("+", ",").split(",") if x}
    dump = JanusDump(
        rundir,
        meta,
        dump_steps=dump_steps,
        full=a.full,
        recompute=a.recompute,
        save_batches=a.save_batches,
        **({"skip": None} if a.dump_all_experts else {}),
        gather_experts=a.gather_experts,
        forward_acts=not a.no_dump_acts,
        full_kinds=a.full_kinds.replace("+", ",") or None,
        **({"full_steps": {int(x) for x in a.full_steps.replace("+", ",").split(",") if x}} if a.full_steps else {}),
    )

    eod = int(hf["eos_token_id"])
    pos = torch.arange(a.seq, device=dev).unsqueeze(0)

    def batches(step):
        out = []
        for m in range(GA):
            row = np.asarray(tokens[(step - 1) * GBS + m * DP + dp_rank, : a.seq + 1]).astype(np.int64)
            tok, lab = row[:-1][None], row[1:][None]
            lm = (tok != eod).astype(np.float32) if a.eod_mask == "on" else np.ones_like(tok, dtype=np.float32)
            b = {
                "tokens": torch.from_numpy(tok).to(dev),
                "labels": torch.from_numpy(lab).to(dev),
                "loss_mask": torch.from_numpy(lm).to(dev),
                "position_ids": pos,
            }
            dump._record_batch(b)  # data/rank<r>.jsonl 指纹与 nonpad；--save-batches 时整批落盘（janus 格式）
            out.append((b["tokens"], b["labels"], b["loss_mask"]))
        return out

    micro = {"n": 0}

    def fstep(it, model):
        ids, labels, lm = next(it)
        mk = micro["n"]  # 本步第几个 microbatch：shared 层的 top-k 按它存取（hy4_mcore.set_topk_micro）
        micro["n"] += 1
        H.set_topk_micro(mk)
        if chc is not None and chc_state["on"]:
            chc.begin_micro(chc_state["m"], ids)
            chc_state["m"] += 1
        # 逐 token loss [1, S]；loss_mask 也交给模型，MTP 按它（滚动后）算（MTP loss 由 MTPLossAutoScaler 加进反向）
        out = model(ids, pos, attention_mask=None, labels=labels, loss_mask=lm)
        if out.requires_grad:  # 这个 microbatch 的反向到达本段输出时切回它的号（之后整层重算的 shared 层取对 top-k）
            out.register_hook(lambda g, mk=mk: H.set_topk_micro(mk))

        if cap is not None and cap.active:
            cap._put("input.labels", labels, "batch", "input", "b,s")
            cap._put("input.loss_mask", lm, "batch", "input", "b,s")

        def loss_func(o):
            if cap is not None and cap.active:
                cap._put("loss.per_token", o, "model", "output", "b,s")
                cap.end_forward()
            if diag_kd["want"] and rank == 0:
                for lname, c0 in kd_dump_cores:
                    x, qr, query, key = c0.last_kd_inputs
                    kd_logged = None if c0.last_indexer_loss is None else float(c0.last_indexer_loss)
                    with torch.no_grad():
                        iq, ik, iw = c0.indexer.forward_before_topk(x, qr, None)
                    torch.save(
                        {
                            "query": query.cpu(),
                            "key": key.cpu(),
                            "indexer_q": iq.cpu(),
                            "indexer_k": ik.cpu(),
                            "indexer_w": iw.cpu(),
                            "topk": c0.last_topk.cpu(),
                            "softmax_scale": c0.softmax_scale,
                            "coeff_per_layer": cfg.dsa_indexer_loss_coeff,
                            "per_token_loss": a.per_token_loss,
                            # 整层重算时前向不建图，KD 在反向的重算里才算：这里还是 None（KD 值见 metrics）
                            "logged_kd_sum_times_coeff": kd_logged,
                            "logged_kd_row_mean_times_coeff": None
                            if kd_logged is None
                            else kd_logged / (a.seq if a.per_token_loss == "on" else 1),
                            "loss_mask_sum": float(lm.sum()),
                            "step": 1,
                            "rank": 0,
                            "micro": 0,
                            "layer": int(lname) if lname.isdigit() else lname,
                        },
                        os.path.join(rundir, f"kd_layer{lname}_step1_rank0_micro0.pt"),
                    )
                diag_kd["want"] = False
            if (
                a.mtp
            ):  # mtp_loss_mt 用：本 microbatch 本 rank 的 MTP 均值（tracker 是累加的，取增量）× 主 loss 有效 token 数
                from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

                cur = MTPLossLoggingHelper.tracker["loss_values"][0].detach().clone()
                mt_rows.append(torch.stack([(cur - mt_prev[0]) * lm.sum(), lm.sum().float()]))
                mt_prev[0] = cur

            # 三元组（loss 和、有效 token 数、记录）：per-token 时调度器不除，finalize_model_grads 按全局 token 数除；
            # 否则调度器除以本 microbatch 的 token 数与 microbatch 数（Megatron 原样）
            s = (o.float() * lm).sum()
            n = lm.sum().to(torch.int)
            return s, n, {"loss": torch.stack([s.detach().clone(), lm.sum().detach().float()])}

        return out, loss_func

    def dup_embed(n):  # PP + MTP：末段那份嵌入是首段的副本（梯度跨段同步），范数只算首段的
        return a.pp > 1 and not is_first and "embedding.word_embeddings" in n

    def pnorm_sq():
        psq = torch.zeros(1, dtype=torch.float64, device=dev)
        if dist_opt:  # 分布式优化器：每个元素只在一个 rank 的 FP32 分片里（DP 切分 / EP 各管各的专家）
            for g in opt.param_groups:
                for p in g["params"]:
                    psq += p.detach().double().pow(2).sum()
        else:  # 不切分：非专家参数每个 rank 一份（除以 DP 份数），专家参数各 EP rank 不同
            for n, p in model.named_parameters():
                if dup_embed(n):
                    continue
                w = getattr(p, "main_param", None)
                w = p if w is None else w
                v = w.detach().double().pow(2).sum()
                psq += v / (DP // a.ep) if not getattr(p, "allreduce", True) else v / DP  # 专家在 DP/EP 组间复制
        dist.all_reduce(psq)
        return psq

    from types import SimpleNamespace

    ctx = SimpleNamespace(
        model=[model],
        optimizer=opt,
        state=SimpleNamespace(train_state=SimpleNamespace(step=0)),
        loss_dict=None,
        grad_norm=None,
        skipped_iter=False,
    )
    dump.on_train_start(ctx)
    diag_kd = {"want": a.diag_kd_dump}
    cap = None
    if a.l1_capture_step:
        assert not a.recompute, "--l1-capture-step 不能和 --recompute 一起用"
        if rank == 0:
            from hy4_capture import Hy4Capture

            cap = Hy4Capture(model.module.module, cfg, rundir, a.l1_capture_step, grad_micro0=a.l1_grad_micro0)
    chc, chc_state = None, {"on": False, "m": 0}
    chc_steps = {int(x) for x in str(a.choices_capture_step).replace("+", ",").split(",") if x}
    if chc_steps:
        from hy4_capture import Hy4ChoiceCapture

        chc = Hy4ChoiceCapture(model.module.module, rundir, min(chc_steps), DP, dp_rank, GBS)
    qk_dump_steps = {int(x) for x in a.qk_dump_steps.replace("+", ",").split(",") if x}
    kd_dump_cores = []  # (名字, core_attention)：第 1 步 rank 0 第 0 个 microbatch 各产出 KD 的层
    if a.diag_kd_dump and rank == 0:
        _mm = model.module.module
        kd_dump_cores = [
            (str(i), layer.self_attention.core_attention)
            for i, layer in enumerate(_mm.decoder.layers)
            if layer.self_attention.core_attention.is_full
        ]
        if a.mtp and getattr(_mm, "mtp_process", False):
            kd_dump_cores.append(("mtp0", _mm.mtp.layers[0].mtp_model_layer.self_attention.core_attention))
        for _, c in kd_dump_cores:
            c.capture = True

    def grad_group(n):
        if ".indexer." in n:
            return "indexer"
        if n.startswith("mtp."):
            return "mtp"
        if "experts.linear_fc" in n and "shared_experts" not in n:
            return "experts"
        if "_hc." in n or "final_layernorm.hc_" in n:
            return "hc"
        if n.startswith(("embedding.", "output_layer.")):
            return "embed_head"
        return "rest"

    gnames = [(n.replace("module.", ""), p) for n, p in model.named_parameters() if p.requires_grad]

    def grad_sq_by_group():
        """main_grad（DP 归约后、裁剪前）的 FP64 平方和，按组；非专家参数每个 rank 一份（÷ DP），专家在 DP/EP 组间复制（÷ DP/EP）。"""
        keys = ["indexer", "mtp", "experts", "hc", "embed_head", "rest"]
        acc = torch.zeros(len(keys), dtype=torch.float64, device=dev)
        for n, p in gnames:
            if dup_embed(n):
                continue
            g = getattr(p, "main_grad", None)
            g = p.grad if g is None else g
            if g is None:
                continue
            v = g.detach().double().square().sum()
            acc[keys.index(grad_group(n))] += v / (DP // a.ep) if not getattr(p, "allreduce", True) else v / DP
        dist.all_reduce(acc)
        out = {f"grad_sq.{k}": float(acc[i]) for i, k in enumerate(keys)}
        out["grad_norm_fp64"] = float(acc.sum().sqrt())
        return out

    # ---- 存档 / 续跑：每个 rank 一个文件（模型含路由偏置 buffer、优化器含 Muon 动量与 FP32 主权重、RNG、数据游标）
    from megatron.core import tensor_parallel

    mm = model.module.module  # GPTModel（本段）
    has_mtp = bool(a.mtp and getattr(mm, "mtp_process", False))
    save_at = {int(x) for x in a.save_at.replace("+", ",").split(",") if x}
    save_root = a.save_dir or os.path.join(rundir, "ckpt")
    par_sig = {"world": world, "dp": DP, "ep": a.ep, "pp": a.pp, "pp_stage_layers": pp_stages, "gbs": GBS}

    def save_ckpt(step):
        d = os.path.join(save_root, f"step_{step}")
        os.makedirs(d, exist_ok=True)
        t_save = time.time()
        torch.save(
            {
                "step": step,
                "next_step": step + 1,  # 数据游标：第 s 步取第 (s-1)·16 … 条，只由步号决定
                "parallelism": par_sig,
                # 已经吃掉的前 step·16 条的 sha256：续跑时核同一份数据
                "data": {
                    "consumed_rows": step * GBS,
                    "consumed_sha256": sha(np.asarray(tokens[: step * GBS])),
                    "layout": meta["data"]["layout"],
                },
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "rng": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(),
                    "numpy": np.random.get_state(),
                    "tracker": tensor_parallel.get_cuda_rng_tracker().get_states(),
                },
            },
            os.path.join(d, f"rank{rank:05d}.pt"),
        )
        dist.barrier()
        size = torch.tensor([os.path.getsize(os.path.join(d, f"rank{rank:05d}.pt"))], dtype=torch.float64, device=dev)
        dist.all_reduce(size)
        if rank == 0:
            info = {"step": step, "dir": d, "seconds": round(time.time() - t_save, 1), "bytes_all_ranks": int(size[0])}
            print("CKPT_SAVED", json.dumps(info), flush=True)

    start = 1
    if a.resume:
        t_load = time.time()
        ck = torch.load(os.path.join(a.resume, f"rank{rank:05d}.pt"), map_location=dev, weights_only=False)
        assert ck["parallelism"] == par_sig, ("续跑的并行设置与存档不同", ck["parallelism"], par_sig)
        assert ck["data"]["consumed_sha256"] == sha(np.asarray(tokens[: ck["data"]["consumed_rows"]])), (
            "续跑的数据与存档时不同"
        )
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["rng"]["torch"].cpu())
        torch.cuda.set_rng_state(ck["rng"]["cuda"].cpu())
        np.random.set_state(ck["rng"]["numpy"])
        tensor_parallel.get_cuda_rng_tracker().set_states(ck["rng"]["tracker"])
        start = int(ck["next_step"])
        del ck
        if rank == 0:
            print(
                "CKPT_LOADED",
                json.dumps({"dir": a.resume, "start_step": start, "seconds": round(time.time() - t_load, 1)}),
                flush=True,
            )

    def state_sha(step):
        """模型 state_dict 与优化器 state_dict 里每个张量的 sha256（按嵌套路径命名），本 rank 一个文件。"""
        out = {}

        def walk(prefix, o):
            if isinstance(o, torch.Tensor):
                t = o.detach().contiguous().cpu()
                out[prefix] = {
                    "dtype": str(t.dtype).replace("torch.", ""),
                    "shape": list(t.shape),
                    "sha256": hashlib.sha256(
                        t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b""
                    ).hexdigest(),
                }
            elif isinstance(o, dict):
                for k, v in o.items():
                    walk(f"{prefix}.{k}" if prefix else str(k), v)
            elif isinstance(o, (list, tuple)):
                for i, v in enumerate(o):
                    walk(f"{prefix}.{i}", v)
            elif isinstance(o, (int, float, str, bool)) or o is None:
                out[prefix] = {"value": o}

        walk("model", model.state_dict())
        walk("optimizer", opt.state_dict())
        os.makedirs(os.path.join(rundir, "state_sha"), exist_ok=True)
        with open(os.path.join(rundir, "state_sha", f"rank{rank}.json"), "w") as f:
            json.dump({"step": step, "tensors": out}, f, indent=0, sort_keys=True)

    def pp_merge(d):
        """各流水线段的本段指标（已在段内 DP 上归约）合成全局的；PP = 1 时原样。"""
        if a.pp == 1:
            return dict(d)
        objs = [None] * a.pp
        dist.all_gather_object(objs, d, group=pp_grp)
        out = {}
        for o in objs:
            for k, v in o.items():
                if k == "max_attention_logit":
                    out[k] = max(out.get(k, v), v)
                elif k.startswith("_sum."):
                    out[k] = out.get(k, 0.0) + v
                else:
                    out[k] = v
        return out

    p0 = float(pnorm_sq().sqrt())  # 第 0 步（载入后、第 1 步前）的 param_norm，和 LOAD_CHECK 一起记
    if rank == 0:
        print("PARAM_NORM_STEP0", p0, flush=True)
        with open(os.path.join(rundir, "load_check.json"), "w") as f:
            json.dump({**load_check, "param_norm_step0": p0, "resumed_from": a.resume or None}, f, indent=1)
    for step in range(start, a.steps + 1):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.time()
        ctx.state.train_state.step = step - 1
        qk_vals = {}
        mt_rows, mt_prev = [], [torch.zeros((), device=dev)]
        micro["n"] = 0
        H.clear_topk_store(cfg)
        dump.on_train_step_start(ctx)
        if cap is not None and step == a.l1_capture_step:
            cap.begin()
        if chc is not None and step in chc_steps:
            chc.begin_step(step)
            chc_state.update(on=True, m=0)
        model.zero_grad_buffer()
        opt.zero_grad()
        stats = fb(
            forward_step_func=fstep,
            data_iterator=iter(batches(step)),
            model=[model],
            num_microbatches=GA,
            seq_length=a.seq,
            micro_batch_size=1,
            forward_only=False,
        )
        if chc is not None and step in chc_steps:
            chc.end_step(
                {"run": a.name, "attn_scores_fp32": bool(cfg.hy4_attn_scores_fp32), "recompute": bool(a.recompute)}
            )
            chc_state["on"] = False
            if step == max(chc_steps):
                chc.close()
                chc = None
        if cap is not None and step == a.l1_capture_step:
            cap.finish({"run": a.name, "attn_scores_fp32": bool(cfg.hy4_attn_scores_fp32)})
            cap = None
        if a.lr_schedule == "cosine":  # 第 step 步是第 step − 1 次更新（MaxText 的 count 从 0 起，第 1 步 lr = 0）
            lr_now = maxtext_lr(step - 1, a.lr, a.warmup_steps, a.lr_final_fraction, a.steps)
            for g in opt.param_groups:
                g["lr"] = lr_now
        gsq = grad_sq_by_group()  # 优化器步之前（裁剪前），FP64，独立于 Megatron 的 grad_norm 核对用
        ok, gnorm, _ = opt.step()
        max_logit = None
        if (
            a.qk_clip or a.log_max_logit
        ):  # MuonClip：优化器步之后按本步最大注意力 logit 缩 q / k 的上投影（Megatron 原样）
            from megatron.core.optimizer.qk_clip import clip_qk

            # 逐层逐头的最大 logit（DP 上取 MAX，与 clip_qk 里的归约相同、取 MAX 两次不变）、缩放 η 与被裁的头数，
            # 以及指定步的 q / kv 上投影裁剪前后；都在 clip_qk 之前取（它会把统计量清零）。层号是全局层号
            qk_attns = [(str(rg.start + i), l.self_attention) for i, l in enumerate(mm.decoder.layers)]
            if has_mtp:
                qk_attns.append(("mtp0", mm.mtp.layers[0].mtp_model_layer.self_attention))
            dp_grp = ps.get_data_parallel_group(with_context_parallel=True)
            for nm, attn in qk_attns:
                cur = attn.core_attention.current_max_attn_logits
                if cur is None:
                    continue
                dist.all_reduce(cur, op=dist.ReduceOp.MAX, group=dp_grp)
                eta = torch.clamp(a.qk_clip_threshold / cur.float(), max=1.0)
                qk_vals[f"qk.max_logit.{nm}"] = [float(x) for x in cur]
                qk_vals[f"qk.eta.{nm}"] = [float(x) for x in eta]
                qk_vals[f"qk.clipped.{nm}"] = int((cur > a.qk_clip_threshold).sum()) if a.qk_clip else 0
            qk_snap = {}
            if a.qk_clip and step in qk_dump_steps and rank == 0:
                for nm, attn in qk_attns:
                    for wn in ("linear_q_up_proj", "linear_kv_up_proj"):
                        w = getattr(attn, wn).weight
                        mp = getattr(w, "main_param", None)
                        qk_snap[f"{nm}.{wn}.main_param.before"] = (
                            (mp if mp is not None else w).detach().float().cpu().clone()
                        )
                        qk_snap[f"{nm}.{wn}.param.before"] = w.detach().cpu().clone()
            max_logit = clip_qk([model], log_max_only=not a.qk_clip)
            if has_mtp and a.qk_clip_mtp == "on":  # clip_qk 只遍历主干；MTP 层照同一流程（MLA 自带的 clip_qk）
                mtp_attn = mm.mtp.layers[0].mtp_model_layer.self_attention
                cur = mtp_attn.core_attention.current_max_attn_logits
                if cur is not None:
                    dist.all_reduce(
                        cur, op=dist.ReduceOp.MAX, group=ps.get_data_parallel_group(with_context_parallel=True)
                    )
                    max_logit = max(max_logit, float(cur.max()))
                    if a.qk_clip:
                        mtp_attn.clip_qk()
                    mtp_attn.core_attention.current_max_attn_logits = None
            if qk_snap:
                from safetensors.torch import save_file

                for nm, attn in qk_attns:
                    for wn in ("linear_q_up_proj", "linear_kv_up_proj"):
                        w = getattr(attn, wn).weight
                        mp = getattr(w, "main_param", None)
                        qk_snap[f"{nm}.{wn}.main_param.after"] = (
                            (mp if mp is not None else w).detach().float().cpu().clone()
                        )
                        qk_snap[f"{nm}.{wn}.param.after"] = w.detach().cpu().clone()
                os.makedirs(os.path.join(rundir, "qk_clip"), exist_ok=True)
                save_file(
                    {k: v.contiguous() for k, v in qk_snap.items()},
                    os.path.join(rundir, "qk_clip", f"step_{step}.safetensors"),
                )
        torch.cuda.synchronize()
        dt = time.time() - t0
        sv = dict(qk_vals)  # 本段的指标（段内 DP 上已归约），最后跨段合并
        if max_logit is not None:
            sv["max_attention_logit"] = float(max_logit)
        if is_last:  # 主 loss 只在末段：[loss 和, token 数]，全局批的 token 平均
            lv = torch.stack([s["loss"] for s in stats]).sum(0)
            dist.all_reduce(lv, group=ps.get_data_parallel_group())
            sv["lm_loss"] = float(lv[0] / lv[1].clamp_min(1))
        psq = pnorm_sq()
        mem = torch.tensor([torch.cuda.max_memory_allocated() / 2**30], device=dev)
        dist.all_reduce(mem, op=dist.ReduceOp.MAX)
        if has_mtp:  # MTPLossLoggingHelper 每个 microbatch 累加一次本 rank 的 MTP loss（已按 token 平均）
            from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

            tr = MTPLossLoggingHelper.tracker
            mv = tr["loss_values"][0].detach().clone().reshape(1) / GA
            dist.all_reduce(mv, group=ps.get_data_parallel_group())
            sv["mtp_loss"] = float(mv[0]) / DP
            MTPLossLoggingHelper.clean_metrics_in_tracker()
            # MaxText 口径（M22 开时）：每个 microbatch 内 0.1 × Σ_rank(均值_r × orig_r) ÷ Σ_rank orig_r，再对 microbatch 取平均
            mr = torch.stack(mt_rows)  # [GA, 2]
            dist.all_reduce(mr, group=ps.get_data_parallel_group())
            sv["mtp_loss_mt"] = float(mtp_factor * (mr[:, 0] / mr[:, 1].clamp_min(1)).mean())
            eb = getattr(mm.mtp.layers[0].mtp_model_layer.mlp.router, "expert_bias", None)
            if eb is not None:  # MTP 层的路由偏置（JanusDump 只记主干各层）
                sv["router.bias_mean.mtp0"] = float(eb.double().mean())
                sv["router.bias_std.mtp0"] = float(eb.double().std())
        if a.pp > 1 and not is_first:  # JanusDump 只在 rank 0（首段）记主干路由偏置；其余段的在这里补
            for i, l in enumerate(mm.decoder.layers):
                eb = getattr(getattr(getattr(l, "mlp", None), "router", None), "expert_bias", None)
                if eb is not None:
                    bf = eb.detach().float()
                    sv[f"router.bias_mean.{rg.start + i}"] = float(bf.mean())
                    sv[f"router.bias_std.{rg.start + i}"] = float(bf.std())
        if (
            a.indexer_loss > 0
        ):  # MaxText 口径：各产出层 KD 的平均 × 系数 = 各层（已乘逐层系数）之和；microbatch 与 DP 取平均
            cores = [l.self_attention.core_attention for l in mm.decoder.layers]
            names = [f"{rg.start + i}" for i in range(len(cores))]
            if has_mtp:
                cores.append(mm.mtp.layers[0].mtp_model_layer.self_attention.core_attention)
                names.append("mtp0")
            per = torch.zeros(len(cores), device=dev)
            for j, c in enumerate(cores):
                if getattr(c, "indexer_loss_accum", None) is not None:
                    per[j] = c.indexer_loss_accum.float()
                    c.indexer_loss_accum = None
            dist.all_reduce(per, group=ps.get_data_parallel_group())
            # per-token 时 Megatron-Core 的 KD 返回逐行之和（未除行数），换成逐行平均再记
            per = per / GA / DP / (a.seq if a.per_token_loss == "on" else 1)
            sv["_sum.indexer_loss"] = float(per.sum())
            for j, nm in enumerate(names):  # 逐个产出层：逐行平均 KL × 逐层系数（shared 层不记）
                if cores[j].is_full:
                    sv[f"indexer_loss.{nm}"] = float(per[j])
        sv = pp_merge(sv)
        if "_sum.indexer_loss" in sv:
            sv["indexer_loss"] = sv.pop("_sum.indexer_loss")
        loss_v = sv.pop("lm_loss")
        vals = {
            "param_norm_fp64": float(psq.sqrt()),
            "update_ok": bool(ok),
            "peak_alloc_gib": float(mem[0]),
            **gsq,
            **sv,
        }
        if any(k.startswith("qk.clipped.") for k in sv):
            vals["qk.clipped_total"] = sum(v for k, v in sv.items() if k.startswith("qk.clipped."))
        max_logit = sv.get("max_attention_logit")
        # janus 运行目录：lm_loss / grad_norm / lr / wd / param_norm / skipped_iter / step_time / router.* 由 JanusDump 写，
        # 上面这些放进 step_vals 一起写
        dump.step_vals.update(vals)
        ctx.loss_dict = {"lm loss": torch.tensor(loss_v)}
        ctx.grad_norm = float(gnorm)
        ctx.skipped_iter = not ok
        dump.on_train_step_end(ctx)
        if rank == 0:
            print(
                f"STEP {step} loss={loss_v:.6f} gnorm={float(gnorm):.6f} pnorm={vals['param_norm_fp64']:.6f} "
                f"dt={dt:.3f}s mem={vals['peak_alloc_gib']:.1f}GiB ok={ok} maxlogit={max_logit}",
                flush=True,
            )
        if step in save_at:
            save_ckpt(step)
    if a.state_sha:
        state_sha(a.steps)
    dump.on_train_end(ctx)
    dist.barrier()
    if rank == 0:
        print("TRAIN_DONE", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
