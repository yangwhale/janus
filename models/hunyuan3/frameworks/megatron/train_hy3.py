"""混元 3 小模型训练入口（janus 第一阶段，GPU 自比）。

用法：torchrun --nproc-per-node 8 train_hy3.py --hf /data/hy3-4l --out /data/runs/<name> \
          [--load /data/hy3-4l-mcore] [--steps 20] [--gbs 16] [--seq 4096] [--deterministic] \
          [--janus-run <目录> --side golden-1 --dump-steps 1]
"""
import argparse
import datetime
import hashlib
import json
import os

p = argparse.ArgumentParser()
p.add_argument("--hf", required=True)
p.add_argument("--out", required=True)
p.add_argument("--load", default=None, help="Megatron 格式初始权重；不给则随机初始化（seed 固定）")
p.add_argument("--steps", type=int, default=20)
p.add_argument("--gbs", type=int, default=16)
p.add_argument("--mbs", type=int, default=1)
p.add_argument("--seq", type=int, default=4096)
p.add_argument("--ep", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--seed", type=int, default=1234)
p.add_argument("--deterministic", action="store_true")
p.add_argument("--data", default=None, help="Megatron 索引数据前缀；不给用 MockGPTDataset")
p.add_argument("--janus-run", default=None, help="janus 运行目录；给了才挂抓数回调")
p.add_argument("--side", default="golden-1")
p.add_argument("--dump-steps", default="1", help="逗号分隔的迭代号（从 1 开始）")
p.add_argument("--dump-grads", action="store_true", help="同时抓逐参数梯度；会关掉分布式优化器")
p.add_argument("--no-dist-opt", action="store_true", help="不用分布式优化器（与 --full 的优化器配置一致，但不挂抓数钩子）")
p.add_argument("--full", action="store_true", help="全量抓数：另抓子模块梯度、更新前后参数与优化器状态（含 --dump-grads）")
p.add_argument("--inject-from", default=None, help="逐层注入：参照运行目录（同一 dump 步）")
p.add_argument("--fault", default=None,
               help="故障注入（只用于验证检查会报警）：weight_scale:<参数正则>:<eps> | grad_scale:<参数正则>:<倍数> | "
                    "data_shift[:<起始步>] | data_seed | router_bf16 | dropout:<比例> | bias_freeze | lr_ulp | "
                    "grad_scale_seen:<参数正则>:<倍数>（抓梯度之前缩放） | grad_nan:<参数正则>:<步> | lr_spike:<步>:<倍数>")
p.add_argument("--tp", type=int, default=1)
p.add_argument("--cp", type=int, default=1)
p.add_argument("--recompute", action="store_true", help="开整层激活重算（抓数会另存重算那次前向）")
p.add_argument("--routing-replay", action="store_true",
               help="路由回放：反向重算沿用首次前向选中的专家（top-k 下标），不重新选；要求 --recompute（见 routing_replay.py）")
p.add_argument("--full-steps", default=None, help="做全量抓数的步（逗号分隔），默认 dump 步里最小的一步")
p.add_argument("--full-only", default=None, help="全量抓数只抓名字匹配这个正则的参数（多步抓数时控制体积）")
p.add_argument("--save-batches", action="store_true", help="每 rank 每步的原始输入整批存下（另一侧原样回放同一批数据用）")
p.add_argument("--save", default=None, help="存 Megatron 检查点到这个目录")
p.add_argument("--save-interval", type=int, default=None, help="每多少步存一次（默认只在最后存）")
p.add_argument("--resume", default=None, help="从这个检查点目录续训（恢复迭代号、优化器状态、数据位置）")
p.add_argument("--observe", default=None, help="可选观测点，逗号分隔：attn,moe,expw,gemm（见 janus/dump/megatron.py）")
p.add_argument("--special", default=None,
               help="特殊值存 / 恢复：参数名正则。不带 --resume 时在最后一步存档前往命中的参数里埋 −0.0 与次正规数；"
                    "带 --resume 时在开训前把命中的张量读出来，写进 <janus-run>/special/")
p.add_argument("--no-tf32", action="store_true",
               help="关 TF32：torch 的 matmul / cuDNN 开关设为 False。TE 的 FP32 GEMM 不看 torch 开关，"
                    "还要在环境里设 NVIDIA_TF32_OVERRIDE=0（run 脚本负责），见 docs/reference/environment.md")
p.add_argument("--ce-fusion", default=None, choices=["on", "off"],
               help="诊断臂：交叉熵融合（cross_entropy_loss_fusion）；默认不改")
p.add_argument("--bias-act-fusion", action="store_true",
               help="诊断臂：门控激活走融合核（bias_activation_fusion=True，silu·up 一次算完，接近 TPU 的「FP32 算完舍一次」）")
p.add_argument("--attention-backend", default=None, choices=["flash", "fused", "unfused", "local", "auto"],
               help="诊断臂：注意力实现（Megatron AttnBackend；默认不改 = 提供者的默认值）")
p.add_argument("--row-stats-every", type=int, default=0,
               help="每 N 步记输出层、embedding 每行的主权重平方和与输出层每行 Adam 更新量（与 TPU 侧同名）")
p.add_argument("--group-norms", action="store_true",
               help="每步另记分组 FP32 主权重平方和（group_sq.*）与每个专家的平方和（expert_sq.*），定位参数范数漂移用")
p.add_argument("--grad-reduce-bf16", action="store_true",
               help="诊断臂：梯度累加与 DP 规约用 BF16（ddp.grad_reduce_in_fp32=False）。默认 FP32；不用于 golden")
p.add_argument("--router-true-fp32", action="store_true",
               help="（只作诊断，不用于 golden）路由真 FP32：路由权重留成 FP32 参数（即主权重本身），路由 logits 的 GEMM 前向、反向都是 FP32 × FP32 的 "
                    "torch.mm；隐含 --no-tf32，并要求环境里 NVIDIA_TF32_OVERRIDE=0（见 router_fp32.py）")
p.add_argument("--resid-probe-every", type=int, default=0,
               help="每 N 步（和第 1 步）在参数更新后用当前参数在固定序列上另跑一次前向（no_grad，不改训练），记各层残差流、"
                    "最后归一化输出的 token 均值与 logits / lse 均值（resid_mean.* / probe_*，与 TPU 侧同名，见 resid_probe.py）；0 = 不记")
p.add_argument("--resid-probe-seq", default=None,
               help="--resid-probe-every 用的固定序列 .npz（tokens、position_ids，[1, S]）：golden 第 1 步 rank 0 第一个 microbatch")
p.add_argument("--rr-arm", default=None,
               help="反向复现诊断臂（逗号分隔）：把一处数值行为换成 TPU（MaxText）的写法，见 rr_arms.py；不用于 golden")
p.add_argument("--export-params-at", default=None,
               help="逗号分隔的步号：该步更新之后把整套 FP32 主权重按 rank 导出到 --export-dir/step_<N>/（param_swap.py）")
p.add_argument("--export-dir", default=None)
p.add_argument("--import-params", default=None,
               help="开训前把整套参数换成这个目录里的（param_swap.py 的格式：rank<r>.safetensors + manifest.json），读回逐位核对")
p.add_argument("--import-round-bf16", action="store_true",
               help="--import-params 导入时先把 FP32 主权重舍到 BF16 再放回（FP32 参照用 BF16 程序实际参与计算的那份权重，只比算术误差）")
p.add_argument("--skip-steps", type=int, default=0,
               help="开训时先丢掉 K 步的数据，本侧第 1 步吃原数据第 K+1 步的 batch（配 --import-params 重算某一步的梯度）")
p.add_argument("--fp32", action="store_true",
               help="（只作诊断参照）全程 FP32：参数、激活、梯度都是 FP32；要配 --no-tf32、NVIDIA_TF32_OVERRIDE=0 与 --attention-backend unfused")
p.add_argument("--bf16-weights", action="store_true",
               help="（配 --fp32，只作诊断参照）BF16 权重、全程 FP32 计算：每步前向前把参数舍到 BF16，优化器更新前换回 FP32 主权重"
                    "（param_swap.Bf16WeightsFp32Compute）。整条轨迹是 BF16 训练的精确版本")
a = p.parse_args()
if a.bf16_weights and not a.fp32:
    p.error("--bf16-weights 要配 --fp32")
if a.export_params_at and not a.export_dir:
    p.error("--export-params-at 要同时给 --export-dir")
if a.resid_probe_every and not (a.resid_probe_seq and a.janus_run):
    p.error("--resid-probe-every 要同时给 --resid-probe-seq 和 --janus-run（值写进 metrics.jsonl）")
if a.routing_replay and not a.recompute:
    p.error("--routing-replay 要求 --recompute：不重算时反向不会重跑路由，没有可回放的地方")
if a.router_true_fp32:
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        p.error("--router-true-fp32 要求 NVIDIA_TF32_OVERRIDE=0（只有它能关掉 TE 与 cuBLAS 的 TF32），"
                "用 run_check.sh 的 TF32=off 或手动 export")
    a.no_tf32 = True
if a.janus_run:                                    # side 写错要在起训练之前就停，不要跑完才在 janus run 时发现
    from janus.store import side_error
    if side_error(a.side):
        p.error("--" + side_error(a.side))

import torch  # noqa: E402

if a.no_tf32:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# TF32 开关读回值与相关环境变量（不碰 CUDA：cuBLAS 句柄要等 Megatron 设好 CUBLAS_WORKSPACE_CONFIG 之后再建）；
# 实测（FP32 GEMM 对 FP64 的误差）在抓数回调的 on_train_start 里做，见 janus/dump/megatron.py 的 tf32_probe
TF32 = {"allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "env": {k: os.environ.get(k) for k in ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE")}}

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain

cfg = _pretrain_common()
m = AutoBridge.from_hf_pretrained(a.hf, trust_remote_code=True).to_megatron_provider(load_weights=False)
m.tensor_model_parallel_size = a.tp
m.context_parallel_size = a.cp
if a.recompute:
    m.recompute_granularity = "full"
    m.recompute_method = "uniform"
    m.recompute_num_layers = 1
if a.routing_replay:
    m.moe_enable_routing_replay = True
m.pipeline_model_parallel_size = 1
m.expert_model_parallel_size = a.ep
m.sequence_parallel = a.tp > 1
m.seq_length = a.seq
m.mtp_num_layers = None
m.moe_token_dispatcher_type = "alltoall"
if a.ce_fusion:
    m.cross_entropy_loss_fusion = a.ce_fusion == "on"
if a.bias_act_fusion:
    m.bias_activation_fusion = True
if a.attention_backend:
    from megatron.core.transformer.enums import AttnBackend
    m.attention_backend = AttnBackend[a.attention_backend]
m.moe_router_bias_update_rate = getattr(m, "moe_router_bias_update_rate", 1e-3) or 1e-3
if a.router_true_fp32:
    import router_fp32
    router_fp32.install(m)
fault = {}
if a.fault:
    kind, *rest = a.fault.split(":")
    fault = {"kind": kind}
    if kind in ("weight_scale", "grad_scale", "grad_scale_seen"):
        fault.update(param=rest[0], **({"eps": rest[1]} if kind == "weight_scale" else {"factor": rest[1]}))
    elif kind == "grad_nan":
        fault.update(param=rest[0], step=int(rest[1]))
    elif kind == "lr_spike":
        fault.update(step=int(rest[0]), factor=float(rest[1]))
    elif kind == "data_shift":
        fault["from_step"] = int(rest[0]) if rest else 1
    elif kind == "data_seed":
        pass                                  # 在下面改 dataset.random_seed：数据顺序整个不同
    elif kind == "router_bf16":
        m.moe_router_dtype = None
    elif kind == "dropout":
        m.hidden_dropout = float(rest[0])
        m.attention_dropout = float(rest[0])
    elif kind == "bias_freeze":
        m.moe_router_bias_update_rate = 0.0
    elif kind != "lr_ulp":
        raise SystemExit(f"未知故障 {kind}")
if a.rr_arm:
    import rr_arms
    rr_arms.apply_model_config(m, rr_arms.parse(a.rr_arm))
if a.deterministic:
    m.deterministic_mode = True
    m.cross_entropy_loss_fusion = False
if a.fp32:
    from megatron.bridge.training.mixed_precision import MixedPrecisionConfig
    m.bf16 = False
    m.params_dtype = torch.float32
    cfg.mixed_precision = MixedPrecisionConfig(fp32=True, params_dtype=torch.float32, pipeline_dtype=torch.float32,
                                               autocast_enabled=False, grad_reduce_in_fp32=True)
cfg.model = m

t = cfg.train
t.train_iters = a.steps
t.global_batch_size = a.gbs
t.micro_batch_size = a.mbs
t.manual_gc = False
cfg.validation.eval_interval = 10 ** 9
cfg.validation.eval_iters = 0
if fault.get("kind") == "lr_ulp":
    import numpy as _np
    a.lr = float(_np.nextafter(_np.float32(a.lr), _np.float32(1)))
cfg.optimizer.lr = a.lr
cfg.optimizer.min_lr = a.lr
cfg.optimizer.log_num_zeros_in_grad = True
cfg.scheduler.lr_warmup_iters = 0
cfg.scheduler.lr_decay_style = "constant"
cfg.ddp.overlap_grad_reduce = False
if a.grad_reduce_bf16:
    cfg.ddp.grad_reduce_in_fp32 = False
cfg.ddp.overlap_param_gather = False
cfg.ddp.check_for_nan_in_grad = fault.get("kind") != "grad_nan"   # 埋 NaN 时让它流到优化器，看检查在哪一步报
if a.dump_grads or a.full or a.no_dist_opt:
    # 抓梯度要每个 rank 上都是完整的已归约梯度：不用分布式优化器、不切分
    cfg.ddp.use_distributed_optimizer = False
    cfg.optimizer.use_distributed_optimizer = False
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"
cfg.dataset.seq_length = a.seq
if a.data:
    cfg.dataset.blend = ([a.data], None)
    cfg.dataset.split = "9999,1,0"
cfg.dataset.random_seed = a.seed + (1 if fault.get("kind") == "data_seed" else 0)
cfg.rng.seed = a.seed
cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
cfg.tokenizer.tokenizer_model = a.hf
cfg.logger.log_interval = 1
cfg.logger.tensorboard_dir = os.path.join(a.out, "tb")
cfg.logger.log_timers_to_tensorboard = False
cfg.checkpoint.save = a.save
cfg.checkpoint.save_interval = (a.save_interval or a.steps) if a.save else 10 ** 9
cfg.checkpoint.load = a.resume
if a.load and not a.resume:
    cfg.checkpoint.pretrained_checkpoint = a.load

callbacks = []
if a.import_params or a.export_params_at:
    import param_swap
    # 排在抓数回调之前：抓数的 params_sha.init 记的是换过的参数
    callbacks.append(param_swap.ParamSwap(import_dir=a.import_params, export_dir=a.export_dir,
                                          export_steps={int(x) for x in (a.export_params_at or "").split(",") if x},
                                          round_bf16=a.import_round_bf16))
if a.bf16_weights:
    import param_swap
    callbacks.append(param_swap.Bf16WeightsFp32Compute())
if a.rr_arm:
    import rr_arms
    rr_arms.install_global(rr_arms.parse(a.rr_arm))
    callbacks.append(rr_arms.RRCallback(rr_arms.parse(a.rr_arm), out=a.janus_run or a.out))
if a.routing_replay:
    from routing_replay import RoutingReplay
    callbacks.append(RoutingReplay())
if a.janus_run:
    from janus.dump.megatron import JanusDump

    def sha(path):
        return hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.isfile(path) else None

    from janus.dump.runinfo import image_ref
    from janus.dump.runinfo import pkg_version as ver

    first_step = 1
    if a.resume and os.path.exists(os.path.join(a.resume, "latest_checkpointed_iteration.txt")):
        first_step = int(open(os.path.join(a.resume, "latest_checkpointed_iteration.txt")).read().strip()) + 1

    def _js(v):
        return v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
    image, image_warn = image_ref("nvcr.io/nvidia/nemo:26.10")
    if image_warn and int(os.environ.get("RANK", 0)) == 0:
        print("[janus] 警告：" + image_warn, flush=True)
    det_env = {k: os.environ.get(k) for k in ("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "CUBLAS_WORKSPACE_CONFIG", "NCCL_ALGO")}
    meta = {
        "schema_version": 1, "run_name": os.path.basename(a.janus_run.rstrip("/")), "side": a.side,
        "framework": "megatron", "model": "hunyuan3-4l",
        "code": {"repo": "NVIDIA-NeMo/Megatron-Bridge (nemo:26.10)", "commit": "container"},
        "image": image,
        "versions": {"torch": torch.__version__, "cuda": torch.version.cuda,
                     "megatron_core": ver("megatron.core", "megatron-core"),
                     "megatron_bridge": ver("megatron.bridge", "megatron-bridge"),
                     "transformer_engine": ver("transformer_engine", "transformer-engine"), "nccl": ".".join(map(str, torch.cuda.nccl.version()))},
        "hardware": {"accelerator": torch.cuda.get_device_name(0), "count": int(os.environ.get("WORLD_SIZE", 1)), "topology": "1 node"},
        "parallelism": {"dp": int(os.environ.get("WORLD_SIZE", 1)) // (a.tp * a.cp), "fsdp": 1, "tp": a.tp, "ep": a.ep, "cp": a.cp, "pp": 1},
        "config": {"path": os.path.abspath(__file__), "sha256": sha(os.path.abspath(__file__)), "args": vars(a)},
        "checkpoint": {"path": a.load or "random-init", "manifest_sha256": sha(os.path.join(a.hf, "tensor_sha256.json"))},
        "data": {"manifest": a.data or "MockGPTDataset", "manifest_sha256": sha(a.data + ".idx") if a.data else None,
                 "seed": cfg.dataset.random_seed},
        "steps": {"first": first_step, "last": a.steps,
                  "convention": f"golden_step = step + {a.skip_steps}" if a.skip_steps else "golden_step = step"},
        "determinism": {"deterministic_mode": a.deterministic, **det_env},
        "tf32": TF32,
        "param_swap": {"import": a.import_params, "export_at": a.export_params_at, "export_dir": a.export_dir,
                       "skip_steps": a.skip_steps, "import_round_bf16": a.import_round_bf16},
        "dtypes": {"compute": "fp32" if a.fp32 else "bf16", "weight": ("fp32 main params, rounded to bf16 for fwd/bwd each step" if a.bf16_weights else "fp32 params" if a.fp32 else "bf16 + fp32 main params") + ("; router weight fp32 param" if a.router_true_fp32 else ""),
                   "optimizer_states": "fp32", "grad_reduce": "fp32" if cfg.ddp.grad_reduce_in_fp32 else "bf16"},
        # 路由 GEMM 怎么算；--router-true-fp32 时训练结束前补上 probe（每次路由 GEMM 对 FP64 的误差，见 router_fp32.py）
        "router": ({"mode": "true_fp32", "weight_param_dtype": "fp32", "gemm": "torch.mm fp32 x fp32 (fwd, dgrad, wgrad)",
                    "patch": "models/hunyuan3/frameworks/megatron/router_fp32.py"} if a.router_true_fp32 else
                   {"mode": "te_general_gemm fp32-in fp32-out（--fp32：全部参数是 FP32" + ("，前反向用其 BF16 舍入）" if a.bf16_weights else "）"),
                    "weight_param_dtype": "fp32"} if a.fp32 else
                   {"mode": "te_general_gemm bf16-in fp32-out", "weight_param_dtype": "bf16"}),
        "hooks": {"janus_dump": "janus.dump.megatron.JanusDump", "dump_steps": a.dump_steps, "dump_grads": a.dump_grads,
                  "full": a.full, "inject_from": a.inject_from},
        "fault": fault or None,
        # 激活重算与路由回放（golden 用的设置写在 models/hunyuan3/config-alignment.md「重算」行）
        "recompute": {k: _js(getattr(m, k, None)) for k in (
            "recompute_granularity", "recompute_method", "recompute_num_layers", "moe_enable_routing_replay")},
        "model_config": {k: _js(getattr(m, k, None)) for k in (
            "num_layers", "hidden_size", "num_attention_heads", "num_query_groups", "kv_channels", "ffn_hidden_size",
            "num_moe_experts", "moe_router_topk", "moe_ffn_hidden_size", "moe_shared_expert_intermediate_size",
            "moe_router_dtype", "moe_router_score_function", "moe_router_enable_expert_bias", "moe_router_bias_update_rate",
            "moe_router_topk_scaling_factor", "moe_aux_loss_coeff", "moe_z_loss_coeff", "hidden_dropout", "attention_dropout",
            "normalization", "layernorm_epsilon", "qk_layernorm", "rotary_base", "position_embedding_type",
            "untie_embeddings_and_output_weights", "init_method_std", "bf16", "fp8", "deterministic_mode", "attention_backend", "cross_entropy_loss_fusion", "bias_activation_fusion")},
        "optimizer_config": {k: _js(getattr(cfg.optimizer, k, None)) for k in (
            "optimizer", "lr", "min_lr", "weight_decay", "adam_beta1", "adam_beta2", "adam_eps", "clip_grad",
            "use_distributed_optimizer", "main_params_dtype", "exp_avg_dtype", "exp_avg_sq_dtype")},
        # 调度器的权重衰减会覆盖 optimizer.weight_decay（Bridge 默认 start/end_weight_decay = 0.033）
        "scheduler_config": {k: _js(getattr(cfg.scheduler, k, None)) for k in (
            "lr_decay_style", "lr_warmup_iters", "start_weight_decay", "end_weight_decay", "weight_decay_incr_style")},
        "created_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds"),
    }
    dump = JanusDump(a.janus_run, meta, dump_steps={int(x) for x in a.dump_steps.split(",") if x},
                     dump_grads=a.dump_grads, full=a.full, inject_from=a.inject_from, fault=fault,
                     full_steps={int(x) for x in a.full_steps.split(",")} if a.full_steps else None,
                     only=a.full_only, recompute=a.recompute, save_batches=a.save_batches, observe=a.observe,
                     special=({"phase": "load" if a.resume else "save", "step": a.steps, "params": a.special}
                              if a.special else None), group_norms=a.group_norms,
                     row_stats_every=a.row_stats_every)
    if a.resid_probe_every:
        from resid_probe import ResidProbe
        probe = ResidProbe(a.resid_probe_every, a.resid_probe_seq)
        probe.dump = dump
        meta["resid_probe"] = {"every": a.resid_probe_every, "seq": os.path.abspath(a.resid_probe_seq),
                               "tokens_sha256": probe.tokens_sha256}
        callbacks.append(probe)                      # 要排在 dump 之前：值进 dump 这一步的 step_vals
    callbacks.append(dump)
    from janus.dump.megatron import wrap_forward_step
    forward_step = wrap_forward_step(forward_step, dump)
    if a.router_true_fp32:
        _orig_start, _orig_end = dump.on_train_start, dump.on_train_end

        def _start(self, context):
            _orig_start(context)
            bad = router_fp32.check_after_setup(dump._params)
            if bad:
                raise RuntimeError(f"--router-true-fp32：这些路由权重不是 FP32 参数：{bad}")

        def _end(self, context):                   # DONE 之前把 rank 0 的探针结果写进 meta.json
            if dump.rank0:
                path = os.path.join(a.janus_run, "meta.json")
                mj = json.load(open(path))
                pr = router_fp32.PROBE
                mj["router"]["probe"] = {
                    **{f"{k}_rel_l2_max": max((r["rel_l2_vs_fp64"] for r in v), default=None) for k, v in pr.items()},
                    "fwd_te_bf16_gemm_rel_l2_max": max((r.get("te_bf16_gemm_rel_l2_vs_fp64", 0) for r in pr["fwd"]), default=None),
                    "calls": pr, "rank": 0, "step": first_step}
                with open(path, "w") as f:
                    json.dump(mj, f, ensure_ascii=False, indent=2)
            _orig_end(context)
        import types  # 回调管理器按绑定方法识别回调，要绑成方法
        dump.on_train_start, dump.on_train_end = types.MethodType(_start, dump), types.MethodType(_end, dump)

if a.skip_steps:
    import param_swap
    forward_step = param_swap.wrap_skip(forward_step, a.skip_steps * (a.gbs // a.mbs // (int(os.environ.get("WORLD_SIZE", 1)) // (a.tp * a.cp))))
pretrain(cfg, forward_step, callbacks=callbacks or None)
if a.router_true_fp32 and int(os.environ.get("RANK", 0)) == 0:
    print("[janus] router_fp32 probe " + json.dumps({k: [round(r["rel_l2_vs_fp64"], 12) for r in v]
                                                     for k, v in router_fp32.PROBE.items()}), flush=True)
