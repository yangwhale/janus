"""混元 4 截层 TPU 训练入口（janus：TPU 对 GPU golden）。从混元 3 的 train_hy3.py 改来，分阶段（--stage A/B/C/D）。

不改 MaxText：用 MaxText 的 setup_train_loop 建模型、优化器、mesh，载入转换好的检查点；训练循环自己写，
每步把 golden 同一步的全局 batch 原样喂进去（replay.py），调 MaxText 自己的 train_step（jit 后）。

用法（每个 TPU 主机上都跑同一条命令；只有进程 0 写运行目录）：
  python train_hy4.py --stage A --load gs://…/hy4-a-maxtext/0/items --golden /data/golden/ga-a \\
      --hf-config /data/hy4-4l/config.json --hf-manifest /data/hy4-4l/tensor_sha256.json --steps 300 \\
      --janus-run /data/janus-runs/t-a --dump-steps 1,300 --full [MaxText key=value …]

  最小可跑（先看 loss）：--dump-steps none --no-param-sha，只写 metrics.jsonl 与输入指纹
  --cpu-small        CPU 冒烟：hy4-tiny 维度、8 个假 CPU 设备（cpu_smoke.py）

和混元 3 的不同：
- 参数类张量写 **GPU golden 的名字与 HF 布局**（GPU golden 是 HF 实现，参数在 GPU 上就是 HF 的形状；名字是 transformers
  加载后的模块名 hf.…，hy4_map.gpu_name），换布局用 hf_view.py（与 L0 逐位核对同一段反向变换）。
  参数清单记 FP32 主权重（GPU 记的也是 FP32 主权重）；全量抓数的 param.* 同样是 FP32 主权重。
- 逐步另记分组参数平方和 group_sq.<组>（组名同 GPU golden；每个叶子 FP32 求平方和、主机 FP64 按组合成；PROTOCOL.md
  第 3 节：param_norm 按参数组分别求平方和再合成），以及合成后的 param_norm_grouped；与 GPU 同样记第 s 步更新之前的参数。
- 梯度累积 2（对 GPU 8 rank × 2 micro）时 MaxText 的加钩子训练步不支持，图内激活不抓（与混元 3 相同）；
  激活与逐层比另用逐层回放（待移植）。状态类（参数、Adam 动量、路由偏置）照抓。

步号：本侧第 s 步吃 golden 第 s 步的 batch，metrics 也记在第 s 步（golden_step = step）。
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mt_common  # noqa: E402
import replay  # noqa: E402
from hf_view import HFView  # noqa: E402
from hy4_map import gpu_name, gpu_rank_pieces  # noqa: E402

from janus.dump import maxtext as jd  # noqa: E402

# CPU 冒烟：hy4-tiny 的维度（结构与 hy4-preview 相同：MLA + indexer + 门控 + sink + iHC + hc_head、稠密 / MoE、MTP）
CPU_SMALL = ["hardware=cpu", "skip_jax_distributed_system=True", "model_name=hy4-tiny", "checkpoint_rope_interleave=true",
             "attention=dot_product", "sparse_matmul=False", "megablox=False", "use_tokamax_splash=False"]


# 分组参数平方和的组：组名与 GPU golden 的 metrics.jsonl（group_sq.<组>）相同。阶段 A 的组按 GPU ga-b 第 1 步的值
# 对 HF 权重逐组复算对上（norms = 每层 input / post_attention / q_a / kv_a 四个 RMSNorm，attn = 五个 MLA 投影）。
# MoE 的组名（router / experts / shared）已对上 GPU gb-a 的 metrics.jsonl；MTP 的组名（mtp_norms / mtp_proj）待阶段 D 的 golden 核
GROUP_RULES = [(r"/token_embedder/", "embedding"), (r"/logits_dense/", "output"), (r"/decoder/decoder_norm/", "final_norm"),
               (r"/decoder/hc_head/", "hc_head"), (r"/mtp_block/.*/mtp_\d+_(embedding_norm|hidden_state_norm|final_norm)/", "mtp_norms"),
               (r"/mtp_block/.*/mtp_\d+_projection/", "mtp_proj"), (r"/mhc_(attention|mlp)/", "ihc"), (r"/indexer/", "indexer"),
               (r"/self_attention/sinks$", "sinks"), (r"/self_attention/gate_proj/", "attn_gate"),
               (r"/self_attention/(wq_a|wq_b|wkv_a|wkv_b|out)/", "attn"),
               (r"/((pre|post)_self_attention_layer_norm|self_attention/(q_norm|kv_norm))/", "norms"),
               (r"/dense_layers/mlp/", "dense_mlp"), (r"/MoeBlock_0/gate/", "router"), (r"/MoeBlock_0/w(i_0|i_1|o)$", "experts"),
               (r"/shared_experts/", "shared")]


def group_of(path: str) -> str:
    import re
    g = next((g for pat, g in GROUP_RULES if re.search(pat, path)), None)
    if g is None:
        raise SystemExit(f"参数 {path} 没有分组规则（train_hy4.GROUP_RULES）")
    return g


def sha_file(path) -> str | None:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=sorted(mt_common.STAGES))
    p.add_argument("--load", required=True, help="MaxText 参数检查点（convert_hf_to_maxtext.py 的 …/0/items）")
    p.add_argument("--golden", required=True, help="golden 运行目录（本地，至少要有 meta.json 与 data/batches/）")
    p.add_argument("--hf-config", required=True, help="截好的 HF 目录的 config.json（换 HF 名字用）")
    p.add_argument("--hf-manifest", default=None, help="HF 目录的 tensor_sha256.json（meta.checkpoint.manifest_sha256 用）")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--first-step", type=int, default=1)
    p.add_argument("--save-every", type=int, default=0,
                   help="每隔这么多步把整个训练状态（参数、优化器状态、路由偏置、步数）存到 --save-dir/step_<N>（Orbax，多主机一起写）")
    p.add_argument("--save-dir", default="", help="存档根目录（gs://…；spot 切片被收走时 Pod 本地盘会丢）")
    p.add_argument("--save-last", action="store_true", help="最后一步也存档（一个作业只跑一段、下一段接着跑时用）")
    p.add_argument("--resume", default="",
                   help="存档目录（…/step_<N>）：建好状态后整个换成存档里的值，从第 N+1 步接着回放；要求同一配置与同样的设备数")
    p.add_argument("--janus-run", required=True, help="运行目录（本地路径；跑完由 run.sh 传到 GCS）")
    p.add_argument("--side", default="tpu")
    p.add_argument("--dump-steps", default="1", help="逗号分隔，本侧步号（= golden 步号）；none = 不抓张量")
    p.add_argument("--full", action="store_true", help="全量抓数：更新前后 FP32 主权重、Adam 两个动量（HF 名字）")
    p.add_argument("--full-steps", default=None, help="做全量抓数的步，默认 dump 步里最小的一步")
    p.add_argument("--full-only", default=None, help="全量抓数只抓 HF 名匹配的参数（逗号分隔的正则按「或」合并）；"
                   "--diag-step1 时 diag_grads.safetensors 只存匹配的张量")
    p.add_argument("--no-param-sha", action="store_true", help="不写参数清单")
    p.add_argument("--state-sha", action="store_true", help="另写训练状态的 FP32 全量清单 state_sha/{init,final}.json")
    p.add_argument("--no-group-norms", action="store_true", help="不记分组参数平方和")
    p.add_argument("--bf16-grid-steps", default="",
                   help="逗号分隔的步号（0 = 训练前的初始权重）：在这些步更新之后数 FP32 主权重里不能被 BF16 精确表示的元素比例，"
                        "记成 bf16_offgrid_frac（全体，不含路由偏置）与 bf16_offgrid_frac.<分组>；默认不记")
    p.add_argument("--hlo-dir", default=None, help="XLA dump 目录：训练结束后把 jit_train_step 优化后的 HLO 拷进运行目录 hlo/")
    p.add_argument("--diag-step1", action="store_true",
                   help="诊断：只对第 --first-step 步算一次梯度（与训练同一个梯度累积函数，不更新），写 diag_step1.json（GPU 同名的"
                        "梯度分组平方和、各产出层的索引器 KD、loss）与 diag_grads.safetensors（HF 名字与布局的 FP32 梯度），然后退出")
    p.add_argument("--hf-snapshot-at", default="",
                   help="逗号分隔的步号：这一步训练之前（= 上一步更新之后）把模型权重按 HF 检查点的名字、布局与 dtype"
                        "（HF 里存 BF16 的张量取 FP32 主权重的 BF16 舍入，存 FP32 的原样；路由偏置一并）写到运行目录"
                        " hf_snapshot/step_<N>/，能直接当 HF 目录给 GPU 与 CPU 参照载入（同状态单步对照用，只作诊断）")
    p.add_argument("--hf-snapshot-only", default="",
                   help="和 --hf-snapshot-at 一起用：只写 HF 名匹配这个预设正则的张量，并且写 FP32 主权重原值（不舍 BF16）。"
                        "预设 gpu-fp32 = GPU 模型里留 FP32、HF 却存 BF16 的参数（输出头、索引器 k_norm）")
    p.add_argument("--hf-snapshot-verify", default="",
                   help="逗号分隔的步号：这一步训练之前只核不写——每个张量按上面的规则转成 HF dtype 后的字节 sha256 对 "
                        "--hf-manifest；第 1 步应全部相同（核快照这条路径本身）")
    p.add_argument("--eval-variants-at", type=int, default=0,
                   help="诊断：训练到这一步之前（= 上一步更新之后）停下，释放优化器状态，对同一份权重按 --eval-variants 逐个"
                        "「把选中的参数舍成 BF16」后只做前向，记 lm_loss / MTP / KD（不更新），写 eval_variants_step<N>.json 后退出。"
                        "只舍 HF 检查点里存 BF16 的张量（存 FP32 的 iHC、sink、hc_head 与路由偏置不动）")
    p.add_argument("--eval-variants", default="none=;all=.",
                   help="分号分隔的「名字=HF 名正则」；正则为空 = 不舍；按 MaxText 叶子整片舍（扫描层的同一叶子各层一起舍）")
    p.add_argument("--diag-free-opt", action="store_true",
                   help="和 --diag-step1 一起用：算梯度前把优化器状态从设备上释放，只把模型变量传进梯度函数（梯度的计算不变）。"
                        "截层 D3 在 v5p 16 片上不释放时差 1.3 GB 装不下")
    p.add_argument("--diag-timing", type=int, default=0,
                   help="和 --diag-step1 一起用：先单独编译（记编译秒数与 memory_analysis），算完梯度后再重复算 N 次计时，"
                        "并记每设备 peak_bytes_in_use，写进 diag_step1.json 的 timing；0 = 不计时（原行为）")
    p.add_argument("--param-override", default=None,
                   help="从别处的状态起步：HF 名字的 FP32 主权重 safetensors（本地或 gs://；如 GPU export_hf_grads.py --kind main_param "
                        "导出的第 N 步更新前主权重），换进训练状态的参数。只换映射里引用到的叶子；缺张量或路由专家不全直接报错，"
                        "不补 0。优化器状态不动（从零开始）。配 --diag-step1 --first-step N 就是第 N 步同参数下的单步前向 / 反向")
    p.add_argument("--override-only", default=None, help="--param-override 只换 HF 名匹配这个正则的张量所在的叶子（默认全换）")
    p.add_argument("--bias-override", default=None,
                   help="路由偏置换成 GPU 全量抓数某一步的 expert_bias：step 目录（本地或 gs://，读其中的 buffer.*.safetensors，"
                        "即该步更新前的偏置）；每个 MoE 层（含 MTP）都要有，缺了报错")
    p.add_argument("--forced-routing", default=None,
                   help="路由回放（MaxText 开关 debug_forced_routing_deepseek，会自动打开）：build_forced_routing.py 的输出（本地或 gs://），"
                        "每一行按 token 逐个相等对上本侧 batch 的行，对不上直接报错。只灌专家下标，权重照算。只和 --diag-step1 一起用；"
                        "diag_step1.json 的 forced_routing 记来源、行对应与审计（各层实际用的下标对文件逐位比、与本侧自选相同的 token 比例）")
    p.add_argument("--forced-indexer", default=None,
                   help="DSA 索引器 top-k 回放（MaxText 开关 debug_forced_indexer_topk，会自动打开）：build_forced_routing.py --indexer "
                        "的输出；行对应与审计同 --forced-routing，可以和它一起用")
    p.add_argument("--capture-choices", action="store_true",
                   help="只抓不灌（MaxText 开关 debug_capture_choices，会自动打开，不改数值）：全局批每一行前向一次，存本侧自选的路由下标、"
                        "加偏置后的路由分数与 DSA 索引器 top-k，写运行目录的 choices_captured.safetensors（格式同 build_forced_routing.py 的"
                        "输出，行序按 GPU 的 R = m·rows + r，另加 router_scores）。只和 --diag-step1 一起用")
    p.add_argument("--capture-only", action="store_true", help="和 --capture-choices 一起用：抓完就退出，不算梯度")
    p.add_argument("--aot-only", action="store_true",
                   help="只编译训练步、打印编译期显存分析（memory_analysis）后退出，不训练（真实切片上的 AOT 显存检查）")
    p.add_argument("--cpu-small", action="store_true")
    p.add_argument("--cpu", action="store_true",
                   help="真实维度在 CPU 上跑（8 个假设备、dot_product 注意力、稠密 MoE）：只作接线核对（第 1 步 loss / grad_norm 的量级），"
                        "不是 TPU 的数值，不进任何比较")
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
    import jax
    env = {k: v for k, v in sorted(os.environ.items()) if k.startswith(FLAG_ENV_PREFIXES)}
    try:
        values = dict(jax.config.values)
    except Exception:  # noqa: BLE001
        values = {}
    return {"env": env, "jax_config": {k: v for k, v in sorted(values.items()) if isinstance(v, (str, int, float, bool, type(None)))}}


def maxtext_full_config(config) -> dict:
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
    """XLA dump 目录里训练步（jit_train_step）的产物 → 运行目录 hlo/：优化后的 HLO、编译器生效的开关，另记指纹。
    同 ../../../hunyuan3/frameworks/maxtext/train_hy3.py 的 collect_hlo；指纹（fingerprint）= 去掉 metadata 与 stack_frame_id
    之后的 sha256（janus.checks.extra.hlo_fingerprint），改源码行号不改它，改融合、调度、算子、精度都改它。
    同名模块编了不止一次时全部按出现顺序编号保留，并记 error（不知道执行的是哪一个）。"""
    import shutil

    from janus.checks.extra import hlo_fingerprint
    src, dst = Path(hlo_dir), Path(run_dir) / "hlo"
    pat = re.compile(r"^module_(\d+)\.jit_train_step(?:\.cl_\d+)?\.((?:cpu_)?after_optimizations\.txt|flagfile|debug_options)$")
    found: dict[str, list[Path]] = {}
    for f in sorted(src.glob("module_*.jit_train_step.*")) if src.is_dir() else []:
        m = pat.match(f.name)
        if m:
            found.setdefault(m.group(1), []).append(f)
    # 只保留真正编译出可执行文件（有优化后 HLO）的模块；只有 flagfile / debug_options 的模块（trace 时先写下的）记在 skipped
    skipped = sorted(cl for cl, fs in found.items() if not any("after_optimizations" in f.name for f in fs))
    found = {cl: fs for cl, fs in found.items() if cl not in skipped}
    info = {"source": str(src), "modules": sorted(found), "skipped_modules": skipped, "files": {}, "fingerprint": {}}
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
            if suffix == "after_optimizations.txt":
                info["fingerprint"][name] = hlo_fingerprint((dst / name).read_text())
    if len(found) > 1:
        info["error"] = f"jit_train_step 编译了 {len(found)} 次"
    return info


def _state_flat(pure) -> dict:
    """存档用：训练状态的叶子按展平顺序编号（l00000 …），去掉大小为 0 的叶子（被掩掉的优化器状态，Orbax 存不了）。
    同一配置建出来的状态展平顺序相同，所以编号对得上。"""
    import jax

    leaves = jax.tree_util.tree_leaves(pure)
    return {f"l{i:05d}": v for i, v in enumerate(leaves) if getattr(v, "size", 1)}


def device_memory_probe(tag: str) -> dict:
    """本进程第一个设备上的显存账：jax.live_arrays() 落在这个设备上的分片字节数与个数，加上运行时的 memory_stats
    （CPU 上没有 memory_stats）。编译训练步之前调，看有没有不该活着的数组（如读档后旧状态的副本）占着显存。"""
    import gc

    import jax

    gc.collect()   # NNX 的图对象有引用环：不收一次，已经没人用的数组还算活着
    dev = jax.local_devices()[0]
    n, nbytes = 0, 0
    for arr in jax.live_arrays():   # 只读分片形状，不取 shard.data（那会另建数组、把自己也算进去）
        if arr.is_deleted() or dev not in arr.sharding.device_set:
            continue
        n += 1
        nbytes += int(np.prod(arr.sharding.shard_shape(arr.shape))) * arr.dtype.itemsize
    stats = dev.memory_stats() or {}
    rec = {"tag": tag, "live_arrays_bytes": nbytes, "live_arrays_n": n,
           **{k: stats[k] for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit") if k in stats}}
    if jax.process_index() == 0:
        print("[janus] memory " + json.dumps(rec), flush=True)
    return rec


def build_meta(a, config, mesh, golden_meta: dict, shard_devices: dict, batches_sha: str, overrides: list[str]) -> dict:
    import flax
    import jax
    import optax
    import orbax.checkpoint as ocp

    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:  # noqa: BLE001
            return None

    def pip_ver(*dists):
        from importlib import metadata
        for d in dists:
            try:
                return f"{d}=={metadata.version(d)}"
            except Exception:  # noqa: BLE001
                continue
        return None
    try:
        from jax.extend.backend import get_backend
        platform_version = get_backend().platform_version
    except Exception:  # noqa: BLE001
        platform_version = None
    c = config
    mesh_shape = dict(mesh.shape)
    layers, mtp = mt_common.STAGES[a.stage]
    lr_constant = c.lr_schedule_type == "wsd" and c.warmup_steps_fraction == 0 and c.wsd_decay_steps_fraction == 0
    return {
        "schema_version": 1, "run_name": Path(a.janus_run).name, "side": a.side, "framework": "maxtext",
        "model": f"hunyuan4-{a.stage}", "stage": a.stage,
        "code": [{"repo": "maxtext", "commit": a.maxtext_commit}, {"repo": "janus", "commit": a.janus_commit}],
        "image": a.image,
        "versions": {"jax": jax.__version__, "jaxlib": ver("jaxlib"), "libtpu": platform_version,
                     # pip 包版本号（platform_version 只有构建信息）；包名随来源不同，取装着的那个
                     "libtpu_pip": pip_ver("libtpu", "libtpu-nightly"), "flax": flax.__version__,
                     "optax": optax.__version__, "orbax_checkpoint": getattr(ocp, "__version__", None)},
        "hardware": {"accelerator": jax.devices()[0].device_kind, "count": jax.device_count(), "topology": a.topology,
                     "processes": jax.process_count()},
        "parallelism": {"dp": mesh_shape.get("data", 1), "fsdp": mesh_shape.get("fsdp", 1) * mesh_shape.get("fsdp_transpose", 1),
                        "tp": mesh_shape.get("tensor", 1), "ep": mesh_shape.get("expert", 1),
                        "cp": mesh_shape.get("context", 1), "pp": mesh_shape.get("stage", 1), "mesh": mesh_shape},
        "config": {"path": str(mt_common.HY4_YML), "sha256": sha_file(mt_common.HY4_YML),
                   "args": {k: v for k, v in vars(a).items() if k != "maxtext_args"},
                   "overrides": mt_common.forced_overrides() + mt_common.stage_overrides(a.stage) + overrides
                                + list(a.maxtext_args),
                   "yml_keys_not_effective": [list(map(str, x)) for x in mt_common.effective_mismatch(c)],
                   "gbs": c.global_batch_size_to_train_on, "seq": c.max_target_length, "lr": c.learning_rate,
                   "seed": c.init_weights_seed},
        "checkpoint": {"path": a.load, "manifest_sha256": sha_file(a.hf_manifest) if a.hf_manifest else None,
                       "manifest": a.hf_manifest, "hf_config_sha256": sha_file(a.hf_config)},
        "data": {"manifest": str(Path(a.golden) / "data" / "batches"), "manifest_sha256": batches_sha,
                 "seed": c.data_shuffle_seed,
                 "replay_of": {"run": golden_meta.get("run_name"), "manifest": golden_meta.get("data", {}).get("manifest"),
                               "manifest_sha256": golden_meta.get("data", {}).get("manifest_sha256")},
                 "device_of_shard": shard_devices,
                 "rows": "全局第 r·M+m 行 = GPU rank r 的 micro m；batch 维第 d 个分片 = GPU rank d"},
        "steps": {"first": a.first_step, "last": a.steps, "convention": "golden_step = step"},
        "determinism": {**jd.env_snapshot(), "remat_policy": c.remat_policy, "moe_routing_maps": c.moe_routing_maps,
                        "attention": c.attention, "matmul_precision": c.matmul_precision},
        "dtypes": {"compute": str(c.dtype), "weight": f"{c.weight_dtype} main params（计算时转 {c.dtype}）",
                   "optimizer_states": str(c.mu_dtype), "grad_reduce": str(c.grad_dtype)},
        "model_config": {"num_layers": c.num_decoder_layers, "first_dense_layers": c.first_num_dense_layers,
                         "mtp_num_layers": c.mtp_num_layers, "hidden_size": c.emb_dim, "num_attention_heads": c.num_query_heads,
                         "q_lora_rank": c.q_lora_rank, "kv_lora_rank": c.kv_lora_rank, "qk_nope_head_dim": c.qk_nope_head_dim,
                         "qk_rope_head_dim": c.qk_rope_head_dim, "v_head_dim": c.v_head_dim, "num_experts": c.num_experts,
                         "num_experts_per_tok": c.num_experts_per_tok, "use_indexer": bool(c.use_indexer),
                         "indexer_topk": c.indexer_topk, "indexer_share_interval": c.indexer_share_interval,
                         "routed_bias_update_rate": c.routed_bias_update_rate, "routed_scaling_factor": c.routed_scaling_factor,
                         "mtp_loss_scaling_factor": c.mtp_loss_scaling_factor, "stage_layers": layers, "stage_mtp": mtp},
        "optimizer_config": {"optimizer": c.opt_type, "lr": c.learning_rate, "weight_decay": c.adam_weight_decay,
                             "adam_beta1": c.adam_b1, "adam_beta2": c.adam_b2, "adam_eps": c.adam_eps,
                             "adam_eps_root": c.adam_eps_root, "clip_grad": c.gradient_clipping_threshold,
                             "adamw_mask": list(c.adamw_mask or []),
                             "trainable_parameters_mask": list(getattr(c, "trainable_parameters_mask", []) or []),
                             "indexer_loss_scaling_factor": c.indexer_loss_scaling_factor,
                             "main_params_dtype": str(c.weight_dtype), "exp_avg_dtype": str(c.mu_dtype)},
        "scheduler_config": {"lr_decay_style": "constant" if lr_constant else str(c.lr_schedule_type),
                             "lr_warmup_iters": int(c.learning_rate_schedule_steps * c.warmup_steps_fraction)},
        "maxtext_config_full": maxtext_full_config(c),
        "flags": {"at_start": FLAG_AT_START, "after_compile": None},
        "hooks": {"janus_dump": "janus.dump.maxtext + hf_view", "dump_steps": a.dump_steps, "full": a.full,
                  "capture": "state-only（梯度累积 > 1，加钩子的训练步不支持）"},
        "restarts": 0,
        "created_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds"),
    }


FLAG_AT_START: dict = {}


class Hy4Dump:
    """按 dump-format 写运行目录；参数类张量用 HF 名字（hf_view）。多进程时所有进程都要调（读设备是集合操作），只有进程 0 写盘。"""

    def __init__(self, out_dir, meta, view: HFView, dump_steps, full, full_steps, process_index, only=None):
        import re

        from janus.store import RunWriter
        self.out_dir, self.view = Path(out_dir), view
        self.dump_steps = set(dump_steps)
        self.full = full
        self.full_steps = set(full_steps) if full_steps else ({min(self.dump_steps)} if self.dump_steps else set())
        self.rank0 = process_index == 0
        self.only = re.compile("|".join(f"(?:{x})" for x in only.split(",") if x)) if only else None
        self.writer = RunWriter(self.out_dir, meta) if self.rank0 else None
        self._pending: dict[str, np.ndarray] = {}
        if self.rank0:
            (self.out_dir / "data").mkdir(parents=True, exist_ok=True)
            (self.out_dir / "params_sha").mkdir(parents=True, exist_ok=True)

    def _keep(self, h):
        return self.only is None or bool(self.only.search(h))

    def write_param_sha(self, flat, tag):
        """params_sha/rank0..7.<tag>.json：与 GPU golden 同一种写法——名字 hy4_map.gpu_name、FP32 主权重、
        第 r 份记 GPU rank r 本地的 FSDP2 分片（hy4_map.gpu_rank_pieces），这样能和 GPU 的清单逐项比。"""
        mans = [dict() for _ in range(8)]
        bufs = {}
        for h, arr in self.view.tensors(flat, jd.to_host, "param"):
            a = np.ascontiguousarray(arr, np.float32)
            gn = gpu_name(h)
            if self.view.is_buffer(h):      # 路由偏置：GPU 上是 buffer，不在它的参数清单里；单独记一份
                bufs[gn] = {"sha256": jd.sha256(a), "shape": list(a.shape), "dtype": "float32"}
                continue
            for r, piece in enumerate(gpu_rank_pieces(gn, a)):
                mans[r][gn] = {"sha256": jd.sha256(np.ascontiguousarray(piece)), "shape": list(a.shape), "dtype": "float32"}
        if self.rank0:
            for r in range(8):
                (self.out_dir / "params_sha" / f"rank{r}.{tag}.json").write_text(json.dumps(mans[r], indent=0, sort_keys=True))
            if bufs:
                (self.out_dir / "params_sha" / f"buffers.{tag}.json").write_text(json.dumps(bufs, indent=0, sort_keys=True))
        return mans[0]

    def write_state_sha(self, flat, tag):
        man = jd.state_manifest(flat)
        if self.rank0:
            d = self.out_dir / "state_sha"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{tag}.json").write_text(json.dumps(man, indent=0, sort_keys=True))
        return man

    def write_modules(self, flat):
        if not self.rank0:
            return
        with open(self.out_dir / "modules.txt", "w") as f:
            for k, v in sorted(flat.items()):
                if hasattr(v, "shape"):
                    f.write(f"param\t{k}\t{tuple(v.shape)}\t{v.dtype}\n")

    def record_batch(self, records):
        if not self.rank0:
            return
        for d, recs in records.items():
            with open(self.out_dir / "data" / f"rank{d}.jsonl", "a") as f:
                for rec in recs:
                    f.write(json.dumps(rec) + "\n")

    def capture(self, step, flat, when: str):
        """when = before / after。buffer（路由偏置）每个抓数步都抓；全量步另抓主权重（before 另写 BF16 参数）与 after 的两个动量。"""
        if step not in self.dump_steps:
            return
        full = self.full and step in self.full_steps
        for h, arr in self.view.tensors(flat, jd.to_host, "param", keep=lambda h: self.view.is_buffer(h) or (full and self._keep(h))):
            if self.view.is_buffer(h):
                self._pending[f"{'buffer' if when == 'before' else 'buffer_after'}.{gpu_name(h)}"] = arr
            elif when == "before":
                self._pending[f"param.{gpu_name(h)}"] = arr
            else:
                self._pending[f"main_param_after.{gpu_name(h)}"] = arr
        if full and when == "after":
            sflat = self._split_fused_glu(flat)
            for kind, prefix in (("mu", "exp_avg"), ("nu", "exp_avg_sq")):
                # Muon 参数只有动量（mu），没有 nu：它们的 nu 叶子是空数组（MaskedNode 之外的 0 元素占位），逐个跳过。
                # Muon 参数的 mu 记成 exp_avg.<名>（GPU 的 Muon 动量 momentum_buffer 不在 GPU 抓数里，比较时按分区区分）
                keep = lambda h, k=kind: (not self.view.is_buffer(h) and self._keep(h)  # noqa: E731
                                          and self._state_nonempty(sflat, h, k))
                for h, arr in self.view.tensors(sflat, jd.to_host, kind, keep=keep):
                    self._pending[f"{prefix}.{gpu_name(h)}"] = arr

    def _state_nonempty(self, flat, h, kind):
        for mt_key, _i in self.view.by_hf[h]:
            sk = self.view.state_key(flat, mt_key, kind)
            if sk is None or int(np.prod(flat[sk].shape)) == 0:
                return False
        return True

    def _split_fused_glu(self, flat):
        """muon_fuse_glu（MaxText optimizers.fuse_glu_for_muon）：一对 GLU（wi_0、wi_1）都走 Muon 时，动量存在 wi_0 的叶子上，
        是 concat([wi_0, wi_1], axis=-1)，wi_1 的叶子是末维为 0 的空数组。拆回两半（只改本地副本，不动训练状态），
        再按 HF 名字取，两半分别是 gate、up 的动量。"""
        out = dict(flat)
        for k, v in flat.items():
            if "/opt_state/" not in k or "/mu/" not in k or "/wi_0" not in k or not hasattr(v, "shape"):
                continue
            rest = k.split("/mu/", 1)[1]
            pk = "params/model/" + rest
            # wi_1 的动量：Muon 一支里是 MaskedNode（没有 shape），另一支里是末维为 0 的空数组；取有数组的那一个
            hits = [x for x in flat if x.endswith("/mu/" + rest.replace("wi_0", "wi_1")) and "/opt_state/" in x
                    and hasattr(flat[x], "shape")]
            if len(hits) != 1 or flat[hits[0]].shape[-1] != 0 or pk not in flat:
                continue
            k1 = hits[0]
            n0 = flat[pk].shape[-1]
            if v.shape[-1] != 2 * n0:
                raise ValueError(f"{k}：融合动量末维 {v.shape[-1]} ≠ 2 × {n0}")
            a = jd.to_host(v)
            out[k], out[k1] = a[..., :n0], a[..., n0:]
        return out

    def log_metrics(self, step, values):
        if self.rank0:
            self.writer.log_metrics(step, **values)

    def flush(self, step):
        import re
        if not self.rank0 or not self._pending:
            self._pending = {}
            return
        kinds = {"param": ("param", "before_update"), "main_param": ("param", "before_update"),
                 "main_param_after": ("param", "after_update"), "buffer": ("param", "before_update"),
                 "buffer_after": ("param", "after_update"), "exp_avg": ("opt_state", "after_update"),
                 "exp_avg_sq": ("opt_state", "after_update")}
        by_scope: dict[str, dict] = {}
        for name, arr in self._pending.items():
            head = name.split(".")[0]
            m = re.search(r"\.(layers|mtp_layers)\.(\d+)\.", name)
            scope = f"{head}.{m.group(1)}.{m.group(2)}" if m else f"{head}.other"
            by_scope.setdefault(scope, {})[name] = arr
        for scope, tensors in sorted(by_scope.items()):
            kind, point = kinds[scope.split(".")[0]]
            self.writer.write_tensors(step, scope, tensors, kind=kind, point=point)
        self._pending = {}

    def done(self):
        if self.rank0:
            self.writer.done()


# 与 GPU 一侧（Megatron 线 grad_sq.<组>）相同的梯度分组，按 HF 名字判，先到先得
def grad_group(h: str) -> str:
    if h.startswith("model.mtp_layers."):
        return "mtp"
    if h in ("model.embed_tokens.weight", "lm_head.weight"):
        return "embed_head"
    if ".mlp.experts." in h:
        return "experts"
    if ".hc_attn_layer." in h or ".hc_mlp_layer." in h or h.startswith("model.hc_head."):
        return "hc"
    if ".indexer." in h:
        return "indexer"
    return "rest"


def diag_step1(a, config, jit_model, state, params_shardings, mesh, gb, put, view, meta):
    """第 --first-step 步的梯度与索引器 KD（不更新参数）。梯度用 MaxText 训练里同一个梯度累积函数算，
    再按训练步里的同一除数（全局有效 token 数）归一，所以和训练第 1 步裁剪前的梯度是同一个量。"""
    import jax
    from flax import nnx
    from maxtext.trainers.pre_train import train
    from maxtext.utils import gradient_accumulation as ga

    s = a.first_step
    host_batch = replay.to_maxtext(gb.global_batch(s))
    fr_info, fr_ids, fi_ids = None, None, None
    if a.forced_routing or a.forced_indexer:
        fr_info = {}
        if a.forced_routing:
            fr_info["routing"], fr_ids = load_replay(a.forced_routing, host_batch["inputs"], "forced_routed_experts", config)
            host_batch["forced_routed_experts"] = fr_ids
        if a.forced_indexer:
            fr_info["indexer"], fi_ids = load_replay(a.forced_indexer, host_batch["inputs"], "forced_indexer_topk", config)
            host_batch["forced_indexer_topk"] = fi_ids
    batch = put(host_batch)
    if a.capture_choices:
        cap_info = capture_choices(a, config, jit_model, state, batch, mesh, host_batch["inputs"])
        if a.capture_only:
            if jax.process_index() == 0:
                d = Path(a.janus_run)
                (d / "diag_step1.json").write_text(json.dumps({"step": s, "captured_choices": cap_info, "code": meta.get("code")},
                                                             ensure_ascii=False, indent=1))
                (d / "DONE").write_text("capture\n")
            return 0

    pflat = mt_common.variable_paths(state)
    model_gdef = None
    if a.diag_free_opt:
        # 梯度函数只用到 .model：把模型单独拆出来，优化器状态的设备数组逐个释放（不复制模型变量）
        if fr_info is not None:
            raise SystemExit("--diag-free-opt 不和 --forced-routing / --forced-indexer 一起用")
        model_gdef, mstate = nnx.split(nnx.merge(jit_model, state).model)
        keep_ids = {id(x) for x in jax.tree_util.tree_leaves(mstate)}
        freed = 0
        for x in jax.tree_util.tree_leaves(state):
            if isinstance(x, jax.Array) and id(x) not in keep_ids and not x.is_deleted():
                freed += x.nbytes
                x.delete()
        state = mstate
        if jax.process_index() == 0:
            print(f"[janus] diag_free_opt: freed {freed} bytes (global) of optimizer state", flush=True)

    def model_of(st):
        return nnx.merge(model_gdef, st) if model_gdef is not None else nnx.merge(jit_model, st).model

    def fn(st, data):
        model = model_of(st)
        loss, aux, grads = ga.gradient_accumulation_loss_and_grad(
            train.loss_fn, config, model, None, params_shardings, data, None)
        return loss, aux, grads

    timing = None
    routing_log = None
    if getattr(config, "debug_routing_digest", False):
        from maxtext.utils import routing_debug
        routing_debug.LOG.clear()
    with jax.set_mesh(mesh), train.logical_axis_rules(config.logical_axis_rules):
        if a.diag_timing > 0:
            t0 = time.time()
            compiled = jax.jit(fn).lower(state, batch).compile()
            t_compile = time.time() - t0
            t0 = time.time()
            loss, aux, grads = jax.block_until_ready(compiled(state, batch))
            runs = [time.time() - t0]
            for _ in range(a.diag_timing):  # 纯函数（不更新参数），重复算同一个梯度只为计时
                t0 = time.time()
                jax.block_until_ready(compiled(state, batch))
                runs.append(time.time() - t0)
            ma = compiled.memory_analysis()
            timing = {
                "xla_flags": os.environ.get("XLA_FLAGS", ""),
                "compile_seconds": t_compile,
                "run_seconds": runs,
                "run_seconds_median_excl_first": float(np.median(runs[1:])),
                "memory_analysis": {k: int(getattr(ma, k)) for k in ("argument_size_in_bytes", "output_size_in_bytes",
                                                                     "temp_size_in_bytes", "generated_code_size_in_bytes")
                                    if hasattr(ma, k)} if ma is not None else None,
                "peak_bytes_in_use": {str(d.id): (d.memory_stats() or {}).get("peak_bytes_in_use")
                                      for d in jax.local_devices()},
            }
            if jax.process_index() == 0:
                print("[janus] diag timing " + json.dumps(timing), flush=True)
        else:
            loss, aux, grads = jax.jit(fn)(state, batch)
    if getattr(config, "debug_routing_digest", False):
        jax.block_until_ready(grads)
        jax.effects_barrier()
        # 本进程设备上的每次 top-k（前向一次；反向若重算路由再记一次）：数重算时选得不同的 (token, 专家) 槽位。
        # --diag-timing 会把同一个梯度重复算几遍，记录里的重复不是重算，不数
        routing_log = ({"process": jax.process_index(), "local_devices": jax.local_device_count(),
                        **routing_debug.count_reselections()} if a.diag_timing == 0 else {"skipped": "diag_timing > 0"})
    gflat = {"params/model/" + "/".join(str(p) for p in path): (v.get_value() if hasattr(v, "get_value") else v)
             for path, v in nnx.to_flat_state(grads)}
    shape_report = {}
    for k, g in gflat.items():
        p = pflat.get(k)
        if p is None or tuple(p.shape) != tuple(g.shape):
            shape_report[k] = {"grad": list(g.shape), "param": None if p is None else list(p.shape)}
    if jax.process_index() == 0:
        print(f"[janus] diag grads: {len(gflat)} leaves, shape mismatches {len(shape_report)}: "
              + json.dumps(dict(list(shape_report.items())[:8])), flush=True)
    sq: dict[str, float] = {}
    per_tensor: dict[str, dict] = {}
    arrays = {}
    import re as _re
    diag_keep = _re.compile("|".join(f"(?:{x})" for x in a.full_only.split(","))) if a.full_only else None
    for h, arr in view.tensors(gflat, jd.to_host, "param", keep=lambda h: not view.is_buffer(h)):
        x = np.asarray(arr, np.float64)
        v = float(np.dot(x.ravel(), x.ravel()))
        sq[grad_group(h)] = sq.get(grad_group(h), 0.0) + v
        per_tensor[h] = {"sq": v, "shape": list(x.shape)}
        if diag_keep is None or diag_keep.search(h):  # --full-only：只存匹配的张量（平方和与 grad_norm 仍按全部张量算）
            arrays[h] = np.ascontiguousarray(arr, np.float32)  # safetensors 按内存写，转置视图要先连续化
    debug = {}
    kd_micro0 = {}
    if getattr(config, "debug_sow_indexer", False):
        # 前向一次（不求导），只取 microbatch 0：与梯度累积同一种切法（第 m 个 microbatch = 全局行 m::GA），
        # 第 0 行 = 设备 0 的 microbatch 0 = GPU rank 0 的 micro 0。各层 sow 的是这一行的 KD 教师 / 学生输入
        ga_n = int(config.gradient_accumulation_steps)

        def fwd(st, data):
            model = model_of(st)
            d0 = jax.tree_util.tree_map(lambda x: x[0::ga_n], data)
            _, aux0 = train.loss_fn(model, config, d0, None, None, is_train=True)
            return aux0["intermediate_outputs"]

        with jax.set_mesh(mesh), train.logical_axis_rules(config.logical_axis_rules):
            inter0 = jax.jit(fwd)(state, batch)

        def grab(path, x):
            keys = [str(getattr(k, "key", getattr(k, "idx", k))) for k in path]
            if any(k.startswith("debug_kd_") for k in keys):
                debug["/".join(keys)] = np.ascontiguousarray(np.asarray(jd.to_host(x), np.float32))
            elif any("indexer_loss" in k for k in keys):  # microbatch 0（8 行）的逐层 KD，已乘 indexer_loss_scaling_factor
                kd_micro0["/".join(keys)] = np.asarray(jd.to_host(x), np.float64).ravel().tolist()

        jax.tree_util.tree_map_with_path(grab, inter0)
    if fr_info is not None:
        try:
            fr_info["audit"] = replay_audit(config, jit_model, state, batch, mesh, fr_ids, fi_ids)
        except Exception as e:  # noqa: BLE001 — 审计出错不丢已算好的梯度，错误写进结果
            fr_info["audit"] = {"error": repr(e), "summary": {"error": repr(e)}}
        if jax.process_index() == 0:
            print("[janus] replay audit " + json.dumps(fr_info["audit"]["summary"], ensure_ascii=False), flush=True)
    kd = {}

    def collect(path, x):
        keys = [str(getattr(k, "key", getattr(k, "idx", k))) for k in path]
        if any("indexer_loss" in k for k in keys):
            kd["/".join(keys)] = np.asarray(jd.to_host(x), np.float64).ravel().tolist()

    jax.tree_util.tree_map_with_path(collect, aux.get("intermediate_outputs", {}))
    qk = {}
    if getattr(config, "use_qk_clip", False) and aux.get("intermediate_outputs"):
        # 第 1 步前向的逐层逐头最大 logit（QK-Clip 统计；qk_clip_max_logit_exclude_sink 决定含不含 sink），列名同 GPU
        from maxtext.utils import qk_clip_utils

        with jax.set_mesh(mesh):
            per_head = qk_clip_utils.per_head_stats(
                nnx.state(model_of(state), nnx.Param).to_pure_dict(), aux["intermediate_outputs"], config)
        per_head = {k: np.asarray(jd.to_host(v)) for k, v in per_head.items()}  # 多主机：先聚到每个进程
        qk = {k: v for k, v in qk_clip_columns(per_head, view).items() if k.startswith("qk.max_logit.")}
    out = {"step": s, "forced_routing": fr_info, "qk_max_logit": qk, "routing_reselection": routing_log, "loss": float(jd.to_host(loss)), "grad_norm": float(np.sqrt(sum(sq.values()))),
           "grad_sq": dict(sorted(sq.items())), "per_tensor": per_tensor,
           "indexer_loss_raw": kd,
           "indexer_loss_micro0": kd_micro0,
           "note": "indexer_loss_raw：各产出层 aux 里的值（已乘 indexer_loss_scaling_factor），梯度累积的各 microbatch 已相加，"
                   "要除以 gradient_accumulation_steps；GPU 的逐层值再乘 1 / 产出层数",
           "gradient_accumulation_steps": int(config.gradient_accumulation_steps), "code": meta.get("code")}
    if timing is not None:
        out["timing"] = timing  # peak_bytes_in_use 只含进程 0 本机的设备
    if jax.process_index() == 0:
        d = Path(a.janus_run)
        d.mkdir(parents=True, exist_ok=True)
        (d / "diag_step1.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        from safetensors.numpy import save_file
        save_file(arrays, str(d / "diag_grads.safetensors"))
        if debug:
            save_file(debug, str(d / "diag_kd_inputs.safetensors"))
        (d / "DONE").write_text("diag\n")
        print("[janus] diag_step1 " + json.dumps({k: out[k] for k in ("loss", "grad_norm", "grad_sq")}), flush=True)
    return 0


SNAPSHOT_ONLY = {"gpu-fp32": r"^lm_head\.weight$|\.indexer\.k_norm\."}

# --eval-variants=@<名字>：预设（运行命令进 Pod 时经过 shell，正则里的 | ( ) ^ 不能直接写在命令行上）
EVAL_PRESETS = {
    "weight-side": "none=;all=.;head=^lm_head\\.;embed=embed_tokens;norm=norm\\.weight$|norm\\.bias$;"
                   "attn=self_attn\\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj|linear_gate)\\.;"
                   "indexer=self_attn\\.indexer\\.;dense=mlp\\.(gate_proj|up_proj|down_proj)\\.;shared=mlp\\.shared_experts\\.;"
                   "experts=mlp\\.experts\\.;router=mlp\\.gate\\.weight$;mtp=^model\\.mtp_layers\\.;mtp_proj=eh_proj",
}


def eval_variants(a, config, jit_model, state, batch, mesh, view, step, meta):
    """--eval-variants-at：同一份权重下，选中的参数舍成 BF16（RNE）后的前向 loss。用来看 TPU 用 FP32 主权重直接参与计算
    （而 GPU 用的是 BF16 模型权重）在某一步造成多大的 loss 差。只作诊断。"""
    import re

    import jax
    import jax.numpy as jnp
    from flax import nnx
    from maxtext.trainers.pre_train import train

    model_gdef, mstate = nnx.split(nnx.merge(jit_model, state).model)
    keep_ids = {id(x) for x in jax.tree_util.tree_leaves(mstate)}
    for x in jax.tree_util.tree_leaves(state):   # 优化器状态用不到，先放掉
        if isinstance(x, jax.Array) and id(x) not in keep_ids and not x.is_deleted():
            x.delete()
    variables = {}
    for path, v in nnx.to_flat_state(mstate):
        if isinstance(v, nnx.RngState):
            continue
        from flax.nnx import variablelib
        col = variablelib.variable_name_from_type(type(v), allow_register=True)
        variables[f"{col}/model/" + "/".join(str(p) for p in path)] = v

    ga_n = int(config.gradient_accumulation_steps)

    def fwd(st, data):
        # 与梯度累积同一种切法：第 m 个 microbatch = 全局行 m::GA；xent 与 token 数相加，MTP / KD 取各 microbatch 的平均
        model = nnx.merge(model_gdef, st)
        acc = {"loss": 0.0, "xent_sum": 0.0, "total_weights": 0.0, "mtp_loss": 0.0, "indexer_loss": 0.0}
        for m in range(ga_n):
            dm = jax.tree_util.tree_map(lambda x, m=m: x[m::ga_n], data)
            loss, aux = train.loss_fn(model, config, dm, None, None, is_train=True)
            acc["loss"] += loss
            acc["xent_sum"] += aux["xent_sum"]
            acc["total_weights"] += aux["total_weights"]
            acc["mtp_loss"] += aux.get("mtp_loss", jnp.zeros(())) / ga_n
            acc["indexer_loss"] += aux.get("indexer_loss", jnp.zeros(())) / ga_n
        return acc

    with jax.set_mesh(mesh), train.logical_axis_rules(config.logical_axis_rules):
        f = jax.jit(fwd)
        out = {}
        specs = EVAL_PRESETS[a.eval_variants[1:]] if a.eval_variants.startswith("@") else a.eval_variants
        for spec in [x for x in specs.split(";") if x]:
            name, _, pat = spec.partition("=")
            rx = re.compile(pat) if pat else None
            keys, names = set(), []
            if rx is not None:
                for h, srcs in view.by_hf.items():
                    if view.stored_fp32(h) or view.is_buffer(h) or not rx.search(h):
                        continue
                    names.append(h)
                    for mt_key, _i in srcs:
                        keys.add(view.state_key({}, mt_key, "param"))
            missing = sorted(k for k in keys if k not in variables)
            if missing:
                raise SystemExit(f"--eval-variants {name}：状态里没有这些叶子 {missing[:4]}")
            orig = {}
            for k in sorted(keys):
                v = variables[k]
                orig[k] = v.get_value()
                v.set_value(orig[k].astype(jnp.bfloat16).astype(orig[k].dtype))
            r = jax.device_get(f(mstate, batch))
            for k, x in orig.items():
                variables[k].set_value(x)
            tw = float(r["total_weights"])
            out[name] = {"pattern": pat, "n_hf_tensors": len(names), "n_leaves": len(keys),
                         "lm_loss": float(r["xent_sum"]) / tw, "total_weights": tw,
                         "mtp_loss": float(r["mtp_loss"]), "indexer_loss": float(r["indexer_loss"]), "loss": float(r["loss"])}
            if jax.process_index() == 0:
                print("[janus] eval_variant " + json.dumps({name: out[name]}, ensure_ascii=False), flush=True)
    if jax.process_index() == 0:
        d = Path(a.janus_run)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"eval_variants_step{step}.json").write_text(json.dumps(
            {"step": step, "note": "第 step 步训练之前的权重；选中的参数舍成 BF16 后只做前向（不更新）", "variants": out,
             "code": meta.get("code")}, ensure_ascii=False, indent=1))
        (d / "DONE").write_text("eval_variants\n")
    return 0


def hf_snapshot(a, view, flat, step, write, process_index):
    """当前模型权重 → HF 检查点的名字、布局与 dtype（HF 存 BF16 的取 BF16 舍入，存 FP32 的原样），按层分文件写到
    <运行目录>/hf_snapshot/step_<N>/，另写 model.safetensors.index.json 与 tensor_sha256.json（同 HF 目录的写法）。
    write=False 时只算每个张量的字节 sha256 对 --hf-manifest。所有进程都要调（读设备是集合操作），只有进程 0 写盘。"""
    import re as _re

    import ml_dtypes

    want = json.loads(Path(a.hf_manifest).read_text()) if (a.hf_manifest and not write) else None
    out_dir = Path(a.janus_run) / "hf_snapshot" / f"step_{step}"
    groups: dict[str, dict[str, np.ndarray]] = {}
    shas: dict[str, str] = {}
    bad, n = [], 0
    t0 = time.time()

    def flush(fname):
        if write and process_index == 0 and groups.get(fname):
            from safetensors.numpy import save_file
            save_file(groups[fname], str(out_dir / fname))
        groups.pop(fname, None)

    if write and process_index == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    only = _re.compile(SNAPSHOT_ONLY[a.hf_snapshot_only]) if a.hf_snapshot_only else None
    cur = None
    for h, arr in view.tensors(flat, jd.to_host, "param", keep=(lambda h: bool(only.search(h))) if only else None):
        if only is not None:
            x = np.ascontiguousarray(arr, np.float32)
        else:
            x = np.ascontiguousarray(arr if view.stored_fp32(h) else arr.astype(ml_dtypes.bfloat16))
        shas[h] = hashlib.sha256(x.tobytes()).hexdigest()
        n += 1
        if want is not None and want.get(h) != shas[h]:
            bad.append(h)
        if write:
            m = _re.match(r"model\.(layers|mtp_layers)\.(\d+)\.", h)
            fname = (f"model-layer{m.group(2)}.safetensors" if m.group(1) == "layers" else "model-mtp.safetensors") if m \
                else "model-other.safetensors"
            if cur is not None and fname != cur and cur != "model-other.safetensors":
                flush(cur)   # 按 HF 名排序时每层的张量连在一起；model-other（lm_head 在最前、embed / norm 在后）留到最后写
            cur = fname
            if process_index == 0:
                groups.setdefault(fname, {})[h] = x
    if write:
        for f in list(groups):
            flush(f)
    info = {"step": step, "tensors": n, "seconds": round(time.time() - t0, 1)}
    if want is not None:
        info["verify_mismatch"] = len(bad)
        info["verify_mismatch_first"] = bad[:8]
    if process_index == 0:
        if write:
            wm = {}
            for f in sorted(out_dir.glob("model-*.safetensors")):
                from janus.store import read_safetensors_header
                header, _ = read_safetensors_header(f)
                for k in header:
                    if k != "__metadata__":
                        wm[k] = f.name
            (out_dir / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"step": step}, "weight_map": wm},
                                                                            indent=1, sort_keys=True))
            (out_dir / "tensor_sha256.json").write_text(json.dumps(shas, indent=1, sort_keys=True))
        print("[janus] hf_snapshot " + json.dumps(info, ensure_ascii=False), flush=True)
    return info


def load_replay(path: str, inputs: np.ndarray, key: str, config) -> tuple[dict, np.ndarray]:
    """build_forced_routing.py 的输出 → 本侧 batch 行序的一个回放数组（key = forced_routed_experts 或 forced_indexer_topk）。
    每一行按 token 逐个相等对上文件里的行（要正好一行）；形状对不上模型直接报错。"""
    from op_replay_mini import fetch

    from janus.store import read_safetensors_header, read_safetensors_tensor

    local = Path(fetch(path))
    header, _ = read_safetensors_header(local)
    if key not in header:
        raise SystemExit(f"{path} 里没有 {key}")
    arr = read_safetensors_tensor(local, key)[0]
    toks = read_safetensors_tensor(local, "tokens")[0]
    meta_info = json.loads((header.get("__metadata__") or {}).get("info", "{}"))
    want = int(config.num_decoder_layers) + int(getattr(config, "mtp_num_layers", 0))
    k = int(config.num_experts_per_tok) if key == "forced_routed_experts" else int(config.indexer_topk)
    if arr.shape[2] != want or arr.shape[3] != k or toks.shape[1] != inputs.shape[1]:
        raise SystemExit(f"{key} 形状 {arr.shape} 与模型（层槽位 {want}、k {k}、序列 {inputs.shape[1]}）不符")
    rows = []
    for i in range(inputs.shape[0]):
        m = [j for j in range(toks.shape[0]) if np.array_equal(toks[j], inputs[i])]
        if len(m) != 1:
            raise SystemExit(f"{key}：本侧第 {i} 行在文件里对上 {len(m)} 行（要正好 1 行）")
        rows.append(m[0])
    variant = (
        "只灌专家下标，权重照算（加偏置前的分数取下标处的值，归一后乘 routed_scaling_factor）"
        if key == "forced_routed_experts"
        else "索引器照算分数与 KD，注意力与 KD 的掩码用灌入的 top-k 位置（自带索引器的层读自己的槽位，共享层沿用）"
    )
    info = {"source": path, "key": key, "sha256": hashlib.sha256(local.read_bytes()).hexdigest(), "file_info": meta_info,
            "row_of_file_for_batch_row": rows, "variant": variant}
    out = arr[rows]
    return info, np.ascontiguousarray(out.astype(np.int32) if key == "forced_routed_experts" else out)


def _slot_of(config, path: str, i: int, indexer: bool) -> int:
    """sow 路径（与扫描 / sow 元组摊平后的第 i 个）→ 层槽位（HF 层号；MTP 排在主干层后）。"""
    n_main = int(config.num_decoder_layers)
    first = int(config.first_num_dense_layers)
    interval = int(getattr(config, "indexer_share_interval", 1))
    if "mtp" in path:
        return n_main + int(re.search(r"mtp_layer_(\d+)", path).group(1)) - 1
    if "dense_layers" in path:
        return i
    if "moe_layers_remainder" in path:
        return first + ((n_main - first) // interval) * interval + int(re.search(r"/layers_(\d+)/", path).group(1))
    m = re.search(r"/layers_(\d+)/", path)
    if m and interval > 1:  # 扫描的 IndexCache 块：第 i 块的第 j 层（索引器只在 j = 0）
        return first + i * interval + int(m.group(1))
    return first + i


def capture_choices(a, config, jit_model, state, batch, mesh, inputs: np.ndarray) -> dict:
    """debug_capture_choices：按梯度累积的切法逐个 microbatch 前向（不求导），收各路由 MoE 自选的下标与加偏置后的分数、
    各自带索引器的层的 top-k，拼回全局批，按 GPU 的行序（R = m·rows_per_micro + r，全局行 i = r·GA + m）写文件。"""
    import jax
    from flax import nnx
    from maxtext.trainers.pre_train import train

    ga_n = int(config.gradient_accumulation_steps)
    n_rows, S = inputs.shape
    per = n_rows // ga_n
    n_slots = int(config.num_decoder_layers) + int(getattr(config, "mtp_num_layers", 0))
    k, E, K = int(config.num_experts_per_tok), int(config.num_experts), int(config.indexer_topk)
    first = int(config.first_num_dense_layers)
    ids = np.full((n_rows, S, n_slots, k), -1, np.int32)
    scores = np.zeros((n_rows, S, n_slots, E), np.float32)
    topk = np.full((n_rows, S, n_slots, K), -1, np.int16)
    have_idx = set()

    def fwd(st, data, m):
        model = nnx.merge(jit_model, st).model
        d0 = jax.tree_util.tree_map(lambda x: x[m::ga_n], data)
        _, aux0 = train.loss_fn(model, config, d0, None, None, is_train=True)
        return aux0["intermediate_outputs"]

    for m in range(ga_n):
        with jax.set_mesh(mesh), train.logical_axis_rules(config.logical_axis_rules):
            inter = jax.jit(fwd, static_argnums=2)(state, batch, m)
        got: dict[str, dict] = {}

        def grab(path, x, got=got):
            keys = [str(getattr(kk, "key", getattr(kk, "idx", kk))) for kk in path]
            for n in ("captured_routed_experts", "captured_router_scores", "captured_indexer_topk"):
                if n in keys:
                    got.setdefault(n, {})["/".join(keys)] = np.asarray(jd.to_host(x))

        jax.tree_util.tree_map_with_path(grab, inter)
        rows = np.arange(per) * ga_n + m  # 这个 microbatch 在全局批里的行
        for n, dst in (("captured_routed_experts", ids), ("captured_router_scores", scores), ("captured_indexer_topk", topk)):
            for path, v in got.get(n, {}).items():
                v = v.reshape((-1,) + v.shape[-3:])
                for i in range(v.shape[0]):
                    slot = _slot_of(config, path, i, n == "captured_indexer_topk")
                    dst[rows, :, slot] = v[i].astype(dst.dtype)
                    if n == "captured_indexer_topk":
                        have_idx.add(slot)
    # IndexCache 的共享层沿用块内第一层的 top-k（与 GPU 的抓数相同：共享层存的就是第一层的选择）
    for slot in range(first, int(config.num_decoder_layers)):
        if slot not in have_idx:
            prev = max(x for x in have_idx if x < slot)
            topk[:, :, slot] = topk[:, :, prev]
    order = np.array([(R % per) * ga_n + R // per for R in range(n_rows)])  # GPU 第 R 行 = 本侧全局第 order[R] 行
    out = {"forced_routed_experts": ids[order], "router_scores": scores[order], "forced_indexer_topk": topk[order],
           "tokens": inputs[order].astype(np.int32)}
    info = {"rows": n_rows, "row_order": "GPU 的 R = m·rows_per_micro + r（全局行 i = r·GA + m）", "top_k": k,
            "indexer_topk": K, "slots": n_slots, "indexer_layers_captured": sorted(have_idx),
            "routed_layers_captured": sorted({s for s in range(n_slots) if (ids[:, :, s] >= 0).any()}),
            "order_within_token": "本侧 top_k 的顺序（加偏置后的分数降序）", "step": a.first_step}
    if jax.process_index() == 0:
        from janus.store import write_safetensors

        f = Path(a.janus_run) / "choices_captured.safetensors"
        f.parent.mkdir(parents=True, exist_ok=True)
        write_safetensors(f, out, metadata={"info": json.dumps(info, ensure_ascii=False)})
        info["file"] = f.name
        info["sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
        print("[janus] captured choices " + json.dumps(info, ensure_ascii=False), flush=True)
    return info


def replay_audit(config, jit_model, state, batch, mesh, routing: np.ndarray | None, indexer: np.ndarray | None) -> dict:
    """前向一次（不求导，只取 microbatch 0：全局行 0::GA），读 MaxText 回放开关 sow 的审计量：
    路由（M45）：各层实际用的专家下标对文件逐位比；本侧自选同一组专家的 token 比例。
    索引器（M46）：各自带索引器的层，每个 query 选中位置之和对文件逐个比；本侧自选同一组位置的 query 比例、
    灌入位置里本侧也会选的比例。"""
    import jax
    from flax import nnx
    from maxtext.trainers.pre_train import train

    ga_n = int(config.gradient_accumulation_steps)

    def fwd(st, data):
        model = nnx.merge(jit_model, st).model
        d0 = jax.tree_util.tree_map(lambda x: x[0::ga_n], data)
        _, aux0 = train.loss_fn(model, config, d0, None, None, is_train=True)
        return aux0["intermediate_outputs"]

    with jax.set_mesh(mesh), train.logical_axis_rules(config.logical_axis_rules):
        inter = jax.jit(fwd)(state, batch)
    got: dict[str, dict] = {}
    names = ("forced_routing_used", "forced_routing_natural_agree", "forced_indexer_index_sum",
             "forced_indexer_natural_agree", "forced_indexer_natural_overlap")

    def grab(path, x):
        keys = [str(getattr(k, "key", getattr(k, "idx", k))) for k in path]
        for n in names:
            if n in keys:
                got.setdefault(n, {})["/".join(keys)] = np.asarray(jd.to_host(x))

    jax.tree_util.tree_map_with_path(grab, inter)
    out: dict = {"rows": "microbatch 0（全局行 0::GA）"}
    n_slots = int(config.num_decoder_layers) + int(getattr(config, "mtp_num_layers", 0))

    def scalars(name, indexer_):
        res = {}
        for path, v in got.get(name, {}).items():
            for i, x in enumerate(np.asarray(v, np.float64).ravel()):
                res[str(_slot_of(config, path, i, indexer_))] = float(x)
        return dict(sorted(res.items(), key=lambda kv: int(kv[0])))

    if routing is not None:
        want = routing[0::ga_n]
        per = {}
        for path, v in got.get("forced_routing_used", {}).items():
            v = v.reshape((-1,) + v.shape[-3:])
            for i in range(v.shape[0]):
                slot = _slot_of(config, path, i, False)
                per[str(slot)] = {"path": path, "mismatch": int(np.sum(v[i] != want[..., slot, :])), "n": int(v[i].size)}
        expect = [L for L in range(n_slots) if L >= int(config.first_num_dense_layers)]
        out["routing"] = {"per_layer_slot": per, "natural_agree_frac": scalars("forced_routing_natural_agree", False),
                          "missing_layers": [L for L in expect if str(L) not in per],
                          "mismatch_total": sum(r["mismatch"] for r in per.values())}
    if indexer is not None:
        want = indexer[0::ga_n].astype(np.int64)
        per = {}
        for path, v in got.get("forced_indexer_index_sum", {}).items():
            v = v.reshape((-1,) + v.shape[-2:])
            for i in range(v.shape[0]):
                slot = _slot_of(config, path, i, True)
                w = want[:, :, slot]
                ref = np.where(w >= 0, w, 0).sum(-1)
                per[str(slot)] = {"path": path, "query_mismatch": int(np.sum(v[i] != ref)), "n": int(v[i].size)}
        out["indexer"] = {"per_layer_slot": per,
                          "natural_agree_frac": scalars("forced_indexer_natural_agree", True),
                          "natural_overlap_frac": scalars("forced_indexer_natural_overlap", True),
                          "query_mismatch_total": sum(r["query_mismatch"] for r in per.values())}
    out["summary"] = {
        k: {x: out[k][x] for x in out[k] if x != "per_layer_slot"} for k in ("routing", "indexer") if k in out
    }
    return out


def qk_clip_columns(per_head: dict, view) -> dict:
    """MaxText 的 qk_clip_per_head（{模块路径: [2, (层,) 头]}）→ 与 GPU 同名的列：qk.max_logit.<层>、qk.eta.<层>（每头一个）、
    qk.clipped.<层>（缩放 < 1 的头数）与 qk.clipped_total。层名按 HF：主干 0…N−1，MTP 记 mtp<k>。
    模块路径与层号来自 HFView 的映射（q_b_proj 的 MaxText 来源）。"""
    import re

    import jax
    host = {k: np.asarray(jax.device_get(v), np.float64) for k, v in per_head.items()}
    out, total = {}, 0
    for h, srcs in view.by_hf.items():
        m = re.fullmatch(r"model\.(layers|mtp_layers)\.(\d+)\.self_attn\.q_b_proj\.weight", h)
        if not m:
            continue
        mt_key, i = srcs[0]
        mod = mt_key.split("-", 1)[1].rsplit("-wq_b-kernel", 1)[0].replace("-", "/")
        if mod not in host:
            raise SystemExit(f"qk_clip_per_head 里没有 {mod}（{h}）：有 {sorted(host)}")
        a = host[mod]
        a = a[:, 0 if i is None else i] if a.ndim == 3 else a
        lab = m.group(2) if m.group(1) == "layers" else f"mtp{m.group(2)}"
        out[f"qk.max_logit.{lab}"] = [float(x) for x in a[0]]
        out[f"qk.eta.{lab}"] = [float(x) for x in a[1]]
        out[f"qk.clipped.{lab}"] = float((a[1] < 1).sum())
        total += int((a[1] < 1).sum())
    out["qk.clipped_total"] = float(total)
    return out


def _strip_dump_prefix(P: dict) -> dict:
    """janus 抓数导出的文件名前缀（main_param_after. / main_param. / param.）去掉，剩 HF 名字。"""
    out = {}
    for k, v in P.items():
        for pre in ("main_param_after.", "main_param.", "param."):
            if k.startswith(pre):
                k = k[len(pre):]
                break
        out[k] = v
    return out


_OVERRIDE_BIAS: dict = {}   # --param-override 文件里顺带读到的路由偏置（HF 名字），给 --bias-override hf:param-override 用


def apply_param_override(a, state) -> dict:
    """--param-override：HF 名字的 FP32 张量 → MaxText 叶子（与 convert_hf_to_maxtext 同一段变换，optim_replay.hf_to_mt），
    按原叶子的 sharding 放回训练状态（多主机也行）。与 op_replay_mini.override_params 同一做法，区别：作用在 TrainState
    （参数路径 model/…），并且严格——映射引用到的 HF 张量缺一个、或路由专家不全（形状不是完整的 [E, …]），都直接报错。"""
    import re

    import jax
    import op_replay_mini
    from flax import nnx
    from make_tiny_hf import hf_shapes
    from op_replay_mini import load_st
    from optim_replay import hf_to_mt

    P = {}
    for f in a.param_override.split(","):   # 逗号分隔的多个文件按顺序合并，后面的覆盖前面的（HF 快照分层存、FP32 覆盖另一个文件）
        P.update(_strip_dump_prefix(load_st(f)))
        if f.startswith("gs://") and f in op_replay_mini._FETCHED:   # 下载到 Pod 本地盘的副本读完就删（D3 快照 67 GB，盘只有 95 GB）
            os.remove(op_replay_mini._FETCHED.pop(f))
    _OVERRIDE_BIAS.update({k: np.asarray(v, np.float32) for k, v in P.items() if k.endswith("e_score_correction_bias")})
    keep = re.compile(a.override_only) if a.override_only else None
    extra = ["hardware=cpu", "skip_jax_distributed_system=True", "run_name=override", "attention=dot_product"]
    if a.cpu_small:
        extra += CPU_SMALL[:2] + [x for x in CPU_SMALL[2:] if x.startswith(("model_name", "checkpoint_rope"))]
    import tempfile

    import convert_hf_to_maxtext as C
    d = Path(tempfile.mkdtemp())
    (d / "config.json").write_text(Path(a.hf_config).read_text())
    # 参数量核对只对真实维度做（CPU 冒烟的 hy4-tiny 维度不是 STAGE_PARAMS 里的数）
    cfg_c, abstract, mapping, hooks, hf_cfg, slash_of = C.build(d, a.stage, extra, strict_params=not a.cpu_small)
    shapes = hf_shapes(hf_cfg, *mt_common.STAGES[a.stage])

    def names(m):
        return [m] if isinstance(m, str) else [x for x in (m if isinstance(m, list) else [m]) if isinstance(x, str)]

    need = {k for k in abstract if slash_of[k].startswith("params/")
            and (keep is None or any(keep.search(x) for x in names(mapping[k])))}
    want = sorted({x for k in need for x in names(mapping[k])})
    missing = [x for x in want if x not in P]
    short = [f"{x} {tuple(P[x].shape)} ≠ {tuple(shapes[x])}" for x in want if x in P and tuple(P[x].shape) != tuple(shapes[x])]
    if missing or short:
        raise SystemExit(f"--param-override 不全：缺 {len(missing)} 个 {missing[:6]}；形状不对 {len(short)} 个 {short[:6]}")
    vals = hf_to_mt(cfg_c, {k: abstract[k] for k in need}, mapping, hooks, slash_of, P, shapes)
    done = []
    for path, v in nnx.to_flat_state(state):
        if len(path) < 2 or str(path[0]) != "model" or not isinstance(v, nnx.Param):
            continue
        key = "params/" + "/".join(str(x) for x in path[1:])
        if key not in vals:
            continue
        old = v.get_value()
        x = np.ascontiguousarray(vals[key].reshape(old.shape)).astype(old.dtype)
        v.set_value(jax.make_array_from_callback(x.shape, old.sharding, lambda idx, x=x: x[idx]))
        done.append(key)
    if sorted(done) != sorted(vals):
        raise SystemExit(f"--param-override：映射出 {len(vals)} 个叶子，训练状态里只找到 {len(done)} 个："
                         f"{sorted(set(vals) - set(done))[:6]}")
    return {"file": a.param_override, "only": a.override_only, "n_hf_tensors": len(want), "n_leaves": len(done),
            "sha256_of_names": hashlib.sha256("\n".join(want).encode()).hexdigest()}


def _gpu_bias_to_hf(name: str) -> str | None:
    """GPU（Megatron）路由偏置 buffer 的名字 → HF 名字。"""
    import re
    m = re.fullmatch(r"(?:buffer\.)?decoder\.layers\.(\d+)\.mlp\.router\.expert_bias", name)
    if m:
        return f"model.layers.{m.group(1)}.mlp.gate.e_score_correction_bias"
    m = re.fullmatch(r"(?:buffer\.)?mtp\.layers\.(\d+)\.mtp_model_layer\.mlp\.router\.expert_bias", name)
    if m:
        return f"model.mtp_layers.{m.group(1)}.mlp.gate.e_score_correction_bias"
    return None


def apply_bias_override(a, state, view) -> dict:
    """--bias-override：GPU 全量抓数 step 目录里 buffer.*.safetensors 的 expert_bias（该步更新前）换进 MoEBiasVar。"""
    import subprocess

    from op_replay_mini import load_st

    src = a.bias_override.rstrip("/")
    if src == "hf:param-override":   # 用 --param-override 那几个文件里的路由偏置（不再下载一遍）
        return _set_bias(state, view, dict(_OVERRIDE_BIAS), src)
    if src.startswith("hf:"):   # HF 名字的 safetensors（逗号分隔，如 HF 快照的各层文件）里的 e_score_correction_bias
        G = {}
        for f in src[3:].split(","):
            from op_replay_mini import fetch

            from janus.store import read_safetensors_header
            names = [k for k in read_safetensors_header(Path(fetch(f)))[0] if k.endswith("e_score_correction_bias")]
            if names:
                G.update({k: np.asarray(v, np.float32) for k, v in load_st(f, names=set(names)).items()})
        return _set_bias(state, view, G, src)
    if src.startswith("gs://"):
        ls = subprocess.run(["gcloud", "storage", "ls", f"{src}/buffer.*.safetensors"], capture_output=True, text=True,
                            check=True).stdout.split()
    else:
        ls = sorted(str(x) for x in Path(src).glob("buffer.*.safetensors"))
    G = {}
    for f in ls:
        if "/buffer_after." in f:
            continue
        for k, v in load_st(f).items():
            h = _gpu_bias_to_hf(k)
            if h is not None:
                G[h] = np.asarray(v, np.float32)
    return _set_bias(state, view, G, src)


def _set_bias(state, view, G, src) -> dict:
    import jax
    flat = mt_common.variable_paths(state)
    want = [h for h in view.by_hf if view.is_buffer(h)]
    missing = [h for h in want if h not in G]
    if missing:
        raise SystemExit(f"--bias-override：{src} 里缺这些路由偏置：{missing}")
    by_key: dict[str, list] = {}
    for h in want:
        for mt_key, i in view.by_hf[h]:
            by_key.setdefault(view.state_key(flat, mt_key, "param"), []).append((i, h))
    from flax import nnx
    n = 0
    for path, v in nnx.to_flat_state(state):
        col = type(v).__name__
        if col != "MoEBiasVar":
            continue
        key = "MoEBiasVar/" + "/".join(str(x) for x in path)
        if key not in by_key:
            continue
        old = v.get_value()
        x = np.array(jd.to_host(old), np.float32)
        for i, h in by_key[key]:
            if i is None:
                x[...] = G[h].reshape(x.shape)
            else:
                x[i] = G[h]
        x = x.astype(old.dtype)
        v.set_value(jax.make_array_from_callback(x.shape, old.sharding, lambda idx, x=x: x[idx]))
        n += 1
    if n != len(by_key):
        raise SystemExit(f"--bias-override：要换 {len(by_key)} 个 MoEBiasVar 叶子，只找到 {n} 个：{sorted(by_key)}")
    return {"dir": src, "n_tensors": len(want), "n_leaves": n}


def main(argv=None):
    global FLAG_AT_START
    a = parse_args(argv)
    if "://" not in a.load:
        a.load = str(Path(a.load).resolve())               # Orbax 要绝对路径
    if a.cpu_small or a.cpu:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        flags = os.environ.get("XLA_FLAGS", "")
        if "xla_force_host_platform_device_count" not in flags:
            os.environ["XLA_FLAGS"] = (flags + " --xla_force_host_platform_device_count=8").strip()
    import jax
    import jax.numpy as jnp
    from flax import nnx
    FLAG_AT_START = flag_snapshot()
    from maxtext.trainers.pre_train import train
    from maxtext.utils import sharding, train_utils

    # 多主机：hy4-4layers.yml 关了存档，MaxText 不会自己调 jax.distributed.initialize()，而载入检查点要它（同混元 3）
    if not (a.cpu_small or a.cpu) and not jax.distributed.is_initialized():
        jax.distributed.initialize()

    dump_steps = _int_steps(a.dump_steps)
    gb = replay.GoldenBatches(a.golden)
    golden_meta = json.loads((Path(a.golden) / "meta.json").read_text()) if (Path(a.golden) / "meta.json").is_file() else {}
    overrides = [f"run_name={Path(a.janus_run).name}", "base_output_directory=/tmp/hy4out", f"steps={a.steps}",
                 f"learning_rate_schedule_steps={a.steps}", f"load_parameters_path={a.load}"]
    if a.cpu_small:
        seq = gb.global_batch(gb.steps[0], verify=False)["tokens"].shape[1]
        overrides += CPU_SMALL + [f"max_target_length={seq}"]
    elif a.cpu:
        overrides += ["hardware=cpu", "skip_jax_distributed_system=True", "attention=dot_product", "sparse_matmul=False",
                      "megablox=False", "use_tokamax_splash=False"]
    if (a.forced_routing or a.forced_indexer) and not a.diag_step1:
        raise SystemExit("--forced-routing / --forced-indexer 只和 --diag-step1 一起用")
    if a.capture_choices:
        if not a.diag_step1 or a.forced_routing or a.forced_indexer:
            raise SystemExit("--capture-choices 只和 --diag-step1 一起用，且不和回放一起用")
        overrides.append("debug_capture_choices=true")
    if a.forced_routing:
        overrides.append("debug_forced_routing_deepseek=true")
    if a.forced_indexer:
        overrides.append("debug_forced_indexer_topk=true")
    config = mt_common.make_config(a.stage, overrides + list(a.maxtext_args))
    bad_cfg = mt_common.effective_mismatch(config)
    if bad_cfg and not a.maxtext_args and not (a.cpu_small or a.cpu):
        raise SystemExit(f"hy4-4layers.yml 里这些键没有按 yml 生效：{bad_cfg}")
    if config.global_batch_size_to_train_on != gb.global_rows:
        raise SystemExit(f"MaxText 全局 batch {config.global_batch_size_to_train_on} ≠ golden 每步 {gb.global_rows} 行")
    micro = gb.micro
    view = HFView(config, json.loads(Path(a.hf_config).read_text()), a.stage)

    (_, _, state_mesh_shardings, _, mesh, lr_schedule, _, _, _, _, state) = train_utils.setup_train_loop(config, None)
    jit_model, state = nnx.split(state)
    params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)
    overrides_info = {}
    if a.param_override:
        overrides_info["param_override"] = apply_param_override(a, state)
    if a.bias_override:
        overrides_info["bias_override"] = apply_bias_override(a, state, view)
    if overrides_info and jax.process_index() == 0:
        print("[janus] override " + json.dumps(overrides_info, ensure_ascii=False), flush=True)
    p_train, _ = train_utils.jit_train_and_eval_step(config, jit_model, mesh, state, state_mesh_shardings, train.train_step,
                                                    params_shardings=params_shardings)
    mem_probe = [device_memory_probe("after_setup")]
    resumed_from = None
    if a.resume:   # 整个训练状态换成存档里的（形状、dtype、切分照现在的状态），从存档的下一步接着回放
        import orbax.checkpoint as ocp
        from flax import nnx

        # to_pure_dict() 的叶子就是状态里那些数组本身（不拷贝）。只要还有一个名字指着它们，换进存档的值以后旧状态
        # 仍整份留在设备上：D3 在 v5p 16 片上每设备多 15.3 GiB，训练步要的 72.8G 临时区就放不下了。
        # 所以只留形状、dtype、切分，先把旧缓冲区释放掉，再按目标切分直接读进设备（读档期间峰值也只有一份状态）
        leaves, treedef = jax.tree_util.tree_flatten(state.to_pure_dict())
        keep = {f"l{i:05d}": i for i, v in enumerate(leaves) if getattr(v, "size", 1)}   # 同 _state_flat 的编号
        target = {k: jax.ShapeDtypeStruct(leaves[i].shape, leaves[i].dtype, sharding=leaves[i].sharding)
                  for k, i in keep.items()}
        for i in keep.values():
            if isinstance(leaves[i], jax.Array) and not leaves[i].is_deleted():
                leaves[i].delete()
        restored = ocp.StandardCheckpointer().restore(a.resume, target)
        for k, i in keep.items():
            leaves[i] = restored.pop(k)
        nnx.replace_by_pure_dict(state, jax.tree_util.tree_unflatten(treedef, leaves))
        del restored, leaves
        resumed_from = a.resume
        a.first_step = int(Path(a.resume.rstrip("/")).name.split("_")[-1]) + 1
        if jax.process_index() == 0:
            print(f"[janus] resumed from {a.resume}, first_step {a.first_step}", flush=True)
        mem_probe.append(device_memory_probe("after_resume"))   # 应与 after_setup 相同：设备上只有一份状态

    data_sharding = sharding.get_input_data_sharding(config, mesh)
    shape = (gb.global_rows, config.max_target_length)
    shard_devices = {}
    dmap = data_sharding.devices_indices_map(shape)
    starts = sorted({idx[0].start or 0 for idx in dmap.values()})
    per_shard = gb.global_rows // len(starts)
    # 每个分片拿 golden 一个 rank 的全部 microbatch（设备数 = GPU rank 数），或其中连续的几个（设备数是 rank 数的整数倍，
    # 如 16 个设备、累积 1：分片 d 拿 rank d // 2 的 micro d % 2），或连续几个 rank 的全部 microbatch（batch 只切在部分轴上，
    # 如张量并行 2 × FSDP 4：4 个分片各 4 行）。全局 batch 的行与行序不变
    if per_shard < 1 or (micro % per_shard and per_shard % micro):
        raise SystemExit(f"每个 batch 分片 {per_shard} 行，与 golden 每 rank {micro} 个 microbatch 不成整数倍")
    for dev, idx in dmap.items():
        shard_devices.setdefault(str((idx[0].start or 0) // per_shard), []).append(
            {"id": dev.id, "process": dev.process_index, "coords": list(getattr(dev, "coords", []) or [])})

    def put(arrs: dict[str, np.ndarray]):
        return {k: jax.make_array_from_callback(v.shape, data_sharding, lambda i, v=v: v[i]) for k, v in arrs.items()}

    if a.aot_only:
        g1 = put(replay.to_maxtext(gb.global_batch(gb.steps[0])))
        with jax.set_mesh(mesh), train.logical_axis_rules(config.logical_axis_rules):
            compiled = p_train.lower(state, g1).compile()
        ma = compiled.memory_analysis()
        rec = {k: getattr(ma, k, None) for k in ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes",
                                                   "temp_size_in_bytes", "generated_code_size_in_bytes")}
        if jax.process_index() == 0:
            Path(a.janus_run).mkdir(parents=True, exist_ok=True)
            (Path(a.janus_run) / "aot_memory.json").write_text(json.dumps(
                {"stage": a.stage, "memory_analysis": rec, "devices": jax.device_count(),
                 "device_kind": jax.devices()[0].device_kind,
                 "bytes_limit": (jax.local_devices()[0].memory_stats() or {}).get("bytes_limit")}, indent=1))
            print("[janus] AOT memory_analysis：" + json.dumps(rec), flush=True)
        return 0

    batches_sha = hashlib.sha256(b"".join((gb.dir / f"rank{r}.jsonl").read_bytes() for r in gb.ranks)).hexdigest()
    meta = build_meta(a, config, mesh, golden_meta, shard_devices, batches_sha, overrides)
    meta["resume"] = {"from": resumed_from, "save_every": a.save_every or None, "save_dir": a.save_dir or None}
    meta["memory_probe"] = mem_probe
    if overrides_info:
        meta["state_override"] = overrides_info
    dump = Hy4Dump(a.janus_run, meta, view, dump_steps, a.full, _int_steps(a.full_steps) or None, jax.process_index(),
                   only=a.full_only)
    flat = mt_common.variable_paths(state)
    dump.write_modules(flat)
    if not a.no_param_sha:
        dump.write_param_sha(flat, "init")
    if a.state_sha:
        dump.write_state_sha(flat, "init")
    # 路由偏置的逐层统计：名字与 GPU golden 相同（router.bias_mean.<HF 层号>；MTP 层记 mtp<j>），层号从映射反查
    import re as _re
    bias_names: dict[str, list] = {}
    for h, srcs in view.by_hf.items():
        if not view.is_buffer(h):
            continue
        m_ = _re.match(r"model\.layers\.(\d+)\.", h) or _re.match(r"model\.mtp_layers\.(\d+)\.", h)
        lname = m_.group(1) if h.startswith("model.layers.") else f"mtp{m_.group(1)}"
        for mt_key, i in srcs:
            bias_names.setdefault(view.state_key(flat, mt_key, "param"), []).append((i, lname))

    # 分组参数平方和：每个参数叶子（MaxText 的叶子，扫描层一个叶子含整栈）在设备上 FP32 求平方和，主机 FP64 按组合成。
    # 与 GPU 同一时刻：第 s 步记的是第 s 步更新之前的参数（GPU ga-b 第 1 步的 group_sq 等于初始权重逐组复算的值）
    group_fn = None
    if not a.no_group_norms:
        param_keys = sorted(k for k in flat if k.startswith("params/") and hasattr(flat[k], "dtype"))
        groups = {k: group_of(k) for k in param_keys}

        @jax.jit
        def group_fn(tree):
            return {k: jnp.sum(jnp.square(v.astype(jnp.float32))) for k, v in tree.items()}

        # 更新之后参数范数的精确值（param_norm_exact）：每 128 个元素一段在设备上 FP32 求平方和（短求和，误差约 1e-8），
        # 段和拿回主机按 FP64 相加；不含路由偏置（同 GPU calc_params_l2_norm，偏置不是参数）。用来核对 MaxText 的
        # learning/param_norm（整棵树在 FP32 里一路相加）本身的求和误差
        exact_keys = [k for k in param_keys if k not in bias_names]

        @jax.jit
        def exact_fn(tree):
            out = {}
            for k, v in tree.items():
                x = jnp.square(v.astype(jnp.float32)).reshape(-1)
                pad = (-x.shape[0]) % 128
                out[k] = jnp.sum(jnp.pad(x, (0, pad)).reshape(-1, 128), axis=-1)
            return out

    # 主权重离开 BF16 网格的比例：从 BF16 检查点起训时，初始的 FP32 主权重全在 BF16 网格上，只作用在权重一侧舍入 / 精度的
    # 开关第 1 步前向必然逐位不变；这一列量主权重随训练「长出」FP32 精度的速度（每 1024 个元素一段在设备上计数，主机 int64 相加）
    grid_steps = {int(x) for x in a.bf16_grid_steps.split(",") if x.strip()}
    offgrid_fn = None
    if grid_steps and group_fn is not None:
        @jax.jit
        def offgrid_fn(tree):
            out = {}
            for k, v in tree.items():
                x = v.astype(jnp.float32).reshape(-1)
                # 不用 x != f32(bf16(x))：XLA 会把这个往返当恒等省掉（excess precision / 代数化简），永远判「在网格上」。
                # 改看位：FP32 能被 BF16 精确表示，当且仅当低 16 位全是 0
                off = ((jax.lax.bitcast_convert_type(x, jnp.uint32) & jnp.uint32(0xFFFF)) != 0).astype(jnp.int32)
                pad = (-off.shape[0]) % 1024
                out[k] = jnp.sum(jnp.pad(off, (0, pad)).reshape(-1, 1024), axis=-1)
            return out

        def offgrid_cols(tree_flat):
            res = offgrid_fn({k: tree_flat[k] for k in exact_keys})
            cnt: dict[str, int] = {}
            tot: dict[str, int] = {}
            for k, v in res.items():
                g_ = groups[k]
                cnt[g_] = cnt.get(g_, 0) + int(np.asarray(jd.to_host(v), np.int64).sum())
                tot[g_] = tot.get(g_, 0) + int(np.prod(tree_flat[k].shape))
            cols = {f"bf16_offgrid_frac.{g_}": cnt[g_] / tot[g_] for g_ in sorted(cnt)}
            cols["bf16_offgrid_frac"] = sum(cnt.values()) / sum(tot.values())
            return cols

        if 0 in grid_steps:                                   # 所有进程都要调（多主机的 jit），只由进程 0 打印
            c0 = offgrid_cols(mt_common.variable_paths(state))
            meta["bf16_offgrid_step0"] = c0
            if jax.process_index() == 0:
                print(f"[janus] step 0 bf16_offgrid_frac {c0['bf16_offgrid_frac']!r}", flush=True)

    if a.diag_step1:
        return diag_step1(a, config, jit_model, state, params_shardings, mesh, gb, put, view, meta)

    snap_steps = set(_int_steps(a.hf_snapshot_at))
    snap_verify = set(_int_steps(a.hf_snapshot_verify))
    t_start = time.time()
    for s in range(a.first_step, a.steps + 1):
        g = gb.global_batch(s)                               # 每个字段核对 sha256
        batch = put(replay.to_maxtext(g))
        fed = replay.from_maxtext({k: jd.to_host(v) for k, v in batch.items()})   # 从设备读回来再算指纹
        dump.record_batch(replay.fingerprints(fed, micro, s, gb.fields))
        if a.eval_variants_at and s == a.eval_variants_at:
            return eval_variants(a, config, jit_model, state, batch, mesh, view, s, meta)
        pre = mt_common.variable_paths(state)
        dump.capture(s, pre, "before")
        if s in snap_steps or s in snap_verify:
            hf_snapshot(a, view, pre, s, write=s in snap_steps, process_index=jax.process_index())
        pre_groups = {}
        if group_fn is not None:
            res = group_fn({k: pre[k] for k in param_keys})
            acc: dict[str, float] = {}
            for k, v in res.items():
                acc[groups[k]] = acc.get(groups[k], 0.0) + float(np.asarray(jd.to_host(v), np.float64))
            pre_groups = {f"group_sq.{g}": v for g, v in sorted(acc.items())}
            pre_groups["param_norm_grouped"] = float(np.sqrt(sum(acc.values())))
        del pre
        t0 = time.time()
        with jax.set_mesh(mesh), train.logical_axis_rules(config.logical_axis_rules):
            state, metrics = p_train(state, batch)
        sc = {k: float(jd.to_host(v)) for k, v in metrics["scalar"].items()}
        step_time = time.time() - t0
        if s == a.first_step:
            meta["flags"]["after_compile"] = flag_snapshot()
        flat = mt_common.variable_paths(state)
        vals = {"lm_loss": sc["learning/lm_loss"], "loss": sc["learning/loss"],
                "grad_norm": sc["learning/raw_grad_norm"], "raw_grad_norm": sc["learning/raw_grad_norm"],
                "grad_norm_clipped": sc["learning/grad_norm"], "param_norm": sc["learning/param_norm"],
                "lr": float(lr_schedule(s - 1)), "wd": float(config.adam_weight_decay), "skipped_iter": 0.0,
                "total_weights": sc["learning/total_weights"], "step_time": step_time}
        for k in ("learning/mtp_loss", "learning/indexer_loss", "learning/moe_lb_loss"):
            if k in sc:
                vals[k.split("/")[1]] = sc[k]
        # MaxText log_indexer_loss_per_layer（默认关）：各产出层的 KD loss，按 MaxText 的收集顺序编号（截层 D1 是第 0 层、MTP 层；
        # hy4-mini 是第 0、1、5 层与 MTP 层）。learning/indexer_loss 是它们的平均；GPU 的 indexer_loss.<层> 是每层乘了
        # 系数 ÷ 产出层数的值，所以 TPU 的 indexer_loss_layer.<i> ≈ 产出层数 × GPU 对应层的值
        for k, v in sc.items():
            if k.startswith("learning/indexer_loss_layer_"):
                vals["indexer_loss_layer." + k.rsplit("_", 1)[1]] = v
        for bk, slots in bias_names.items():
            b_all = np.asarray(jd.to_host(flat[bk]), np.float64)
            for i, lname in slots:
                b = b_all if i is None else np.take(b_all, i, axis=0)
                vals[f"router.bias_mean.{lname}"] = float(b.mean())
                vals[f"router.bias_std.{lname}"] = float(b.std(ddof=1))
        vals.update(pre_groups)
        if "qk_clip_per_head" in metrics:   # MaxText qk_clip_log_per_head：逐层逐头的最大 logit 与缩放，记成与 GPU 同名的列
            vals.update(qk_clip_columns(metrics["qk_clip_per_head"], view))
        if not np.isfinite(sc["learning/param_norm"]) and group_fn is not None:
            # 定位：哪些参数叶子在这一步更新之后出现非有限值（每个叶子在设备上求和，NaN / Inf 会传到和里）
            bad = [k for k in param_keys if not np.isfinite(float(np.asarray(jd.to_host(jnp.sum(flat[k].astype(jnp.float32))))))]
            if jax.process_index() == 0:
                print(f"[janus] step {s}: {len(bad)} 个参数叶子有非有限值：{bad[:40]}", flush=True)
        if group_fn is not None:
            ex = exact_fn({k: flat[k] for k in exact_keys})
            vals["param_norm_exact"] = float(np.sqrt(sum(float(np.asarray(jd.to_host(v), np.float64).sum()) for v in ex.values())))
        if offgrid_fn is not None and s in grid_steps:
            vals.update(offgrid_cols(flat))
        # 与 GPU（Megatron 线）同名的两列，T1 的检查项目录按这两个名字比：
        # param_norm_fp64 = 分组平方和 FP64 合成（GPU 是它自己的 FP64 复算）；mtp_loss_mt = MaxText 口径的 MTP loss（已乘系数）
        if "param_norm_grouped" in vals:
            vals["param_norm_fp64"] = vals["param_norm_grouped"]
        if "mtp_loss" in vals:
            vals["mtp_loss_mt"] = vals["mtp_loss"]
        dump.log_metrics(s, vals)
        if s in dump.dump_steps:
            dump.capture(s, flat, "after")
            dump.flush(s)
        if jax.process_index() == 0:
            print(f"[janus] step {s} lm_loss {vals['lm_loss']!r} grad_norm {vals['grad_norm']!r} "
                  f"param_norm {vals['param_norm']!r} ({step_time:.2f} s)", flush=True)
        if a.save_every and a.save_dir and (s % a.save_every == 0 or (s == a.steps and a.save_last)):
            import orbax.checkpoint as ocp

            t_save = time.time()
            ck = ocp.StandardCheckpointer()
            ck.save(f"{a.save_dir.rstrip('/')}/step_{s}", _state_flat(state.to_pure_dict()))
            ck.wait_until_finished()   # 训练步会捐出状态的缓冲区，存完再往下走
            if jax.process_index() == 0:
                print(f"[janus] saved step {s} to {a.save_dir} ({time.time() - t_save:.1f} s)", flush=True)
    if not a.no_param_sha:
        dump.write_param_sha(mt_common.variable_paths(state), "final")
    if a.state_sha:
        dump.write_state_sha(mt_common.variable_paths(state), "final")
    if jax.process_index() == 0:
        meta["wall_seconds"] = time.time() - t_start
        if a.hlo_dir:
            meta["hlo"] = collect_hlo(a.hlo_dir, a.janus_run)
        try:
            meta["memory"] = {str(d.id): {k: v for k, v in (d.memory_stats() or {}).items()
                                          if k in ("peak_bytes_in_use", "bytes_limit", "bytes_in_use")} for d in jax.local_devices()}
        except Exception as e:  # noqa: BLE001
            meta["memory"] = {"error": str(e)}
        (Path(a.janus_run) / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    dump.done()
    return 0


if __name__ == "__main__":
    sys.exit(main())
