"""L0 配置逐项机读对照（hy4-mini T1）：TPU 运行实际生效的 MaxText 配置（meta.json 的 maxtext_config_full，
train_hy4.py 在建好配置之后整份写下）对 GPU 运行的 meta.json 与两边共同的 HF config.json，逐项判。

三种规则：
- same_hf：模型结构。GPU 按 HF 目录的 config.json 建模型（meta.config.hf_config_filled_from_release 为空 = 没有补任何键），
  所以 GPU 的值就是 HF config 的值；TPU 的值必须等于它（必要时经过写明的换算，如层类型列表 → 稠密层数、共享间隔）。
  两边是同一个 HF 目录：TPU meta.checkpoint.manifest_sha256 与 GPU meta.checkpoint.tensor_sha256_manifest 必须相同，
  TPU 记的 hf_config_sha256 必须等于本地 config.json 的 sha256（这两条也作为行列出）。
- same_gpu：训练设置。TPU 的值必须等于 GPU meta.config 里记的值（经写明的换算）。
- registered：两边都必须等于登记值（PROTOCOL 第 1.2 节的设置，如 dropout 0、权重衰减 0）。
不在表里的项不判：MaxText 有几百个键，大部分和这次对齐无关（并行、调度、检查点、评估等）。默认只核「两边语义设置相同」。

可选 --precision-switches：另加第四种规则 precision，逐项核 TPU 实际生效的精度开关等于 01-mini/config-alignment.md 登记的
对齐配置值（「数值精度」表 TPU 一栏「默认 → T1」箭头右边的值，和「为对齐改过的代码」开关表里的开关）。这些开关 GPU 一侧
没有对应的键（GPU 的舍入位置是写在代码里的），所以参照是登记值，不是 GPU meta。登记值抄在下面的 PRECISION 表里，改
config-alignment.md 时要同步改这张表。

  python l0_config_vs_gpu.py --tpu-meta <TPU 运行>/meta.json --gpu-meta <GPU 运行>/meta.json --hf-config <HF 目录>/config.json \
      --out <结果 json> [--stage <阶段名>] [--precision-switches]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def full_layers(types: list[str]) -> list[int]:
    return [i for i, t in enumerate(types) if t == "full"]


def tpu_full_layers(t: dict) -> list[int]:
    """MaxText 混元 4：稠密层都自带索引器；MoE 层按 indexer_share_interval 一组，组内第一层自带、其余复用（模型代码的规则）。"""
    fd, n, k = int(t["first_num_dense_layers"]), int(t["base_num_decoder_layers"]), int(t["indexer_share_interval"])
    return [i for i in range(n) if i < fd or (i - fd) % k == 0]


_CA = "config-alignment.md"
# (键, 登记的对齐配置值, 出处行)。值抄自 01-mini/config-alignment.md（2026-11-11 C 口径版：现行比较 tc-a 那一版；上一代 tz-off-rep 没有最后加的 M52、M58 两行）。
PRECISION = [
    ("dtype", "bfloat16", "数值精度「计算、主权重、梯度、优化器状态」"),
    ("weight_dtype", "float32", "数值精度「计算、主权重、梯度、优化器状态」"),
    ("grad_dtype", "float32", "数值精度「计算、主权重、梯度、优化器状态」"),
    ("mu_dtype", "float32", "数值精度「计算、主权重、梯度、优化器状态」"),
    ("matmul_precision", "default", "数值精度「矩阵乘默认精度」"),
    ("dense_fp32_accumulation", True, "数值精度「GEMM 累加只舍一次」（M12）"),
    ("rmsnorm_single_rounding", True, "数值精度「RMSNorm 舍入次数」（M9）"),
    ("routed_gate_bf16_rounding", True, "数值精度「路由 GEMM」（M8）"),
    ("float32_gate_logits", True, "数值精度「路由 GEMM」"),
    ("router_matmul_precision", "highest", "数值精度「路由 GEMM」"),
    ("mlp_activation_round_per_op", True, "数值精度「门控激活舍入」（M10）"),
    ("indexer_fp32_scores", True, "数值精度「索引器打分」（M23）"),
    ("indexer_weights_proj_bf16_weight", True, "数值精度「索引器打分」（M27）"),
    ("hc_matmul_precision", "highest", "数值精度「iHC 混合系数」"),
    ("ihc_stream_sum_unrolled", True, "数值精度「iHC 四路流加权和」（M28）"),
    ("ihc_force_bf16_rounding", True, "数值精度「iHC 分支输入与更新后的流的 BF16 舍入」（M36）"),
    ("ihc_stream_grad_gpu_order", True, "数值精度「iHC 与 hc_head 的流梯度、分支输出的 BF16 舍入」（M58，2026-11-11 起 C 口径）"),
    ("hc_head_force_bf16_rounding", True, "数值精度「hc_head 收拢之后的 BF16 舍入」（M52，2026-11-11 起 C 口径）"),
    ("logits_dot_in_fp32", True, "数值精度「输出头」（M13）"),
    ("logits_matmul_precision", "highest", "数值精度「输出头」（M13）"),
    ("transcendental_accuracy_highest", True, "数值精度「FP32 超越函数」（M33、M37）"),
    ("accurate_sigmoid_attn_gate", None, "开关表 M38（默认 null，跟随 transcendental_accuracy_highest）"),
    ("accurate_sigmoid_hc_head", None, "开关表 M38（同上）"),
    ("accurate_sigmoid_ihc", None, "开关表 M38（同上）"),
    ("accurate_sigmoid_router", None, "开关表 M38（同上）"),
    ("accurate_silu_dense", None, "开关表 M38（同上）"),
    ("accurate_silu_routed", None, "开关表 M38（同上）"),
    ("accurate_silu_shared", None, "开关表 M38（同上）"),
    ("muon_mcore_ns_bf16", True, "数值精度「Newton–Schulz 的矩阵乘」（M31）"),
    ("muon_adam_accurate", True, "数值精度「Adam 的 FP32 除法、sqrt 与偏差修正」（M34）"),
    ("rmsnorm_scale_grad_fp32", False, "数值精度「三个梯度 FP32 开关」（T1 不开）"),
    ("logits_dense_wgrad_fp32", False, "数值精度「三个梯度 FP32 开关」（T1 不开）"),
    ("embedding_grad_fp32", False, "数值精度「三个梯度 FP32 开关」（T1 不开）"),
    ("moe_mcore_prob_before_fc2", True, "开关表 M40"),
    ("mlp_fused_fc1_dgrad", True, "开关表 M41"),
    ("splash_dsinks_fp32", False, "开关表 M42（对齐配置不开）"),
    ("splash_dsinks_gpu_order", True, "开关表 M42b（M47）"),
    ("sa_use_base2_exp", False, "开关表 M43；并行与确定性「注意力 实现」（2026-11-11 起关）"),
    ("indexer_kd_generalized_kl", True, "开关表 M50（只改记录值）"),
]


def precision_rows(t: dict) -> list[dict]:
    return [
        {"category": "精度开关", "item": k, "rule": "precision", "tpu": t.get(k, "<缺>"), "ref": want,
         "ref_source": f"{_CA} {src}", "equal": t.get(k, "<缺>") == want, "note": ""}
        for k, want, src in PRECISION
    ]


def rows_for(t: dict, g: dict, h: dict, tm: dict, gm: dict, hf_sha: str):
    gc = g
    R = []

    def add(cat, item, rule, tpu, ref, src, note=""):
        R.append({"category": cat, "item": item, "rule": rule, "tpu": tpu, "ref": ref, "ref_source": src,
                  "equal": tpu == ref, "note": note})

    add("来源", "同一个 HF 目录（张量清单 sha256）", "same_gpu", tm["checkpoint"]["manifest_sha256"],
        gm["checkpoint"]["tensor_sha256_manifest"], "GPU meta.checkpoint.tensor_sha256_manifest")
    add("来源", "TPU 读的 config.json 就是这一份", "same_hf", tm["checkpoint"]["hf_config_sha256"], hf_sha, "本地 config.json 的 sha256")
    add("来源", "GPU 没有补 HF config 的键", "registered", None, None, "GPU meta.config.hf_config_filled_from_release",
        note=f"GPU 记的是 {gc['config'].get('hf_config_filled_from_release')!r}")
    R[-1]["equal"] = gc["config"].get("hf_config_filled_from_release") == {}
    S = [
        ("结构", "主干层数", "base_num_decoder_layers", "num_hidden_layers", None),
        ("结构", "MTP 层数", "mtp_num_layers", "num_nextn_predict_layers", None),
        ("结构", "隐藏宽度", "base_emb_dim", "hidden_size", None),
        ("结构", "注意力头数", "base_num_query_heads", "num_attention_heads", None),
        ("结构", "q 低秩", "q_lora_rank", "q_lora_rank", None),
        ("结构", "kv 低秩", "kv_lora_rank", "kv_lora_rank", None),
        ("结构", "qk 非 RoPE 维", "qk_nope_head_dim", "qk_nope_head_dim", None),
        ("结构", "qk RoPE 维", "qk_rope_head_dim", "qk_rope_head_dim", None),
        ("结构", "v 每头维", "v_head_dim", "v_head_dim", None),
        ("结构", "稠密 FFN 宽", "base_mlp_dim", "intermediate_size", None),
        ("结构", "路由专家 FFN 宽", "base_moe_mlp_dim", "moe_intermediate_size", None),
        ("结构", "路由专家数", "num_experts", "n_routed_experts", None),
        ("结构", "每 token 选几个专家", "num_experts_per_tok", "num_experts_per_tok", None),
        ("结构", "共享专家数", "shared_experts", "n_shared_experts", None),
        ("结构", "词表", "vocab_size", "vocab_size", None),
        ("结构", "路由缩放系数", "routed_scaling_factor", "routed_scaling_factor", None),
        ("结构", "路由打分函数", "routed_score_func", "scoring_func", "sigmoid"),
        ("结构", "选中专家的权重归一化", "routed_weight_normalize", "norm_topk_prob", None),
        ("结构", "RMSNorm eps", "normalization_layer_epsilon", "rms_norm_eps", None),
        ("结构", "MLA 两个潜空间 norm 的 eps", "mla_lora_norm_epsilon", "rms_norm_eps", None),
        ("结构", "索引器头数", "indexer_n_heads", "index_n_heads", None),
        ("结构", "索引器每头维", "indexer_head_dim", "index_head_dim", None),
        ("结构", "索引器 top-k", "indexer_topk", "index_topk", None),
        ("结构", "iHC 流数", "mhc_expansion_rate", "hc_mult", None),
        ("结构", "iHC eps", "hc_eps", "hc_eps", None),
        ("结构", "iHC 写回幅度", "hc_post_magnitude", "hc_magnitude", None),
        ("结构", "SwiGLU 截断", "mlp_activations_limit", "swiglu_limit", None),
        ("结构", "可学习 sink", "attention_sink", "learnable_sink", None),
        ("结构", "门控 MLA（逐元素）", "attention_output_gate", "gating_type", None),
        ("结构", "输出头与词嵌入不共享", "logits_via_embedding", "tie_word_embeddings", None),
        ("结构", "注意力 bias", "attention_bias", "attention_bias", None),
    ]
    for cat, item, tk, hk, hf_default in S:
        hv = h.get(hk, hf_default)
        add(cat, item, "same_hf", t.get(tk, "<缺>"), hv, f"HF config.{hk}" + ("（缺省 sigmoid，模型代码默认）" if hf_default and hk not in h else ""))
    add("结构", "稠密层数（mlp_layer_types 里 dense 的个数，且都在最前面）", "same_hf", t["first_num_dense_layers"],
        h["mlp_layer_types"].index("sparse"), "HF config.mlp_layer_types")
    R[-1]["equal"] = R[-1]["equal"] and h["mlp_layer_types"].count("dense") == h["mlp_layer_types"].index("sparse")
    add("结构", "自带索引器的层（其余复用上一个）", "same_hf", tpu_full_layers(t), full_layers(h["indexer_types"]),
        "HF config.indexer_types", note="TPU 按 first_num_dense_layers 与 indexer_share_interval 推出")
    add("结构", "RoPE θ", "same_hf", t["rope_max_timescale"], h["rope_parameters"]["rope_theta"], "HF config.rope_parameters.rope_theta")
    add("结构", "RoPE 不缩放（yarn factor 1 = 普通 RoPE）", "same_hf", t["rope_factor"], 1,
        f"HF config.rope_parameters.rope_type = {h['rope_parameters']['rope_type']}（factor 1）")
    add("结构", "MTP loss 系数", "same_gpu", t["mtp_loss_scaling_factor"], gc["config"]["mtp_loss_factor"], "GPU meta.config.mtp_loss_factor")
    R.append({"category": "结构", "item": "MTP loss 系数（HF）", "rule": "same_hf", "tpu": t["mtp_loss_scaling_factor"],
              "ref": h["mtp_loss_factor"], "ref_source": "HF config.mtp_loss_factor", "equal": t["mtp_loss_scaling_factor"] == h["mtp_loss_factor"], "note": ""})

    c = gc["config"]
    G = [
        ("训练", "学习率", t["learning_rate"], c["lr"], "lr"),
        ("训练", "Adam β1", t["adam_b1"], c["betas"][0], "betas[0]"),
        ("训练", "Adam β2", t["adam_b2"], c["adam_beta2"], "adam_beta2"),
        ("训练", "Adam eps", t["adam_eps"], c["adam_eps"], "adam_eps"),
        ("训练", "梯度裁剪阈值", t["gradient_clipping_threshold"], c["clip"], "clip"),
        ("训练", "序列长度", t["max_target_length"], c["seq"], "seq"),
        ("训练", "全局批（条）", t["global_batch_size_to_train_on"], gm["parallelism"]["gbs"], "parallelism.gbs"),
        ("训练", "梯度累积步数", t["gradient_accumulation_steps"], gm["parallelism"]["ga"], "parallelism.ga"),
        ("训练", "QK-Clip 开", t["use_qk_clip"], c["qk_clip"]["on"], "qk_clip.on"),
        ("训练", "QK-Clip 阈值", t["qk_clip_threshold"], c["qk_clip"]["threshold"], "qk_clip.threshold"),
        ("训练", "索引器 KD 系数", t["indexer_loss_scaling_factor"], c["indexer_loss"]["maxtext_coeff"], "indexer_loss.maxtext_coeff"),
        ("训练", "索引器参与训练（稀疏 KD）", t["indexer_sparse_training"], c["indexer_loss"]["sparse"] and c["indexer_trained"], "indexer_loss.sparse 且 indexer_trained"),
        ("训练", "路由偏置更新率", t["routed_bias_update_rate"], c["routing_bias_update_rate"], "routing_bias_update_rate"),
        ("训练", "优化器 Muon", t["opt_type"], c["optimizer"], "optimizer"),
        ("训练", "Muon Nesterov", t["muon_mcore_nesterov"], str(c["optimizer_extra"]["muon_nesterov"]) == "True", "optimizer_extra.muon_nesterov"),
        ("训练", "iHC 与 hc_head 不走 Muon", t["muon_exclude_ihc"], c["muon_exclude_ihc"] == "ihc+head", "muon_exclude_ihc"),
        ("训练", "输出门每头正交化", not t["muon_attention_gate_whole_matrix"], c["muon_gate"] == "per_head", "muon_gate"),
        ("训练", "MTP 层也做 QK-Clip", True, c["qk_clip_mtp"] == "on", "qk_clip_mtp（TPU 一侧取 True 的依据：τ 0.1 的 TPU 运行逐步记 qk.clipped.mtp0，第 1 步 MTP 层 8 个头被裁，QK-Clip 对 MTP 层生效）"),
    ]
    for cat, item, tv, gv, src in G:
        add(cat, item, "same_gpu", tv, gv, f"GPU meta.config.{src}")
    REG = [
        ("训练", "学习率常数（warmup 0）", t["warmup_steps_fraction"], 0.0),
        ("训练", "学习率常数（不衰减）", t["learning_rate_final_fraction"], 1.0),
        ("训练", "Muon 动量 0.95（GPU 入口没覆盖 muon_momentum，取 Megatron-Core 0.20 OptimizerConfig 的默认 0.95）", t["muon_beta"],
         0.95 if "muon_momentum" not in c["optimizer_extra"] else float(c["optimizer_extra"]["muon_momentum"])),
        ("训练", "Muon 权重衰减 0", t["muon_weight_decay"], c["weight_decay_effective"]),
        ("训练", "Adam 权重衰减 0", t["adam_weight_decay"], c["weight_decay_effective"]),
        ("训练", "dropout 0", t["dropout_rate"], 0.0),
        ("训练", "MoE 负载均衡 loss 0", t["load_balance_loss_weight"], 0.0),
        ("训练", "z-loss 0", t["z_loss_multiplier"], 0.0),
        ("训练", "MoE 不丢 token（capacity_factor −1）", t["capacity_factor"], -1.0),
        ("训练", "路由偏置不做零均值更新（原版 Megatron-Core 没有，台账 G14）", t["routed_bias_zero_mean_update"], False),
    ]
    for cat, item, tv, want in REG:
        add(cat, item, "registered", tv, want, "登记值（PROTOCOL 第 1.2 节）")
    R.append({"category": "数据", "item": "loss 掩码（eod）", "rule": "registered", "tpu": "回放 GPU 存下的 loss_mask",
              "ref": c["loss_mask"]["rule"], "ref_source": "GPU meta.config.loss_mask",
              "equal": c["loss_mask"]["eod_mask"] == "on",
              "note": "TPU 不自己算掩码（eod_mask_loss 关），逐步回放 GPU 存下的 loss_mask；逐位一致见 t1zg-l0-data（L0.data.fingerprint 含 loss_mask 字段）"})
    return R


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tpu-meta", required=True)
    p.add_argument("--gpu-meta", required=True)
    p.add_argument("--hf-config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stage", default="t1-l0-config", help="结果里记的阶段名（默认 t1-l0-config）")
    p.add_argument("--precision-switches", action="store_true",
                   help="另核精度开关：TPU 实际生效值对 config-alignment.md 登记的对齐配置值（规则 precision）")
    a = p.parse_args()
    tm = json.loads(Path(a.tpu_meta).read_text())
    gm = json.loads(Path(a.gpu_meta).read_text())
    h = json.loads(Path(a.hf_config).read_text())
    hf_sha = hashlib.sha256(Path(a.hf_config).read_bytes()).hexdigest()
    rows = rows_for(tm["maxtext_config_full"], gm, h, tm, gm, hf_sha)
    rules = ["same_hf", "same_gpu", "registered"]
    if a.precision_switches:
        rows += precision_rows(tm["maxtext_config_full"])
        rules.append("precision")
    bad = [f"{r['item']}：TPU {r['tpu']!r}，参照 {r['ref']!r}（{r['ref_source']}）" for r in rows if not r["equal"]]
    res = {"schema_version": 1, "stage": a.stage, "model": "hunyuan4-mini",
           "inputs": {"tpu_run": tm["run_name"], "gpu_run": gm["run_name"], "hf_config_sha256": hf_sha},
           "checks": [{"id": "L0.config.items", "type": "config_diff", "level": "L0", "status": "fail" if bad else "pass",
                       "threshold": {"kind": "exact"},
                       "metrics": {"items": len(rows), "differ": len(bad),
                                   "by_rule": {k: sum(r["rule"] == k for r in rows) for k in rules}},
                       "message": f"{len(rows)} 项（结构对 HF config {sum(r['rule'] == 'same_hf' for r in rows)}、训练设置对 GPU meta "
                                  f"{sum(r['rule'] == 'same_gpu' for r in rows)}、登记值 {sum(r['rule'] == 'registered' for r in rows)}"
                                  + (f"、精度开关对登记的对齐配置 {sum(r['rule'] == 'precision' for r in rows)}" if a.precision_switches else "")
                                  + f"）：不同 {len(bad)}",
                       "details_inline": {"differ": bad, "rows": rows}}]}
    res["summary"] = {k: sum(c["status"] == k for c in res["checks"]) for k in ("pass", "fail", "inconclusive", "error")}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(res["checks"][0]["status"], res["checks"][0]["message"])
    for b in bad:
        print("  ", b)


if __name__ == "__main__":
    main()
