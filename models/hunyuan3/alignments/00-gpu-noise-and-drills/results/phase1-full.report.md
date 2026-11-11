# janus 报告：hunyuan3-4l / phase1-full

- 被测：`s-a`；golden：`s-b`, `s-c`, `s-d`
- janus 0.0.1；目录 `catalogs/hunyuan3-4l/phase1-full.yaml`（sha256 60a768098bd2…）
- 结果：pass 38、fail 0、inconclusive 8、error 0

**证据不足（inconclusive）**：`L2.layer0.bwd`、`L2.layer3.bwd`、`L2.layer0.grads`、`L2.layer3.grads`、`L4.loss`、`L4.param_norm`、`L4.router.bias_std`、`L4.router.load_cv`

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L0.config.diff` | config_diff | L0 | pass | fields=17, must_equal_differ=0, other_differ=0, missing=0 | 必须相同的字段都相同；其他不同 0 项 |
| `L0.data.fingerprint` | data_fingerprint | L0 | pass | units_per_field={tokens:1600 labels:1600 loss_mask:1600 position_ids:1600}, first_loss={test:11.5 ref:11.5}, units=6500, differ=0 | token 与首步 loss 逐位相同 |
| `L0.weights.init` | checkpoint_equal | L0 | pass | params=1480, differ=0, only_one_side=0 | 全部参数逐位相同 |
| `L0.weights.loaded` | bitwise_equal | L0 | pass | tensors=50, differ=0 | 全部逐位相同 |
| `L0.main_params.loaded` | bitwise_equal | L0 | pass | tensors=50, differ=0 | 全部逐位相同 |
| `L0.buffers.loaded` | bitwise_equal | L0 | pass | tensors=3, differ=0 | 全部逐位相同 |
| `L0.dtype.manifest` | dtype_manifest | L0 | pass | entries=2960, wrong_dtype=0, unmatched=0, rule_hits={buffer.*expert_bias:48 * !buffer.*:2912} | 2960 项 dtype 全部符合规则 |
| `L0.model.structure` | artifact_check | L0 | pass | rules=6, bad=0 | 全部规则满足 |
| `L1.optim.adamw.replay` | optimizer_step | L1 | pass | rel_l2_max=1.788e-03, max_abs_max=1.172e-07, tensors=141, lr=1.000e-05 | 全部在门槛内 |
| `L1.router.topk.fp64` | discrete_agreement | L1 | pass | set_frac_min=1.000e+00, order_frac_min=1.000e+00 | set_frac 最低 1.000000 |
| `L2.layer0.fwd` | tensor_tolerance | L2 | pass | rel_l2_max=0.000e+00, cosine_min=1.000e+00, mean_shift_max=0.000e+00, max_abs_max=0.000e+00 | 在噪声参照内 |
| `L2.layer1.fwd` | tensor_tolerance | L2 | pass | rel_l2_max=0.000e+00, cosine_min=1.000e+00, mean_shift_max=0.000e+00, max_abs_max=0.000e+00 | 在噪声参照内 |
| `L2.layer2.fwd` | tensor_tolerance | L2 | pass | rel_l2_max=0.000e+00, cosine_min=1.000e+00, mean_shift_max=0.000e+00, max_abs_max=0.000e+00 | 在噪声参照内 |
| `L2.layer3.fwd` | tensor_tolerance | L2 | pass | rel_l2_max=0.000e+00, cosine_min=1.000e+00, mean_shift_max=0.000e+00, max_abs_max=0.000e+00 | 在噪声参照内 |
| `L2.layer0.bwd` | tensor_tolerance | L2 | inconclusive | rel_l2_max=4.285e-03, cosine_min=1.000e+00, norm_ratio_dev_max=3.913e-04, tensors=12 | 介于噪声参照一到两倍之间，或参照不足 |
| `L2.layer1.bwd` | tensor_tolerance | L2 | pass | rel_l2_max=3.483e-03, cosine_min=1.000e+00, norm_ratio_dev_max=3.210e-05, tensors=12 | 在噪声参照内 |
| `L2.layer2.bwd` | tensor_tolerance | L2 | pass | rel_l2_max=5.217e-04, cosine_min=1.000e+00, norm_ratio_dev_max=1.374e-06, tensors=12 | 在噪声参照内 |
| `L2.layer3.bwd` | tensor_tolerance | L2 | inconclusive | rel_l2_max=1.042e-05, cosine_min=1.000e+00, norm_ratio_dev_max=6.567e-09, tensors=12 | 介于噪声参照一到两倍之间，或参照不足 |
| `L2.layer0.grads` | tensor_tolerance | L2 | inconclusive | rel_l2_max=4.129e-04, cosine_min=1.000e+00, norm_ratio_dev_max=2.481e-04, tensors=8 | 介于噪声参照一到两倍之间，或参照不足 |
| `L2.layer1.grads` | tensor_tolerance | L2 | pass | rel_l2_max=5.097e-04, cosine_min=1.000e+00, norm_ratio_dev_max=3.528e-04, tensors=13 | 在噪声参照内 |
| `L2.layer2.grads` | tensor_tolerance | L2 | pass | rel_l2_max=1.451e-04, cosine_min=1.000e+00, norm_ratio_dev_max=3.793e-06, tensors=13 | 在噪声参照内 |
| `L2.layer3.grads` | tensor_tolerance | L2 | inconclusive | rel_l2_max=4.349e-06, cosine_min=1.000e+00, norm_ratio_dev_max=4.669e-08, tensors=13 | 介于噪声参照一到两倍之间，或参照不足 |
| `L2.layer1.router` | discrete_agreement | L2 | pass | set_frac_min=1.000e+00, order_frac_min=1.000e+00 | set_frac 最低 1.000000 |
| `L2.layer2.router` | discrete_agreement | L2 | pass | set_frac_min=1.000e+00, order_frac_min=1.000e+00 | set_frac 最低 1.000000 |
| `L2.layer3.router` | discrete_agreement | L2 | pass | set_frac_min=1.000e+00, order_frac_min=1.000e+00 | set_frac 最低 1.000000 |
| `L3.step1.act.bitwise` | bitwise_equal | L3 | pass | tensors=46, differ=0 | 全部逐位相同 |
| `L3.step1.loss.per_token` | tensor_tolerance | L3 | pass | rel_l2_max=0.000e+00, max_abs_max=0.000e+00, mean_shift_max=0.000e+00, tensors=1 | 在噪声参照内 |
| `L3.step1.logits` | tensor_tolerance | L3 | pass | rel_l2_max=0.000e+00, cosine_min=1.000e+00, max_abs_max=0.000e+00, kl_max=0.000e+00 | 在噪声参照内 |
| `L3.step1.grads.all` | tensor_tolerance | L3 | pass | rel_l2_max=5.097e-04, cosine_min=1.000e+00, norm_ratio_dev_max=3.528e-04, tensors=50 | 在噪声参照内 |
| `L3.step1.grads.dist` | tensor_tolerance | L3 | pass | std_ratio_dev_max=3.529e-04, zero_frac_diff_max=2.543e-06, rel_err_p99_max=4.278e-02, tensors=50 | 在噪声参照内 |
| `L3.step1.grads.ulp` | ulp_distribution | L3 | pass | zero_frac_min=1.073e-04, one_frac=1.272e-02, buckets={0:0.225 1:0.0127 2-4:0.0324 5-8:0.0255 9-16:0.0312 17+:0.673}, p999_max=3.529e+06 | 诊断：只报分布，不判 |
| `L3.step1.updated` | tensor_tolerance | L3 | pass | rel_l2_max=1.836e-06, max_abs_max=1.627e-05, tensors=50 | 在噪声参照内 |
| `L3.step1.optim_state` | tensor_tolerance | L3 | pass | rel_l2_max=1.216e-03, cosine_min=1.000e+00, tensors=100 | 在噪声参照内 |
| `L3.step1.fake_fp32.main_param` | fake_fp32 | L3 | pass | low16_zero_frac_max=2.755e-03, tensors=50, fake=0, skipped_non_fp32=0 | 50 个 float32 张量低 16 位全 0 的比例最高 2.75e-03 |
| `L3.step1.fake_fp32.optim_state` | fake_fp32 | L3 | pass | low16_zero_frac_max=1.844e-05, tensors=100, fake=0, skipped_non_fp32=0 | 100 个 float32 张量低 16 位全 0 的比例最高 1.84e-05 |
| `L3.step1.bias_after` | tensor_tolerance | L3 | pass | rel_l2_max=0.000e+00, max_abs_max=0.000e+00, tensors=3 | 在噪声参照内 |
| `L3.step1.coverage` | coverage | L3 | pass |  | 全部比到 |
| `L4.loss` | multi_golden | L4 | inconclusive | d0_max=0.000e+00, signed_mean_max=2.873e-05, mare_max=3.939e-05, max_step_max=2.406e-04 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.lr` | scalar_series | L4 | pass | steps=100, bad_steps=0, first_bad=None | 逐步在容差内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-1.251e-05, signed_mean=-1.779e-03, mare=7.662e-03, max_step=6.417e-02 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | inconclusive | d0=0.000e+00, signed_mean=5.457e-10, mare=1.364e-08, max_step=1.080e-07 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.bias_std` | router_stats | L4 | inconclusive | signed_mean_max=7.110e-04, mare_max=1.256e-03, max_step_max=4.975e-03, stats=3 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.router.load_cv` | router_stats | L4 | inconclusive | signed_mean_max=1.304e-03, mare_max=8.842e-03, max_step_max=6.118e-02, stats=3 | 介于噪声参照一到两倍之间，或参照不足 |
| `L4.health` | health_signals | L4 | pass | fields=3, problems=0 | 健康 |
| `L2.dp.replica` | replica_consistency | L2 | pass | objects=41, inconsistent=0 | 41 个对象副本全部一致 |
| `SELF.poison.tensor_tolerance` | poison_selftest | L3 | pass | cases=5 | 干净数据通过、每种注入都被发现 |

数字全部来自 `results.json`，本报告不手写数字。
