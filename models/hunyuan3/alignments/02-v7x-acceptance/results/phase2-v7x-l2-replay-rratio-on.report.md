# janus 报告：hunyuan3-4l / phase2-v7x-l2-replay-rratio-on

- 被测：`r-v7x-l2-on`；golden：`g1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/analysis/fwd_rratio.py`（sha256 c02387df92a0…）
- 结果：pass 10、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L2.layer0.fwd` | op_replay | L2 | pass | rel_l2=3.625e-03, cr_match_frac=3.393e-01, ulp_max=14910, ulp_p999=1.392e+04 | 全部在门槛内 |
| `L2.layer1.fwd` | op_replay | L2 | pass | rel_l2=1.541e-02, cr_match_frac=3.052e-05, ulp_max=72442256, ulp_p999=3.730e+05 | 全部在门槛内 |
| `L2.layer1.router` | discrete_agreement | L2 | pass | set_frac_min=9.543e-01, order_frac_min=9.543e-01, gpu_vs_fp64_set_frac=9.517e-01 | set_frac 最低 0.954346 |
| `L1.router.topk.fp64.layer1` | discrete_agreement | L1 | pass | set_frac_min=1.000e+00, order_frac_min=1.000e+00 | set_frac 最低 1.000000 |
| `L2.layer2.fwd` | op_replay | L2 | pass | rel_l2=1.972e-02, cr_match_frac=4.832e-05, ulp_max=17122253, ulp_p999=1.373e+05 | 全部在门槛内 |
| `L2.layer2.router` | discrete_agreement | L2 | pass | set_frac_min=9.688e-01, order_frac_min=9.688e-01, gpu_vs_fp64_set_frac=9.656e-01 | set_frac 最低 0.968750 |
| `L1.router.topk.fp64.layer2` | discrete_agreement | L1 | pass | set_frac_min=9.998e-01, order_frac_min=9.998e-01 | set_frac 最低 0.999756 |
| `L2.layer3.fwd` | op_replay | L2 | pass | rel_l2=2.443e-02, cr_match_frac=6.994e-05, ulp_max=55921721, ulp_p999=1.289e+05 | 全部在门槛内 |
| `L2.layer3.router` | discrete_agreement | L2 | pass | set_frac_min=9.692e-01, order_frac_min=9.692e-01, gpu_vs_fp64_set_frac=9.707e-01 | set_frac 最低 0.969238 |
| `L1.router.topk.fp64.layer3` | discrete_agreement | L1 | pass | set_frac_min=1.000e+00, order_frac_min=1.000e+00 | set_frac 最低 1.000000 |

数字全部来自 `results.json`，本报告不手写数字。
