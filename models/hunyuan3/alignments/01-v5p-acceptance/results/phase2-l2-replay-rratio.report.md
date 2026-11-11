# janus 报告：hunyuan3-4l / phase2-l2-replay-rratio

- 被测：`r-l2-norm-perop`；golden：`g1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/analysis/fwd_rratio.py`（sha256 c02387df92a0…）
- 结果：pass 6、fail 4、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L2.layer0.fwd` | op_replay | L2 | fail | rel_l2=3.855e-03, cr_match_frac=3.139e-01, ulp_max=14850, ulp_p999=1.393e+04 | 超线：['r_ratio'] |
| `L2.layer1.fwd` | op_replay | L2 | fail | rel_l2=1.461e-02, cr_match_frac=3.306e-05, ulp_max=31682495, ulp_p999=3.314e+05 | 超线：['r_ratio'] |
| `L2.layer1.router` | discrete_agreement | L2 | pass | set_frac_min=9.534e-01, order_frac_min=9.534e-01, gpu_vs_fp64_set_frac=9.517e-01 | set_frac 最低 0.953369 |
| `L1.router.topk.fp64.layer1` | discrete_agreement | L1 | pass | set_frac_min=9.998e-01, order_frac_min=9.998e-01 | set_frac 最低 0.999756 |
| `L2.layer2.fwd` | op_replay | L2 | fail | rel_l2=1.966e-02, cr_match_frac=4.959e-05, ulp_max=26850207, ulp_p999=1.338e+05 | 超线：['r_ratio'] |
| `L2.layer2.router` | discrete_agreement | L2 | pass | set_frac_min=9.683e-01, order_frac_min=9.683e-01, gpu_vs_fp64_set_frac=9.656e-01 | set_frac 最低 0.968262 |
| `L1.router.topk.fp64.layer2` | discrete_agreement | L1 | pass | set_frac_min=1.000e+00, order_frac_min=1.000e+00 | set_frac 最低 1.000000 |
| `L2.layer3.fwd` | op_replay | L2 | fail | rel_l2=2.650e-02, cr_match_frac=5.976e-05, ulp_max=58744249, ulp_p999=1.239e+05 | 超线：['r_ratio'] |
| `L2.layer3.router` | discrete_agreement | L2 | pass | set_frac_min=9.670e-01, order_frac_min=9.670e-01, gpu_vs_fp64_set_frac=9.707e-01 | set_frac 最低 0.967041 |
| `L1.router.topk.fp64.layer3` | discrete_agreement | L1 | pass | set_frac_min=1.000e+00, order_frac_min=1.000e+00 | set_frac 最低 1.000000 |

数字全部来自 `results.json`，本报告不手写数字。
