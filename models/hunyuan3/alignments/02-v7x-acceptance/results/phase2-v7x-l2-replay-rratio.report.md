# janus 报告：hunyuan3-4l / phase2-v7x-l2-replay-rratio

- 被测：`r-v7x-l2`；golden：`g1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/analysis/fwd_rratio.py`（sha256 c02387df92a0…）
- 结果：pass 6、fail 4、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L2.layer0.fwd` | op_replay | L2 | fail | rel_l2=4.047e-03, cr_match_frac=2.985e-01, ulp_max=15330, ulp_p999=1.391e+04 | 超线：['r_ratio'] |
| `L2.layer1.fwd` | op_replay | L2 | fail | rel_l2=1.555e-02, cr_match_frac=2.416e-05, ulp_max=28931666, ulp_p999=3.580e+05 | 超线：['r_ratio'] |
| `L2.layer1.router` | discrete_agreement | L2 | pass | set_frac_min=9.526e-01, order_frac_min=9.526e-01, gpu_vs_fp64_set_frac=9.517e-01 | set_frac 最低 0.952637 |
| `L1.router.topk.fp64.layer1` | discrete_agreement | L1 | pass | set_frac_min=1.000e+00, order_frac_min=1.000e+00 | set_frac 最低 1.000000 |
| `L2.layer2.fwd` | op_replay | L2 | fail | rel_l2=1.897e-02, cr_match_frac=4.069e-05, ulp_max=56500809, ulp_p999=1.367e+05 | 超线：['r_ratio'] |
| `L2.layer2.router` | discrete_agreement | L2 | pass | set_frac_min=9.697e-01, order_frac_min=9.697e-01, gpu_vs_fp64_set_frac=9.656e-01 | set_frac 最低 0.969727 |
| `L1.router.topk.fp64.layer2` | discrete_agreement | L1 | pass | set_frac_min=9.998e-01, order_frac_min=9.998e-01 | set_frac 最低 0.999756 |
| `L2.layer3.fwd` | op_replay | L2 | fail | rel_l2=2.481e-02, cr_match_frac=4.578e-05, ulp_max=79747753, ulp_p999=1.424e+05 | 超线：['r_ratio'] |
| `L2.layer3.router` | discrete_agreement | L2 | pass | set_frac_min=9.714e-01, order_frac_min=9.714e-01, gpu_vs_fp64_set_frac=9.707e-01 | set_frac 最低 0.971436 |
| `L1.router.topk.fp64.layer3` | discrete_agreement | L1 | pass | set_frac_min=1.000e+00, order_frac_min=1.000e+00 | set_frac 最低 1.000000 |

数字全部来自 `results.json`，本报告不手写数字。
