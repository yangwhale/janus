# janus 报告：hunyuan3-4l / phase2-v7x-l1-ops-own

- 被测：`r-v7x-l2`；golden：`g1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/analysis/fwd_rratio.py`（sha256 c02387df92a0…）
- 结果：pass 7、fail 2、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L1.ops.op.qkv` | op_replay | L1 | fail | rel_l2=2.946e-03, cr_match_frac=4.861e-01, ulp_max=14724, ulp_p999=1.860e+02 | 超线：['r_ratio_own_inputs'] |
| `L1.ops.op.qk_norm` | op_replay | L1 | pass | rel_l2=1.685e-03, cr_match_frac=1.000e+00, ulp_max=1, ulp_p999=0.000e+00 | 全部在门槛内 |
| `L1.ops.op.attn_proj` | op_replay | L1 | pass | rel_l2=1.901e-03, cr_match_frac=9.999e-01, ulp_max=12741, ulp_p999=0.000e+00 | 全部在门槛内 |
| `L1.ops.op.mlp_norm` | op_replay | L1 | pass | rel_l2=1.923e-03, cr_match_frac=8.239e-01, ulp_max=1, ulp_p999=1.000e+00 | 全部在门槛内 |
| `L1.ops.op.router_gemm` | op_replay | L1 | pass | rel_l2=5.557e-08, cr_match_frac=5.149e-01, ulp_max=10188, ulp_p999=7.000e+00 | 全部在门槛内 |
| `L1.ops.op.dense_fc1` | op_replay | L1 | fail | rel_l2=3.152e-03, cr_match_frac=5.133e-01, ulp_max=14974, ulp_p999=1.250e+02 | 超线：['r_ratio_own_inputs'] |
| `L1.ops.op.dense_act_fc2` | op_replay | L1 | pass | rel_l2=1.907e-03, cr_match_frac=6.109e-01, ulp_max=1923, ulp_p999=1.210e+02 | 全部在门槛内 |
| `L1.ops.kernel.attn_core` | op_replay | L1 | pass | rel_l2=3.744e-03, cr_match_frac=5.448e-01, ulp_max=13741, ulp_p999=1.430e+02 | 全部在门槛内 |
| `L1.ops.kernel.moe` | op_replay | L1 | pass | rel_l2=4.093e-03, cr_match_frac=3.621e-01, ulp_max=14782, ulp_p999=3.120e+02 | 全部在门槛内 |

数字全部来自 `results.json`，本报告不手写数字。
