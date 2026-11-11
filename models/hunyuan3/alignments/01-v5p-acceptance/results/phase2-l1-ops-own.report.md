# janus 报告：hunyuan3-4l / phase2-l1-ops-own

- 被测：`r-l2-norm-perop`；golden：`g1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/analysis/fwd_rratio.py`（sha256 c02387df92a0…）
- 结果：pass 7、fail 2、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L1.ops.op.qkv` | op_replay | L1 | fail | rel_l2=2.335e-03, cr_match_frac=5.677e-01, ulp_max=2438, ulp_p999=1.240e+02 | 超线：['r_ratio_own_inputs'] |
| `L1.ops.op.qk_norm` | op_replay | L1 | pass | rel_l2=1.685e-03, cr_match_frac=1.000e+00, ulp_max=1, ulp_p999=0.000e+00 | 全部在门槛内 |
| `L1.ops.op.attn_proj` | op_replay | L1 | pass | rel_l2=1.899e-03, cr_match_frac=9.998e-01, ulp_max=12684, ulp_p999=0.000e+00 | 全部在门槛内 |
| `L1.ops.op.mlp_norm` | op_replay | L1 | pass | rel_l2=1.923e-03, cr_match_frac=8.238e-01, ulp_max=1, ulp_p999=1.000e+00 | 全部在门槛内 |
| `L1.ops.op.router_gemm` | op_replay | L1 | pass | rel_l2=5.717e-08, cr_match_frac=4.998e-01, ulp_max=19863, ulp_p999=7.000e+00 | 全部在门槛内 |
| `L1.ops.op.dense_fc1` | op_replay | L1 | fail | rel_l2=2.577e-03, cr_match_frac=5.888e-01, ulp_max=2175, ulp_p999=9.700e+01 | 超线：['r_ratio_own_inputs'] |
| `L1.ops.op.dense_act_fc2` | op_replay | L1 | pass | rel_l2=1.896e-03, cr_match_frac=6.108e-01, ulp_max=2160, ulp_p999=1.200e+02 | 全部在门槛内 |
| `L1.ops.kernel.attn_core` | op_replay | L1 | pass | rel_l2=3.741e-03, cr_match_frac=5.446e-01, ulp_max=14164, ulp_p999=1.420e+02 | 全部在门槛内 |
| `L1.ops.kernel.moe` | op_replay | L1 | pass | rel_l2=3.792e-03, cr_match_frac=3.898e-01, ulp_max=14674, ulp_p999=2.960e+02 | 全部在门槛内 |

数字全部来自 `results.json`，本报告不手写数字。
