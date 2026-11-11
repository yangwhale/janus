# janus 报告：hunyuan3-4l / phase2-l1-ops-wcoff

- 被测：`r-wc-off-r4`；golden：`g1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/analysis/fwd_rratio.py`（sha256 c02387df92a0…）
- 结果：pass 8、fail 1、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L1.ops.op.qkv` | op_replay | L1 | pass | rel_l2=1.890e-03, cr_match_frac=6.592e-01, ulp_max=13844, ulp_p999=1.050e+02 | 全部在门槛内 |
| `L1.ops.op.qk_norm` | op_replay | L1 | pass | rel_l2=1.769e-03, cr_match_frac=9.143e-01, ulp_max=1, ulp_p999=1.000e+00 | 全部在门槛内 |
| `L1.ops.op.attn_proj` | op_replay | L1 | pass | rel_l2=1.899e-03, cr_match_frac=9.998e-01, ulp_max=12383, ulp_p999=0.000e+00 | 全部在门槛内 |
| `L1.ops.op.mlp_norm` | op_replay | L1 | fail | rel_l2=2.735e-03, cr_match_frac=6.876e-01, ulp_max=14900, ulp_p999=5.600e+01 | 超线：['r_ratio_own_inputs'] |
| `L1.ops.op.router_gemm` | op_replay | L1 | pass | rel_l2=8.251e-08, cr_match_frac=3.915e-01, ulp_max=7947, ulp_p999=7.000e+00 | 全部在门槛内 |
| `L1.ops.op.dense_fc1` | op_replay | L1 | pass | rel_l2=2.199e-03, cr_match_frac=6.590e-01, ulp_max=2093, ulp_p999=7.900e+01 | 全部在门槛内 |
| `L1.ops.op.dense_act_fc2` | op_replay | L1 | pass | rel_l2=1.903e-03, cr_match_frac=6.109e-01, ulp_max=2050, ulp_p999=1.210e+02 | 全部在门槛内 |
| `L1.ops.kernel.attn_core` | op_replay | L1 | pass | rel_l2=3.751e-03, cr_match_frac=5.442e-01, ulp_max=14062, ulp_p999=1.430e+02 | 全部在门槛内 |
| `L1.ops.kernel.moe` | op_replay | L1 | pass | rel_l2=3.694e-03, cr_match_frac=4.051e-01, ulp_max=14208, ulp_p999=1.920e+02 | 全部在门槛内 |

数字全部来自 `results.json`，本报告不手写数字。
