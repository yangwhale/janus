# janus 报告：hunyuan3-4l / phase2-v7x-l1-ops-own-on

- 被测：`r-v7x-l2-on`；golden：`g1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/analysis/fwd_rratio.py`（sha256 c02387df92a0…）
- 结果：pass 8、fail 1、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L1.ops.op.qkv` | op_replay | L1 | pass | rel_l2=1.890e-03, cr_match_frac=6.592e-01, ulp_max=2348, ulp_p999=1.050e+02 | 全部在门槛内 |
| `L1.ops.op.qk_norm` | op_replay | L1 | pass | rel_l2=1.769e-03, cr_match_frac=9.143e-01, ulp_max=1, ulp_p999=1.000e+00 | 全部在门槛内 |
| `L1.ops.op.attn_proj` | op_replay | L1 | pass | rel_l2=1.899e-03, cr_match_frac=9.999e-01, ulp_max=148, ulp_p999=0.000e+00 | 全部在门槛内 |
| `L1.ops.op.mlp_norm` | op_replay | L1 | pass | rel_l2=1.923e-03, cr_match_frac=8.239e-01, ulp_max=1, ulp_p999=1.000e+00 | 全部在门槛内 |
| `L1.ops.op.router_gemm` | op_replay | L1 | pass | rel_l2=5.549e-08, cr_match_frac=5.137e-01, ulp_max=25656, ulp_p999=7.000e+00 | 全部在门槛内 |
| `L1.ops.op.dense_fc1` | op_replay | L1 | pass | rel_l2=2.226e-03, cr_match_frac=6.537e-01, ulp_max=14374, ulp_p999=7.900e+01 | 全部在门槛内 |
| `L1.ops.op.dense_act_fc2` | op_replay | L1 | pass | rel_l2=2.112e-03, cr_match_frac=5.327e-01, ulp_max=1914, ulp_p999=1.530e+02 | 全部在门槛内 |
| `L1.ops.kernel.attn_core` | op_replay | L1 | pass | rel_l2=3.719e-03, cr_match_frac=5.447e-01, ulp_max=13803, ulp_p999=1.430e+02 | 全部在门槛内 |
| `L1.ops.kernel.moe` | op_replay | L1 | fail | rel_l2=4.618e-03, cr_match_frac=3.874e-01, ulp_max=14769, ulp_p999=2.760e+02 | 超线：['r_ratio_own_inputs'] |

数字全部来自 `results.json`，本报告不手写数字。
