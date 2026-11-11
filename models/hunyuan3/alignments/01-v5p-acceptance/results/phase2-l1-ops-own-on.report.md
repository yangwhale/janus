# janus 报告：hunyuan3-4l / phase2-l1-ops-own-on

- 被测：`r-l2-on`；golden：`g1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/analysis/fwd_rratio.py`（sha256 c02387df92a0…）
- 结果：pass 9、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L1.ops.op.qkv` | op_replay | L1 | pass | rel_l2=1.890e-03, cr_match_frac=6.592e-01, ulp_max=2348, ulp_p999=1.050e+02 | 全部在门槛内 |
| `L1.ops.op.qk_norm` | op_replay | L1 | pass | rel_l2=1.769e-03, cr_match_frac=9.143e-01, ulp_max=1, ulp_p999=1.000e+00 | 全部在门槛内 |
| `L1.ops.op.attn_proj` | op_replay | L1 | pass | rel_l2=1.899e-03, cr_match_frac=9.998e-01, ulp_max=12245, ulp_p999=0.000e+00 | 全部在门槛内 |
| `L1.ops.op.mlp_norm` | op_replay | L1 | pass | rel_l2=1.924e-03, cr_match_frac=8.238e-01, ulp_max=1, ulp_p999=1.000e+00 | 全部在门槛内 |
| `L1.ops.op.router_gemm` | op_replay | L1 | pass | rel_l2=5.726e-08, cr_match_frac=4.997e-01, ulp_max=5442, ulp_p999=7.000e+00 | 全部在门槛内 |
| `L1.ops.op.dense_fc1` | op_replay | L1 | pass | rel_l2=2.225e-03, cr_match_frac=6.539e-01, ulp_max=13653, ulp_p999=7.900e+01 | 全部在门槛内 |
| `L1.ops.op.dense_act_fc2` | op_replay | L1 | pass | rel_l2=2.113e-03, cr_match_frac=5.326e-01, ulp_max=2121, ulp_p999=1.520e+02 | 全部在门槛内 |
| `L1.ops.kernel.attn_core` | op_replay | L1 | pass | rel_l2=3.751e-03, cr_match_frac=5.442e-01, ulp_max=14062, ulp_p999=1.430e+02 | 全部在门槛内 |
| `L1.ops.kernel.moe` | op_replay | L1 | pass | rel_l2=3.677e-03, cr_match_frac=4.048e-01, ulp_max=14922, ulp_p999=2.740e+02 | 全部在门槛内 |

数字全部来自 `results.json`，本报告不手写数字。
