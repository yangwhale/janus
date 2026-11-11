# janus 报告：hunyuan3-4l / phase2-l1-ops-on

- 被测：`r-l2-on`；golden：`g1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/analysis/l1_replay_results.py`（sha256 116c9323cf9b…）
- 结果：pass 7、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L1.ops.router.gemm` | op_replay | L1 | pass | rel_l2=5.720e-08, cr_match_frac=5.002e-01, ulp_max=40288, ulp_p999=8.000e+00 | 全部在门槛内 |
| `L1.ops.router.topk` | op_replay | L1 | pass | expect_match_frac=1.000e+00, cases=3 | 逐位全中（expect_match_frac） |
| `L1.ops.router.precision` | op_replay | L1 | pass | bit_match_frac=1.000e+00, cases=3 | 逐位全中（bit_match_frac） |
| `L1.ops.rmsnorm` | op_replay | L1 | pass | rel_l2=1.670e-03, cr_match_frac=1.000e+00, ulp_max=1, ulp_p999=0.000e+00 | 全部在门槛内 |
| `L1.ops.act` | op_replay | L1 | pass | rel_l2=2.347e-03, cr_match_frac=7.372e-01, ulp_max=1, ulp_p999=1.000e+00 | 逐位全中（emulate_match_frac） |
| `L1.ops.attn.core` | op_replay | L1 | pass | rel_l2=3.371e-03, cr_match_frac=5.884e-01, ulp_max=13888, ulp_p999=1.250e+02 | 全部在门槛内 |
| `L1.ops.moe` | op_replay | L1 | pass | rel_l2=3.764e-03, cr_match_frac=4.044e-01, ulp_max=14616, ulp_p999=2.670e+02 | 全部在门槛内 |

数字全部来自 `results.json`，本报告不手写数字。
