# janus 报告：hunyuan3-4l / phase2-v7x-l2-replay-bwd-fp64-on

- 被测：`r-v7x-l2-on`；golden：`g1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/analysis/fwd_rratio.py`（sha256 c02387df92a0…）
- 结果：pass 4、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L2.layer0.bwd` | op_replay | L2 | pass | rel_l2=2.513e-03, cr_match_frac=6.753e-01, ulp_max=13317, ulp_p999=5.100e+01 | 全部在门槛内 |
| `L2.layer1.bwd` | op_replay | L2 | pass | rel_l2=5.763e-03, cr_match_frac=6.166e-01, ulp_max=1443, ulp_p999=9.600e+01 | 全部在门槛内 |
| `L2.layer2.bwd` | op_replay | L2 | pass | rel_l2=4.594e-03, cr_match_frac=5.838e-01, ulp_max=1923, ulp_p999=1.150e+02 | 全部在门槛内 |
| `L2.layer3.bwd` | op_replay | L2 | pass | rel_l2=5.516e-03, cr_match_frac=5.425e-01, ulp_max=12399, ulp_p999=1.440e+02 | 全部在门槛内 |

数字全部来自 `results.json`，本报告不手写数字。
