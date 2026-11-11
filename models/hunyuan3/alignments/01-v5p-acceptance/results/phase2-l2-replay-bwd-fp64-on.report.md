# janus 报告：hunyuan3-4l / phase2-l2-replay-bwd-fp64-on

- 被测：`r-l2-on`；golden：`g1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/analysis/fwd_rratio.py`（sha256 c02387df92a0…）
- 结果：pass 4、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L2.layer0.bwd` | op_replay | L2 | pass | rel_l2=2.508e-03, cr_match_frac=6.752e-01, ulp_max=2073, ulp_p999=5.100e+01 | 全部在门槛内 |
| `L2.layer1.bwd` | op_replay | L2 | pass | rel_l2=4.482e-03, cr_match_frac=6.177e-01, ulp_max=1543, ulp_p999=9.500e+01 | 全部在门槛内 |
| `L2.layer2.bwd` | op_replay | L2 | pass | rel_l2=4.355e-03, cr_match_frac=5.850e-01, ulp_max=1967, ulp_p999=1.140e+02 | 全部在门槛内 |
| `L2.layer3.bwd` | op_replay | L2 | pass | rel_l2=5.832e-03, cr_match_frac=5.399e-01, ulp_max=1818, ulp_p999=1.480e+02 | 全部在门槛内 |

数字全部来自 `results.json`，本报告不手写数字。
