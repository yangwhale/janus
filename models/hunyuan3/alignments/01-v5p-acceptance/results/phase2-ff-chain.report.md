# janus 报告：hunyuan3-4l / phase2-ff-chain

- 被测：`r-ff-prod`；golden：`g1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/analysis/fwd_rratio.py`（sha256 c02387df92a0…）
- 结果：pass 2、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L3.step1.logits` | op_replay | L3 | pass | rel_l2=1.047e-02, cr_match_frac=2.064e-01, ulp_max=15676, ulp_p999=3.420e+02 | 全部在门槛内 |
| `L3.step1.loss.per_token` | op_replay | L3 | pass | rel_l2=1.156e-03, cr_match_frac=0.000e+00, ulp_max=171713, ulp_p999=1.307e+05 | 全部在门槛内 |

数字全部来自 `results.json`，本报告不手写数字。
