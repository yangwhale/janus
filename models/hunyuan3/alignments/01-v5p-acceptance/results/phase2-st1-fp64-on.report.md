# janus 报告：hunyuan3-4l / phase2-st1-fp64-on

- 被测：`t-st1n-base-gm`；golden：`g1k-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/analysis/fwd_rratio.py`（sha256 c02387df92a0…）
- 结果：pass 1、fail 7、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L2.layer0.grads` | op_replay | L2 | fail | rel_l2=1.272e-03, cr_match_frac=0.000e+00, ulp_max=923099980, ulp_p999=1.134e+07 | 超线：['r_ratio'] |
| `L2.layer1.grads` | op_replay | L2 | fail | rel_l2=3.827e-02, cr_match_frac=0.000e+00, ulp_max=938717941, ulp_p999=5.636e+07 | 超线：['r_ratio'] |
| `L2.layer2.grads` | op_replay | L2 | fail | rel_l2=6.168e-02, cr_match_frac=0.000e+00, ulp_max=926109388, ulp_p999=6.037e+07 | 超线：['r_ratio'] |
| `L2.layer3.grads` | op_replay | L2 | fail | rel_l2=8.421e-02, cr_match_frac=0.000e+00, ulp_max=889743488, ulp_p999=6.526e+07 | 超线：['r_ratio'] |
| `L3.step1.grads.all` | op_replay | L3 | fail | rel_l2=8.421e-02, cr_match_frac=0.000e+00, ulp_max=938717941, ulp_p999=6.526e+07 | 超线：['r_ratio'] |
| `L3.step1.updated` | op_replay | L3 | fail | rel_l2=1.348e-01, cr_match_frac=0.000e+00, ulp_max=4531985161360065024, ulp_p999=4.446e+18 | 超线：['r_ratio_own'] |
| `L3.step1.optim_state` | op_replay | L3 | fail | rel_l2=8.421e-02, cr_match_frac=0.000e+00, ulp_max=875682889, ulp_p999=1.418e+08 | 超线：['r_ratio_own'] |
| `L3.step1.bias_after` | op_replay | L3 | pass | rel_l2=1.443e-01, cr_match_frac=0.000e+00, ulp_max=9345848836, ulp_p999=9.059e+09 | 全部在门槛内 |

数字全部来自 `results.json`，本报告不手写数字。
