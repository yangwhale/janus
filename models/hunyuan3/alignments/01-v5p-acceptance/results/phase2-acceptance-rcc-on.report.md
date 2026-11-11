# janus 报告：hunyuan3-4l / phase2-acceptance-rcc-on

- 被测：`t-rmap-on-1000`；golden：`rcc-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/01-b200x8-to-v5p8-4layers/catalogs/phase2-acceptance-rcc-on.yaml`（sha256 2c0dc6e291e5…）
- 结果：pass 7、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.loss` | multi_golden | L4 | pass | d0_max=1.031e-05, signed_mean_max=3.848e-05, mare_max=1.098e-04, max_step_max=5.901e-04 | 在噪声参照内 |
| `L4.loss.tail` | multi_golden | L4 | pass | d0_max=1.031e-05, signed_mean_max=3.341e-05, mare_max=1.154e-04, max_step_max=4.712e-04 | 在噪声参照内 |
| `L4.grad_norm` | loss_series | L4 | pass | d0=-2.238e-04, signed_mean=1.123e-03, mare=2.796e-02, max_step=2.721e-01 | 在噪声参照内 |
| `L4.grad_norm.golden_den` | loss_series | L4 | pass | d0=-2.238e-04, signed_mean=1.983e-03, mare=2.801e-02, max_step=2.841e-01 | 在噪声参照内 |
| `L4.param_norm` | loss_series | L4 | pass | d0=-1.309e-07, signed_mean=1.593e-08, mare=5.427e-08, max_step=2.443e-07 | 在噪声参照内 |
| `L4.param_norm.raw` | loss_series | L4 | pass | d0=5.615e-08, signed_mean=5.184e-08, mare=8.221e-08, max_step=3.554e-07 | 在噪声参照内 |
| `L4.lr` | scalar_series | L4 | pass | steps=1000, bad_steps=0, first_bad=None | 逐步在容差内 |

数字全部来自 `results.json`，本报告不手写数字。
