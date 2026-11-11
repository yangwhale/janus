# janus 报告：hunyuan4-mini / gpu-scale-vs-single-node

- 被测：`mn-b`；golden：`g3-a`
- janus 0.1.0.dev0；目录 `models/hunyuan4/alignments/05-gpu-scale/catalogs/vs-single-node.yaml`（内容指纹 debcff08c0c5…）
- 结果：pass 8、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `C.loss` | loss_series | L4 | pass | d0=-1.628e-06, signed_mean=-1.724e-06, mare=5.399e-06, max_step=2.279e-05 | 在噪声参照内 |
| `C.loss.tail` | loss_series | L4 | pass | d0=-1.628e-06, signed_mean=-2.030e-06, mare=6.811e-06, max_step=2.279e-05 | 在噪声参照内 |
| `C.mtp_loss` | loss_series | L4 | pass | d0=5.089e-07, signed_mean=-6.526e-06, mare=1.382e-05, max_step=1.175e-04 | 在噪声参照内 |
| `C.indexer_loss` | loss_series | L4 | pass | d0=4.397e-06, signed_mean=2.806e-05, mare=2.062e-04, max_step=9.299e-04 | 在噪声参照内 |
| `C.grad_norm` | loss_series | L4 | pass | d0=-6.269e-05, signed_mean=4.308e-05, mare=3.123e-04, max_step=1.458e-03 | 在噪声参照内 |
| `C.param_norm` | loss_series | L4 | pass | d0=3.881e-09, signed_mean=3.212e-08, mare=4.318e-08, max_step=2.224e-07 | 在噪声参照内 |
| `C.router.bias_mean` | router_stats | L4 | pass | signed_mean_max=1.570e-01, mare_max=1.587e-01, max_step_max=1.346e+00, stats=7 | 在噪声参照内 |
| `C.router.bias_std` | router_stats | L4 | pass | signed_mean_max=1.319e-03, mare_max=4.439e-03, max_step_max=2.508e-02, stats=7 | 在噪声参照内 |

数字全部来自 `results.json`，本报告不手写数字。
