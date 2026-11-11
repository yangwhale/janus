# janus 报告：hunyuan3-4l / phase1-parallel

- 被测：`p-ep4`；golden：`d-a`
- janus 0.0.1；目录 `catalogs/hunyuan3-4l/phase1-parallel.yaml`（sha256 9558dabf0405…）
- 结果：pass 6、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L0.data.fingerprint` | data_fingerprint | L0 | pass | first_loss={'test': 11.4657564163208, 'ref': 11.4657564163208}, units=195, differ=0 | token 与首步 loss 逐位相同 |
| `L2.parallel.fwd.bitwise` | bitwise_equal | L2 | pass | tensors=43, differ=0 | 全部逐位相同 |
| `L2.parallel.fwd.ulp` | ulp_distribution | L2 | pass | zero_frac_min=1.000e+00, p999_max=0.000e+00, max_max=0, cross_zero=0 | 诊断：只报分布，不判 |
| `L2.parallel.fwd.tol` | tensor_tolerance | L2 | pass | rel_l2_max=0.000e+00, cosine_min=1.000e+00, max_abs_max=0.000e+00, tensors=43 | 全部在门槛内 |
| `L2.parallel.router` | discrete_agreement | L2 | pass | set_frac_min=1.000e+00, order_frac_min=1.000e+00 | set_frac 最低 1.000000 |
| `L3.parallel.loss` | tensor_tolerance | L3 | pass | rel_l2_max=0.000e+00, max_abs_max=0.000e+00, tensors=1 | 全部在门槛内 |

数字全部来自 `results.json`，本报告不手写数字。
