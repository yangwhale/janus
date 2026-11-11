# janus 报告：hunyuan3-4l / phase1-moe-stages

- 被测：`g61-obs`
- janus 0.1.0.dev0；目录 `models/hunyuan3/frameworks/megatron/moe_stages.py`（sha256 cda9aa3183ec…）
- 结果：pass 5、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L2.moe.expert_weights` | op_replay | L2 | pass | compared=1152, mismatch=0 | HF 原件转成 Megatron 布局后 1152 个专家权重与训练加载时的 sha256 全部相同 |
| `L2.moe.stages.chain` | op_replay | L2 | pass | rel_l2=4.033e-03, cr_match_frac=1.780e-05, ulp_max=472864, ulp_p999=2.950e+02 | 诊断：只报数，不判 |
| `L2.moe.stages.local` | op_replay | L2 | pass | rel_l2=3.240e-03, cr_match_frac=1.780e-05, ulp_max=472864, ulp_p999=1.980e+02 | 诊断：只报数，不判 |
| `L2.moe.movement` | op_replay | L2 | pass | bit_match_frac=1.000e+00, cases=12 | 逐位全中（bit_match_frac） |
| `L2.moe.topk.fp64` | op_replay | L2 | pass | expect_match_frac=1.000e+00, cases=3 | 诊断：只报数，不判 |


## 环境与开关（进程内实测）

- `torch`：2.14.0a0+b2c75dd062.nv26.09
- `transformer_engine`：2.20.2+44f31bbe
- `cuda`：13.4
- `cublas`：13.8.0.4
- `device`：NVIDIA B200
- `allow_tf32_matmul_default`：True
- `allow_tf32_cudnn_default`：True
- `float32_matmul_precision_default`：high
- `env`：{'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': '1', 'NVIDIA_TF32_OVERRIDE': None, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'NVTE_ALLOW_NONDETERMINISTIC_ALGO': '0'}

## 逐用例明细

列：stage 阶段；rel_l2 对 FP64（`L2.moe.stages.chain` 对链式、`.local` 对本阶段）；cr_match_frac 对 FP64 正确舍入到输出 dtype 的逐位相同率；bit_match_frac 搬运检查的逐位相同率；expect_match_frac top-k 选中集合与 FP64 链式一致的行比例。

### `L2.moe.expert_weights`（pass）

| 用例 | match |
|---|---|
| `layers.1.experts.0.linear_fc1` | True |
| `layers.1.experts.0.linear_fc2` | True |
| `layers.1.experts.1.linear_fc1` | True |
| `layers.1.experts.1.linear_fc2` | True |
| `layers.1.experts.2.linear_fc1` | True |
| `layers.1.experts.2.linear_fc2` | True |
| `layers.1.experts.3.linear_fc1` | True |
| `layers.1.experts.3.linear_fc2` | True |
| `layers.1.experts.4.linear_fc1` | True |
| `layers.1.experts.4.linear_fc2` | True |
| `layers.1.experts.5.linear_fc1` | True |
| `layers.1.experts.5.linear_fc2` | True |
| `layers.1.experts.6.linear_fc1` | True |
| `layers.1.experts.6.linear_fc2` | True |
| `layers.1.experts.7.linear_fc1` | True |
| `layers.1.experts.7.linear_fc2` | True |
| `layers.1.experts.8.linear_fc1` | True |
| `layers.1.experts.8.linear_fc2` | True |
| `layers.1.experts.9.linear_fc1` | True |
| `layers.1.experts.9.linear_fc2` | True |
| `layers.1.experts.10.linear_fc1` | True |
| `layers.1.experts.10.linear_fc2` | True |
| `layers.1.experts.11.linear_fc1` | True |
| `layers.1.experts.11.linear_fc2` | True |
| `layers.1.experts.12.linear_fc1` | True |
| `layers.1.experts.12.linear_fc2` | True |
| `layers.1.experts.13.linear_fc1` | True |
| `layers.1.experts.13.linear_fc2` | True |
| `layers.1.experts.14.linear_fc1` | True |
| `layers.1.experts.14.linear_fc2` | True |
| `layers.1.experts.15.linear_fc1` | True |
| `layers.1.experts.15.linear_fc2` | True |
| `layers.1.experts.16.linear_fc1` | True |
| `layers.1.experts.16.linear_fc2` | True |
| `layers.1.experts.17.linear_fc1` | True |
| `layers.1.experts.17.linear_fc2` | True |
| `layers.1.experts.18.linear_fc1` | True |
| `layers.1.experts.18.linear_fc2` | True |
| `layers.1.experts.19.linear_fc1` | True |
| `layers.1.experts.19.linear_fc2` | True |
| `layers.1.experts.20.linear_fc1` | True |
| `layers.1.experts.20.linear_fc2` | True |
| `layers.1.experts.21.linear_fc1` | True |
| `layers.1.experts.21.linear_fc2` | True |
| `layers.1.experts.22.linear_fc1` | True |
| `layers.1.experts.22.linear_fc2` | True |
| `layers.1.experts.23.linear_fc1` | True |
| `layers.1.experts.23.linear_fc2` | True |
| `layers.1.experts.24.linear_fc1` | True |
| `layers.1.experts.24.linear_fc2` | True |
| `layers.1.experts.25.linear_fc1` | True |
| `layers.1.experts.25.linear_fc2` | True |
| `layers.1.experts.26.linear_fc1` | True |
| `layers.1.experts.26.linear_fc2` | True |
| `layers.1.experts.27.linear_fc1` | True |
| `layers.1.experts.27.linear_fc2` | True |
| `layers.1.experts.28.linear_fc1` | True |
| `layers.1.experts.28.linear_fc2` | True |
| `layers.1.experts.29.linear_fc1` | True |
| `layers.1.experts.29.linear_fc2` | True |
| `layers.1.experts.30.linear_fc1` | True |
| `layers.1.experts.30.linear_fc2` | True |
| `layers.1.experts.31.linear_fc1` | True |
| `layers.1.experts.31.linear_fc2` | True |
| `layers.1.experts.32.linear_fc1` | True |
| `layers.1.experts.32.linear_fc2` | True |
| `layers.1.experts.33.linear_fc1` | True |
| `layers.1.experts.33.linear_fc2` | True |
| `layers.1.experts.34.linear_fc1` | True |
| `layers.1.experts.34.linear_fc2` | True |
| `layers.1.experts.35.linear_fc1` | True |
| `layers.1.experts.35.linear_fc2` | True |
| `layers.1.experts.36.linear_fc1` | True |
| `layers.1.experts.36.linear_fc2` | True |
| `layers.1.experts.37.linear_fc1` | True |
| `layers.1.experts.37.linear_fc2` | True |
| `layers.1.experts.38.linear_fc1` | True |
| `layers.1.experts.38.linear_fc2` | True |
| `layers.1.experts.39.linear_fc1` | True |
| `layers.1.experts.39.linear_fc2` | True |
| `layers.1.experts.40.linear_fc1` | True |
| `layers.1.experts.40.linear_fc2` | True |
| `layers.1.experts.41.linear_fc1` | True |
| `layers.1.experts.41.linear_fc2` | True |
| `layers.1.experts.42.linear_fc1` | True |
| `layers.1.experts.42.linear_fc2` | True |
| `layers.1.experts.43.linear_fc1` | True |
| `layers.1.experts.43.linear_fc2` | True |
| `layers.1.experts.44.linear_fc1` | True |
| `layers.1.experts.44.linear_fc2` | True |
| `layers.1.experts.45.linear_fc1` | True |
| `layers.1.experts.45.linear_fc2` | True |
| `layers.1.experts.46.linear_fc1` | True |
| `layers.1.experts.46.linear_fc2` | True |
| `layers.1.experts.47.linear_fc1` | True |
| `layers.1.experts.47.linear_fc2` | True |
| `layers.1.experts.48.linear_fc1` | True |
| `layers.1.experts.48.linear_fc2` | True |
| `layers.1.experts.49.linear_fc1` | True |
| `layers.1.experts.49.linear_fc2` | True |
| `layers.1.experts.50.linear_fc1` | True |
| `layers.1.experts.50.linear_fc2` | True |
| `layers.1.experts.51.linear_fc1` | True |
| `layers.1.experts.51.linear_fc2` | True |
| `layers.1.experts.52.linear_fc1` | True |
| `layers.1.experts.52.linear_fc2` | True |
| `layers.1.experts.53.linear_fc1` | True |
| `layers.1.experts.53.linear_fc2` | True |
| `layers.1.experts.54.linear_fc1` | True |
| `layers.1.experts.54.linear_fc2` | True |
| `layers.1.experts.55.linear_fc1` | True |
| `layers.1.experts.55.linear_fc2` | True |
| `layers.1.experts.56.linear_fc1` | True |
| `layers.1.experts.56.linear_fc2` | True |
| `layers.1.experts.57.linear_fc1` | True |
| `layers.1.experts.57.linear_fc2` | True |
| `layers.1.experts.58.linear_fc1` | True |
| `layers.1.experts.58.linear_fc2` | True |
| `layers.1.experts.59.linear_fc1` | True |
| `layers.1.experts.59.linear_fc2` | True |
| `layers.1.experts.60.linear_fc1` | True |
| `layers.1.experts.60.linear_fc2` | True |
| `layers.1.experts.61.linear_fc1` | True |
| `layers.1.experts.61.linear_fc2` | True |
| `layers.1.experts.62.linear_fc1` | True |
| `layers.1.experts.62.linear_fc2` | True |
| `layers.1.experts.63.linear_fc1` | True |
| `layers.1.experts.63.linear_fc2` | True |
| `layers.1.experts.64.linear_fc1` | True |
| `layers.1.experts.64.linear_fc2` | True |
| `layers.1.experts.65.linear_fc1` | True |
| `layers.1.experts.65.linear_fc2` | True |
| `layers.1.experts.66.linear_fc1` | True |
| `layers.1.experts.66.linear_fc2` | True |
| `layers.1.experts.67.linear_fc1` | True |
| `layers.1.experts.67.linear_fc2` | True |
| `layers.1.experts.68.linear_fc1` | True |
| `layers.1.experts.68.linear_fc2` | True |
| `layers.1.experts.69.linear_fc1` | True |
| `layers.1.experts.69.linear_fc2` | True |
| `layers.1.experts.70.linear_fc1` | True |
| `layers.1.experts.70.linear_fc2` | True |
| `layers.1.experts.71.linear_fc1` | True |
| `layers.1.experts.71.linear_fc2` | True |
| `layers.1.experts.72.linear_fc1` | True |
| `layers.1.experts.72.linear_fc2` | True |
| `layers.1.experts.73.linear_fc1` | True |
| `layers.1.experts.73.linear_fc2` | True |
| `layers.1.experts.74.linear_fc1` | True |
| `layers.1.experts.74.linear_fc2` | True |
| `layers.1.experts.75.linear_fc1` | True |
| `layers.1.experts.75.linear_fc2` | True |
| `layers.1.experts.76.linear_fc1` | True |
| `layers.1.experts.76.linear_fc2` | True |
| `layers.1.experts.77.linear_fc1` | True |
| `layers.1.experts.77.linear_fc2` | True |
| `layers.1.experts.78.linear_fc1` | True |
| `layers.1.experts.78.linear_fc2` | True |
| `layers.1.experts.79.linear_fc1` | True |
| `layers.1.experts.79.linear_fc2` | True |
| `layers.1.experts.80.linear_fc1` | True |
| `layers.1.experts.80.linear_fc2` | True |
| `layers.1.experts.81.linear_fc1` | True |
| `layers.1.experts.81.linear_fc2` | True |
| `layers.1.experts.82.linear_fc1` | True |
| `layers.1.experts.82.linear_fc2` | True |
| `layers.1.experts.83.linear_fc1` | True |
| `layers.1.experts.83.linear_fc2` | True |
| `layers.1.experts.84.linear_fc1` | True |
| `layers.1.experts.84.linear_fc2` | True |
| `layers.1.experts.85.linear_fc1` | True |
| `layers.1.experts.85.linear_fc2` | True |
| `layers.1.experts.86.linear_fc1` | True |
| `layers.1.experts.86.linear_fc2` | True |
| `layers.1.experts.87.linear_fc1` | True |
| `layers.1.experts.87.linear_fc2` | True |
| `layers.1.experts.88.linear_fc1` | True |
| `layers.1.experts.88.linear_fc2` | True |
| `layers.1.experts.89.linear_fc1` | True |
| `layers.1.experts.89.linear_fc2` | True |
| `layers.1.experts.90.linear_fc1` | True |
| `layers.1.experts.90.linear_fc2` | True |
| `layers.1.experts.91.linear_fc1` | True |
| `layers.1.experts.91.linear_fc2` | True |
| `layers.1.experts.92.linear_fc1` | True |
| `layers.1.experts.92.linear_fc2` | True |
| `layers.1.experts.93.linear_fc1` | True |
| `layers.1.experts.93.linear_fc2` | True |
| `layers.1.experts.94.linear_fc1` | True |
| `layers.1.experts.94.linear_fc2` | True |
| `layers.1.experts.95.linear_fc1` | True |
| `layers.1.experts.95.linear_fc2` | True |
| `layers.1.experts.96.linear_fc1` | True |
| `layers.1.experts.96.linear_fc2` | True |
| `layers.1.experts.97.linear_fc1` | True |
| `layers.1.experts.97.linear_fc2` | True |
| `layers.1.experts.98.linear_fc1` | True |
| `layers.1.experts.98.linear_fc2` | True |
| `layers.1.experts.99.linear_fc1` | True |
| `layers.1.experts.99.linear_fc2` | True |
| `layers.1.experts.100.linear_fc1` | True |
| `layers.1.experts.100.linear_fc2` | True |
| `layers.1.experts.101.linear_fc1` | True |
| `layers.1.experts.101.linear_fc2` | True |
| `layers.1.experts.102.linear_fc1` | True |
| `layers.1.experts.102.linear_fc2` | True |
| `layers.1.experts.103.linear_fc1` | True |
| `layers.1.experts.103.linear_fc2` | True |
| `layers.1.experts.104.linear_fc1` | True |
| `layers.1.experts.104.linear_fc2` | True |
| `layers.1.experts.105.linear_fc1` | True |
| `layers.1.experts.105.linear_fc2` | True |
| `layers.1.experts.106.linear_fc1` | True |
| `layers.1.experts.106.linear_fc2` | True |
| `layers.1.experts.107.linear_fc1` | True |
| `layers.1.experts.107.linear_fc2` | True |
| `layers.1.experts.108.linear_fc1` | True |
| `layers.1.experts.108.linear_fc2` | True |
| `layers.1.experts.109.linear_fc1` | True |
| `layers.1.experts.109.linear_fc2` | True |
| `layers.1.experts.110.linear_fc1` | True |
| `layers.1.experts.110.linear_fc2` | True |
| `layers.1.experts.111.linear_fc1` | True |
| `layers.1.experts.111.linear_fc2` | True |
| `layers.1.experts.112.linear_fc1` | True |
| `layers.1.experts.112.linear_fc2` | True |
| `layers.1.experts.113.linear_fc1` | True |
| `layers.1.experts.113.linear_fc2` | True |
| `layers.1.experts.114.linear_fc1` | True |
| `layers.1.experts.114.linear_fc2` | True |
| `layers.1.experts.115.linear_fc1` | True |
| `layers.1.experts.115.linear_fc2` | True |
| `layers.1.experts.116.linear_fc1` | True |
| `layers.1.experts.116.linear_fc2` | True |
| `layers.1.experts.117.linear_fc1` | True |
| `layers.1.experts.117.linear_fc2` | True |
| `layers.1.experts.118.linear_fc1` | True |
| `layers.1.experts.118.linear_fc2` | True |
| `layers.1.experts.119.linear_fc1` | True |
| `layers.1.experts.119.linear_fc2` | True |
| `layers.1.experts.120.linear_fc1` | True |
| `layers.1.experts.120.linear_fc2` | True |
| `layers.1.experts.121.linear_fc1` | True |
| `layers.1.experts.121.linear_fc2` | True |
| `layers.1.experts.122.linear_fc1` | True |
| `layers.1.experts.122.linear_fc2` | True |
| `layers.1.experts.123.linear_fc1` | True |
| `layers.1.experts.123.linear_fc2` | True |
| `layers.1.experts.124.linear_fc1` | True |
| `layers.1.experts.124.linear_fc2` | True |
| `layers.1.experts.125.linear_fc1` | True |
| `layers.1.experts.125.linear_fc2` | True |
| `layers.1.experts.126.linear_fc1` | True |
| `layers.1.experts.126.linear_fc2` | True |
| `layers.1.experts.127.linear_fc1` | True |
| `layers.1.experts.127.linear_fc2` | True |
| `layers.1.experts.128.linear_fc1` | True |
| `layers.1.experts.128.linear_fc2` | True |
| `layers.1.experts.129.linear_fc1` | True |
| `layers.1.experts.129.linear_fc2` | True |
| `layers.1.experts.130.linear_fc1` | True |
| `layers.1.experts.130.linear_fc2` | True |
| `layers.1.experts.131.linear_fc1` | True |
| `layers.1.experts.131.linear_fc2` | True |
| `layers.1.experts.132.linear_fc1` | True |
| `layers.1.experts.132.linear_fc2` | True |
| `layers.1.experts.133.linear_fc1` | True |
| `layers.1.experts.133.linear_fc2` | True |
| `layers.1.experts.134.linear_fc1` | True |
| `layers.1.experts.134.linear_fc2` | True |
| `layers.1.experts.135.linear_fc1` | True |
| `layers.1.experts.135.linear_fc2` | True |
| `layers.1.experts.136.linear_fc1` | True |
| `layers.1.experts.136.linear_fc2` | True |
| `layers.1.experts.137.linear_fc1` | True |
| `layers.1.experts.137.linear_fc2` | True |
| `layers.1.experts.138.linear_fc1` | True |
| `layers.1.experts.138.linear_fc2` | True |
| `layers.1.experts.139.linear_fc1` | True |
| `layers.1.experts.139.linear_fc2` | True |
| `layers.1.experts.140.linear_fc1` | True |
| `layers.1.experts.140.linear_fc2` | True |
| `layers.1.experts.141.linear_fc1` | True |
| `layers.1.experts.141.linear_fc2` | True |
| `layers.1.experts.142.linear_fc1` | True |
| `layers.1.experts.142.linear_fc2` | True |
| `layers.1.experts.143.linear_fc1` | True |
| `layers.1.experts.143.linear_fc2` | True |
| `layers.1.experts.144.linear_fc1` | True |
| `layers.1.experts.144.linear_fc2` | True |
| `layers.1.experts.145.linear_fc1` | True |
| `layers.1.experts.145.linear_fc2` | True |
| `layers.1.experts.146.linear_fc1` | True |
| `layers.1.experts.146.linear_fc2` | True |
| `layers.1.experts.147.linear_fc1` | True |
| `layers.1.experts.147.linear_fc2` | True |
| `layers.1.experts.148.linear_fc1` | True |
| `layers.1.experts.148.linear_fc2` | True |
| `layers.1.experts.149.linear_fc1` | True |
| `layers.1.experts.149.linear_fc2` | True |
| `layers.1.experts.150.linear_fc1` | True |
| `layers.1.experts.150.linear_fc2` | True |
| `layers.1.experts.151.linear_fc1` | True |
| `layers.1.experts.151.linear_fc2` | True |
| `layers.1.experts.152.linear_fc1` | True |
| `layers.1.experts.152.linear_fc2` | True |
| `layers.1.experts.153.linear_fc1` | True |
| `layers.1.experts.153.linear_fc2` | True |
| `layers.1.experts.154.linear_fc1` | True |
| `layers.1.experts.154.linear_fc2` | True |
| `layers.1.experts.155.linear_fc1` | True |
| `layers.1.experts.155.linear_fc2` | True |
| `layers.1.experts.156.linear_fc1` | True |
| `layers.1.experts.156.linear_fc2` | True |
| `layers.1.experts.157.linear_fc1` | True |
| `layers.1.experts.157.linear_fc2` | True |
| `layers.1.experts.158.linear_fc1` | True |
| `layers.1.experts.158.linear_fc2` | True |
| `layers.1.experts.159.linear_fc1` | True |
| `layers.1.experts.159.linear_fc2` | True |
| `layers.1.experts.160.linear_fc1` | True |
| `layers.1.experts.160.linear_fc2` | True |
| `layers.1.experts.161.linear_fc1` | True |
| `layers.1.experts.161.linear_fc2` | True |
| `layers.1.experts.162.linear_fc1` | True |
| `layers.1.experts.162.linear_fc2` | True |
| `layers.1.experts.163.linear_fc1` | True |
| `layers.1.experts.163.linear_fc2` | True |
| `layers.1.experts.164.linear_fc1` | True |
| `layers.1.experts.164.linear_fc2` | True |
| `layers.1.experts.165.linear_fc1` | True |
| `layers.1.experts.165.linear_fc2` | True |
| `layers.1.experts.166.linear_fc1` | True |
| `layers.1.experts.166.linear_fc2` | True |
| `layers.1.experts.167.linear_fc1` | True |
| `layers.1.experts.167.linear_fc2` | True |
| `layers.1.experts.168.linear_fc1` | True |
| `layers.1.experts.168.linear_fc2` | True |
| `layers.1.experts.169.linear_fc1` | True |
| `layers.1.experts.169.linear_fc2` | True |
| `layers.1.experts.170.linear_fc1` | True |
| `layers.1.experts.170.linear_fc2` | True |
| `layers.1.experts.171.linear_fc1` | True |
| `layers.1.experts.171.linear_fc2` | True |
| `layers.1.experts.172.linear_fc1` | True |
| `layers.1.experts.172.linear_fc2` | True |
| `layers.1.experts.173.linear_fc1` | True |
| `layers.1.experts.173.linear_fc2` | True |
| `layers.1.experts.174.linear_fc1` | True |
| `layers.1.experts.174.linear_fc2` | True |
| `layers.1.experts.175.linear_fc1` | True |
| `layers.1.experts.175.linear_fc2` | True |
| `layers.1.experts.176.linear_fc1` | True |
| `layers.1.experts.176.linear_fc2` | True |
| `layers.1.experts.177.linear_fc1` | True |
| `layers.1.experts.177.linear_fc2` | True |
| `layers.1.experts.178.linear_fc1` | True |
| `layers.1.experts.178.linear_fc2` | True |
| `layers.1.experts.179.linear_fc1` | True |
| `layers.1.experts.179.linear_fc2` | True |
| `layers.1.experts.180.linear_fc1` | True |
| `layers.1.experts.180.linear_fc2` | True |
| `layers.1.experts.181.linear_fc1` | True |
| `layers.1.experts.181.linear_fc2` | True |
| `layers.1.experts.182.linear_fc1` | True |
| `layers.1.experts.182.linear_fc2` | True |
| `layers.1.experts.183.linear_fc1` | True |
| `layers.1.experts.183.linear_fc2` | True |
| `layers.1.experts.184.linear_fc1` | True |
| `layers.1.experts.184.linear_fc2` | True |
| `layers.1.experts.185.linear_fc1` | True |
| `layers.1.experts.185.linear_fc2` | True |
| `layers.1.experts.186.linear_fc1` | True |
| `layers.1.experts.186.linear_fc2` | True |
| `layers.1.experts.187.linear_fc1` | True |
| `layers.1.experts.187.linear_fc2` | True |
| `layers.1.experts.188.linear_fc1` | True |
| `layers.1.experts.188.linear_fc2` | True |
| `layers.1.experts.189.linear_fc1` | True |
| `layers.1.experts.189.linear_fc2` | True |
| `layers.1.experts.190.linear_fc1` | True |
| `layers.1.experts.190.linear_fc2` | True |
| `layers.1.experts.191.linear_fc1` | True |
| `layers.1.experts.191.linear_fc2` | True |
| `layers.2.experts.0.linear_fc1` | True |
| `layers.2.experts.0.linear_fc2` | True |
| `layers.2.experts.1.linear_fc1` | True |
| `layers.2.experts.1.linear_fc2` | True |
| `layers.2.experts.2.linear_fc1` | True |
| `layers.2.experts.2.linear_fc2` | True |
| `layers.2.experts.3.linear_fc1` | True |
| `layers.2.experts.3.linear_fc2` | True |
| `layers.2.experts.4.linear_fc1` | True |
| `layers.2.experts.4.linear_fc2` | True |
| `layers.2.experts.5.linear_fc1` | True |
| `layers.2.experts.5.linear_fc2` | True |
| `layers.2.experts.6.linear_fc1` | True |
| `layers.2.experts.6.linear_fc2` | True |
| `layers.2.experts.7.linear_fc1` | True |
| `layers.2.experts.7.linear_fc2` | True |
| `layers.2.experts.8.linear_fc1` | True |
| `layers.2.experts.8.linear_fc2` | True |
| `layers.2.experts.9.linear_fc1` | True |
| `layers.2.experts.9.linear_fc2` | True |
| `layers.2.experts.10.linear_fc1` | True |
| `layers.2.experts.10.linear_fc2` | True |
| `layers.2.experts.11.linear_fc1` | True |
| `layers.2.experts.11.linear_fc2` | True |
| `layers.2.experts.12.linear_fc1` | True |
| `layers.2.experts.12.linear_fc2` | True |
| `layers.2.experts.13.linear_fc1` | True |
| `layers.2.experts.13.linear_fc2` | True |
| `layers.2.experts.14.linear_fc1` | True |
| `layers.2.experts.14.linear_fc2` | True |
| `layers.2.experts.15.linear_fc1` | True |
| `layers.2.experts.15.linear_fc2` | True |
| `layers.2.experts.16.linear_fc1` | True |
| `layers.2.experts.16.linear_fc2` | True |
| `layers.2.experts.17.linear_fc1` | True |
| `layers.2.experts.17.linear_fc2` | True |
| `layers.2.experts.18.linear_fc1` | True |
| `layers.2.experts.18.linear_fc2` | True |
| `layers.2.experts.19.linear_fc1` | True |
| `layers.2.experts.19.linear_fc2` | True |
| `layers.2.experts.20.linear_fc1` | True |
| `layers.2.experts.20.linear_fc2` | True |
| `layers.2.experts.21.linear_fc1` | True |
| `layers.2.experts.21.linear_fc2` | True |
| `layers.2.experts.22.linear_fc1` | True |
| `layers.2.experts.22.linear_fc2` | True |
| `layers.2.experts.23.linear_fc1` | True |
| `layers.2.experts.23.linear_fc2` | True |
| `layers.2.experts.24.linear_fc1` | True |
| `layers.2.experts.24.linear_fc2` | True |
| `layers.2.experts.25.linear_fc1` | True |
| `layers.2.experts.25.linear_fc2` | True |
| `layers.2.experts.26.linear_fc1` | True |
| `layers.2.experts.26.linear_fc2` | True |
| `layers.2.experts.27.linear_fc1` | True |
| `layers.2.experts.27.linear_fc2` | True |
| `layers.2.experts.28.linear_fc1` | True |
| `layers.2.experts.28.linear_fc2` | True |
| `layers.2.experts.29.linear_fc1` | True |
| `layers.2.experts.29.linear_fc2` | True |
| `layers.2.experts.30.linear_fc1` | True |
| `layers.2.experts.30.linear_fc2` | True |
| `layers.2.experts.31.linear_fc1` | True |
| `layers.2.experts.31.linear_fc2` | True |
| `layers.2.experts.32.linear_fc1` | True |
| `layers.2.experts.32.linear_fc2` | True |
| `layers.2.experts.33.linear_fc1` | True |
| `layers.2.experts.33.linear_fc2` | True |
| `layers.2.experts.34.linear_fc1` | True |
| `layers.2.experts.34.linear_fc2` | True |
| `layers.2.experts.35.linear_fc1` | True |
| `layers.2.experts.35.linear_fc2` | True |
| `layers.2.experts.36.linear_fc1` | True |
| `layers.2.experts.36.linear_fc2` | True |
| `layers.2.experts.37.linear_fc1` | True |
| `layers.2.experts.37.linear_fc2` | True |
| `layers.2.experts.38.linear_fc1` | True |
| `layers.2.experts.38.linear_fc2` | True |
| `layers.2.experts.39.linear_fc1` | True |
| `layers.2.experts.39.linear_fc2` | True |
| `layers.2.experts.40.linear_fc1` | True |
| `layers.2.experts.40.linear_fc2` | True |
| `layers.2.experts.41.linear_fc1` | True |
| `layers.2.experts.41.linear_fc2` | True |
| `layers.2.experts.42.linear_fc1` | True |
| `layers.2.experts.42.linear_fc2` | True |
| `layers.2.experts.43.linear_fc1` | True |
| `layers.2.experts.43.linear_fc2` | True |
| `layers.2.experts.44.linear_fc1` | True |
| `layers.2.experts.44.linear_fc2` | True |
| `layers.2.experts.45.linear_fc1` | True |
| `layers.2.experts.45.linear_fc2` | True |
| `layers.2.experts.46.linear_fc1` | True |
| `layers.2.experts.46.linear_fc2` | True |
| `layers.2.experts.47.linear_fc1` | True |
| `layers.2.experts.47.linear_fc2` | True |
| `layers.2.experts.48.linear_fc1` | True |
| `layers.2.experts.48.linear_fc2` | True |
| `layers.2.experts.49.linear_fc1` | True |
| `layers.2.experts.49.linear_fc2` | True |
| `layers.2.experts.50.linear_fc1` | True |
| `layers.2.experts.50.linear_fc2` | True |
| `layers.2.experts.51.linear_fc1` | True |
| `layers.2.experts.51.linear_fc2` | True |
| `layers.2.experts.52.linear_fc1` | True |
| `layers.2.experts.52.linear_fc2` | True |
| `layers.2.experts.53.linear_fc1` | True |
| `layers.2.experts.53.linear_fc2` | True |
| `layers.2.experts.54.linear_fc1` | True |
| `layers.2.experts.54.linear_fc2` | True |
| `layers.2.experts.55.linear_fc1` | True |
| `layers.2.experts.55.linear_fc2` | True |
| `layers.2.experts.56.linear_fc1` | True |
| `layers.2.experts.56.linear_fc2` | True |
| `layers.2.experts.57.linear_fc1` | True |
| `layers.2.experts.57.linear_fc2` | True |
| `layers.2.experts.58.linear_fc1` | True |
| `layers.2.experts.58.linear_fc2` | True |
| `layers.2.experts.59.linear_fc1` | True |
| `layers.2.experts.59.linear_fc2` | True |
| `layers.2.experts.60.linear_fc1` | True |
| `layers.2.experts.60.linear_fc2` | True |
| `layers.2.experts.61.linear_fc1` | True |
| `layers.2.experts.61.linear_fc2` | True |
| `layers.2.experts.62.linear_fc1` | True |
| `layers.2.experts.62.linear_fc2` | True |
| `layers.2.experts.63.linear_fc1` | True |
| `layers.2.experts.63.linear_fc2` | True |
| `layers.2.experts.64.linear_fc1` | True |
| `layers.2.experts.64.linear_fc2` | True |
| `layers.2.experts.65.linear_fc1` | True |
| `layers.2.experts.65.linear_fc2` | True |
| `layers.2.experts.66.linear_fc1` | True |
| `layers.2.experts.66.linear_fc2` | True |
| `layers.2.experts.67.linear_fc1` | True |
| `layers.2.experts.67.linear_fc2` | True |
| `layers.2.experts.68.linear_fc1` | True |
| `layers.2.experts.68.linear_fc2` | True |
| `layers.2.experts.69.linear_fc1` | True |
| `layers.2.experts.69.linear_fc2` | True |
| `layers.2.experts.70.linear_fc1` | True |
| `layers.2.experts.70.linear_fc2` | True |
| `layers.2.experts.71.linear_fc1` | True |
| `layers.2.experts.71.linear_fc2` | True |
| `layers.2.experts.72.linear_fc1` | True |
| `layers.2.experts.72.linear_fc2` | True |
| `layers.2.experts.73.linear_fc1` | True |
| `layers.2.experts.73.linear_fc2` | True |
| `layers.2.experts.74.linear_fc1` | True |
| `layers.2.experts.74.linear_fc2` | True |
| `layers.2.experts.75.linear_fc1` | True |
| `layers.2.experts.75.linear_fc2` | True |
| `layers.2.experts.76.linear_fc1` | True |
| `layers.2.experts.76.linear_fc2` | True |
| `layers.2.experts.77.linear_fc1` | True |
| `layers.2.experts.77.linear_fc2` | True |
| `layers.2.experts.78.linear_fc1` | True |
| `layers.2.experts.78.linear_fc2` | True |
| `layers.2.experts.79.linear_fc1` | True |
| `layers.2.experts.79.linear_fc2` | True |
| `layers.2.experts.80.linear_fc1` | True |
| `layers.2.experts.80.linear_fc2` | True |
| `layers.2.experts.81.linear_fc1` | True |
| `layers.2.experts.81.linear_fc2` | True |
| `layers.2.experts.82.linear_fc1` | True |
| `layers.2.experts.82.linear_fc2` | True |
| `layers.2.experts.83.linear_fc1` | True |
| `layers.2.experts.83.linear_fc2` | True |
| `layers.2.experts.84.linear_fc1` | True |
| `layers.2.experts.84.linear_fc2` | True |
| `layers.2.experts.85.linear_fc1` | True |
| `layers.2.experts.85.linear_fc2` | True |
| `layers.2.experts.86.linear_fc1` | True |
| `layers.2.experts.86.linear_fc2` | True |
| `layers.2.experts.87.linear_fc1` | True |
| `layers.2.experts.87.linear_fc2` | True |
| `layers.2.experts.88.linear_fc1` | True |
| `layers.2.experts.88.linear_fc2` | True |
| `layers.2.experts.89.linear_fc1` | True |
| `layers.2.experts.89.linear_fc2` | True |
| `layers.2.experts.90.linear_fc1` | True |
| `layers.2.experts.90.linear_fc2` | True |
| `layers.2.experts.91.linear_fc1` | True |
| `layers.2.experts.91.linear_fc2` | True |
| `layers.2.experts.92.linear_fc1` | True |
| `layers.2.experts.92.linear_fc2` | True |
| `layers.2.experts.93.linear_fc1` | True |
| `layers.2.experts.93.linear_fc2` | True |
| `layers.2.experts.94.linear_fc1` | True |
| `layers.2.experts.94.linear_fc2` | True |
| `layers.2.experts.95.linear_fc1` | True |
| `layers.2.experts.95.linear_fc2` | True |
| `layers.2.experts.96.linear_fc1` | True |
| `layers.2.experts.96.linear_fc2` | True |
| `layers.2.experts.97.linear_fc1` | True |
| `layers.2.experts.97.linear_fc2` | True |
| `layers.2.experts.98.linear_fc1` | True |
| `layers.2.experts.98.linear_fc2` | True |
| `layers.2.experts.99.linear_fc1` | True |
| `layers.2.experts.99.linear_fc2` | True |
| `layers.2.experts.100.linear_fc1` | True |
| `layers.2.experts.100.linear_fc2` | True |
| `layers.2.experts.101.linear_fc1` | True |
| `layers.2.experts.101.linear_fc2` | True |
| `layers.2.experts.102.linear_fc1` | True |
| `layers.2.experts.102.linear_fc2` | True |
| `layers.2.experts.103.linear_fc1` | True |
| `layers.2.experts.103.linear_fc2` | True |
| `layers.2.experts.104.linear_fc1` | True |
| `layers.2.experts.104.linear_fc2` | True |
| `layers.2.experts.105.linear_fc1` | True |
| `layers.2.experts.105.linear_fc2` | True |
| `layers.2.experts.106.linear_fc1` | True |
| `layers.2.experts.106.linear_fc2` | True |
| `layers.2.experts.107.linear_fc1` | True |
| `layers.2.experts.107.linear_fc2` | True |
| `layers.2.experts.108.linear_fc1` | True |
| `layers.2.experts.108.linear_fc2` | True |
| `layers.2.experts.109.linear_fc1` | True |
| `layers.2.experts.109.linear_fc2` | True |
| `layers.2.experts.110.linear_fc1` | True |
| `layers.2.experts.110.linear_fc2` | True |
| `layers.2.experts.111.linear_fc1` | True |
| `layers.2.experts.111.linear_fc2` | True |
| `layers.2.experts.112.linear_fc1` | True |
| `layers.2.experts.112.linear_fc2` | True |
| `layers.2.experts.113.linear_fc1` | True |
| `layers.2.experts.113.linear_fc2` | True |
| `layers.2.experts.114.linear_fc1` | True |
| `layers.2.experts.114.linear_fc2` | True |
| `layers.2.experts.115.linear_fc1` | True |
| `layers.2.experts.115.linear_fc2` | True |
| `layers.2.experts.116.linear_fc1` | True |
| `layers.2.experts.116.linear_fc2` | True |
| `layers.2.experts.117.linear_fc1` | True |
| `layers.2.experts.117.linear_fc2` | True |
| `layers.2.experts.118.linear_fc1` | True |
| `layers.2.experts.118.linear_fc2` | True |
| `layers.2.experts.119.linear_fc1` | True |
| `layers.2.experts.119.linear_fc2` | True |
| `layers.2.experts.120.linear_fc1` | True |
| `layers.2.experts.120.linear_fc2` | True |
| `layers.2.experts.121.linear_fc1` | True |
| `layers.2.experts.121.linear_fc2` | True |
| `layers.2.experts.122.linear_fc1` | True |
| `layers.2.experts.122.linear_fc2` | True |
| `layers.2.experts.123.linear_fc1` | True |
| `layers.2.experts.123.linear_fc2` | True |
| `layers.2.experts.124.linear_fc1` | True |
| `layers.2.experts.124.linear_fc2` | True |
| `layers.2.experts.125.linear_fc1` | True |
| `layers.2.experts.125.linear_fc2` | True |
| `layers.2.experts.126.linear_fc1` | True |
| `layers.2.experts.126.linear_fc2` | True |
| `layers.2.experts.127.linear_fc1` | True |
| `layers.2.experts.127.linear_fc2` | True |
| `layers.2.experts.128.linear_fc1` | True |
| `layers.2.experts.128.linear_fc2` | True |
| `layers.2.experts.129.linear_fc1` | True |
| `layers.2.experts.129.linear_fc2` | True |
| `layers.2.experts.130.linear_fc1` | True |
| `layers.2.experts.130.linear_fc2` | True |
| `layers.2.experts.131.linear_fc1` | True |
| `layers.2.experts.131.linear_fc2` | True |
| `layers.2.experts.132.linear_fc1` | True |
| `layers.2.experts.132.linear_fc2` | True |
| `layers.2.experts.133.linear_fc1` | True |
| `layers.2.experts.133.linear_fc2` | True |
| `layers.2.experts.134.linear_fc1` | True |
| `layers.2.experts.134.linear_fc2` | True |
| `layers.2.experts.135.linear_fc1` | True |
| `layers.2.experts.135.linear_fc2` | True |
| `layers.2.experts.136.linear_fc1` | True |
| `layers.2.experts.136.linear_fc2` | True |
| `layers.2.experts.137.linear_fc1` | True |
| `layers.2.experts.137.linear_fc2` | True |
| `layers.2.experts.138.linear_fc1` | True |
| `layers.2.experts.138.linear_fc2` | True |
| `layers.2.experts.139.linear_fc1` | True |
| `layers.2.experts.139.linear_fc2` | True |
| `layers.2.experts.140.linear_fc1` | True |
| `layers.2.experts.140.linear_fc2` | True |
| `layers.2.experts.141.linear_fc1` | True |
| `layers.2.experts.141.linear_fc2` | True |
| `layers.2.experts.142.linear_fc1` | True |
| `layers.2.experts.142.linear_fc2` | True |
| `layers.2.experts.143.linear_fc1` | True |
| `layers.2.experts.143.linear_fc2` | True |
| `layers.2.experts.144.linear_fc1` | True |
| `layers.2.experts.144.linear_fc2` | True |
| `layers.2.experts.145.linear_fc1` | True |
| `layers.2.experts.145.linear_fc2` | True |
| `layers.2.experts.146.linear_fc1` | True |
| `layers.2.experts.146.linear_fc2` | True |
| `layers.2.experts.147.linear_fc1` | True |
| `layers.2.experts.147.linear_fc2` | True |
| `layers.2.experts.148.linear_fc1` | True |
| `layers.2.experts.148.linear_fc2` | True |
| `layers.2.experts.149.linear_fc1` | True |
| `layers.2.experts.149.linear_fc2` | True |
| `layers.2.experts.150.linear_fc1` | True |
| `layers.2.experts.150.linear_fc2` | True |
| `layers.2.experts.151.linear_fc1` | True |
| `layers.2.experts.151.linear_fc2` | True |
| `layers.2.experts.152.linear_fc1` | True |
| `layers.2.experts.152.linear_fc2` | True |
| `layers.2.experts.153.linear_fc1` | True |
| `layers.2.experts.153.linear_fc2` | True |
| `layers.2.experts.154.linear_fc1` | True |
| `layers.2.experts.154.linear_fc2` | True |
| `layers.2.experts.155.linear_fc1` | True |
| `layers.2.experts.155.linear_fc2` | True |
| `layers.2.experts.156.linear_fc1` | True |
| `layers.2.experts.156.linear_fc2` | True |
| `layers.2.experts.157.linear_fc1` | True |
| `layers.2.experts.157.linear_fc2` | True |
| `layers.2.experts.158.linear_fc1` | True |
| `layers.2.experts.158.linear_fc2` | True |
| `layers.2.experts.159.linear_fc1` | True |
| `layers.2.experts.159.linear_fc2` | True |
| `layers.2.experts.160.linear_fc1` | True |
| `layers.2.experts.160.linear_fc2` | True |
| `layers.2.experts.161.linear_fc1` | True |
| `layers.2.experts.161.linear_fc2` | True |
| `layers.2.experts.162.linear_fc1` | True |
| `layers.2.experts.162.linear_fc2` | True |
| `layers.2.experts.163.linear_fc1` | True |
| `layers.2.experts.163.linear_fc2` | True |
| `layers.2.experts.164.linear_fc1` | True |
| `layers.2.experts.164.linear_fc2` | True |
| `layers.2.experts.165.linear_fc1` | True |
| `layers.2.experts.165.linear_fc2` | True |
| `layers.2.experts.166.linear_fc1` | True |
| `layers.2.experts.166.linear_fc2` | True |
| `layers.2.experts.167.linear_fc1` | True |
| `layers.2.experts.167.linear_fc2` | True |
| `layers.2.experts.168.linear_fc1` | True |
| `layers.2.experts.168.linear_fc2` | True |
| `layers.2.experts.169.linear_fc1` | True |
| `layers.2.experts.169.linear_fc2` | True |
| `layers.2.experts.170.linear_fc1` | True |
| `layers.2.experts.170.linear_fc2` | True |
| `layers.2.experts.171.linear_fc1` | True |
| `layers.2.experts.171.linear_fc2` | True |
| `layers.2.experts.172.linear_fc1` | True |
| `layers.2.experts.172.linear_fc2` | True |
| `layers.2.experts.173.linear_fc1` | True |
| `layers.2.experts.173.linear_fc2` | True |
| `layers.2.experts.174.linear_fc1` | True |
| `layers.2.experts.174.linear_fc2` | True |
| `layers.2.experts.175.linear_fc1` | True |
| `layers.2.experts.175.linear_fc2` | True |
| `layers.2.experts.176.linear_fc1` | True |
| `layers.2.experts.176.linear_fc2` | True |
| `layers.2.experts.177.linear_fc1` | True |
| `layers.2.experts.177.linear_fc2` | True |
| `layers.2.experts.178.linear_fc1` | True |
| `layers.2.experts.178.linear_fc2` | True |
| `layers.2.experts.179.linear_fc1` | True |
| `layers.2.experts.179.linear_fc2` | True |
| `layers.2.experts.180.linear_fc1` | True |
| `layers.2.experts.180.linear_fc2` | True |
| `layers.2.experts.181.linear_fc1` | True |
| `layers.2.experts.181.linear_fc2` | True |
| `layers.2.experts.182.linear_fc1` | True |
| `layers.2.experts.182.linear_fc2` | True |
| `layers.2.experts.183.linear_fc1` | True |
| `layers.2.experts.183.linear_fc2` | True |
| `layers.2.experts.184.linear_fc1` | True |
| `layers.2.experts.184.linear_fc2` | True |
| `layers.2.experts.185.linear_fc1` | True |
| `layers.2.experts.185.linear_fc2` | True |
| `layers.2.experts.186.linear_fc1` | True |
| `layers.2.experts.186.linear_fc2` | True |
| `layers.2.experts.187.linear_fc1` | True |
| `layers.2.experts.187.linear_fc2` | True |
| `layers.2.experts.188.linear_fc1` | True |
| `layers.2.experts.188.linear_fc2` | True |
| `layers.2.experts.189.linear_fc1` | True |
| `layers.2.experts.189.linear_fc2` | True |
| `layers.2.experts.190.linear_fc1` | True |
| `layers.2.experts.190.linear_fc2` | True |
| `layers.2.experts.191.linear_fc1` | True |
| `layers.2.experts.191.linear_fc2` | True |
| `layers.3.experts.0.linear_fc1` | True |
| `layers.3.experts.0.linear_fc2` | True |
| `layers.3.experts.1.linear_fc1` | True |
| `layers.3.experts.1.linear_fc2` | True |
| `layers.3.experts.2.linear_fc1` | True |
| `layers.3.experts.2.linear_fc2` | True |
| `layers.3.experts.3.linear_fc1` | True |
| `layers.3.experts.3.linear_fc2` | True |
| `layers.3.experts.4.linear_fc1` | True |
| `layers.3.experts.4.linear_fc2` | True |
| `layers.3.experts.5.linear_fc1` | True |
| `layers.3.experts.5.linear_fc2` | True |
| `layers.3.experts.6.linear_fc1` | True |
| `layers.3.experts.6.linear_fc2` | True |
| `layers.3.experts.7.linear_fc1` | True |
| `layers.3.experts.7.linear_fc2` | True |
| `layers.3.experts.8.linear_fc1` | True |
| `layers.3.experts.8.linear_fc2` | True |
| `layers.3.experts.9.linear_fc1` | True |
| `layers.3.experts.9.linear_fc2` | True |
| `layers.3.experts.10.linear_fc1` | True |
| `layers.3.experts.10.linear_fc2` | True |
| `layers.3.experts.11.linear_fc1` | True |
| `layers.3.experts.11.linear_fc2` | True |
| `layers.3.experts.12.linear_fc1` | True |
| `layers.3.experts.12.linear_fc2` | True |
| `layers.3.experts.13.linear_fc1` | True |
| `layers.3.experts.13.linear_fc2` | True |
| `layers.3.experts.14.linear_fc1` | True |
| `layers.3.experts.14.linear_fc2` | True |
| `layers.3.experts.15.linear_fc1` | True |
| `layers.3.experts.15.linear_fc2` | True |
| `layers.3.experts.16.linear_fc1` | True |
| `layers.3.experts.16.linear_fc2` | True |
| `layers.3.experts.17.linear_fc1` | True |
| `layers.3.experts.17.linear_fc2` | True |
| `layers.3.experts.18.linear_fc1` | True |
| `layers.3.experts.18.linear_fc2` | True |
| `layers.3.experts.19.linear_fc1` | True |
| `layers.3.experts.19.linear_fc2` | True |
| `layers.3.experts.20.linear_fc1` | True |
| `layers.3.experts.20.linear_fc2` | True |
| `layers.3.experts.21.linear_fc1` | True |
| `layers.3.experts.21.linear_fc2` | True |
| `layers.3.experts.22.linear_fc1` | True |
| `layers.3.experts.22.linear_fc2` | True |
| `layers.3.experts.23.linear_fc1` | True |
| `layers.3.experts.23.linear_fc2` | True |
| `layers.3.experts.24.linear_fc1` | True |
| `layers.3.experts.24.linear_fc2` | True |
| `layers.3.experts.25.linear_fc1` | True |
| `layers.3.experts.25.linear_fc2` | True |
| `layers.3.experts.26.linear_fc1` | True |
| `layers.3.experts.26.linear_fc2` | True |
| `layers.3.experts.27.linear_fc1` | True |
| `layers.3.experts.27.linear_fc2` | True |
| `layers.3.experts.28.linear_fc1` | True |
| `layers.3.experts.28.linear_fc2` | True |
| `layers.3.experts.29.linear_fc1` | True |
| `layers.3.experts.29.linear_fc2` | True |
| `layers.3.experts.30.linear_fc1` | True |
| `layers.3.experts.30.linear_fc2` | True |
| `layers.3.experts.31.linear_fc1` | True |
| `layers.3.experts.31.linear_fc2` | True |
| `layers.3.experts.32.linear_fc1` | True |
| `layers.3.experts.32.linear_fc2` | True |
| `layers.3.experts.33.linear_fc1` | True |
| `layers.3.experts.33.linear_fc2` | True |
| `layers.3.experts.34.linear_fc1` | True |
| `layers.3.experts.34.linear_fc2` | True |
| `layers.3.experts.35.linear_fc1` | True |
| `layers.3.experts.35.linear_fc2` | True |
| `layers.3.experts.36.linear_fc1` | True |
| `layers.3.experts.36.linear_fc2` | True |
| `layers.3.experts.37.linear_fc1` | True |
| `layers.3.experts.37.linear_fc2` | True |
| `layers.3.experts.38.linear_fc1` | True |
| `layers.3.experts.38.linear_fc2` | True |
| `layers.3.experts.39.linear_fc1` | True |
| `layers.3.experts.39.linear_fc2` | True |
| `layers.3.experts.40.linear_fc1` | True |
| `layers.3.experts.40.linear_fc2` | True |
| `layers.3.experts.41.linear_fc1` | True |
| `layers.3.experts.41.linear_fc2` | True |
| `layers.3.experts.42.linear_fc1` | True |
| `layers.3.experts.42.linear_fc2` | True |
| `layers.3.experts.43.linear_fc1` | True |
| `layers.3.experts.43.linear_fc2` | True |
| `layers.3.experts.44.linear_fc1` | True |
| `layers.3.experts.44.linear_fc2` | True |
| `layers.3.experts.45.linear_fc1` | True |
| `layers.3.experts.45.linear_fc2` | True |
| `layers.3.experts.46.linear_fc1` | True |
| `layers.3.experts.46.linear_fc2` | True |
| `layers.3.experts.47.linear_fc1` | True |
| `layers.3.experts.47.linear_fc2` | True |
| `layers.3.experts.48.linear_fc1` | True |
| `layers.3.experts.48.linear_fc2` | True |
| `layers.3.experts.49.linear_fc1` | True |
| `layers.3.experts.49.linear_fc2` | True |
| `layers.3.experts.50.linear_fc1` | True |
| `layers.3.experts.50.linear_fc2` | True |
| `layers.3.experts.51.linear_fc1` | True |
| `layers.3.experts.51.linear_fc2` | True |
| `layers.3.experts.52.linear_fc1` | True |
| `layers.3.experts.52.linear_fc2` | True |
| `layers.3.experts.53.linear_fc1` | True |
| `layers.3.experts.53.linear_fc2` | True |
| `layers.3.experts.54.linear_fc1` | True |
| `layers.3.experts.54.linear_fc2` | True |
| `layers.3.experts.55.linear_fc1` | True |
| `layers.3.experts.55.linear_fc2` | True |
| `layers.3.experts.56.linear_fc1` | True |
| `layers.3.experts.56.linear_fc2` | True |
| `layers.3.experts.57.linear_fc1` | True |
| `layers.3.experts.57.linear_fc2` | True |
| `layers.3.experts.58.linear_fc1` | True |
| `layers.3.experts.58.linear_fc2` | True |
| `layers.3.experts.59.linear_fc1` | True |
| `layers.3.experts.59.linear_fc2` | True |
| `layers.3.experts.60.linear_fc1` | True |
| `layers.3.experts.60.linear_fc2` | True |
| `layers.3.experts.61.linear_fc1` | True |
| `layers.3.experts.61.linear_fc2` | True |
| `layers.3.experts.62.linear_fc1` | True |
| `layers.3.experts.62.linear_fc2` | True |
| `layers.3.experts.63.linear_fc1` | True |
| `layers.3.experts.63.linear_fc2` | True |
| `layers.3.experts.64.linear_fc1` | True |
| `layers.3.experts.64.linear_fc2` | True |
| `layers.3.experts.65.linear_fc1` | True |
| `layers.3.experts.65.linear_fc2` | True |
| `layers.3.experts.66.linear_fc1` | True |
| `layers.3.experts.66.linear_fc2` | True |
| `layers.3.experts.67.linear_fc1` | True |
| `layers.3.experts.67.linear_fc2` | True |
| `layers.3.experts.68.linear_fc1` | True |
| `layers.3.experts.68.linear_fc2` | True |
| `layers.3.experts.69.linear_fc1` | True |
| `layers.3.experts.69.linear_fc2` | True |
| `layers.3.experts.70.linear_fc1` | True |
| `layers.3.experts.70.linear_fc2` | True |
| `layers.3.experts.71.linear_fc1` | True |
| `layers.3.experts.71.linear_fc2` | True |
| `layers.3.experts.72.linear_fc1` | True |
| `layers.3.experts.72.linear_fc2` | True |
| `layers.3.experts.73.linear_fc1` | True |
| `layers.3.experts.73.linear_fc2` | True |
| `layers.3.experts.74.linear_fc1` | True |
| `layers.3.experts.74.linear_fc2` | True |
| `layers.3.experts.75.linear_fc1` | True |
| `layers.3.experts.75.linear_fc2` | True |
| `layers.3.experts.76.linear_fc1` | True |
| `layers.3.experts.76.linear_fc2` | True |
| `layers.3.experts.77.linear_fc1` | True |
| `layers.3.experts.77.linear_fc2` | True |
| `layers.3.experts.78.linear_fc1` | True |
| `layers.3.experts.78.linear_fc2` | True |
| `layers.3.experts.79.linear_fc1` | True |
| `layers.3.experts.79.linear_fc2` | True |
| `layers.3.experts.80.linear_fc1` | True |
| `layers.3.experts.80.linear_fc2` | True |
| `layers.3.experts.81.linear_fc1` | True |
| `layers.3.experts.81.linear_fc2` | True |
| `layers.3.experts.82.linear_fc1` | True |
| `layers.3.experts.82.linear_fc2` | True |
| `layers.3.experts.83.linear_fc1` | True |
| `layers.3.experts.83.linear_fc2` | True |
| `layers.3.experts.84.linear_fc1` | True |
| `layers.3.experts.84.linear_fc2` | True |
| `layers.3.experts.85.linear_fc1` | True |
| `layers.3.experts.85.linear_fc2` | True |
| `layers.3.experts.86.linear_fc1` | True |
| `layers.3.experts.86.linear_fc2` | True |
| `layers.3.experts.87.linear_fc1` | True |
| `layers.3.experts.87.linear_fc2` | True |
| `layers.3.experts.88.linear_fc1` | True |
| `layers.3.experts.88.linear_fc2` | True |
| `layers.3.experts.89.linear_fc1` | True |
| `layers.3.experts.89.linear_fc2` | True |
| `layers.3.experts.90.linear_fc1` | True |
| `layers.3.experts.90.linear_fc2` | True |
| `layers.3.experts.91.linear_fc1` | True |
| `layers.3.experts.91.linear_fc2` | True |
| `layers.3.experts.92.linear_fc1` | True |
| `layers.3.experts.92.linear_fc2` | True |
| `layers.3.experts.93.linear_fc1` | True |
| `layers.3.experts.93.linear_fc2` | True |
| `layers.3.experts.94.linear_fc1` | True |
| `layers.3.experts.94.linear_fc2` | True |
| `layers.3.experts.95.linear_fc1` | True |
| `layers.3.experts.95.linear_fc2` | True |
| `layers.3.experts.96.linear_fc1` | True |
| `layers.3.experts.96.linear_fc2` | True |
| `layers.3.experts.97.linear_fc1` | True |
| `layers.3.experts.97.linear_fc2` | True |
| `layers.3.experts.98.linear_fc1` | True |
| `layers.3.experts.98.linear_fc2` | True |
| `layers.3.experts.99.linear_fc1` | True |
| `layers.3.experts.99.linear_fc2` | True |
| `layers.3.experts.100.linear_fc1` | True |
| `layers.3.experts.100.linear_fc2` | True |
| `layers.3.experts.101.linear_fc1` | True |
| `layers.3.experts.101.linear_fc2` | True |
| `layers.3.experts.102.linear_fc1` | True |
| `layers.3.experts.102.linear_fc2` | True |
| `layers.3.experts.103.linear_fc1` | True |
| `layers.3.experts.103.linear_fc2` | True |
| `layers.3.experts.104.linear_fc1` | True |
| `layers.3.experts.104.linear_fc2` | True |
| `layers.3.experts.105.linear_fc1` | True |
| `layers.3.experts.105.linear_fc2` | True |
| `layers.3.experts.106.linear_fc1` | True |
| `layers.3.experts.106.linear_fc2` | True |
| `layers.3.experts.107.linear_fc1` | True |
| `layers.3.experts.107.linear_fc2` | True |
| `layers.3.experts.108.linear_fc1` | True |
| `layers.3.experts.108.linear_fc2` | True |
| `layers.3.experts.109.linear_fc1` | True |
| `layers.3.experts.109.linear_fc2` | True |
| `layers.3.experts.110.linear_fc1` | True |
| `layers.3.experts.110.linear_fc2` | True |
| `layers.3.experts.111.linear_fc1` | True |
| `layers.3.experts.111.linear_fc2` | True |
| `layers.3.experts.112.linear_fc1` | True |
| `layers.3.experts.112.linear_fc2` | True |
| `layers.3.experts.113.linear_fc1` | True |
| `layers.3.experts.113.linear_fc2` | True |
| `layers.3.experts.114.linear_fc1` | True |
| `layers.3.experts.114.linear_fc2` | True |
| `layers.3.experts.115.linear_fc1` | True |
| `layers.3.experts.115.linear_fc2` | True |
| `layers.3.experts.116.linear_fc1` | True |
| `layers.3.experts.116.linear_fc2` | True |
| `layers.3.experts.117.linear_fc1` | True |
| `layers.3.experts.117.linear_fc2` | True |
| `layers.3.experts.118.linear_fc1` | True |
| `layers.3.experts.118.linear_fc2` | True |
| `layers.3.experts.119.linear_fc1` | True |
| `layers.3.experts.119.linear_fc2` | True |
| `layers.3.experts.120.linear_fc1` | True |
| `layers.3.experts.120.linear_fc2` | True |
| `layers.3.experts.121.linear_fc1` | True |
| `layers.3.experts.121.linear_fc2` | True |
| `layers.3.experts.122.linear_fc1` | True |
| `layers.3.experts.122.linear_fc2` | True |
| `layers.3.experts.123.linear_fc1` | True |
| `layers.3.experts.123.linear_fc2` | True |
| `layers.3.experts.124.linear_fc1` | True |
| `layers.3.experts.124.linear_fc2` | True |
| `layers.3.experts.125.linear_fc1` | True |
| `layers.3.experts.125.linear_fc2` | True |
| `layers.3.experts.126.linear_fc1` | True |
| `layers.3.experts.126.linear_fc2` | True |
| `layers.3.experts.127.linear_fc1` | True |
| `layers.3.experts.127.linear_fc2` | True |
| `layers.3.experts.128.linear_fc1` | True |
| `layers.3.experts.128.linear_fc2` | True |
| `layers.3.experts.129.linear_fc1` | True |
| `layers.3.experts.129.linear_fc2` | True |
| `layers.3.experts.130.linear_fc1` | True |
| `layers.3.experts.130.linear_fc2` | True |
| `layers.3.experts.131.linear_fc1` | True |
| `layers.3.experts.131.linear_fc2` | True |
| `layers.3.experts.132.linear_fc1` | True |
| `layers.3.experts.132.linear_fc2` | True |
| `layers.3.experts.133.linear_fc1` | True |
| `layers.3.experts.133.linear_fc2` | True |
| `layers.3.experts.134.linear_fc1` | True |
| `layers.3.experts.134.linear_fc2` | True |
| `layers.3.experts.135.linear_fc1` | True |
| `layers.3.experts.135.linear_fc2` | True |
| `layers.3.experts.136.linear_fc1` | True |
| `layers.3.experts.136.linear_fc2` | True |
| `layers.3.experts.137.linear_fc1` | True |
| `layers.3.experts.137.linear_fc2` | True |
| `layers.3.experts.138.linear_fc1` | True |
| `layers.3.experts.138.linear_fc2` | True |
| `layers.3.experts.139.linear_fc1` | True |
| `layers.3.experts.139.linear_fc2` | True |
| `layers.3.experts.140.linear_fc1` | True |
| `layers.3.experts.140.linear_fc2` | True |
| `layers.3.experts.141.linear_fc1` | True |
| `layers.3.experts.141.linear_fc2` | True |
| `layers.3.experts.142.linear_fc1` | True |
| `layers.3.experts.142.linear_fc2` | True |
| `layers.3.experts.143.linear_fc1` | True |
| `layers.3.experts.143.linear_fc2` | True |
| `layers.3.experts.144.linear_fc1` | True |
| `layers.3.experts.144.linear_fc2` | True |
| `layers.3.experts.145.linear_fc1` | True |
| `layers.3.experts.145.linear_fc2` | True |
| `layers.3.experts.146.linear_fc1` | True |
| `layers.3.experts.146.linear_fc2` | True |
| `layers.3.experts.147.linear_fc1` | True |
| `layers.3.experts.147.linear_fc2` | True |
| `layers.3.experts.148.linear_fc1` | True |
| `layers.3.experts.148.linear_fc2` | True |
| `layers.3.experts.149.linear_fc1` | True |
| `layers.3.experts.149.linear_fc2` | True |
| `layers.3.experts.150.linear_fc1` | True |
| `layers.3.experts.150.linear_fc2` | True |
| `layers.3.experts.151.linear_fc1` | True |
| `layers.3.experts.151.linear_fc2` | True |
| `layers.3.experts.152.linear_fc1` | True |
| `layers.3.experts.152.linear_fc2` | True |
| `layers.3.experts.153.linear_fc1` | True |
| `layers.3.experts.153.linear_fc2` | True |
| `layers.3.experts.154.linear_fc1` | True |
| `layers.3.experts.154.linear_fc2` | True |
| `layers.3.experts.155.linear_fc1` | True |
| `layers.3.experts.155.linear_fc2` | True |
| `layers.3.experts.156.linear_fc1` | True |
| `layers.3.experts.156.linear_fc2` | True |
| `layers.3.experts.157.linear_fc1` | True |
| `layers.3.experts.157.linear_fc2` | True |
| `layers.3.experts.158.linear_fc1` | True |
| `layers.3.experts.158.linear_fc2` | True |
| `layers.3.experts.159.linear_fc1` | True |
| `layers.3.experts.159.linear_fc2` | True |
| `layers.3.experts.160.linear_fc1` | True |
| `layers.3.experts.160.linear_fc2` | True |
| `layers.3.experts.161.linear_fc1` | True |
| `layers.3.experts.161.linear_fc2` | True |
| `layers.3.experts.162.linear_fc1` | True |
| `layers.3.experts.162.linear_fc2` | True |
| `layers.3.experts.163.linear_fc1` | True |
| `layers.3.experts.163.linear_fc2` | True |
| `layers.3.experts.164.linear_fc1` | True |
| `layers.3.experts.164.linear_fc2` | True |
| `layers.3.experts.165.linear_fc1` | True |
| `layers.3.experts.165.linear_fc2` | True |
| `layers.3.experts.166.linear_fc1` | True |
| `layers.3.experts.166.linear_fc2` | True |
| `layers.3.experts.167.linear_fc1` | True |
| `layers.3.experts.167.linear_fc2` | True |
| `layers.3.experts.168.linear_fc1` | True |
| `layers.3.experts.168.linear_fc2` | True |
| `layers.3.experts.169.linear_fc1` | True |
| `layers.3.experts.169.linear_fc2` | True |
| `layers.3.experts.170.linear_fc1` | True |
| `layers.3.experts.170.linear_fc2` | True |
| `layers.3.experts.171.linear_fc1` | True |
| `layers.3.experts.171.linear_fc2` | True |
| `layers.3.experts.172.linear_fc1` | True |
| `layers.3.experts.172.linear_fc2` | True |
| `layers.3.experts.173.linear_fc1` | True |
| `layers.3.experts.173.linear_fc2` | True |
| `layers.3.experts.174.linear_fc1` | True |
| `layers.3.experts.174.linear_fc2` | True |
| `layers.3.experts.175.linear_fc1` | True |
| `layers.3.experts.175.linear_fc2` | True |
| `layers.3.experts.176.linear_fc1` | True |
| `layers.3.experts.176.linear_fc2` | True |
| `layers.3.experts.177.linear_fc1` | True |
| `layers.3.experts.177.linear_fc2` | True |
| `layers.3.experts.178.linear_fc1` | True |
| `layers.3.experts.178.linear_fc2` | True |
| `layers.3.experts.179.linear_fc1` | True |
| `layers.3.experts.179.linear_fc2` | True |
| `layers.3.experts.180.linear_fc1` | True |
| `layers.3.experts.180.linear_fc2` | True |
| `layers.3.experts.181.linear_fc1` | True |
| `layers.3.experts.181.linear_fc2` | True |
| `layers.3.experts.182.linear_fc1` | True |
| `layers.3.experts.182.linear_fc2` | True |
| `layers.3.experts.183.linear_fc1` | True |
| `layers.3.experts.183.linear_fc2` | True |
| `layers.3.experts.184.linear_fc1` | True |
| `layers.3.experts.184.linear_fc2` | True |
| `layers.3.experts.185.linear_fc1` | True |
| `layers.3.experts.185.linear_fc2` | True |
| `layers.3.experts.186.linear_fc1` | True |
| `layers.3.experts.186.linear_fc2` | True |
| `layers.3.experts.187.linear_fc1` | True |
| `layers.3.experts.187.linear_fc2` | True |
| `layers.3.experts.188.linear_fc1` | True |
| `layers.3.experts.188.linear_fc2` | True |
| `layers.3.experts.189.linear_fc1` | True |
| `layers.3.experts.189.linear_fc2` | True |
| `layers.3.experts.190.linear_fc1` | True |
| `layers.3.experts.190.linear_fc2` | True |
| `layers.3.experts.191.linear_fc1` | True |
| `layers.3.experts.191.linear_fc2` | True |

### `L2.moe.stages.chain`（pass）

| 用例 | stage | rel_l2 | cr_match_frac | ulp_p999 |
|---|---|---|---|---|
| `layers.1.router.logits` | router.logits | 3.758e-06 | 3.751e-04 | 1.730e+02 |
| `layers.1.router.scores` | router.scores | 1.331e-06 | 6.835e-03 | 1.590e+02 |
| `layers.1.router.probs` | router.probs | 7.873e-07 | 6.165e-02 | 3.400e+01 |
| `layers.1.permute` | permute | 0 | 1 | 0 |
| `layers.1.gemm1` | gemm1 | 1.660e-03 | 0.996927 | 1 |
| `layers.1.act` | act | 3.727e-03 | 5.440e-01 | 2.000e+00 |
| `layers.1.gemm2` | gemm2 | 3.665e-03 | 3.942e-01 | 2.120e+02 |
| `layers.1.unpermute` | unpermute | 3.421e-03 | 4.140e-01 | 1.980e+02 |
| `layers.1.shared` | shared | 3.034e-03 | 4.244e-01 | 1.980e+02 |
| `layers.1.output` | output | 3.314e-03 | 4.125e-01 | 2.410e+02 |
| `layers.2.router.logits` | router.logits | 3.797e-06 | 4.069e-05 | 1.090e+02 |
| `layers.2.router.scores` | router.scores | 2.382e-06 | 4.756e-04 | 1.726e+02 |
| `layers.2.router.probs` | router.probs | 1.359e-06 | 3.107e-02 | 5.400e+01 |
| `layers.2.permute` | permute | 0 | 1 | 0 |
| `layers.2.gemm1` | gemm1 | 1.659e-03 | 0.997368 | 1 |
| `layers.2.act` | act | 3.726e-03 | 5.377e-01 | 2.000e+00 |
| `layers.2.gemm2` | gemm2 | 3.763e-03 | 3.908e-01 | 2.150e+02 |
| `layers.2.unpermute` | unpermute | 3.752e-03 | 4.033e-01 | 2.050e+02 |
| `layers.2.shared` | shared | 3.161e-03 | 4.281e-01 | 1.950e+02 |
| `layers.2.output` | output | 3.640e-03 | 3.947e-01 | 2.950e+02 |
| `layers.3.router.logits` | router.logits | 3.956e-06 | 1.780e-05 | 8.000e+01 |
| `layers.3.router.scores` | router.scores | 3.136e-06 | 2.060e-04 | 1.910e+02 |
| `layers.3.router.probs` | router.probs | 1.794e-06 | 2.719e-02 | 7.023e+01 |
| `layers.3.permute` | permute | 0 | 1 | 0 |
| `layers.3.gemm1` | gemm1 | 1.660e-03 | 0.997754 | 1 |
| `layers.3.act` | act | 3.780e-03 | 5.390e-01 | 2.000e+00 |
| `layers.3.gemm2` | gemm2 | 3.985e-03 | 3.881e-01 | 2.150e+02 |
| `layers.3.unpermute` | unpermute | 3.951e-03 | 3.963e-01 | 2.060e+02 |
| `layers.3.shared` | shared | 3.240e-03 | 4.270e-01 | 1.940e+02 |
| `layers.3.output` | output | 4.033e-03 | 3.799e-01 | 2.650e+02 |

### `L2.moe.stages.local`（pass）

| 用例 | stage | rel_l2 | cr_match_frac | ulp_p999 |
|---|---|---|---|---|
| `layers.1.router.logits` | router.logits | 3.758e-06 | 3.751e-04 | 1.730e+02 |
| `layers.1.router.scores` | router.scores | 4.570e-08 | 5.996e-01 | 2.000e+00 |
| `layers.1.router.probs` | router.probs | 6.433e-08 | 4.981e-01 | 2.000e+00 |
| `layers.1.permute` | permute | 0 | 1 | 0 |
| `layers.1.gemm1` | gemm1 | 1.660e-03 | 0.996927 | 1 |
| `layers.1.act` | act | 2.871e-03 | 6.611e-01 | 2.000e+00 |
| `layers.1.gemm2` | gemm2 | 1.657e-03 | 0.998938 | 1 |
| `layers.1.unpermute` | unpermute | 1.659e-03 | 1 | 0 |
| `layers.1.shared` | shared | 3.034e-03 | 4.244e-01 | 1.980e+02 |
| `layers.1.output` | output | 1.773e-03 | 1 | 0 |
| `layers.2.router.logits` | router.logits | 3.797e-06 | 4.069e-05 | 1.090e+02 |
| `layers.2.router.scores` | router.scores | 4.493e-08 | 6.374e-01 | 2.000e+00 |
| `layers.2.router.probs` | router.probs | 6.461e-08 | 5.057e-01 | 3.000e+00 |
| `layers.2.permute` | permute | 0 | 1 | 0 |
| `layers.2.gemm1` | gemm1 | 1.659e-03 | 0.997368 | 1 |
| `layers.2.act` | act | 2.842e-03 | 6.488e-01 | 2.000e+00 |
| `layers.2.gemm2` | gemm2 | 1.659e-03 | 0.998959 | 1 |
| `layers.2.unpermute` | unpermute | 1.660e-03 | 1.000000 | 0 |
| `layers.2.shared` | shared | 3.161e-03 | 4.281e-01 | 1.950e+02 |
| `layers.2.output` | output | 1.771e-03 | 1 | 0 |
| `layers.3.router.logits` | router.logits | 3.956e-06 | 1.780e-05 | 8.000e+01 |
| `layers.3.router.scores` | router.scores | 4.520e-08 | 6.270e-01 | 2.000e+00 |
| `layers.3.router.probs` | router.probs | 6.624e-08 | 4.917e-01 | 3.000e+00 |
| `layers.3.permute` | permute | 0 | 1 | 0 |
| `layers.3.gemm1` | gemm1 | 1.660e-03 | 0.997754 | 1 |
| `layers.3.act` | act | 2.991e-03 | 6.511e-01 | 2.000e+00 |
| `layers.3.gemm2` | gemm2 | 1.657e-03 | 0.998968 | 1 |
| `layers.3.unpermute` | unpermute | 1.660e-03 | 1 | 0 |
| `layers.3.shared` | shared | 3.240e-03 | 4.270e-01 | 1.940e+02 |
| `layers.3.output` | output | 1.730e-03 | 1 | 0 |

### `L2.moe.movement`（pass）

| 用例 | bit_match_frac | rows |
|---|---|---|
| `layers.1.permute.tokens` | 1 | 32768 |
| `layers.1.permute.probs` | 1 |  |
| `layers.1.dispatch.rank0_to_local` | 1 | 3352 |
| `layers.1.combine.local_back` | 1 | 3352 |
| `layers.2.permute.tokens` | 1 | 32768 |
| `layers.2.permute.probs` | 1 |  |
| `layers.2.dispatch.rank0_to_local` | 1 | 4430 |
| `layers.2.combine.local_back` | 1 | 4430 |
| `layers.3.permute.tokens` | 1 | 32768 |
| `layers.3.permute.probs` | 1 |  |
| `layers.3.dispatch.rank0_to_local` | 1 | 4128 |
| `layers.3.combine.local_back` | 1 | 4128 |

### `L2.moe.topk.fp64`（pass）

| 用例 | expect_match_frac | rows_differ | boundary_tie_rows_fp64 |
|---|---|---|---|
| `layers.1.topk.vs_fp64_chain` | 1 | 0 | 0 |
| `layers.2.topk.vs_fp64_chain` | 1 | 0 | 0 |
| `layers.3.topk.vs_fp64_chain` | 1 | 0 | 0 |

数字全部来自 `results.json` 与 `details/`，本报告不手写数字。
