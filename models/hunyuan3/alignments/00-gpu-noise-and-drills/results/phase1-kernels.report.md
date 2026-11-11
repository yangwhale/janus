# janus 报告：hunyuan3-4l / phase1-kernels

- 被测：`g61-obs, g61-notf32`
- janus 0.1.0.dev0；目录 `models/hunyuan3/frameworks/megatron/l1_kernels.py`（sha256 cc18755f7e07…）
- 结果：pass 16、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `TF32.switch.matrix` | op_replay | L1 | pass | rel_l2=2.942e-04, cr_match_frac=7.248e-05, ulp_max=73059380, ulp_p999=2.104e+06 | 诊断：只报数，不判 |
| `TF32.switch.override_off` | op_replay | L1 | pass | rel_l2=1.146e-06, cr_match_frac=2.644e-02, ulp_max=3139288, ulp_p999=4.680e+03 | 全部在门槛内 |
| `TF32.router_bwd` | op_replay | L1 | pass | rel_l2=1.698e-03, cr_match_frac=8.966e-01, ulp_max=2129, ulp_p999=2.400e+01 | 诊断：只报数，不判 |
| `TF32.in_run` | op_replay | L1 | pass | cases=2 | 关 TF32 的运行里，torch 与 TE 的 FP32 GEMM 实测都是 IEEE FP32 |
| `TF32.gemm_inventory` | op_replay | L1 | pass | calls=154, fp32_input_calls=12, fp32_input_sites=2, cases=14 | 第 1 步 rank 0 共 154 次 GEMM 调用，输入含 FP32 的 12 次（2 处调用点） |
| `L1.kernel.experts.fwd.replay` | op_replay | L1 | pass | rel_l2=2.909e-03, cr_match_frac=6.485e-01, ulp_max=1130, ulp_p999=2.000e+00 | 逐位全中（bit_match_frac） |
| `L1.kernel.experts.bwd.replay` | op_replay | L1 | pass | rel_l2=1.659e-03, cr_match_frac=9.978e-01, ulp_max=1290, ulp_p999=1.000e+00 | 逐位全中（bit_match_frac） |
| `L1.kernel.experts.vs_loop` | op_replay | L1 | pass | rel_l2=1.659e-03, cr_match_frac=9.969e-01, ulp_max=1130, ulp_p999=1.000e+00 | 诊断：只报数，不判 |
| `L1.kernel.experts.wgrad` | op_replay | L1 | pass | rel_l2=3.499e-06, cr_match_frac=7.935e-03, ulp_max=744577648, ulp_p999=4.206e+03 | 全部在门槛内 |
| `L1.kernel.gemm_row_invariance` | op_replay | L1 | pass | bit_match_frac=9.969e-01, cases=208 | 诊断：只报数，不判 |
| `L1.kernel.attn.fwd.replay` | op_replay | L1 | pass | rel_l2=1.947e-03, cr_match_frac=7.842e-01, ulp_max=2033, ulp_p999=5.500e+01 | 逐位全中（bit_match_frac） |
| `L1.kernel.attn.bwd.replay` | op_replay | L1 | pass | rel_l2=3.969e-03, cr_match_frac=5.254e-01, ulp_max=14004, ulp_p999=1.780e+02 | 逐位全中（bit_match_frac） |
| `L1.poison.grouped_gemm` | op_replay | L1 | pass | bit_match_frac=1.000e+00, cases=9 | 逐位全中（bit_match_frac） |
| `L1.poison.attn_padding` | op_replay | L1 | pass | bit_match_frac=1.000e+00, cases=8 | 逐位全中（bit_match_frac） |
| `L1.causal.attn` | op_replay | L1 | pass | bit_match_frac=1.000e+00, cases=8 | 逐位全中（bit_match_frac） |
| `L1.causal.attn_future_nan` | op_replay | L1 | pass | bit_match_frac=0.000e+00, cases=8 | 诊断：只报数，不判 |


## 环境与开关（进程内实测）

- `torch`：2.14.0a0+b2c75dd062.nv26.09
- `transformer_engine`：2.20.2+44f31bbe
- `cuda`：13.4
- `cublas`：13.8.0.4
- `device`：NVIDIA B200
- `allow_tf32_matmul_default`：True
- `allow_tf32_cudnn_default`：True
- `float32_matmul_precision_default`：high
- `env`：{'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': '1', 'NVIDIA_TF32_OVERRIDE': None, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'NVTE_ALLOW_NONDETERMINISTIC_ALGO': '0', 'NVTE_FLASH_ATTN': None, 'NVTE_FUSED_ATTN': None}
- `torch_deterministic_algorithms`：False

## 逐用例明细

列：rel_l2 对 FP64；cr_match_frac 对 FP64 正确舍入到输出 dtype 的逐位相同率；ulp 对正确舍入；bit_match_frac 对应当逐位相同的结果（训练抓数 / CPU 同一运算）；emulate_match_frac 对 CPU 模拟。

### `TF32.switch.matrix`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | bit_match_frac | nonfinite_mismatch | variant | op | readback.allow_tf32_matmul |
|---|---|---|---|---|---|---|---|---|---|
| `default.torch_mm_fp32` | 2.942e-04 | 7.248e-05 | 73059380 | 2.104e+06 | 9.537e-05 | 0 | default | torch_mm_fp32 | True |
| `default.te_general_gemm_fp32` | 2.942e-04 | 7.248e-05 | 73059380 | 2.104e+06 | 9.537e-05 | 0 | default | te_general_gemm_fp32 | True |
| `torch_false.torch_mm_fp32` | 1.146e-06 | 2.644e-02 | 3139288 | 4.680e+03 | 1 | 0 | torch_false | torch_mm_fp32 | False |
| `torch_false.te_general_gemm_fp32` | 2.942e-04 | 7.248e-05 | 73059380 | 2.104e+06 | 9.537e-05 | 0 | torch_false | te_general_gemm_fp32 | False |
| `highest.torch_mm_fp32` | 1.146e-06 | 2.644e-02 | 3139288 | 4.680e+03 | 1 | 0 | highest | torch_mm_fp32 | False |
| `highest.te_general_gemm_fp32` | 2.942e-04 | 7.248e-05 | 73059380 | 2.104e+06 | 9.537e-05 | 0 | highest | te_general_gemm_fp32 | False |
| `no_cublas_override_env.torch_mm_fp32` | 1.146e-06 | 2.644e-02 | 3139288 | 4.680e+03 | 1 | 0 | no_cublas_override_env | torch_mm_fp32 | False |
| `no_cublas_override_env.te_general_gemm_fp32` | 2.942e-04 | 7.248e-05 | 73059380 | 2.104e+06 | 9.537e-05 | 0 | no_cublas_override_env | te_general_gemm_fp32 | False |
| `NVIDIA_TF32_OVERRIDE=0.torch_mm_fp32` | 1.146e-06 | 2.644e-02 | 3139288 | 4.680e+03 | 1 | 0 | NVIDIA_TF32_OVERRIDE=0 | torch_mm_fp32 | True |
| `NVIDIA_TF32_OVERRIDE=0.te_general_gemm_fp32` | 1.146e-06 | 2.644e-02 | 3139288 | 4.680e+03 | 1 | 0 | NVIDIA_TF32_OVERRIDE=0 | te_general_gemm_fp32 | True |
| `NVIDIA_TF32_OVERRIDE=0+torch_false+no_cublas_override_env.torch_mm_fp32` | 1.146e-06 | 2.644e-02 | 3139288 | 4.680e+03 | 1 | 0 | NVIDIA_TF32_OVERRIDE=0+torch_false+no_cublas_override_env | torch_mm_fp32 | False |
| `NVIDIA_TF32_OVERRIDE=0+torch_false+no_cublas_override_env.te_general_gemm_fp32` | 1.146e-06 | 2.644e-02 | 3139288 | 4.680e+03 | 1 | 0 | NVIDIA_TF32_OVERRIDE=0+torch_false+no_cublas_override_env | te_general_gemm_fp32 | False |

### `TF32.switch.override_off`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | bit_match_frac | nonfinite_mismatch | variant | op | readback.allow_tf32_matmul |
|---|---|---|---|---|---|---|---|---|---|
| `NVIDIA_TF32_OVERRIDE=0.torch_mm_fp32` | 1.146e-06 | 2.644e-02 | 3139288 | 4.680e+03 | 1 | 0 | NVIDIA_TF32_OVERRIDE=0 | torch_mm_fp32 | True |
| `NVIDIA_TF32_OVERRIDE=0.te_general_gemm_fp32` | 1.146e-06 | 2.644e-02 | 3139288 | 4.680e+03 | 1 | 0 | NVIDIA_TF32_OVERRIDE=0 | te_general_gemm_fp32 | True |
| `NVIDIA_TF32_OVERRIDE=0+torch_false+no_cublas_override_env.torch_mm_fp32` | 1.146e-06 | 2.644e-02 | 3139288 | 4.680e+03 | 1 | 0 | NVIDIA_TF32_OVERRIDE=0+torch_false+no_cublas_override_env | torch_mm_fp32 | False |
| `NVIDIA_TF32_OVERRIDE=0+torch_false+no_cublas_override_env.te_general_gemm_fp32` | 1.146e-06 | 2.644e-02 | 3139288 | 4.680e+03 | 1 | 0 | NVIDIA_TF32_OVERRIDE=0+torch_false+no_cublas_override_env | te_general_gemm_fp32 | False |

### `TF32.router_bwd`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | bit_match_frac | nonfinite_mismatch | variant |
|---|---|---|---|---|---|---|---|
| `default.router_bwd.dx` | 1.671e-03 | 8.980e-01 | 2129 | 2.300e+01 | 1 | 0 | default |
| `default.router_bwd.dw` | 1.698e-03 | 8.966e-01 | 866 | 2.400e+01 | 1 | 0 | default |
| `torch_false.router_bwd.dx` | 1.671e-03 | 8.980e-01 | 2129 | 2.300e+01 | 1 | 0 | torch_false |
| `torch_false.router_bwd.dw` | 1.698e-03 | 8.966e-01 | 866 | 2.400e+01 | 1 | 0 | torch_false |
| `highest.router_bwd.dx` | 1.671e-03 | 8.980e-01 | 2129 | 2.300e+01 | 1 | 0 | highest |
| `highest.router_bwd.dw` | 1.698e-03 | 8.966e-01 | 866 | 2.400e+01 | 1 | 0 | highest |
| `no_cublas_override_env.router_bwd.dx` | 1.671e-03 | 8.980e-01 | 2129 | 2.300e+01 | 1 | 0 | no_cublas_override_env |
| `no_cublas_override_env.router_bwd.dw` | 1.698e-03 | 8.966e-01 | 866 | 2.400e+01 | 1 | 0 | no_cublas_override_env |
| `NVIDIA_TF32_OVERRIDE=0.router_bwd.dx` | 1.658e-03 | 0.999809 | 216 | 0 | 8.980e-01 | 0 | NVIDIA_TF32_OVERRIDE=0 |
| `NVIDIA_TF32_OVERRIDE=0.router_bwd.dw` | 1.685e-03 | 0.999493 | 41 | 0 | 8.966e-01 | 0 | NVIDIA_TF32_OVERRIDE=0 |
| `NVIDIA_TF32_OVERRIDE=0+torch_false+no_cublas_override_env.router_bwd.dx` | 1.658e-03 | 0.999809 | 216 | 0 | 8.980e-01 | 0 | NVIDIA_TF32_OVERRIDE=0+torch_false+no_cublas_override_env |
| `NVIDIA_TF32_OVERRIDE=0+torch_false+no_cublas_override_env.router_bwd.dw` | 1.685e-03 | 0.999493 | 41 | 0 | 8.966e-01 | 0 | NVIDIA_TF32_OVERRIDE=0+torch_false+no_cublas_override_env |

### `TF32.in_run`（pass）

| 用例 | probe_rel_l2_torch_mm_fp32 | probe_rel_l2_te_general_gemm_fp32 | allow_tf32_matmul | env.NVIDIA_TF32_OVERRIDE | env.TORCH_ALLOW_TF32_CUBLAS_OVERRIDE |
|---|---|---|---|---|---|
| `g61-obs` | 2.936e-04 | 2.936e-04 | True | None | 1 |
| `g61-notf32` | 2.301e-07 | 2.301e-07 | False | 0 | None |

### `TF32.gemm_inventory`（pass）

| 用例 | op | a_dtype | b_dtype | out_dtype | layout | grad | count |
|---|---|---|---|---|---|---|---|
| `te.general_gemm @ linear.py:forward < linear.py:_linear_eager < eval_frame.py:_fn` | te.general_gemm | bfloat16 | bfloat16 | bfloat16 | TN | False | 22 |
| `te.general_gemm @ linear.py:backward` | te.general_gemm | bfloat16 | bfloat16 | bfloat16 | NN | True | 22 |
| `te.general_gemm @ linear.py:_linear_backward_impl < linear.py:backward` | te.general_gemm | bfloat16 | bfloat16 | float32 | NT | True | 22 |
| `te.general_grouped_gemm @ grouped_linear.py:forward < eval_frame.py:_fn < jit.py:wrapper` | te.general_grouped_gemm | list[bfloat16] | list[bfloat16] | None | TN | False | 12 |
| `te.general_grouped_gemm @ ` | te.general_grouped_gemm | list[bfloat16] | list[bfloat16] | None | NN | True | 12 |
| `te.general_grouped_gemm @ ` | te.general_grouped_gemm | list[bfloat16] | list[bfloat16] | bfloat16 | NT | True | 12 |
| `te.general_gemm @ layernorm_linear.py:forward < eval_frame.py:_fn < jit.py:wrapper` | te.general_gemm | bfloat16 | bfloat16 | bfloat16 | TN | False | 10 |
| `te.general_gemm @ ` | te.general_gemm | bfloat16 | bfloat16 | bfloat16 | NN | True | 10 |
| `te.general_gemm @ layernorm_linear.py:backward` | te.general_gemm | bfloat16 | bfloat16 | float32 | NT | True | 10 |
| `te.general_gemm @ moe_utils.py:forward < moe_utils.py:router_gating_linear < router.py:gating` | te.general_gemm | bfloat16 | bfloat16 | float32 | TN | False | 6 |
| `te.general_gemm @ moe_utils.py:backward` | te.general_gemm | float32 | float32 | float32 | NN | True | 6 |
| `te.general_gemm @ moe_utils.py:backward` | te.general_gemm | float32 | float32 | float32 | NT | True | 6 |
| `torch.mm @ _compile.py:inner < layers.py:_linear_forward < layers.py:forward` | torch.mm | bfloat16 | bfloat16 |  |  | False | 2 |
| `torch.mm @ _compile.py:inner < layers.py:backward < autocast_mode.py:decorate_bwd` | torch.mm | bfloat16 | bfloat16 |  |  | False | 2 |

### `L1.kernel.experts.fwd.replay`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | bit_match_frac | nonfinite_mismatch | impl | rows | m_mod128_nonzero |
|---|---|---|---|---|---|---|---|---|---|
| `layers.1.experts.gemm1` | 1.659e-03 | 0.996902 | 868 | 1 | 1 | 0 | TE GroupedLinear（训练同一模块类） | 27527 | 24 |
| `layers.1.experts.act` | 2.875e-03 | 6.610e-01 | 2 | 2.000e+00 | 1 | 0 | silu(前半)·(后半+0)（BF16）× FP32 概率 → BF16 |  |  |
| `layers.1.experts.gemm2` | 1.657e-03 | 0.998938 | 584 | 1 | 1 | 0 | TE GroupedLinear |  |  |
| `layers.2.experts.gemm1` | 1.659e-03 | 0.997397 | 1130 | 1 | 1 | 0 | TE GroupedLinear（训练同一模块类） | 38631 | 24 |
| `layers.2.experts.act` | 2.869e-03 | 6.485e-01 | 2 | 2.000e+00 | 1 | 0 | silu(前半)·(后半+0)（BF16）× FP32 概率 → BF16 |  |  |
| `layers.2.experts.gemm2` | 1.659e-03 | 0.998959 | 709 | 1 | 1 | 0 | TE GroupedLinear |  |  |
| `layers.3.experts.gemm1` | 1.659e-03 | 0.997746 | 826 | 1 | 1 | 0 | TE GroupedLinear（训练同一模块类） | 36218 | 24 |
| `layers.3.experts.act` | 2.909e-03 | 6.504e-01 | 2 | 2.000e+00 | 1 | 0 | silu(前半)·(后半+0)（BF16）× FP32 概率 → BF16 |  |  |
| `layers.3.experts.gemm2` | 1.657e-03 | 0.998968 | 704 | 1 | 1 | 0 | TE GroupedLinear |  |  |

### `L1.kernel.experts.bwd.replay`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | bit_match_frac | nonfinite_mismatch | impl |
|---|---|---|---|---|---|---|---|
| `layers.1.experts.gemm2.dgrad` | 1.659e-03 | 0.997942 | 768 | 1 | 1 | 0 |  |
| `layers.1.experts.act.bwd` |  |  |  |  | 1 |  | autograd 过激活（与训练同一组 torch 算子） |
| `layers.1.experts.gemm1.dgrad` | 1.659e-03 | 0.998134 | 763 | 1 | 1 | 0 |  |
| `layers.2.experts.gemm2.dgrad` | 1.658e-03 | 0.998008 | 805 | 1 | 1 | 0 |  |
| `layers.2.experts.act.bwd` |  |  |  |  | 1 |  | autograd 过激活（与训练同一组 torch 算子） |
| `layers.2.experts.gemm1.dgrad` | 1.659e-03 | 0.998148 | 890 | 1 | 1 | 0 |  |
| `layers.3.experts.gemm2.dgrad` | 1.659e-03 | 0.997753 | 1290 | 1 | 1 | 0 |  |
| `layers.3.experts.act.bwd` |  |  |  |  | 1 |  | autograd 过激活（与训练同一组 torch 算子） |
| `layers.3.experts.gemm1.dgrad` | 1.659e-03 | 0.998140 | 1002 | 1 | 1 | 0 |  |

### `L1.kernel.experts.vs_loop`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | bit_match_frac | nonfinite_mismatch |
|---|---|---|---|---|---|---|
| `layers.1.experts.gemm1.fwd` | 1.659e-03 | 0.996902 | 868 | 1 | 1 | 0 |
| `layers.1.experts.gemm2.fwd` | 1.657e-03 | 0.998938 | 584 | 1 | 1 | 0 |
| `layers.1.experts.gemm1.dgrad` | 1.659e-03 | 0.998134 | 763 | 1 | 1 | 0 |
| `layers.2.experts.gemm1.fwd` | 1.659e-03 | 0.997397 | 1130 | 1 | 1 | 0 |
| `layers.2.experts.gemm2.fwd` | 1.659e-03 | 0.998959 | 709 | 1 | 1 | 0 |
| `layers.2.experts.gemm1.dgrad` | 1.659e-03 | 0.998148 | 890 | 1 | 1 | 0 |
| `layers.3.experts.gemm1.fwd` | 1.659e-03 | 0.997746 | 826 | 1 | 1 | 0 |
| `layers.3.experts.gemm2.fwd` | 1.657e-03 | 0.998968 | 704 | 1 | 1 | 0 |
| `layers.3.experts.gemm1.dgrad` | 1.659e-03 | 0.998140 | 1002 | 1 | 1 | 0 |

### `L1.kernel.experts.wgrad`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | bit_match_frac | nonfinite_mismatch | out_dtype |
|---|---|---|---|---|---|---|---|
| `layers.1.experts.gemm1.wgrad` | 2.263e-06 | 1.492e-02 | 677457920 | 2.687e+03 | 1 | 0 | float32 |
| `layers.1.experts.gemm2.wgrad` | 1.940e-06 | 1.295e-02 | 39830016 | 1.755e+03 | 1 | 0 | float32 |
| `layers.2.experts.gemm1.wgrad` | 3.262e-06 | 9.663e-03 | 49766400 | 4.067e+03 | 1 | 0 | float32 |
| `layers.2.experts.gemm2.wgrad` | 3.499e-06 | 7.935e-03 | 38409408 | 2.476e+03 | 1 | 0 | float32 |
| `layers.3.experts.gemm1.wgrad` | 2.313e-06 | 1.168e-02 | 744577648 | 4.206e+03 | 1 | 0 | float32 |
| `layers.3.experts.gemm2.wgrad` | 2.560e-06 | 8.899e-03 | 36819968 | 2.688e+03 | 1 | 0 | float32 |

### `L1.kernel.gemm_row_invariance`（pass）

| 用例 | bit_match_frac | impl | M | extra_rows | rows_changed |
|---|---|---|---|---|---|
| `gemm1.torch.M1+1` | 0.998047 | torch F.linear | 1 | 1 | 1 |
| `gemm1.grouped.M1+1` | 0.998047 | TE GroupedLinear | 1 | 1 | 1 |
| `gemm1.torch.M1+8` | 0.997396 | torch F.linear | 1 | 8 | 1 |
| `gemm1.grouped.M1+8` | 0.997396 | TE GroupedLinear | 1 | 8 | 1 |
| `gemm1.torch.M1+64` | 1 | torch F.linear | 1 | 64 | 0 |
| `gemm1.grouped.M1+64` | 1 | TE GroupedLinear | 1 | 64 | 0 |
| `gemm1.torch.M1+512` | 1 | torch F.linear | 1 | 512 | 0 |
| `gemm1.grouped.M1+512` | 1 | TE GroupedLinear | 1 | 512 | 0 |
| `gemm1.torch.M2+1` | 1 | torch F.linear | 2 | 1 | 0 |
| `gemm1.grouped.M2+1` | 1 | TE GroupedLinear | 2 | 1 | 0 |
| `gemm1.torch.M2+8` | 0.998047 | torch F.linear | 2 | 8 | 2 |
| `gemm1.grouped.M2+8` | 0.998047 | TE GroupedLinear | 2 | 8 | 2 |
| `gemm1.torch.M2+64` | 0.997559 | torch F.linear | 2 | 64 | 2 |
| `gemm1.grouped.M2+64` | 0.997559 | TE GroupedLinear | 2 | 64 | 2 |
| `gemm1.torch.M2+512` | 0.997559 | torch F.linear | 2 | 512 | 2 |
| `gemm1.grouped.M2+512` | 0.997559 | TE GroupedLinear | 2 | 512 | 2 |
| `gemm1.torch.M4+1` | 1 | torch F.linear | 4 | 1 | 0 |
| `gemm1.grouped.M4+1` | 1 | TE GroupedLinear | 4 | 1 | 0 |
| `gemm1.torch.M4+8` | 0.998372 | torch F.linear | 4 | 8 | 4 |
| `gemm1.grouped.M4+8` | 0.998372 | TE GroupedLinear | 4 | 8 | 4 |
| `gemm1.torch.M4+64` | 0.997640 | torch F.linear | 4 | 64 | 4 |
| `gemm1.grouped.M4+64` | 0.997640 | TE GroupedLinear | 4 | 64 | 4 |
| `gemm1.torch.M4+512` | 0.997640 | torch F.linear | 4 | 512 | 4 |
| `gemm1.grouped.M4+512` | 0.997640 | TE GroupedLinear | 4 | 512 | 4 |
| `gemm1.torch.M8+1` | 0.998332 | torch F.linear | 8 | 1 | 8 |
| `gemm1.grouped.M8+1` | 0.998332 | TE GroupedLinear | 8 | 1 | 8 |
| `gemm1.torch.M8+8` | 0.998332 | torch F.linear | 8 | 8 | 8 |
| `gemm1.grouped.M8+8` | 0.998332 | TE GroupedLinear | 8 | 8 | 8 |
| `gemm1.torch.M8+64` | 0.997640 | torch F.linear | 8 | 64 | 8 |
| `gemm1.grouped.M8+64` | 0.997640 | TE GroupedLinear | 8 | 64 | 8 |
| `gemm1.torch.M8+512` | 0.997640 | torch F.linear | 8 | 512 | 8 |
| `gemm1.grouped.M8+512` | 0.997640 | TE GroupedLinear | 8 | 512 | 8 |
| `gemm1.torch.M16+1` | 1 | torch F.linear | 16 | 1 | 0 |
| `gemm1.grouped.M16+1` | 1 | TE GroupedLinear | 16 | 1 | 0 |
| `gemm1.torch.M16+8` | 1 | torch F.linear | 16 | 8 | 0 |
| `gemm1.grouped.M16+8` | 1 | TE GroupedLinear | 16 | 8 | 0 |
| `gemm1.torch.M16+64` | 0.996948 | torch F.linear | 16 | 64 | 16 |
| `gemm1.grouped.M16+64` | 0.996948 | TE GroupedLinear | 16 | 64 | 16 |
| `gemm1.torch.M16+512` | 0.996948 | torch F.linear | 16 | 512 | 16 |
| `gemm1.grouped.M16+512` | 0.996948 | TE GroupedLinear | 16 | 512 | 16 |
| `gemm1.torch.M24+1` | 1 | torch F.linear | 24 | 1 | 0 |
| `gemm1.grouped.M24+1` | 1 | TE GroupedLinear | 24 | 1 | 0 |
| `gemm1.torch.M24+8` | 1 | torch F.linear | 24 | 8 | 0 |
| `gemm1.grouped.M24+8` | 1 | TE GroupedLinear | 24 | 8 | 0 |
| `gemm1.torch.M24+64` | 0.996935 | torch F.linear | 24 | 64 | 24 |
| `gemm1.grouped.M24+64` | 0.996935 | TE GroupedLinear | 24 | 64 | 24 |
| `gemm1.torch.M24+512` | 0.996935 | torch F.linear | 24 | 512 | 24 |
| `gemm1.grouped.M24+512` | 0.996935 | TE GroupedLinear | 24 | 512 | 24 |
| `gemm1.torch.M32+1` | 0.997060 | torch F.linear | 32 | 1 | 32 |
| `gemm1.grouped.M32+1` | 0.997060 | TE GroupedLinear | 32 | 1 | 32 |
| `gemm1.torch.M32+8` | 0.997060 | torch F.linear | 32 | 8 | 32 |
| `gemm1.grouped.M32+8` | 0.997060 | TE GroupedLinear | 32 | 8 | 32 |
| `gemm1.torch.M32+64` | 0.997060 | torch F.linear | 32 | 64 | 32 |
| `gemm1.grouped.M32+64` | 0.997060 | TE GroupedLinear | 32 | 64 | 32 |
| `gemm1.torch.M32+512` | 0.997060 | torch F.linear | 32 | 512 | 32 |
| `gemm1.grouped.M32+512` | 0.997060 | TE GroupedLinear | 32 | 512 | 32 |
| `gemm1.torch.M33+1` | 1 | torch F.linear | 33 | 1 | 0 |
| `gemm1.grouped.M33+1` | 1 | TE GroupedLinear | 33 | 1 | 0 |
| `gemm1.torch.M33+8` | 1 | torch F.linear | 33 | 8 | 0 |
| `gemm1.grouped.M33+8` | 1 | TE GroupedLinear | 33 | 8 | 0 |
| `gemm1.torch.M33+64` | 1 | torch F.linear | 33 | 64 | 0 |
| `gemm1.grouped.M33+64` | 1 | TE GroupedLinear | 33 | 64 | 0 |
| `gemm1.torch.M33+512` | 1 | torch F.linear | 33 | 512 | 0 |
| `gemm1.grouped.M33+512` | 1 | TE GroupedLinear | 33 | 512 | 0 |
| `gemm1.torch.M48+1` | 1 | torch F.linear | 48 | 1 | 0 |
| `gemm1.grouped.M48+1` | 1 | TE GroupedLinear | 48 | 1 | 0 |
| `gemm1.torch.M48+8` | 1 | torch F.linear | 48 | 8 | 0 |
| `gemm1.grouped.M48+8` | 1 | TE GroupedLinear | 48 | 8 | 0 |
| `gemm1.torch.M48+64` | 1 | torch F.linear | 48 | 64 | 0 |
| `gemm1.grouped.M48+64` | 1 | TE GroupedLinear | 48 | 64 | 0 |
| `gemm1.torch.M48+512` | 1 | torch F.linear | 48 | 512 | 0 |
| `gemm1.grouped.M48+512` | 1 | TE GroupedLinear | 48 | 512 | 0 |
| `gemm1.torch.M64+1` | 1 | torch F.linear | 64 | 1 | 0 |
| `gemm1.grouped.M64+1` | 1 | TE GroupedLinear | 64 | 1 | 0 |
| `gemm1.torch.M64+8` | 1 | torch F.linear | 64 | 8 | 0 |
| `gemm1.grouped.M64+8` | 1 | TE GroupedLinear | 64 | 8 | 0 |
| `gemm1.torch.M64+64` | 1 | torch F.linear | 64 | 64 | 0 |
| `gemm1.grouped.M64+64` | 1 | TE GroupedLinear | 64 | 64 | 0 |
| `gemm1.torch.M64+512` | 1 | torch F.linear | 64 | 512 | 0 |
| `gemm1.grouped.M64+512` | 1 | TE GroupedLinear | 64 | 512 | 0 |
| `gemm1.torch.M128+1` | 1 | torch F.linear | 128 | 1 | 0 |
| `gemm1.grouped.M128+1` | 1 | TE GroupedLinear | 128 | 1 | 0 |
| `gemm1.torch.M128+8` | 1 | torch F.linear | 128 | 8 | 0 |
| `gemm1.grouped.M128+8` | 1 | TE GroupedLinear | 128 | 8 | 0 |
| `gemm1.torch.M128+64` | 1 | torch F.linear | 128 | 64 | 0 |
| `gemm1.grouped.M128+64` | 1 | TE GroupedLinear | 128 | 64 | 0 |
| `gemm1.torch.M128+512` | 1 | torch F.linear | 128 | 512 | 0 |
| `gemm1.grouped.M128+512` | 1 | TE GroupedLinear | 128 | 512 | 0 |
| `gemm1.torch.M256+1` | 1 | torch F.linear | 256 | 1 | 0 |
| `gemm1.grouped.M256+1` | 1 | TE GroupedLinear | 256 | 1 | 0 |
| `gemm1.torch.M256+8` | 1 | torch F.linear | 256 | 8 | 0 |
| `gemm1.grouped.M256+8` | 1 | TE GroupedLinear | 256 | 8 | 0 |
| `gemm1.torch.M256+64` | 1 | torch F.linear | 256 | 64 | 0 |
| `gemm1.grouped.M256+64` | 1 | TE GroupedLinear | 256 | 64 | 0 |
| `gemm1.torch.M256+512` | 1 | torch F.linear | 256 | 512 | 0 |
| `gemm1.grouped.M256+512` | 1 | TE GroupedLinear | 256 | 512 | 0 |
| `gemm1.torch.M1024+1` | 1 | torch F.linear | 1024 | 1 | 0 |
| `gemm1.grouped.M1024+1` | 1 | TE GroupedLinear | 1024 | 1 | 0 |
| `gemm1.torch.M1024+8` | 1 | torch F.linear | 1024 | 8 | 0 |
| `gemm1.grouped.M1024+8` | 1 | TE GroupedLinear | 1024 | 8 | 0 |
| `gemm1.torch.M1024+64` | 1 | torch F.linear | 1024 | 64 | 0 |
| `gemm1.grouped.M1024+64` | 1 | TE GroupedLinear | 1024 | 64 | 0 |
| `gemm1.torch.M1024+512` | 1 | torch F.linear | 1024 | 512 | 0 |
| `gemm1.grouped.M1024+512` | 1 | TE GroupedLinear | 1024 | 512 | 0 |
| `gemm2.torch.M1+1` | 1 | torch F.linear | 1 | 1 | 0 |
| `gemm2.grouped.M1+1` | 1 | TE GroupedLinear | 1 | 1 | 0 |
| `gemm2.torch.M1+8` | 1 | torch F.linear | 1 | 8 | 0 |
| `gemm2.grouped.M1+8` | 1 | TE GroupedLinear | 1 | 8 | 0 |
| `gemm2.torch.M1+64` | 1 | torch F.linear | 1 | 64 | 0 |
| `gemm2.grouped.M1+64` | 1 | TE GroupedLinear | 1 | 64 | 0 |
| `gemm2.torch.M1+512` | 1 | torch F.linear | 1 | 512 | 0 |
| `gemm2.grouped.M1+512` | 1 | TE GroupedLinear | 1 | 512 | 0 |
| `gemm2.torch.M2+1` | 1 | torch F.linear | 2 | 1 | 0 |
| `gemm2.grouped.M2+1` | 1 | TE GroupedLinear | 2 | 1 | 0 |
| `gemm2.torch.M2+8` | 1 | torch F.linear | 2 | 8 | 0 |
| `gemm2.grouped.M2+8` | 1 | TE GroupedLinear | 2 | 8 | 0 |
| `gemm2.torch.M2+64` | 1 | torch F.linear | 2 | 64 | 0 |
| `gemm2.grouped.M2+64` | 1 | TE GroupedLinear | 2 | 64 | 0 |
| `gemm2.torch.M2+512` | 1 | torch F.linear | 2 | 512 | 0 |
| `gemm2.grouped.M2+512` | 1 | TE GroupedLinear | 2 | 512 | 0 |
| `gemm2.torch.M4+1` | 1 | torch F.linear | 4 | 1 | 0 |
| `gemm2.grouped.M4+1` | 1 | TE GroupedLinear | 4 | 1 | 0 |
| `gemm2.torch.M4+8` | 1 | torch F.linear | 4 | 8 | 0 |
| `gemm2.grouped.M4+8` | 1 | TE GroupedLinear | 4 | 8 | 0 |
| `gemm2.torch.M4+64` | 1 | torch F.linear | 4 | 64 | 0 |
| `gemm2.grouped.M4+64` | 1 | TE GroupedLinear | 4 | 64 | 0 |
| `gemm2.torch.M4+512` | 1 | torch F.linear | 4 | 512 | 0 |
| `gemm2.grouped.M4+512` | 1 | TE GroupedLinear | 4 | 512 | 0 |
| `gemm2.torch.M8+1` | 1 | torch F.linear | 8 | 1 | 0 |
| `gemm2.grouped.M8+1` | 1 | TE GroupedLinear | 8 | 1 | 0 |
| `gemm2.torch.M8+8` | 1 | torch F.linear | 8 | 8 | 0 |
| `gemm2.grouped.M8+8` | 1 | TE GroupedLinear | 8 | 8 | 0 |
| `gemm2.torch.M8+64` | 1 | torch F.linear | 8 | 64 | 0 |
| `gemm2.grouped.M8+64` | 1 | TE GroupedLinear | 8 | 64 | 0 |
| `gemm2.torch.M8+512` | 1 | torch F.linear | 8 | 512 | 0 |
| `gemm2.grouped.M8+512` | 1 | TE GroupedLinear | 8 | 512 | 0 |
| `gemm2.torch.M16+1` | 1 | torch F.linear | 16 | 1 | 0 |
| `gemm2.grouped.M16+1` | 1 | TE GroupedLinear | 16 | 1 | 0 |
| `gemm2.torch.M16+8` | 1 | torch F.linear | 16 | 8 | 0 |
| `gemm2.grouped.M16+8` | 1 | TE GroupedLinear | 16 | 8 | 0 |
| `gemm2.torch.M16+64` | 1 | torch F.linear | 16 | 64 | 0 |
| `gemm2.grouped.M16+64` | 1 | TE GroupedLinear | 16 | 64 | 0 |
| `gemm2.torch.M16+512` | 1 | torch F.linear | 16 | 512 | 0 |
| `gemm2.grouped.M16+512` | 1 | TE GroupedLinear | 16 | 512 | 0 |
| `gemm2.torch.M24+1` | 1 | torch F.linear | 24 | 1 | 0 |
| `gemm2.grouped.M24+1` | 1 | TE GroupedLinear | 24 | 1 | 0 |
| `gemm2.torch.M24+8` | 1 | torch F.linear | 24 | 8 | 0 |
| `gemm2.grouped.M24+8` | 1 | TE GroupedLinear | 24 | 8 | 0 |
| `gemm2.torch.M24+64` | 1 | torch F.linear | 24 | 64 | 0 |
| `gemm2.grouped.M24+64` | 1 | TE GroupedLinear | 24 | 64 | 0 |
| `gemm2.torch.M24+512` | 1 | torch F.linear | 24 | 512 | 0 |
| `gemm2.grouped.M24+512` | 1 | TE GroupedLinear | 24 | 512 | 0 |
| `gemm2.torch.M32+1` | 1 | torch F.linear | 32 | 1 | 0 |
| `gemm2.grouped.M32+1` | 1 | TE GroupedLinear | 32 | 1 | 0 |
| `gemm2.torch.M32+8` | 1 | torch F.linear | 32 | 8 | 0 |
| `gemm2.grouped.M32+8` | 1 | TE GroupedLinear | 32 | 8 | 0 |
| `gemm2.torch.M32+64` | 1 | torch F.linear | 32 | 64 | 0 |
| `gemm2.grouped.M32+64` | 1 | TE GroupedLinear | 32 | 64 | 0 |
| `gemm2.torch.M32+512` | 1 | torch F.linear | 32 | 512 | 0 |
| `gemm2.grouped.M32+512` | 1 | TE GroupedLinear | 32 | 512 | 0 |
| `gemm2.torch.M33+1` | 1 | torch F.linear | 33 | 1 | 0 |
| `gemm2.grouped.M33+1` | 1 | TE GroupedLinear | 33 | 1 | 0 |
| `gemm2.torch.M33+8` | 1 | torch F.linear | 33 | 8 | 0 |
| `gemm2.grouped.M33+8` | 1 | TE GroupedLinear | 33 | 8 | 0 |
| `gemm2.torch.M33+64` | 1 | torch F.linear | 33 | 64 | 0 |
| `gemm2.grouped.M33+64` | 1 | TE GroupedLinear | 33 | 64 | 0 |
| `gemm2.torch.M33+512` | 1 | torch F.linear | 33 | 512 | 0 |
| `gemm2.grouped.M33+512` | 1 | TE GroupedLinear | 33 | 512 | 0 |
| `gemm2.torch.M48+1` | 1 | torch F.linear | 48 | 1 | 0 |
| `gemm2.grouped.M48+1` | 1 | TE GroupedLinear | 48 | 1 | 0 |
| `gemm2.torch.M48+8` | 1 | torch F.linear | 48 | 8 | 0 |
| `gemm2.grouped.M48+8` | 1 | TE GroupedLinear | 48 | 8 | 0 |
| `gemm2.torch.M48+64` | 1 | torch F.linear | 48 | 64 | 0 |
| `gemm2.grouped.M48+64` | 1 | TE GroupedLinear | 48 | 64 | 0 |
| `gemm2.torch.M48+512` | 1 | torch F.linear | 48 | 512 | 0 |
| `gemm2.grouped.M48+512` | 1 | TE GroupedLinear | 48 | 512 | 0 |
| `gemm2.torch.M64+1` | 1 | torch F.linear | 64 | 1 | 0 |
| `gemm2.grouped.M64+1` | 1 | TE GroupedLinear | 64 | 1 | 0 |
| `gemm2.torch.M64+8` | 1 | torch F.linear | 64 | 8 | 0 |
| `gemm2.grouped.M64+8` | 1 | TE GroupedLinear | 64 | 8 | 0 |
| `gemm2.torch.M64+64` | 1 | torch F.linear | 64 | 64 | 0 |
| `gemm2.grouped.M64+64` | 1 | TE GroupedLinear | 64 | 64 | 0 |
| `gemm2.torch.M64+512` | 1 | torch F.linear | 64 | 512 | 0 |
| `gemm2.grouped.M64+512` | 1 | TE GroupedLinear | 64 | 512 | 0 |
| `gemm2.torch.M128+1` | 1 | torch F.linear | 128 | 1 | 0 |
| `gemm2.grouped.M128+1` | 1 | TE GroupedLinear | 128 | 1 | 0 |
| `gemm2.torch.M128+8` | 1 | torch F.linear | 128 | 8 | 0 |
| `gemm2.grouped.M128+8` | 1 | TE GroupedLinear | 128 | 8 | 0 |
| `gemm2.torch.M128+64` | 1 | torch F.linear | 128 | 64 | 0 |
| `gemm2.grouped.M128+64` | 1 | TE GroupedLinear | 128 | 64 | 0 |
| `gemm2.torch.M128+512` | 1 | torch F.linear | 128 | 512 | 0 |
| `gemm2.grouped.M128+512` | 1 | TE GroupedLinear | 128 | 512 | 0 |
| `gemm2.torch.M256+1` | 1 | torch F.linear | 256 | 1 | 0 |
| `gemm2.grouped.M256+1` | 1 | TE GroupedLinear | 256 | 1 | 0 |
| `gemm2.torch.M256+8` | 1 | torch F.linear | 256 | 8 | 0 |
| `gemm2.grouped.M256+8` | 1 | TE GroupedLinear | 256 | 8 | 0 |
| `gemm2.torch.M256+64` | 1 | torch F.linear | 256 | 64 | 0 |
| `gemm2.grouped.M256+64` | 1 | TE GroupedLinear | 256 | 64 | 0 |
| `gemm2.torch.M256+512` | 1 | torch F.linear | 256 | 512 | 0 |
| `gemm2.grouped.M256+512` | 1 | TE GroupedLinear | 256 | 512 | 0 |
| `gemm2.torch.M1024+1` | 1 | torch F.linear | 1024 | 1 | 0 |
| `gemm2.grouped.M1024+1` | 1 | TE GroupedLinear | 1024 | 1 | 0 |
| `gemm2.torch.M1024+8` | 1 | torch F.linear | 1024 | 8 | 0 |
| `gemm2.grouped.M1024+8` | 1 | TE GroupedLinear | 1024 | 8 | 0 |
| `gemm2.torch.M1024+64` | 1 | torch F.linear | 1024 | 64 | 0 |
| `gemm2.grouped.M1024+64` | 1 | TE GroupedLinear | 1024 | 64 | 0 |
| `gemm2.torch.M1024+512` | 1 | torch F.linear | 1024 | 512 | 0 |
| `gemm2.grouped.M1024+512` | 1 | TE GroupedLinear | 1024 | 512 | 0 |

### `L1.kernel.attn.fwd.replay`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | bit_match_frac | nonfinite_mismatch | impl |
|---|---|---|---|---|---|---|---|
| `layers.0.attn.core.out` | 1.659e-03 | 9.666e-01 | 1038 | 5.000e+00 | 1 | 0 | TE DotProductAttention（sbhd、causal、GQA 64/8） |
| `layers.1.attn.core.out` | 1.866e-03 | 8.194e-01 | 2033 | 4.600e+01 | 1 | 0 | TE DotProductAttention（sbhd、causal、GQA 64/8） |
| `layers.2.attn.core.out` | 1.914e-03 | 7.957e-01 | 1806 | 5.400e+01 | 1 | 0 | TE DotProductAttention（sbhd、causal、GQA 64/8） |
| `layers.3.attn.core.out` | 1.947e-03 | 7.842e-01 | 2012 | 5.500e+01 | 1 | 0 | TE DotProductAttention（sbhd、causal、GQA 64/8） |

### `L1.kernel.attn.bwd.replay`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | bit_match_frac | nonfinite_mismatch |
|---|---|---|---|---|---|---|
| `layers.0.attn.core.dq` | 2.100e-03 | 6.960e-01 | 11998 | 1.330e+02 | 1 | 0 |
| `layers.0.attn.core.dk` | 2.921e-03 | 5.970e-01 | 12955 | 1.200e+02 | 1 | 0 |
| `layers.0.attn.core.dv` | 2.449e-03 | 6.399e-01 | 13490 | 1.050e+02 | 1 | 0 |
| `layers.1.attn.core.dq` | 2.908e-03 | 6.270e-01 | 12000 | 1.540e+02 | 1 | 0 |
| `layers.1.attn.core.dk` | 3.140e-03 | 5.254e-01 | 12976 | 1.430e+02 | 1 | 0 |
| `layers.1.attn.core.dv` | 2.585e-03 | 5.477e-01 | 14004 | 1.400e+02 | 1 | 0 |
| `layers.2.attn.core.dq` | 3.423e-03 | 5.888e-01 | 11654 | 1.710e+02 | 1 | 0 |
| `layers.2.attn.core.dk` | 2.706e-03 | 5.466e-01 | 12796 | 1.348e+02 | 1 | 0 |
| `layers.2.attn.core.dv` | 2.600e-03 | 5.426e-01 | 13835 | 1.440e+02 | 1 | 0 |
| `layers.3.attn.core.dq` | 3.969e-03 | 5.738e-01 | 12556 | 1.780e+02 | 1 | 0 |
| `layers.3.attn.core.dk` | 2.783e-03 | 5.368e-01 | 12873 | 1.400e+02 | 1 | 0 |
| `layers.3.attn.core.dv` | 2.609e-03 | 5.439e-01 | 13937 | 1.380e+02 | 1 | 0 |

### `L1.poison.grouped_gemm`（pass）

| 用例 | bit_match_frac | m_mod128_nonzero |
|---|---|---|
| `tail_nan.y` | 1 | 24 |
| `tail_nan.dx` | 1 | 24 |
| `tail_nan.dw` | 1 | 24 |
| `tail_inf.y` | 1 | 24 |
| `tail_inf.dx` | 1 | 24 |
| `tail_inf.dw` | 1 | 24 |
| `group_isolation.y_other_rows` | 1 |  |
| `group_isolation.dx_other_rows` | 1 |  |
| `group_isolation.dw_other_experts` | 1 |  |

### `L1.poison.attn_padding`（pass）

| 用例 | bit_match_frac |
|---|---|
| `thd_pad_nan.out` | 1 |
| `thd_pad_nan.dq` | 1 |
| `thd_pad_nan.dk` | 1 |
| `thd_pad_nan.dv` | 1 |
| `thd_pad_inf.out` | 1 |
| `thd_pad_inf.dq` | 1 |
| `thd_pad_inf.dk` | 1 |
| `thd_pad_inf.dv` | 1 |

### `L1.causal.attn`（pass）

| 用例 | bit_match_frac |
|---|---|
| `p2048.future_changed.out` | 1 |
| `p2048.future_changed.dq` | 1 |
| `p2048.future_changed.dk` | 1 |
| `p2048.future_changed.dv` | 1 |
| `p3001.future_changed.out` | 1 |
| `p3001.future_changed.dq` | 1 |
| `p3001.future_changed.dk` | 1 |
| `p3001.future_changed.dv` | 1 |

### `L1.causal.attn_future_nan`（pass）

| 用例 | bit_match_frac | past_rows_nonfinite | first_bad_row |
|---|---|---|---|
| `p2048.future_nan.out` | 1 | 0 | None |
| `p2048.future_nan.dq` | 1 | 0 | None |
| `p2048.future_nan.dk` | 0 | 2048 | 0 |
| `p2048.future_nan.dv` | 0 | 2048 | 0 |
| `p3001.future_nan.out` | 9.384e-01 | 185 | 2816 |
| `p3001.future_nan.dq` | 9.384e-01 | 185 | 2816 |
| `p3001.future_nan.dk` | 0 | 3001 | 0 |
| `p3001.future_nan.dv` | 0 | 3001 | 0 |

数字全部来自 `results.json` 与 `details/`，本报告不手写数字。
