# janus 报告：hunyuan3-4l / phase1-l1

- 被测：`d-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/frameworks/megatron/l1_ops.py`（sha256 84b09668f343…）
- 结果：pass 25、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L1.gemm.router.replay` | op_replay | L1 | pass | rel_l2=3.956e-06, cr_match_frac=1.780e-05, ulp_max=472864, ulp_p999=1.730e+02 | 逐位全中（bit_match_frac） |
| `L1.gemm.router.upcast_fp32` | op_replay | L1 | pass | rel_l2=7.704e-06, cr_match_frac=1.017e-05, ulp_max=896544, ulp_p999=3.370e+02 | 诊断：只报数，不判 |
| `L1.gemm.attn_proj.replay` | op_replay | L1 | pass | rel_l2=1.900e-03, cr_match_frac=9.948e-01, ulp_max=812, ulp_p999=1.000e+00 | 逐位全中（bit_match_frac） |
| `L1.gemm.qkv.replay` | op_replay | L1 | pass | rel_l2=1.665e-03, cr_match_frac=9.978e-01, ulp_max=834, ulp_p999=1.000e+00 | 逐位全中（bit_match_frac） |
| `L1.norm.replay` | op_replay | L1 | pass | rel_l2=1.686e-03, cr_match_frac=1.000e+00, ulp_max=1, ulp_p999=0.000e+00 | 逐位全中（bit_match_frac） |
| `L1.elem.exp.fp32` | op_replay | L1 | pass | rel_l2=3.970e-08, cr_match_frac=6.904e-01, ulp_max=2, ulp_p999=2.000e+00 | 全部在门槛内 |
| `L1.elem.sigmoid.fp32` | op_replay | L1 | pass | rel_l2=4.536e-08, cr_match_frac=6.213e-01, ulp_max=3, ulp_p999=2.000e+00 | 全部在门槛内 |
| `L1.elem.silu.fp32` | op_replay | L1 | pass | rel_l2=5.086e-08, cr_match_frac=6.085e-01, ulp_max=3, ulp_p999=2.000e+00 | 全部在门槛内 |
| `L1.elem.rsqrt.fp32` | op_replay | L1 | pass | rel_l2=3.651e-08, cr_match_frac=7.407e-01, ulp_max=2, ulp_p999=1.000e+00 | 全部在门槛内 |
| `L1.elem.tail` | op_replay | L1 | pass | rel_l2=1.451e-08, cr_match_frac=7.293e-01, ulp_max=45182983, ulp_p999=4.289e+07 | 诊断：只报数，不判 |
| `L1.elem.bf16_vs_fp32` | op_replay | L1 | pass | rel_l2=1.245e-03, cr_match_frac=7.027e-01, ulp_max=42396024, ulp_p999=2.000e+00 | 诊断：只报数，不判 |
| `L1.elem.basic` | op_replay | L1 | pass | rel_l2=1.531e-03, cr_match_frac=1.000e+00, ulp_max=0, ulp_p999=0.000e+00 | 全部在门槛内 |
| `L1.rratio.clean` | op_replay | L1 | pass | rel_l2=1.652e-03, cr_match_frac=1.000e+00, ulp_max=1, ulp_p999=0.000e+00 | 全部在门槛内 |
| `L1.rratio.drill` | op_replay | L1 | pass | rel_l2=2.332e-03, cr_match_frac=6.231e-05, ulp_max=121160, ulp_p999=9.629e+04 | 演练：门槛按期望报警（超线：['r_ratio']） |
| `L1.emulate.gemm` | op_replay | L1 | pass | emulate_match_frac=1.000e+00, cases=11 | 逐位全中（emulate_match_frac） |
| `L1.emulate.other_guesses` | op_replay | L1 | pass | emulate_match_frac=3.751e-04, cases=17 | 诊断：只报数，不判 |
| `L1.extreme.rmsnorm` | op_replay | L1 | pass | rel_l2=1.992e-03, cr_match_frac=0.000e+00, ulp_max=1, ulp_p999=1.000e+00 | 全部在门槛内 |
| `L1.extreme.rmsnorm.overflow` | op_replay | L1 | pass | rel_l2=1.000e+00, cr_match_frac=0.000e+00, ulp_max=1115684864, ulp_p999=1.079e+09 | 诊断：只报数，不判 |
| `L1.extreme.sigmoid` | op_replay | L1 | pass | rel_l2=8.085e-10, cr_match_frac=9.412e-01, ulp_max=1, ulp_p999=9.850e-01 | 全部在门槛内 |
| `L1.extreme.softmax` | op_replay | L1 | pass | rel_l2=5.225e-08, cr_match_frac=2.188e-01, ulp_max=15, ulp_p999=1.462e+01 | 全部在门槛内 |
| `L1.ties.synthetic` | op_replay | L1 | pass | expect_match_frac=1.000e+00, cases=3 | 逐位全中（expect_match_frac） |
| `L1.ties.cpu_torch` | op_replay | L1 | pass | expect_match_frac=0.000e+00, cases=6 | 诊断：只报数，不判 |
| `L1.ties.real` | op_replay | L1 | pass | expect_match_frac=1.000e+00, cases=6 | 逐位全中（expect_match_frac） |
| `L1.vjp.fp32` | op_replay | L1 | pass | rel_l2=1.882e-07, cr_match_frac=2.068e-01, ulp_max=659847, ulp_p999=7.911e+02 | 全部在门槛内 |
| `L1.vjp.bf16` | op_replay | L1 | pass | rel_l2=1.725e-03, cr_match_frac=8.998e-01, ulp_max=14243, ulp_p999=2.300e+01 | 全部在门槛内 |


## 环境与开关（进程内实测）

- `torch`：2.14.0a0+b2c75dd062.nv26.09
- `transformer_engine`：2.20.2+44f31bbe
- `cuda`：13.4
- `cublas`：13.8.0.4
- `device`：NVIDIA B200
- `allow_tf32_matmul_default`：True
- `allow_tf32_cudnn_default`：True
- `float32_matmul_precision_default`：high
- `env`：{'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': '1', 'NVIDIA_TF32_OVERRIDE': None, 'CUBLAS_WORKSPACE_CONFIG': None, 'NVTE_ALLOW_NONDETERMINISTIC_ALGO': None}

## 逐用例明细

列：rel_l2 对 FP64；cr_match_frac 对 FP64 正确舍入到输出 dtype 的逐位相同率；ulp 对正确舍入；bit_match_frac 对应当逐位相同的结果（训练抓数 / CPU 同一运算）；emulate_match_frac 对 CPU 模拟。

### `L1.gemm.router.replay`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | bit_match_frac | nonfinite_mismatch |
|---|---|---|---|---|---|---|
| `layers.1.router.logits` | 3.758e-06 | 3.751e-04 | 152037 | 1.730e+02 | 1 | 0 |
| `layers.2.router.logits` | 3.797e-06 | 4.069e-05 | 472864 | 1.090e+02 | 1 | 0 |
| `layers.3.router.logits` | 3.956e-06 | 1.780e-05 | 16088 | 8.000e+01 | 1 | 0 |

### `L1.gemm.router.upcast_fp32`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | bit_match_frac | nonfinite_mismatch |
|---|---|---|---|---|---|---|
| `layers.1.router.logits.upcast_fp32_mm.tf32_off` | 7.637e-07 | 5.123e-02 | 26395 | 4.200e+01 | 4.921e-04 | 0 |
| `layers.1.router.logits.upcast_fp32_mm.tf32_on` | 7.361e-06 | 1.882e-04 | 330981 | 3.370e+02 | 4.094e-04 | 0 |
| `layers.2.router.logits.upcast_fp32_mm.tf32_off` | 7.464e-07 | 4.836e-02 | 35872 | 3.500e+01 | 4.323e-05 | 0 |
| `layers.2.router.logits.upcast_fp32_mm.tf32_on` | 7.429e-06 | 1.780e-05 | 896544 | 2.066e+02 | 5.213e-05 | 0 |
| `layers.3.router.logits.upcast_fp32_mm.tf32_off` | 7.644e-07 | 4.555e-02 | 1848 | 3.400e+01 | 2.035e-05 | 0 |
| `layers.3.router.logits.upcast_fp32_mm.tf32_on` | 7.704e-06 | 1.017e-05 | 27960 | 1.490e+02 | 2.670e-05 | 0 |

### `L1.gemm.attn_proj.replay`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | bit_match_frac | nonfinite_mismatch |
|---|---|---|---|---|---|---|
| `layers.0.attn.proj.te_gemm` | 1.900e-03 | 0.994784 | 709 | 1 | 1 | 0 |
| `layers.0.attn.proj.torch_linear` | 1.900e-03 | 0.994784 | 709 | 1 | 1 | 0 |
| `layers.1.attn.proj.te_gemm` | 1.662e-03 | 0.995332 | 749 | 1 | 1 | 0 |
| `layers.1.attn.proj.torch_linear` | 1.662e-03 | 0.995332 | 749 | 1 | 1 | 0 |
| `layers.2.attn.proj.te_gemm` | 1.659e-03 | 0.996120 | 812 | 1 | 1 | 0 |
| `layers.2.attn.proj.torch_linear` | 1.659e-03 | 0.996120 | 812 | 1 | 1 | 0 |
| `layers.3.attn.proj.te_gemm` | 1.659e-03 | 0.995615 | 663 | 1 | 1 | 0 |
| `layers.3.attn.proj.torch_linear` | 1.659e-03 | 0.995615 | 663 | 1 | 1 | 0 |

### `L1.gemm.qkv.replay`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | bit_match_frac | nonfinite_mismatch |
|---|---|---|---|---|---|---|
| `layers.0.attn.qkv` | 1.665e-03 | 0.997778 | 662 | 1 | 1 | 0 |
| `layers.1.attn.qkv` | 1.651e-03 | 0.997916 | 722 | 1 | 1 | 0 |
| `layers.2.attn.qkv` | 1.660e-03 | 0.998044 | 716 | 1 | 1 | 0 |
| `layers.3.attn.qkv` | 1.662e-03 | 0.998061 | 834 | 1 | 1 | 0 |

### `L1.norm.replay`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | bit_match_frac | nonfinite_mismatch |
|---|---|---|---|---|---|---|
| `layers.0.attn.q_norm` | 1.649e-03 | 0.999992 | 1 | 0 | 1 | 0 |
| `layers.0.attn.k_norm` | 1.686e-03 | 0.999993 | 1 | 0 | 1 | 0 |
| `layers.1.attn.q_norm` | 1.666e-03 | 0.999990 | 1 | 0 | 1 | 0 |
| `layers.1.attn.k_norm` | 1.683e-03 | 0.999991 | 1 | 0 | 1 | 0 |
| `layers.1.mlp_norm` | 1.670e-03 | 0.999988 | 1 | 0 | 1 | 0 |
| `layers.2.attn.q_norm` | 1.675e-03 | 0.999991 | 1 | 0 | 1 | 0 |
| `layers.2.attn.k_norm` | 1.683e-03 | 0.999990 | 1 | 0 | 1 | 0 |
| `layers.2.mlp_norm` | 1.661e-03 | 0.999988 | 1 | 0 | 1 | 0 |
| `layers.3.attn.q_norm` | 1.663e-03 | 0.999990 | 1 | 0 | 1 | 0 |
| `layers.3.attn.k_norm` | 1.665e-03 | 0.999989 | 1 | 0 | 1 | 0 |
| `layers.3.mlp_norm` | 1.662e-03 | 0.999988 | 1 | 0 | 1 | 0 |

### `L1.elem.exp.fp32`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | nonfinite_mismatch | n_ulp_gt4 |
|---|---|---|---|---|---|---|
| `exp.fp32.real.softmax_in` | 3.915e-08 | 6.914e-01 | 2 | 2.000e+00 | 0 | 0 |
| `exp.fp32.real.router_logits` | 3.970e-08 | 6.904e-01 | 2 | 2.000e+00 | 0 | 0 |
| `exp.fp32.unif[-87,88]` | 3.758e-08 | 6.968e-01 | 2 | 1 | 0 | 0 |
| `exp.fp32.unif[-10,10]` | 3.925e-08 | 6.980e-01 | 2 | 1 | 0 | 0 |

### `L1.elem.sigmoid.fp32`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | nonfinite_mismatch | n_ulp_gt4 |
|---|---|---|---|---|---|---|
| `sigmoid.fp32.real.router_logits` | 4.536e-08 | 6.213e-01 | 3 | 2.000e+00 | 0 | 0 |
| `sigmoid.fp32.unif[-30,30]` | 2.808e-08 | 6.641e-01 | 3 | 2.000e+00 | 0 | 0 |

### `L1.elem.silu.fp32`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | nonfinite_mismatch | n_ulp_gt4 |
|---|---|---|---|---|---|---|
| `silu.fp32.real.mlp_gate` | 5.086e-08 | 6.085e-01 | 2 | 2.000e+00 | 0 | 0 |
| `silu.fp32.unif[-30,30]` | 1.801e-08 | 7.128e-01 | 3 | 2.000e+00 | 0 | 0 |

### `L1.elem.rsqrt.fp32`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | nonfinite_mismatch | n_ulp_gt4 |
|---|---|---|---|---|---|---|
| `rsqrt.fp32.real.rmsnorm_ms` | 3.651e-08 | 7.407e-01 | 1 | 1 | 0 | 0 |
| `rsqrt.fp32.all[1,4)` | 3.245e-08 | 7.834e-01 | 2 | 1 | 0 | 0 |
| `rsqrt.fp32.logu[1e-37,1e37]` | 3.535e-08 | 7.749e-01 | 2 | 1 | 0 | 0 |

### `L1.elem.tail`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | nonfinite_mismatch | n_ulp_gt4 | x_ulp_gt4_min | x_ulp_gt4_max |
|---|---|---|---|---|---|---|---|---|
| `sigmoid.fp32.unif[-110,110]` | 1.451e-08 | 7.382e-01 | 2097023 | 1.682e+06 | 0 | 248970 | -1.018e+02 | -8.872e+01 |
| `silu.fp32.unif[-110,110]` | 2.564e-09 | 7.293e-01 | 45182983 | 4.289e+07 | 0 | 339171 | -1.064e+02 | -8.872e+01 |

### `L1.elem.bf16_vs_fp32`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | bit_match_frac | nonfinite_mismatch | n_ulp_gt4 | x_ulp_gt4_min | x_ulp_gt4_max |
|---|---|---|---|---|---|---|---|---|---|
| `exp.bf16_direct` | 1.245e-03 | 1 | 0 | 0 | 1 | 0 | 0 |  |  |
| `exp.fp32_then_bf16` | 1.245e-03 | 1 | 0 | 0 |  | 0 |  |  |  |
| `exp.fp32_out_on_bf16_grid` | 3.023e-08 | 9.637e-01 | 2 | 1 |  | 0 |  |  |  |
| `sigmoid.bf16_direct` | 3.153e-04 | 0.999877 | 24 | 0 | 1 | 0 | 4 | -9.050e+01 | -8.900e+01 |
| `sigmoid.fp32_then_bf16` | 3.153e-04 | 0.999877 | 24 | 0 |  | 0 |  |  |  |
| `sigmoid.fp32_out_on_bf16_grid` | 1.166e-08 | 9.615e-01 | 1589500 | 2.000e+00 |  | 0 |  |  |  |
| `silu.bf16_direct` | 3.332e-41 | 0.999740 | 647 | 0 | 1 | 0 | 13 | -9.500e+01 | -8.900e+01 |
| `silu.fp32_then_bf16` | 3.332e-41 | 0.999740 | 647 | 0 |  | 0 |  |  |  |
| `silu.fp32_out_on_bf16_grid` | 2.113e-45 | 9.599e-01 | 42396024 | 2.000e+00 |  | 0 |  |  |  |
| `rsqrt.bf16_direct` | 1.203e-03 | 1 | 0 | 0 | 1 | 0 | 0 |  |  |
| `rsqrt.fp32_then_bf16` | 1.203e-03 | 1 | 0 | 0 |  | 0 |  |  |  |
| `rsqrt.fp32_out_on_bf16_grid` | 5.660e-08 | 7.027e-01 | 1 | 1 |  | 0 |  |  |  |

### `L1.elem.basic`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | bit_match_frac | nonfinite_mismatch |
|---|---|---|---|---|---|---|
| `add.fp32` | 7.330e-09 | 1 | 0 | 0 | 1 | 0 |
| `mul.fp32` | 2.338e-08 | 1 | 0 | 0 | 1 | 0 |
| `add.bf16` | 2.973e-04 | 1 | 0 | 0 | 1 | 0 |
| `mul.bf16` | 1.531e-03 | 1 | 0 | 0 | 1 | 0 |
| `cast.fp32_to_bf16` | 1.472e-03 | 1 | 0 | 0 | 1 | 0 |
| `cast.fp32_to_fp16` | 1.847e-04 | 1 | 0 | 0 | 1 | 0 |
| `cast.bf16_to_fp32` | 0 | 1 | 0 | 0 | 1 | 0 |
| `cast.fp64_to_fp32` | 2.334e-08 | 1 | 0 | 0 | 1 | 0 |

### `L1.rratio.clean`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | r_ratio | cosine_test | nonfinite_mismatch |
|---|---|---|---|---|---|---|---|
| `layers.1.rmsnorm.torch_fp32_formula` | 1.652e-03 | 0.999988 | 1 | 0 | 1.000e+00 | 0.999999 | 0 |
| `layers.1.router.sigmoid.fp64_then_fp32` | 2.239e-08 | 1 | 0 | 0 | 4.899e-01 | 1.000000 | 0 |

### `L1.rratio.drill`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | r_ratio | cosine_test | nonfinite_mismatch |
|---|---|---|---|---|---|---|---|
| `layers.1.rmsnorm.extra_bf16_round` | 2.332e-03 | 7.417e-01 | 1 | 1 | 1.412e+00 | 0.999997 | 0 |
| `layers.1.router.sigmoid.logits_bf16` | 6.013e-04 | 6.231e-05 | 121160 | 9.629e+04 | 1.316e+04 | 1.000000 | 0 |

### `L1.emulate.gemm`（pass）

| 用例 | emulate_match_frac |
|---|---|
| `layers.1.router.logits` | 1 |
| `layers.2.router.logits` | 1 |
| `layers.3.router.logits` | 1 |
| `layers.0.attn.proj` | 1 |
| `layers.0.attn.qkv.gemm_part` | 1 |
| `layers.1.attn.proj` | 1 |
| `layers.1.attn.qkv.gemm_part` | 1 |
| `layers.2.attn.proj` | 1 |
| `layers.2.attn.qkv.gemm_part` | 1 |
| `layers.3.attn.proj` | 1 |
| `layers.3.attn.qkv.gemm_part` | 1 |

### `L1.emulate.other_guesses`（pass）

| 用例 | emulate_match_frac |
|---|---|
| `layers.1.router.logits.fp64` | 3.751e-04 |
| `layers.1.router.logits.seq` | 4.921e-04 |
| `layers.1.router.logits.tc_g16_b25_rne` | 4.476e-04 |
| `layers.1.router.logits.tc_g8_b25_rz` | 4.272e-04 |
| `layers.1.router.logits.tc_g16_b24_rz` | 8.789e-03 |
| `layers.0.attn.qkv.rmsnorm_part` | 0.999996 |
| `layers.0.attn.qkv.full_cpu` | 0.999435 |
| `layers.0.attn.qkv.fp64_round` | 0.997780 |
| `layers.1.attn.qkv.rmsnorm_part` | 0.999989 |
| `layers.1.attn.qkv.full_cpu` | 1 |
| `layers.1.attn.qkv.fp64_round` | 0.997894 |
| `layers.2.attn.qkv.rmsnorm_part` | 0.999990 |
| `layers.2.attn.qkv.full_cpu` | 0.999185 |
| `layers.2.attn.qkv.fp64_round` | 0.998010 |
| `layers.3.attn.qkv.rmsnorm_part` | 0.999991 |
| `layers.3.attn.qkv.full_cpu` | 1 |
| `layers.3.attn.qkv.fp64_round` | 0.997946 |

### `L1.extreme.rmsnorm`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | nonfinite_mismatch | out_zero_frac |
|---|---|---|---|---|---|---|
| `rmsnorm.torch_formula_fp32.normal` | 6.201e-08 | 3.127e-01 | 1 | 1 | 0 | 0 |
| `rmsnorm.F.rms_norm_fp32.normal` | 6.201e-08 | 3.127e-01 | 1 | 1 | 0 | 0 |
| `rmsnorm.te_fp32.normal` | 6.201e-08 | 3.127e-01 | 1 | 1 | 0 | 0 |
| `rmsnorm.te_bf16.normal` | 1.522e-03 | 1 | 0 | 0 | 0 | 0 |
| `rmsnorm.torch_formula_fp32.zero` | 0 | 1 | 0 | 0 | 0 | 1 |
| `rmsnorm.F.rms_norm_fp32.zero` | 0 | 1 | 0 | 0 | 0 | 1 |
| `rmsnorm.te_fp32.zero` | 0 | 1 | 0 | 0 | 0 | 1 |
| `rmsnorm.te_bf16.zero` | 0 | 1 | 0 | 0 | 0 | 1 |
| `rmsnorm.torch_formula_fp32.tiny_1e-20` | 6.300e-08 | 0 | 1 | 1 | 0 | 0 |
| `rmsnorm.F.rms_norm_fp32.tiny_1e-20` | 6.300e-08 | 0 | 1 | 1 | 0 | 0 |
| `rmsnorm.te_fp32.tiny_1e-20` | 6.300e-08 | 0 | 1 | 1 | 0 | 0 |
| `rmsnorm.te_bf16.tiny_1e-20` | 1.992e-03 | 1 | 0 | 0 | 0 | 0 |
| `rmsnorm.torch_formula_fp32.subnormal_1e-40` | 8.147e-08 | 0 | 1 | 1 | 0 | 0 |
| `rmsnorm.F.rms_norm_fp32.subnormal_1e-40` | 8.147e-08 | 0 | 1 | 1 | 0 | 0 |
| `rmsnorm.te_fp32.subnormal_1e-40` | 8.147e-08 | 0 | 1 | 1 | 0 | 0 |
| `rmsnorm.te_bf16.subnormal_1e-40` | 7.203e-04 | 1 | 0 | 0 | 0 | 0 |

### `L1.extreme.rmsnorm.overflow`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | nonfinite_mismatch | out_zero_frac |
|---|---|---|---|---|---|---|
| `rmsnorm.torch_formula_fp32.large_1e18` | 1 | 0 | 1080507156 | 1.079e+09 | 0 | 1 |
| `rmsnorm.F.rms_norm_fp32.large_1e18` | 1 | 0 | 1080507156 | 1.079e+09 | 0 | 1 |
| `rmsnorm.te_fp32.large_1e18` | 1 | 0 | 1080507156 | 1.079e+09 | 0 | 1 |
| `rmsnorm.te_bf16.large_1e18` | 1 | 0 | 16487 | 1.646e+04 | 0 | 1 |
| `rmsnorm.torch_formula_fp32.overflow_1e20` | 1 | 0 | 1080207830 | 1.078e+09 | 0 | 1 |
| `rmsnorm.F.rms_norm_fp32.overflow_1e20` | 1 | 0 | 1080207830 | 1.078e+09 | 0 | 1 |
| `rmsnorm.te_fp32.overflow_1e20` | 1 | 0 | 1080207830 | 1.078e+09 | 0 | 1 |
| `rmsnorm.te_bf16.overflow_1e20` | 1 | 0 | 16482 | 1.645e+04 | 0 | 1 |
| `rmsnorm.torch_formula_fp32.one_huge_3e38` | 1 | 0 | 1115684864 | 4.307e+07 | 0 | 1 |
| `rmsnorm.F.rms_norm_fp32.one_huge_3e38` | 1 | 0 | 1115684864 | 4.307e+07 | 0 | 1 |
| `rmsnorm.te_fp32.one_huge_3e38` | 1 | 0 | 1115684864 | 4.307e+07 | 0 | 1 |
| `rmsnorm.te_bf16.one_huge_3e38` | 1 | 0 | 17024 | 6.570e+02 | 0 | 1 |

### `L1.extreme.sigmoid`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | nonfinite_mismatch |
|---|---|---|---|---|---|
| `sigmoid.fp32` | 8.085e-10 | 9.412e-01 | 1 | 9.850e-01 | 0 |
| `sigmoid.bf16` | 8.085e-10 | 1 | 0 | 0 | 0 |

### `L1.extreme.softmax`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | nonfinite_mismatch |
|---|---|---|---|---|---|
| `softmax.fp32.all_zero` | 2.980e-08 | 1 | 0 | 0 | 0 |
| `softmax.fp32.all_1e30` | 2.980e-08 | 1 | 0 | 0 | 0 |
| `softmax.fp32.one_3e38` | 0 | 1 | 0 | 0 | 0 |
| `softmax.fp32.spread_1e30` | 0 | 1 | 0 | 0 | 0 |
| `softmax.fp32.half_neg_inf` | 4.896e-08 | 6.719e-01 | 4 | 3.809e+00 | 0 |
| `softmax.fp32.all_neg_inf` | 0 | 1 |  |  | 0 |
| `softmax.fp32.has_pos_inf` | 0 | 1 |  |  | 0 |
| `softmax.fp32.subnormal` | 2.980e-08 | 1 | 0 | 0 | 0 |
| `softmax.fp32.normal` | 5.225e-08 | 2.188e-01 | 15 | 1.462e+01 | 0 |

### `L1.ties.synthetic`（pass）

| 用例 | expect_match_frac | rows_differ | boundary_tie_rows |
|---|---|---|---|
| `levels4.gpu_topk_vs_stable` | 1 | 0 | 4096 |
| `levels32.gpu_topk_vs_stable` | 1 | 0 | 3401 |
| `all_equal.gpu_topk_vs_stable` | 1 | 0 | 256 |

### `L1.ties.cpu_torch`（pass）

| 用例 | expect_match_frac | rows_differ | boundary_tie_rows |
|---|---|---|---|
| `levels4.cpu_topk_vs_stable` | 0 | 4096 | 4096 |
| `levels4.gpu_vs_cpu_topk` | 0 | 4096 | 4096 |
| `levels32.cpu_topk_vs_stable` | 2.642e-01 | 3014 | 3401 |
| `levels32.gpu_vs_cpu_topk` | 2.642e-01 | 3014 | 3401 |
| `all_equal.cpu_topk_vs_stable` | 0 | 256 | 256 |
| `all_equal.gpu_vs_cpu_topk` | 0 | 256 | 256 |

### `L1.ties.real`（pass）

| 用例 | expect_match_frac | rows_differ | boundary_tie_rows |
|---|---|---|---|
| `layers.1.replay_vs_captured` | 1 | 0 | 0 |
| `layers.1.gpu_topk_vs_stable` | 1 | 0 | 0 |
| `layers.2.replay_vs_captured` | 1 | 0 | 0 |
| `layers.2.gpu_topk_vs_stable` | 1 | 0 | 0 |
| `layers.3.replay_vs_captured` | 1 | 0 | 0 |
| `layers.3.gpu_topk_vs_stable` | 1 | 0 | 0 |

### `L1.vjp.fp32`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | nonfinite_mismatch | fd_check_rel_err |
|---|---|---|---|---|---|---|
| `layers.1.rmsnorm.torch_fp32.dx` | 7.498e-08 | 4.403e-01 | 91213 | 1.100e+01 | 0 | 7.743e-11 |
| `layers.1.rmsnorm.torch_fp32.dw` | 1.882e-07 | 2.068e-01 | 659847 | 7.911e+02 | 0 |  |
| `layers.1.router.score_fn_fp32.dlogits` | 8.385e-08 | 9.714e-01 | 4166 | 1.100e+01 | 0 | 1.359e-10 |

### `L1.vjp.bf16`（pass）

| 用例 | rel_l2 | cr_match_frac | ulp_max | ulp_p999 | nonfinite_mismatch |
|---|---|---|---|---|---|
| `layers.1.rmsnorm.te_bf16.dx` | 1.658e-03 | 0.999986 | 3 | 0 | 0 |
| `layers.1.rmsnorm.te_bf16.dw` | 1.725e-03 | 1 | 0 | 0 | 0 |
| `layers.1.router.full.dx` | 1.675e-03 | 9.089e-01 | 14243 | 1.900e+01 | 0 |
| `layers.1.router.full.dw` | 1.676e-03 | 8.998e-01 | 1038 | 2.300e+01 | 0 |

数字全部来自 `results.json` 与 `details/`，本报告不手写数字。
