# janus 报告：hunyuan3-4l / phase1-causal-model

- 被测：`g61-obs`
- janus 0.1.0.dev0；目录 `models/hunyuan3/frameworks/megatron/causal_model.py`（sha256 e006be83e889…）
- 结果：pass 3、fail 3、inconclusive 0、error 0；**有硬检查失败**

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L2.causal.model.repeat` | op_replay | L2 | pass | bit_match_frac=1.000e+00, cases=1287 | 逐位全中（bit_match_frac） |
| `L2.causal.model.fwd` | op_replay | L2 | fail | bit_match_frac=2.441e-03, cases=98 | 没有逐位全中：bit_match_frac=0.00244140625 |
| `L2.causal.model.act_grad` | op_replay | L2 | fail | bit_match_frac=2.741e-01, cases=96 | 没有逐位全中：bit_match_frac=0.27405981046682354 |
| `L2.causal.model.param_grad.dense` | op_replay | L2 | fail | bit_match_frac=8.521e-02, cases=76 | 没有逐位全中：bit_match_frac=0.085205078125 |
| `L2.causal.model.param_grad.experts` | op_replay | L2 | pass | bit_match_frac=1.323e-02, cases=2304 | 诊断：只报数，不判 |
| `L2.causal.model.attribution` | op_replay | L2 | pass | past_tokens_out_differ=144, explained=144 | 诊断：第一个 MoE 层里过去位置输出不同的 token，有多少路由到了 token 数变了且 ≤ 32 的专家 |


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
- `determinism`：{'NVTE_ALLOW_NONDETERMINISTIC_ALGO': '0', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'NCCL_ALGO': 'Ring', 'TRITON_CACHE_AUTOTUNING': '1', 'torch_deterministic': True}
- `model_build_seconds`：9.4
- `experts_type`：TEGroupedMLP

## 逐用例明细

列：bit_match_frac 改了未来 token 之后对不改时的逐位相同率（只取位置 < p）；rel_diff 参数梯度的相对 L2 差。

### `L2.causal.model.repeat`（pass）

| 用例 | bit_match_frac |
|---|---|
| `repeat.embed.out` | 1 |
| `repeat.embed.out.grad` | 1 |
| `repeat.final_norm.out` | 1 |
| `repeat.final_norm.out.grad` | 1 |
| `repeat.layers.0.attn.core.out` | 1 |
| `repeat.layers.0.attn.core.out.grad` | 1 |
| `repeat.layers.0.attn.k_norm.out` | 1 |
| `repeat.layers.0.attn.k_norm.out.grad` | 1 |
| `repeat.layers.0.attn.out` | 1 |
| `repeat.layers.0.attn.out.grad` | 1 |
| `repeat.layers.0.attn.proj.out` | 1 |
| `repeat.layers.0.attn.proj.out.grad` | 1 |
| `repeat.layers.0.attn.q_norm.out` | 1 |
| `repeat.layers.0.attn.q_norm.out.grad` | 1 |
| `repeat.layers.0.attn.qkv.out` | 1 |
| `repeat.layers.0.attn.qkv.out.grad` | 1 |
| `repeat.layers.0.in` | 1 |
| `repeat.layers.0.in.grad` | 1 |
| `repeat.layers.0.mlp.fc1.out` | 1 |
| `repeat.layers.0.mlp.fc1.out.grad` | 1 |
| `repeat.layers.0.mlp.fc2.out` | 1 |
| `repeat.layers.0.mlp.fc2.out.grad` | 1 |
| `repeat.layers.0.mlp.out` | 1 |
| `repeat.layers.0.mlp.out.grad` | 1 |
| `repeat.layers.0.mlp_norm.out` | 1 |
| `repeat.layers.0.mlp_norm.out.grad` | 1 |
| `repeat.layers.0.out` | 1 |
| `repeat.layers.0.out.grad` | 1 |
| `repeat.layers.1.attn.core.out` | 1 |
| `repeat.layers.1.attn.core.out.grad` | 1 |
| `repeat.layers.1.attn.k_norm.out` | 1 |
| `repeat.layers.1.attn.k_norm.out.grad` | 1 |
| `repeat.layers.1.attn.out` | 1 |
| `repeat.layers.1.attn.out.grad` | 1 |
| `repeat.layers.1.attn.proj.out` | 1 |
| `repeat.layers.1.attn.proj.out.grad` | 1 |
| `repeat.layers.1.attn.q_norm.out` | 1 |
| `repeat.layers.1.attn.q_norm.out.grad` | 1 |
| `repeat.layers.1.attn.qkv.out` | 1 |
| `repeat.layers.1.attn.qkv.out.grad` | 1 |
| `repeat.layers.1.in` | 1 |
| `repeat.layers.1.in.grad` | 1 |
| `repeat.layers.1.mlp.out` | 1 |
| `repeat.layers.1.mlp.out.grad` | 1 |
| `repeat.layers.1.mlp.shared.out` | 1 |
| `repeat.layers.1.mlp.shared.out.grad` | 1 |
| `repeat.layers.1.mlp_norm.out` | 1 |
| `repeat.layers.1.mlp_norm.out.grad` | 1 |
| `repeat.layers.1.out` | 1 |
| `repeat.layers.1.out.grad` | 1 |
| `repeat.layers.2.attn.core.out` | 1 |
| `repeat.layers.2.attn.core.out.grad` | 1 |
| `repeat.layers.2.attn.k_norm.out` | 1 |
| `repeat.layers.2.attn.k_norm.out.grad` | 1 |
| `repeat.layers.2.attn.out` | 1 |
| `repeat.layers.2.attn.out.grad` | 1 |
| `repeat.layers.2.attn.proj.out` | 1 |
| `repeat.layers.2.attn.proj.out.grad` | 1 |
| `repeat.layers.2.attn.q_norm.out` | 1 |
| `repeat.layers.2.attn.q_norm.out.grad` | 1 |
| `repeat.layers.2.attn.qkv.out` | 1 |
| `repeat.layers.2.attn.qkv.out.grad` | 1 |
| `repeat.layers.2.in` | 1 |
| `repeat.layers.2.in.grad` | 1 |
| `repeat.layers.2.mlp.out` | 1 |
| `repeat.layers.2.mlp.out.grad` | 1 |
| `repeat.layers.2.mlp.shared.out` | 1 |
| `repeat.layers.2.mlp.shared.out.grad` | 1 |
| `repeat.layers.2.mlp_norm.out` | 1 |
| `repeat.layers.2.mlp_norm.out.grad` | 1 |
| `repeat.layers.2.out` | 1 |
| `repeat.layers.2.out.grad` | 1 |
| `repeat.layers.3.attn.core.out` | 1 |
| `repeat.layers.3.attn.core.out.grad` | 1 |
| `repeat.layers.3.attn.k_norm.out` | 1 |
| `repeat.layers.3.attn.k_norm.out.grad` | 1 |
| `repeat.layers.3.attn.out` | 1 |
| `repeat.layers.3.attn.out.grad` | 1 |
| `repeat.layers.3.attn.proj.out` | 1 |
| `repeat.layers.3.attn.proj.out.grad` | 1 |
| `repeat.layers.3.attn.q_norm.out` | 1 |
| `repeat.layers.3.attn.q_norm.out.grad` | 1 |
| `repeat.layers.3.attn.qkv.out` | 1 |
| `repeat.layers.3.attn.qkv.out.grad` | 1 |
| `repeat.layers.3.in` | 1 |
| `repeat.layers.3.in.grad` | 1 |
| `repeat.layers.3.mlp.out` | 1 |
| `repeat.layers.3.mlp.out.grad` | 1 |
| `repeat.layers.3.mlp.shared.out` | 1 |
| `repeat.layers.3.mlp.shared.out.grad` | 1 |
| `repeat.layers.3.mlp_norm.out` | 1 |
| `repeat.layers.3.mlp_norm.out.grad` | 1 |
| `repeat.layers.3.out` | 1 |
| `repeat.layers.3.out.grad` | 1 |
| `repeat.logits` | 1 |
| `repeat.logits.grad` | 1 |
| `repeat.loss.per_token` | 1 |
| `repeat.grad.decoder.final_layernorm.weight` | 1 |
| `repeat.grad.decoder.layers.0.mlp.linear_fc1.layer_norm_weight` | 1 |
| `repeat.grad.decoder.layers.0.mlp.linear_fc1.weight` | 1 |
| `repeat.grad.decoder.layers.0.mlp.linear_fc2.weight` | 1 |
| `repeat.grad.decoder.layers.0.self_attention.k_layernorm.weight` | 1 |
| `repeat.grad.decoder.layers.0.self_attention.linear_proj.weight` | 1 |
| `repeat.grad.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight` | 1 |
| `repeat.grad.decoder.layers.0.self_attention.linear_qkv.weight` | 1 |
| `repeat.grad.decoder.layers.0.self_attention.q_layernorm.weight` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight0` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight1` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight10` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight100` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight101` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight102` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight103` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight104` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight105` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight106` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight107` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight108` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight109` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight11` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight110` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight111` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight112` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight113` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight114` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight115` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight116` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight117` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight118` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight119` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight12` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight120` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight121` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight122` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight123` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight124` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight125` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight126` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight127` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight128` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight129` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight13` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight130` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight131` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight132` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight133` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight134` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight135` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight136` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight137` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight138` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight139` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight14` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight140` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight141` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight142` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight143` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight144` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight145` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight146` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight147` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight148` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight149` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight15` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight150` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight151` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight152` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight153` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight154` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight155` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight156` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight157` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight158` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight159` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight16` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight160` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight161` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight162` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight163` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight164` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight165` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight166` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight167` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight168` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight169` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight17` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight170` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight171` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight172` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight173` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight174` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight175` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight176` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight177` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight178` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight179` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight18` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight180` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight181` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight182` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight183` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight184` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight185` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight186` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight187` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight188` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight189` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight19` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight190` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight191` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight2` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight20` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight21` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight22` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight23` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight24` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight25` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight26` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight27` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight28` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight29` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight3` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight30` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight31` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight32` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight33` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight34` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight35` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight36` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight37` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight38` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight39` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight4` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight40` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight41` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight42` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight43` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight44` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight45` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight46` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight47` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight48` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight49` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight5` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight50` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight51` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight52` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight53` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight54` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight55` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight56` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight57` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight58` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight59` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight6` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight60` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight61` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight62` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight63` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight64` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight65` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight66` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight67` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight68` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight69` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight7` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight70` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight71` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight72` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight73` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight74` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight75` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight76` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight77` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight78` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight79` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight8` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight80` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight81` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight82` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight83` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight84` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight85` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight86` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight87` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight88` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight89` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight9` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight90` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight91` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight92` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight93` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight94` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight95` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight96` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight97` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight98` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc1.weight99` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight1` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight10` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight100` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight101` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight102` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight103` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight104` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight105` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight106` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight107` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight108` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight109` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight11` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight110` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight111` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight112` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight113` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight114` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight115` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight116` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight117` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight118` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight119` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight12` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight120` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight121` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight122` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight123` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight124` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight125` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight126` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight127` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight128` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight129` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight13` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight130` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight131` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight132` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight133` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight134` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight135` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight136` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight137` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight138` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight139` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight14` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight140` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight141` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight142` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight143` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight144` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight145` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight146` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight147` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight148` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight149` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight15` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight150` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight151` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight152` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight153` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight154` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight155` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight156` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight157` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight158` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight159` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight16` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight160` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight161` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight162` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight163` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight164` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight165` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight166` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight167` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight168` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight169` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight17` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight170` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight171` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight172` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight173` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight174` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight175` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight176` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight177` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight178` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight179` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight18` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight180` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight181` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight182` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight183` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight184` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight185` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight186` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight187` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight188` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight189` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight19` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight190` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight191` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight2` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight20` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight21` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight22` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight23` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight24` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight25` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight26` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight27` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight28` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight29` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight3` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight30` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight31` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight32` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight33` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight34` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight35` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight36` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight37` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight38` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight39` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight4` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight40` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight41` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight42` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight43` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight44` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight45` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight46` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight47` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight48` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight49` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight5` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight50` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight51` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight52` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight53` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight54` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight55` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight56` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight57` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight58` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight59` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight6` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight60` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight61` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight62` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight63` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight64` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight65` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight66` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight67` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight68` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight69` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight7` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight70` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight71` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight72` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight73` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight74` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight75` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight76` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight77` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight78` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight79` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight8` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight80` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight81` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight82` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight83` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight84` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight85` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight86` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight87` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight88` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight89` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight9` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight90` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight91` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight92` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight93` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight94` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight95` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight96` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight97` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight98` | 1 |
| `repeat.grad.decoder.layers.1.mlp.experts.linear_fc2.weight99` | 1 |
| `repeat.grad.decoder.layers.1.mlp.router.weight` | 1 |
| `repeat.grad.decoder.layers.1.mlp.shared_experts.linear_fc1.weight` | 1 |
| `repeat.grad.decoder.layers.1.mlp.shared_experts.linear_fc2.weight` | 1 |
| `repeat.grad.decoder.layers.1.pre_mlp_layernorm.weight` | 1 |
| `repeat.grad.decoder.layers.1.self_attention.k_layernorm.weight` | 1 |
| `repeat.grad.decoder.layers.1.self_attention.linear_proj.weight` | 1 |
| `repeat.grad.decoder.layers.1.self_attention.linear_qkv.layer_norm_weight` | 1 |
| `repeat.grad.decoder.layers.1.self_attention.linear_qkv.weight` | 1 |
| `repeat.grad.decoder.layers.1.self_attention.q_layernorm.weight` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight0` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight1` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight10` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight100` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight101` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight102` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight103` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight104` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight105` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight106` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight107` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight108` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight109` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight11` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight110` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight111` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight112` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight113` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight114` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight115` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight116` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight117` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight118` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight119` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight12` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight120` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight121` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight122` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight123` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight124` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight125` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight126` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight127` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight128` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight129` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight13` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight130` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight131` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight132` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight133` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight134` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight135` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight136` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight137` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight138` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight139` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight14` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight140` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight141` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight142` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight143` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight144` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight145` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight146` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight147` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight148` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight149` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight15` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight150` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight151` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight152` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight153` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight154` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight155` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight156` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight157` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight158` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight159` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight16` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight160` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight161` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight162` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight163` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight164` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight165` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight166` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight167` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight168` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight169` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight17` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight170` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight171` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight172` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight173` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight174` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight175` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight176` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight177` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight178` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight179` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight18` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight180` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight181` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight182` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight183` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight184` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight185` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight186` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight187` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight188` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight189` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight19` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight190` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight191` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight2` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight20` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight21` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight22` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight23` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight24` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight25` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight26` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight27` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight28` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight29` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight3` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight30` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight31` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight32` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight33` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight34` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight35` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight36` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight37` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight38` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight39` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight4` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight40` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight41` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight42` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight43` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight44` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight45` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight46` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight47` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight48` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight49` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight5` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight50` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight51` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight52` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight53` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight54` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight55` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight56` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight57` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight58` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight59` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight6` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight60` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight61` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight62` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight63` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight64` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight65` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight66` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight67` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight68` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight69` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight7` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight70` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight71` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight72` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight73` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight74` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight75` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight76` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight77` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight78` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight79` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight8` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight80` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight81` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight82` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight83` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight84` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight85` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight86` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight87` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight88` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight89` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight9` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight90` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight91` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight92` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight93` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight94` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight95` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight96` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight97` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight98` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc1.weight99` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight0` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight1` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight10` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight100` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight101` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight102` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight103` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight104` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight105` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight106` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight107` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight108` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight109` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight11` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight110` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight111` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight112` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight113` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight114` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight115` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight116` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight117` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight118` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight119` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight12` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight120` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight121` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight122` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight123` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight124` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight125` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight126` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight127` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight128` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight129` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight13` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight130` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight131` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight132` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight133` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight134` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight135` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight136` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight137` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight138` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight139` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight14` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight140` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight141` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight142` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight143` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight144` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight145` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight146` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight147` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight148` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight149` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight15` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight150` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight151` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight152` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight153` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight154` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight155` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight156` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight157` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight158` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight159` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight16` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight160` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight161` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight162` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight163` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight164` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight165` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight166` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight167` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight168` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight169` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight17` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight170` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight171` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight172` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight173` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight174` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight175` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight176` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight177` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight178` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight179` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight18` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight180` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight181` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight182` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight183` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight184` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight185` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight186` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight187` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight188` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight189` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight19` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight190` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight191` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight2` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight20` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight21` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight22` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight23` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight24` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight25` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight26` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight27` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight28` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight29` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight3` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight30` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight31` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight32` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight33` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight34` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight35` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight36` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight37` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight38` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight39` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight4` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight40` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight41` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight42` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight43` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight44` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight45` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight46` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight47` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight48` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight49` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight5` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight50` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight51` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight52` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight53` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight54` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight55` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight56` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight57` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight58` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight59` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight6` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight60` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight61` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight62` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight63` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight64` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight65` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight66` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight67` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight68` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight69` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight7` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight70` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight71` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight72` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight73` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight74` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight75` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight76` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight77` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight78` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight79` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight8` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight80` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight81` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight82` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight83` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight84` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight85` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight86` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight87` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight88` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight89` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight9` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight90` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight91` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight92` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight93` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight94` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight95` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight96` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight97` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight98` | 1 |
| `repeat.grad.decoder.layers.2.mlp.experts.linear_fc2.weight99` | 1 |
| `repeat.grad.decoder.layers.2.mlp.router.weight` | 1 |
| `repeat.grad.decoder.layers.2.mlp.shared_experts.linear_fc1.weight` | 1 |
| `repeat.grad.decoder.layers.2.mlp.shared_experts.linear_fc2.weight` | 1 |
| `repeat.grad.decoder.layers.2.pre_mlp_layernorm.weight` | 1 |
| `repeat.grad.decoder.layers.2.self_attention.k_layernorm.weight` | 1 |
| `repeat.grad.decoder.layers.2.self_attention.linear_proj.weight` | 1 |
| `repeat.grad.decoder.layers.2.self_attention.linear_qkv.layer_norm_weight` | 1 |
| `repeat.grad.decoder.layers.2.self_attention.linear_qkv.weight` | 1 |
| `repeat.grad.decoder.layers.2.self_attention.q_layernorm.weight` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight0` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight1` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight10` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight100` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight101` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight102` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight103` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight104` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight105` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight106` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight107` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight108` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight109` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight11` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight110` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight111` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight112` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight113` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight114` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight115` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight116` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight117` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight118` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight119` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight12` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight120` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight121` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight122` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight123` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight124` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight125` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight126` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight127` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight128` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight129` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight13` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight130` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight131` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight132` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight133` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight134` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight135` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight136` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight137` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight138` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight139` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight14` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight140` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight141` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight142` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight143` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight144` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight145` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight146` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight147` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight148` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight149` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight15` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight150` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight151` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight152` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight153` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight154` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight155` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight156` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight157` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight158` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight159` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight16` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight160` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight161` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight162` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight163` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight164` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight165` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight166` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight167` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight168` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight169` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight17` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight170` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight171` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight172` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight173` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight174` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight175` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight176` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight177` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight178` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight179` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight18` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight180` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight181` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight182` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight183` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight184` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight185` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight186` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight187` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight188` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight189` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight19` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight190` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight191` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight2` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight20` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight21` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight22` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight23` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight24` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight25` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight26` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight27` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight28` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight29` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight3` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight30` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight31` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight32` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight33` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight34` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight35` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight36` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight37` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight38` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight39` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight4` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight40` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight41` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight42` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight43` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight44` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight45` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight46` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight47` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight48` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight49` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight5` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight50` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight51` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight52` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight53` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight54` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight55` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight56` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight57` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight58` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight59` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight6` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight60` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight61` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight62` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight63` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight64` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight65` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight66` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight67` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight68` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight69` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight7` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight70` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight71` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight72` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight73` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight74` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight75` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight76` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight77` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight78` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight79` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight8` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight80` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight81` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight82` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight83` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight84` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight85` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight86` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight87` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight88` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight89` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight9` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight90` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight91` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight92` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight93` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight94` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight95` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight96` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight97` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight98` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc1.weight99` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight0` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight1` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight10` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight100` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight101` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight102` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight103` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight104` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight105` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight106` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight107` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight108` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight109` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight11` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight110` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight111` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight112` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight113` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight114` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight115` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight116` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight117` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight118` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight119` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight12` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight120` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight121` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight122` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight123` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight124` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight125` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight126` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight127` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight128` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight129` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight13` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight130` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight131` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight132` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight133` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight134` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight135` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight136` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight137` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight138` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight139` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight14` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight140` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight141` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight142` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight143` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight144` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight145` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight146` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight147` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight148` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight149` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight15` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight150` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight151` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight152` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight153` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight154` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight155` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight156` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight157` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight158` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight159` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight16` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight160` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight161` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight162` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight163` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight164` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight165` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight166` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight167` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight168` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight169` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight17` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight170` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight171` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight172` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight173` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight174` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight175` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight176` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight177` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight178` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight179` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight18` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight180` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight181` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight182` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight183` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight184` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight185` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight186` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight187` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight188` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight189` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight19` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight190` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight191` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight2` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight20` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight21` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight22` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight23` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight24` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight25` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight26` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight27` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight28` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight29` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight3` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight30` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight31` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight32` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight33` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight34` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight35` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight36` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight37` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight38` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight39` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight4` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight40` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight41` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight42` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight43` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight44` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight45` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight46` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight47` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight48` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight49` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight5` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight50` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight51` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight52` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight53` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight54` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight55` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight56` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight57` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight58` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight59` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight6` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight60` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight61` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight62` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight63` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight64` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight65` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight66` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight67` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight68` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight69` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight7` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight70` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight71` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight72` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight73` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight74` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight75` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight76` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight77` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight78` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight79` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight8` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight80` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight81` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight82` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight83` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight84` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight85` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight86` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight87` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight88` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight89` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight9` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight90` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight91` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight92` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight93` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight94` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight95` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight96` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight97` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight98` | 1 |
| `repeat.grad.decoder.layers.3.mlp.experts.linear_fc2.weight99` | 1 |
| `repeat.grad.decoder.layers.3.mlp.router.weight` | 1 |
| `repeat.grad.decoder.layers.3.mlp.shared_experts.linear_fc1.weight` | 1 |
| `repeat.grad.decoder.layers.3.mlp.shared_experts.linear_fc2.weight` | 1 |
| `repeat.grad.decoder.layers.3.pre_mlp_layernorm.weight` | 1 |
| `repeat.grad.decoder.layers.3.self_attention.k_layernorm.weight` | 1 |
| `repeat.grad.decoder.layers.3.self_attention.linear_proj.weight` | 1 |
| `repeat.grad.decoder.layers.3.self_attention.linear_qkv.layer_norm_weight` | 1 |
| `repeat.grad.decoder.layers.3.self_attention.linear_qkv.weight` | 1 |
| `repeat.grad.decoder.layers.3.self_attention.q_layernorm.weight` | 1 |
| `repeat.grad.embedding.word_embeddings.weight` | 1 |
| `repeat.grad.output_layer.weight` | 1 |

### `L2.causal.model.fwd`（fail）

| 用例 | bit_match_frac | p | p_mod128 | changed_tokens |
|---|---|---|---|---|
| `p2048.embed.out` | 1 | 2048 | 0 | 2048 |
| `p2048.final_norm.out` | 6.526e-01 | 2048 | 0 | 2048 |
| `p2048.layers.0.attn.core.out` | 1 | 2048 | 0 | 2048 |
| `p2048.layers.0.attn.k_norm.out` | 1 | 2048 | 0 | 2048 |
| `p2048.layers.0.attn.out` | 1 | 2048 | 0 | 2048 |
| `p2048.layers.0.attn.proj.out` | 1 | 2048 | 0 | 2048 |
| `p2048.layers.0.attn.q_norm.out` | 1 | 2048 | 0 | 2048 |
| `p2048.layers.0.attn.qkv.out` | 1 | 2048 | 0 | 2048 |
| `p2048.layers.0.in` | 1 | 2048 | 0 | 2048 |
| `p2048.layers.0.mlp.fc1.out` | 1 | 2048 | 0 | 2048 |
| `p2048.layers.0.mlp.fc2.out` | 1 | 2048 | 0 | 2048 |
| `p2048.layers.0.mlp.out` | 1 | 2048 | 0 | 2048 |
| `p2048.layers.0.mlp_norm.out` | 1 | 2048 | 0 | 2048 |
| `p2048.layers.0.out` | 1 | 2048 | 0 | 2048 |
| `p2048.layers.1.attn.core.out` | 1 | 2048 | 0 | 2048 |
| `p2048.layers.1.attn.k_norm.out` | 1 | 2048 | 0 | 2048 |
| `p2048.layers.1.attn.out` | 1 | 2048 | 0 | 2048 |
| `p2048.layers.1.attn.proj.out` | 1 | 2048 | 0 | 2048 |
| `p2048.layers.1.attn.q_norm.out` | 1 | 2048 | 0 | 2048 |
| `p2048.layers.1.attn.qkv.out` | 1 | 2048 | 0 | 2048 |
| `p2048.layers.1.in` | 1 | 2048 | 0 | 2048 |
| `p2048.layers.1.mlp.out` | 0.999925 | 2048 | 0 | 2048 |
| `p2048.layers.1.mlp.shared.out` | 1 | 2048 | 0 | 2048 |
| `p2048.layers.1.mlp_norm.out` | 1 | 2048 | 0 | 2048 |
| `p2048.layers.1.out` | 0.999988 | 2048 | 0 | 2048 |
| `p2048.layers.2.attn.core.out` | 0.992316 | 2048 | 0 | 2048 |
| `p2048.layers.2.attn.k_norm.out` | 0.999579 | 2048 | 0 | 2048 |
| `p2048.layers.2.attn.out` | 9.316e-01 | 2048 | 0 | 2048 |
| `p2048.layers.2.attn.proj.out` | 9.316e-01 | 2048 | 0 | 2048 |
| `p2048.layers.2.attn.q_norm.out` | 0.999575 | 2048 | 0 | 2048 |
| `p2048.layers.2.attn.qkv.out` | 0.999566 | 2048 | 0 | 2048 |
| `p2048.layers.2.in` | 0.999988 | 2048 | 0 | 2048 |
| `p2048.layers.2.mlp.out` | 7.012e-01 | 2048 | 0 | 2048 |
| `p2048.layers.2.mlp.shared.out` | 7.151e-01 | 2048 | 0 | 2048 |
| `p2048.layers.2.mlp_norm.out` | 9.872e-01 | 2048 | 0 | 2048 |
| `p2048.layers.2.out` | 9.159e-01 | 2048 | 0 | 2048 |
| `p2048.layers.3.attn.core.out` | 6.212e-01 | 2048 | 0 | 2048 |
| `p2048.layers.3.attn.k_norm.out` | 7.730e-01 | 2048 | 0 | 2048 |
| `p2048.layers.3.attn.out` | 4.354e-01 | 2048 | 0 | 2048 |
| `p2048.layers.3.attn.proj.out` | 4.354e-01 | 2048 | 0 | 2048 |
| `p2048.layers.3.attn.q_norm.out` | 7.788e-01 | 2048 | 0 | 2048 |
| `p2048.layers.3.attn.qkv.out` | 7.747e-01 | 2048 | 0 | 2048 |
| `p2048.layers.3.in` | 9.159e-01 | 2048 | 0 | 2048 |
| `p2048.layers.3.mlp.out` | 4.065e-01 | 2048 | 0 | 2048 |
| `p2048.layers.3.mlp.shared.out` | 4.164e-01 | 2048 | 0 | 2048 |
| `p2048.layers.3.mlp_norm.out` | 7.320e-01 | 2048 | 0 | 2048 |
| `p2048.layers.3.out` | 6.395e-01 | 2048 | 0 | 2048 |
| `p2048.logits` | 4.717e-01 | 2048 | 0 | 2048 |
| `p2048.loss.per_token` | 2.441e-03 | 2048 | 0 | 2048 |
| `p3001.embed.out` | 1 | 3001 | 57 | 1095 |
| `p3001.final_norm.out` | 6.457e-01 | 3001 | 57 | 1095 |
| `p3001.layers.0.attn.core.out` | 1 | 3001 | 57 | 1095 |
| `p3001.layers.0.attn.k_norm.out` | 1 | 3001 | 57 | 1095 |
| `p3001.layers.0.attn.out` | 1 | 3001 | 57 | 1095 |
| `p3001.layers.0.attn.proj.out` | 1 | 3001 | 57 | 1095 |
| `p3001.layers.0.attn.q_norm.out` | 1 | 3001 | 57 | 1095 |
| `p3001.layers.0.attn.qkv.out` | 1 | 3001 | 57 | 1095 |
| `p3001.layers.0.in` | 1 | 3001 | 57 | 1095 |
| `p3001.layers.0.mlp.fc1.out` | 1 | 3001 | 57 | 1095 |
| `p3001.layers.0.mlp.fc2.out` | 1 | 3001 | 57 | 1095 |
| `p3001.layers.0.mlp.out` | 1 | 3001 | 57 | 1095 |
| `p3001.layers.0.mlp_norm.out` | 1 | 3001 | 57 | 1095 |
| `p3001.layers.0.out` | 1 | 3001 | 57 | 1095 |
| `p3001.layers.1.attn.core.out` | 1 | 3001 | 57 | 1095 |
| `p3001.layers.1.attn.k_norm.out` | 1 | 3001 | 57 | 1095 |
| `p3001.layers.1.attn.out` | 1 | 3001 | 57 | 1095 |
| `p3001.layers.1.attn.proj.out` | 1 | 3001 | 57 | 1095 |
| `p3001.layers.1.attn.q_norm.out` | 1 | 3001 | 57 | 1095 |
| `p3001.layers.1.attn.qkv.out` | 1 | 3001 | 57 | 1095 |
| `p3001.layers.1.in` | 1 | 3001 | 57 | 1095 |
| `p3001.layers.1.mlp.out` | 0.999934 | 3001 | 57 | 1095 |
| `p3001.layers.1.mlp.shared.out` | 1 | 3001 | 57 | 1095 |
| `p3001.layers.1.mlp_norm.out` | 1 | 3001 | 57 | 1095 |
| `p3001.layers.1.out` | 0.999989 | 3001 | 57 | 1095 |
| `p3001.layers.2.attn.core.out` | 0.991656 | 3001 | 57 | 1095 |
| `p3001.layers.2.attn.k_norm.out` | 0.999628 | 3001 | 57 | 1095 |
| `p3001.layers.2.attn.out` | 9.297e-01 | 3001 | 57 | 1095 |
| `p3001.layers.2.attn.proj.out` | 9.297e-01 | 3001 | 57 | 1095 |
| `p3001.layers.2.attn.q_norm.out` | 0.999611 | 3001 | 57 | 1095 |
| `p3001.layers.2.attn.qkv.out` | 0.999607 | 3001 | 57 | 1095 |
| `p3001.layers.2.in` | 0.999989 | 3001 | 57 | 1095 |
| `p3001.layers.2.mlp.out` | 6.912e-01 | 3001 | 57 | 1095 |
| `p3001.layers.2.mlp.shared.out` | 7.080e-01 | 3001 | 57 | 1095 |
| `p3001.layers.2.mlp_norm.out` | 9.869e-01 | 3001 | 57 | 1095 |
| `p3001.layers.2.out` | 9.117e-01 | 3001 | 57 | 1095 |
| `p3001.layers.3.attn.core.out` | 6.101e-01 | 3001 | 57 | 1095 |
| `p3001.layers.3.attn.k_norm.out` | 7.642e-01 | 3001 | 57 | 1095 |
| `p3001.layers.3.attn.out` | 4.266e-01 | 3001 | 57 | 1095 |
| `p3001.layers.3.attn.proj.out` | 4.266e-01 | 3001 | 57 | 1095 |
| `p3001.layers.3.attn.q_norm.out` | 7.702e-01 | 3001 | 57 | 1095 |
| `p3001.layers.3.attn.qkv.out` | 7.662e-01 | 3001 | 57 | 1095 |
| `p3001.layers.3.in` | 9.117e-01 | 3001 | 57 | 1095 |
| `p3001.layers.3.mlp.out` | 4.013e-01 | 3001 | 57 | 1095 |
| `p3001.layers.3.mlp.shared.out` | 4.051e-01 | 3001 | 57 | 1095 |
| `p3001.layers.3.mlp_norm.out` | 7.247e-01 | 3001 | 57 | 1095 |
| `p3001.layers.3.out` | 6.327e-01 | 3001 | 57 | 1095 |
| `p3001.logits` | 4.669e-01 | 3001 | 57 | 1095 |
| `p3001.loss.per_token` | 2.999e-03 | 3001 | 57 | 1095 |

### `L2.causal.model.act_grad`（fail）

| 用例 | bit_match_frac | p | p_mod128 | changed_tokens |
|---|---|---|---|---|
| `p2048.embed.out.grad` | 4.988e-01 | 2048 | 0 | 2048 |
| `p2048.final_norm.out.grad` | 9.495e-01 | 2048 | 0 | 2048 |
| `p2048.layers.0.attn.core.out.grad` | 2.969e-01 | 2048 | 0 | 2048 |
| `p2048.layers.0.attn.k_norm.out.grad` | 4.468e-01 | 2048 | 0 | 2048 |
| `p2048.layers.0.attn.out.grad` | 5.067e-01 | 2048 | 0 | 2048 |
| `p2048.layers.0.attn.proj.out.grad` | 5.067e-01 | 2048 | 0 | 2048 |
| `p2048.layers.0.attn.q_norm.out.grad` | 2.766e-01 | 2048 | 0 | 2048 |
| `p2048.layers.0.attn.qkv.out.grad` | 3.016e-01 | 2048 | 0 | 2048 |
| `p2048.layers.0.in.grad` | 4.988e-01 | 2048 | 0 | 2048 |
| `p2048.layers.0.mlp.fc1.out.grad` | 3.654e-01 | 2048 | 0 | 2048 |
| `p2048.layers.0.mlp.fc2.out.grad` | 5.244e-01 | 2048 | 0 | 2048 |
| `p2048.layers.0.mlp.out.grad` | 5.244e-01 | 2048 | 0 | 2048 |
| `p2048.layers.0.mlp_norm.out.grad` | 5.067e-01 | 2048 | 0 | 2048 |
| `p2048.layers.0.out.grad` | 5.244e-01 | 2048 | 0 | 2048 |
| `p2048.layers.1.attn.core.out.grad` | 3.516e-01 | 2048 | 0 | 2048 |
| `p2048.layers.1.attn.k_norm.out.grad` | 3.375e-01 | 2048 | 0 | 2048 |
| `p2048.layers.1.attn.out.grad` | 5.580e-01 | 2048 | 0 | 2048 |
| `p2048.layers.1.attn.proj.out.grad` | 5.580e-01 | 2048 | 0 | 2048 |
| `p2048.layers.1.attn.q_norm.out.grad` | 3.165e-01 | 2048 | 0 | 2048 |
| `p2048.layers.1.attn.qkv.out.grad` | 3.090e-01 | 2048 | 0 | 2048 |
| `p2048.layers.1.in.grad` | 5.244e-01 | 2048 | 0 | 2048 |
| `p2048.layers.1.mlp.out.grad` | 5.740e-01 | 2048 | 0 | 2048 |
| `p2048.layers.1.mlp.shared.out.grad` | 5.740e-01 | 2048 | 0 | 2048 |
| `p2048.layers.1.mlp_norm.out.grad` | 3.245e-01 | 2048 | 0 | 2048 |
| `p2048.layers.1.out.grad` | 5.740e-01 | 2048 | 0 | 2048 |
| `p2048.layers.2.attn.core.out.grad` | 4.187e-01 | 2048 | 0 | 2048 |
| `p2048.layers.2.attn.k_norm.out.grad` | 3.896e-01 | 2048 | 0 | 2048 |
| `p2048.layers.2.attn.out.grad` | 6.270e-01 | 2048 | 0 | 2048 |
| `p2048.layers.2.attn.proj.out.grad` | 6.270e-01 | 2048 | 0 | 2048 |
| `p2048.layers.2.attn.q_norm.out.grad` | 3.627e-01 | 2048 | 0 | 2048 |
| `p2048.layers.2.attn.qkv.out.grad` | 3.572e-01 | 2048 | 0 | 2048 |
| `p2048.layers.2.in.grad` | 5.740e-01 | 2048 | 0 | 2048 |
| `p2048.layers.2.mlp.out.grad` | 6.595e-01 | 2048 | 0 | 2048 |
| `p2048.layers.2.mlp.shared.out.grad` | 6.595e-01 | 2048 | 0 | 2048 |
| `p2048.layers.2.mlp_norm.out.grad` | 3.528e-01 | 2048 | 0 | 2048 |
| `p2048.layers.2.out.grad` | 6.595e-01 | 2048 | 0 | 2048 |
| `p2048.layers.3.attn.core.out.grad` | 5.938e-01 | 2048 | 0 | 2048 |
| `p2048.layers.3.attn.k_norm.out.grad` | 3.412e-01 | 2048 | 0 | 2048 |
| `p2048.layers.3.attn.out.grad` | 8.163e-01 | 2048 | 0 | 2048 |
| `p2048.layers.3.attn.proj.out.grad` | 8.163e-01 | 2048 | 0 | 2048 |
| `p2048.layers.3.attn.q_norm.out.grad` | 3.684e-01 | 2048 | 0 | 2048 |
| `p2048.layers.3.attn.qkv.out.grad` | 3.381e-01 | 2048 | 0 | 2048 |
| `p2048.layers.3.in.grad` | 6.595e-01 | 2048 | 0 | 2048 |
| `p2048.layers.3.mlp.out.grad` | 9.365e-01 | 2048 | 0 | 2048 |
| `p2048.layers.3.mlp.shared.out.grad` | 9.365e-01 | 2048 | 0 | 2048 |
| `p2048.layers.3.mlp_norm.out.grad` | 3.802e-01 | 2048 | 0 | 2048 |
| `p2048.layers.3.out.grad` | 9.365e-01 | 2048 | 0 | 2048 |
| `p2048.logits.grad` | 6.397e-01 | 2048 | 0 | 2048 |
| `p3001.embed.out.grad` | 4.947e-01 | 3001 | 57 | 1095 |
| `p3001.final_norm.out.grad` | 9.461e-01 | 3001 | 57 | 1095 |
| `p3001.layers.0.attn.core.out.grad` | 2.933e-01 | 3001 | 57 | 1095 |
| `p3001.layers.0.attn.k_norm.out.grad` | 4.168e-01 | 3001 | 57 | 1095 |
| `p3001.layers.0.attn.out.grad` | 5.024e-01 | 3001 | 57 | 1095 |
| `p3001.layers.0.attn.proj.out.grad` | 5.024e-01 | 3001 | 57 | 1095 |
| `p3001.layers.0.attn.q_norm.out.grad` | 2.741e-01 | 3001 | 57 | 1095 |
| `p3001.layers.0.attn.qkv.out.grad` | 2.975e-01 | 3001 | 57 | 1095 |
| `p3001.layers.0.in.grad` | 4.947e-01 | 3001 | 57 | 1095 |
| `p3001.layers.0.mlp.fc1.out.grad` | 3.622e-01 | 3001 | 57 | 1095 |
| `p3001.layers.0.mlp.fc2.out.grad` | 5.199e-01 | 3001 | 57 | 1095 |
| `p3001.layers.0.mlp.out.grad` | 5.199e-01 | 3001 | 57 | 1095 |
| `p3001.layers.0.mlp_norm.out.grad` | 5.024e-01 | 3001 | 57 | 1095 |
| `p3001.layers.0.out.grad` | 5.199e-01 | 3001 | 57 | 1095 |
| `p3001.layers.1.attn.core.out.grad` | 3.473e-01 | 3001 | 57 | 1095 |
| `p3001.layers.1.attn.k_norm.out.grad` | 3.397e-01 | 3001 | 57 | 1095 |
| `p3001.layers.1.attn.out.grad` | 5.533e-01 | 3001 | 57 | 1095 |
| `p3001.layers.1.attn.proj.out.grad` | 5.533e-01 | 3001 | 57 | 1095 |
| `p3001.layers.1.attn.q_norm.out.grad` | 3.147e-01 | 3001 | 57 | 1095 |
| `p3001.layers.1.attn.qkv.out.grad` | 3.075e-01 | 3001 | 57 | 1095 |
| `p3001.layers.1.in.grad` | 5.199e-01 | 3001 | 57 | 1095 |
| `p3001.layers.1.mlp.out.grad` | 5.697e-01 | 3001 | 57 | 1095 |
| `p3001.layers.1.mlp.shared.out.grad` | 5.697e-01 | 3001 | 57 | 1095 |
| `p3001.layers.1.mlp_norm.out.grad` | 3.211e-01 | 3001 | 57 | 1095 |
| `p3001.layers.1.out.grad` | 5.697e-01 | 3001 | 57 | 1095 |
| `p3001.layers.2.attn.core.out.grad` | 4.148e-01 | 3001 | 57 | 1095 |
| `p3001.layers.2.attn.k_norm.out.grad` | 3.948e-01 | 3001 | 57 | 1095 |
| `p3001.layers.2.attn.out.grad` | 6.219e-01 | 3001 | 57 | 1095 |
| `p3001.layers.2.attn.proj.out.grad` | 6.219e-01 | 3001 | 57 | 1095 |
| `p3001.layers.2.attn.q_norm.out.grad` | 3.622e-01 | 3001 | 57 | 1095 |
| `p3001.layers.2.attn.qkv.out.grad` | 3.569e-01 | 3001 | 57 | 1095 |
| `p3001.layers.2.in.grad` | 5.697e-01 | 3001 | 57 | 1095 |
| `p3001.layers.2.mlp.out.grad` | 6.558e-01 | 3001 | 57 | 1095 |
| `p3001.layers.2.mlp.shared.out.grad` | 6.558e-01 | 3001 | 57 | 1095 |
| `p3001.layers.2.mlp_norm.out.grad` | 3.501e-01 | 3001 | 57 | 1095 |
| `p3001.layers.2.out.grad` | 6.558e-01 | 3001 | 57 | 1095 |
| `p3001.layers.3.attn.core.out.grad` | 5.899e-01 | 3001 | 57 | 1095 |
| `p3001.layers.3.attn.k_norm.out.grad` | 3.410e-01 | 3001 | 57 | 1095 |
| `p3001.layers.3.attn.out.grad` | 8.128e-01 | 3001 | 57 | 1095 |
| `p3001.layers.3.attn.proj.out.grad` | 8.128e-01 | 3001 | 57 | 1095 |
| `p3001.layers.3.attn.q_norm.out.grad` | 3.681e-01 | 3001 | 57 | 1095 |
| `p3001.layers.3.attn.qkv.out.grad` | 3.362e-01 | 3001 | 57 | 1095 |
| `p3001.layers.3.in.grad` | 6.558e-01 | 3001 | 57 | 1095 |
| `p3001.layers.3.mlp.out.grad` | 9.325e-01 | 3001 | 57 | 1095 |
| `p3001.layers.3.mlp.shared.out.grad` | 9.325e-01 | 3001 | 57 | 1095 |
| `p3001.layers.3.mlp_norm.out.grad` | 3.764e-01 | 3001 | 57 | 1095 |
| `p3001.layers.3.out.grad` | 9.325e-01 | 3001 | 57 | 1095 |
| `p3001.logits.grad` | 6.300e-01 | 3001 | 57 | 1095 |

### `L2.causal.model.param_grad.dense`（fail）

| 用例 | bit_match_frac | rel_diff | p | p_mod128 | changed_tokens |
|---|---|---|---|---|---|
| `p2048.grad.decoder.final_layernorm.weight` | 6.333e-01 | 1.475e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.0.mlp.linear_fc1.layer_norm_weight` | 5.134e-01 | 5.991e-04 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.0.mlp.linear_fc1.weight` | 4.621e-01 | 1.578e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.0.mlp.linear_fc2.weight` | 7.265e-01 | 1.511e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.0.self_attention.k_layernorm.weight` | 7.422e-01 | 2.132e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.0.self_attention.linear_proj.weight` | 8.009e-01 | 1.356e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight` | 6.667e-01 | 2.154e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.0.self_attention.linear_qkv.weight` | 4.826e-01 | 1.524e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.0.self_attention.q_layernorm.weight` | 7.422e-01 | 9.842e-04 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.router.weight` | 3.098e-01 | 4.453e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.shared_experts.linear_fc1.weight` | 4.494e-01 | 1.851e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.shared_experts.linear_fc2.weight` | 7.048e-01 | 1.651e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.pre_mlp_layernorm.weight` | 4.553e-01 | 2.040e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.self_attention.k_layernorm.weight` | 4.609e-01 | 3.197e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.self_attention.linear_proj.weight` | 7.318e-01 | 1.549e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.self_attention.linear_qkv.layer_norm_weight` | 6.033e-01 | 1.614e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.self_attention.linear_qkv.weight` | 4.492e-01 | 1.584e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.self_attention.q_layernorm.weight` | 4.766e-01 | 2.973e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.router.weight` | 2.956e-01 | 1.133e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.shared_experts.linear_fc1.weight` | 4.131e-01 | 1.933e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.shared_experts.linear_fc2.weight` | 5.785e-01 | 1.578e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.pre_mlp_layernorm.weight` | 3.154e-01 | 3.862e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.self_attention.k_layernorm.weight` | 5.156e-01 | 2.666e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.self_attention.linear_proj.weight` | 7.313e-01 | 1.545e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.self_attention.linear_qkv.layer_norm_weight` | 6.543e-01 | 1.554e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.self_attention.linear_qkv.weight` | 4.428e-01 | 1.506e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.self_attention.q_layernorm.weight` | 5.156e-01 | 1.643e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.router.weight` | 1.716e-01 | 5.787e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.shared_experts.linear_fc1.weight` | 4.478e-01 | 1.742e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.shared_experts.linear_fc2.weight` | 4.901e-01 | 1.146e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.pre_mlp_layernorm.weight` | 8.521e-02 | 4.833e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.self_attention.k_layernorm.weight` | 6.016e-01 | 2.376e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.self_attention.linear_proj.weight` | 6.740e-01 | 1.828e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.self_attention.linear_qkv.layer_norm_weight` | 6.526e-01 | 1.481e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.self_attention.linear_qkv.weight` | 3.968e-01 | 1.427e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.self_attention.q_layernorm.weight` | 6.641e-01 | 2.465e-03 | 2048 | 0 | 2048 |
| `p2048.grad.embedding.word_embeddings.weight` | 0.996769 | 5.001e-03 | 2048 | 0 | 2048 |
| `p2048.grad.output_layer.weight` | 6.732e-01 | 2.561e-03 | 2048 | 0 | 2048 |
| `p3001.grad.decoder.final_layernorm.weight` | 6.682e-01 | 1.300e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.0.mlp.linear_fc1.layer_norm_weight` | 5.486e-01 | 6.984e-04 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.0.mlp.linear_fc1.weight` | 4.973e-01 | 1.526e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.0.mlp.linear_fc2.weight` | 7.559e-01 | 1.395e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.0.self_attention.k_layernorm.weight` | 5.312e-01 | 2.186e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.0.self_attention.linear_proj.weight` | 8.239e-01 | 1.240e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight` | 6.707e-01 | 1.815e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.0.self_attention.linear_qkv.weight` | 4.959e-01 | 1.457e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.0.self_attention.q_layernorm.weight` | 5.391e-01 | 2.813e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.router.weight` | 3.339e-01 | 4.286e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.shared_experts.linear_fc1.weight` | 4.878e-01 | 1.666e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.shared_experts.linear_fc2.weight` | 7.371e-01 | 1.487e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.pre_mlp_layernorm.weight` | 4.978e-01 | 1.383e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.self_attention.k_layernorm.weight` | 5.078e-01 | 3.261e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.self_attention.linear_proj.weight` | 7.636e-01 | 1.386e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.self_attention.linear_qkv.layer_norm_weight` | 6.362e-01 | 1.460e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.self_attention.linear_qkv.weight` | 4.818e-01 | 1.400e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.self_attention.q_layernorm.weight` | 4.297e-01 | 2.715e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.router.weight` | 2.952e-01 | 9.956e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.shared_experts.linear_fc1.weight` | 4.398e-01 | 1.701e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.shared_experts.linear_fc2.weight` | 6.104e-01 | 1.396e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.pre_mlp_layernorm.weight` | 3.633e-01 | 2.019e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.self_attention.k_layernorm.weight` | 6.016e-01 | 1.849e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.self_attention.linear_proj.weight` | 7.584e-01 | 1.391e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.self_attention.linear_qkv.layer_norm_weight` | 6.892e-01 | 3.036e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.self_attention.linear_qkv.weight` | 4.722e-01 | 1.379e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.self_attention.q_layernorm.weight` | 6.484e-01 | 1.045e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.router.weight` | 1.420e-01 | 4.430e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.shared_experts.linear_fc1.weight` | 4.729e-01 | 1.550e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.shared_experts.linear_fc2.weight` | 5.201e-01 | 1.050e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.pre_mlp_layernorm.weight` | 1.067e-01 | 4.096e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.self_attention.k_layernorm.weight` | 5.859e-01 | 2.874e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.self_attention.linear_proj.weight` | 6.948e-01 | 1.675e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.self_attention.linear_qkv.layer_norm_weight` | 6.882e-01 | 1.091e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.self_attention.linear_qkv.weight` | 4.156e-01 | 1.288e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.self_attention.q_layernorm.weight` | 6.797e-01 | 1.709e-03 | 3001 | 57 | 1095 |
| `p3001.grad.embedding.word_embeddings.weight` | 0.995577 | 5.058e-03 | 3001 | 57 | 1095 |
| `p3001.grad.output_layer.weight` | 7.012e-01 | 1.893e-03 | 3001 | 57 | 1095 |

### `L2.causal.model.param_grad.experts`（pass）

| 用例 | bit_match_frac | rel_diff | p | p_mod128 | changed_tokens |
|---|---|---|---|---|---|
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight0` | 3.437e-01 | 4.417e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight1` | 3.621e-01 | 3.796e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight10` | 2.999e-01 | 4.962e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight100` | 3.296e-01 | 4.833e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight101` | 3.579e-01 | 4.061e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight102` | 2.346e-01 | 7.766e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight103` | 4.387e-01 | 2.843e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight104` | 3.228e-01 | 4.831e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight105` | 1.952e-01 | 1.107e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight106` | 2.402e-01 | 6.234e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight107` | 2.666e-01 | 5.394e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight108` | 3.589e-01 | 3.454e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight109` | 3.442e-01 | 3.432e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight11` | 3.244e-01 | 3.827e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight110` | 2.765e-01 | 4.484e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight111` | 3.394e-01 | 3.657e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight112` | 2.408e-01 | 7.183e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight113` | 4.256e-01 | 3.339e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight114` | 2.900e-01 | 4.394e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight115` | 3.889e-01 | 3.711e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight116` | 3.484e-01 | 4.268e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight117` | 3.535e-01 | 4.617e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight118` | 3.575e-01 | 3.844e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight119` | 3.262e-01 | 4.097e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight12` | 3.786e-01 | 3.466e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight120` | 4.284e-01 | 2.902e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight121` | 3.250e-01 | 5.076e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight122` | 2.744e-01 | 5.382e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight123` | 3.325e-01 | 3.968e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight124` | 3.679e-01 | 4.343e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight125` | 3.024e-01 | 4.261e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight126` | 2.875e-01 | 4.447e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight127` | 4.549e-01 | 2.925e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight128` | 3.408e-01 | 3.123e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight129` | 3.187e-01 | 4.888e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight13` | 2.971e-01 | 5.560e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight130` | 2.912e-01 | 3.991e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight131` | 2.640e-01 | 6.688e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight132` | 3.045e-01 | 4.249e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight133` | 2.886e-01 | 6.710e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight134` | 3.590e-01 | 3.818e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight135` | 3.293e-01 | 6.161e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight136` | 2.554e-01 | 6.310e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight137` | 4.414e-01 | 3.228e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight138` | 3.721e-01 | 3.337e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight139` | 3.195e-01 | 3.588e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight14` | 3.486e-01 | 4.169e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight140` | 3.733e-01 | 3.511e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight141` | 2.868e-01 | 5.521e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight142` | 2.862e-01 | 4.839e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight143` | 3.309e-01 | 4.078e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight144` | 2.806e-01 | 5.480e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight145` | 4.207e-01 | 2.772e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight146` | 2.843e-01 | 5.972e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight147` | 2.631e-01 | 6.506e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight148` | 2.647e-01 | 5.549e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight149` | 3.018e-01 | 3.822e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight15` | 3.256e-01 | 4.137e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight150` | 3.156e-01 | 5.494e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight151` | 3.155e-01 | 4.610e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight152` | 2.228e-01 | 9.927e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight153` | 3.344e-01 | 4.441e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight154` | 2.315e-01 | 7.210e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight155` | 3.729e-01 | 3.740e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight156` | 3.272e-01 | 5.440e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight157` | 2.557e-01 | 6.502e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight158` | 2.763e-01 | 5.459e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight159` | 3.240e-01 | 4.043e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight16` | 3.579e-01 | 4.759e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight160` | 3.966e-01 | 3.597e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight161` | 2.936e-01 | 5.068e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight162` | 3.093e-01 | 3.575e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight163` | 2.752e-01 | 6.058e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight164` | 5.980e-01 | 1.985e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight165` | 2.299e-01 | 6.034e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight166` | 3.033e-01 | 4.618e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight167` | 4.138e-01 | 3.736e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight168` | 2.997e-01 | 4.637e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight169` | 2.546e-01 | 6.604e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight17` | 2.244e-01 | 9.113e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight170` | 3.129e-01 | 5.558e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight171` | 2.632e-01 | 6.020e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight172` | 2.996e-01 | 5.750e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight173` | 3.174e-01 | 4.082e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight174` | 2.935e-01 | 4.934e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight175` | 3.299e-01 | 4.495e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight176` | 2.264e-01 | 5.309e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight177` | 3.124e-01 | 4.366e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight178` | 3.512e-01 | 3.547e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight179` | 2.500e-01 | 8.085e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight18` | 2.860e-01 | 5.472e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight180` | 2.645e-01 | 6.017e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight181` | 2.926e-01 | 5.379e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight182` | 3.137e-01 | 5.253e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight183` | 4.074e-01 | 4.818e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight184` | 3.288e-01 | 4.102e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight185` | 3.055e-01 | 5.266e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight186` | 4.427e-01 | 4.261e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight187` | 4.182e-01 | 3.118e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight188` | 4.008e-01 | 3.409e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight189` | 2.912e-01 | 5.484e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight19` | 2.773e-01 | 4.128e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight190` | 3.542e-01 | 3.007e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight191` | 2.057e-01 | 1.189e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight2` | 3.167e-01 | 3.898e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight20` | 3.719e-01 | 4.080e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight21` | 3.228e-01 | 4.764e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight22` | 2.868e-01 | 5.423e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight23` | 3.219e-01 | 4.901e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight24` | 2.005e-01 | 7.892e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight25` | 4.663e-01 | 2.691e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight26` | 2.823e-01 | 4.939e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight27` | 2.782e-01 | 6.352e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight28` | 3.575e-01 | 3.623e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight29` | 3.113e-01 | 3.639e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight3` | 3.499e-01 | 4.423e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight30` | 3.006e-01 | 5.053e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight31` | 2.456e-01 | 5.560e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight32` | 2.875e-01 | 5.679e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight33` | 2.897e-01 | 4.242e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight34` | 3.066e-01 | 8.045e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight35` | 2.608e-01 | 7.305e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight36` | 3.247e-01 | 4.258e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight37` | 3.534e-01 | 4.129e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight38` | 2.859e-01 | 5.506e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight39` | 2.820e-01 | 5.060e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight4` | 2.451e-01 | 5.925e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight40` | 2.351e-01 | 8.458e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight41` | 3.473e-01 | 3.439e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight42` | 2.709e-01 | 4.504e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight43` | 3.550e-01 | 3.996e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight44` | 3.145e-01 | 3.837e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight45` | 2.800e-01 | 5.103e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight46` | 2.602e-01 | 6.131e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight47` | 2.804e-01 | 7.702e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight48` | 3.795e-01 | 3.830e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight49` | 3.397e-01 | 4.723e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight5` | 2.479e-01 | 6.164e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight50` | 3.992e-01 | 3.505e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight51` | 3.625e-01 | 3.300e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight52` | 3.114e-01 | 5.092e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight53` | 4.271e-01 | 2.686e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight54` | 2.014e-01 | 1.096e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight55` | 2.835e-01 | 6.152e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight56` | 1.268e-01 | 1.858e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight57` | 2.874e-01 | 4.902e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight58` | 3.539e-01 | 3.396e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight59` | 4.391e-01 | 2.966e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight6` | 2.385e-01 | 7.428e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight60` | 2.688e-01 | 7.183e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight61` | 3.075e-01 | 5.491e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight62` | 2.906e-01 | 4.731e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight63` | 3.630e-01 | 4.264e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight64` | 4.347e-01 | 2.991e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight65` | 2.130e-01 | 7.415e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight66` | 2.504e-01 | 5.724e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight67` | 3.868e-01 | 3.482e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight68` | 3.914e-01 | 3.778e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight69` | 2.883e-01 | 5.095e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight7` | 3.357e-01 | 3.631e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight70` | 3.497e-01 | 3.238e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight71` | 3.436e-01 | 3.929e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight72` | 3.297e-01 | 4.468e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight73` | 3.635e-01 | 4.035e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight74` | 3.511e-01 | 3.904e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight75` | 2.807e-01 | 4.610e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight76` | 3.478e-01 | 4.312e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight77` | 3.715e-01 | 3.534e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight78` | 4.500e-01 | 2.987e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight79` | 3.687e-01 | 4.037e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight8` | 3.324e-01 | 5.026e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight80` | 3.269e-01 | 4.621e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight81` | 3.163e-01 | 4.861e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight82` | 4.179e-01 | 3.674e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight83` | 3.370e-01 | 4.440e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight84` | 3.712e-01 | 3.336e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight85` | 3.599e-01 | 4.442e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight86` | 3.453e-01 | 3.455e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight87` | 4.128e-01 | 3.885e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight88` | 3.450e-01 | 4.217e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight89` | 2.771e-01 | 5.977e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight9` | 1.784e-01 | 1.280e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight90` | 3.241e-01 | 4.964e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight91` | 4.692e-01 | 2.767e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight92` | 4.149e-01 | 3.345e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight93` | 2.699e-01 | 5.961e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight94` | 3.326e-01 | 3.974e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight95` | 3.876e-01 | 3.705e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight96` | 2.413e-01 | 6.985e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight97` | 4.589e-01 | 2.715e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight98` | 3.322e-01 | 4.410e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc1.weight99` | 3.439e-01 | 3.775e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 4.654e-01 | 3.574e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight1` | 4.586e-01 | 3.385e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight10` | 4.176e-01 | 4.142e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight100` | 4.684e-01 | 3.565e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight101` | 4.794e-01 | 3.271e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight102` | 3.256e-01 | 6.280e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight103` | 5.726e-01 | 2.467e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight104` | 4.216e-01 | 4.119e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight105` | 2.678e-01 | 9.775e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight106` | 3.405e-01 | 5.263e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight107` | 3.684e-01 | 4.958e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight108` | 4.704e-01 | 3.939e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight109` | 5.044e-01 | 3.099e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight11` | 4.435e-01 | 3.005e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight110` | 4.026e-01 | 3.977e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight111` | 4.898e-01 | 4.133e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight112` | 3.834e-01 | 5.321e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight113` | 5.194e-01 | 2.777e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight114` | 4.241e-01 | 3.366e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight115` | 4.756e-01 | 3.186e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight116` | 4.669e-01 | 3.765e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight117` | 4.757e-01 | 3.420e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight118` | 4.904e-01 | 3.612e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight119` | 4.175e-01 | 3.748e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight12` | 4.707e-01 | 2.904e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight120` | 5.698e-01 | 2.988e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight121` | 4.389e-01 | 3.652e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight122` | 3.796e-01 | 4.630e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight123` | 4.681e-01 | 3.720e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight124` | 4.935e-01 | 3.308e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight125` | 4.489e-01 | 3.437e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight126` | 3.975e-01 | 3.785e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight127` | 5.981e-01 | 2.416e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight128` | 5.007e-01 | 2.784e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight129` | 4.166e-01 | 4.011e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight13` | 3.868e-01 | 4.687e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight130` | 4.783e-01 | 3.130e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight131` | 3.469e-01 | 1.955e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight132` | 4.535e-01 | 3.603e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight133` | 3.683e-01 | 5.199e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight134` | 4.779e-01 | 3.100e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight135` | 4.778e-01 | 4.164e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight136` | 3.538e-01 | 5.370e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight137` | 5.286e-01 | 2.810e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight138` | 4.936e-01 | 2.724e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight139` | 4.805e-01 | 2.986e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight14` | 4.543e-01 | 3.698e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight140` | 4.828e-01 | 3.072e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight141` | 3.691e-01 | 4.754e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight142` | 3.837e-01 | 4.432e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight143` | 4.473e-01 | 3.549e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight144` | 4.475e-01 | 4.270e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight145` | 5.485e-01 | 2.444e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight146` | 3.969e-01 | 4.885e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight147` | 3.819e-01 | 5.889e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight148` | 3.759e-01 | 4.830e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight149` | 4.397e-01 | 3.736e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight15` | 4.624e-01 | 3.481e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight150` | 4.145e-01 | 4.498e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight151` | 4.044e-01 | 4.131e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight152` | 2.931e-01 | 1.020e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight153` | 4.400e-01 | 5.907e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight154` | 3.725e-01 | 5.132e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight155` | 5.040e-01 | 2.624e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight156` | 4.373e-01 | 3.619e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight157` | 3.548e-01 | 5.732e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight158` | 3.714e-01 | 4.865e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight159` | 4.829e-01 | 3.380e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight16` | 4.792e-01 | 3.320e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight160` | 4.871e-01 | 3.032e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight161` | 3.864e-01 | 4.525e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight162` | 4.732e-01 | 2.950e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight163` | 3.733e-01 | 4.701e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight164` | 6.580e-01 | 1.937e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight165` | 3.576e-01 | 4.229e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight166` | 4.401e-01 | 3.734e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight167` | 5.394e-01 | 2.815e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight168` | 4.604e-01 | 3.596e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight169` | 3.396e-01 | 6.167e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight17` | 2.940e-01 | 1.354e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight170` | 4.240e-01 | 7.220e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight171` | 3.629e-01 | 5.160e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight172` | 4.145e-01 | 4.087e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight173` | 4.549e-01 | 3.562e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight174` | 3.727e-01 | 4.639e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight175` | 4.321e-01 | 3.665e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight176` | 3.588e-01 | 4.272e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight177` | 4.283e-01 | 3.479e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight178` | 5.174e-01 | 2.690e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight179` | 3.529e-01 | 6.087e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight18` | 3.887e-01 | 4.764e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight180` | 3.537e-01 | 5.369e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight181` | 3.669e-01 | 4.832e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight182` | 3.953e-01 | 4.172e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight183` | 5.581e-01 | 3.461e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight184` | 5.445e-01 | 2.862e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight185` | 4.083e-01 | 3.543e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight186` | 5.596e-01 | 3.113e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight187` | 5.646e-01 | 2.521e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight188` | 5.222e-01 | 2.824e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight189` | 3.956e-01 | 4.814e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight19` | 4.498e-01 | 3.937e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight190` | 5.083e-01 | 2.713e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight191` | 2.810e-01 | 8.721e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight2` | 4.558e-01 | 3.197e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight20` | 4.977e-01 | 3.255e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight21` | 4.384e-01 | 4.073e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight22` | 3.969e-01 | 4.885e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight23` | 4.413e-01 | 3.676e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight24` | 2.904e-01 | 5.641e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight25` | 5.789e-01 | 2.099e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight26` | 4.040e-01 | 4.138e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight27` | 3.704e-01 | 5.327e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight28` | 5.092e-01 | 3.541e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight29` | 4.617e-01 | 3.371e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight3` | 4.810e-01 | 3.409e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight30` | 3.895e-01 | 4.392e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight31` | 3.367e-01 | 5.006e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight32` | 3.806e-01 | 5.905e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight33` | 4.304e-01 | 3.218e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight34` | 4.105e-01 | 2.059e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight35` | 3.355e-01 | 6.730e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight36` | 4.660e-01 | 3.572e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight37` | 4.810e-01 | 3.132e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight38` | 3.817e-01 | 4.082e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight39` | 3.771e-01 | 4.277e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight4` | 3.849e-01 | 5.274e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight40` | 3.231e-01 | 7.007e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight41` | 4.741e-01 | 2.832e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight42` | 4.476e-01 | 3.729e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight43` | 4.646e-01 | 3.228e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight44` | 4.601e-01 | 3.186e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight45` | 3.999e-01 | 4.056e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight46` | 3.331e-01 | 5.763e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight47` | 3.558e-01 | 7.346e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight48` | 4.735e-01 | 3.335e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight49` | 4.300e-01 | 3.979e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight5` | 3.390e-01 | 5.599e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight50` | 4.908e-01 | 3.437e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight51` | 5.483e-01 | 2.472e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight52` | 4.143e-01 | 3.802e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight53` | 5.552e-01 | 2.547e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight54` | 2.515e-01 | 1.334e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight55` | 3.772e-01 | 5.334e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight56` | 1.685e-01 | 1.806e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight57` | 4.036e-01 | 4.153e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight58` | 4.843e-01 | 3.209e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight59` | 5.348e-01 | 2.454e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight6` | 3.147e-01 | 6.696e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight60` | 3.680e-01 | 5.359e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight61` | 4.043e-01 | 4.161e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight62` | 3.922e-01 | 3.717e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight63` | 4.746e-01 | 3.333e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight64` | 5.164e-01 | 2.661e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight65` | 2.979e-01 | 4.742e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight66` | 3.416e-01 | 5.194e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight67` | 5.347e-01 | 3.073e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight68` | 5.008e-01 | 3.031e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight69` | 4.075e-01 | 4.075e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight7` | 5.041e-01 | 2.944e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight70` | 5.183e-01 | 3.113e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight71` | 4.656e-01 | 3.255e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight72` | 4.633e-01 | 3.147e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight73` | 4.898e-01 | 3.435e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight74` | 4.629e-01 | 3.416e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight75` | 3.898e-01 | 4.252e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight76` | 4.494e-01 | 3.612e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight77` | 5.286e-01 | 2.692e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight78` | 5.768e-01 | 2.312e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight79` | 5.102e-01 | 3.129e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight8` | 4.536e-01 | 3.864e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight80` | 4.079e-01 | 3.973e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight81` | 3.956e-01 | 4.299e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight82` | 5.006e-01 | 3.073e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight83` | 4.981e-01 | 3.216e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight84` | 5.379e-01 | 3.261e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight85` | 4.802e-01 | 3.887e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight86` | 5.260e-01 | 2.653e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight87` | 5.555e-01 | 3.145e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight88` | 4.695e-01 | 3.452e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight89` | 3.741e-01 | 4.426e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight9` | 2.500e-01 | 1.120e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight90` | 4.325e-01 | 4.184e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight91` | 5.892e-01 | 2.332e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight92` | 5.644e-01 | 3.136e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight93` | 3.650e-01 | 5.354e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight94` | 4.597e-01 | 3.499e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight95` | 5.276e-01 | 3.238e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight96` | 3.787e-01 | 7.167e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight97` | 5.844e-01 | 2.221e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight98` | 4.621e-01 | 3.546e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.1.mlp.experts.linear_fc2.weight99` | 5.063e-01 | 3.501e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight0` | 3.034e-01 | 5.538e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight1` | 2.567e-01 | 4.583e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight10` | 3.090e-01 | 4.951e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight100` | 2.636e-01 | 6.778e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight101` | 3.274e-01 | 5.035e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight102` | 4.056e-01 | 4.121e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight103` | 4.124e-01 | 3.953e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight104` | 4.070e-01 | 3.223e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight105` | 3.489e-01 | 4.338e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight106` | 4.470e-01 | 3.380e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight107` | 3.429e-01 | 4.841e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight108` | 4.131e-01 | 3.354e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight109` | 2.308e-01 | 9.368e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight11` | 3.819e-01 | 3.990e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight110` | 4.502e-01 | 3.184e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight111` | 3.607e-01 | 3.549e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight112` | 2.955e-01 | 5.293e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight113` | 8.047e-02 | 8.088e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight114` | 2.447e-01 | 9.007e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight115` | 3.285e-01 | 5.118e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight116` | 8.672e-02 | 8.684e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight117` | 4.849e-01 | 2.703e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight118` | 4.528e-01 | 2.932e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight119` | 3.883e-01 | 4.302e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight12` | 3.416e-01 | 5.105e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight120` | 2.820e-01 | 7.183e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight121` | 3.426e-01 | 4.819e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight122` | 2.663e-01 | 4.664e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight123` | 3.727e-01 | 4.508e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight124` | 2.908e-01 | 4.289e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight125` | 2.997e-01 | 3.872e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight126` | 3.978e-01 | 4.280e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight127` | 2.516e-01 | 7.701e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight128` | 3.485e-01 | 4.264e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight129` | 2.996e-01 | 5.971e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight13` | 3.479e-01 | 4.025e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight130` | 3.703e-01 | 4.253e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight131` | 2.711e-01 | 6.555e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight132` | 4.053e-01 | 3.410e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight133` | 4.247e-01 | 3.434e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight134` | 3.033e-01 | 6.324e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight135` | 3.467e-01 | 4.043e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight136` | 3.822e-01 | 3.792e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight137` | 4.017e-01 | 4.457e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight138` | 3.688e-01 | 4.254e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight139` | 2.922e-01 | 5.716e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight14` | 4.038e-01 | 3.538e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight140` | 3.024e-01 | 4.526e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight141` | 2.894e-01 | 6.004e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight142` | 3.786e-01 | 4.132e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight143` | 2.817e-01 | 4.841e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight144` | 2.991e-01 | 5.150e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight145` | 3.772e-01 | 4.259e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight146` | 2.894e-01 | 5.884e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight147` | 3.561e-01 | 4.612e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight148` | 3.939e-01 | 3.761e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight149` | 3.635e-01 | 4.082e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight15` | 2.545e-01 | 7.454e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight150` | 2.533e-01 | 5.809e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight151` | 2.505e-01 | 1.036e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight152` | 1.214e-01 | 4.475e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight153` | 3.693e-01 | 4.718e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight154` | 3.561e-01 | 4.091e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight155` | 4.174e-01 | 3.172e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight156` | 3.285e-01 | 4.634e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight157` | 3.086e-01 | 5.953e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight158` | 4.014e-01 | 4.029e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight159` | 2.786e-01 | 5.828e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight16` | 7.467e-02 | 9.856e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight160` | 3.065e-01 | 5.038e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight161` | 3.889e-01 | 4.058e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight162` | 4.340e-01 | 3.791e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight163` | 4.261e-01 | 3.082e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight164` | 3.431e-01 | 5.007e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight165` | 2.905e-01 | 5.731e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight166` | 3.796e-01 | 3.525e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight167` | 4.767e-01 | 4.081e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight168` | 3.675e-01 | 4.503e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight169` | 4.043e-01 | 3.796e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight17` | 3.956e-01 | 3.581e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight170` | 4.594e-01 | 4.124e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight171` | 2.861e-01 | 5.134e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight172` | 3.797e-01 | 4.284e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight173` | 4.918e-02 | 1.523e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight174` | 2.856e-01 | 5.838e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight175` | 9.068e-02 | 8.692e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight176` | 2.655e-01 | 6.676e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight177` | 3.708e-01 | 4.348e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight178` | 4.009e-01 | 4.257e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight179` | 4.009e-01 | 3.138e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight18` | 3.300e-01 | 4.596e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight180` | 4.560e-01 | 2.790e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight181` | 1.817e-01 | 1.487e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight182` | 3.894e-01 | 4.344e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight183` | 3.290e-01 | 5.279e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight184` | 1.045e-01 | 6.893e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight185` | 3.615e-01 | 4.609e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight186` | 3.490e-01 | 4.390e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight187` | 3.813e-01 | 4.191e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight188` | 3.345e-01 | 4.515e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight189` | 3.627e-01 | 4.568e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight19` | 2.685e-01 | 8.020e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight190` | 3.124e-01 | 6.116e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight191` | 3.914e-01 | 4.228e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight2` | 2.850e-01 | 6.271e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight20` | 4.329e-01 | 3.122e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight21` | 2.498e-01 | 9.447e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight22` | 4.430e-01 | 3.085e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight23` | 3.566e-01 | 3.546e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight24` | 2.965e-01 | 5.771e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight25` | 2.457e-01 | 8.012e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight26` | 3.328e-01 | 4.276e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight27` | 3.422e-01 | 4.223e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight28` | 2.128e-01 | 7.236e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight29` | 2.402e-01 | 7.444e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight3` | 4.025e-01 | 4.169e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight30` | 3.431e-01 | 5.280e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight31` | 8.931e-02 | 8.272e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight32` | 3.875e-01 | 3.814e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight33` | 3.437e-01 | 4.703e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight34` | 2.970e-01 | 5.830e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight35` | 3.122e-01 | 4.455e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight36` | 1.860e-01 | 9.619e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight37` | 4.596e-01 | 3.906e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight38` | 2.469e-01 | 6.250e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight39` | 3.631e-01 | 4.751e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight4` | 2.176e-01 | 8.804e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight40` | 2.949e-01 | 6.733e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight41` | 7.850e-02 | 9.258e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight42` | 2.498e-01 | 7.676e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight43` | 2.645e-01 | 5.498e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight44` | 2.536e-01 | 5.519e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight45` | 3.865e-01 | 3.592e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight46` | 3.189e-01 | 5.407e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight47` | 2.100e-01 | 1.273e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight48` | 3.811e-01 | 4.262e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight49` | 3.658e-01 | 4.643e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight5` | 4.255e-01 | 3.401e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight50` | 3.600e-01 | 3.876e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight51` | 3.532e-01 | 3.829e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight52` | 2.547e-01 | 7.782e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight53` | 2.311e-01 | 1.346e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight54` | 2.831e-01 | 6.883e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight55` | 5.633e-01 | 2.231e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight56` | 3.380e-01 | 4.462e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight57` | 2.349e-01 | 6.567e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight58` | 2.903e-01 | 6.974e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight59` | 3.487e-01 | 4.210e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight6` | 3.931e-01 | 4.525e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight60` | 3.656e-01 | 4.298e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight61` | 2.887e-01 | 5.907e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight62` | 3.257e-01 | 4.824e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight63` | 3.213e-01 | 4.435e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight64` | 2.985e-01 | 5.293e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight65` | 4.762e-01 | 2.706e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight66` | 2.894e-01 | 7.592e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight67` | 1.063e-01 | 5.348e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight68` | 4.511e-01 | 3.274e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight69` | 4.762e-01 | 4.037e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight7` | 3.658e-01 | 4.944e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight70` | 3.524e-01 | 3.869e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight71` | 2.794e-01 | 5.942e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight72` | 3.918e-01 | 3.779e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight73` | 4.694e-01 | 4.229e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight74` | 3.742e-01 | 4.367e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight75` | 4.068e-01 | 4.310e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight76` | 3.649e-01 | 4.799e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight77` | 4.118e-01 | 3.482e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight78` | 5.556e-01 | 2.272e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight79` | 9.528e-02 | 8.023e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight8` | 3.475e-01 | 4.103e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight80` | 2.402e-01 | 8.826e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight81` | 3.609e-01 | 3.730e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight82` | 3.560e-01 | 3.869e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight83` | 4.286e-01 | 2.973e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight84` | 2.810e-01 | 6.834e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight85` | 4.055e-01 | 3.990e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight86` | 3.879e-01 | 3.495e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight87` | 4.279e-01 | 3.226e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight88` | 2.730e-01 | 5.920e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight89` | 2.856e-01 | 4.901e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight9` | 1.648e-01 | 3.749e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight90` | 4.461e-01 | 3.014e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight91` | 2.577e-01 | 7.074e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight92` | 4.419e-01 | 3.185e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight93` | 3.351e-01 | 4.609e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight94` | 4.134e-01 | 2.628e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight95` | 2.050e-01 | 1.516e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight96` | 3.808e-01 | 4.105e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight97` | 3.866e-01 | 3.108e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight98` | 4.167e-01 | 3.782e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc1.weight99` | 2.413e-01 | 9.138e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight0` | 3.787e-01 | 4.813e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight1` | 4.302e-01 | 3.252e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight10` | 4.353e-01 | 4.174e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight100` | 3.433e-01 | 5.259e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight101` | 4.360e-01 | 3.837e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight102` | 4.843e-01 | 3.581e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight103` | 5.246e-01 | 2.836e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight104` | 4.771e-01 | 2.628e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight105` | 4.407e-01 | 3.037e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight106` | 5.484e-01 | 2.720e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight107` | 4.414e-01 | 3.708e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight108` | 5.270e-01 | 2.681e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight109` | 2.766e-01 | 1.156e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight11` | 4.518e-01 | 3.495e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight110` | 5.561e-01 | 2.698e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight111` | 5.033e-01 | 3.351e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight112` | 4.064e-01 | 3.931e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight113` | 1.254e-01 | 7.107e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight114` | 3.143e-01 | 7.920e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight115` | 4.281e-01 | 4.396e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight116` | 9.424e-02 | 8.155e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight117` | 5.800e-01 | 2.227e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight118` | 5.348e-01 | 2.503e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight119` | 4.977e-01 | 2.976e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight12` | 4.121e-01 | 4.069e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight120` | 3.860e-01 | 4.926e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight121` | 4.384e-01 | 3.778e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight122` | 3.833e-01 | 4.217e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight123` | 4.598e-01 | 3.606e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight124` | 4.529e-01 | 3.342e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight125` | 4.698e-01 | 3.211e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight126` | 5.096e-01 | 3.192e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight127` | 3.273e-01 | 6.474e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight128` | 4.459e-01 | 3.473e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight129` | 3.787e-01 | 5.165e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight13` | 4.589e-01 | 3.425e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight130` | 4.555e-01 | 2.909e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight131` | 3.442e-01 | 5.452e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight132` | 4.816e-01 | 2.935e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight133` | 5.303e-01 | 2.605e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight134` | 3.704e-01 | 5.909e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight135` | 4.712e-01 | 3.084e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight136` | 4.963e-01 | 2.732e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight137` | 5.140e-01 | 3.601e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight138` | 4.862e-01 | 3.020e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight139` | 3.600e-01 | 5.049e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight14` | 5.268e-01 | 3.214e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight140` | 4.236e-01 | 3.513e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight141` | 3.581e-01 | 5.303e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight142` | 4.618e-01 | 3.466e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight143` | 4.059e-01 | 3.811e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight144` | 4.321e-01 | 4.640e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight145` | 4.622e-01 | 3.598e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight146` | 3.693e-01 | 7.913e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight147` | 4.420e-01 | 3.916e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight148` | 5.232e-01 | 3.020e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight149` | 4.359e-01 | 3.629e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight15` | 3.426e-01 | 5.491e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight150` | 3.811e-01 | 4.371e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight151` | 3.103e-01 | 8.739e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight152` | 1.410e-01 | 2.939e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight153` | 4.667e-01 | 3.600e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight154` | 4.532e-01 | 3.147e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight155` | 5.435e-01 | 2.557e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight156` | 4.189e-01 | 3.918e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight157` | 3.859e-01 | 5.689e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight158` | 4.846e-01 | 3.327e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight159` | 3.639e-01 | 4.120e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight16` | 9.931e-02 | 1.039e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight160` | 3.965e-01 | 4.073e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight161` | 5.014e-01 | 3.250e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight162` | 5.217e-01 | 3.088e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight163` | 5.289e-01 | 2.382e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight164` | 4.253e-01 | 4.062e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight165` | 3.526e-01 | 5.316e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight166` | 5.125e-01 | 2.999e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight167` | 5.906e-01 | 3.049e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight168` | 4.687e-01 | 3.541e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight169` | 5.138e-01 | 2.959e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight17` | 5.231e-01 | 2.694e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight170` | 6.150e-01 | 2.841e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight171` | 3.785e-01 | 3.870e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight172` | 4.948e-01 | 3.215e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight173` | 5.985e-02 | 1.517e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight174` | 3.888e-01 | 5.108e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight175` | 1.052e-01 | 1.030e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight176` | 3.736e-01 | 6.419e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight177` | 4.820e-01 | 3.393e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight178` | 4.913e-01 | 3.444e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight179` | 5.533e-01 | 2.510e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight18` | 4.548e-01 | 3.756e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight180` | 5.281e-01 | 2.252e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight181` | 2.392e-01 | 1.308e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight182` | 5.295e-01 | 2.889e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight183` | 4.077e-01 | 4.734e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight184` | 1.059e-01 | 1.016e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight185` | 4.494e-01 | 3.827e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight186` | 4.583e-01 | 3.312e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight187` | 5.067e-01 | 3.301e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight188` | 4.216e-01 | 4.089e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight189` | 4.554e-01 | 3.637e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight19` | 3.593e-01 | 6.300e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight190` | 3.948e-01 | 6.207e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight191` | 5.144e-01 | 3.076e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight2` | 3.861e-01 | 5.367e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight20` | 5.156e-01 | 2.716e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight21` | 3.501e-01 | 7.617e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight22` | 5.404e-01 | 2.603e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight23` | 5.251e-01 | 2.736e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight24` | 3.780e-01 | 4.035e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight25` | 3.022e-01 | 5.440e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight26` | 4.741e-01 | 3.382e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight27` | 4.750e-01 | 3.349e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight28` | 2.979e-01 | 5.668e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight29` | 3.184e-01 | 5.986e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight3` | 4.881e-01 | 3.238e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight30` | 4.075e-01 | 4.435e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight31` | 1.087e-01 | 7.433e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight32` | 5.175e-01 | 2.893e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight33` | 4.103e-01 | 4.260e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight34` | 4.113e-01 | 4.982e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight35` | 4.960e-01 | 3.680e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight36` | 2.989e-01 | 7.135e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight37` | 6.033e-01 | 2.961e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight38` | 4.102e-01 | 4.404e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight39` | 4.567e-01 | 3.789e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight4` | 2.962e-01 | 6.981e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight40` | 4.007e-01 | 5.530e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight41` | 8.581e-02 | 8.397e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight42` | 3.210e-01 | 6.305e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight43` | 3.781e-01 | 3.574e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight44` | 3.929e-01 | 4.178e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight45` | 5.030e-01 | 2.839e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight46` | 4.009e-01 | 4.603e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight47` | 2.743e-01 | 1.275e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight48` | 5.024e-01 | 3.234e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight49` | 4.822e-01 | 3.959e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight5` | 5.558e-01 | 2.670e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight50` | 4.965e-01 | 2.855e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight51` | 4.913e-01 | 3.028e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight52` | 3.033e-01 | 7.714e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight53` | 2.944e-01 | 1.134e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight54` | 3.574e-01 | 5.982e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight55` | 6.575e-01 | 1.910e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight56` | 4.025e-01 | 4.254e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight57` | 3.986e-01 | 4.779e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight58` | 3.867e-01 | 5.095e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight59` | 4.697e-01 | 3.188e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight6` | 4.784e-01 | 3.832e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight60` | 4.559e-01 | 3.698e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight61` | 3.911e-01 | 4.073e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight62` | 3.998e-01 | 4.381e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight63` | 4.458e-01 | 3.483e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight64` | 4.122e-01 | 3.573e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight65` | 5.581e-01 | 2.119e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight66` | 3.635e-01 | 6.495e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight67` | 1.278e-01 | 4.307e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight68` | 5.446e-01 | 2.654e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight69` | 4.787e-01 | 3.768e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight7` | 4.466e-01 | 3.933e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight70` | 4.610e-01 | 3.052e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight71` | 3.472e-01 | 5.244e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight72` | 5.138e-01 | 3.001e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight73` | 6.137e-01 | 3.166e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight74` | 4.646e-01 | 3.440e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight75` | 4.716e-01 | 3.663e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight76` | 4.876e-01 | 3.650e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight77` | 4.753e-01 | 3.216e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight78` | 6.365e-01 | 1.955e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight79` | 1.202e-01 | 7.480e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight8` | 4.831e-01 | 3.122e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight80` | 3.167e-01 | 9.605e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight81` | 4.629e-01 | 3.750e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight82` | 4.915e-01 | 2.860e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight83` | 5.279e-01 | 2.594e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight84` | 3.520e-01 | 5.655e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight85` | 5.313e-01 | 2.981e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight86` | 5.316e-01 | 3.067e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight87` | 5.516e-01 | 2.936e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight88` | 3.839e-01 | 4.920e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight89` | 4.143e-01 | 3.752e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight9` | 1.893e-01 | 2.907e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight90` | 5.705e-01 | 2.332e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight91` | 3.660e-01 | 4.883e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight92` | 5.321e-01 | 2.672e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight93` | 4.458e-01 | 3.994e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight94` | 4.947e-01 | 2.578e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight95` | 2.851e-01 | 3.050e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight96` | 4.983e-01 | 4.860e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight97` | 5.325e-01 | 2.613e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight98` | 5.108e-01 | 3.012e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.2.mlp.experts.linear_fc2.weight99` | 3.095e-01 | 8.845e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight0` | 3.416e-01 | 4.496e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight1` | 4.044e-01 | 3.450e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight10` | 3.434e-01 | 4.444e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight100` | 9.933e-02 | 5.926e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight101` | 1.600e-01 | 1.824e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight102` | 3.607e-01 | 3.821e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight103` | 4.554e-02 | 2.366e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight104` | 2.384e-01 | 1.211e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight105` | 3.187e-01 | 4.732e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight106` | 3.363e-01 | 4.236e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight107` | 1.435e-01 | 3.807e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight108` | 3.309e-01 | 4.744e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight109` | 1.480e-01 | 3.067e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight11` | 2.723e-01 | 6.624e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight110` | 3.758e-01 | 3.874e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight111` | 3.735e-01 | 3.834e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight112` | 7.483e-02 | 1.313e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight113` | 8.647e-02 | 9.639e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight114` | 3.670e-01 | 4.398e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight115` | 1.201e-01 | 4.731e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight116` | 4.054e-01 | 4.050e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight117` | 6.065e-02 | 1.639e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight118` | 3.236e-01 | 5.131e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight119` | 2.874e-01 | 6.117e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight12` | 3.432e-01 | 5.157e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight120` | 2.326e-01 | 8.966e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight121` | 3.669e-01 | 4.712e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight122` | 3.269e-01 | 5.436e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight123` | 3.572e-01 | 4.450e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight124` | 3.133e-01 | 7.245e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight125` | 3.175e-01 | 4.766e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight126` | 3.441e-01 | 3.893e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight127` | 7.020e-02 | 1.330e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight128` | 4.601e-02 | 2.593e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight129` | 3.814e-01 | 3.897e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight13` | 5.138e-02 | 9.056e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight130` | 3.261e-01 | 4.572e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight131` | 3.745e-01 | 3.667e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight132` | 8.921e-02 | 9.370e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight133` | 1.220e-01 | 4.934e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight134` | 3.297e-01 | 5.052e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight135` | 3.541e-01 | 3.749e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight136` | 3.608e-01 | 4.591e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight137` | 3.253e-01 | 4.966e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight138` | 7.655e-02 | 3.948e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight139` | 1.089e-01 | 5.190e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight14` | 4.338e-01 | 2.310e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight140` | 8.860e-02 | 8.807e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight141` | 3.347e-01 | 4.795e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight142` | 4.047e-01 | 4.045e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight143` | 2.700e-01 | 7.167e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight144` | 3.454e-01 | 5.075e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight145` | 8.785e-02 | 7.702e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight146` | 3.564e-01 | 4.005e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight147` | 3.311e-01 | 4.293e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight148` | 2.185e-01 | 1.204e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight149` | 3.323e-01 | 4.669e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight15` | 3.368e-01 | 4.992e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight150` | 2.925e-01 | 5.670e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight151` | 1.634e-01 | 2.351e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight152` | 1.008e-01 | 6.126e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight153` | 4.269e-02 | 6.583e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight154` | 3.416e-01 | 3.928e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight155` | 3.621e-01 | 4.524e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight156` | 3.437e-01 | 4.641e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight157` | 6.162e-02 | 1.774e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight158` | 8.805e-02 | 8.397e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight159` | 2.163e-02 | 1.211e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight16` | 1.968e-01 | 1.587e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight160` | 4.165e-02 | 3.323e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight161` | 3.369e-01 | 5.054e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight162` | 3.618e-01 | 4.835e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight163` | 3.404e-01 | 4.551e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight164` | 1.119e-01 | 5.139e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight165` | 6.897e-02 | 1.449e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight166` | 3.478e-01 | 4.237e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight167` | 1.048e-01 | 6.705e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight168` | 1.105e-01 | 5.010e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight169` | 3.172e-01 | 5.748e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight17` | 3.720e-01 | 4.272e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight170` | 4.415e-01 | 3.279e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight171` | 4.510e-02 | 9.433e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight172` | 3.839e-01 | 3.855e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight173` | 2.435e-01 | 7.997e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight174` | 3.278e-01 | 4.586e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight175` | 3.449e-01 | 4.431e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight176` | 4.080e-01 | 4.495e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight177` | 3.462e-01 | 4.620e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight178` | 3.464e-01 | 3.909e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight179` | 3.309e-01 | 4.498e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight18` | 3.254e-01 | 5.050e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight180` | 4.663e-02 | 1.010e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight181` | 4.622e-01 | 2.765e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight182` | 3.530e-01 | 4.119e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight183` | 9.629e-02 | 8.767e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight184` | 3.574e-01 | 3.923e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight185` | 1.631e-01 | 2.673e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight186` | 7.490e-02 | 9.661e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight187` | 2.965e-01 | 5.534e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight188` | 3.998e-01 | 3.913e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight189` | 4.157e-02 | 1.113e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight19` | 3.051e-02 | 1.077e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight190` | 1.583e-01 | 3.748e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight191` | 6.689e-02 | 1.513e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight2` | 3.382e-01 | 5.013e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight20` | 3.505e-01 | 3.802e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight21` | 3.601e-01 | 4.217e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight22` | 1.356e-01 | 3.717e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight23` | 2.797e-02 | 2.567e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight24` | 1.756e-01 | 2.228e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight25` | 1.455e-01 | 3.320e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight26` | 4.505e-02 | 1.036e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight27` | 7.668e-02 | 4.981e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight28` | 3.445e-01 | 3.728e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight29` | 3.092e-01 | 5.475e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight3` | 4.042e-01 | 3.772e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight30` | 1.619e-01 | 2.191e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight31` | 3.521e-01 | 4.674e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight32` | 4.238e-01 | 3.224e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight33` | 3.511e-01 | 4.389e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight34` | 3.460e-01 | 4.362e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight35` | 3.505e-01 | 4.117e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight36` | 2.097e-02 | 1.443e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight37` | 9.839e-02 | 6.201e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight38` | 7.330e-02 | 1.402e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight39` | 3.171e-01 | 5.310e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight4` | 3.540e-01 | 4.232e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight40` | 9.494e-02 | 7.954e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight41` | 9.287e-02 | 4.481e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight42` | 1.161e-01 | 5.245e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight43` | 3.373e-01 | 4.483e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight44` | 2.469e-01 | 1.288e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight45` | 3.654e-01 | 3.846e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight46` | 3.426e-01 | 4.223e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight47` | 3.486e-01 | 4.245e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight48` | 3.710e-02 | 3.623e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight49` | 3.673e-01 | 3.892e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight5` | 3.596e-01 | 3.794e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight50` | 3.341e-01 | 4.736e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight51` | 4.646e-02 | 2.387e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight52` | 5.836e-02 | 7.539e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight53` | 5.000e-01 | 0 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight54` | 3.382e-01 | 4.700e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight55` | 1.074e-01 | 3.516e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight56` | 5.000e-01 | 0 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight57` | 3.276e-01 | 5.464e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight58` | 3.385e-02 | 1.048e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight59` | 9.384e-02 | 7.152e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight6` | 3.464e-01 | 4.811e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight60` | 4.089e-01 | 3.949e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight61` | 3.935e-01 | 4.379e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight62` | 8.577e-02 | 9.122e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight63` | 3.570e-01 | 3.849e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight64` | 1.715e-01 | 2.355e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight65` | 7.636e-02 | 1.242e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight66` | 3.609e-01 | 4.805e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight67` | 3.566e-01 | 5.235e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight68` | 3.748e-01 | 3.614e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight69` | 3.319e-01 | 5.012e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight7` | 2.385e-02 | 1.750e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight70` | 3.564e-01 | 4.457e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight71` | 3.653e-02 | 7.105e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight72` | 1.631e-01 | 2.761e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight73` | 3.281e-01 | 4.561e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight74` | 3.493e-01 | 4.093e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight75` | 3.162e-01 | 5.568e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight76` | 3.131e-01 | 5.215e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight77` | 3.864e-01 | 3.871e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight78` | 3.037e-02 | 1.679e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight79` | 1.114e-01 | 4.637e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight8` | 3.300e-01 | 3.596e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight80` | 3.256e-01 | 4.573e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight81` | 3.489e-01 | 4.805e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight82` | 3.406e-01 | 4.584e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight83` | 3.308e-01 | 5.205e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight84` | 3.850e-01 | 4.433e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight85` | 4.475e-02 | 7.596e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight86` | 3.767e-01 | 4.301e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight87` | 4.388e-01 | 3.227e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight88` | 3.259e-01 | 4.703e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight89` | 3.454e-01 | 4.221e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight9` | 3.828e-01 | 4.339e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight90` | 1.246e-01 | 4.335e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight91` | 3.376e-01 | 4.533e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight92` | 4.611e-01 | 3.917e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight93` | 3.642e-02 | 1.470e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight94` | 4.931e-01 | 2.775e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight95` | 4.426e-01 | 2.473e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight96` | 3.098e-02 | 8.878e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight97` | 1.064e-01 | 5.908e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight98` | 3.601e-01 | 4.470e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc1.weight99` | 3.153e-01 | 5.325e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight0` | 3.469e-01 | 4.651e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight1` | 4.184e-01 | 3.188e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight10` | 3.461e-01 | 4.360e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight100` | 1.391e-01 | 3.451e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight101` | 2.224e-01 | 1.109e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight102` | 3.795e-01 | 3.430e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight103` | 5.093e-02 | 2.412e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight104` | 2.700e-01 | 1.119e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight105` | 3.735e-01 | 3.843e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight106` | 4.065e-01 | 3.238e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight107` | 1.751e-01 | 2.555e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight108` | 3.889e-01 | 3.896e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight109` | 1.837e-01 | 2.521e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight11` | 3.253e-01 | 4.647e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight110` | 4.188e-01 | 3.336e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight111` | 4.411e-01 | 3.151e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight112` | 6.591e-02 | 1.233e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight113` | 9.444e-02 | 8.582e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight114` | 3.663e-01 | 4.291e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight115` | 1.605e-01 | 2.018e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight116` | 4.173e-01 | 3.900e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight117` | 7.628e-02 | 1.358e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight118` | 3.582e-01 | 3.953e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight119` | 3.215e-01 | 5.448e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight12` | 3.470e-01 | 4.167e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight120` | 2.633e-01 | 8.393e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight121` | 4.080e-01 | 4.047e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight122` | 3.542e-01 | 4.855e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight123` | 3.910e-01 | 4.186e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight124` | 3.266e-01 | 8.447e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight125` | 3.478e-01 | 3.764e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight126` | 4.207e-01 | 3.388e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight127` | 6.368e-02 | 1.155e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight128` | 4.633e-02 | 2.837e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight129` | 4.435e-01 | 2.320e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight13` | 8.937e-02 | 5.862e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight130` | 3.480e-01 | 4.379e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight131` | 4.130e-01 | 3.302e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight132` | 1.088e-01 | 7.279e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight133` | 1.656e-01 | 3.581e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight134` | 3.303e-01 | 5.101e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight135` | 4.313e-01 | 3.012e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight136` | 3.938e-01 | 3.618e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight137` | 3.480e-01 | 4.464e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight138` | 1.273e-01 | 2.822e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight139` | 1.397e-01 | 3.884e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight14` | 4.541e-01 | 2.316e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight140` | 1.044e-01 | 8.400e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight141` | 3.376e-01 | 3.954e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight142` | 4.371e-01 | 3.100e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight143` | 3.292e-01 | 3.842e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight144` | 3.445e-01 | 5.175e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight145` | 1.077e-01 | 3.557e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight146` | 4.018e-01 | 3.640e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight147` | 4.084e-01 | 3.575e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight148` | 2.697e-01 | 8.473e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight149` | 3.859e-01 | 4.175e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight15` | 3.339e-01 | 4.846e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight150` | 3.232e-01 | 4.665e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight151` | 2.189e-01 | 1.149e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight152` | 1.065e-01 | 6.353e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight153` | 6.174e-02 | 3.248e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight154` | 3.784e-01 | 3.523e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight155` | 3.711e-01 | 4.298e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight156` | 3.657e-01 | 4.275e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight157` | 6.200e-02 | 1.966e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight158` | 1.198e-01 | 4.084e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight159` | 2.394e-02 | 1.061e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight16` | 2.097e-01 | 1.451e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight160` | 4.874e-02 | 3.020e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight161` | 3.413e-01 | 4.866e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight162` | 3.551e-01 | 5.031e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight163` | 4.072e-01 | 2.927e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight164` | 1.502e-01 | 2.863e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight165` | 7.337e-02 | 1.501e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight166` | 4.097e-01 | 3.785e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight167` | 1.306e-01 | 4.750e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight168` | 1.229e-01 | 3.855e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight169` | 3.043e-01 | 5.793e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight17` | 3.918e-01 | 3.731e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight170` | 4.415e-01 | 3.333e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight171` | 6.961e-02 | 5.089e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight172` | 3.989e-01 | 3.408e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight173` | 2.834e-01 | 5.518e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight174` | 3.784e-01 | 4.040e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight175` | 4.274e-01 | 2.929e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight176` | 4.319e-01 | 3.854e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight177` | 3.783e-01 | 4.257e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight178` | 4.581e-01 | 3.126e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight179` | 3.816e-01 | 3.608e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight18` | 3.670e-01 | 2.770e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight180` | 7.569e-02 | 5.980e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight181` | 5.050e-01 | 2.668e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight182` | 4.264e-01 | 3.305e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight183` | 1.087e-01 | 7.592e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight184` | 4.022e-01 | 3.535e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight185` | 2.362e-01 | 1.594e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight186` | 9.059e-02 | 7.113e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight187` | 3.328e-01 | 4.497e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight188` | 4.129e-01 | 3.226e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight189` | 5.663e-02 | 8.831e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight19` | 4.100e-02 | 8.281e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight190` | 1.737e-01 | 3.566e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight191` | 7.939e-02 | 1.319e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight2` | 3.608e-01 | 2.374e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight20` | 3.720e-01 | 3.312e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight21` | 4.167e-01 | 3.612e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight22` | 1.822e-01 | 2.019e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight23` | 2.428e-02 | 1.983e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight24` | 1.904e-01 | 2.169e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight25` | 2.005e-01 | 1.833e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight26` | 4.208e-02 | 9.508e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight27` | 9.321e-02 | 4.295e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight28` | 3.958e-01 | 3.418e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight29` | 3.606e-01 | 4.764e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight3` | 5.151e-01 | 3.004e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight30` | 2.321e-01 | 1.730e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight31` | 3.709e-01 | 4.151e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight32` | 4.135e-01 | 3.215e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight33` | 3.882e-01 | 3.410e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight34` | 4.053e-01 | 3.425e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight35` | 4.015e-01 | 3.722e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight36` | 2.503e-02 | 1.315e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight37` | 1.428e-01 | 5.586e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight38` | 6.343e-02 | 1.257e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight39` | 3.310e-01 | 4.981e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight4` | 3.813e-01 | 3.805e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight40` | 1.212e-01 | 5.369e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight41` | 1.021e-01 | 3.433e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight42` | 1.202e-01 | 4.337e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight43` | 4.086e-01 | 3.575e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight44` | 2.701e-01 | 1.524e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight45` | 4.630e-01 | 2.898e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight46` | 3.730e-01 | 3.853e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight47` | 4.253e-01 | 3.752e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight48` | 4.115e-02 | 3.465e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight49` | 3.693e-01 | 3.501e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight5` | 4.203e-01 | 3.302e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight50` | 3.654e-01 | 4.357e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight51` | 4.889e-02 | 2.223e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight52` | 8.094e-02 | 5.989e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight53` | 5.098e-01 | 0 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight54` | 3.644e-01 | 4.225e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight55` | 1.433e-01 | 2.384e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight56` | 4.759e-01 | 0 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight57` | 3.463e-01 | 5.552e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight58` | 3.564e-02 | 7.528e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight59` | 9.924e-02 | 6.338e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight6` | 3.511e-01 | 4.623e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight60` | 4.223e-01 | 3.707e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight61` | 4.146e-01 | 1.841e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight62` | 9.112e-02 | 8.328e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight63` | 4.082e-01 | 3.141e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight64` | 2.178e-01 | 2.046e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight65` | 8.861e-02 | 1.180e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight66` | 3.565e-01 | 4.736e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight67` | 3.489e-01 | 3.260e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight68` | 4.360e-01 | 3.231e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight69` | 3.332e-01 | 5.190e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight7` | 2.807e-02 | 1.563e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight70` | 3.826e-01 | 3.756e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight71` | 7.385e-02 | 3.972e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight72` | 1.805e-01 | 2.139e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight73` | 3.925e-01 | 3.688e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight74` | 4.181e-01 | 3.447e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight75` | 3.130e-01 | 5.437e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight76` | 3.467e-01 | 4.289e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight77` | 4.023e-01 | 3.584e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight78` | 3.662e-02 | 1.658e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight79` | 1.259e-01 | 2.445e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight8` | 3.572e-01 | 2.735e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight80` | 3.618e-01 | 3.993e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight81` | 3.466e-01 | 2.071e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight82` | 3.920e-01 | 4.154e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight83` | 3.394e-01 | 5.035e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight84` | 3.841e-01 | 4.353e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight85` | 6.251e-02 | 4.636e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight86` | 3.968e-01 | 3.878e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight87` | 4.821e-01 | 2.930e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight88` | 3.709e-01 | 4.099e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight89` | 3.923e-01 | 3.802e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight9` | 3.946e-01 | 4.086e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight90` | 1.433e-01 | 2.160e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight91` | 4.017e-01 | 3.617e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight92` | 4.627e-01 | 4.061e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight93` | 4.265e-02 | 1.365e-01 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight94` | 4.857e-01 | 2.747e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight95` | 4.980e-01 | 2.683e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight96` | 3.243e-02 | 7.983e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight97` | 1.610e-01 | 3.610e-02 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight98` | 3.615e-01 | 4.357e-03 | 2048 | 0 | 2048 |
| `p2048.grad.decoder.layers.3.mlp.experts.linear_fc2.weight99` | 3.139e-01 | 4.379e-03 | 2048 | 0 | 2048 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight0` | 3.430e-01 | 4.227e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight1` | 4.043e-01 | 3.193e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight10` | 2.691e-01 | 5.527e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight100` | 3.467e-01 | 3.392e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight101` | 3.603e-01 | 3.765e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight102` | 2.993e-01 | 5.320e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight103` | 4.628e-01 | 2.598e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight104` | 3.123e-01 | 4.534e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight105` | 3.493e-01 | 4.279e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight106` | 2.617e-01 | 4.955e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight107` | 3.260e-01 | 3.786e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight108` | 3.354e-01 | 3.677e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight109` | 3.031e-01 | 3.583e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight11` | 3.556e-01 | 3.557e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight110` | 3.632e-01 | 3.162e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight111` | 3.556e-01 | 4.022e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight112` | 2.961e-01 | 4.570e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight113` | 4.480e-01 | 2.942e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight114` | 2.819e-01 | 4.189e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight115` | 3.843e-01 | 3.678e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight116` | 3.237e-01 | 4.385e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight117` | 3.106e-01 | 5.161e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight118` | 3.288e-01 | 4.046e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight119` | 3.212e-01 | 4.119e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight12` | 3.618e-01 | 3.773e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight120` | 4.238e-01 | 2.592e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight121` | 3.698e-01 | 4.057e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight122` | 2.607e-01 | 5.171e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight123` | 3.385e-01 | 3.779e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight124` | 3.637e-01 | 3.979e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight125` | 3.185e-01 | 3.944e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight126` | 3.267e-01 | 3.574e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight127` | 4.462e-01 | 2.932e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight128` | 3.498e-01 | 2.885e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight129` | 3.199e-01 | 4.256e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight13` | 3.676e-01 | 3.639e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight130` | 2.865e-01 | 3.923e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight131` | 2.962e-01 | 4.848e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight132` | 2.801e-01 | 4.800e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight133` | 3.511e-01 | 4.146e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight134` | 2.796e-01 | 5.304e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight135` | 3.263e-01 | 5.451e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight136` | 2.574e-01 | 5.612e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight137` | 4.546e-01 | 2.985e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight138` | 3.556e-01 | 3.345e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight139` | 3.621e-01 | 3.067e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight14` | 1.992e-01 | 8.489e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight140` | 3.762e-01 | 3.553e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight141` | 3.136e-01 | 4.643e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight142` | 3.071e-01 | 4.010e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight143` | 3.412e-01 | 3.701e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight144` | 3.235e-01 | 4.240e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight145` | 4.382e-01 | 2.949e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight146` | 2.236e-01 | 8.618e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight147` | 2.340e-01 | 6.303e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight148` | 2.730e-01 | 4.929e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight149` | 3.052e-01 | 3.219e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight15` | 3.269e-01 | 3.654e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight150` | 3.515e-01 | 4.795e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight151` | 3.486e-01 | 3.987e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight152` | 3.368e-01 | 4.323e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight153` | 3.273e-01 | 4.069e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight154` | 2.635e-01 | 5.311e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight155` | 2.845e-01 | 6.508e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight156` | 3.356e-01 | 4.354e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight157` | 2.526e-01 | 6.054e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight158` | 2.905e-01 | 4.934e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight159` | 2.405e-01 | 6.684e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight16` | 2.250e-01 | 7.639e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight160` | 3.815e-01 | 4.113e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight161` | 3.128e-01 | 4.548e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight162` | 3.538e-01 | 3.024e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight163` | 3.040e-01 | 4.635e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight164` | 6.450e-01 | 1.962e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight165` | 2.571e-01 | 5.068e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight166` | 2.942e-01 | 3.861e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight167` | 3.279e-01 | 5.175e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight168` | 3.433e-01 | 3.457e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight169` | 2.640e-01 | 5.258e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight17` | 2.796e-01 | 5.115e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight170` | 3.200e-01 | 4.670e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight171` | 2.981e-01 | 4.730e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight172` | 3.195e-01 | 4.524e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight173` | 3.452e-01 | 3.519e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight174` | 3.241e-01 | 4.300e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight175` | 3.468e-01 | 3.902e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight176` | 2.959e-01 | 3.566e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight177` | 2.978e-01 | 4.430e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight178` | 3.326e-01 | 3.547e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight179` | 2.712e-01 | 6.283e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight18` | 3.019e-01 | 4.936e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight180` | 2.907e-01 | 4.933e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight181` | 3.540e-01 | 4.010e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight182` | 3.270e-01 | 4.680e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight183` | 3.333e-01 | 5.045e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight184` | 3.667e-01 | 3.589e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight185` | 3.466e-01 | 4.080e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight186` | 3.335e-01 | 8.881e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight187` | 3.260e-01 | 3.852e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight188` | 3.749e-01 | 3.602e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight189` | 2.961e-01 | 4.850e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight19` | 3.268e-01 | 3.442e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight190` | 3.944e-01 | 2.634e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight191` | 2.653e-01 | 7.863e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight2` | 3.132e-01 | 3.432e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight20` | 3.738e-01 | 3.648e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight21` | 3.221e-01 | 4.449e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight22` | 3.011e-01 | 4.667e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight23` | 2.584e-01 | 7.269e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight24` | 2.317e-01 | 6.544e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight25` | 4.306e-01 | 3.003e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight26` | 3.137e-01 | 4.077e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight27` | 2.603e-01 | 6.875e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight28` | 3.654e-01 | 3.282e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight29` | 2.857e-01 | 3.672e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight3` | 3.566e-01 | 4.102e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight30` | 2.728e-01 | 5.448e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight31` | 3.402e-01 | 3.622e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight32` | 3.230e-01 | 4.476e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight33` | 3.402e-01 | 3.631e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight34` | 3.512e-01 | 4.896e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight35` | 2.832e-01 | 6.243e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight36` | 3.368e-01 | 4.528e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight37` | 2.953e-01 | 5.239e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight38` | 2.871e-01 | 5.294e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight39` | 3.206e-01 | 4.290e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight4` | 3.332e-01 | 4.010e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight40` | 2.935e-01 | 5.140e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight41` | 4.119e-01 | 2.802e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight42` | 2.531e-01 | 4.994e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight43` | 3.182e-01 | 4.740e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight44` | 3.140e-01 | 3.451e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight45` | 3.316e-01 | 3.846e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight46` | 2.405e-01 | 5.939e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight47` | 3.343e-01 | 4.476e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight48` | 3.698e-01 | 3.757e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight49` | 3.066e-01 | 5.233e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight5` | 3.363e-01 | 4.032e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight50` | 3.901e-01 | 3.925e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight51` | 3.259e-01 | 3.710e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight52` | 3.529e-01 | 3.798e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight53` | 4.495e-01 | 2.773e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight54` | 2.433e-01 | 8.091e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight55` | 2.852e-01 | 5.550e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight56` | 3.839e-01 | 4.447e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight57` | 3.279e-01 | 4.125e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight58` | 3.505e-01 | 3.307e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight59` | 3.850e-01 | 3.470e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight6` | 3.438e-01 | 4.102e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight60` | 3.091e-01 | 4.983e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight61` | 3.208e-01 | 5.006e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight62` | 2.774e-01 | 5.017e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight63` | 3.508e-01 | 4.963e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight64` | 4.463e-01 | 3.081e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight65` | 2.670e-01 | 5.767e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight66` | 3.136e-01 | 4.353e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight67` | 4.450e-01 | 2.839e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight68` | 2.784e-01 | 6.040e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight69` | 2.707e-01 | 6.028e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight7` | 3.271e-01 | 3.704e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight70` | 2.983e-01 | 4.323e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight71` | 3.363e-01 | 3.984e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight72` | 2.748e-01 | 5.449e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight73` | 3.318e-01 | 4.376e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight74` | 3.502e-01 | 3.821e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight75` | 3.317e-01 | 3.669e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight76` | 3.516e-01 | 3.853e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight77` | 3.777e-01 | 3.473e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight78` | 3.460e-01 | 4.354e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight79` | 3.601e-01 | 4.149e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight8` | 3.426e-01 | 4.695e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight80` | 3.706e-01 | 3.790e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight81` | 2.998e-01 | 5.305e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight82` | 4.527e-01 | 3.295e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight83` | 3.310e-01 | 4.112e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight84` | 3.489e-01 | 3.540e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight85` | 3.519e-01 | 4.372e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight86` | 2.906e-01 | 4.227e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight87` | 3.949e-01 | 3.683e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight88` | 3.421e-01 | 4.166e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight89` | 2.793e-01 | 5.068e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight9` | 2.558e-01 | 8.207e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight90` | 3.169e-01 | 4.972e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight91` | 4.911e-01 | 2.760e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight92` | 3.852e-01 | 3.801e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight93` | 2.389e-01 | 6.168e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight94` | 2.831e-01 | 4.713e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight95` | 3.272e-01 | 4.136e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight96` | 2.912e-01 | 4.110e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight97` | 4.854e-01 | 2.473e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight98` | 3.318e-01 | 4.374e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc1.weight99` | 2.887e-01 | 4.180e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 4.605e-01 | 3.511e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight1` | 5.270e-01 | 2.668e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight10` | 3.810e-01 | 4.814e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight100` | 4.849e-01 | 3.129e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight101` | 4.841e-01 | 3.098e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight102` | 4.130e-01 | 4.142e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight103` | 5.978e-01 | 2.254e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight104` | 4.137e-01 | 4.794e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight105` | 4.716e-01 | 3.379e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight106` | 3.712e-01 | 4.268e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight107` | 4.328e-01 | 3.635e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight108` | 4.486e-01 | 3.797e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight109` | 4.587e-01 | 3.054e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight11` | 4.943e-01 | 2.662e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight110` | 5.131e-01 | 2.644e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight111` | 5.206e-01 | 4.775e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight112` | 4.828e-01 | 3.430e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight113` | 5.331e-01 | 2.597e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight114` | 4.302e-01 | 2.895e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight115` | 4.614e-01 | 3.324e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight116` | 4.481e-01 | 3.694e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight117` | 4.117e-01 | 4.072e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight118` | 4.406e-01 | 3.885e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight119` | 4.166e-01 | 3.800e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight12` | 4.599e-01 | 3.525e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight120` | 5.757e-01 | 2.403e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight121` | 4.776e-01 | 3.052e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight122` | 3.590e-01 | 4.783e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight123` | 4.884e-01 | 2.861e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight124` | 4.996e-01 | 3.138e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight125` | 4.612e-01 | 3.405e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight126` | 4.425e-01 | 3.095e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight127` | 5.880e-01 | 2.425e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight128` | 5.306e-01 | 2.476e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight129` | 4.288e-01 | 3.775e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight13` | 4.845e-01 | 3.659e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight130` | 4.962e-01 | 2.826e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight131` | 4.014e-01 | 8.980e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight132` | 4.245e-01 | 3.877e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight133` | 4.372e-01 | 3.664e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight134` | 3.780e-01 | 4.633e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight135` | 4.836e-01 | 3.780e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight136` | 3.527e-01 | 5.011e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight137` | 5.363e-01 | 2.785e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight138` | 4.741e-01 | 2.737e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight139` | 5.523e-01 | 2.526e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight14` | 2.690e-01 | 5.417e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight140` | 4.971e-01 | 2.978e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight141` | 4.000e-01 | 4.384e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight142` | 4.241e-01 | 3.691e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight143` | 4.622e-01 | 3.257e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight144` | 5.047e-01 | 3.289e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight145` | 5.779e-01 | 2.504e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight146` | 3.074e-01 | 8.011e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight147` | 3.347e-01 | 5.833e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight148` | 3.835e-01 | 4.352e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight149` | 4.603e-01 | 3.135e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight15` | 4.768e-01 | 2.941e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight150` | 4.544e-01 | 3.708e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight151` | 4.349e-01 | 3.537e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight152` | 4.345e-01 | 3.852e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight153` | 4.307e-01 | 4.240e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight154` | 4.263e-01 | 3.985e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight155` | 3.743e-01 | 4.225e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight156` | 4.505e-01 | 5.209e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight157` | 3.426e-01 | 5.734e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight158` | 3.887e-01 | 4.335e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight159` | 3.725e-01 | 6.373e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight16` | 3.297e-01 | 4.081e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight160` | 4.826e-01 | 3.010e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight161` | 4.128e-01 | 3.997e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight162` | 5.247e-01 | 2.495e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight163` | 4.117e-01 | 3.787e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight164` | 6.918e-01 | 1.746e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight165` | 3.973e-01 | 3.659e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight166` | 4.367e-01 | 3.156e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight167` | 4.349e-01 | 4.159e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight168` | 5.402e-01 | 2.812e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight169` | 3.634e-01 | 5.351e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight17` | 3.605e-01 | 6.594e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight170` | 4.338e-01 | 4.947e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight171` | 4.176e-01 | 3.983e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight172` | 4.564e-01 | 3.359e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight173` | 4.963e-01 | 2.945e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight174` | 4.143e-01 | 3.844e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight175` | 4.671e-01 | 3.172e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight176` | 4.663e-01 | 2.911e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight177` | 4.149e-01 | 4.131e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight178` | 5.020e-01 | 2.772e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight179` | 3.774e-01 | 5.009e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight18` | 3.991e-01 | 4.159e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight180` | 4.034e-01 | 4.304e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight181` | 4.510e-01 | 3.542e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight182` | 4.232e-01 | 3.891e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight183` | 4.428e-01 | 4.015e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight184` | 5.979e-01 | 2.508e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight185` | 4.477e-01 | 3.139e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight186` | 4.793e-01 | 4.822e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight187` | 4.516e-01 | 3.910e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight188` | 4.993e-01 | 2.879e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight189` | 4.078e-01 | 4.090e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight19` | 5.144e-01 | 2.985e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight190` | 5.538e-01 | 2.301e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight191` | 3.462e-01 | 6.670e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight2` | 4.429e-01 | 2.825e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight20` | 5.114e-01 | 2.986e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight21` | 4.321e-01 | 4.128e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight22` | 4.316e-01 | 4.070e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight23` | 3.536e-01 | 5.504e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight24` | 3.336e-01 | 4.716e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight25` | 5.331e-01 | 2.583e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight26` | 4.548e-01 | 3.330e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight27` | 3.350e-01 | 6.586e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight28` | 5.158e-01 | 3.187e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight29` | 4.355e-01 | 3.271e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight3` | 4.855e-01 | 3.197e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight30` | 3.640e-01 | 4.894e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight31` | 4.452e-01 | 3.228e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight32` | 4.377e-01 | 3.748e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight33` | 4.982e-01 | 2.820e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight34` | 4.461e-01 | 1.467e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight35` | 3.608e-01 | 5.568e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight36` | 4.965e-01 | 3.269e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight37` | 4.124e-01 | 4.027e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight38` | 3.944e-01 | 4.343e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight39` | 4.084e-01 | 3.578e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight4` | 5.007e-01 | 3.442e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight40` | 3.978e-01 | 4.441e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight41` | 5.436e-01 | 2.338e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight42` | 4.298e-01 | 4.313e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight43` | 4.047e-01 | 4.020e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight44` | 4.557e-01 | 3.055e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight45` | 4.759e-01 | 3.766e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight46` | 3.218e-01 | 5.523e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight47` | 4.297e-01 | 3.882e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight48` | 4.426e-01 | 3.670e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight49` | 3.980e-01 | 4.593e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight5` | 4.353e-01 | 3.461e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight50` | 4.918e-01 | 3.618e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight51` | 5.037e-01 | 2.859e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight52` | 4.526e-01 | 3.240e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight53` | 5.687e-01 | 2.433e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight54` | 3.058e-01 | 7.340e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight55` | 3.799e-01 | 4.646e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight56` | 5.142e-01 | 3.341e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight57` | 4.585e-01 | 3.386e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight58` | 4.744e-01 | 3.154e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight59` | 4.653e-01 | 3.053e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight6` | 4.668e-01 | 3.345e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight60` | 4.279e-01 | 4.123e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight61` | 4.151e-01 | 4.032e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight62` | 3.774e-01 | 3.944e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight63` | 4.836e-01 | 3.535e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight64` | 5.372e-01 | 2.359e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight65` | 3.711e-01 | 4.712e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight66` | 4.199e-01 | 3.809e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight67` | 5.984e-01 | 2.699e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight68` | 3.475e-01 | 5.557e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight69` | 3.996e-01 | 4.896e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight7` | 5.114e-01 | 3.036e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight70` | 4.748e-01 | 3.861e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight71` | 4.497e-01 | 3.489e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight72` | 3.867e-01 | 3.677e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight73` | 4.450e-01 | 3.965e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight74` | 4.591e-01 | 3.292e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight75` | 4.547e-01 | 3.320e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight76` | 4.593e-01 | 3.375e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight77` | 5.346e-01 | 2.599e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight78` | 4.733e-01 | 3.218e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight79` | 4.966e-01 | 3.306e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight8` | 4.566e-01 | 3.726e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight80` | 4.497e-01 | 3.282e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight81` | 3.837e-01 | 4.672e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight82` | 5.323e-01 | 2.780e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight83` | 4.926e-01 | 3.126e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight84` | 5.019e-01 | 4.370e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight85` | 4.588e-01 | 3.992e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight86` | 4.751e-01 | 3.606e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight87` | 5.334e-01 | 3.986e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight88` | 4.556e-01 | 3.441e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight89` | 3.852e-01 | 3.722e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight9` | 3.554e-01 | 7.139e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight90` | 4.210e-01 | 4.139e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight91` | 6.124e-01 | 2.196e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight92` | 5.261e-01 | 3.285e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight93` | 3.304e-01 | 5.635e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight94` | 3.973e-01 | 4.180e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight95` | 4.571e-01 | 3.771e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight96` | 4.565e-01 | 3.808e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight97` | 6.113e-01 | 2.067e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight98` | 4.496e-01 | 3.658e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.1.mlp.experts.linear_fc2.weight99` | 4.209e-01 | 5.628e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight0` | 2.938e-01 | 5.686e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight1` | 2.873e-01 | 3.735e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight10` | 3.428e-01 | 3.840e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight100` | 1.678e-01 | 2.001e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight101` | 3.098e-01 | 5.092e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight102` | 3.868e-01 | 4.190e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight103` | 3.715e-01 | 4.509e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight104` | 3.682e-01 | 3.834e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight105` | 3.385e-01 | 4.507e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight106` | 4.129e-01 | 3.708e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight107` | 2.957e-01 | 5.773e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight108` | 2.563e-01 | 1.168e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight109` | 1.782e-01 | 1.092e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight11` | 5.411e-02 | 7.569e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight110` | 4.536e-01 | 2.789e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight111` | 3.574e-01 | 3.473e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight112` | 2.978e-01 | 4.876e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight113` | 2.961e-01 | 4.548e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight114` | 1.136e-01 | 7.075e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight115` | 3.150e-01 | 5.168e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight116` | 3.021e-01 | 5.415e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight117` | 5.012e-01 | 2.582e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight118` | 3.981e-01 | 3.464e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight119` | 3.756e-01 | 4.280e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight12` | 3.667e-01 | 4.453e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight120` | 2.925e-01 | 6.148e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight121` | 3.072e-01 | 5.324e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight122` | 1.335e-01 | 2.509e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight123` | 3.554e-01 | 4.552e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight124` | 2.920e-01 | 4.059e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight125` | 3.291e-01 | 3.185e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight126` | 3.901e-01 | 4.436e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight127` | 3.681e-01 | 4.116e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight128` | 3.485e-01 | 3.972e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight129` | 3.007e-01 | 5.213e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight13` | 3.290e-01 | 4.145e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight130` | 2.668e-01 | 7.534e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight131` | 4.127e-01 | 3.539e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight132` | 8.874e-02 | 8.876e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight133` | 4.238e-01 | 3.199e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight134` | 3.347e-01 | 5.174e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight135` | 3.586e-01 | 3.682e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight136` | 3.032e-01 | 4.931e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight137` | 4.000e-01 | 3.762e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight138` | 4.147e-01 | 3.306e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight139` | 1.248e-01 | 4.008e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight14` | 3.452e-01 | 3.733e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight140` | 2.797e-01 | 4.522e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight141` | 3.208e-01 | 4.682e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight142` | 2.828e-01 | 6.530e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight143` | 3.135e-01 | 4.070e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight144` | 2.781e-01 | 5.623e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight145` | 3.734e-01 | 4.017e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight146` | 2.864e-01 | 4.913e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight147` | 2.858e-01 | 6.434e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight148` | 4.324e-01 | 3.371e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight149` | 4.483e-02 | 8.148e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight15` | 3.031e-01 | 5.221e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight150` | 2.646e-01 | 5.190e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight151` | 3.901e-01 | 4.160e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight152` | 1.481e-01 | 2.975e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight153` | 2.357e-01 | 9.667e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight154` | 3.484e-01 | 4.128e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight155` | 3.786e-01 | 3.325e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight156` | 1.500e-01 | 2.722e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight157` | 3.113e-01 | 5.302e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight158` | 3.786e-01 | 4.096e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight159` | 2.930e-01 | 5.038e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight16` | 8.476e-02 | 5.447e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight160` | 2.792e-01 | 5.316e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight161` | 3.084e-01 | 5.802e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight162` | 4.000e-01 | 4.148e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight163` | 4.003e-01 | 3.375e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight164` | 2.753e-01 | 7.166e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight165` | 3.070e-01 | 5.077e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight166` | 3.861e-01 | 3.291e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight167` | 3.793e-01 | 4.857e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight168` | 3.025e-01 | 6.587e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight169` | 3.310e-01 | 4.874e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight17` | 3.222e-01 | 4.359e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight170` | 2.237e-01 | 1.361e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight171` | 3.238e-01 | 3.973e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight172` | 3.042e-01 | 6.455e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight173` | 5.916e-02 | 1.286e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight174` | 3.067e-01 | 4.643e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight175` | 9.764e-02 | 6.269e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight176` | 3.747e-01 | 3.781e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight177` | 3.652e-01 | 4.293e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight178` | 3.043e-01 | 6.684e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight179` | 3.432e-01 | 4.001e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight18` | 3.472e-01 | 4.109e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight180` | 1.500e-01 | 3.726e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight181` | 3.230e-01 | 5.664e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight182` | 3.722e-01 | 4.281e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight183` | 3.794e-01 | 4.127e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight184` | 1.310e-01 | 4.054e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight185` | 2.985e-01 | 6.295e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight186` | 3.607e-01 | 4.303e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight187` | 2.951e-01 | 6.803e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight188` | 3.528e-01 | 4.025e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight189` | 2.536e-01 | 7.812e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight19` | 3.971e-01 | 3.563e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight190` | 3.264e-01 | 5.571e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight191` | 3.010e-01 | 6.150e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight2` | 2.866e-01 | 5.874e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight20` | 3.893e-01 | 3.689e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight21` | 2.754e-01 | 7.293e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight22` | 4.062e-01 | 3.288e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight23` | 3.186e-01 | 3.874e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight24` | 3.023e-01 | 5.552e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight25` | 2.125e-01 | 7.587e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight26` | 3.328e-01 | 4.117e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight27` | 3.171e-01 | 4.559e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight28` | 2.476e-01 | 6.074e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight29` | 3.325e-01 | 4.734e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight3` | 2.936e-01 | 6.584e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight30` | 4.352e-01 | 3.331e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight31` | 1.114e-01 | 6.264e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight32` | 3.814e-01 | 3.763e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight33` | 2.962e-01 | 5.756e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight34` | 3.169e-01 | 4.530e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight35` | 3.703e-01 | 3.513e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight36` | 2.230e-01 | 7.221e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight37` | 3.547e-01 | 4.838e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight38` | 1.356e-01 | 3.603e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight39` | 7.393e-02 | 1.184e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight4` | 3.043e-01 | 5.869e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight40` | 2.711e-01 | 6.232e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight41` | 3.663e-01 | 4.274e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight42` | 2.788e-01 | 5.826e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight43` | 3.636e-01 | 3.489e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight44` | 3.065e-01 | 4.447e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight45` | 3.636e-01 | 3.734e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight46` | 8.766e-02 | 7.388e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight47` | 2.409e-01 | 9.871e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight48` | 3.270e-01 | 5.263e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight49` | 3.906e-01 | 3.774e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight5` | 3.176e-01 | 5.102e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight50` | 3.485e-01 | 3.964e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight51` | 3.561e-01 | 3.651e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight52` | 2.845e-01 | 6.711e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight53` | 3.843e-01 | 4.065e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight54` | 2.257e-01 | 8.219e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight55` | 5.882e-01 | 2.079e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight56` | 3.443e-01 | 4.381e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight57` | 3.339e-01 | 3.900e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight58` | 2.918e-01 | 6.536e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight59` | 3.665e-01 | 4.415e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight6` | 3.573e-01 | 4.779e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight60` | 3.626e-01 | 4.102e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight61` | 3.625e-01 | 4.215e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight62` | 3.212e-01 | 5.065e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight63` | 3.089e-01 | 4.367e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight64` | 2.910e-01 | 5.220e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight65` | 4.070e-01 | 3.326e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight66` | 2.612e-01 | 7.810e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight67` | 1.136e-01 | 4.258e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight68` | 3.740e-01 | 4.116e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight69` | 4.178e-01 | 4.420e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight7` | 4.060e-01 | 4.378e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight70` | 3.307e-01 | 3.619e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight71` | 2.961e-01 | 5.213e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight72` | 3.245e-01 | 5.000e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight73` | 3.405e-01 | 6.174e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight74` | 3.708e-01 | 4.237e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight75` | 3.823e-01 | 4.622e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight76` | 3.627e-01 | 4.491e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight77` | 4.363e-01 | 3.201e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight78` | 5.813e-01 | 2.259e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight79` | 3.581e-01 | 3.762e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight8` | 3.248e-01 | 4.229e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight80` | 2.857e-01 | 4.931e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight81` | 3.283e-01 | 4.290e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight82` | 3.549e-01 | 3.880e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight83` | 4.334e-01 | 2.781e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight84` | 3.541e-01 | 4.643e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight85` | 3.890e-01 | 4.066e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight86` | 2.848e-01 | 4.986e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight87` | 3.781e-01 | 3.408e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight88` | 2.858e-01 | 4.911e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight89` | 3.789e-01 | 3.138e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight9` | 1.595e-01 | 3.155e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight90` | 4.591e-01 | 2.802e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight91` | 3.582e-01 | 3.956e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight92` | 4.373e-01 | 3.167e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight93` | 3.643e-01 | 3.835e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight94` | 4.501e-01 | 2.268e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight95` | 3.019e-01 | 6.174e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight96` | 3.990e-01 | 3.495e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight97` | 3.597e-01 | 3.204e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight98` | 3.557e-01 | 4.704e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc1.weight99` | 2.255e-01 | 8.286e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight0` | 3.725e-01 | 5.039e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight1` | 4.785e-01 | 2.735e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight10` | 4.950e-01 | 2.941e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight100` | 2.066e-01 | 1.572e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight101` | 4.358e-01 | 3.801e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight102` | 4.701e-01 | 3.496e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight103` | 4.690e-01 | 4.022e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight104` | 4.345e-01 | 3.364e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight105` | 4.259e-01 | 3.144e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight106` | 5.120e-01 | 2.901e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight107` | 3.915e-01 | 4.353e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight108` | 3.324e-01 | 9.541e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight109` | 2.204e-01 | 1.254e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight11` | 5.898e-02 | 8.088e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight110` | 5.530e-01 | 2.607e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight111` | 5.129e-01 | 3.068e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight112` | 4.264e-01 | 3.595e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight113` | 4.439e-01 | 3.611e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight114` | 1.353e-01 | 6.746e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight115` | 4.195e-01 | 4.140e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight116` | 3.888e-01 | 4.766e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight117` | 5.932e-01 | 2.047e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight118` | 4.864e-01 | 2.971e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight119` | 4.897e-01 | 2.930e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight12` | 4.317e-01 | 3.719e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight120` | 4.056e-01 | 3.486e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight121` | 3.996e-01 | 3.772e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight122` | 2.074e-01 | 2.243e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight123` | 4.348e-01 | 3.763e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight124` | 4.612e-01 | 3.270e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight125` | 5.156e-01 | 2.705e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight126` | 5.039e-01 | 3.102e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight127` | 4.536e-01 | 3.442e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight128` | 4.566e-01 | 3.267e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight129` | 3.861e-01 | 4.582e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight13` | 4.398e-01 | 3.506e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight130` | 3.378e-01 | 4.868e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight131` | 5.145e-01 | 2.881e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight132` | 9.857e-02 | 8.986e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight133` | 5.243e-01 | 2.704e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight134` | 3.918e-01 | 4.677e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight135` | 4.834e-01 | 2.816e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight136` | 4.070e-01 | 3.596e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight137` | 4.872e-01 | 3.248e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight138` | 5.472e-01 | 2.558e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight139` | 1.487e-01 | 3.797e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight14` | 4.752e-01 | 3.228e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight140` | 4.012e-01 | 3.836e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight141` | 4.033e-01 | 4.073e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight142` | 3.532e-01 | 6.059e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight143` | 4.516e-01 | 3.239e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight144` | 4.100e-01 | 4.916e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight145` | 4.550e-01 | 3.395e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight146` | 3.872e-01 | 5.457e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight147` | 3.725e-01 | 5.454e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight148` | 5.746e-01 | 2.694e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight149` | 4.621e-02 | 7.814e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight15` | 4.175e-01 | 3.776e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight150` | 3.919e-01 | 3.801e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight151` | 4.925e-01 | 3.303e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight152` | 1.770e-01 | 2.008e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight153` | 2.932e-01 | 8.431e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight154` | 4.430e-01 | 3.150e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight155` | 5.269e-01 | 2.604e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight156` | 1.966e-01 | 2.535e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight157` | 3.878e-01 | 4.966e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight158` | 4.733e-01 | 3.348e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight159` | 3.914e-01 | 3.699e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight16` | 1.136e-01 | 5.284e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight160` | 3.700e-01 | 4.487e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight161` | 3.757e-01 | 5.248e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight162` | 4.879e-01 | 3.343e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight163` | 5.076e-01 | 2.628e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight164` | 3.635e-01 | 5.778e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight165` | 3.791e-01 | 4.410e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight166` | 5.261e-01 | 2.747e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight167` | 4.905e-01 | 3.444e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight168` | 3.838e-01 | 5.575e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight169` | 4.429e-01 | 3.886e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight17` | 4.550e-01 | 2.996e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight170` | 2.931e-01 | 1.359e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight171` | 4.428e-01 | 3.213e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight172` | 4.155e-01 | 5.110e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight173` | 7.277e-02 | 1.260e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight174` | 4.373e-01 | 3.833e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight175` | 1.139e-01 | 7.429e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight176` | 5.048e-01 | 2.844e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight177` | 4.758e-01 | 3.284e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight178` | 3.772e-01 | 5.936e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight179` | 4.792e-01 | 3.328e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight18` | 4.777e-01 | 3.280e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight180` | 1.659e-01 | 3.125e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight181` | 4.014e-01 | 4.757e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight182` | 5.141e-01 | 2.968e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight183` | 4.746e-01 | 2.990e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight184` | 1.355e-01 | 4.105e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight185` | 3.899e-01 | 5.662e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight186` | 4.618e-01 | 3.454e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight187` | 4.050e-01 | 5.091e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight188` | 4.555e-01 | 3.148e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight189` | 3.278e-01 | 6.853e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight19` | 5.055e-01 | 2.934e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight190` | 4.208e-01 | 4.527e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight191` | 4.179e-01 | 4.646e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight2` | 4.022e-01 | 4.649e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight20` | 4.455e-01 | 3.483e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight21` | 3.951e-01 | 5.606e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight22` | 5.068e-01 | 2.877e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight23` | 4.988e-01 | 2.975e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight24` | 3.890e-01 | 4.043e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight25` | 2.820e-01 | 5.505e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight26` | 4.771e-01 | 3.269e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight27` | 4.564e-01 | 3.650e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight28` | 3.481e-01 | 5.911e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight29` | 4.330e-01 | 3.652e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight3` | 3.720e-01 | 5.247e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight30` | 5.058e-01 | 2.770e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight31` | 1.390e-01 | 5.505e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight32` | 5.188e-01 | 2.826e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight33` | 3.618e-01 | 5.348e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight34` | 4.446e-01 | 5.466e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight35` | 5.635e-01 | 3.177e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight36` | 3.646e-01 | 5.054e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight37` | 4.110e-01 | 4.426e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight38` | 2.024e-01 | 3.464e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight39` | 9.744e-02 | 1.071e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight4` | 4.039e-01 | 4.199e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight40` | 3.847e-01 | 4.854e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight41` | 4.431e-01 | 3.705e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight42` | 3.664e-01 | 4.637e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight43` | 4.926e-01 | 3.173e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight44` | 4.647e-01 | 3.332e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight45` | 4.816e-01 | 2.883e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight46` | 9.605e-02 | 7.924e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight47` | 3.118e-01 | 9.781e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight48` | 4.240e-01 | 4.332e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight49` | 5.070e-01 | 3.172e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight5` | 4.254e-01 | 4.196e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight50` | 4.821e-01 | 3.301e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight51` | 5.074e-01 | 2.982e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight52` | 3.388e-01 | 6.406e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight53` | 4.674e-01 | 3.458e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight54` | 2.889e-01 | 7.412e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight55` | 6.794e-01 | 1.805e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight56` | 4.110e-01 | 4.071e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight57` | 5.302e-01 | 2.843e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight58` | 3.956e-01 | 4.835e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight59` | 4.967e-01 | 3.633e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight6` | 4.240e-01 | 4.161e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight60` | 4.534e-01 | 3.431e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight61` | 4.814e-01 | 3.595e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight62` | 3.905e-01 | 4.611e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight63` | 4.376e-01 | 3.462e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight64` | 4.004e-01 | 3.615e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight65` | 4.629e-01 | 2.914e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight66` | 3.334e-01 | 6.129e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight67` | 1.387e-01 | 3.729e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight68` | 4.611e-01 | 3.522e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight69` | 4.753e-01 | 3.162e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight7` | 4.809e-01 | 3.579e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight70` | 4.363e-01 | 2.751e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight71` | 3.718e-01 | 4.436e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight72` | 4.309e-01 | 4.479e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight73` | 4.099e-01 | 5.148e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight74` | 4.604e-01 | 3.321e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight75` | 4.760e-01 | 3.683e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight76` | 5.177e-01 | 3.273e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight77` | 5.143e-01 | 2.785e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight78` | 6.560e-01 | 1.806e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight79` | 4.984e-01 | 3.049e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight8` | 4.559e-01 | 3.444e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight80` | 3.657e-01 | 3.465e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight81` | 4.245e-01 | 3.822e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight82` | 4.956e-01 | 3.789e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight83` | 5.292e-01 | 2.519e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight84` | 4.276e-01 | 4.001e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight85` | 5.090e-01 | 3.069e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight86` | 4.206e-01 | 4.996e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight87` | 4.780e-01 | 3.902e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight88` | 3.989e-01 | 4.155e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight89` | 5.329e-01 | 2.847e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight9` | 1.885e-01 | 2.519e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight90` | 5.810e-01 | 2.219e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight91` | 4.809e-01 | 3.055e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight92` | 5.324e-01 | 2.652e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight93` | 4.901e-01 | 3.144e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight94` | 5.349e-01 | 2.380e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight95` | 4.057e-01 | 5.441e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight96` | 5.140e-01 | 4.036e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight97` | 5.100e-01 | 2.711e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight98` | 4.405e-01 | 3.741e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.2.mlp.experts.linear_fc2.weight99` | 2.873e-01 | 7.972e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight0` | 3.994e-02 | 1.238e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight1` | 1.348e-01 | 4.306e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight10` | 5.356e-02 | 7.701e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight100` | 3.386e-01 | 4.285e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight101` | 1.053e-01 | 3.106e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight102` | 8.675e-02 | 6.383e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight103` | 3.411e-01 | 4.844e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight104` | 8.935e-02 | 4.292e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight105` | 1.693e-01 | 1.712e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight106` | 1.399e-01 | 3.464e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight107` | 1.410e-01 | 3.404e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight108` | 2.928e-01 | 5.949e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight109` | 1.685e-01 | 2.327e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight11` | 3.371e-01 | 4.487e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight110` | 1.552e-01 | 3.210e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight111` | 3.710e-01 | 3.775e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight112` | 3.437e-01 | 4.383e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight113` | 1.052e-01 | 6.880e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight114` | 3.557e-01 | 4.594e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight115` | 3.604e-01 | 4.273e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight116` | 3.840e-01 | 4.298e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight117` | 2.732e-02 | 1.765e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight118` | 3.551e-01 | 4.391e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight119` | 7.760e-02 | 1.098e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight12` | 3.275e-01 | 5.479e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight120` | 5.708e-02 | 9.759e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight121` | 3.601e-01 | 4.646e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight122` | 5.633e-02 | 1.996e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight123` | 3.553e-01 | 4.306e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight124` | 3.714e-01 | 4.255e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight125` | 3.207e-01 | 4.686e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight126` | 3.413e-01 | 3.859e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight127` | 8.753e-02 | 9.867e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight128` | 3.498e-01 | 4.327e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight129` | 7.024e-02 | 4.548e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight13` | 1.032e-01 | 6.580e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight130` | 3.456e-01 | 4.378e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight131` | 3.715e-01 | 3.533e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight132` | 9.899e-02 | 7.350e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight133` | 1.384e-01 | 4.050e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight134` | 3.306e-01 | 4.904e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight135` | 3.443e-01 | 3.863e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight136` | 3.474e-01 | 4.742e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight137` | 1.340e-01 | 4.228e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight138` | 7.216e-02 | 3.158e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight139` | 1.433e-01 | 3.665e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight14` | 4.630e-01 | 2.159e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight140` | 1.044e-01 | 6.011e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight141` | 8.792e-02 | 9.143e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight142` | 3.072e-01 | 7.451e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight143` | 1.487e-01 | 2.546e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight144` | 3.352e-01 | 5.065e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight145` | 8.407e-02 | 5.320e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight146` | 3.612e-01 | 3.871e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight147` | 3.430e-01 | 3.926e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight148` | 3.902e-01 | 3.586e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight149` | 3.425e-01 | 4.561e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight15` | 3.394e-01 | 4.808e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight150` | 3.511e-01 | 3.936e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight151` | 1.840e-01 | 1.768e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight152` | 2.456e-02 | 1.296e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight153` | 6.932e-02 | 4.805e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight154` | 2.108e-01 | 1.152e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight155` | 3.612e-01 | 4.350e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight156` | 9.177e-02 | 9.746e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight157` | 3.334e-01 | 4.801e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight158` | 1.016e-01 | 6.880e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight159` | 3.364e-02 | 7.399e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight16` | 2.268e-01 | 9.936e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight160` | 4.802e-02 | 2.675e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight161` | 7.509e-02 | 1.100e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight162` | 1.251e-01 | 5.085e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight163` | 3.472e-01 | 4.415e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight164` | 1.678e-01 | 2.440e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight165` | 7.691e-02 | 1.236e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight166` | 3.447e-01 | 4.294e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight167` | 1.099e-01 | 6.145e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight168` | 2.693e-01 | 8.487e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight169` | 3.309e-01 | 4.960e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight17` | 1.134e-01 | 6.023e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight170` | 4.449e-01 | 3.282e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight171` | 3.710e-02 | 7.612e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight172` | 3.883e-01 | 3.647e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight173` | 1.828e-01 | 2.011e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight174` | 1.172e-01 | 4.804e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight175` | 2.449e-02 | 1.064e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight176` | 3.348e-01 | 5.201e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight177` | 1.625e-01 | 2.638e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight178` | 3.539e-01 | 3.720e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight179` | 3.397e-01 | 4.326e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight18` | 2.946e-01 | 5.342e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight180` | 2.995e-02 | 9.997e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight181` | 4.367e-01 | 2.976e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight182` | 3.327e-01 | 4.209e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight183` | 3.639e-01 | 4.172e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight184` | 3.654e-01 | 3.744e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight185` | 3.668e-01 | 3.647e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight186` | 2.769e-01 | 5.761e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight187` | 3.292e-01 | 4.561e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight188` | 3.994e-01 | 3.824e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight189` | 1.317e-01 | 3.689e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight19` | 3.389e-02 | 7.810e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight190` | 1.685e-01 | 3.181e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight191` | 3.371e-01 | 4.674e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight2` | 3.013e-01 | 5.228e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight20` | 3.479e-01 | 3.980e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight21` | 3.537e-01 | 4.164e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight22` | 1.459e-01 | 3.271e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight23` | 3.588e-02 | 1.900e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight24` | 1.024e-01 | 3.654e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight25` | 3.683e-01 | 3.835e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight26` | 2.618e-02 | 1.459e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight27` | 4.041e-02 | 6.857e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight28` | 3.252e-01 | 4.048e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight29` | 3.216e-01 | 5.175e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight3` | 8.906e-02 | 3.990e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight30` | 3.508e-01 | 3.768e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight31` | 1.435e-01 | 3.880e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight32` | 4.433e-01 | 3.239e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight33` | 3.572e-01 | 4.374e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight34` | 3.458e-01 | 4.303e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight35` | 3.504e-01 | 3.996e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight36` | 1.317e-01 | 4.754e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight37` | 2.837e-02 | 1.195e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight38` | 7.800e-02 | 1.250e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight39` | 3.210e-01 | 5.150e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight4` | 3.648e-01 | 4.035e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight40` | 3.374e-01 | 4.280e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight41` | 1.593e-01 | 3.070e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight42` | 7.426e-02 | 5.292e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight43` | 8.738e-02 | 9.766e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight44` | 2.793e-01 | 9.648e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight45` | 3.735e-01 | 3.490e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight46` | 1.337e-01 | 3.812e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight47` | 1.599e-01 | 2.628e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight48` | 3.305e-01 | 5.308e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight49` | 2.963e-01 | 5.386e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight5` | 3.738e-01 | 3.617e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight50` | 3.521e-01 | 4.138e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight51` | 1.643e-02 | 2.969e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight52` | 1.593e-01 | 3.191e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight53` | 5.000e-01 | 0 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight54` | 3.345e-01 | 4.799e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight55` | 1.093e-01 | 4.056e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight56` | 3.466e-01 | 4.922e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight57` | 3.397e-01 | 4.428e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight58` | 4.561e-02 | 7.597e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight59` | 3.182e-01 | 5.339e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight6` | 3.257e-01 | 5.010e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight60` | 3.743e-01 | 4.236e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight61` | 9.632e-02 | 7.197e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight62` | 1.034e-01 | 7.560e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight63` | 1.719e-01 | 1.890e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight64` | 1.647e-01 | 2.731e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight65` | 3.176e-02 | 1.537e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight66` | 1.323e-02 | 4.316e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight67` | 3.203e-01 | 5.420e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight68` | 3.658e-01 | 3.621e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight69` | 7.390e-02 | 1.136e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight7` | 3.062e-02 | 1.171e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight70` | 4.410e-02 | 1.056e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight71` | 4.567e-02 | 5.407e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight72` | 1.258e-01 | 2.341e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight73` | 3.555e-01 | 3.933e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight74` | 1.949e-01 | 1.494e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight75` | 3.256e-01 | 5.307e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight76` | 3.384e-01 | 4.551e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight77` | 4.153e-02 | 1.079e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight78` | 8.136e-02 | 1.138e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight79` | 3.253e-01 | 4.907e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight8` | 3.710e-01 | 3.306e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight80` | 3.663e-01 | 3.785e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight81` | 2.736e-01 | 5.948e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight82` | 1.483e-01 | 3.688e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight83` | 2.835e-01 | 6.166e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight84` | 3.821e-01 | 4.226e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight85` | 1.153e-01 | 5.014e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight86` | 3.478e-01 | 5.078e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight87` | 4.413e-01 | 3.207e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight88` | 3.761e-02 | 8.976e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight89` | 3.564e-01 | 4.131e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight9` | 3.496e-01 | 4.733e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight90` | 1.433e-01 | 3.182e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight91` | 3.121e-01 | 5.240e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight92` | 2.616e-01 | 7.065e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight93` | 4.431e-02 | 1.143e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight94` | 4.647e-01 | 2.988e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight95` | 4.586e-01 | 2.281e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight96` | 2.645e-02 | 1.031e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight97` | 1.532e-01 | 2.758e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight98` | 3.517e-01 | 4.478e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc1.weight99` | 2.987e-01 | 5.487e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight0` | 4.304e-02 | 1.300e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight1` | 1.598e-01 | 3.715e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight10` | 6.051e-02 | 7.023e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight100` | 4.299e-01 | 2.989e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight101` | 1.581e-01 | 1.978e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight102` | 1.027e-01 | 3.758e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight103` | 3.534e-01 | 4.473e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight104` | 1.113e-01 | 4.413e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight105` | 2.522e-01 | 7.398e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight106` | 2.096e-01 | 1.886e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight107` | 1.779e-01 | 2.346e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight108` | 3.498e-01 | 5.651e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight109` | 2.131e-01 | 1.902e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight11` | 4.073e-01 | 4.105e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight110` | 1.867e-01 | 2.627e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight111` | 4.499e-01 | 3.045e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight112` | 3.811e-01 | 4.014e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight113` | 1.198e-01 | 5.912e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight114` | 3.684e-01 | 4.370e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight115` | 4.069e-01 | 3.015e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight116` | 4.042e-01 | 4.075e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight117` | 3.508e-02 | 1.469e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight118` | 3.954e-01 | 3.595e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight119` | 9.248e-02 | 1.069e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight12` | 3.355e-01 | 4.637e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight120` | 7.518e-02 | 8.423e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight121` | 4.086e-01 | 4.111e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight122` | 7.535e-02 | 1.701e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight123` | 4.075e-01 | 3.903e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight124` | 4.194e-01 | 4.105e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight125` | 3.553e-01 | 4.140e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight126` | 4.262e-01 | 3.769e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight127` | 8.193e-02 | 8.382e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight128` | 3.628e-01 | 4.280e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight129` | 1.041e-01 | 2.397e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight13` | 1.612e-01 | 4.457e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight130` | 3.694e-01 | 4.309e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight131` | 4.096e-01 | 3.229e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight132` | 1.261e-01 | 5.428e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight133` | 1.860e-01 | 3.006e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight134` | 3.344e-01 | 4.907e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight135` | 4.287e-01 | 3.101e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight136` | 3.863e-01 | 3.863e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight137` | 1.447e-01 | 3.391e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight138` | 1.270e-01 | 2.361e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight139` | 1.899e-01 | 2.640e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight14` | 4.869e-01 | 2.168e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight140` | 1.310e-01 | 5.147e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight141` | 1.018e-01 | 7.676e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight142` | 3.397e-01 | 5.932e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight143` | 1.996e-01 | 1.393e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight144` | 3.400e-01 | 5.183e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight145` | 1.089e-01 | 3.051e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight146` | 4.136e-01 | 3.580e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight147` | 4.487e-01 | 3.116e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight148` | 4.441e-01 | 2.951e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight149` | 3.958e-01 | 4.092e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight15` | 3.433e-01 | 4.539e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight150` | 4.007e-01 | 3.325e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight151` | 2.486e-01 | 8.753e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight152` | 2.835e-02 | 1.287e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight153` | 1.055e-01 | 2.494e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight154` | 2.620e-01 | 7.322e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight155` | 3.811e-01 | 4.030e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight156` | 1.187e-01 | 7.384e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight157` | 3.511e-01 | 4.591e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight158` | 1.382e-01 | 3.598e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight159` | 4.033e-02 | 6.480e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight16` | 2.512e-01 | 9.052e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight160` | 5.826e-02 | 2.416e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight161` | 7.424e-02 | 1.262e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight162` | 1.249e-01 | 5.693e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight163` | 4.289e-01 | 2.997e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight164` | 2.471e-01 | 1.663e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight165` | 8.506e-02 | 1.243e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight166` | 4.170e-01 | 3.608e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight167` | 1.530e-01 | 3.617e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight168` | 3.087e-01 | 6.527e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight169` | 3.351e-01 | 4.921e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight17` | 1.349e-01 | 4.573e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight170` | 4.512e-01 | 3.314e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight171` | 6.269e-02 | 4.552e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight172` | 4.053e-01 | 3.245e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight173` | 2.241e-01 | 1.657e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight174` | 1.700e-01 | 3.430e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight175` | 4.358e-02 | 6.582e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight176` | 3.482e-01 | 4.753e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight177` | 1.920e-01 | 2.501e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight178` | 4.848e-01 | 2.804e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight179` | 3.854e-01 | 4.119e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight18` | 3.434e-01 | 3.869e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight180` | 4.670e-02 | 6.854e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight181` | 4.876e-01 | 2.816e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight182` | 4.307e-01 | 3.130e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight183` | 3.951e-01 | 3.732e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight184` | 4.182e-01 | 3.402e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight185` | 4.711e-01 | 2.933e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight186` | 3.191e-01 | 4.938e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight187` | 3.699e-01 | 3.947e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight188` | 4.193e-01 | 3.163e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight189` | 1.536e-01 | 2.760e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight19` | 4.568e-02 | 6.074e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight190` | 1.837e-01 | 3.129e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight191` | 3.666e-01 | 4.333e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight2` | 3.442e-01 | 4.132e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight20` | 3.895e-01 | 3.571e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight21` | 4.191e-01 | 3.603e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight22` | 1.967e-01 | 1.860e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight23` | 3.358e-02 | 1.396e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight24` | 1.263e-01 | 3.039e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight25` | 4.456e-01 | 2.940e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight26` | 3.128e-02 | 1.276e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight27` | 5.312e-02 | 5.837e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight28` | 3.741e-01 | 3.349e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight29` | 3.775e-01 | 4.544e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight3` | 1.671e-01 | 1.830e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight30` | 4.755e-01 | 3.703e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight31` | 1.687e-01 | 3.327e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight32` | 4.261e-01 | 3.201e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight33` | 4.136e-01 | 3.459e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight34` | 4.165e-01 | 3.454e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight35` | 4.210e-01 | 3.536e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight36` | 1.237e-01 | 4.173e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight37` | 4.007e-02 | 9.643e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight38` | 6.721e-02 | 1.144e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight39` | 3.380e-01 | 4.802e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight4` | 3.955e-01 | 3.629e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight40` | 3.946e-01 | 3.310e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight41` | 1.758e-01 | 2.532e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight42` | 8.127e-02 | 4.924e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight43` | 1.102e-01 | 6.701e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight44` | 3.119e-01 | 1.086e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight45` | 4.913e-01 | 2.646e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight46` | 1.627e-01 | 3.021e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight47` | 2.272e-01 | 1.767e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight48` | 3.417e-01 | 4.908e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight49` | 3.199e-01 | 4.563e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight5` | 4.463e-01 | 3.021e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight50` | 4.005e-01 | 3.798e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight51` | 1.872e-02 | 2.752e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight52` | 1.993e-01 | 2.840e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight53` | 5.098e-01 | 0 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight54` | 3.667e-01 | 4.388e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight55` | 1.286e-01 | 2.658e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight56` | 3.619e-01 | 4.627e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight57` | 3.613e-01 | 3.842e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight58` | 4.907e-02 | 6.106e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight59` | 3.282e-01 | 5.053e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight6` | 3.443e-01 | 4.838e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight60` | 4.040e-01 | 3.838e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight61` | 1.093e-01 | 3.257e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight62` | 1.138e-01 | 5.636e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight63` | 1.962e-01 | 1.003e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight64` | 2.270e-01 | 2.332e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight65` | 3.684e-02 | 1.437e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight66` | 1.462e-02 | 4.289e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight67` | 3.345e-01 | 4.763e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight68` | 4.459e-01 | 3.354e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight69` | 8.123e-02 | 1.007e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight7` | 3.430e-02 | 1.060e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight70` | 5.384e-02 | 6.119e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight71` | 9.393e-02 | 3.057e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight72` | 1.533e-01 | 1.369e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight73` | 4.374e-01 | 2.939e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight74` | 2.629e-01 | 9.418e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight75` | 3.323e-01 | 5.040e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight76` | 3.861e-01 | 4.371e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight77` | 4.888e-02 | 8.978e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight78` | 9.244e-02 | 1.063e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight79` | 3.664e-01 | 3.847e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight8` | 4.050e-01 | 2.497e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight80` | 3.971e-01 | 3.519e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight81` | 2.855e-01 | 3.298e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight82` | 1.911e-01 | 2.797e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight83` | 3.081e-01 | 5.839e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight84` | 3.847e-01 | 4.005e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight85` | 1.436e-01 | 3.388e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight86` | 3.679e-01 | 4.696e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight87` | 4.973e-01 | 2.884e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight88` | 5.209e-02 | 8.003e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight89` | 4.017e-01 | 3.711e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight9` | 3.663e-01 | 4.479e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight90` | 1.731e-01 | 1.606e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight91` | 3.773e-01 | 5.189e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight92` | 2.616e-01 | 7.633e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight93` | 5.289e-02 | 1.054e-01 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight94` | 4.624e-01 | 2.972e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight95` | 5.232e-01 | 2.298e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight96` | 2.850e-02 | 8.836e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight97` | 2.454e-01 | 1.627e-02 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight98` | 3.632e-01 | 4.313e-03 | 3001 | 57 | 1095 |
| `p3001.grad.decoder.layers.3.mlp.experts.linear_fc2.weight99` | 2.956e-01 | 4.786e-03 | 3001 | 57 | 1095 |

### `L2.causal.model.attribution`（pass）

| 用例 | p | past_routing_equal | experts_count_changed | experts_changed_and_small | past_tokens_out_differ | explained_by_small_changed_expert | min_count_changed_experts |
|---|---|---|---|---|---|---|---|
| `p2048.layers.1` | 2048 | True | 192 | 13 | 61 | 61 | 9 |
| `p3001.layers.1` | 3001 | True | 189 | 13 | 83 | 83 | 9 |

数字全部来自 `results.json` 与 `details/`，本报告不手写数字。
