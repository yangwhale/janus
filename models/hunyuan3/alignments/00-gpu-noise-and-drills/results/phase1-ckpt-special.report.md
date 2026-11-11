# janus 报告：hunyuan3-4l / phase1-ckpt-special

- 被测：`g61-sv-save, g61-sv-load`
- janus 0.1.0.dev0；目录 `models/hunyuan3/frameworks/megatron/ckpt_special.py`（sha256 4722bc237d73…）
- 结果：pass 3、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `CKPT.special.poked` | op_replay | L0 | pass | bit_match_frac=1.000e+00, cases=72 | 逐位全中（bit_match_frac） |
| `CKPT.special.full` | op_replay | L0 | pass | bit_match_frac=1.000e+00, cases=72 | 逐位全中（bit_match_frac） |
| `CKPT.special.values` | op_replay | L0 | pass | bit_match_frac=1.000e+00, cases=72 | 逐位全中（bit_match_frac） |


## 环境与开关（进程内实测）

- `save_run_args`：{'hf': '/data/hy3-4l', 'out': '/data/runs/g61-sv-save', 'load': '/data/hy3-4l-mcore', 'steps': 2, 'gbs': 16, 'mbs': 1, 'seq': 4096, 'ep': 8, 'lr': 1e-05, 'seed': 1234, 'deterministic': True, 'data': '/data/ds/fw_hy3_text_document', 'janus_run': '/data/janus-runs2/g61-sv-save', 'side': 'golden-g61-sv-save', 'dump_steps': '', 'dump_grads': False, 'no_dist_opt': True, 'full': False, 'inject_from': None, 'fault': None, 'tp': 1, 'cp': 1, 'recompute': False, 'routing_replay': False, 'full_steps': None, 'full_only': None, 'save_batches': False, 'save': '/data/ckpt-g61-sv', 'save_interval': None, 'resume': None, 'observe': None, 'special': '^decoder\\.layers\\.(0\\.self_attention\\.q_layernorm\\.weight|1\\.mlp\\.experts\\.linear_fc2\\.weight0|1\\.mlp\\.router\\.expert_bias)$', 'no_tf32': False, 'ce_fusion': None, 'bias_act_fusion': False, 'attention_backend': None, 'row_stats_every': 0, 'group_norms': False, 'grad_reduce_bf16': False, 'router_true_fp32': False, 'resid_probe_every': 0, 'resid_probe_seq': None, 'rr_arm': None, 'export_params_at': None, 'export_dir': None, 'import_params': None, 'import_round_bf16': False, 'skip_steps': 0, 'fp32': False, 'bf16_weights': False}
- `load_run_args`：{'hf': '/data/hy3-4l', 'out': '/data/runs/g61-sv-load', 'load': '/data/hy3-4l-mcore', 'steps': 3, 'gbs': 16, 'mbs': 1, 'seq': 4096, 'ep': 8, 'lr': 1e-05, 'seed': 1234, 'deterministic': True, 'data': '/data/ds/fw_hy3_text_document', 'janus_run': '/data/janus-runs2/g61-sv-load', 'side': 'golden-g61-sv-load', 'dump_steps': '', 'dump_grads': False, 'no_dist_opt': True, 'full': False, 'inject_from': None, 'fault': None, 'tp': 1, 'cp': 1, 'recompute': False, 'routing_replay': False, 'full_steps': None, 'full_only': None, 'save_batches': False, 'save': None, 'save_interval': None, 'resume': '/data/ckpt-g61-sv', 'observe': None, 'special': '^decoder\\.layers\\.(0\\.self_attention\\.q_layernorm\\.weight|1\\.mlp\\.experts\\.linear_fc2\\.weight0|1\\.mlp\\.router\\.expert_bias)$', 'no_tf32': False, 'ce_fusion': None, 'bias_act_fusion': False, 'attention_backend': None, 'row_stats_every': 0, 'group_norms': False, 'grad_reduce_bf16': False, 'router_true_fp32': False, 'resid_probe_every': 0, 'resid_probe_seq': None, 'rr_arm': None, 'export_params_at': None, 'export_dir': None, 'import_params': None, 'import_round_bf16': False, 'skip_steps': 0, 'fp32': False, 'bf16_weights': False}
- `special_bits_fp32`：['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000']

## 逐用例明细

列：bit_match_frac 恢复后对存档前（或对期望埋入的位模式）的逐位相同率；*_neg_zero / *_subnormal 是张量里 −0.0 与次正规数的个数。

### `CKPT.special.poked`（pass）

| 用例 | bit_match_frac | dtype |
|---|---|---|
| `rank0.buffer.decoder.layers.1.mlp.router.expert_bias[:8]` | 1 | float32 |
| `rank0.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank0.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank0.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank0.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank0.main_param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank0.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank0.param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | bfloat16 |
| `rank0.param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | bfloat16 |
| `rank1.buffer.decoder.layers.1.mlp.router.expert_bias[:8]` | 1 | float32 |
| `rank1.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank1.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank1.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank1.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank1.main_param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank1.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank1.param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | bfloat16 |
| `rank1.param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | bfloat16 |
| `rank2.buffer.decoder.layers.1.mlp.router.expert_bias[:8]` | 1 | float32 |
| `rank2.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank2.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank2.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank2.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank2.main_param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank2.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank2.param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | bfloat16 |
| `rank2.param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | bfloat16 |
| `rank3.buffer.decoder.layers.1.mlp.router.expert_bias[:8]` | 1 | float32 |
| `rank3.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank3.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank3.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank3.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank3.main_param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank3.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank3.param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | bfloat16 |
| `rank3.param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | bfloat16 |
| `rank4.buffer.decoder.layers.1.mlp.router.expert_bias[:8]` | 1 | float32 |
| `rank4.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank4.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank4.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank4.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank4.main_param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank4.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank4.param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | bfloat16 |
| `rank4.param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | bfloat16 |
| `rank5.buffer.decoder.layers.1.mlp.router.expert_bias[:8]` | 1 | float32 |
| `rank5.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank5.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank5.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank5.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank5.main_param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank5.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank5.param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | bfloat16 |
| `rank5.param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | bfloat16 |
| `rank6.buffer.decoder.layers.1.mlp.router.expert_bias[:8]` | 1 | float32 |
| `rank6.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank6.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank6.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank6.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank6.main_param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank6.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank6.param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | bfloat16 |
| `rank6.param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | bfloat16 |
| `rank7.buffer.decoder.layers.1.mlp.router.expert_bias[:8]` | 1 | float32 |
| `rank7.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank7.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank7.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank7.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank7.main_param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 |
| `rank7.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 |
| `rank7.param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | bfloat16 |
| `rank7.param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | bfloat16 |

### `CKPT.special.full`（pass）

| 用例 | bit_match_frac | dtype | numel | saved_neg_zero | saved_subnormal | loaded_neg_zero | loaded_subnormal |
|---|---|---|---|---|---|---|---|
| `rank0.buffer.decoder.layers.1.mlp.router.expert_bias` | 1 | float32 | 192 | 1 | 5 | 1 | 5 |
| `rank0.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 1 | 5 | 1 | 5 |
| `rank0.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 1 | 5 | 1 | 5 |
| `rank0.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 0 | 5 | 0 | 5 |
| `rank0.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 0 | 5 | 0 | 5 |
| `rank0.main_param.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 1 | 5 | 1 | 5 |
| `rank0.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 1 | 5 | 1 | 5 |
| `rank0.param.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | bfloat16 | 128 | 2 | 2 | 2 | 2 |
| `rank0.param.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | bfloat16 | 6291456 | 2 | 2 | 2 | 2 |
| `rank1.buffer.decoder.layers.1.mlp.router.expert_bias` | 1 | float32 | 192 | 1 | 5 | 1 | 5 |
| `rank1.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 1 | 5 | 1 | 5 |
| `rank1.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 1 | 5 | 1 | 5 |
| `rank1.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 0 | 5 | 0 | 5 |
| `rank1.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 0 | 5 | 0 | 5 |
| `rank1.main_param.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 1 | 5 | 1 | 5 |
| `rank1.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 1 | 5 | 1 | 5 |
| `rank1.param.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | bfloat16 | 128 | 2 | 2 | 2 | 2 |
| `rank1.param.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | bfloat16 | 6291456 | 2 | 2 | 2 | 2 |
| `rank2.buffer.decoder.layers.1.mlp.router.expert_bias` | 1 | float32 | 192 | 1 | 5 | 1 | 5 |
| `rank2.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 1 | 5 | 1 | 5 |
| `rank2.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 1 | 5 | 1 | 5 |
| `rank2.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 0 | 5 | 0 | 5 |
| `rank2.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 0 | 5 | 0 | 5 |
| `rank2.main_param.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 1 | 5 | 1 | 5 |
| `rank2.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 1 | 5 | 1 | 5 |
| `rank2.param.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | bfloat16 | 128 | 2 | 2 | 2 | 2 |
| `rank2.param.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | bfloat16 | 6291456 | 2 | 2 | 2 | 2 |
| `rank3.buffer.decoder.layers.1.mlp.router.expert_bias` | 1 | float32 | 192 | 1 | 5 | 1 | 5 |
| `rank3.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 1 | 5 | 1 | 5 |
| `rank3.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 1 | 5 | 1 | 5 |
| `rank3.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 0 | 5 | 0 | 5 |
| `rank3.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 0 | 5 | 0 | 5 |
| `rank3.main_param.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 1 | 5 | 1 | 5 |
| `rank3.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 1 | 5 | 1 | 5 |
| `rank3.param.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | bfloat16 | 128 | 2 | 2 | 2 | 2 |
| `rank3.param.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | bfloat16 | 6291456 | 2 | 2 | 2 | 2 |
| `rank4.buffer.decoder.layers.1.mlp.router.expert_bias` | 1 | float32 | 192 | 1 | 5 | 1 | 5 |
| `rank4.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 1 | 5 | 1 | 5 |
| `rank4.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 1 | 5 | 1 | 5 |
| `rank4.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 0 | 5 | 0 | 5 |
| `rank4.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 0 | 5 | 0 | 5 |
| `rank4.main_param.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 1 | 5 | 1 | 5 |
| `rank4.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 1 | 5 | 1 | 5 |
| `rank4.param.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | bfloat16 | 128 | 2 | 2 | 2 | 2 |
| `rank4.param.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | bfloat16 | 6291456 | 2 | 2 | 2 | 2 |
| `rank5.buffer.decoder.layers.1.mlp.router.expert_bias` | 1 | float32 | 192 | 1 | 5 | 1 | 5 |
| `rank5.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 1 | 5 | 1 | 5 |
| `rank5.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 1 | 5 | 1 | 5 |
| `rank5.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 0 | 5 | 0 | 5 |
| `rank5.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 0 | 5 | 0 | 5 |
| `rank5.main_param.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 1 | 5 | 1 | 5 |
| `rank5.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 1 | 5 | 1 | 5 |
| `rank5.param.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | bfloat16 | 128 | 2 | 2 | 2 | 2 |
| `rank5.param.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | bfloat16 | 6291456 | 2 | 2 | 2 | 2 |
| `rank6.buffer.decoder.layers.1.mlp.router.expert_bias` | 1 | float32 | 192 | 1 | 5 | 1 | 5 |
| `rank6.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 1 | 5 | 1 | 5 |
| `rank6.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 1 | 5 | 1 | 5 |
| `rank6.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 0 | 5 | 0 | 5 |
| `rank6.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 0 | 5 | 0 | 5 |
| `rank6.main_param.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 1 | 5 | 1 | 5 |
| `rank6.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 1 | 5 | 1 | 5 |
| `rank6.param.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | bfloat16 | 128 | 2 | 2 | 2 | 2 |
| `rank6.param.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | bfloat16 | 6291456 | 2 | 2 | 2 | 2 |
| `rank7.buffer.decoder.layers.1.mlp.router.expert_bias` | 1 | float32 | 192 | 1 | 5 | 1 | 5 |
| `rank7.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 1 | 5 | 1 | 5 |
| `rank7.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 1 | 5 | 1 | 5 |
| `rank7.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 0 | 5 | 0 | 5 |
| `rank7.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 0 | 5 | 0 | 5 |
| `rank7.main_param.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | float32 | 128 | 1 | 5 | 1 | 5 |
| `rank7.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | float32 | 6291456 | 1 | 5 | 1 | 5 |
| `rank7.param.decoder.layers.0.self_attention.q_layernorm.weight` | 1 | bfloat16 | 128 | 2 | 2 | 2 | 2 |
| `rank7.param.decoder.layers.1.mlp.experts.linear_fc2.weight0` | 1 | bfloat16 | 6291456 | 2 | 2 | 2 | 2 |

### `CKPT.special.values`（pass）

| 用例 | bit_match_frac | dtype | loaded_bits |
|---|---|---|---|
| `rank0.buffer.decoder.layers.1.mlp.router.expert_bias[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank0.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank0.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank0.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x00000000', '0x00000000', '0x000116c2', '0x000116c2', '0x00000001', '0x00000001', '0x007fffff', '0x00800000'] |
| `rank0.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x00000000', '0x00000000', '0x000116c2', '0x000116c2', '0x00000001', '0x00000001', '0x007fffff', '0x00800000'] |
| `rank0.main_param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank0.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank0.param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | bfloat16 | ['0x8000', '0x0000', '0x0001', '0x8001', '0x0000', '0x8000', '0x0080', '0x0080'] |
| `rank0.param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | bfloat16 | ['0x8000', '0x0000', '0x0001', '0x8001', '0x0000', '0x8000', '0x0080', '0x0080'] |
| `rank1.buffer.decoder.layers.1.mlp.router.expert_bias[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank1.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank1.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank1.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x00000000', '0x00000000', '0x000116c2', '0x000116c2', '0x00000001', '0x00000001', '0x007fffff', '0x00800000'] |
| `rank1.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x00000000', '0x00000000', '0x000116c2', '0x000116c2', '0x00000001', '0x00000001', '0x007fffff', '0x00800000'] |
| `rank1.main_param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank1.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank1.param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | bfloat16 | ['0x8000', '0x0000', '0x0001', '0x8001', '0x0000', '0x8000', '0x0080', '0x0080'] |
| `rank1.param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | bfloat16 | ['0x8000', '0x0000', '0x0001', '0x8001', '0x0000', '0x8000', '0x0080', '0x0080'] |
| `rank2.buffer.decoder.layers.1.mlp.router.expert_bias[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank2.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank2.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank2.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x00000000', '0x00000000', '0x000116c2', '0x000116c2', '0x00000001', '0x00000001', '0x007fffff', '0x00800000'] |
| `rank2.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x00000000', '0x00000000', '0x000116c2', '0x000116c2', '0x00000001', '0x00000001', '0x007fffff', '0x00800000'] |
| `rank2.main_param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank2.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank2.param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | bfloat16 | ['0x8000', '0x0000', '0x0001', '0x8001', '0x0000', '0x8000', '0x0080', '0x0080'] |
| `rank2.param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | bfloat16 | ['0x8000', '0x0000', '0x0001', '0x8001', '0x0000', '0x8000', '0x0080', '0x0080'] |
| `rank3.buffer.decoder.layers.1.mlp.router.expert_bias[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank3.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank3.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank3.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x00000000', '0x00000000', '0x000116c2', '0x000116c2', '0x00000001', '0x00000001', '0x007fffff', '0x00800000'] |
| `rank3.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x00000000', '0x00000000', '0x000116c2', '0x000116c2', '0x00000001', '0x00000001', '0x007fffff', '0x00800000'] |
| `rank3.main_param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank3.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank3.param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | bfloat16 | ['0x8000', '0x0000', '0x0001', '0x8001', '0x0000', '0x8000', '0x0080', '0x0080'] |
| `rank3.param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | bfloat16 | ['0x8000', '0x0000', '0x0001', '0x8001', '0x0000', '0x8000', '0x0080', '0x0080'] |
| `rank4.buffer.decoder.layers.1.mlp.router.expert_bias[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank4.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank4.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank4.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x00000000', '0x00000000', '0x000116c2', '0x000116c2', '0x00000001', '0x00000001', '0x007fffff', '0x00800000'] |
| `rank4.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x00000000', '0x00000000', '0x000116c2', '0x000116c2', '0x00000001', '0x00000001', '0x007fffff', '0x00800000'] |
| `rank4.main_param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank4.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank4.param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | bfloat16 | ['0x8000', '0x0000', '0x0001', '0x8001', '0x0000', '0x8000', '0x0080', '0x0080'] |
| `rank4.param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | bfloat16 | ['0x8000', '0x0000', '0x0001', '0x8001', '0x0000', '0x8000', '0x0080', '0x0080'] |
| `rank5.buffer.decoder.layers.1.mlp.router.expert_bias[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank5.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank5.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank5.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x00000000', '0x00000000', '0x000116c2', '0x000116c2', '0x00000001', '0x00000001', '0x007fffff', '0x00800000'] |
| `rank5.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x00000000', '0x00000000', '0x000116c2', '0x000116c2', '0x00000001', '0x00000001', '0x007fffff', '0x00800000'] |
| `rank5.main_param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank5.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank5.param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | bfloat16 | ['0x8000', '0x0000', '0x0001', '0x8001', '0x0000', '0x8000', '0x0080', '0x0080'] |
| `rank5.param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | bfloat16 | ['0x8000', '0x0000', '0x0001', '0x8001', '0x0000', '0x8000', '0x0080', '0x0080'] |
| `rank6.buffer.decoder.layers.1.mlp.router.expert_bias[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank6.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank6.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank6.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x00000000', '0x00000000', '0x000116c2', '0x000116c2', '0x00000001', '0x00000001', '0x007fffff', '0x00800000'] |
| `rank6.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x00000000', '0x00000000', '0x000116c2', '0x000116c2', '0x00000001', '0x00000001', '0x007fffff', '0x00800000'] |
| `rank6.main_param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank6.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank6.param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | bfloat16 | ['0x8000', '0x0000', '0x0001', '0x8001', '0x0000', '0x8000', '0x0080', '0x0080'] |
| `rank6.param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | bfloat16 | ['0x8000', '0x0000', '0x0001', '0x8001', '0x0000', '0x8000', '0x0080', '0x0080'] |
| `rank7.buffer.decoder.layers.1.mlp.router.expert_bias[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank7.exp_avg.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank7.exp_avg.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank7.exp_avg_sq.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x00000000', '0x00000000', '0x000116c2', '0x000116c2', '0x00000001', '0x00000001', '0x007fffff', '0x00800000'] |
| `rank7.exp_avg_sq.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x00000000', '0x00000000', '0x000116c2', '0x000116c2', '0x00000001', '0x00000001', '0x007fffff', '0x00800000'] |
| `rank7.main_param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank7.main_param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | float32 | ['0x80000000', '0x00000000', '0x000116c2', '0x800116c2', '0x00000001', '0x80000001', '0x007fffff', '0x00800000'] |
| `rank7.param.decoder.layers.0.self_attention.q_layernorm.weight[:8]` | 1 | bfloat16 | ['0x8000', '0x0000', '0x0001', '0x8001', '0x0000', '0x8000', '0x0080', '0x0080'] |
| `rank7.param.decoder.layers.1.mlp.experts.linear_fc2.weight0[:8]` | 1 | bfloat16 | ['0x8000', '0x0000', '0x0001', '0x8001', '0x0000', '0x8000', '0x0080', '0x0080'] |

数字全部来自 `results.json` 与 `details/`，本报告不手写数字。
