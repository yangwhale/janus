```
[T1 · v5p · 2026-11-19 01:02] 数值 FAIL
窗口：LKG reg-t1-fa-a@0622774a16 → sync0119-t1-a@7e49199ee9（声明：升级依赖）
依赖与开关：有变化或没判出（看 R04、R05）      对照臂：本次没跑（手动阶段）
第一个不同：R04 R04.versions：必须相同（或等于给定值）的字段不符：['image ↔ image', "versions ↔ versions（6 个子键：['flax', 'jax', 'jaxlib']）"]
各行：pass 10 / fail 2 / inconclusive 7 / error 0      只报：R21, R22
下一步：先跑对照臂（LKG 的提交在今天的环境上重跑一次）分清环境还是代码；对照臂逐位就二分，判据取第一个不同的量（标准第八节）
```

- 档位配置 `models/hunyuan3/regression/tiers/t1.yaml`（sha256 2f9c5908788e…）；目录 `regression-t1.yaml`（内容指纹 39f2cf8e8940…）；janus 0.1.0.dev0
- 被测提交：{maxtext: 7e49199ee99b5d84cffc8192006ed95301e3c11c, janus: b4e84eaa3862cb754efbbf43847477a58f6ef6f2}；LKG 提交：{maxtext: 0622774a160c17ba8f3f66941cf1ee49cd9158b2, janus: 4db01d7a27a1db8f302e7afe705e0786df4e4680}
- LKG 候选：否

| 行 | 状态 | 检查 | 关键数 | 说明 |
|---|---|---|---|---|
| R01 | pass | `R01.run_complete`（pass） | planned=1000, logged=1000, duplicate=0, missing=0, extra=0 | 1000 步全部完成、无重复、无重启 |
| R02 | pass | `R02.inputs`（pass） | fields=7, must_equal_differ=0, other_differ=0, missing=0, equals_rules=3 | 必须相同的字段都相同；其他不同 0 项 |
|  |  | `R02.registry`（pass） | tier_config_sha256=2f9c5908788e, catalog_files=2, catalog_hash=content-v1 | 档位配置的 sha256、目录的内容指纹与 lkg.yaml 登记的相同 |
| R03 | pass | `R03.data_fingerprint`（pass） | units_per_field={tokens: 16000, labels: 16000, loss_mask: 16000, position_ids: 16000}, units=65000, differ=0 | token 与首步 loss 逐位相同 |
| R04 | fail | `R04.versions`（fail） | fields=7, must_equal_differ=2, other_differ=1, missing=0, equals_rules=0 | 必须相同（或等于给定值）的字段不符：['image ↔ image', "versions ↔ versions（6 个子键：['flax', 'jax', 'jaxlib']）"] |
| R05 | fail | `R05.process_flags`（fail） | fields=6, must_equal_differ=5, other_differ=1, missing=0, equals_rules=0 | 必须相同（或等于给定值）的字段不符：["flags.at_start.env ↔ flags.at_start.env（1 个子键：['TPU_ML_PLATFORM_VERSION']）", "flags.after_compile.env ↔ flags.after_comp |
|  |  | `R05.compiler_flagfile`（fail） | rules=0, bad=0, fingerprint=6f47d9db7a9f642d, ref_fingerprint=8494a7374f44f508, fingerprint_equal=False | 不满足：['指纹不同'] |
|  |  | `R05.compiler_debug_options`（pass） | rules=0, bad=0, fingerprint=59407bf7dd627e57, ref_fingerprint=59407bf7dd627e57, fingerprint_equal=True | 全部规则满足，指纹与参照相同 |
| R06 | pass | `R06.must_equal`（pass） | fields=29, must_equal_differ=0, other_differ=0, missing=0, equals_rules=29 | 必须相同的字段都相同；其他不同 0 项 |
| R07 | pass | `R07.state_init`（pass） | params=91, differ=0, only_one_side=0 | 全部参数逐位相同 |
| R08 | inconclusive | `R08.step1_loss`（fail） | fields=2, first_differ=[lm_loss@1, loss@1], tensors=0, tensors_differ=0 | 不可复现：['lm_loss@1', 'loss@1']，张量不同 0；声明「升级依赖」且不逐位：按执行级判（门槛待标定时为 inconclusive），并按标准第九节做归因 |
|  |  | `R08.loss_tolerance`（pass，只报） | **0.97 倍尺子**, d0=2.66158e-06, signed_mean=6.44102e-06, mare=0.000115841, max_step=0.000518373, max_step_at=683 | 在噪声参照内 |
| R09 | inconclusive | `R09.step1_norms`（fail） | fields=2, first_differ=[raw_grad_norm@1], tensors=0, tensors_differ=0 | 不可复现：['raw_grad_norm@1']，张量不同 0；声明「升级依赖」且不逐位：按执行级判（门槛待标定时为 inconclusive），并按标准第九节做归因 |
|  |  | `R09.step1_grad_norm_tolerance`（pass，只报） | **0.27 倍尺子**, d0=8.47942e-06, signed_mean=0, mare=0, max_step=0, max_step_at=None | 在噪声参照内 |
| R12 | inconclusive | `R12.loss`（fail） | d0=2.66158e-06, signed_mean=6.44102e-06, mare=0.000115841, max_step=0.000518373, max_step_at=683 | ；声明「升级依赖」且不逐位：按执行级判（门槛待标定时为 inconclusive），并按标准第九节做归因 |
|  |  | `R12.loss_tolerance`（pass，只报） | **0.97 倍尺子**, d0=2.66158e-06, signed_mean=6.44102e-06, mare=0.000115841, max_step=0.000518373, max_step_at=683 | 在噪声参照内 |
| R13 | inconclusive | `R13.raw_grad_norm`（fail） | d0=8.47942e-06, signed_mean=0.00238521, mare=0.0262895, max_step=0.206535, max_step_at=168 | ；声明「升级依赖」且不逐位：按执行级判（门槛待标定时为 inconclusive），并按标准第九节做归因 |
|  |  | `R13.param_norm`（fail） | d0=0, signed_mean=-3.57435e-08, mare=8.32957e-08, max_step=3.20978e-07, max_step_at=242 |  |
|  |  | `R13.raw_grad_norm_tolerance`（inconclusive，只报） | **1.01 倍尺子**, d0=8.47942e-06, signed_mean=0.00238521, mare=0.0262895, max_step=0.206535, max_step_at=168 | 介于噪声参照一到两倍之间，或参照不足 |
|  |  | `R13.param_norm_tolerance`（pass，只报） | **0.82 倍尺子**, d0=0, signed_mean=-3.57435e-08, mare=8.32957e-08, max_step=3.20978e-07, max_step_at=242 | 在噪声参照内 |
| R14 | pass | `R14.lr_wd`（pass） | fields=2, first_differ=[], tensors=0, tensors_differ=0 | 逐位可复现 |
| R15 | inconclusive | `R15.state_final`（fail） | params=91, differ=88, only_one_side=0 | 不同 88 个：['state/MoEBiasVar/model/decoder/moe_layers/Hunyuan3MoeBlock_0/MoeBlock_0/gate/bias', 'state/OptVariable/optimizer/opt_state/0/mu/de；声明「升级依赖」且不逐位：按执行级判（门槛待标定时为 inconclusive），并按标准第九节做归因 |
| R16 | inconclusive | `R16.bias_stats`（fail） | signed_mean_max=0.017285, mare_max=0.018897, max_step_max=0.125178, stats=6 | ；声明「升级依赖」且不逐位：按执行级判（门槛待标定时为 inconclusive），并按标准第九节做归因 |
|  |  | `R16.bias_stats_tolerance`（pass，只报） | **0.54 倍尺子**, signed_mean_max=0.017285, mare_max=0.018897, max_step_max=0.125178, stats=6 | 在噪声参照内 |
| R17 | pass | `R17.health`（pass） | fields=3, problems=0 | 健康 |
| R18 | pass | `R18.dtypes`（pass） | entries=182, wrong_dtype=0, unmatched=0, rule_hits={params/*: 58, MoEBiasVar/*: 2, */opt_state/*/mu/*: 58, */opt_state/*/nu/*: 58, */opt_state/*/count: 4, OptState/optimizer/step: 2} | 182 项 dtype 全部符合规则 |
| R20 | pass | `R20.artifact_counts`（pass） | rules=7, bad=0 | 全部规则满足 |
| R21 | report | `R21.hlo_fingerprint`（fail，只报） | rules=0, bad=0, fingerprint=89e42526bc0fe49a, ref_fingerprint=97592661a9934200, fingerprint_equal=False | 不满足：['指纹不同'] |
| R22 | report | （regress.py） | test_median_s=0.878328, lkg_median_s=0.903432, rel_diff=-0.0277877, steps_used=998, skip_first=2 | 只报。阈值待标定（标准 R22：同一提交 ≥ 5 次运行的步时极差）；峰值 HBM、编译期临时区还没记 |
| R23 | pass | （regress.py） | rows=21, problems=0 | 每一行都出了结论、单元数对得上 |
| R25 | inconclusive | `R25.loss_vs_golden`（pass） | **0.77 倍尺子**, d0_max=1.86314e-05, signed_mean_max=1.90236e-05, mare_max=0.000105573, max_step_max=0.000544143, goldens=10 | 在噪声参照内；对齐门本身（对 golden 基线、golden 噪声）pass；数值变了：「不变差」要 TPU 执行噪声当尺子，还没标定或已作废，只报新值与 LKG 值的差（标准第三节）；没给 LKG 的回归结果（--lkg-results），只报被测值 |

数字全部来自 janus 的 `results.json` 与运行目录（`regression.json` 是同一份数据），本报告不手写数字。
