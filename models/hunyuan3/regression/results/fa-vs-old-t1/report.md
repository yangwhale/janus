```
[T1 · v5p · 2026-11-05 22:26] ERROR（工具）
窗口：LKG reg-t1-lkg2-a@237a32705f → reg-t1-fa-a@0622774a16（声明：数学级，范围 R02,R05,R06,R08,R09,R12,R13,R15,R16,R20,R21）
依赖与开关：有变化或没判出（看 R04、R05）      对照臂：本次没跑（手动阶段）
第一个不同：R06 R06.must_equal：缺字段 ["ref:'maxtext_config_full.dense_fp32_accumulation'"]
各行：pass 8 / fail 0 / inconclusive 0 / error 2      只报：R02, R05, R08, R09, R12, R13, R15, R16, R20, R21, R22
下一步：看报 error 的行：运行目录缺文件、目录写错或 janus 版本变了；修好后重判，不动 LKG
```

- 档位配置 `models/hunyuan3/regression/tiers/t1.yaml`（sha256 abd58de70a97…）；目录 `regression-t1.yaml`（sha256 9837ad44c3a5…）；janus 0.1.0.dev0
- 被测提交：{maxtext: 0622774a160c17ba8f3f66941cf1ee49cd9158b2, janus: 4db01d7a27a1db8f302e7afe705e0786df4e4680}；LKG 提交：{maxtext: 237a32705feb9f028910eb28216299139415d007, janus: 3fff02dc81307f9128565a8a9d4ec61725f7aa46}
- LKG 候选：否

| 行 | 状态 | 检查 | 关键数 | 说明 |
|---|---|---|---|---|
| R01 | pass | `R01.run_complete`（pass） | planned=1000, logged=1000, duplicate=0, missing=0, extra=0 | 1000 步全部完成、无重复、无重启 |
| R02 | report | `R02.inputs`（fail） | fields=7, must_equal_differ=1, other_differ=0, missing=0, equals_rules=3 | 必须相同（或等于给定值）的字段不符：["ref:config.sha256 == 'a238883b7f914d5c99f313d2d5ef68fa472164ef81e95d48d1046cb7e593ee6b'"]；声明「数学级」改了这一行：变化在声明范围内，只报（看 details 核对确实只变了声明的字段） |
|  |  | `R02.registry`（fail） | tier_config_sha256=abd58de70a97, catalog_files=2 | 与 lkg.yaml 登记的不同：['regression-t1.yaml', 'regression-common.yaml']（改档位定义要重建 LKG） |
| R03 | pass | `R03.data_fingerprint`（pass） | units_per_field={tokens: 16000, labels: 16000, loss_mask: 16000, position_ids: 16000}, units=65000, differ=0 | token 与首步 loss 逐位相同 |
| R04 | pass | `R04.versions`（pass） | fields=7, must_equal_differ=0, other_differ=1, missing=0, equals_rules=0 | 必须相同的字段都相同；其他不同 1 项 |
| R05 | report | `R05.process_flags`（fail） | fields=6, must_equal_differ=1, other_differ=0, missing=0, equals_rules=0 | 必须相同（或等于给定值）的字段不符：["maxtext_config_full ↔ maxtext_config_full（6 个子键：['dense_fp32_accumulation', 'lm_head_dh_gauge_scale', 'lm_head_dh_wbar_r；声明「数学级」改了这一行：变化在声明范围内，只报（看 details 核对确实只变了声明的字段） |
|  |  | `R05.compiler_flagfile`（pass） | rules=0, bad=0, fingerprint=8494a7374f44f508, ref_fingerprint=8494a7374f44f508, fingerprint_equal=True | 全部规则满足，指纹与参照相同 |
|  |  | `R05.compiler_debug_options`（pass） | rules=0, bad=0, fingerprint=59407bf7dd627e57, ref_fingerprint=59407bf7dd627e57, fingerprint_equal=True | 全部规则满足，指纹与参照相同 |
| R06 | error | `R06.must_equal`（error） | fields=29, must_equal_differ=3, other_differ=0, missing=1, equals_rules=29 | 缺字段 ["ref:'maxtext_config_full.dense_fp32_accumulation'"] |
| R07 | pass | `R07.state_init`（pass） | params=91, differ=0, only_one_side=0 | 全部参数逐位相同 |
| R08 | report | `R08.step1_loss`（fail） | fields=2, first_differ=[lm_loss@1, loss@1], tensors=0, tensors_differ=0 | 不可复现：['lm_loss@1', 'loss@1']，张量不同 0；声明「数学级」且在范围内：只报，由对齐门（R25）拦 |
|  |  | `R08.loss_tolerance`（pass，只报） | d0=1.4223e-05, signed_mean=-6.9838e-06, mare=0.000112873, max_step=0.000588492, max_step_at=107 | 在噪声参照内 |
| R09 | report | `R09.step1_norms`（fail） | fields=2, first_differ=[raw_grad_norm@1], tensors=0, tensors_differ=0 | 不可复现：['raw_grad_norm@1']，张量不同 0；声明「数学级」且在范围内：只报，由对齐门（R25）拦 |
| R12 | report | `R12.loss`（fail） | d0=1.4223e-05, signed_mean=-6.9838e-06, mare=0.000112873, max_step=0.000588492, max_step_at=107 | ；声明「数学级」且在范围内：只报，由对齐门（R25）拦 |
|  |  | `R12.loss_tolerance`（pass，只报） | d0=1.4223e-05, signed_mean=-6.9838e-06, mare=0.000112873, max_step=0.000588492, max_step_at=107 | 在噪声参照内 |
| R13 | report | `R13.raw_grad_norm`（fail） | d0=0.000265542, signed_mean=0.000389493, mare=0.028345, max_step=0.303517, max_step_at=107 | ；声明「数学级」且在范围内：只报，由对齐门（R25）拦 |
|  |  | `R13.param_norm`（fail） | d0=0, signed_mean=1.85544e-08, mare=5.6307e-08, max_step=2.40736e-07, max_step_at=116 |  |
|  |  | `R13.raw_grad_norm_tolerance`（pass，只报） | d0=0.000265542, signed_mean=0.000389493, mare=0.028345, max_step=0.303517, max_step_at=107 | 在噪声参照内 |
|  |  | `R13.param_norm_tolerance`（pass，只报） | d0=0, signed_mean=1.85544e-08, mare=5.6307e-08, max_step=2.40736e-07, max_step_at=116 | 在噪声参照内 |
| R14 | pass | `R14.lr_wd`（pass） | fields=2, first_differ=[], tensors=0, tensors_differ=0 | 逐位可复现 |
| R15 | report | `R15.state_final`（fail） | params=91, differ=88, only_one_side=0 | 不同 88 个：['state/MoEBiasVar/model/decoder/moe_layers/Hunyuan3MoeBlock_0/MoeBlock_0/gate/bias', 'state/OptVariable/optimizer/opt_state/0/mu/de；声明「数学级」且在范围内：只报，由对齐门（R25）拦 |
| R16 | report | `R16.bias_stats`（fail） | signed_mean_max=0.0320659, mare_max=0.037901, max_step_max=0.125178, stats=6 | ；声明「数学级」且在范围内：只报，由对齐门（R25）拦 |
|  |  | `R16.bias_stats_tolerance`（pass，只报） | signed_mean_max=0.0320659, mare_max=0.037901, max_step_max=0.125178, stats=6 | 在噪声参照内 |
| R17 | pass | `R17.health`（pass） | fields=3, problems=0 | 健康 |
| R18 | pass | `R18.dtypes`（pass） | entries=182, wrong_dtype=0, unmatched=0, rule_hits={params/*: 58, MoEBiasVar/*: 2, */opt_state/*/mu/*: 58, */opt_state/*/nu/*: 58, */opt_state/*/count: 4, OptState/optimizer/step: 2} | 182 项 dtype 全部符合规则 |
| R20 | report | `R20.artifact_counts`（fail） | rules=7, bad=2 | 不满足：['all_reduce=8（参照 9）', 'reduce_precision=63（参照 31）']；声明「数学级」改了这一行：变化在声明范围内，只报（看 details 核对确实只变了声明的字段） |
| R21 | report | `R21.hlo_fingerprint`（fail，只报） | rules=0, bad=0, fingerprint=97592661a9934200, ref_fingerprint=209341bb971a111d, fingerprint_equal=False | 不满足：['指纹不同'] |
| R22 | report | （regress.py） | test_median_s=0.903432, lkg_median_s=0.898192, rel_diff=0.0058343, steps_used=998, skip_first=2, threshold=0.00570345 | 只报。偏离 LKG +0.58%，超过阈值 0.57%，连续两次超过就开性能 issue（阈值出自 calibration.yaml）；峰值 HBM、编译期临时区还没记 |
| R23 | error | （regress.py） | rows=21, problems=1 | R06 没出结论 |
| R25 | pass | `R25.loss_vs_golden`（fail，只报） | d0_max=1.67184e-05, signed_mean_max=0.000136965, mare_max=0.000174189, max_step_max=0.00069146, goldens=3 | 超过噪声参照两倍；不变差：对 GPU golden 的三个量，新值 − LKG 值都不大于 TPU 执行噪声上限（标准第三节） |

数字全部来自 janus 的 `results.json` 与运行目录（`regression.json` 是同一份数据），本报告不手写数字。
