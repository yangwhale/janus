```
[T0 · v5p · 2026-11-05 22:27] PASS
窗口：LKG reg-t0-fa-a@0622774a16 → reg-t0-fa-e@0622774a16（声明：数学不变）
依赖与开关：无变化（R04、R05 逐位）      对照臂：本次没跑（手动阶段）
第一个不同：无
各行：pass 15 / fail 0 / inconclusive 0 / error 0      只报：R21, R22
下一步：无。数值与 LKG 逐位相同，LKG 指针可前移到被测提交（数值没变，不用批准）
```

- 档位配置 `models/hunyuan3/regression/tiers/t0.yaml`（sha256 d95fa250a233…）；目录 `regression-t0.yaml`（sha256 ed85e06a6723…）；janus 0.1.0.dev0
- 被测提交：{maxtext: 0622774a160c17ba8f3f66941cf1ee49cd9158b2, janus: 4db01d7a27a1db8f302e7afe705e0786df4e4680}；LKG 提交：{maxtext: 0622774a160c17ba8f3f66941cf1ee49cd9158b2, janus: 4db01d7a27a1db8f302e7afe705e0786df4e4680}
- LKG 候选：是

| 行 | 状态 | 检查 | 关键数 | 说明 |
|---|---|---|---|---|
| R01 | pass | `R01.run_complete`（pass） | planned=20, logged=20, duplicate=0, missing=0, extra=0 | 20 步全部完成、无重复、无重启 |
| R02 | pass | `R02.inputs`（pass） | fields=7, must_equal_differ=0, other_differ=0, missing=0, equals_rules=3 | 必须相同的字段都相同；其他不同 0 项 |
|  |  | `R02.registry`（pass） | tier_config_sha256=d95fa250a233, catalog_files=2 | 档位配置与目录的 sha256 与 lkg.yaml 登记的相同 |
| R03 | pass | `R03.data_fingerprint`（pass） | units_per_field={tokens: 320, labels: 320, loss_mask: 320, position_ids: 320}, units=1300, differ=0 | token 与首步 loss 逐位相同 |
| R04 | pass | `R04.versions`（pass） | fields=7, must_equal_differ=0, other_differ=0, missing=0, equals_rules=0 | 必须相同的字段都相同；其他不同 0 项 |
| R05 | pass | `R05.process_flags`（pass） | fields=6, must_equal_differ=0, other_differ=0, missing=0, equals_rules=0 | 必须相同的字段都相同；其他不同 0 项 |
|  |  | `R05.compiler_flagfile`（pass） | rules=0, bad=0, fingerprint=8494a7374f44f508, ref_fingerprint=8494a7374f44f508, fingerprint_equal=True | 全部规则满足，指纹与参照相同 |
|  |  | `R05.compiler_debug_options`（pass） | rules=0, bad=0, fingerprint=59407bf7dd627e57, ref_fingerprint=59407bf7dd627e57, fingerprint_equal=True | 全部规则满足，指纹与参照相同 |
| R06 | pass | `R06.must_equal`（pass） | fields=29, must_equal_differ=0, other_differ=0, missing=0, equals_rules=29 | 必须相同的字段都相同；其他不同 0 项 |
| R07 | pass | `R07.state_init`（pass） | params=91, differ=0, only_one_side=0 | 全部参数逐位相同 |
| R08 | pass | `R08.step1_loss`（pass） | fields=2, first_differ=[], tensors=0, tensors_differ=0 | 逐位可复现 |
|  |  | `R08.loss_tolerance`（pass，只报） | d0=0, signed_mean=0, mare=0, max_step=0, max_step_at=2 | 在噪声参照内 |
| R09 | pass | `R09.step1_norms`（pass） | fields=2, first_differ=[], tensors=0, tensors_differ=0 | 逐位可复现 |
| R14 | pass | `R14.lr_wd`（pass） | fields=2, first_differ=[], tensors=0, tensors_differ=0 | 逐位可复现 |
| R15 | pass | `R15.state_final`（pass） | params=91, differ=0, only_one_side=0 | 全部参数逐位相同 |
| R17 | pass | `R17.health`（pass） | fields=3, problems=0 | 健康 |
| R18 | pass | `R18.dtypes`（pass） | entries=182, wrong_dtype=0, unmatched=0, rule_hits={params/*: 58, MoEBiasVar/*: 2, */opt_state/*/mu/*: 58, */opt_state/*/nu/*: 58, */opt_state/*/count: 4, OptState/optimizer/step: 2} | 182 项 dtype 全部符合规则 |
| R20 | pass | `R20.artifact_counts`（pass） | rules=7, bad=0 | 全部规则满足 |
| R21 | report | `R21.hlo_fingerprint`（pass，只报） | rules=0, bad=0, fingerprint=97592661a9934200, ref_fingerprint=97592661a9934200, fingerprint_equal=True | 全部规则满足，指纹与参照相同 |
| R22 | report | （regress.py） | test_median_s=0.91094, lkg_median_s=0.905232, rel_diff=0.00630528, steps_used=18, skip_first=2, threshold=0.00627678 | 只报。偏离 LKG +0.63%，超过阈值 0.63%，连续两次超过就开性能 issue（阈值出自 calibration.yaml）；峰值 HBM、编译期临时区还没记 |
| R23 | pass | （regress.py） | rows=17, problems=0 | 每一行都出了结论、单元数对得上 |

数字全部来自 janus 的 `results.json` 与运行目录（`regression.json` 是同一份数据），本报告不手写数字。
