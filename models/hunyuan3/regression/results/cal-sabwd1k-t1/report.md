```
[T1 · v5p · 2026-11-05 12:59] 数值 FAIL
窗口：LKG reg-t1-lkg2-a@237a32705f → cal-sabwd1k-t1@237a32705f（声明：数学不变）
依赖与开关：有变化或没判出（看 R04、R05）      对照臂：本次没跑（手动阶段）
第一个不同：R02 R02.inputs：必须相同（或等于给定值）的字段不符：['config.overrides ↔ config.overrides']
各行：pass 11 / fail 7 / inconclusive 1 / error 0      只报：R21, R22
下一步：先跑对照臂（LKG 的提交在今天的环境上重跑一次）分清环境还是代码；对照臂逐位就二分，判据取第一个不同的量（标准第八节）
```

- 档位配置 `models/hunyuan3/regression/tiers/t1.yaml`（sha256 abd58de70a97…）；目录 `regression-t1.yaml`（sha256 c68049cc4bb8…）；janus 0.1.0.dev0
- 被测提交：{maxtext: 237a32705feb9f028910eb28216299139415d007, janus: 3fff02dc81307f9128565a8a9d4ec61725f7aa46}；LKG 提交：{maxtext: 237a32705feb9f028910eb28216299139415d007, janus: 3fff02dc81307f9128565a8a9d4ec61725f7aa46}
- LKG 候选：否

| 行 | 状态 | 检查 | 关键数 | 说明 |
|---|---|---|---|---|
| R01 | pass | `R01.run_complete`（pass） | planned=1000, logged=1000, duplicate=0, missing=0, extra=0 | 1000 步全部完成、无重复、无重启 |
| R02 | fail | `R02.inputs`（fail） | fields=7, must_equal_differ=1, other_differ=0, missing=0, equals_rules=3 | 必须相同（或等于给定值）的字段不符：['config.overrides ↔ config.overrides'] |
|  |  | `R02.registry`（pass） | tier_config_sha256=abd58de70a97, catalog_files=2 | 档位配置与目录的 sha256 与 lkg.yaml 登记的相同 |
| R03 | pass | `R03.data_fingerprint`（pass） | units_per_field={tokens: 16000, labels: 16000, loss_mask: 16000, position_ids: 16000}, units=65000, differ=0 | token 与首步 loss 逐位相同 |
| R04 | pass | `R04.versions`（pass） | fields=7, must_equal_differ=0, other_differ=0, missing=0, equals_rules=0 | 必须相同的字段都相同；其他不同 0 项 |
| R05 | fail | `R05.process_flags`（fail） | fields=6, must_equal_differ=1, other_differ=0, missing=0, equals_rules=0 | 必须相同（或等于给定值）的字段不符：["maxtext_config_full ↔ maxtext_config_full（10 个子键：['local_sa_block_kv_dkv', 'local_sa_block_kv_dkv_compute', 'local_sa_bl |
|  |  | `R05.compiler_flagfile`（pass） | rules=0, bad=0, fingerprint=8494a7374f44f508, ref_fingerprint=8494a7374f44f508, fingerprint_equal=True | 全部规则满足，指纹与参照相同 |
|  |  | `R05.compiler_debug_options`（pass） | rules=0, bad=0, fingerprint=59407bf7dd627e57, ref_fingerprint=59407bf7dd627e57, fingerprint_equal=True | 全部规则满足，指纹与参照相同 |
| R06 | pass | `R06.must_equal`（pass） | fields=25, must_equal_differ=0, other_differ=0, missing=0, equals_rules=25 | 必须相同的字段都相同；其他不同 0 项 |
| R07 | pass | `R07.state_init`（pass） | params=91, differ=0, only_one_side=0 | 全部参数逐位相同 |
| R08 | pass | `R08.step1_loss`（pass） | fields=2, first_differ=[], tensors=0, tensors_differ=0 | 逐位可复现 |
|  |  | `R08.loss_tolerance`（inconclusive，只报） | d0=0, signed_mean=1.3065e-05, mare=0.000100217, max_step=0.000545444, max_step_at=203 | 门槛待标定（TPU 执行噪声待标定（README 第五节）），只报被测值 |
| R09 | fail | `R09.step1_norms`（fail） | fields=2, first_differ=[raw_grad_norm@1], tensors=0, tensors_differ=0 | 不可复现：['raw_grad_norm@1']，张量不同 0 |
| R12 | fail | `R12.loss`（fail） | d0=0, signed_mean=1.3065e-05, mare=0.000100217, max_step=0.000545444, max_step_at=203 |  |
|  |  | `R12.loss_tolerance`（inconclusive，只报） | d0=0, signed_mean=1.3065e-05, mare=0.000100217, max_step=0.000545444, max_step_at=203 | 门槛待标定（TPU 执行噪声待标定（README 第五节）），只报被测值 |
| R13 | fail | `R13.raw_grad_norm`（fail） | d0=-3.26218e-07, signed_mean=0.00170943, mare=0.0294628, max_step=0.453278, max_step_at=201 |  |
|  |  | `R13.param_norm`（fail） | d0=0, signed_mean=4.68286e-08, mare=6.24115e-08, max_step=2.40734e-07, max_step_at=202 |  |
|  |  | `R13.raw_grad_norm_tolerance`（inconclusive，只报） | d0=-3.26218e-07, signed_mean=0.00170943, mare=0.0294628, max_step=0.453278, max_step_at=201 | 门槛待标定（TPU 执行噪声待标定（README 第五节）），只报被测值 |
|  |  | `R13.param_norm_tolerance`（inconclusive，只报） | d0=0, signed_mean=4.68286e-08, mare=6.24115e-08, max_step=2.40734e-07, max_step_at=202 | 门槛待标定（TPU 执行噪声待标定（README 第五节）），只报被测值 |
| R14 | pass | `R14.lr_wd`（pass） | fields=2, first_differ=[], tensors=0, tensors_differ=0 | 逐位可复现 |
| R15 | fail | `R15.state_final`（fail） | params=91, differ=88, only_one_side=0 | 不同 88 个：['state/MoEBiasVar/model/decoder/moe_layers/Hunyuan3MoeBlock_0/MoeBlock_0/gate/bias', 'state/OptVariable/optimizer/opt_state/0/mu/de |
| R16 | fail | `R16.bias_stats`（fail） | signed_mean_max=0.0267051, mare_max=0.0304891, max_step_max=0.0649563, stats=6 |  |
|  |  | `R16.bias_stats_tolerance`（inconclusive，只报） | signed_mean_max=0.0267051, mare_max=0.0304891, max_step_max=0.0649563, stats=6 | 门槛待标定（TPU 执行噪声待标定（README 第五节）），只报被测值 |
| R17 | pass | `R17.health`（pass） | fields=3, problems=0 | 健康 |
| R18 | pass | `R18.dtypes`（pass） | entries=182, wrong_dtype=0, unmatched=0, rule_hits={params/*: 58, MoEBiasVar/*: 2, */opt_state/*/mu/*: 58, */opt_state/*/nu/*: 58, */opt_state/*/count: 4, OptState/optimizer/step: 2} | 182 项 dtype 全部符合规则 |
| R20 | pass | `R20.artifact_counts`（pass） | rules=7, bad=0 | 全部规则满足 |
| R21 | report | `R21.hlo_fingerprint`（fail，只报） | rules=0, bad=0, fingerprint=acc3b78859bdb55b, ref_fingerprint=209341bb971a111d, fingerprint_equal=False | 不满足：['指纹不同'] |
| R22 | report | （regress.py） | test_median_s=0.89099, lkg_median_s=0.898192, rel_diff=-0.0080177, steps_used=998, skip_first=2 | 只报。阈值待标定（标准 R22：同一提交 ≥ 5 次运行的步时极差）；峰值 HBM、编译期临时区还没记 |
| R23 | pass | （regress.py） | rows=21, problems=0 | 每一行都出了结论、单元数对得上 |
| R25 | inconclusive | `R25.loss_vs_golden`（fail，只报） | d0_max=2.49528e-06, signed_mean_max=0.000126467, mare_max=0.000159982, max_step_max=0.000781236, goldens=3 | 超过噪声参照两倍；数值变了：「不变差」要 TPU 执行噪声当尺子，还没标定或已作废，只报新值与 LKG 值的差（标准第三节）；没给 LKG 的回归结果（--lkg-results），只报被测值 |

数字全部来自 janus 的 `results.json` 与运行目录（`regression.json` 是同一份数据），本报告不手写数字。
