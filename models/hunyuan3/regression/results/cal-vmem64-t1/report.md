```
[T1 · v5p · 2026-11-05 06:26] INCONCLUSIVE
窗口：LKG reg-t1-lkg2-a@237a32705f → cal-vmem64-t1@237a32705f（声明：执行级，范围 R02,R05,R08,R09,R12,R13,R15,R16,R20,R21）
依赖与开关：有变化或没判出（看 R04、R05）      对照臂：本次没跑（手动阶段）
第一个不同：R08 R08.step1_loss：不可复现：['lm_loss@1', 'loss@1']，张量不同 0，['lm_loss@1', 'loss@1']
各行：pass 11 / fail 0 / inconclusive 7 / error 0      只报：R05, R21, R22
下一步：待标定的门槛（TPU 执行噪声）标完才能判；在那之前人工看报告里的数
```

- 档位配置 `models/hunyuan3/regression/tiers/t1.yaml`（sha256 abd58de70a97…）；目录 `regression-t1.yaml`（sha256 c68049cc4bb8…）；janus 0.1.0.dev0
- 被测提交：{maxtext: 237a32705feb9f028910eb28216299139415d007, janus: 3fff02dc81307f9128565a8a9d4ec61725f7aa46}；LKG 提交：{maxtext: 237a32705feb9f028910eb28216299139415d007, janus: 3fff02dc81307f9128565a8a9d4ec61725f7aa46}
- LKG 候选：否

| 行 | 状态 | 检查 | 关键数 | 说明 |
|---|---|---|---|---|
| R01 | pass | `R01.run_complete`（pass） | planned=1000, logged=1000, duplicate=0, missing=0, extra=0 | 1000 步全部完成、无重复、无重启 |
| R02 | pass | `R02.inputs`（pass） | fields=7, must_equal_differ=0, other_differ=0, missing=0, equals_rules=3 | 必须相同的字段都相同；其他不同 0 项 |
|  |  | `R02.registry`（pass） | tier_config_sha256=abd58de70a97, catalog_files=2 | 档位配置与目录的 sha256 与 lkg.yaml 登记的相同 |
| R03 | pass | `R03.data_fingerprint`（pass） | units_per_field={tokens: 16000, labels: 16000, loss_mask: 16000, position_ids: 16000}, units=65000, differ=0 | token 与首步 loss 逐位相同 |
| R04 | pass | `R04.versions`（pass） | fields=7, must_equal_differ=0, other_differ=0, missing=0, equals_rules=0 | 必须相同的字段都相同；其他不同 0 项 |
| R05 | report | `R05.process_flags`（fail） | fields=6, must_equal_differ=3, other_differ=0, missing=0, equals_rules=0 | 必须相同（或等于给定值）的字段不符：["flags.at_start.env ↔ flags.at_start.env（1 个子键：['LIBTPU_INIT_ARGS']）", "flags.after_compile.env ↔ flags.after_compile.env；声明「执行级」改了这一行：变化在声明范围内，只报（看 details 核对确实只变了声明的字段） |
|  |  | `R05.compiler_flagfile`（fail） | rules=0, bad=0, fingerprint=26957bb3db050391, ref_fingerprint=8494a7374f44f508, fingerprint_equal=False | 不满足：['指纹不同'] |
|  |  | `R05.compiler_debug_options`（pass） | rules=0, bad=0, fingerprint=59407bf7dd627e57, ref_fingerprint=59407bf7dd627e57, fingerprint_equal=True | 全部规则满足，指纹与参照相同 |
| R06 | pass | `R06.must_equal`（pass） | fields=25, must_equal_differ=0, other_differ=0, missing=0, equals_rules=25 | 必须相同的字段都相同；其他不同 0 项 |
| R07 | pass | `R07.state_init`（pass） | params=91, differ=0, only_one_side=0 | 全部参数逐位相同 |
| R08 | inconclusive | `R08.step1_loss`（fail） | fields=2, first_differ=[lm_loss@1, loss@1], tensors=0, tensors_differ=0 | 不可复现：['lm_loss@1', 'loss@1']，张量不同 0；声明「执行级」且在范围内：不逐位，改按容差门判（门槛待标定时为 inconclusive） |
|  |  | `R08.loss_tolerance`（inconclusive，只报） | d0=4.15878e-07, signed_mean=-2.76263e-05, mare=0.000102522, max_step=0.000512645, max_step_at=748 | 门槛待标定（TPU 执行噪声待标定（README 第五节）），只报被测值 |
| R09 | inconclusive | `R09.step1_norms`（fail） | fields=2, first_differ=[raw_grad_norm@1, param_norm@1], tensors=0, tensors_differ=0 | 不可复现：['raw_grad_norm@1', 'param_norm@1']，张量不同 0；声明「执行级」且在范围内：不逐位；这一行没有容差门，待 TPU 执行噪声标定后才能判 |
| R12 | inconclusive | `R12.loss`（fail） | d0=4.15878e-07, signed_mean=-2.76263e-05, mare=0.000102522, max_step=0.000512645, max_step_at=748 | ；声明「执行级」且在范围内：不逐位，改按容差门判（门槛待标定时为 inconclusive） |
|  |  | `R12.loss_tolerance`（inconclusive，只报） | d0=4.15878e-07, signed_mean=-2.76263e-05, mare=0.000102522, max_step=0.000512645, max_step_at=748 | 门槛待标定（TPU 执行噪声待标定（README 第五节）），只报被测值 |
| R13 | inconclusive | `R13.raw_grad_norm`（fail） | d0=1.31575e-05, signed_mean=-0.000218424, mare=0.0246408, max_step=0.308, max_step_at=179 | ；声明「执行级」且在范围内：不逐位，改按容差门判（门槛待标定时为 inconclusive） |
|  |  | `R13.param_norm`（fail） | d0=-1.60493e-07, signed_mean=3.93583e-08, mare=6.58654e-08, max_step=2.40734e-07, max_step_at=211 |  |
|  |  | `R13.raw_grad_norm_tolerance`（inconclusive，只报） | d0=1.31575e-05, signed_mean=-0.000218424, mare=0.0246408, max_step=0.308, max_step_at=179 | 门槛待标定（TPU 执行噪声待标定（README 第五节）），只报被测值 |
|  |  | `R13.param_norm_tolerance`（inconclusive，只报） | d0=-1.60493e-07, signed_mean=3.93583e-08, mare=6.58654e-08, max_step=2.40734e-07, max_step_at=211 | 门槛待标定（TPU 执行噪声待标定（README 第五节）），只报被测值 |
| R14 | pass | `R14.lr_wd`（pass） | fields=2, first_differ=[], tensors=0, tensors_differ=0 | 逐位可复现 |
| R15 | inconclusive | `R15.state_final`（fail） | params=91, differ=88, only_one_side=0 | 不同 88 个：['state/MoEBiasVar/model/decoder/moe_layers/Hunyuan3MoeBlock_0/MoeBlock_0/gate/bias', 'state/OptVariable/optimizer/opt_state/0/mu/de；声明「执行级」且在范围内：不逐位；这一行没有容差门，待 TPU 执行噪声标定后才能判 |
| R16 | inconclusive | `R16.bias_stats`（fail） | signed_mean_max=0.0475481, mare_max=0.0482992, max_step_max=0.125178, stats=6 | ；声明「执行级」且在范围内：不逐位，改按容差门判（门槛待标定时为 inconclusive） |
|  |  | `R16.bias_stats_tolerance`（inconclusive，只报） | signed_mean_max=0.0475481, mare_max=0.0482992, max_step_max=0.125178, stats=6 | 门槛待标定（TPU 执行噪声待标定（README 第五节）），只报被测值 |
| R17 | pass | `R17.health`（pass） | fields=3, problems=0 | 健康 |
| R18 | pass | `R18.dtypes`（pass） | entries=182, wrong_dtype=0, unmatched=0, rule_hits={params/*: 58, MoEBiasVar/*: 2, */opt_state/*/mu/*: 58, */opt_state/*/nu/*: 58, */opt_state/*/count: 4, OptState/optimizer/step: 2} | 182 项 dtype 全部符合规则 |
| R20 | pass | `R20.artifact_counts`（pass） | rules=7, bad=0 | 全部规则满足 |
| R21 | report | `R21.hlo_fingerprint`（fail，只报） | rules=0, bad=0, fingerprint=c09411cf2b169834, ref_fingerprint=209341bb971a111d, fingerprint_equal=False | 不满足：['指纹不同'] |
| R22 | report | （regress.py） | test_median_s=0.865896, lkg_median_s=0.898192, rel_diff=-0.0359562, steps_used=998, skip_first=2 | 只报。阈值待标定（标准 R22：同一提交 ≥ 5 次运行的步时极差）；峰值 HBM、编译期临时区还没记 |
| R23 | pass | （regress.py） | rows=21, problems=0 | 每一行都出了结论、单元数对得上 |
| R25 | inconclusive | `R25.loss_vs_golden`（fail，只报） | d0_max=2.91116e-06, signed_mean_max=0.000167155, mare_max=0.00018099, max_step_max=0.00071248, goldens=3 | 超过噪声参照两倍；数值变了：「不变差」要 TPU 执行噪声当尺子，还没标定或已作废，只报新值与 LKG 值的差（标准第三节） |

数字全部来自 janus 的 `results.json` 与运行目录（`regression.json` 是同一份数据），本报告不手写数字。
