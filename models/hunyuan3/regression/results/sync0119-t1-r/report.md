```
[T1 · v5p · 2026-11-19 11:54] PASS
窗口：LKG sync0119-t1-a@7e49199ee9 → sync0119-t1-b@7e49199ee9（声明：数学不变）
依赖与开关：无变化（R04、R05 逐位）      对照臂：本次没跑（手动阶段）
第一个不同：无
各行：pass 19 / fail 0 / inconclusive 0 / error 0      只报：R21, R22
下一步：无。数值与 LKG 逐位相同，LKG 指针可前移到被测提交（数值没变，不用批准）
```

- 档位配置 `models/hunyuan3/regression/tiers/t1.yaml`（sha256 2f9c5908788e…）；目录 `regression-t1.yaml`（内容指纹 8b3eeed25a02…）；janus 0.1.0.dev0
- 被测提交：{maxtext: 7e49199ee99b5d84cffc8192006ed95301e3c11c, janus: b4e84eaa3862cb754efbbf43847477a58f6ef6f2}；LKG 提交：{maxtext: 7e49199ee99b5d84cffc8192006ed95301e3c11c, janus: b4e84eaa3862cb754efbbf43847477a58f6ef6f2}
- LKG 候选：是

| 行 | 状态 | 检查 | 关键数 | 说明 |
|---|---|---|---|---|
| R01 | pass | `R01.run_complete`（pass） | planned=1000, logged=1000, duplicate=0, missing=0, extra=0 | 1000 步全部完成、无重复、无重启 |
| R02 | pass | `R02.inputs`（pass） | fields=7, must_equal_differ=0, other_differ=0, missing=0, equals_rules=3 | 必须相同的字段都相同；其他不同 0 项 |
|  |  | `R02.registry`（pass） | tier_config_sha256=2f9c5908788e, catalog_files=2, catalog_hash=content-v1 | 档位配置的 sha256、目录的内容指纹与 lkg.yaml 登记的相同 |
| R03 | pass | `R03.data_fingerprint`（pass） | units_per_field={tokens: 16000, labels: 16000, loss_mask: 16000, position_ids: 16000}, units=65000, differ=0 | token 与首步 loss 逐位相同 |
| R04 | pass | `R04.versions`（pass） | fields=7, must_equal_differ=0, other_differ=0, missing=0, equals_rules=0 | 必须相同的字段都相同；其他不同 0 项 |
| R05 | pass | `R05.process_flags`（pass） | fields=6, must_equal_differ=0, other_differ=0, missing=0, equals_rules=0 | 必须相同的字段都相同；其他不同 0 项 |
|  |  | `R05.compiler_flagfile`（pass） | rules=0, bad=0, fingerprint=6f47d9db7a9f642d, ref_fingerprint=6f47d9db7a9f642d, fingerprint_equal=True | 全部规则满足，指纹与参照相同 |
|  |  | `R05.compiler_debug_options`（pass） | rules=0, bad=0, fingerprint=59407bf7dd627e57, ref_fingerprint=59407bf7dd627e57, fingerprint_equal=True | 全部规则满足，指纹与参照相同 |
| R06 | pass | `R06.must_equal`（pass） | fields=29, must_equal_differ=0, other_differ=0, missing=0, equals_rules=29 | 必须相同的字段都相同；其他不同 0 项 |
| R07 | pass | `R07.state_init`（pass） | params=91, differ=0, only_one_side=0 | 全部参数逐位相同 |
| R08 | pass | `R08.step1_loss`（pass） | fields=2, first_differ=[], tensors=0, tensors_differ=0 | 逐位可复现 |
|  |  | `R08.loss_tolerance`（pass，只报） | **0.00 倍尺子**, d0=0, signed_mean=0, mare=0, max_step=0, max_step_at=2 | 在噪声参照内 |
| R09 | pass | `R09.step1_norms`（pass） | fields=2, first_differ=[], tensors=0, tensors_differ=0 | 逐位可复现 |
|  |  | `R09.step1_grad_norm_tolerance`（pass，只报） | **0.00 倍尺子**, d0=0, signed_mean=0, mare=0, max_step=0, max_step_at=None | 在噪声参照内 |
| R12 | pass | `R12.loss`（pass） | d0=0, signed_mean=0, mare=0, max_step=0, max_step_at=2 |  |
|  |  | `R12.loss_tolerance`（pass，只报） | **0.00 倍尺子**, d0=0, signed_mean=0, mare=0, max_step=0, max_step_at=2 | 在噪声参照内 |
| R13 | pass | `R13.raw_grad_norm`（pass） | d0=0, signed_mean=0, mare=0, max_step=0, max_step_at=2 |  |
|  |  | `R13.param_norm`（pass） | d0=0, signed_mean=0, mare=0, max_step=0, max_step_at=2 |  |
|  |  | `R13.raw_grad_norm_tolerance`（pass，只报） | **0.00 倍尺子**, d0=0, signed_mean=0, mare=0, max_step=0, max_step_at=2 | 在噪声参照内 |
|  |  | `R13.param_norm_tolerance`（pass，只报） | **0.00 倍尺子**, d0=0, signed_mean=0, mare=0, max_step=0, max_step_at=2 | 在噪声参照内 |
| R14 | pass | `R14.lr_wd`（pass） | fields=2, first_differ=[], tensors=0, tensors_differ=0 | 逐位可复现 |
| R15 | pass | `R15.state_final`（pass） | params=91, differ=0, only_one_side=0 | 全部参数逐位相同 |
| R16 | pass | `R16.bias_stats`（pass） | signed_mean_max=0, mare_max=0, max_step_max=0, stats=6 |  |
|  |  | `R16.bias_stats_tolerance`（pass，只报） | **0.00 倍尺子**, signed_mean_max=0, mare_max=0, max_step_max=0, stats=6 | 在噪声参照内 |
| R17 | pass | `R17.health`（pass） | fields=3, problems=0 | 健康 |
| R18 | pass | `R18.dtypes`（pass） | entries=182, wrong_dtype=0, unmatched=0, rule_hits={params/*: 58, MoEBiasVar/*: 2, */opt_state/*/mu/*: 58, */opt_state/*/nu/*: 58, */opt_state/*/count: 4, OptState/optimizer/step: 2} | 182 项 dtype 全部符合规则 |
| R20 | pass | `R20.artifact_counts`（pass） | rules=7, bad=0 | 全部规则满足 |
| R21 | report | `R21.hlo_fingerprint`（pass，只报） | rules=0, bad=0, fingerprint=89e42526bc0fe49a, ref_fingerprint=89e42526bc0fe49a, fingerprint_equal=True | 全部规则满足，指纹与参照相同 |
| R22 | report | （regress.py） | test_median_s=0.878729, lkg_median_s=0.878328, rel_diff=0.000457115, steps_used=998, skip_first=2 | 只报。阈值待标定（标准 R22：同一提交 ≥ 5 次运行的步时极差）；峰值 HBM、编译期临时区还没记 |
| R23 | pass | （regress.py） | rows=21, problems=0 | 每一行都出了结论、单元数对得上 |
| R25 | pass | `R25.loss_vs_golden`（pass） | **0.77 倍尺子**, d0_max=1.86314e-05, signed_mean_max=1.90236e-05, mare_max=0.000105573, max_step_max=0.000544143, goldens=10 | 在噪声参照内；回归门逐位通过：数值与 LKG 相同，对齐门结论继承 LKG（标准第二节） |

数字全部来自 janus 的 `results.json` 与运行目录（`regression.json` 是同一份数据），本报告不手写数字。
