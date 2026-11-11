# janus 报告：hunyuan3-4l / phase2-acc-health-rcc

- 被测：`t-rmap-on-1000`；golden：`rcc-a`
- janus 0.1.0.dev0；目录 `models/hunyuan3/alignments/01-b200x8-to-v5p8-4layers/catalogs/phase2-acc-health-rcc.yaml`（sha256 2b6c29d00840…）
- 结果：pass 2、fail 0、inconclusive 0、error 0

| 检查 | 类型 | 级别 | 状态 | 关键指标 | 说明 |
|---|---|---|---|---|---|
| `L4.router.bias_std` | router_stats | L4 | pass | signed_mean_max=4.185e-04, mare_max=2.744e-03, max_step_max=1.299e-02, stats=3 | 在噪声参照内 |
| `L4.health` | health_signals | L4 | pass | fields=3, problems=0 | 健康 |

数字全部来自 `results.json`，本报告不手写数字。
