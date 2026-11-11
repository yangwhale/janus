# janus

> 这是 janus 的公开版，由开发仓库定期同步生成。文中提到、但没有公开的东西（我们修改过的 MaxText、开发历史、问题跟踪记录）见[关于公开版](docs/public-edition.md)。许可：Apache License 2.0，见 [`LICENSE`](LICENSE)。

GPU（Megatron）与 TPU（MaxText）训练的数值对齐工具。它回答一个问题：

> **TPU 和 GPU 的差，是不是不大于 GPU 自己重跑一遍的差？**

例子与全部证据来自混元 3（截 4 层，B200 × 8 → TPU v5p-8 / v7x-4）。本仓库的 v5p-8 指 8 颗芯片（拓扑 2x2x2），云上型号名按 TensorCore 计数叫 v5p-16。

## 结论（截至 2026-11-06）

- L4（1000 步训练）在 v5p 和 v7x 上都对齐了：对最终 golden [`rcc`](docs/glossary.md#rcc)（GPU 生产程序只把输出头 dgrad 改成分段 FP32 累加）的每一项都在 GPU 自己重跑的噪声内。判定量 r = TPU 对参照的差 ÷ GPU 重跑的噪声上限，最差 0.85（≤ 1 判过；[对齐报告](models/hunyuan3/ALIGNMENT-REPORT.md)第 3 节）。
- 参照只在 GPU 一侧修了一处：B200 上输出头 dgrad 沿 12 万词表的长求和有系统偏差，`rcc` 修的就是它。TPU 这一步没有算错（第 2 节）。
- 第 1 步的梯度与优化器状态有超出噪声的地方，按签字原则这一项不满足，是否接受由使用方决定；原因都已查清，且不累积：大头是近似平局的路由在两种硬件上各自选到不同的专家；另一处是 TPU 让一维 norm 权重、输出层与 embedding 的梯度先以 BF16 落地，三个前向不变的可选开关（步时 v5p +0.8%、v7x +1.6%）能消掉它，但打开后 v7x 的 L4 grad_norm 待定（1.05 倍），所以不在当前对齐配置里（第 4、6 节）。专家参数在第 1 步只比了每层 192 个里的 2 个。
- 适用范围：混元 3 截 4 层、1000 步、常数学习率、BF16、TPU 主配置 FSDP 8；80 层生产形状没有验证。范围写在对齐报告开头，全部未覆盖项列在检查清单「[已知的覆盖缺口](models/hunyuan3/CHECKS.md#已知的覆盖缺口)」。
- 每一项检查比了什么、怎么判、两平台的结果，逐项列在[检查清单](models/hunyuan3/CHECKS.md)。

> **看代码找对不齐，先读[精度边界](docs/reference/precision-boundaries.md)。** 到目前为止，两边对不齐几乎都出在一个值从 16 位进 32 位、或从 32 位落回 16 位的地方，再加上 GPU 上悄悄打开的 TF32。那一页列出每一类边界在本仓库里的实例与开关，以及十分钟看代码的做法。

## 从哪进

| 你想 | 读这个 |
|---|---|
| 知道结论 | [对齐报告 `ALIGNMENT-REPORT.md`](models/hunyuan3/ALIGNMENT-REPORT.md)：结论、偏差台账、开关表，一份读完 |
| 知道比了哪些量 | [检查清单 `CHECKS.md`](models/hunyuan3/CHECKS.md)：每一级比的模块与张量、指标与门槛、v5p / v7x 的结果，一页列全 |
| 知道每个指标怎么定义、尺子怎么造 | [专题《尺子》](docs/guide/ruler.md)：逐步曲线的 dₜ、窗口、三个统计量、噪声上限 N 与 r；第 1 步逐张量的相对 L2 与对 FP64 定责；带完整算例和速查表 |
| 看代码找对不齐 | [精度边界 `precision-boundaries.md`](docs/reference/precision-boundaries.md)：每个算子两边的输入、乘法、累加、输出 dtype 与舍入位置逐格比；边界类型清单与开关 |
| 知道 TPU 与 GPU 到底哪里不一样、怎么改 | [逐条清单 `tpu-vs-gpu-differences.md`](docs/reference/tpu-vs-gpu-differences.md)：每条差异的现象、原理、开关、修前修后与生产建议，还没查清的单列 |
| 自己做一遍 | [用户指南](docs/guide/README.md)：8 章，从准备环境到日常回归，按顺序读、按顺序做 |
| 查证据 | [混元 3 模型目录](models/hunyuan3/README.md)：golden 出处、配置对照、四个对齐实践的结果文件、[总览表](models/hunyuan3/dashboard/dashboard.md) |

名词不熟先查[术语表](docs/glossary.md)。

## 仓库地图

| 文件夹 | 放什么 |
|---|---|
| [`docs/`](docs/README.md) | 用户指南（主线）、查用的参考、术语表；`docs/dev/` 是给改 janus 本身的人看的 |
| `models/` | 每个模型一个文件夹：对齐报告、golden、配置对照、对齐实践的结果、两边的操作手册。[`hunyuan3/`](models/hunyuan3/README.md)（混元 3，已完成）、[`hunyuan4/`](models/hunyuan4/README.md)（混元 4，进行中）、[`glm52/`](models/glm52/README.md)（GLM-5.2，暂停）；归档说明在 [`archived/`](models/archived/README.md) |
| [`janus/`](janus/) | 工具本体：读检查项目录（YAML）、判定、出报告 |
| [`templates/`](templates/) | 新模型、新对齐实践照着填的模板 |
| [`tests/`](tests/README.md) | 单元测试与文档检查，只要 CPU |

根目录的 `AGENTS.md`、`CLAUDE.md` 是给自动化工具看的协作规矩，读者可以忽略。

## 五分钟试一下

只要 CPU 和 Python ≥ 3.10。

```bash
pip install -e '.[dev]'                       # 核心只依赖 numpy、ml_dtypes、pyyaml
janus demo /tmp/janus-demo                    # 合成 6 份参照运行 + 2 份被测
janus run /tmp/janus-demo/catalog.yaml --out /tmp/janus-demo/out-bad --test t-bad   # 模型第 1 层与 loss 不过，第 0 层过
```

然后打开 `/tmp/janus-demo/out-bad/report.md` 看报告。demo 造了什么、为什么这样判：[`docs/reference/cli.md`](docs/reference/cli.md#janus-demo)。

## 其他

- 问题与建议在本仓库开 issue，参与方式见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。文中的 `maxtext` 指我们修改过、尚未公开的 MaxText，见[关于公开版](docs/public-edition.md)。
- 版本：`janus --version` 与 [`CHANGELOG.md`](CHANGELOG.md)。改 janus 本身：[`CONTRIBUTING.md`](CONTRIBUTING.md)。数据与机密：[`SECURITY.md`](SECURITY.md)。许可：Apache License 2.0，见 [`LICENSE`](LICENSE) 与 [`NOTICE`](NOTICE)。
