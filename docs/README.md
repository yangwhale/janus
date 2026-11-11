[janus](../README.md) › 文档

# 文档

**第一次来，读[用户指南](guide/README.md)。** 它是主线：8 章，按上手顺序写，每章末尾有「下一步」。其余文档都是查用的。

想知道每个指标怎么定义、尺子怎么造、r 怎么算出来，读专题[《尺子》](guide/ruler.md)。

| 文件夹 | 是什么 | 什么时候读 |
|---|---|---|
| [`guide/`](guide/README.md) | 用户指南：准备环境 → GPU golden → TPU 准备 → 逐级对齐 → 读结果 → 排查 → 日常回归 → 接新模型 | 从头读 |
| [`glossary.md`](glossary.md) | 术语表：golden 代号、噪声参照、LKG、calibrated…… | 碰到不熟的词 |
| [`reference/`](#reference参考) | 参考：判法的完整论证、指标定义、每种检查的规格、文件格式、命令行 | 想知道「为什么这样判」或写目录、写工具时 |
| [`dev/`](dev/README.md) | 给改 janus 本身的人：架构、命名约定、仓库布局、设计文档、路径变更记录 | 只用 janus 的读者不用看 |

## reference/：参考

**为什么这样做**

| 文件 | 讲什么 |
|---|---|
| [`principles.md`](reference/principles.md) | 为什么以 GPU 为参照、FP64 用来干什么、差从哪来、什么时候能签字 |
| [`levels.md`](reference/levels.md) | L0–L4 每级做什么、为什么这样设计、跨平台怎么判、准出、不过时先查什么、用哪些检查类型 |
| [`judging.md`](reference/judging.md) | 指标与判定：哪一级用哪把尺子、每个指标管什么、loss 的四个数、噪声参照、1 倍 / 2 倍、两档门槛 |
| [`precision-boundaries.md`](reference/precision-boundaries.md) | 精度边界：对不齐几乎都出在 16 位与 32 位之间；每类边界的实例、开关，十分钟看代码的做法 |
| [`tpu-vs-gpu-differences.md`](reference/tpu-vs-gpu-differences.md) | TPU 与 GPU 训练数值哪里不一样：逐条清单（差异、原理、开关、修前修后、生产建议、还没查清的） |
| [`environment.md`](reference/environment.md) | 两边要钉死的版本与开关、GPU 侧踩到的坑、TPU 侧原则 |
| [`dump.md`](reference/dump.md) | 抓什么、怎么抓、怎么证明抓数不改数值 |
| [`regression.md`](reference/regression.md) | 日常回归验收标准：清单 R01–R28、TPU 自噪声标定、档位、LKG |

**规格**（写目录、写工具时查）

| 文件 | 讲什么 |
|---|---|
| [`metrics.md`](reference/metrics.md) | 每个指标怎么算、噪声上限怎么从参照对算出、边界情况、混元 3 实测量级 |
| [`checks/`](reference/checks/README.md) | 每种检查类型一份规格：输入、判据、输出、边界 |
| [`catalog-schema.md`](reference/catalog-schema.md) | 检查项目录（`catalogs/*.yaml`）的写法 |
| [`result-schema.md`](reference/result-schema.md) | 结果格式（`results.json`）、状态判法、退出码 |
| [`dump-format.md`](reference/dump-format.md) | 抓数格式：运行目录、`meta.json`、逐步标量、张量文件 |
| [`cli.md`](reference/cli.md) | `janus` 的子命令、参数、退出码 |

混元 3 的材料（对齐报告、检查清单、golden 出处、配置对照、各实践结果）不在这里，在 [`models/hunyuan3/`](../models/hunyuan3/README.md)。混元 4（主线 hy4-mini 的现状、现行比较、判定口径、根源一览）在 [`models/hunyuan4/`](../models/hunyuan4/README.md)。
