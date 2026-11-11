[janus](../../README.md) › [用户指南](README.md) › 第 8 章

[← 上一章：第 7 章 日常回归](07-daily-regression.md) ｜ [回目录](README.md) ｜ [附录 →](appendix.md)

# 第 8 章 · 接入新模型

本章说明换一个模型（如混元 4）时，哪些原样复用，哪些每个模型都要重做，以及按什么顺序做。例子都用混元 3。目录布局见 [`layout.md`](../dev/layout.md)。

| 这一步 | |
|---|---|
| 做什么 | 分清复用和重做，按 13 步建出新模型的 `models/<模型>/` 与第一个对齐实践 |
| 怎么做 | 照第三节的步骤表；结构差异按第四节清单加检查或改代码 |
| 做完看到什么算对 | 第三节每步「怎么算做完」一栏都满足（要点见下） |
| 下一步 | 新模型照第 1–7 章同样的顺序走一遍 |

「做完看到什么算对」的要点：

- 转换后逐张量逐位。
- GPU 自比的噪声参照 ≥ 3 对（≥ 3 份运行）。
- `dashboard/build.py` 生成总览。
- golden 先查有没有 GPU 一侧的系统偏差；有就像混元 3 的 [`rcc`](../glossary.md#rcc)（最终 golden）那样只修那一处。

## 本章做完手里有什么

一张接新模型的清单：

- 哪些原样复用（第一节）；
- 哪些每个模型都要重做（第二节）；
- 按什么顺序做（第三节 13 步，每步写了产出和「怎么算做完」）；
- 哪些结构差异要加检查或改代码（第四节）。

照第三节做完，新模型就有了自己的 `models/<模型>/` 与第一个对齐实践，之后按第 1–7 章同样的顺序走。

## 一、原样复用（不用改）

| 东西 | 在哪 | 说明 |
|---|---|---|
| 工具代码 | `janus/` | 全部原样复用，有两处例外，见下面的「janus 里和模型有关的两处」 |
| 指标与判法 | [方法详解：指标与判定](../reference/judging.md)、[`reference/metrics.md`](../reference/metrics.md) | 噪声参照、1 倍 / 2 倍判定、loss 判定量 |
| 每级的定义 | [`levels.md`](../reference/levels.md) | L0–L4 查什么、准出标准 |
| 检查项目录格式 | [`reference/catalog-schema.md`](../reference/catalog-schema.md) | 目录内容要按模型重写（见下），格式不变 |
| 抓数格式 | [`reference/dump-format.md`](../reference/dump-format.md) | 运行目录、`meta.json`、张量索引、规范名 |
| 结果格式 | [`reference/result-schema.md`](../reference/result-schema.md) | `results.json`、状态、退出码 |
| 模板 | `templates/` | 报告、配置对照表、实践文件夹的 `setup.md` / `runs.yaml` |
| 分析工具 | `janus/analysis/` | 原样复用；模型特有的只有两份数据（见下） |
| 总览 | [`models/hunyuan3/dashboard/build.py`](../../models/hunyuan3/dashboard/build.py) | 复制后只改写死的实践名、运行名 |

表里各行的细节：

- 工具代码包括：
  - `janus run`、全部检查类型、指标、FP64 重放；
  - 逐级大表（[`janus/report/tables.py`](../../janus/report/tables.py)）；
  - 配图（[`janus/report/figures.py`](../../janus/report/figures.py)）。
- 指标与判法：
  - [噪声参照](../glossary.md#noise-reference)、1 倍 / 2 倍判定；
  - loss 的三个判定量，外加只报不判的首步差。
- 每级的定义：每种检查类型的规格在 `reference/checks/`。
- 分析工具：
  - 命令：`janus curves` / `janus offsets` / `janus drift` / `janus fault-matrix`。
  - 运行分组：照 [`models/hunyuan3/analysis/phase1-curves.yaml`](../../models/hunyuan3/analysis/phase1-curves.yaml) 写。
  - 故障期望：照 `phase1-faults.yaml` 写。
- 总览：复制到新模型目录后，只改里面写死的 `LEGACY_FIGS`、`SERIES_COMPARE`。

### janus 里和模型有关的两处

- [`janus/dump/maxtext.py`](../../janus/dump/maxtext.py) 的 `install_capture()` 只给列出来的 MaxText 层类套壳，现在是 `hunyuan3.Hunyuan3DenseLayer`、`hunyuan3.Hunyuan3MoELayer` 和 `qwen3.AttentionWithNorm` 等。新模型的层类不同时，要把它的层类加进去，否则 TPU 侧抓不到逐层激活。
- [`janus/dump/names.py`](../../janus/dump/names.py) 的 `_RULES` 把 Megatron 模块名映射到规范名，按 GPT 结构写（`self_attention.linear_qkv`、`mlp.router`、`mlp.experts`…）。新模型的模块名不同时要补规则。MLA 就是这种情况，它有 q / kv 的降维投影。

这两处改动和新模型同一个 PR 提交，并补单测：

- [`tests/dump/test_names.py`](../../tests/dump/test_names.py)
- [`tests/dump/test_maxtext.py`](../../tests/dump/test_maxtext.py)

## 二、每个模型都要重做

| 东西 | 混元 3 的样子 | 说明 |
|---|---|---|
| 名字与布局映射 | [`models/hunyuan3/model/name_map.yaml`](../../models/hunyuan3/model/name_map.yaml) | HF ↔ Megatron ↔ MaxText ↔ 规范名（见下） |
| 截层与权重转换 | `model/` 下 5 个脚本（见下） | 截层要同时截配置、去 MTP、写逐张量 sha256 清单 |
| 框架训练与抓数脚本 | `frameworks/` 下训练脚本、配置、`run.sh`（见下） | 结构相近时大部分能照抄；结构不同的部分要改，见第四节清单 |
| 配置对照表 | `config-alignment.md` | 照模板填；每个值都要有出处（见下） |
| GPU golden | 起步 `g1k`（第一阶段 GPU golden），最终 `rcc`（GPU 生产程序只修一处），见下 | 至少 3 份同设置；签字前建议 6 份以上 |
| 噪声参照 | 第一阶段 GPU 自比，`alignments/00-gpu-noise-and-drills/` | 噪声大小和模型、数据、并行有关，不能沿用混元 3 的数 |
| 检查项目录 | `alignments/*/catalogs/*.yaml` | 层号、张量名、门槛的出处都和模型有关 |
| 总览清单 | `dashboard/items.yaml` | 每行一项检查，每列一个实践 |
| 日常回归实例 | `regression/README.md` | 投产后才需要（见下） |

### 各行的细节

- 名字与布局映射：含转置、拆分、拼接、专家分组。解释器 [`janus/dump/name_map.py`](../../janus/dump/name_map.py) 复用。
- 截层与权重转换的 5 个脚本：
  - `model/truncate_hy3.py`
  - `model/import_hy3.py`
  - `model/export_hy3.py`
  - `model/convert_hf_to_maxtext.py`
  - `model/verify_ckpt.py`
- 框架训练与抓数脚本：
  - `frameworks/megatron/train_hy3.py`
  - `frameworks/maxtext/train_hy3.py`
  - `hy3-4layers.yml`
  - `run.sh`
- 配置对照表：照 [`../../templates/config-alignment.md`](../../templates/config-alignment.md) 填。每个值都要有出处，按运行时实际生效的值填。
- GPU golden（代号怎么读见[术语表](../glossary.md#golden-codenames)）：
  - 起步：[`g1k`](../glossary.md#g1k)（第一阶段 GPU golden）的 `g1k-a` / `g1k-b` / `g1k-c`。默认设置，1000 步，第 1 步全量抓数、第 1000 步只抓前向。
  - 最终：生产配置的 golden，查出系统偏差时只修那一处。混元 3 是 [`rcc`](../glossary.md#rcc) 的 `rcc-a..j` 10 份，[最终 golden](../glossary.md#final-golden)，见[第 2 章](02-gpu-golden.md)。
  - [精确轨迹](../glossary.md#exact-trajectory)用来当裁判查偏差。
  - 份数：至少 3 份同设置的 golden 才凑得出 3 对噪声参照；签字前建议 6 份以上（15 对）。
- 日常回归实例：清单与判法照 [`reference/regression.md`](../reference/regression.md)，按模型填固定输入、平台档案、各行的名字与个数。TPU 执行噪声按平台重新标定。

## 三、按步骤走（每步写清产出的文件）

| 步 | 做什么 | 产出（混元 3 的对应文件） | 怎么算做完 |
|---|---|---|---|
| 1 | 建模型目录，写概况 | `models/<模型>/README.md` | — |
| 2 | 截层 | `model/truncate_<模型>.py`，产出 HF 目录和 `tensor_sha256.json` | 张量数、清单 sha256 写进 README，作为锚点 |
| 3 | 写名字映射 | `model/name_map.yaml` | 覆盖全部张量、形状全对；单测通过 |
| 4 | GPU 侧：HF → Megatron，训练脚本，抓数 | `model/import_<模型>.py`、`frameworks/megatron/train_<模型>.py`、`run.sh` | 参数数对得上 HF；跑通一次 `--full` 抓数 |
| 5 | GPU 自比实践 | `alignments/00-gpu-noise-and-drills/`：目录、结果、`faults/` | 确定性逐位；埋的错都被抓到；噪声参照 ≥ 3 对 |
| 6 | 填配置对照表（GPU 一栏先填） | `config-alignment.md` | 每个值有出处；「待查」逐条列出 |
| 7 | GPU golden | golden 运行目录（桶里），`results/series/<运行>.metrics.jsonl` | `meta.json` 记全实际生效的配置 |
| 8 | TPU 侧：转换、配置对照、训练与抓数 | `model/convert_hf_to_maxtext.py`、`frameworks/maxtext/<模型>.yml`、`train_<模型>.py` | 转换后逐张量逐位；CPU 冒烟跑通 |
| 9 | 开对齐实践：设置、运行清单、目录 | `alignments/01-<目标平台>-acceptance/` 的 `setup.md`、`runs.yaml`、`catalogs/` | — |
| 10 | 按 L0 到 L4 跑；总览清单加一列 | `results/*.results.json`、`dashboard/items.yaml` | `dashboard/build.py` 生成总览和逐级大表 |
| 11 | TPU 扰动参照、TPU 侧故障演练 | `tpu-perturb-*` 运行、`faults/` | 见下 |
| 12 | 写结论 | 实践 README 表格下方 | 数字只引用表里的行或结果文件 |
| 13 | 投产后：建日常回归 | `models/<模型>/regression/` | 每档首个 [LKG](../glossary.md#lkg) 双跑逐位；TPU 执行噪声 ≥ 15 对 |

### 各步的细节

- 第 1 步：概况写代码基准、截几层、为什么。
- 第 2 步：配置改层数、去 MTP；只拷需要的张量；写逐张量 sha256 清单。
- 第 3 步「怎么算做完」：`verify_ckpt.py names` 覆盖全部张量、形状全对；[`tests/dump/test_name_map.py`](../../tests/dump/test_name_map.py) 照抄改名后通过。
- 第 5 步做的是确定性自证、抓数不改数值、噪声参照、故障演练。做完的标准是确定性运行逐位，每种埋的错都被期望的检查抓到，噪声参照 ≥ 3 对。
- 第 7 步：至少 3 份同设置的长跑，第 1 步和最后一步全量抓数。
- 第 8 步做的是 HF → MaxText 转换与核对、MaxText 配置逐项对照、训练与抓数。逐张量逐位的例子是混元 3 的 01 `phase2-ckpt-convert`。
- 第 10 步：结果放进 `results/`。
- 第 11 步：混元 3 的 TPU 执行噪声见 [`regression/README.md`](../../models/hunyuan3/regression/README.md)「五、TPU 自噪声标定」；TPU 侧故障演练只做了 NaN（01 [`README.md`](../../models/hunyuan3/alignments/01-v5p-acceptance/README.md)）。
- 第 13 步：混元 3 的对应文件是 [`regression/README.md`](../../models/hunyuan3/regression/README.md)。

## 四、结构差异清单：哪些差异会要求加检查或改代码

接新模型前，逐项对照混元 3 看。有差异的项，在配置对照表里单列，并按下面「要做什么」处理。

### 路由类型

- 包括：softmax / sigmoid、分组路由、top-k 归一、缩放、无辅助损失的偏置。
- 混元 3：sigmoid + expert_bias（更新率 0.001），top-k 归一后乘 2.826。
- 会影响到：路由 GEMM 写法、翻转率、偏置更新。
- 要做什么：
  - L1 路由回放与 `discrete_agreement`；
  - 偏置的 dtype 进 `dtype_manifest`；
  - L3 比更新后偏置；
  - 两边的归一与缩放写法逐行核对。

### 注意力：MLA 对 GQA / MHA

- 混元 3：GQA（64 头、8 个 KV 头、每头 128）。
- 会影响到：投影个数与形状、qkv 拆分、rope 维。
- 要做什么：
  - 改 `name_map.yaml` 与 [`janus/dump/names.py`](../../janus/dump/names.py) 的规则。
  - 规则要覆盖 q / kv 降维投影、`kv_lora_rank`、rope 维单独一段。
  - 不要套「hidden = 头数 × 每头维度」。

### 共享专家

- 混元 3：1 个，FFN 维 1536。
- 会影响到：每层多一条 FFN。
- 要做什么：抓数要单独抓共享专家输出（`mlp.shared.out`）。

### 前几层稠密

- 混元 3：第 0 层稠密。
- 会影响到：层类不同。
- 要做什么：
  - `install_capture()` 两种层类都要套壳；
  - 目录里按层区分稠密层和 MoE 层。

### MTP

- 混元 3：截层时关掉。
- 会影响到：多一个预测头和 loss。
- 要做什么：
  - 对齐先关。
  - 要开就另开实践文件夹，并加 MTP loss 的 L4 检查。

### 归一化

- 包括：RMSNorm 以外的归一化、qk 归一、前后归一位置。
- 混元 3：RMSNorm（eps 1e-5）+ qk 归一。
- 会影响到：归一化的舍入。第一阶段发现 RMSNorm 溢出时静默输出 0。
- 要做什么：
  - L1 归一化回放；
  - 新的归一化先做 FP64 参照。

### RoPE 变体

- 包括：theta、YaRN、部分维度旋转。
- 混元 3：default，theta 11158840。
- 会影响到：位置编码。
- 要做什么：
  - L2 比 q / k；
  - 配置对照表写清缩放参数。

### 词表与 embedding 是否共享

- 混元 3：不共享，词表 120832。
- 会影响到：输出层权重、embedding 梯度。
- 要做什么：共享时 embedding 梯度来自两处，L3 梯度检查要拆开看。

### 精度路径

- 包括：FP8、FP32 路由、FP32 lm_head、FP32 合并。
- 混元 3：计算 BF16；路由 FP32 出；HF `enable_lm_head_fp32: true`。
- 会影响到：每个 FP32 / FP8 段都是一个舍入点。
- 要做什么：
  - 配置对照表逐段列；
  - FP8 另开实践文件夹。

### 专家执行

- 包括：丢不丢 token、分组 GEMM 实现、EP 方式。
- 混元 3：不丢 token；GPU alltoall、TPU megablox。
- 会影响到：专家输出缓冲区不能跨运行逐位比（00 README 第 3 条）。
- 要做什么：比 token 空间的 `mlp.out`，不比分发后的缓冲区。

---

下一步：新模型照第 1–7 章同样的顺序走一遍。术语、指标定义、设计文档与仓库布局在附录。

[← 上一章：第 7 章 日常回归](07-daily-regression.md) ｜ [回目录](README.md) ｜ [附录 →](appendix.md)
