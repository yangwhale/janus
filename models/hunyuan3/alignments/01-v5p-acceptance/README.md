[janus](../../../../README.md) › [用户指南](../../../../docs/guide/README.md) › [混元 3](../../README.md) ｜ 对齐实践 01；怎么读见指南[第 5 章 读结果与签字](../../../../docs/guide/05-reading-results.md)

# 混元 3 · 对齐实践 01：B200×8 → TPU v5p-8（截 4 层）

这个实践是 TPU v5p 对 GPU golden 的迁移验收，从 L0 到 L4 逐级做。整个排查的主线也在这里：TPU loss 偏低万分之一点五，查下去发现 GPU 参照自己有偏（案例 01），换成最终 golden `rcc` 后 L4 全部在噪声内。

- 第二阶段：TPU（MaxText，v5p 8 片）对 GPU golden（B200×8，Megatron）。
- GPU 侧的噪声参照与 golden 来自 [`../00-gpu-noise-and-drills/`](../00-gpu-noise-and-drills/README.md)。
- 每一项检查在各实践上的状态见 [`../../dashboard/dashboard.md`](../../dashboard/dashboard.md)。

本页依次是：这个文件夹、golden 的分工、运行根目录、逐级大表（脚本生成）、结论、排查案例、已有结果。

## 这个文件夹

| 文件 / 文件夹 | 放什么 |
|---|---|
| [`setup.md`](setup.md) | 两边硬件、切分、镜像与提交号，和模型对照表不一样的地方 |
| [`runs.yaml`](runs.yaml) | 运行清单 |
| `catalogs/` | 检查项目录（见表下） |
| `results/` | 结果文件（平铺，文件名带检查项目录名） |
| `figures/` | 图 |
| `faults/` | TPU 侧故障演练（见表下） |

**`catalogs/` 里的检查项目录**

- 正式迁移验收第一层：对最终 golden 的基线 [`rcc-a`](../../../../docs/glossary.md#rcc-a)（GPU 生产程序加输出头 dgrad 分段 FP32 累加），噪声取 `rcc-a..j` 10 份重跑两两配对得到的 45 对比较。
  - `phase2-acceptance-rcc-on.yaml`：当前配置 `t-rmap-on-1000`（v5p 当前对齐配置 1000 步），总览取这份；
  - `phase2-acc-health-rcc.yaml`：路由偏置曲线与健康信号；
  - `phase2-acceptance-rcc-ep8.yaml`：并行方式改成 EP = 8。
- 第二层：`phase2-acceptance-rc.yaml`，对 GPU 生产 golden [`rc`](../../../../docs/glossary.md#rc) 的 `rc-a..rc-e`，差写成已解释差异。
- 其余：
  - `phase2-tpu.yaml`：L0–L4 草稿，对 `g1k-*`（第一阶段 GPU golden，不开重算，1000 步，3 份），归因用；
  - `phase2-ckpt-convert.yaml`：L0 权重转换逐张量；
  - `phase2-t-loss30.yaml`：30 步 loss，对 `g1k-*`。

**`faults/`**

TPU 侧只做了 NaN 演练（`results/phase2-t-nan8`）。其余几种埋错（权重缩放、梯度缩放、学习率差一个 ULP、数据错位一步、路由改 BF16、冻结 expert_bias）没在 TPU 上做，文件夹还没有建。检查项本身的抓错能力已在 GPU 侧第一阶段演练里证明，见 [`../00-gpu-noise-and-drills/faults/phase1-faults.md`](../00-gpu-noise-and-drills/faults/phase1-faults.md)。

TPU 侧怎么跑见 [`../../frameworks/maxtext/README.md`](../../frameworks/maxtext/README.md)。

## golden 的分工

2026-11-06 定。golden 是 GPU 生产训练程序跑出的结果，TPU 拿它当参照答案。

### 第一层（正式迁移验收）：对最终 golden `rcc`

[`rcc`](../../../../docs/glossary.md#rcc) 沿用 `03` 的 `rc` 配置（BF16 生产程序，整层重算加路由回放），其余不变，只加一处：输出头 dgrad 改成分段 FP32 累加（`rr_arms.py` 的 `dgrad-chunk`），修掉 B200 上这次长求和的系统偏差。

- 基线：TPU 只和事先指定的一份 [`rcc-a`](../../../../docs/glossary.md#rcc-a) 比。
- 噪声带：取 `rcc-a..j` 10 份同设置重跑、两两配对得到的 45 对比较。噪声带和基线用同一种设置，这是一般规矩（[`../../../../docs/reference/judging.md`](../../../../docs/reference/judging.md#golden-本身有系统偏差时修掉偏差再按一般规矩比)）。
- 结果：目录 [`catalogs/phase2-acceptance-rcc-on.yaml`](catalogs/phase2-acceptance-rcc-on.yaml) 里，当前配置 `t-rmap-on-1000`（v5p 当前对齐配置 1000 步）全部过，r 最差 0.85。过的项是 loss、loss 尾段、grad_norm、param_norm、学习率；路由偏置曲线与健康信号对 `rcc-a` 也过（`phase2-acc-health-rcc`）。

依据与全部数字见 [`../../ALIGNMENT-REPORT.md`](../../ALIGNMENT-REPORT.md)。

### 第二层：对 GPU 生产 golden `rc`

[`rc`](../../../../docs/glossary.md#rc) 是 GPU 生产 golden，即原版生产程序（整层重算加路由回放），共 `rc-a..e` 5 份。TPU 对它的差写成已解释差异，全部来自 GPU 输出头 dgrad 的长求和（[`../../config-alignment.md`](../../config-alignment.md)「输出头 dgrad」行）：

- 第 1000 步 param_norm +2.09e-2；
- 第 301–1000 步 loss 逐步相对差带符号均值 −1.37e-4，对 `rc-a..e` 同步均值（[`results/case01-tpu-vs-rc.json`](results/case01-tpu-vs-rc.json)）。

### 只作参考、只做归因

- 只作参考：精确轨迹 [`rce`](../../../../docs/glossary.md#rce)（全程 FP32、BF16 权重语义）与 [`rcec`](../../../../docs/glossary.md#rcec)（再加分段 dgrad，真精确），用来交叉核对。
- 只做归因：[`rcx`](../../../../docs/glossary.md#rcx)（只修 BF16 那一部分）与 `00` 的 [`g1k`](../../../../docs/glossary.md#g1k)`-*`（第一阶段 GPU golden）。

2026-11-05 晚到 11-06 早，第一层曾以 `rce-a` 为基线，噪声例外地取 `rcx-a..e` 5 份重跑两两配对的 10 对，称「方案 A」；`rcx` 是生产 golden 把输出头 dgrad 换成 FP32 一次累加。后来查清 `rce` 的输出头 dgrad 也带 cuBLAS FP32 长累加偏差，方案 A 随即撤销，不再使用（[案例 01](cases/01-loss-offset-param-norm.md)「TPU 对精确轨迹那一成的账」）。

### 并行方式对照（诊断臂，不计入判定）

- TPU 当前配置只把专家并行度改成和 GPU 相同的 EP = 8（词嵌入表沿 expert 轴切 8 份（读 HLO 核过），其余非专家参数按逻辑轴规则推断同样切，≈ FSDP 8，和 GPU 的 DP 8 不同，见[对齐报告注 11](../../ALIGNMENT-REPORT.md)），按第一层同一把尺子判（`phase2-acceptance-rcc-ep8`）：五项全过（r 最差 0.78）。
- v7x 同样的对照见 02 的 `phase2-v7x-acceptance-rcc-ep8`。
- FSDP 8 与 EP 8 两种切分对 `rcc-a`（最终 golden 的第 1 份，验收的比较基线）都在噪声内。
- EP = 8 在 v5p-8 上步时更短（1.607 → 0.999 s）。

## 运行根目录

- 检查项目录的 `runs.root` 一律写 `/data/janus-runs`：把目录里用到的运行目录都下载到这一个目录下再跑。
- 放在别处时不用改目录文件，用 `janus run --root <目录>` 换（`results.json` 里记的是实际路径）。
- L4 类检查只读 `meta.json`、`metrics.jsonl`、`DONE`，可以只拉这三个文件：

```bash
R=/data/janus-runs; B=gs://<桶>/janus/hunyuan3-4l
for n in golden/rcc-{a,b,c,d,e,f,g,h,i,j} golden/rc-{a,b,c,d,e} tpu/t-rmap-on-1000; do       # 换成目录里 runs 列的运行
  mkdir -p $R/$(basename $n); for f in meta.json metrics.jsonl DONE; do gcloud storage cp $B/$n/$f $R/$(basename $n)/; done
done
python -m janus run models/hunyuan3/alignments/01-v5p-acceptance/catalogs/phase2-acceptance-rcc-on.yaml --root $R --out /tmp/acc-rcc-on   # 第一层
python -m janus run models/hunyuan3/alignments/01-v5p-acceptance/catalogs/phase2-acceptance-rc.yaml --root $R --out /tmp/acc-rc     # 第二层
```

L0–L3 类检查要张量，得把整个运行目录拉下来（`gcloud storage cp -r`）。

<!-- janus:table:begin（以下由脚本生成，不要手改） -->

## 逐级大表

### 来源

- 由 `models/<模型>/dashboard/build.py` 调 `janus/report/tables.py` 生成。
- 数据来自本文件夹 `results/` 下的 `results.json`，数字不手写。

### 两档门槛

- 含义见 `docs/reference/judging.md`「两档门槛」。
- **严格档**是验收判定：
  - calibrated 检查对噪声上限 N 判：≤ N 过，> 2N 不过，中间待定；
  - 参照不足 3 对最多待定；
  - 逐位类必须逐位相同；
  - 数值门槛（如 FP64 R 比）照目录里写的线。
  - 符号：✅ 过 ｜ 🟡 待定 ｜ ❌ 不过或出错 ｜ ⬜ 还没跑 ｜ — 不适用。
- **参考档**是同类 GPU→TPU 迁移项目的实践门槛：
  - 只并列报告、不决定验收。
  - 符号：✅ 过 ｜ ☑️ 记录了的都过，但有门槛项没记录 ｜ ❌ 不过 ｜ ❔ 所需指标未记录 ｜ — 无对应门槛。

### 说明栏怎么读

- 格式一律是「实测（指标）｜严格档 门槛 → 离门槛多远，判定｜参考档 门槛：实测（实测 / 门槛的倍数），判定」。
- calibrated 检查先写主判指标 rel_L2：
  - 没有 rel_L2 就写最差的指标；
  - 最差的不是它时，补一句最差的那个。
- L4 逐步量三个指标各占一行：
  - 各自按同一规则判；
  - 整条检查取最差，见报告。
- 首步差：严格档只报不判，参考档要判（MARE＝平均绝对相对差）。
- 进行中的项在证据栏写证据位置。
- 严格档里「N=…（k 对）」的 k 是算 N 用的噪声参照对数：
  - 一对是两份运行的一次比较，k 对不是 k 份运行；
  - 用了哪几份运行、怎么配对见该检查项目录的 `runs.noise`。

### 逐行结果

- 共 127 行。
- 严格档：✅ 103　🟡 1　❌ 13　— 10
- 参考档（有对应门槛的 26 行）：✅ 18　❔ 8

| 级 | 检查项 | 严格档 | 参考档 | 说明 | 证据 |
|---|---|---|---|---|---|
| L0 | 配置逐项对照<br>`L0.config.diff.dropout` | ✅ | — | 实测 全部 6 个字段一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-t-loss30](results/phase2-t-loss30.report.md) |
| L0 | 每 rank 每步输入指纹（回放数据）<br>`L0.data.fingerprint` | ✅ | — | 实测 全部 1,950 个单元一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-t-loss30](results/phase2-t-loss30.report.md) |
| L0 | 加载后全部参数逐位<br>`L0.weights.init` | ✅ | — | 实测 全部 1,480 个参数一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-l0-state](results/phase2-l0-state.report.md) |
| L0 | 加载后全部参数逐位<br>`L0.weights.init.prod` | ✅ | — | 实测 全部 1,480 个参数一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-l0-state](results/phase2-l0-state.report.md) |
| L0 | 加载后全部参数逐位<br>`L0.weights.loaded` | ✅ | — | 实测 全部 50 个张量一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-l0-state](results/phase2-l0-state.report.md) |
| L0 | FP32 主权重逐位<br>`L0.main_params.loaded` | ✅ | — | 实测 全部 50 个张量一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-l0-state](results/phase2-l0-state.report.md) |
| L0 | 浮点 buffer（expert_bias）逐位<br>`L0.buffers.loaded` | ✅ | — | 实测 全部 3 个张量一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-l0-state](results/phase2-l0-state.report.md) |
| L0 | dtype 清单<br>`L0.dtype.manifest` | ✅ | — | 实测 全部 2,960 项一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-l0-state](results/phase2-l0-state.report.md) |
| L0 | dtype 清单<br>`L0.dtype.state` | ✅ | — | 实测 全部 182 项一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-l0-state](results/phase2-l0-state.report.md) |
| L0 | 模型结构（实例化结果）<br>`L0.model.structure` | ✅ | — | 实测 全部规则满足｜严格档 按规则，过｜参考档 无对应门槛 | [phase2-l0-state](results/phase2-l0-state.report.md) |
| L0 | 权重转换 → MaxText 逐张量逐位（同一份检查点给 v5p、v7x 用；含 dtype 清单）<br>`L0.ckpt.convert.bitwise` | ✅ | — | 实测 全部 1,781 个参数一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-ckpt-convert](results/phase2-ckpt-convert.report.md) |
| L0 | 权重转换 → MaxText 逐张量逐位（同一份检查点给 v5p、v7x 用；含 dtype 清单）<br>`L0.ckpt.convert.dtype` | ✅ | — | 实测 全部 1,781 项一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-ckpt-convert](results/phase2-ckpt-convert.report.md) |
| L0 | 续训数据位置逐位<br>`L0.resume.data` | ✅ | — | 实测 全部 650 个单元一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-resume](results/phase2-resume.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.gpu_integrity` | ✅ | — | 实测 全部 64,000 个数组一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.index` | ✅ | — | 实测 三个索引两边重建结果逐字节相同（文件 sha256 也相同）｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.stream` | ✅ | — | 实测 全部 1,000 步一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.device_map.native` | ❌ | — | 实测 原样布局只有 2000 / 16000 个（设备 · microbatch）单元与 GPU 同号 rank 相同：MaxText 梯度累积按…｜严格档 逐位 / 逐项相同，不过｜参考档 无对应门槛 | [phase2-native-loader](results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.fingerprint` | ✅ | — | 实测 全部 64,000 个单元一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.nonpad_total` | ✅ | — | 实测 全部 1,000 步一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.fields` | ✅ | — | 实测 字段约定一致：MaxText 用 int32、按全局批 [gbs, L] 存，GPU 用 int64 / float32、按 [mbs, L…｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.readers` | ✅ | — | 实测 grain_worker_count 0 / 8 与 1 逐步逐位相同｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.hosts` | ✅ | — | 实测 2 host：读进来的行按 host 交错（16000/16000 行符合 host 切分公式），用单 host 换序只对上 36000/6…｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.epochs` | ✅ | — | 实测 16000 个样本（1 个 epoch）：三个索引相同，8 个样本（含跨 epoch 边界的 0 个）逐位相同；356736 个样本（2 个…｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.eod_cases` | ✅ | — | 实测 合成数据（15 篇文档，含 1 token 空文档、恰好 seq / seq+1 长、跨两个样本的长文档，序列 16，2 个 epoch）：…｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.pad_id` | ❌ | — | 实测 含 pad id 的样本在 4 种组合下共 8 处两边不同（字段 ['labels', 'loss_mask', 'tokens']）：Me…｜严格档 逐位 / 逐项相同，不过｜参考档 无对应门槛 | [phase2-native-loader](results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.negative_controls` | ✅ | — | 实测 MaxText 默认的 reset_attention_mask 与 packing：68/80 个单元不同（position）；不设 mm…｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](results/phase2-native-loader.report.md) |
| L0 | TPU 原生数据加载器与 GPU 逐样本一致<br>`L0.native.coverage` | ✅ | — | 实测 全部比到｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-native-loader](results/phase2-native-loader.report.md) |
| L1 | 优化器单步 FP64 重放<br>`L1.optim.adamw.replay` | ✅ | ✅ | 实测 0.00213（rel_L2）｜严格档 ≤ 0.005 → 0.425×，过｜参考档 rel_L2（Frobenius） ≤ 0.02：0.00213（0.106×）；cos ≥ 0.9999：0.999999，过 | [phase2-st1-on-state-gpuonly](results/phase2-st1-on-state-gpuonly.report.md) |
| L1 | 优化器连续 8 步重放<br>`L1.optim.adamw.8steps` | ✅ | — | 实测 0.0048（rel_L2）｜严格档 ≤ 0.005 → 0.9593×，过｜参考档 无对应门槛 | [phase2-st8-opt8](results/phase2-st8-opt8.report.md) |
| L1 | 路由选中集合对 FP64<br>`L1.router.topk.fp64.layer1` | ✅ | — | 实测 0.9998（集合相同比例）｜严格档 ≥ 0.97 → 过｜参考档 无对应门槛 | [phase2-l2-replay-rratio-on](results/phase2-l2-replay-rratio-on.report.md) |
| L1 | 路由选中集合对 FP64<br>`L1.router.topk.fp64.layer2` | ✅ | — | 实测 0.9995（集合相同比例）｜严格档 ≥ 0.97 → 过｜参考档 无对应门槛 | [phase2-l2-replay-rratio-on](results/phase2-l2-replay-rratio-on.report.md) |
| L1 | 路由选中集合对 FP64<br>`L1.router.topk.fp64.layer3` | ✅ | — | 实测 1（集合相同比例）｜严格档 ≥ 0.97 → 过｜参考档 无对应门槛 | [phase2-l2-replay-rratio-on](results/phase2-l2-replay-rratio-on.report.md) |
| L1 | 单算子回放（路由 / 投影 / qkv / norm）与 CPU 复现舍入<br>`L1.ops.op.qkv` | ✅ | ✅ | 实测 1（R 比·各用自己的输入）｜严格档 ≤ 1.2 → 0.833×，过｜参考档 rel_L2 ≤ 2^-10：1.79e-04（0.183×）；归一化最大误差 ≤ 2^-7：0.005（0.64×），过（rel_L2 最大的是 `layers.0.attn.qkv.out`） | [phase2-l1-ops-own-on](results/phase2-l1-ops-own-on.report.md) |
| L1 | 单算子回放（路由 / 投影 / qkv / norm）与 CPU 复现舍入<br>`L1.ops.op.qk_norm` | ✅ | ❔ | 实测 1.073（R 比·各用自己的输入）｜严格档 ≤ 1.2 → 0.894×，过｜参考档 rel_L2 ≤ 2^-10、归一化最大误差 ≤ 2^-7：所需指标未记录（结果里没有同输入的被测对参照统计（rel_L2 是对 FP64 真值；两侧算子输入不同时不比）） | [phase2-l1-ops-own-on](results/phase2-l1-ops-own-on.report.md) |
| L1 | 单算子回放（路由 / 投影 / qkv / norm）与 CPU 复现舍入<br>`L1.ops.op.attn_proj` | ✅ | ❔ | 实测 1.001（R 比·各用自己的输入）｜严格档 ≤ 1.2 → 0.834×，过｜参考档 rel_L2 ≤ 2^-10、归一化最大误差 ≤ 2^-7：所需指标未记录（结果里没有同输入的被测对参照统计（rel_L2 是对 FP64 真值；两侧算子输入不同时不比）） | [phase2-l1-ops-own-on](results/phase2-l1-ops-own-on.report.md) |
| L1 | 单算子回放（路由 / 投影 / qkv / norm）与 CPU 复现舍入<br>`L1.ops.op.mlp_norm` | ✅ | ❔ | 实测 1.158（R 比·各用自己的输入）｜严格档 ≤ 1.2 → 0.9648×，过｜参考档 rel_L2 ≤ 2^-10、归一化最大误差 ≤ 2^-7：所需指标未记录（结果里没有同输入的被测对参照统计（rel_L2 是对 FP64 真值；两侧算子输入不同时不比）） | [phase2-l1-ops-own-on](results/phase2-l1-ops-own-on.report.md) |
| L1 | 单算子回放（路由 / 投影 / qkv / norm）与 CPU 复现舍入<br>`L1.ops.op.router_gemm` | ✅ | ❔ | 实测 0.0152（R 比·各用自己的输入）｜严格档 ≤ 1.2 → 0.0127×，过｜参考档 rel_L2 ≤ 2^-10、归一化最大误差 ≤ 2^-7：所需指标未记录（结果里没有同输入的被测对参照统计（rel_L2 是对 FP64 真值；两侧算子输入不同时不比）） | [phase2-l1-ops-own-on](results/phase2-l1-ops-own-on.report.md) |
| L1 | 单算子回放（路由 / 投影 / qkv / norm）与 CPU 复现舍入<br>`L1.ops.op.dense_fc1` | ✅ | ❔ | 实测 1.117（R 比·各用自己的输入）｜严格档 ≤ 1.2 → 0.9308×，过｜参考档 rel_L2 ≤ 2^-10、归一化最大误差 ≤ 2^-7：所需指标未记录（结果里没有同输入的被测对参照统计（rel_L2 是对 FP64 真值；两侧算子输入不同时不比）） | [phase2-l1-ops-own-on](results/phase2-l1-ops-own-on.report.md) |
| L1 | 单算子回放（路由 / 投影 / qkv / norm）与 CPU 复现舍入<br>`L1.ops.op.dense_act_fc2` | ✅ | ❔ | 实测 1.113（R 比·各用自己的输入）｜严格档 ≤ 1.2 → 0.9271×，过｜参考档 rel_L2 ≤ 2^-10、归一化最大误差 ≤ 2^-7：所需指标未记录（结果里没有同输入的被测对参照统计（rel_L2 是对 FP64 真值；两侧算子输入不同时不比）） | [phase2-l1-ops-own-on](results/phase2-l1-ops-own-on.report.md) |
| L1 | 专家 GEMM / 核心注意力回放、MoE 分阶段<br>`L1.ops.kernel.attn_core` | ✅ | ❔ | 实测 1.009（R 比·各用自己的输入）｜严格档 ≤ 1.2 → 0.841×，过｜参考档 rel_L2 ≤ 2^-10、归一化最大误差 ≤ 2^-7：所需指标未记录（结果里没有同输入的被测对参照统计（rel_L2 是对 FP64 真值；两侧算子输入不同时不比）） | [phase2-l1-ops-own-on](results/phase2-l1-ops-own-on.report.md) |
| L1 | 专家 GEMM / 核心注意力回放、MoE 分阶段<br>`L1.ops.kernel.moe` | ✅ | ❔ | 实测 0.9412（R 比·各用自己的输入）｜严格档 ≤ 1.2 → 0.784×，过｜参考档 rel_L2 ≤ 2^-10、归一化最大误差 ≤ 2^-7：所需指标未记录（结果里没有同输入的被测对参照统计（rel_L2 是对 FP64 真值；两侧算子输入不同时不比）） | [phase2-l1-ops-own-on](results/phase2-l1-ops-own-on.report.md) |
| L2 | 第 0 层前向<br>`L2.layer0.fwd` | ✅ | ✅ | 实测 1.059（R 比）｜严格档 ≤ 1.2 → 0.883×，过｜参考档 rel_L2 < 0.01：0.00399（0.399×）；超差元素比例 < 1%：2.61e-04（0.0261×）；均值差 < 0.01：0.0039（0.39×）；cos ≥ 0.99995：0.999992，只报，过（rel_L2 最大的是 `layers.0.mlp.fc2.out`） | [phase2-l2-replay-rratio-on](results/phase2-l2-replay-rratio-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-st1-on-state-gpuonly.png) |
| L2 | 第 1 层前向<br>`L2.layer1.fwd` | ✅ | — | 实测 1.01（R 比）｜严格档 ≤ 1.2 → 0.841×，过｜参考档 参考档不适用：这一层是 MoE 层，路由之后的观测点受近似平局翻转影响，按严格档 R 比判 | [phase2-l2-replay-rratio-on](results/phase2-l2-replay-rratio-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-st1-on-state-gpuonly.png) |
| L2 | 第 2 层前向<br>`L2.layer2.fwd` | ✅ | — | 实测 1.011（R 比）｜严格档 ≤ 1.2 → 0.843×，过｜参考档 参考档不适用：这一层是 MoE 层，路由之后的观测点受近似平局翻转影响，按严格档 R 比判 | [phase2-l2-replay-rratio-on](results/phase2-l2-replay-rratio-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-st1-on-state-gpuonly.png) |
| L2 | 第 3 层前向<br>`L2.layer3.fwd` | ✅ | — | 实测 1.095（R 比）｜严格档 ≤ 1.2 → 0.9122×，过｜参考档 参考档不适用：这一层是 MoE 层，路由之后的观测点受近似平局翻转影响，按严格档 R 比判 | [phase2-l2-replay-rratio-on](results/phase2-l2-replay-rratio-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-st1-on-state-gpuonly.png) |
| L2 | 各层反向（激活梯度）<br>`L2.layer0.bwd` | ✅ | ✅ | 实测 0.9801（R 比）｜严格档 ≤ 1.2 → 0.817×，过｜参考档 rel_L2 < 0.01：0.00212（0.212×）；超差元素比例 < 1%：1.08e-04（0.0108×）；均值差 < 0.01：8.23e-04（0.0823×）；cos ≥ 0.99995：0.999998，只报，过（rel_L2 最大的是 `layers.0.in.grad`） | [phase2-l2-replay-bwd-fp64-on](results/phase2-l2-replay-bwd-fp64-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-st1-on-state-gpuonly.png) |
| L2 | 各层反向（激活梯度）<br>`L2.layer1.bwd` | ✅ | — | 实测 0.7448（R 比）｜严格档 ≤ 1.2 → 0.621×，过｜参考档 参考档不适用：这一层是 MoE 层，路由之后的观测点受近似平局翻转影响，按严格档 R 比判 | [phase2-l2-replay-bwd-fp64-on](results/phase2-l2-replay-bwd-fp64-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-st1-on-state-gpuonly.png) |
| L2 | 各层反向（激活梯度）<br>`L2.layer2.bwd` | ✅ | — | 实测 0.8338（R 比）｜严格档 ≤ 1.2 → 0.695×，过｜参考档 参考档不适用：这一层是 MoE 层，路由之后的观测点受近似平局翻转影响，按严格档 R 比判 | [phase2-l2-replay-bwd-fp64-on](results/phase2-l2-replay-bwd-fp64-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-st1-on-state-gpuonly.png) |
| L2 | 各层反向（激活梯度）<br>`L2.layer3.bwd` | ✅ | — | 实测 1.005（R 比）｜严格档 ≤ 1.2 → 0.838×，过｜参考档 参考档不适用：这一层是 MoE 层，路由之后的观测点受近似平局翻转影响，按严格档 R 比判 | [phase2-l2-replay-bwd-fp64-on](results/phase2-l2-replay-bwd-fp64-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-st1-on-state-gpuonly.png) |
| L2 | 各层参数梯度<br>`L2.layer0.grads` | ❌ | — | 实测 0.00216（rel_L2），cos 0.999999｜严格档 N=5.34e-04（3 对）→ 4.05×N；最差的是 1−cos 1.33e-06 = 10.6×N，不过｜参考档 无对应门槛 | [phase2-st1-on-state-gpuonly](results/phase2-st1-on-state-gpuonly.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-st1-on-state-gpuonly.png) |
| L2 | 各层参数梯度<br>`L2.layer1.grads` | ❌ | — | 实测 0.0384（rel_L2），cos 0.999263｜严格档 N=5.27e-04（3 对）→ 72.9×N；最差的是 1−cos 7.37e-04 = 5.31e+03×N，不过｜参考档 无对应门槛 | [phase2-st1-on-state-gpuonly](results/phase2-st1-on-state-gpuonly.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-st1-on-state-gpuonly.png) |
| L2 | 各层参数梯度<br>`L2.layer2.grads` | ❌ | — | 实测 0.0722（rel_L2），cos 0.997408｜严格档 N=1.36e-04（3 对）→ 530×N；最差的是 1−cos 0.00259 = 2.8e5×N，不过｜参考档 无对应门槛 | [phase2-st1-on-state-gpuonly](results/phase2-st1-on-state-gpuonly.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-st1-on-state-gpuonly.png) |
| L2 | 各层参数梯度<br>`L2.layer3.grads` | ❌ | — | 实测 0.105（rel_L2），cos 0.994592｜严格档 N=3.85e-06（3 对）→ 2.7e4×N；最差的是 1−cos 0.00541 = 7.3e8×N，不过｜参考档 无对应门槛 | [phase2-st1-on-state-gpuonly](results/phase2-st1-on-state-gpuonly.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-st1-on-state-gpuonly.png) |
| L2 | 各层路由选中集合<br>`L2.layer1.router` | ✅ | — | 实测 0.9553（集合相同比例）｜严格档 ≥ 0.942 → 过｜参考档 无对应门槛 | [phase2-l2-replay-rratio-on](results/phase2-l2-replay-rratio-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-st1-on-state-gpuonly.png) |
| L2 | 各层路由选中集合<br>`L2.layer2.router` | ✅ | — | 实测 0.9692（集合相同比例）｜严格档 ≥ 0.959 → 过｜参考档 无对应门槛 | [phase2-l2-replay-rratio-on](results/phase2-l2-replay-rratio-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-st1-on-state-gpuonly.png) |
| L2 | 各层路由选中集合<br>`L2.layer3.router` | ✅ | — | 实测 0.9675（集合相同比例）｜严格档 ≥ 0.965 → 过｜参考档 无对应门槛 | [phase2-l2-replay-rratio-on](results/phase2-l2-replay-rratio-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-st1-on-state-gpuonly.png) |
| L2 | 逐层注入（误差隔离到单层）<br>`L2.layer0.fwd` | ✅ | ✅ | 实测 1.059（R 比）｜严格档 ≤ 1.2 → 0.883×，过｜参考档 rel_L2 < 0.01：0.00399（0.399×）；超差元素比例 < 1%：2.61e-04（0.0261×）；均值差 < 0.01：0.0039（0.39×）；cos ≥ 0.99995：0.999992，只报，过（rel_L2 最大的是 `layers.0.mlp.fc2.out`） | [phase2-l2-replay-rratio-on](results/phase2-l2-replay-rratio-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-st1-on-state-gpuonly.png) |
| L2 | 逐层注入（误差隔离到单层）<br>`L2.layer1.fwd` | ✅ | — | 实测 1.01（R 比）｜严格档 ≤ 1.2 → 0.841×，过｜参考档 参考档不适用：这一层是 MoE 层，路由之后的观测点受近似平局翻转影响，按严格档 R 比判 | [phase2-l2-replay-rratio-on](results/phase2-l2-replay-rratio-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-st1-on-state-gpuonly.png) |
| L2 | 逐层注入（误差隔离到单层）<br>`L2.layer2.fwd` | ✅ | — | 实测 1.011（R 比）｜严格档 ≤ 1.2 → 0.843×，过｜参考档 参考档不适用：这一层是 MoE 层，路由之后的观测点受近似平局翻转影响，按严格档 R 比判 | [phase2-l2-replay-rratio-on](results/phase2-l2-replay-rratio-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-st1-on-state-gpuonly.png) |
| L2 | 逐层注入（误差隔离到单层）<br>`L2.layer3.fwd` | ✅ | — | 实测 1.095（R 比）｜严格档 ≤ 1.2 → 0.9122×，过｜参考档 参考档不适用：这一层是 MoE 层，路由之后的观测点受近似平局翻转影响，按严格档 R 比判 | [phase2-l2-replay-rratio-on](results/phase2-l2-replay-rratio-on.report.md)<br>[图·gpuonly](figures/L2-step1-layers.phase2-st1-on-state-gpuonly.png) |
| L3 | 全部激活逐位<br>`L3.step1.act.bitwise` | ❌ | — | 实测 41 / 41 个张量不同｜严格档 逐位 / 逐项相同，不过｜参考档 无对应门槛 | [phase2-l2-replay-bitwise](results/phase2-l2-replay-bitwise.report.md) |
| L3 | 逐 token loss<br>`L3.step1.loss.per_token` | ✅ | — | 实测 0.9811（R 比）｜严格档 ≤ 1.2 → 0.818×，过｜参考档 无对应门槛 | [phase2-ff-chain](results/phase2-ff-chain.report.md) |
| L3 | logits（含 KL）<br>`L3.step1.logits` | ✅ | — | 实测 0.957（R 比）｜严格档 ≤ 1.2 → 0.798×，过｜参考档 无对应门槛 | [phase2-ff-chain](results/phase2-ff-chain.report.md) |
| L3 | 全部参数梯度<br>`L3.step1.grads.all` | ❌ | — | 实测 0.105（rel_L2），cos 0.994592｜严格档 N=5.34e-04（3 对）→ 196×N；最差的是 1−cos 0.00541 = 3.9e4×N，不过｜参考档 无对应门槛 | [phase2-st1-on-state-gpuonly](results/phase2-st1-on-state-gpuonly.report.md) |
| L3 | 梯度分布（std 比、零比例、逐元素 p99）<br>`L3.step1.grads.dist` | ❌ | — | 实测 7.26e-04（零元素比例差）｜严格档 N=1.40e-05（3 对）→ 51.9×N，不过｜参考档 无对应门槛 | [phase2-st1-on-state-gpuonly](results/phase2-st1-on-state-gpuonly.report.md) |
| L3 | 更新后主权重<br>`L3.step1.updated` | 🟡 | ✅ | 实测 5.28e-05（rel_L2），cos 1.000000｜严格档 N=4.13e-05（3 对）→ 1.28×N；最差的是 1−cos 1.39e-09 = 1.63×N，待定｜参考档 rel_L2（Frobenius） ≤ 0.02：5.28e-05（0.00264×）；cos ≥ 0.9999：1.000000，过 | [phase2-st1-on-state-gpuonly](results/phase2-st1-on-state-gpuonly.report.md) |
| L3 | 优化器状态（两个动量）<br>`L3.step1.optim_state` | ❌ | — | 实测 0.105（rel_L2），cos 0.994592｜严格档 N=0.00101（3 对）→ 104×N；最差的是 1−cos 0.00541 = 1.2e4×N，不过｜参考档 无对应门槛 | [phase2-st1-on-state-gpuonly](results/phase2-st1-on-state-gpuonly.report.md) |
| L3 | 更新后 expert_bias<br>`L3.step1.bias_after` | ✅ | — | 实测 0（rel_L2）｜严格档 N=0（3 对）→ 0×N，过｜参考档 无对应门槛 | [phase2-st1-on-state-gpuonly](results/phase2-st1-on-state-gpuonly.report.md) |
| L3 | 第 1 步全部参数梯度对 FP64（R-ratio，定责用：TPU 不比 GPU 离精确算术远）<br>`L3.step1.grads.all` | ❌ | — | 实测 2.2（R 比）｜严格档 ≤ 1.2 → 1.83×，不过｜参考档 无对应门槛 | [phase2-st1-fp64-on](results/phase2-st1-fp64-on.report.md) |
| L3 | 可选：三个梯度 FP32 开关打开后，第 1 步全部参数梯度对 FP64（R-ratio，定责用）<br>`L3.step1.grads.all` | ❌ | — | 实测 1.785（R 比）｜严格档 ≤ 1.2 → 1.49×，不过｜参考档 无对应门槛 | [phase2-st1-fp64-gradfp32](results/phase2-st1-fp64-gradfp32.report.md) |
| L3 | 比全了没有（覆盖率）<br>`L3.step1.coverage` | ✅ | — | 实测 全部比到｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-t-a-coverage](results/phase2-t-a-coverage.report.md) |
| L3 | 存档续训重叠步逐位<br>`L3.resume.state` | ✅ | — | 实测 全部 91 个参数一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-resume](results/phase2-resume.report.md) |
| L3 | 存档续训重叠步逐位<br>`L3.resume.step11` | ✅ | — | 实测 全部 256 个张量一致｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-resume](results/phase2-resume.report.md) |
| L3 | 存档续训重叠步逐位<br>`L3.resume.coverage` | ✅ | — | 实测 全部比到｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-resume](results/phase2-resume.report.md) |
| L4 | 设置核对：学习率逐步（学习率是设置的常数，不是观测量；检查 id 仍是 L4.lr）<br>`L4.lr` | ✅ | — | 实测 逐步在容差内｜严格档 rtol ≤ 0、abs_ulp ≤ 0，过｜参考档 无对应门槛 | [phase2-acceptance-rcc-on](results/phase2-acceptance-rcc-on.report.md) |
| L4 | loss（首步、带符号均值、MARE、最大单步）<br>`L4.loss` | — | ✅ | 实测 +1.03e-05（首步差）｜严格档 只报，不判｜参考档 首步差 ≤ 2e-4：1.03e-05（0.0516×），过 | [phase2-acceptance-rcc-on](results/phase2-acceptance-rcc-on.report.md) |
| L4 | loss（首步、带符号均值、MARE、最大单步）<br>`L4.loss` | ✅ | ✅ | 实测 3.85e-05（带符号均值）｜严格档 N=9.47e-05（45 对）→ 0.407×N，过｜参考档 带符号均值 ≤ 2e-4：3.85e-05（0.192×），过 | [phase2-acceptance-rcc-on](results/phase2-acceptance-rcc-on.report.md) |
| L4 | loss（首步、带符号均值、MARE、最大单步）<br>`L4.loss` | ✅ | — | 实测 1.10e-04（MARE）｜严格档 N=1.38e-04（45 对）→ 0.797×N，过｜参考档 这个量无对应门槛 | [phase2-acceptance-rcc-on](results/phase2-acceptance-rcc-on.report.md) |
| L4 | loss（首步、带符号均值、MARE、最大单步）<br>`L4.loss` | ✅ | — | 实测 5.90e-04（最大单步）｜严格档 N=9.75e-04（45 对）→ 0.605×N，过｜参考档 这个量无对应门槛 | [phase2-acceptance-rcc-on](results/phase2-acceptance-rcc-on.report.md) |
| L4 | loss（首步、带符号均值、MARE、最大单步）<br>`L4.loss.tail` | — | ✅ | 实测 +1.03e-05（首步差）｜严格档 只报，不判｜参考档 首步差 ≤ 2e-4：1.03e-05（0.0516×），过 | [phase2-acceptance-rcc-on](results/phase2-acceptance-rcc-on.report.md) |
| L4 | loss（首步、带符号均值、MARE、最大单步）<br>`L4.loss.tail` | ✅ | ✅ | 实测 3.34e-05（带符号均值）｜严格档 N=1.21e-04（45 对）→ 0.275×N，过｜参考档 带符号均值 ≤ 2e-4：3.34e-05（0.167×），过 | [phase2-acceptance-rcc-on](results/phase2-acceptance-rcc-on.report.md) |
| L4 | loss（首步、带符号均值、MARE、最大单步）<br>`L4.loss.tail` | ✅ | — | 实测 1.15e-04（MARE）｜严格档 N=1.50e-04（45 对）→ 0.771×N，过｜参考档 这个量无对应门槛 | [phase2-acceptance-rcc-on](results/phase2-acceptance-rcc-on.report.md) |
| L4 | loss（首步、带符号均值、MARE、最大单步）<br>`L4.loss.tail` | ✅ | — | 实测 4.71e-04（最大单步）｜严格档 N=5.91e-04（45 对）→ 0.797×N，过｜参考档 这个量无对应门槛 | [phase2-acceptance-rcc-on](results/phase2-acceptance-rcc-on.report.md) |
| L4 | grad_norm<br>`L4.grad_norm` | — | — | 实测 -2.24e-04（首步差）｜严格档 只报，不判｜参考档 无对应门槛 | [phase2-acceptance-rcc-on](results/phase2-acceptance-rcc-on.report.md) |
| L4 | grad_norm<br>`L4.grad_norm` | ✅ | — | 实测 0.00112（带符号均值）｜严格档 N=0.00453（45 对）→ 0.248×N，过｜参考档 无对应门槛 | [phase2-acceptance-rcc-on](results/phase2-acceptance-rcc-on.report.md) |
| L4 | grad_norm<br>`L4.grad_norm` | ✅ | — | 实测 0.028（MARE）｜严格档 N=0.033（45 对）→ 0.846×N，过｜参考档 无对应门槛 | [phase2-acceptance-rcc-on](results/phase2-acceptance-rcc-on.report.md) |
| L4 | grad_norm<br>`L4.grad_norm` | ✅ | — | 实测 0.272（最大单步）｜严格档 N=0.459（45 对）→ 0.592×N，过｜参考档 无对应门槛 | [phase2-acceptance-rcc-on](results/phase2-acceptance-rcc-on.report.md) |
| L4 | param_norm<br>`L4.param_norm` | — | — | 实测 -1.31e-07（首步差）｜严格档 只报，不判｜参考档 这个量无对应门槛 | [phase2-acceptance-rcc-on](results/phase2-acceptance-rcc-on.report.md) |
| L4 | param_norm<br>`L4.param_norm` | ✅ | ✅ | 实测 1.59e-08（带符号均值）｜严格档 N=2.27e-07（45 对）→ 0.0702×N，过｜参考档 带符号均值 ≤ 5e-6：1.59e-08（0.00319×），过 | [phase2-acceptance-rcc-on](results/phase2-acceptance-rcc-on.report.md) |
| L4 | param_norm<br>`L4.param_norm` | ✅ | — | 实测 5.43e-08（MARE）｜严格档 N=2.33e-07（45 对）→ 0.233×N，过｜参考档 这个量无对应门槛 | [phase2-acceptance-rcc-on](results/phase2-acceptance-rcc-on.report.md) |
| L4 | param_norm<br>`L4.param_norm` | ✅ | ✅ | 实测 2.44e-07（最大单步）｜严格档 N=4.86e-07（45 对）→ 0.502×N，过｜参考档 最大单步 ≤ 1.25e-5：2.44e-07（0.0195×），过 | [phase2-acceptance-rcc-on](results/phase2-acceptance-rcc-on.report.md) |
| L4 | 路由偏置 / 负载曲线<br>`L4.router.bias_std` | ✅ | — | 实测 4.18e-04（带符号均值）｜严格档 N=0.00193（45 对）→ 0.217×N，过｜参考档 无对应门槛 | [phase2-acc-health-rcc](results/phase2-acc-health-rcc.report.md) |
| L4 | 路由偏置 / 负载曲线<br>`L4.router.bias_std` | ✅ | — | 实测 0.00274（MARE）｜严格档 N=0.00353（45 对）→ 0.778×N，过｜参考档 无对应门槛 | [phase2-acc-health-rcc](results/phase2-acc-health-rcc.report.md) |
| L4 | 路由偏置 / 负载曲线<br>`L4.router.bias_std` | ✅ | — | 实测 0.013（最大单步）｜严格档 N=0.016（45 对）→ 0.81×N，过｜参考档 无对应门槛 | [phase2-acc-health-rcc](results/phase2-acc-health-rcc.report.md) |
| L4 | 健康信号（NaN、突跳）<br>`L4.health` | ✅ | — | 实测 健康｜严格档 spike_ratio ≤ 3，过｜参考档 无对应门槛 | [phase2-acc-health-rcc](results/phase2-acc-health-rcc.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.loss` | — | ✅ | 实测 +1.03e-05（首步差）｜严格档 只报，不判｜参考档 首步差 ≤ 2e-4：1.03e-05（0.0516×），过 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.loss` | ✅ | ✅ | 实测 8.04e-05（带符号均值）｜严格档 N=9.47e-05（45 对）→ 0.849×N，过｜参考档 带符号均值 ≤ 2e-4：8.04e-05（0.402×），过 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.loss` | ✅ | — | 实测 1.21e-04（MARE）｜严格档 N=1.38e-04（45 对）→ 0.875×N，过｜参考档 这个量无对应门槛 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.loss` | ✅ | — | 实测 5.32e-04（最大单步）｜严格档 N=9.75e-04（45 对）→ 0.545×N，过｜参考档 这个量无对应门槛 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.loss.tail` | — | ✅ | 实测 +1.03e-05（首步差）｜严格档 只报，不判｜参考档 首步差 ≤ 2e-4：1.03e-05（0.0516×），过 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.loss.tail` | ✅ | ✅ | 实测 9.76e-05（带符号均值）｜严格档 N=1.21e-04（45 对）→ 0.804×N，过｜参考档 带符号均值 ≤ 2e-4：9.76e-05（0.488×），过 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.loss.tail` | ✅ | — | 实测 1.36e-04（MARE）｜严格档 N=1.50e-04（45 对）→ 0.9067×N，过｜参考档 这个量无对应门槛 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.loss.tail` | ✅ | — | 实测 5.32e-04（最大单步）｜严格档 N=5.91e-04（45 对）→ 0.899×N，过｜参考档 这个量无对应门槛 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.grad_norm` | — | — | 实测 -3.60e-05（首步差）｜严格档 只报，不判｜参考档 无对应门槛 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.grad_norm` | ✅ | — | 实测 0.00283（带符号均值）｜严格档 N=0.00453（45 对）→ 0.625×N，过｜参考档 无对应门槛 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.grad_norm` | ✅ | — | 实测 0.026（MARE）｜严格档 N=0.033（45 对）→ 0.786×N，过｜参考档 无对应门槛 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.grad_norm` | ✅ | — | 实测 0.187（最大单步）｜严格档 N=0.459（45 对）→ 0.408×N，过｜参考档 无对应门槛 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.grad_norm.golden_den` | — | — | 实测 -3.60e-05（首步差）｜严格档 只报，不判｜参考档 无对应门槛 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.grad_norm.golden_den` | ✅ | — | 实测 0.00352（带符号均值）｜严格档 N=0.00614（45 对）→ 0.574×N，过｜参考档 无对应门槛 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.grad_norm.golden_den` | ✅ | — | 实测 0.0261（MARE）｜严格档 N=0.0335（45 对）→ 0.778×N，过｜参考档 无对应门槛 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.grad_norm.golden_den` | ✅ | — | 实测 0.207（最大单步）｜严格档 N=0.559（45 对）→ 0.37×N，过｜参考档 无对应门槛 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.param_norm` | — | — | 实测 -1.31e-07（首步差）｜严格档 只报，不判｜参考档 这个量无对应门槛 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.param_norm` | ✅ | ✅ | 实测 9.42e-08（带符号均值）｜严格档 N=2.27e-07（45 对）→ 0.415×N，过｜参考档 带符号均值 ≤ 5e-6：9.42e-08（0.0188×），过 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.param_norm` | ✅ | — | 实测 9.78e-08（MARE）｜严格档 N=2.33e-07（45 对）→ 0.42×N，过｜参考档 这个量无对应门槛 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.param_norm` | ✅ | ✅ | 实测 2.91e-07（最大单步）｜严格档 N=4.86e-07（45 对）→ 0.598×N，过｜参考档 最大单步 ≤ 1.25e-5：2.91e-07（0.0233×），过 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.param_norm.raw` | — | — | 实测 +5.61e-08（首步差）｜严格档 只报，不判｜参考档 无对应门槛 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.param_norm.raw` | ✅ | — | 实测 5.72e-08（带符号均值）｜严格档 N=2.27e-07（45 对）→ 0.252×N，过｜参考档 无对应门槛 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.param_norm.raw` | ✅ | — | 实测 8.30e-08（MARE）｜严格档 N=2.33e-07（45 对）→ 0.356×N，过｜参考档 无对应门槛 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.param_norm.raw` | ✅ | — | 实测 3.05e-07（最大单步）｜严格档 N=4.86e-07（45 对）→ 0.627×N，过｜参考档 无对应门槛 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的 L4 第一层（loss、grad_norm、param_norm、学习率；台账 #20，不是正式配置）<br>`L4.lr` | ✅ | — | 实测 逐步在容差内｜严格档 rtol ≤ 0、abs_ulp ≤ 0，过｜参考档 无对应门槛 | [phase2-acceptance-rcc-gradfp32](results/phase2-acceptance-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的路由偏置曲线与健康信号<br>`L4.router.bias_std` | ✅ | — | 实测 0.00164（带符号均值）｜严格档 N=0.00193（45 对）→ 0.852×N，过｜参考档 无对应门槛 | [phase2-acc-health-rcc-gradfp32](results/phase2-acc-health-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的路由偏置曲线与健康信号<br>`L4.router.bias_std` | ✅ | — | 实测 0.00258（MARE）｜严格档 N=0.00353（45 对）→ 0.731×N，过｜参考档 无对应门槛 | [phase2-acc-health-rcc-gradfp32](results/phase2-acc-health-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的路由偏置曲线与健康信号<br>`L4.router.bias_std` | ✅ | — | 实测 0.0107（最大单步）｜严格档 N=0.016（45 对）→ 0.666×N，过｜参考档 无对应门槛 | [phase2-acc-health-rcc-gradfp32](results/phase2-acc-health-rcc-gradfp32.report.md) |
| L4 | 可选：三个梯度 FP32 开关打开后的路由偏置曲线与健康信号<br>`L4.health` | ✅ | — | 实测 健康｜严格档 spike_ratio ≤ 3，过｜参考档 无对应门槛 | [phase2-acc-health-rcc-gradfp32](results/phase2-acc-health-rcc-gradfp32.report.md) |
| L4 | 梯度 NaN 先于 loss 报警<br>`L4.health.grad_norm` | ✅ | — | 演练（期望报警）：按期望报警｜本条原判定 不过：grad_norm@5=非有限 | [phase2-t-nan8](results/phase2-t-nan8.report.md) |
| L4 | 观测不改数值（开 / 不开抓数）<br>`DET.hooks.noop` | ❌ | — | 实测 不可复现：['lm_loss@1', 'grad_norm@1', 'param_norm@2']，张量不同 0｜严格档 逐位 / 逐项相同，不过｜参考档 无对应门槛 | [phase2-obs-gate](results/phase2-obs-gate.report.md) |
| L4 | 存档续训重叠步逐位<br>`L4.resume.overlap` | ✅ | — | 实测 重叠步全部逐位相同｜严格档 逐位 / 逐项相同，过｜参考档 无对应门槛 | [phase2-resume](results/phase2-resume.report.md) |

<!-- janus:table:end -->

## 结论（截至 2026-11-06，1000 步与案例 01）

### 第一层验收：全过

当前配置 `t-rmap-on-1000`（v5p 当前对齐配置 1000 步）的正式迁移验收第一层全过（[`results/phase2-acceptance-rcc-on.report.md`](results/phase2-acceptance-rcc-on.report.md)）。比较对象是最终 golden 的基线 [`rcc-a`](../../../../docs/glossary.md#rcc-a)，噪声取 `rcc-a..j` 10 份重跑两两配对得到的 45 对比较。

- 每格是三个判定量里对 N 最差的比值：loss 全程 0.80、第 301–1000 步 0.80、梯度范数 0.85、参数范数 0.50、学习率逐位。
- 路由偏置曲线 0.81，健康信号无报警（[`results/phase2-acc-health-rcc.report.md`](results/phase2-acc-health-rcc.report.md)）。
- 并行方式改成 EP = 8 同样全过（`phase2-acceptance-rcc-ep8`）。

数字与读法见[对齐报告](../../ALIGNMENT-REPORT.md)第 3 节。

### 对生产 golden `rc`：已解释差异

对生产 golden `rc` 的差是已解释差异（`phase2-acceptance-rc`，[`results/phase2-acceptance-rc.report.md`](results/phase2-acceptance-rc.report.md)）。原因是 GPU 生产路径的输出层 dgrad 在张量核里系统地朝 0 截断（[`../../config-alignment.md`](../../config-alignment.md)「输出头 dgrad」）。

- 改成单 golden 协议（基线 `rc-a`，GPU 生产 golden 的第 1 份）后重判：
  - `L4.loss`、`L4.loss.tail`、`L4.grad_norm` 待定；
  - `L4.param_norm` 不过；
  - `L4.lr` 过。
- 原来的读数：
  - 301–1000 步 loss 带符号均值对 `rc` 均值 −1.34e-4（`rc` 各份对自己均值的标准差 2.6e-5）；
  - 第 1000 步 param_norm 多 +2.1e-2。

### 根因：在 GPU 参照一侧

根因已定到 GPU 参照一侧（[案例 01](cases/01-loss-offset-param-norm.md)，已结案）。loss 偏低、param_norm 漂移、logits 整体下移、第 0 层注意力输出投影长歪，这四个现象是同一股把 logits 整体往下压的力造成的。这股力来自 GPU 输出头 dgrad 沿词表的长求和在 B200 上有偏：BF16 张量核偏得最多，cuBLAS FP32 一次累加也偏。

GPU 生产只把这一处修成分段精确累加（`rcc`），TPU 生产就落在它的散布里。TPU 生产与真精确之间的 −0.63e-2，来自两边生产程序共有的 dlogits 舍 BF16（[案例 01「一页看完」](cases/01-loss-offset-param-norm.md)）。

早先「把 GPU 输出层 dgrad 换成精确的训练不动」的结论已改判：那一臂实际跑的是 TF32（案例 01 第 0 节「输出头反向」行）。

### 已排除

- 重算（两侧）、片上内存上限、注意力实现（两侧）；
- 权重衰减掩码与 Adam 公式；
- 输出层矩阵乘与 logits 精度、梯度 BF16 规约；
- CE 融合、激活融合、TF32；
- 路由偏置更新不是 param_norm 漂移的来源：冻结后漂移不变；loss 偏移少一半，推断是偏置更新把小差别放大（案例 01 第 6 节）。

### L0–L3 正式检查（2026-11-05）

TPU 上加钩子抓的激活与梯度只是诊断值。L0–L3 改用三种不经过钩子的测量（[`../../frameworks/maxtext/README.md`](../../frameworks/maxtext/README.md#在-tpu-上拿第-1-步的正式证据三条路)「在 TPU 上拿第 1 步的正式证据」）。

- L0（`phase2-l0-state`）：载入后的参数、FP32 主权重、expert_bias、dtype 清单、模型结构全部逐位 / 合规。
- L1 单算子（`phase2-l1-ops-own`，FP64 R 比，两侧各用自己的算子输入）：
  - 注意力 kernel、MoE 块、qk 归一化、输出投影、路由 GEMM、门控激活都不比 GPU 离 FP64 远。
  - 开关打开之前只有「归一化 + 投影」两处超 1.2：qkv（R 1.235）与第 0 层 fc1（R 1.294）。
  - 当时推断是归一化输出先舍 BF16，这个推断已被取代：根因是窗口化 collective matmul 的 BF16 部分和。打开 `dense_fp32_accumulation`（FSDP 切开收缩维的投影改用 FP32 累加）后 9 条全过，见下表 A 行。
- L2 逐层前向（`phase2-l2-replay-rratio`，逐层隔离回放加 FP64 R 比）：
  - 开关打开之前 4 层都只差在 `attn.qkv.out`，原因同上一条；打开后全过，见下表 A 行。
  - 路由翻转率对 FP64 与 GPU 同量级。
- L2 逐层反向现行按 FP64 R 比判，4 层全过（`phase2-l2-replay-bwd-fp64-on`）。
  - 按 GPU 噪声判的旧结果（`phase2-l2-replay-bwd-on`）：第 0 层过，1–3 层超。隔离回放里 GPU 重跑不翻路由，越往后的层反向噪声越小，这把尺子量不到，所以改用 FP64（定责见下表 B 行）。
- 第 1 步状态读：总览与检查清单的 L2、L3 现行取 `phase2-st1-on-state-gpuonly`（被测 `t-st1n-base`，当前对齐配置，噪声只用 GPU 3 对）。
  - AdamW FP64 重放过。
  - 更新后 expert_bias 与 GPU 逐位相同（过，路由计数一致）。
  - 梯度、两个动量超 GPU 噪声 2 倍以上：
    - GPU `g1k-a/b/c`（第一阶段 GPU golden）3 份两两 3 对的逐层梯度噪声上限 3.85e-6（第 3 层）到 5.34e-4（第 0 层）；
    - TPU 对 GPU 各层最差的张量 2.16e-3（第 0 层）到 0.105（第 3 层）。
  - 更新后主权重 rel_L2 1.28 倍、1−cos 1.63 倍，介于 1–2 倍，待定。
  - 更早的 `phase2-st1-prod-state`（被测 `t-st1-prod`，换代前配置）只留作历史，见检查清单「[未列入本页的结果文件](../../CHECKS.md#未列入本页的结果文件)」。

### 观测构建：没过标定三关

观测构建（每一步都跑同一个加钩子的程序）没过标定三关（`phase2-obs-gate`、`phase2-obs-grad`、`results/phase2-obs-calibrate.json`）：

- 第一关过。
- 第二关不过：HLO 里 all-reduce 33 对 9、reduce-precision 27 对 31。
- 第三关 loss 过：带符号均值 2.80e-5、MARE 1.03e-4、最大单步 4.57e-4；GPU 噪声上限 5.39e-5 / 1.19e-4 / 8.72e-4（GPU 默认设置 10 份 1000 步运行两两 45 对）。
- 第三关第 1 步梯度不过：对同提交生产构建最大 rel_L2 8.95e-2，GPU 上限 5.3e-4。

它的张量因此不当正式证据。「观测不改数值」一格由它对 T1 LKG 的逐步比较判 ❌：第 1 步 loss 就不逐位，根因是多出的输出改变 XLA 融合（[`../../frameworks/maxtext/README.md`](../../frameworks/maxtext/README.md#上-tpu-才能核对的九项结果与现状)「上 TPU 才能核对的九项：结果与现状」第 2 条）。

同一个 TPU 程序只改融合，第 1 步梯度就差 1e-3 到 9e-2，远大于 GPU 自己的梯度噪声。这只说明 GPU 尺子量不到路由翻转；被测一侧的波动不拿来定门槛，超出部分改对 FP64 定责（下表 B 行）。

### 对齐配置打开 `dense_fp32_accumulation`

2026-11-05 决定对齐优先，打开这个开关，让 FSDP 切开收缩维的投影改用 FP32 累加（yml `a238883b7f91…`，maxtext ≥ `0622774`）。生产 rmap 配置开 / 关各跑 100 步对比代价。步时取第 11–100 步的中位数，显存取各设备峰值的最大值：

| 平台 | 关（`t-st-off-100` / `t-v7x-st-off-100`） | 开（`t-st-on-100` / `t-v7x-st-on-100`） | 步时 | 显存峰值 |
|---|---|---|---|---|
| v5p-8 | 1.6005 s | 1.6110 s | +0.66% | 23.08 GiB，不变 |
| v7x-4 | 0.9615 s | 0.9798 s | +1.9% | 23.09 GiB，不变 |

- 受影响的 L1 / L2 正式检查已按新配置重跑，结果见下表 A 行。
- 前向投影的累加精度对齐了，但它不是 L4 偏差的来源：打开之前的配置对 `rcc-a`（最终 golden 的第 1 份）也全过（[`../../config-alignment.md`](../../config-alignment.md)「收缩维被 FSDP 切开的投影」行）。

### v5p 列的 ❌ 怎么读

本节 2026-11-05 写，B 行的定责 11-06 更新。判定都来自结果文件，没有手改。下表按「门槛用的是什么尺子」分几类读，详细读法在表下。

| 类 | 行（结果文件） | 尺子 | 读法 |
|---|---|---|---|
| A 已解决：对齐配置打开 `dense_fp32_accumulation` 后转 ✅ | 第 0–3 层前向、逐层注入、单算子回放 | FP64 R 比，门槛 1.2（见注 A） | 根因是窗口化 collective matmul 在 BF16 里累加分片部分积；打开开关后全过（见注 A） |
| A′ 真实系统偏差 | 对 GPU 生产 `rc` 的 param_norm（第二层 `phase2-acceptance-rc`） | GPU `rc-a..e` 5 份重跑两两 10 对 | 方向一致、不随步数消失；已查清来自 GPU 输出头 dgrad 的长求和，对最终 golden `rcc` 在噪声内（案例 01） |
| B GPU 尺子量不到路由翻转 | 各层参数梯度、全部参数梯度、梯度分布、优化器状态、更新后主权重（现行 `phase2-st1-on-state-gpuonly`；并集结果 `phase2-st1-on-state-xnoise`、换代前的 `phase2-st1-prod-state` 留作历史；逐层反向现行按 FP64 R 判，见注 A 末） | 现行：只用 GPU `g1k-a/b/c` 3 对（2026-11-08 起）；超出部分对 FP64 定责。曾并上 TPU 只改执行参数的 7 份两两 21 对，已撤回 | 已定责：大头是第 1 步路由近似平局翻转（台账 #5）；第 0 层是稠密层，它的超出来自梯度以 BF16 落地（台账 #20）。都不累积（见注 B） |
| D 程序本身改了数值 | 观测不改数值（`phase2-obs-gate` 的 `DET.hooks.noop`） | 逐位 | 加钩子改变 XLA 融合，见上一小节 |
| C 按定义不逐位 | 全部激活逐位（`phase2-l2-replay-bitwise`，hard: false） | 逐位 | 跨平台不要求逐位，只记事实 |

注 A（A 类）。`dense_fp32_accumulation` 让 FSDP 切开收缩维的投影改用 FP32 累加。门槛 1.2 是推断值，取在干净实现 R = 1.00 与多舍一次 R = 1.41 之间，见术语表「R-ratio」。打开开关之前只差在「归一化 + 投影」两处：qkv R 1.21–1.24、第 0 层 fc1 R 1.29（[`../../config-alignment.md`](../../config-alignment.md)）。yml `a238883b7f91…` 打开开关后重跑（`r-l2-on`，当前对齐配置下的 TPU 逐层隔离回放）：

- 单算子 9 条全过（qkv 1.000、fc1 1.117，`phase2-l1-ops-own-on`）；
- 逐层前向 4 层全过（最大 R 1.06 / 1.01 / 1.01 / 1.09，`phase2-l2-replay-rratio-on`）；
- golden 输入的单算子 7 条全过（`phase2-l1-ops-on`）。

逐层反向不受这个开关影响：按 GPU 噪声判仍是第 0 层过、1–3 层超（`phase2-l2-replay-bwd-on`）；现行按 FP64 R 比判，4 层全过（`phase2-l2-replay-bwd-fp64-on`）。

注 B（B 类）。更新后 expert_bias 在生产配置下已逐位相同，不在这一类。GPU 第 1 步只有反向的原子加不确定，噪声极小（梯度 3.8e-6–5.3e-4）；而同一个 TPU 程序加不加钩子，第 0 层梯度就差 2.17e-3。

定责：第 1 步每层 0.44% – 0.56% 的路由槽位在近似平局处两边各落一边，分到 token 少的专家梯度相对差到 0.10。GPU 重跑第 1 步路由从不翻转，这把尺子对路由翻转的宽度是 0。按梯度能量加权，整网只差 1.3e-3，不累积（对齐报告第 4 节、台账 #5）。生产配置下这几行仍超，第 0 层梯度最差 2.15e-3，路由权重梯度到 0.10。

2026-11-06 补了第 1 步的 TPU 执行噪声：当前配置 7 份只改执行参数，两两配对 21 对，判法见 [`docs/reference/judging.md`](../../../../docs/reference/judging.md#第-1-步张量两侧执行噪声的并集)。

- 整步检查曾改按 GPU 3 对与 TPU 21 对的并集判（`phase2-st1-on-state-xnoise`，历史）。2026-11-08 起撤回并集、只用 GPU 3 对（`phase2-st1-on-state-gpuonly`）：全部超 2 倍，判「不过 · 已解释」（FP64 定责见对齐报告台账 #5 / #20）。
- 逐层反向改按 FP64 R 判，全过（`phase2-l2-replay-bwd-fp64-on`）。
- 整步对 FP64 的定责见案例 01「第 1 步张量换尺子」。一维 norm 权重、输出层与 embedding 的梯度先以 BF16 落地，机制与开关见对齐报告台账 #20。

### 30 步阶段的结论

30 步阶段（`t-loss30`，TPU 30 步冒烟）的结论仍然成立：L0 地基（检查点逐位、数据回放逐位、配置）已过；30 步上 TPU 比 GPU 自己的噪声大。

GPU 噪声带后来用 10 份同设置运行重估过，比 3 份 golden 两两 3 对的差宽（案例 01 第 6 节「噪声带」）。这 10 份是 `g1k-a/b/c`（第一阶段 GPU golden）加 7 份，各自对 `g1k-a` 比；301–1000 步 loss 带符号均值 −2.5e-5 ± 2.7e-5。

## 排查案例

[案例 01：TPU 1000 步 loss 稳定偏低万分之一点五](cases/01-loss-offset-param-norm.md)，已结案。TPU 没有算错，偏差来自 GPU 输出头 dgrad 的长求和；最终 golden `rcc` 只修这一处，TPU 生产就在噪声内。

## 已有结果（CPU 上能做的部分）

| 文件 | 比什么 | 结论 |
|---|---|---|
| `results/phase2-ckpt-convert` | HF 转成 MaxText 后，按 `name_map.yaml` 还原回 HF 命名与形状，对 HF 原件的 `tensor_sha256.json` | 见 [`results/phase2-ckpt-convert.report.md`](results/phase2-ckpt-convert.report.md)（由 `.results.json` 渲染） |
| `results/phase2-cpu/` | 名字映射覆盖、Megatron 布局对 golden 参数清单、数据回放、诊断前向、CPU 冒烟的命令输出 | 见 [`../../frameworks/maxtext/README.md`](../../frameworks/maxtext/README.md#已在-cpu-上核对)「已在 CPU 上核对」 |
| `results/phase2-native-loader` | TPU 原生数据加载器（Grain `mmap_npy`）与 GPU golden 逐样本对拍 | 见 [`results/phase2-native-loader.report.md`](results/phase2-native-loader.report.md)；做法见 [`../../data/README.md`](../../data/README.md) |
