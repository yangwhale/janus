[janus](../../../../README.md) › [用户指南](../../../../docs/guide/README.md) › [混元 3](../../README.md) ｜ TPU 侧操作手册；对应指南[第 3 章 TPU 侧准备](../../../../docs/guide/03-tpu-setup.md)

# 混元 3 · TPU 侧（MaxText）

和 [`../megatron/README.md`](../megatron/README.md) 对称：

- 同一个截 4 层的混元 3（第 0 层稠密、1–3 层 MoE，关 MTP）、同一份 HF 权重、同一批数据。
- 在 TPU 上用 MaxText 训练，按 [`docs/reference/dump-format.md`](../../../../docs/reference/dump-format.md) 抓数，再用 janus 对 GPU golden。

本页结构：先是文件与步骤（怎么跑），再是权重、数据回放、抓数、配置对齐（各环节怎么对上 GPU），最后是核对结果与已知限制。

- 平台：先在 **v5p 8 片（2x2x2，2 台主机，8 个 JAX 设备）** 上调通，之后 **v7x 4 片（2x2x1，1 台主机，8 个 JAX 设备）**。
  - 两者都是 8 个设备，对上 GPU 8 卡。
- 代码：`maxtext` 的 `hunyuan` 分支。**验收（正式被测、日常回归的 LKG）钉在 `0622774a`，结论只对这个提交成立；用更新的提交先跑日常回归 T0、T1，对 LKG 逐位相同再用**（[日常回归](../../regression/README.md)）。复现验收时给 [`stage.sh`](stage.sh) 第二个参数显式传 `0622774a`，不传默认拉 `origin/hunyuan` 最新。
  - [`hy3-4layers.yml`](hy3-4layers.yml) 打开四个精度开关：
    - `routed_gate_bf16_rounding`（路由核先舍 BF16 再乘）、`rmsnorm_single_rounding`（RMSNorm 只舍一次）、`mlp_activation_round_per_op`（门控激活按算子各舍一次）从 `4d4ffac` 起有；
    - `dense_fp32_accumulation`（FSDP 切开收缩维的投影改用 FP32 累加）从 `0622774`起有。
  - 具体提交由 [`stage.sh`](stage.sh) 记进每个运行的 `meta.json` 的 `code`。
  - 训练循环、数据回放、抓数都在本目录与 [`janus/dump/maxtext.py`](../../../../janus/dump/maxtext.py)。
  - 为对齐改到 MaxText 里的只有这几个开关（[`../../config-alignment.md`](../../config-alignment.md#为对齐改过的代码)「为对齐改过的代码」）。
- 镜像：MaxText 官方 `tpu_pre_training` 0.2.5，按 digest 钉死（[`run.sh`](run.sh) 里的 `IMAGE`）。
  - 实际的 jax / jaxlib / libtpu 版本在运行时读出来，写进 `meta.json` 的 `versions`。
- CPU 验证环境：jax / jaxlib 0.11.1、flax 0.12.10、optax 0.2.8、orbax-checkpoint 0.12.7。
  - 这是 `hunyuan` 分支 `tpu-requirements.txt` 的下限版本，去掉 libtpu 装在 CPU 上。

## 文件

| 文件 | 作用 |
|---|---|
| [`../../model/name_map.yaml`](../../model/name_map.yaml) | HF ↔ Megatron ↔ MaxText ↔ 规范名：参数（含转置、拆分、拼接、专家分组）与激活观测点。解释器 [`janus/dump/name_map.py`](../../../../janus/dump/name_map.py) |
| [`hy3-4layers.yml`](hy3-4layers.yml) | MaxText 训练配置，逐项写了 GPU 那一侧的对应值与出处 |
| [`../../model/convert_hf_to_maxtext.py`](../../model/convert_hf_to_maxtext.py) | HF → MaxText 参数检查点（CPU；跨平台共用，放在模型一层） |
| [`../../model/verify_ckpt.py`](../../model/verify_ckpt.py) | 逐张量核对（模型一层），三个子命令见注 1 |
| [`replay.py`](replay.py) | 数据回放：golden 的 `data/batches` 拼成每步全局 batch，`check` 子命令逐条核对 |
| [`train_hy3.py`](train_hy3.py) | 训练入口：MaxText 建模型与优化器、载入检查点，每步喂 golden 同一步的 batch，抓数 |
| [`run.sh`](run.sh) / [`stage.sh`](stage.sh) | 提交 JobSet / 打包代码上传 |
| [`profile.sh`](profile.sh) | 提交方式（见注 2） |
| [`site-profile.example.sh`](site-profile.example.sh) | 站点配置的最小示例（占位值） |
| [`probe_forward.py`](probe_forward.py) | 诊断用的单独前向：转换后的权重对 golden 第 1 步逐层比（核对名字与布局，不当精度结论） |
| [`cpu_smoke.py`](cpu_smoke.py) / [`smoke_checks.py`](smoke_checks.py) | CPU 冒烟：小维度、8 个假 CPU 设备，把整条链路跑一遍再用 janus 读 |
| [`mt_common.py`](mt_common.py) | 上面几个脚本共用的 MaxText 辅助 |

- 注 1：`verify_ckpt.py` 的三个子命令。
  - `names`：映射覆盖。
  - `megatron`：对 GPU 的 params_sha。
  - `maxtext`：检查点反向取回，对 tensor_sha256.json。
- 注 2：`profile.sh` 默认提交到自己的节点池。
  - 集群特有的设置由仓库外的站点配置（`JANUS_SITE_PROFILE`）覆盖。
  - `run.sh` 与日常回归共用它。

## 步骤

### golden 放在哪

- `<桶>` 是 POC 桶。golden 都在 `gs://<桶>/janus/hunyuan3-4l/golden/<运行名>/` 下：
  - 最终 golden `rcc-a..j`（正式验收的基线 `rcc-a`，噪声 `rcc-a..j`）；
  - GPU 生产 `rc-a..e`；
  - 第一阶段 `g1k-a/b/c`（不开重算，1000 步，3 份）、`g1k-det`（确定性设置版，只用来归因）。
- 各自的角色见 [`../../goldens.md`](../../goldens.md)，代号的意思见[术语表](../../../../docs/glossary.md#golden-codenames)。
- 下面的步骤和命令以第一阶段的 `g1k-*` 为例，换成别的 golden 只改运行名。

### 九步总表

| 步 | 在哪 | 命令 | 产物 |
|---|---|---|---|
| 1. 权重 | 任意 | 见注 1 | HF 目录 + `tensor_sha256.json`（见注 1） |
| 2. 核对映射 | CPU（以下 2–4 步在 `models/hunyuan3/model/` 下执行） | 见注 2 | 都打印 `OK` |
| 3. 转换 | CPU（约 60 GB 内存，4 分钟） | `JAX_PLATFORMS=cpu python convert_hf_to_maxtext.py --hf <HF 目录> --out <输出>` | `<输出>/0/items`（Orbax，FP32，约 22 GB） |
| 4. 核对检查点（L0 正式检查） | CPU | 见注 3 | 3 条 L0 检查 pass（见注 3） |
| 5. 上传检查点 | 任意 | `gcloud storage cp -r <输出> gs://<桶>/janus/hunyuan3-4l/maxtext/` | `…/maxtext/hy3-4l-maxtext/0/items` |
| 6. 核对数据 | 任意 | `python replay.py check <golden 目录>`（只要 `data/`） | `OK` |
| 7. 打包代码 | 任意 | `export BUCKET=gs://<桶>; source <(bash stage.sh <maxtext 克隆>)` | 导出 `MAXTEXT_PKG` / `JANUS_PKG` / 提交号 / sha256 |
| 8. 训练并抓数 | TPU | 见下「在 TPU 上跑」 | `gs://<桶>/janus/hunyuan3-4l/tpu/<运行名>/` |
| 9. 比对 | CPU | 见下「跑 janus 比对」 | `results.json` / `report.md` |

- 注 1（第 1 步）：
  - 命令：GPU 侧截好的 HF 目录（[`../megatron/README.md`](../megatron/README.md#3-截-4-层去-mtp)「截 4 层、去 MTP」）原样传一份到 `gs://<桶>/janus/hunyuan3-4l/hf/hy3-4l/`。
  - 产物：`tensor_sha256.json` 的 sha256 应等于 golden `meta.checkpoint.manifest_sha256`。
- 注 2（第 2 步），两条命令：

  ```bash
  python verify_ckpt.py names --hf <HF 目录>
  python verify_ckpt.py megatron --hf <HF 目录> --gpu-run <golden 目录>
  ```

- 注 3（第 4 步），三条命令（前两条在 `models/hunyuan3/model/` 下，第三条在仓库根目录）：

  ```bash
  python verify_ckpt.py hf-manifest --hf <HF 目录> --janus-run <运行根>/hy3-4l-hf
  JAX_PLATFORMS=cpu python verify_ckpt.py maxtext --ckpt <输出>/0/items --hf <HF 目录> --janus-run <运行根>/hy3-4l-maxtext-ckpt
  python -m janus run models/hunyuan3/alignments/01-v5p-acceptance/catalogs/phase2-ckpt-convert.yaml --root <运行根>
  ```

  - 产物：3 条 L0 检查 pass。
    - 1781 个张量逐位；
    - dtype 清单（expert_bias FP32、其余 BF16）。

### 装 MaxText 环境（CPU）

第 3 步之前先装：

```bash
git clone https://github.com/yangwhale/maxtext maxtext && git -C maxtext checkout hunyuan   # 公开分支，见 docs/public-edition.md
uv venv -p 3.12 mt-venv
grep -v -E '^(libtpu|torch|nvidia|triton)' maxtext/src/dependencies/requirements/generated_requirements/tpu-requirements.txt \
  | sed 's/>=/==/;s/ ;.*//' > req.txt
VIRTUAL_ENV=mt-venv uv pip install -r req.txt pyyaml && VIRTUAL_ENV=mt-venv uv pip install --no-deps -e maxtext
```

### 在 TPU 上跑

```bash
export BUCKET=gs://<桶> CONTEXT=<TPU 集群的 kubectl 上下文>
source <(bash stage.sh <maxtext 克隆>)
# 最小可跑：先看 TPU 的 loss 曲线。不加钩子、不抓张量、不写参数清单，只写 metrics.jsonl 与输入指纹
PLATFORM=v5p bash run.sh t-loss30 30 --no-hooks --dump-steps none --no-param-sha
# 被测：1000 步，第 1、1000 步抓数，第 1 步全量（梯度、更新前后主权重、Adam 动量）
PLATFORM=v5p bash run.sh t-a 1000 --dump-steps 1,1000 --full
# 「抓数不改数值」的对照：同配置不加钩子
PLATFORM=v5p bash run.sh t-nohooks 1000 --no-hooks --dump-steps 1,1000
# v7x
PLATFORM=v7x NODEPOOL=<v7x 节点池> bash run.sh t-v7x-a 1000 --dump-steps 1,1000 --full
# 复现正式被测 t-rmap-on-1000 的程序：stage.sh 第二个参数传 0622774a，yml 用 a238883b 那一版（见 ../../regression/README.md「状态」），
# 命令行加这组重算覆盖（照 01 runs.yaml 的 overrides；抓数参数按要比的检查选）。只用 yml 默认跑出来的不是验收的那个程序
PLATFORM=v5p bash run.sh <运行名> 1000 context=remat query_proj=remat key_proj=remat value_proj=remat out_proj=remat mlpwi_0=remat mlpwi_1=remat mlpwo=remat moe_mlpwi_0=remat moe_mlpwi_1=remat moe_mlpwo=remat moe_routing_maps=device
```

#### 提交、取回、清理

- 每台主机跑同一条命令；进程 0 写运行目录。
- [`run.sh`](run.sh) 提交后一直等，直到写运行目录的 Pod 打出 `JANUS_READY`（训练结束，不论成败）。
  - 这个 Pod 是 JAX 进程 0 所在的那个，不一定是 0 号 Pod。
- 然后用 `kubectl exec … tar` 把运行目录与日志取回提交机（默认 `~/janus-runs/`）。
- **从提交机**传到 `gs://<桶>/janus/hunyuan3-4l/tpu/<运行名>/` 与 `…/tpu/logs/<运行名>.train_w0.log`，再通知 Pod 退出。
- `NO_WAIT=1` 只提交；之后 `COLLECT_ONLY=1 bash run.sh <运行名> 0` 取回。
- 跑完 `kubectl --context $CONTEXT delete jobset janus-<运行名>`。

#### 多主机初始化与网络

- 多主机时 [`train_hy3.py`](train_hy3.py) 在建配置前自己调 `jax.distributed.initialize()`。
  - MaxText 只在开存档时才初始化，而我们关了存档。
  - 载入检查点（orbax v1）的跨进程同步又离不开它。
  - 不初始化会报 `ThreadSafeKeyValueSignalingClient should only be used in a single controller setup, process count: 2`。
- **必须 `hostNetwork: true` + `dnsPolicy: ClusterFirstWithHostNet`**（`run.sh` 已带）。
  - 不开时多主机的 `jax.distributed` 初始化卡住，TPU backend 初始化超过 60 s 不返回。

#### 为什么由提交机上传

- **为什么不从 Pod 里上传**：
  - Pod 用了 `hostNetwork`，GKE 的元数据服务器拦不到它的请求，Pod 的身份就是节点的服务账号（Workload Identity 不生效）。
  - 节点池的访问范围是只读存储，给 KSA 授权也写不进桶（v5p `t-loss30`（TPU 30 步冒烟）实测上传报 `Provided scope(s) are not authorized`）。
  - 改节点池的访问范围要重建节点，所以由提交机上传。
- 权限分工：Pod 只需要读 golden、检查点、代码包；提交机的账号要能写 `…/tpu/` 前缀。
- 写运行目录的 Pod 训练结束后最多等 `HOLD` 秒（默认 3600），让提交机取走。

#### 盘与性能参数

- 本地盘：golden 数据约 1.8 GB。
  - 全量抓数的运行目录与 GPU 同量级：GPU 四份 golden 共 198 GB（`gcloud storage du` 实测），平均每份约 50 GB。
- `LIBTPU_INIT_ARGS` 默认留空：对齐阶段先不加性能类参数。
  - 要加，先用「同程序只改这一个开关」证明它不改数学（[`docs/reference/environment.md`](../../../../docs/reference/environment.md)）。

#### `train_hy3.py` 的参数

基本：

| 参数 | 作用 |
|---|---|
| `--load` | MaxText 检查点（`…/0/items`，可以是 `gs://`） |
| `--golden` | golden 运行目录（本地；`run.sh` 先拉下来） |
| `--hf-manifest` | HF 目录的 `tensor_sha256.json`，写进 `meta.checkpoint.manifest_sha256` |
| `--steps` | 步数；本侧第 s 步吃 golden 第 s 步的 batch（`golden_step = step`） |
| `--janus-run` `--side` | 运行目录、`side`（默认 `tpu`；扰动臂用 `tpu-perturb-<n>`） |
| 末尾 `key=value` | 追加给 MaxText 的配置（写进 `meta.config.overrides`），例如扰动臂改 `remat_policy` |

抓数：

| 参数 | 作用 |
|---|---|
| `--dump-steps 1,1000` | 抓数的步（本侧步号 = golden 步号）；`none` 不抓张量 |
| `--full` / `--full-steps` | 全量抓数（默认只在抓数步里最小的那一步做） |
| `--full-only` | 全量抓数只抓 Megatron 名匹配的参数（见注 1）；路由偏置照抓 |
| `--no-hooks` | 不加图内钩子：只写逐步标量、输入指纹、参数清单、状态类张量 |
| `--no-selfcheck` | 抓数步直接用加钩子程序的输出往下训练、不跑对照（见注 2） |
| `--observe` | 观测构建（见下「观测构建」）；要求 GA 1（见注 3） |
| `--acts-steps` | 只在这些抓数步写激活、logits、逐 token loss（默认所有抓数步；`none` 都不写）；程序不变，只是不取回 |
| `--capture` / `--probe-tiers` | 抓数档位；在第一个抓数步逐档和生产图逐位比 |
| `--no-param-sha` | 不写参数清单（省掉训练前后各一次全量读状态） |
| `--state-sha` | 另写训练状态的 FP32 全量清单 `state_sha/{init,final}.json`（见注 4） |
| `--hlo-dir` | XLA dump 目录（见注 5） |

存档、换参数、故障演练：

| 参数 | 作用 |
|---|---|
| `--save-state-at N` / `--state-dir` | 第 N 步更新之后存整个训练状态（见注 6） |
| `--resume-state` + `--first-step N+1` | 开训前从存档恢复整个训练状态，从第 N+1 步接着训（见注 7） |
| `--export-params-at N[,…]` | 第 N 步更新之后导出整套 FP32 主权重（见注 8） |
| `--params-from <目录>` / `--first-step K` | 开训前把整套参数换成目录里的，从第 K 步开始（见注 9） |
| `--inject-nan-grad N` / `--nan-target` | 故障演练：第 N 步在参数梯度里埋一个 NaN（见注 10）。要求 GA 1 |

梯度探针：

| 参数 | 作用 |
|---|---|
| `--grad-probe-at s[,…]` | 第 s 步训练之前另编三个独立 jit 程序算梯度（见注 11） |
| `--probe-ref-weights bf16\|master` | `--grad-probe-at` 的参照程序用哪份权重（见注 12） |

- 注 1（`--full-only`）：逗号分隔的几个正则按「或」合并。
  - 不要用括号、竖线：参数要经 JobSet 里的 bash 转一道。
- 注 2（`--no-selfcheck`）：省两份训练状态的 HBM；轨迹可能偏离生产图。
- 注 3（`--observe`）：每一步都用同一个加钩子的程序往下训练，额外输出只在 `--dump-steps` 取回。
- 注 4（`--state-sha`）：不转 BF16，日常回归 R07 / R15 / R18 用；训练前后各多一次全量读状态。
- 注 5（`--hlo-dir`）：配 `run.sh` 的 `XLA_DUMP=1`，即 `/data/xla`。
  - 训练结束后把 `jit_train_step` 优化后的 HLO、`flagfile`、`debug_options` 拷进运行目录 `hlo/`。
  - 日常回归 R05 / R20 / R21 用。
- 注 6（`--save-state-at`）：用 orbax 把整个训练状态存到 `--state-dir`（本地或 gs://）。
  - 训练状态包括主权重、两个动量、路由偏置、计数器、随机数键。
  - 同时写 `state_sha/step<N>.json`。
- 注 7（`--resume-state`）：恢复后写 `state_sha/step<N>.json`，可和存档那次逐项比（`phase2-resume`）。
- 注 8（`--export-params-at`）：按 GPU 的 rank 划分导出到 `<运行目录>/export/step_N/`。
  - 与 GPU 侧 `param_swap.py` 同一种格式，见 [`param_io.py`](param_io.py)。
- 注 9（`--params-from`）：目录可以是本地或 `gs://`；放回后读出逐张量核 sha256。
  - 第 K 步吃 golden 第 K 步的 batch。
  - 用途：在另一侧来的状态上重算某一步的梯度。
- 注 10（`--inject-nan-grad`）：默认埋在第 0 层注意力输出投影，裁剪与优化器之前，同 GPU 演练 `phase1-nan`。
  - 其余步照常；`meta.fault` 记埋了什么。
- 注 11（`--grad-probe-at`）：用当前状态与第 s 步的 batch 编程序，不进训练图、不捐出状态，轨迹不变。三个程序：
  - 生产精度的梯度 `gp_bf16.*`；
  - 参照精度（`--probe-ref`，默认全程 FP32、`matmul_precision=highest`、dot-product 注意力、ragged dot）的梯度 `gp_f32.*`；
  - 参照精度下 logits 均值对参数的梯度 `gp_lvl.*`（「所有 logits 一起动」的方向）。
  - 只写 `--full-only` 选中的参数（专家留 0、1），输出层另写沿词表求和与逐行平方和。
  - 损失值记进 `metrics.jsonl` 的 `gp_loss_*`。
  - 程序本身就是测量对象，不靠钩子。
- 注 12（`--probe-ref-weights`）：
  - 默认 `bf16` = 主权重先舍到 BF16（生产程序实际参与计算的那份，只比算术误差）；
  - `master` = FP32 主权重原样。

### 站点配置（`JANUS_SITE_PROFILE`）

默认提交到自己的节点池（`CONTEXT` + `NODEPOOL`）。

- 集群特有的提交设置（队列标签、节点选择、配额检查等）放在仓库外的站点配置文件里，通过 `JANUS_SITE_PROFILE` 指定。
- 只换提交方式：镜像、拓扑、Pod 里的命令都不变。

```bash
export JANUS_SITE_PROFILE=/path/to/site-profile.sh   # 仓库外；从 site-profile.example.sh 复制后按自己的集群改
export BUCKET=gs://<POC 桶>                           # 结果（OUT）在这里
export DATA_BUCKET=gs://<Pod 读的桶>                  # 可选：golden、检查点、代码包放这里（同样的相对路径），例如与 TPU 同区域的副本
source <(bash stage.sh <maxtext 克隆>)       # 给了 DATA_BUCKET 时代码包传到它下面
bash run.sh t-loss30 30 --no-hooks --dump-steps none --no-param-sha
# 日常回归同样认 JANUS_SITE_PROFILE：bash ../../regression/run_regression.sh T0 <运行名>
```

#### 可重定义的函数

[`profile.sh`](profile.sh) 先定义默认函数再 source 站点配置，站点配置重定义哪个函数就用哪个：

| 函数 | 做什么 | 默认实现 |
|---|---|---|
| `profile_resolve` | 设 `CONTEXT`、`NAMESPACE`、`COLLECT`，以及 JobSet 里随站点变化的四段（见注 1） | `nodeSelector` 指定 `NODEPOOL`；`COLLECT=exec` |
| `profile_ours` | 列出我们还没结束的 `janus-*` JobSet | 同节点池上的 `janus-*` |
| `profile_busy_reason` | 日常回归提交前：不该提交的理由（空 = 可以） | 同节点池上有别的 `janus-*` 没结束 |
| `profile_admit` | [`run.sh`](run.sh) 提交前检查（例如配额、排队），不该提交就 `exit 2`；约定认 `ALLOW_QUEUE=1` 跳过 | 不查 |
| `profile_delete` | 删一个 JobSet，只删我们的 | 直接删 |

- 注 1（`profile_resolve`）：`COLLECT` 取 `exec` / `gcs`。随站点变化的四段：
  - `JS_META`（JobSet 的 labels / annotations）；
  - `POD_META`（Pod 模板的 metadata）；
  - `NODE_SELECTOR`；
  - `TOLERATIONS`（追加的行）。

#### 取回方式与权限

- `COLLECT=gcs`：用于没有 `pods/exec` 权限、但节点服务账号能写 `OUT` 的集群。
  - 写运行目录的 Pod 自己把运行目录、HLO、日志传到 `$OUT`。
  - 打出 `JANUS_READY … upload=0` 就退出（不占着切片等人取）。
  - 提交机从 `$OUT` 拉回 `~/janus-runs/`。
- **读和写的权限都是节点的服务账号**（hostNetwork 下 Workload Identity 不生效）。
  - 能读写哪些桶，取决于节点的访问范围与桶的 IAM。
  - 换集群或换桶之前，先用一个 CPU 小 Job（同样 `hostNetwork: true`）试读 `DATA_BUCKET`、试写 `OUT`。

#### 换提交方式不改数值

- 做法：用同一份代码包换一种提交方式重跑 `t-loss30`（TPU 30 步冒烟；2026-11-05 实测）。
- 逐位相同的部分：
  - 30 步 `metrics.jsonl` 的全部 17 个字段（`step_time` 除外）；
  - `data/` 输入指纹；
  - `meta.json` 里的版本（jax、libtpu build）。
- `meta.json` 只差运行名、检查点路径、时间。

#### 节点被回收

- 例如可抢占节点：Pod 收到 `imminent node shutdown`，JobSet 变成 Failed，运行目录没写出来。
- 删掉 JobSet 原名重提即可（`OUT` 里没有产物）。

### 跑 janus 比对

```bash
mkdir -p /data/janus-runs
gcloud storage cp -r gs://<桶>/janus/hunyuan3-4l/golden/g1k-{a,b,c} gs://<桶>/janus/hunyuan3-4l/tpu/{t-a,t-nohooks} /data/janus-runs/
# 把 models/hunyuan3/alignments/01-v5p-acceptance/catalogs/phase2-tpu.yaml 的 runs.test / runs.lkg 换成实际运行名、runs.root 换成 /data/janus-runs
python -m janus run models/hunyuan3/alignments/01-v5p-acceptance/catalogs/phase2-tpu.yaml --out /data/janus-results/phase2-tpu --verbose --html
```

- `phase2-tpu.yaml` 是第二阶段开始时的草稿目录。
- L2 / L3 的第 1 步前向检查标了 `hard: false`（目录文件头写了原因）：
  - GPU 默认设置下第 1 步前向逐位可复现，GPU 运行对的噪声是 0；
  - 所以 calibrated 门槛等于「逐位」。
- L2 / L3 的正式判定改用下面「在 TPU 上拿第 1 步的正式证据：三条路」的第 1、2 条。
- L4 的正式验收目录：
  - `phase2-acceptance-rcc-on`（v5p）；
  - `phase2-v7x-acceptance-rcc`（v7x）；
  - 见 [`../../ALIGNMENT-REPORT.md`](../../ALIGNMENT-REPORT.md)。

## 权重：映射与核对

全部映射写在 [`../../model/name_map.yaml`](../../model/name_map.yaml)。只有转置、改形状、切片、拼接，没有算术。

| 参数 | HF | Megatron | MaxText（scan_layers，层轴 = 第 1 维，除注明外） |
|---|---|---|---|
| 词表 | `embed_tokens` [V, D] | 原样 | `token_embedder/embedding` [V, D]，原样 |
| 输出层 | `lm_head` [V, D] | 原样 | `logits_dense/kernel` [D, V]，**转置** |
| q / k / v | 三个 [out, D] | **按 KV 组交错拼成** `linear_qkv` [10240, D]（见注 1） | `query` [D, L, 64, 128]、`key` / `value` [D, L, 8, 128]：**转置 + 拆头** |
| 输出投影 | `o_proj` [D, 8192] | 原样 | `out/kernel` [64, L, 128, D]：**转置 + 拆头** |
| 输入归一化 | `input_layernorm` | `linear_qkv.layer_norm_weight`（融进 qkv 线性层） | `pre_self_attention_layer_norm/scale` [D, L] |
| MLP 前归一化 | `post_attention_layernorm` | 第 0 层 `linear_fc1.layer_norm_weight`（融进 fc1）；1–3 层 `pre_mlp_layernorm.weight` | `post_self_attention_layer_norm/scale` [D, L] |
| 稠密 MLP | `gate/up/down_proj` | `linear_fc1` = **[gate; up] 拼接**，`linear_fc2` 原样 | `wi_0` / `wi_1` / `wo`：**转置** |
| 路由 | `router.gate` [192, D] | `router.weight` 原样 | `gate/kernel` [D, L, 192]：**转置** |
| 路由偏置 | `expert_bias` [192] FP32 | buffer `router.expert_bias` FP32 | `MoEBiasVar …/gate/bias` [L, 192]：**层轴在第 0 维**（见注 2） |
| 路由专家 | `experts.<e>.{gate,up,down}_proj` | rank r 上 `experts.linear_fc{1,2}.weight<k>`，**e = 24r + k**；fc1 = [gate; up] | `MoeBlock_0/wi_0`、`wi_1` [E, L, D, 1536]、`wo` [E, L, 1536, D]：**转置**，专家轴第 0 维 |
| 共享专家 | `shared_mlp.*` | `shared_experts.linear_fc1` = [gate; up] | `shared_experts/wi_0`、`wi_1`、`wo`：**转置** |

- 注 1：8 组，每组 8 个 q 头、1 个 k 头、1 个 v 头。
- 注 2：不进优化器的变量不按 param_scan_axis 堆叠。

### 转换的三条事实

- 参数总数：转换后 12,386,079,744（由 name_map 推出的 MaxText 叶子形状求和）。
  - 与 GPU 侧 `import_hy3.py` 报的 4 层参数量相同。
- RoPE：两侧都是前后两半配对，转换不重排 q / k 的行。
  - CPU 诊断前向对 golden 的 `attn.core.out` 吻合在 BF16 末位量级，见下。
- 检查点存 FP32：MaxText 的 `weight_dtype=float32` 就是主权重。
  - BF16 → FP32 精确，转回 BF16 逐位还原（`verify_ckpt.py maxtext` 就是这么核的）。

### L0 权重转换检查

转换后的逐张量核对是 janus 的正式检查，不是脚本打印。

- `verify_ckpt.py maxtext --janus-run` 写 MaxText 一侧的参数清单：
  - 读 MaxText 检查点，按 `name_map.yaml` 把每个张量还原回 HF 的命名与形状（拼接 / 拆分 / 转置反做）；
  - 按检查点里**实际存的 dtype** 还原：存 FP32 的，只有在 HF 原件是 BF16、且每个值都是 BF16 精确值时才转回 BF16；
  - 逐张量算原始字节的 sha256，写成一个运行目录的参数清单（另记 `stored_dtype`）。
- `verify_ckpt.py hf-manifest` 写另一个：把 HF 原件的 `tensor_sha256.json` 加上 safetensors 头里的 dtype / 形状。
- 目录 [`models/hunyuan3/alignments/01-v5p-acceptance/catalogs/phase2-ckpt-convert.yaml`](../../alignments/01-v5p-acceptance/catalogs/phase2-ckpt-convert.yaml) 有两条检查：
  - `checkpoint_equal`：清单模式，逐项比 sha256，数清两边张量数；
  - `dtype_manifest`：expert_bias 必须 FP32、其余 BF16。
  - 同样两条也写在 `phase2-tpu.yaml` 的 L0（`select.test_run` / `ref_run` / `run` 指到这两个运行）。
- 结果：`../../alignments/01-v5p-acceptance/results/phase2-ckpt-convert.results.json` / `.report.md`。
  - 1781 个张量逐位相同、0 个只在一边；
  - dtype 3 个 FP32（expert_bias）、1778 个 BF16。
- 自证：CPU 冒烟里喂一份故意改坏的小检查点（一个专家权重翻一位、三个 expert_bias 存成 BF16）。
  - `checkpoint_equal` 报 4 个张量不同、`dtype_manifest` 报 3 个 dtype 不符（`../../alignments/01-v5p-acceptance/results/phase2-cpu/cpu-smoke.txt`）。

## 数据回放

不重新分词、不用 MaxText 的数据管线。

### 怎么拼全局 batch

- [`replay.py`](replay.py) 读 golden 的 `data/batches/rank<r>.{bin,jsonl}`。
  - 里面是 GPU 每 rank 每步每个 microbatch 原样的 tokens / labels / loss_mask / position_ids。
- 每个字段先核 sha256。
- 再按 **全局第 r·M + m 行 = GPU rank r 的第 m 个 microbatch**（M = 2）拼成 [16, 4096] 的全局 batch。
- MaxText 把全局 batch 按 batch 维均分到 8 个分片，第 d 个分片拿第 2d、2d+1 行，正好是 GPU rank d 的两个 microbatch。
  - **TPU 设备 d ↔ GPU rank d** 指的是这个分片号；
  - 它落在哪个物理设备上，写在 `meta.data.device_of_shard`。

### 字段对应

- `inputs = tokens`、`targets = labels`、`inputs_position = targets_position = position_ids`。
- `inputs_segmentation = 1`：整条 4096 一段，和 GPU 一样不按文档切注意力。
- `targets_segmentation = (loss_mask ≠ 0)`。
- golden 的 loss_mask 全是 1、每个 microbatch 的 nonpad 都是 4096（`replay-check.txt`）。
  - 所以 GPU 的「每个 microbatch 按 token 平均再平均」与 MaxText 的「全局按 token 平均」数学上相同（求和顺序不同）。

### 输入指纹

- TPU 侧每步把喂进设备的 batch **从设备读回来**，换回 GPU 的 dtype 与形状算 sha256，写同格式的 `data/rank<d>.jsonl`。
- `data_fingerprint` 按 `<步>.r<rank>.m<microbatch>` 逐位对 golden。

### 梯度累积

- [`hy3-4layers.yml`](hy3-4layers.yml) 默认 `per_device_batch_size=1 gradient_accumulation_steps=2`：和 GPU 一样每个 microbatch 单独前反向再累加。
  - MaxText 按 `行 % 2` 取第 m 个累积步，设备 d 拿第 d 行。
  - v5p `t-ga2-30`（TPU 改成 GA 2 的 30 步运行）的输入指纹逐个 microbatch 逐位。
- 旧排法 `per_device_batch_size=2 gradient_accumulation_steps=1`（一次处理两行）也对得上。
  - CPU 冒烟里两种排法的输入指纹逐位相同、loss 只差求和顺序。
- 加钩子的全量抓数只支持不累积。
  - GA>1 时 [`train_hy3.py`](train_hy3.py) 自动按 `--no-hooks` 跑，激活改用 [`layer_replay.py`](layer_replay.py)。

## 抓数

实现在 [`janus/dump/maxtext.py`](../../../../janus/dump/maxtext.py)，原则按 [`docs/reference/dump.md`](../../../../docs/reference/dump.md)。

### 抓什么、怎么拿

| 类 | 内容 | 怎么拿 | 扰动 |
|---|---|---|---|
| 逐步标量 | `lm_loss`、`grad_norm`（裁剪前，和 golden 同义）、`grad_norm_clipped`、`param_norm`、`lr`、`wd`、`skipped_iter`、`total_weights`、每层 `router.bias_mean/std` | 见注 1 | 无 |
| 输入指纹 | `data/rank<d>.jsonl` | 每步喂进去的 batch 从设备读回 | 无 |
| 参数清单 | `params_sha/rank<r>.{init,final}.json` | 训练前后从状态读出（见注 2） | 无 |
| 状态类张量 | `param.*`（主权重转 BF16）、`main_param.*` / `main_param_after.*`、`exp_avg*`、`buffer*` | 抓数步训练前后从状态读（见注 3） | 无 |
| 参数梯度 | `grad.*`（已归约、裁剪前，与 GPU 的 `main_grad` 同义） | 加钩子的训练步（见注 4） | 额外输出，靠自检兜底 |
| 激活 | `embed.out`、`layers.<i>.{in,out}`、`attn.{qkv,q_norm,k_norm,core,proj}.out`、`attn.out`、`mlp_norm.out`、`mlp.{fc1,fc2,out}`（稠密）/ `mlp.{shared,routed,out}`（MoE）、`final_norm.out`、`logits`、`loss.per_token`、路由 `logits / scores / topk_idx / probs / counts` | 加钩子的训练步（见注 5） | 额外输出，靠自检兜底 |

- 注 1（逐步标量）：MaxText 训练步本来就输出的标量。
  - `lr` 按调度函数、`wd` 按配置在主机上算；
  - 偏置统计在步与步之间从状态读。
- 注 2（参数清单）：**按 GPU EP8 的持有关系分 rank**。
  - rank r：全部非专家参数 + 专家 24r … 24r+23；
  - 换成 Megatron 名字与布局；
  - 参数存 BF16（主权重转 BF16，与 GPU 的 BF16 参数同义），expert_bias 存 FP32。
- 注 3（状态类张量）：名字是 Megatron 名、布局是 Megatron 布局。
  - 专家只留每层专家 0、1（= GPU rank 0 的 weight0 / weight1）。
- 注 4（参数梯度）：把 `train` 模块里的 `jax` 换成只拦 `value_and_grad` 的代理，取 MaxText 自己算出的梯度。
  - 专家叶子在图里切成专家 0、1 再输出。
- 注 5（激活）：
  - 按实例路径给模块打标记，给几个模块类的 `__call__` 套壳；
  - 用 nnx 的 `sow` 存成 Intermediate，随 MaxText 原有的通道带出；
  - 只留设备 0 的两行，在图里切片。

### 自检与生产轨迹

- 每个抓数步，往下训练**一律用不加钩子的程序**（MaxText 原样的 `train_step`）的输出。
- 加钩子的程序在同一输入状态的副本上跑，只取它的额外输出。
  - 再逐叶子逐位比两者输出的训练状态与 loss / 三个范数，结果写进 `meta.hooks.selfcheck`。
- 逐位相同时，额外输出算「生产图里的值」。
  - 否则 `meta.hooks.extras_are` 标为诊断值。
  - 训练状态、逐步标量、状态类张量不受影响，都来自生产图。
- 这只证明「输出逐位相同」。
  - 抓到的激活是否就是不加钩子时图里的那个值，原理上看不到（[`docs/reference/dump.md`](../../../../docs/reference/dump.md) 第 2 条）。
- 档位 `--capture full|boundary|router|grads|none`（[`janus/dump/maxtext.py`](../../../../janus/dump/maxtext.py) 的 `TIERS`）。
  - `--probe-tiers boundary,grads,none` 在第一个抓数步把各档都和生产图比一遍，看哪一档在这个配置上不改数值。
- v5p 实测（`t-a` 第一次提交，janus `3eb043d`；`t-a` 是 TPU 第 1 步全量抓数的 1000 步运行）：`full` 档加钩子程序的输出与生产图**不逐位相同**。
  - 91 个状态叶子里 90 个不同，loss 不同。
  - 所以当时往下训练用加钩子程序的做法，会让整条轨迹偏离生产图——这就是改成上面做法的原因。
  - CPU 上各档都逐位相同，CPU 的结论不能搬到 TPU。

### 和 GPU 抓数的不同

- **激活的梯度（`<名>.grad`）不抓**：余切量不落在任何模块输出上。
  - 要往图里插探针输入才能拿到，会改图。
- GPU 的 `mlp.experts.out`（专家并行分发缓冲区）TPU 没有对应量。TPU 多写两个：
  - `mlp.routed.out`：路由专家合并后、加共享专家前，token 空间；
  - `mlp.router.scores`：sigmoid 之后、加偏置之前。
- `topk_idx` / `probs` / `counts` 是**主机上派生的**：稀疏路径里 top-k 在 `shard_map` 里，抓不出来。
  - 主机用图内抓到的同一份 FP32 分数重算：加偏置后取 top-8，并列取小下标，同 `jax.lax.top_k`；
  - 权重按 `deepseek_scale_weights` 的顺序在 FP32 里算。
- 第 0 层的 `mlp_norm.out` 是残差和（层输入 + 注意力输出），不是归一化后的值。
  - Megatron 第 0 层的 `pre_mlp_layernorm` 是恒等（归一化融进 fc1），GPU 抓到的就是残差和。
- `metrics.jsonl` 不写 `router.load_*`。
  - GPU 记的是 rank 0 本地两个 microbatch 的专家计数，TPU 要每步都抓路由才有。

## 配置对齐

- 逐项写在 [`hy3-4layers.yml`](hy3-4layers.yml) 的注释里。
- `config_diff` 用的映射在 [`models/hunyuan3/alignments/01-v5p-acceptance/catalogs/phase2-tpu.yaml`](../../alignments/01-v5p-acceptance/catalogs/phase2-tpu.yaml) 的 `L0.config.diff`。

要点：

| 项 | GPU（golden `meta.json`） | TPU |
|---|---|---|
| 学习率 | 1e-5 常数，无预热 | `lr_schedule_type=wsd`，预热与衰减都为 0 → `optax.constant_schedule(1e-5)` |
| 权重衰减 | **实际 0.033**（见注 1）；一维参数不衰减 | `adam_weight_decay=0.033`；`adamw_mask: ["norm/scale"]`（见注 2） |
| AdamW | β1 0.9、β2 0.95、eps 1e-8（加在根号外），裁剪 1.0 | 同；`mu_dtype=float32` |
| 精度 | BF16 计算、FP32 主权重、FP32 梯度归约、FP32 动量 | `dtype=bfloat16`、`weight_dtype=float32`、`grad_dtype=float32` |
| 路由 | sigmoid + expert_bias，top-8，`moe_router_dtype=fp32`，偏置更新率 1e-3，缩放 2.826，aux loss 0 | 模型 yml 同值；`float32_gate_logits: True`；`matmul_precision=default`（见注 3） |
| dropout / 随机路由 | 0 / 无 | `enable_dropout: False`、`dropout_rate=0`、`use_random_routing=false` |
| MTP | 关（`mtp_num_layers=None`） | `mtp_num_layers=0`（必须命令行覆盖模型 yml，[`train_hy3.py`](train_hy3.py) 自动加） |
| 批与序列 | gbs 16、mbs 1、8 卡 → 每卡 2 个 microbatch，seq 4096 | `per_device_batch_size=1`、`gradient_accumulation_steps=2`、8 个设备、`max_target_length=4096` |
| 重算 | 见注 4 | 正式验收与生产：`remat_policy=custom`，命令行另加 11 项重算覆盖（`context`、`query_proj`、`key_proj`、`value_proj`、`out_proj`、`mlpwi_0`、`mlpwi_1`、`mlpwo`、`moe_mlpwi_0`、`moe_mlpwi_1`、`moe_mlpwo` 都 `=remat`）与 `moe_routing_maps=device`，除层输入与路由结果外都重算。yml 默认（各投影存 HBM、不重算）只给日常回归 T0 / T1（见注 5） |
| 并行 | DP8 + EP8 | FSDP 8（归约顺序本来就不同，列为已知差异） |
| 种子 | 1234 | 1234（权重载入、数据回放，种子不影响数值） |

- 注 1（GPU 权重衰减）：`scheduler_config` 覆盖了 `optimizer_config.weight_decay=0.1`。
- 注 2（TPU `adamw_mask`）：扫描层把一维 scale 堆成 [D, L]，只能按名字排除。
- 注 3（TPU 路由 `matmul_precision=default`）：操作数先舍 BF16、单遍乘、FP32 累加。
  - 操作数和 GPU 的路由 GEMM 相同；
  - 累加方式不同：GPU 张量核截断、TPU 就近舍入。
- 注 4（GPU 重算）：
  - 最终 golden `rcc-*` 与 GPU 生产 `rc-*`：整层重算 + 路由回放（`03-gpu-remat-golden`）；
  - 第一阶段 `g1k-*` 不开。
- 注 5（TPU 重算）：`moe_routing_maps=device` 等。
  - 原因：整层重算会让反向里重算出的路由与前向不同。

### 权重衰减的实测依据

一维参数不衰减、实际衰减 0.033 这两条，两侧都有实测依据：

- GPU 侧：`models/hunyuan3/alignments/00-gpu-noise-and-drills/results/phase1-full.results.json` 的 `L1.optim.adamw.replay`。
  - 按「一维参数 wd = 0、其余 0.033」做 FP64 重放，141 个张量全部在门槛内。
- TPU 侧：生产配置下同一种重放都在门槛内。
  - 第 1 步状态读：144 个张量（48 个参数 × 更新量与两个动量）；
  - 连续 8 步重放：66 个张量；
  - 运行：01 `phase2-st1-prod-state`、`phase2-st8-opt8`，02 `phase2-v7x-st1-prod-state`、`phase2-v7x-st8-opt8`。
- 能验证到什么程度，见 [`../../config-alignment.md`](../../config-alignment.md)「优化器与调度」。

## 已在 CPU 上核对

命令输出都在 [`../../alignments/01-v5p-acceptance/results/phase2-cpu/`](../../alignments/01-v5p-acceptance/results/phase2-cpu/)：

| 核对 | 结果文件 | 结论 |
|---|---|---|
| name_map 覆盖 HF 目录的全部张量、形状一致 | `verify-names.txt` | 1781 / 1781，形状全对 |
| HF 按 name_map 拼成 Megatron 布局，对 golden 的 `params_sha/rank<r>.init.json` | `verify-megatron.txt` | 8 个 rank × 185 个参数全部逐位相同（见注 1） |
| HF → MaxText 转换后反向取回，对 `tensor_sha256.json`（janus 正式检查） | `convert.txt`、`verify-maxtext.txt`、`../phase2-ckpt-convert.results.json` | 1781 个张量逐位相同、dtype 清单符合（见注 2） |
| golden 数据回放 | `replay-check.txt` | 1000 步 × 16 个 microbatch，sha256 全对（见注 3） |
| 诊断前向：真权重、golden 第 1 步 rank 0 第一个 microbatch 前 256 个 token（CPU、稠密 MoE 路径、点积注意力） | `probe-forward-256.txt` / `.json` | 名字与布局没有错位（见注 4） |
| CPU 冒烟（小维度、8 个假设备、3 步）：转换 → 训练 → 抓数 | `cpu-smoke.txt` | 见注 5 |

- 注 1：qkv 交错、gate/up 拼接、专家本地号、expert_bias 都核到。
- 注 2：MaxText 抽象模型的 30 个变量与 name_map 推出的形状逐个一致。
- 注 3：拼成 MaxText 字段再还原，指纹与 golden 的 `data/rank*.jsonl` 全部相同。
- 注 4（诊断前向）：
  - `embed.out`、`layers.0.in` 逐位相同；
  - 其余 49 个连续值观测点 rel_L2 1.3e-3 – 3.2e-2、cosine ≥ 0.9994，名字与布局没有错位（错位是 O(1) 的差）；
  - 路由 top-8 集合逐 token 一致 0.91–0.92，`probs` 随之不同；
  - 路由差别多大算正常不在这里判：诊断前向不是生产图，CPU 上路由矩阵乘的输入精度也和 GPU 不同【推断】。
- 注 5（CPU 冒烟，`cpu-smoke.txt` 里能看到的）：
  - 转换检查对改坏的检查点报警（4 个张量 sha256 不同、3 个 dtype 不符）；
  - 第 1 步钩子自检逐位相同；
  - 加 / 不加钩子两次运行逐步 loss、grad_norm 逐位相同。
  - 当时另有两份 janus 结果（`cpu-smoke`、`cpu-smoke-init`），用的目录是冒烟目录里的临时文件、没进仓库，对不上指纹，已删除；
    它们核过的几项在真规模上另有 01 的结果：初始参数清单 `phase2-l0-state`，FP64 重放 AdamW `phase2-st1-prod-state`、`phase2-st8-opt8`，数据指纹 `phase2-t-loss30`。

CPU 冒烟另外核对过（ad hoc，不在结果文件里）：

- 层与层之间 `layers.i.out == layers.(i+1).in` 逐位；
- `mlp.out == routed.out + shared.out` 逐位；
- 用抓到的参数在 FP64 里重算 `qkv.out`、`fc1.out`、`attn.proj.out`、路由专家合并输出，都吻合在 BF16 末位量级。

## 上 TPU 才能核对的九项：结果与现状

第二阶段开始时列的九条清单，每条后面加粗的是现状：

- 6 条已核：1、2、3、6、7、9。
- 1 条部分已核：4（精度已核，图内与主机重算的逐 token 比较改由单独编译的回放承担）。
- 1 条改了做法：5。
- 1 条仍开：8（关闭条件写在该条）。

1. `hunyuan` 分支在 0.2.5 镜像上能跑（依赖版本只在 CPU 上对过下限）；v5p 两台主机的 `jax.distributed` 初始化与数据切分（`meta.data.device_of_shard`）。
   - **已核**：v5p、v7x 的全部运行；
   - 输入指纹逐个 microbatch 逐位（`phase2-t-loss30` 等的 `L0.data.fingerprint`）。
2. **加钩子的训练步在 TPU 上和不加钩子逐位相同**（`hooks.selfcheck`，以及 `t-a`（抓数的 1000 步运行）对 `t-nohooks`（同配置不加钩子）的 `L0.hooks.determinism`）。
   - 要核的原因：CPU 上逐位不代表 TPU 编译器的融合决策不变。
   - **已核，结论是不逐位。v5p 实测结果（2026-11-04）**：
     - **只有不多出额外输出的档逐位相同。** `none` 档逐位；`grads` 档不带梯度切片时（`t-hlo`）也逐位，因为加钩子的程序编出来和生产程序是同一个。
     - **多出输出就不逐位。** `grads` 档带梯度切片时（`t-hlo2`），91 个状态叶子里 15 个不同，标量里 `grad_norm` 不同，loss 与 `raw_grad_norm` 逐位；`boundary` 档 86 / 91 个叶子不同。
     - **根因：多出的输出改变了 XLA 的融合决策。** 见下面三条：
       - 在 `t-hlo2` 的 XLA dump 里，两份程序优化前的 HLO 除了多出的输出（32 条 reshape / slice，ROOT 相应多出几项），计算部分逐行相同（6762 对 6794 条指令）。
       - 优化后 fusion 从 856 个变成 859 个，梯度范数的逐叶子平方和、RMSNorm scale 梯度的 `dot_general` 转置等 `reduce_sum` 被重新分组进不同的 fusion。FP32 规约的分组和顺序一变，末位就变。
       - 推断：要把某个中间值变成程序输出，它就必须被物化，这会改变它前后 fusion 的划分。
     - **结论：** 在 TPU 上，往编译好的训练步里加输出不是数值中性的，加钩子抓到的激活和梯度只能当诊断值。
       - 逐层判定改用 [`layer_replay.py`](layer_replay.py)：每一层单独编译，回放程序本身就是测量对象。
       - 生产轨迹不受影响：`t-a`（抓数）与 `t-nohooks` 1000 步逐步逐位相同，最终参数 sha256 也相同。
3. MaxText 在 TPU 上从 GCS 载入检查点后，初始参数清单对 golden 逐位（`L0.weights.init`；CPU 冒烟只在小模型上核过载入路径）。
   - **已核**：1480 个参数 0 个不同；
   - 载入后主权重 50 个张量逐位（`phase2-l0-state`、`phase2-v7x-l0-state`）。
4. 路由：图内 top-k 与主机按同一份分数重算的结果一致（稀疏路径在 `shard_map` 里，CPU 冒烟走的是稠密路径）；`matmul_precision=default` 下路由矩阵乘在 TPU 上的实际精度（看 HLO）。
   - **精度已核**：v5p 实测 DEFAULT 下 FP32 操作数先舍 BF16、单遍乘（[`config-alignment.md`](../../config-alignment.md)「路由 GEMM」行）。
   - **图内与主机重算的逐 token 比较没有单独做**：加钩子的程序在 TPU 上不逐位（第 2 条）。
     - 路由的正式判定改在单独编译的回放里做（`L1.router.topk.fp64`、`L2.layer{i}.router`）。
5. 各激活观测点在**生产图**里抓到的值（CPU 诊断前向只核了名字与布局）；第 0 层 `mlp_norm.out` 与 MoE 层的对应关系、`qkv.out` 融合与否的舍入点（看 HLO）。
   - **改了做法**：生产图里抓不到不改数值的激活（第 2 条），改用下一节的隔离回放。
   - `qkv.out` 的舍入点查清是窗口化 collective matmul 的 BF16 部分和（`dense_fp32_accumulation`）。
6. 优化器状态树的路径（`OptVariable/optimizer/opt_state/0/{mu,nu}/…`，CPU 上读出）在 TPU 上一致；`exp_avg*` 与 GPU 同名张量的语义一致（第 1 步 FP64 重放与 golden 对比）。
   - **已核**：`L0.dtype.state` 按这些路径匹配 182 项（v5p `phase2-l0-state`）；
   - FP64 重放见上「配置对齐」末段。
7. 内存：自检要多一份训练状态；全量抓数的额外输出（全部非专家参数的 FP32 梯度、logits 两行）放得下。
   - **已核**：v5p `t-a`（`--dump-steps 1,1000 --full`）跑完 1000 步。
8. `remat_policy=custom` + 各张量存 HBM 是否真的不重算路由（`L1.router.topk.fp64` 与 GPU 同量级）；它与「整层重算」能不能当扰动臂。
   - **仍开**：编译后的程序里没有核实。
     - 日常回归 R20 ⑥ 暂缓：路由矩阵乘被融进 fusion，还没有可靠的匹配规则。
   - 关闭条件：写出 R20 ⑥ 的 HLO 匹配规则并在 LKG 上核过。
   - 间接证据：L4 对 `rcc-a`（最终 golden 的第 1 份，验收的比较基线）全部在噪声内。
   - 见 [`config-alignment.md`](../../config-alignment.md)「路由是否在重算里重选」行。
9. v7x 2x2x1 上同样的命令跑通。
   - 已核：02 的全部运行，见 [`../../alignments/02-v7x-acceptance/README.md`](../../alignments/02-v7x-acceptance/README.md)。

## 在 TPU 上拿第 1 步的正式证据：三条路

- 加钩子的训练步在 TPU 上和生产程序不逐位（上面「上 TPU 才能核对的九项：结果与现状」第 2 条）。
- 抓到的激活与梯度只能当诊断值。
- 要正式判定，换成下面三种「程序本身就是测量对象」的办法。

1. **状态读 + 动量反推梯度（零扰动）**：训练步之间从状态读出的主权重、两个动量、expert_bias 来自生产图。
   - 梯度从一阶动量反推：g_k = (m_k − β1·m_{k−1}) / (1 − β1) / 裁剪系数。
     - [`../../analysis/grads_from_moments.py`](../../analysis/grads_from_moments.py) 写派生运行目录。
   - 二阶动量留给优化器 FP64 重放核对。
   - 连续几步抓状态时用 `--full-only` 只留一小组参数，控制数据量。
2. **隔离回放**：[`layer_replay.py`](layer_replay.py) 把 golden 的输入喂进单独编译的程序。
   - 逐层：`--layers`、`--grad`。
   - 单算子与 kernel：`--l1`。
     - 覆盖路由 GEMM、MLP 前归一化、门控激活、注意力（RoPE + 缩放 + attention_op）、MoE 块；
     - 生产开关下的 TPU 输出存进运行目录 `l1.*`。
   - 输出头：`--head`。
   - 整网前向：`--full-forward`，只前向、不加任何抓数输出，记 logits 与逐 token loss。
   - 前向的判定用 FP64 R-ratio（[`../../analysis/fwd_rratio.py`](../../analysis/fwd_rratio.py)）。
     - GPU 第 1 步前向逐位可复现，噪声上限 0，当不了尺子。
3. **观测构建**（`--observe`）：每一步都用同一个加钩子的程序往下训练。
   - 整条轨迹只有这一个编译好的程序，抓到的就是它自己的中间值。
   - 它和生产构建不逐位，要先当一个执行级扰动臂过标定三关：
     - 规则见 [`docs/reference/regression.md`](../../../../docs/reference/regression.md) 第五节；
     - 目录 [`../../alignments/01-v5p-acceptance/catalogs/phase2-obs-gate.yaml`](../../alignments/01-v5p-acceptance/catalogs/phase2-obs-gate.yaml)。
   - 过了，它的第 1 步张量才当 L2 / L3 的正式证据；L4 正式验收仍用生产构建。
   - **v5p 实测（2026-11-05）：没过，不进参照。**
     - 第一关过：`obs-t0-a` / `obs-t0-b` 20 步逐位。
     - 第二关不过：优化后 HLO 里 all-reduce 33 个（生产 9 个）、reduce-precision 27 个（生产 31 个）。
       - 多出的输出带来额外的跨设备规约，还少了 4 处显式舍入，不只是改了执行。
     - 第三关：1000 步 loss 三个量都在 GPU 噪声上限内。
       - 噪声上限来自 GPU 默认设置 10 份 1000 步运行：`g1k-a/b/c`（第一阶段 GPU golden）与只差日志选项的 7 份，两两 45 对。
       - 但第 1 步参数梯度对同提交生产构建差到 8.95e-2（GPU 上限取 `g1k-a/b/c` 3 份两两 3 对，5.3e-4）。
     - 所以在 v5p 上观测构建抓到的张量仍只当诊断值，L2 / L3 的正式判定用第 1、2 条路。
     - 结果：01 实践 `results/phase2-obs-gate.*`、`phase2-obs-grad.*`、`phase2-obs-calibrate.json`。

## 已知限制

- 逐层注入（GPU 侧 `--inject-from`）TPU 侧没有：扫描层里拿不到层号。
  - 要么关 `scan_layers`（检查点布局随之变化），要么改 MaxText。
- 路由回放：MaxText 有 `forced_routed_experts` 通道，但 `Hunyuan3MoELayer.__call__` 没有这个参数。
  - 要改 `hunyuan3.py` 才能接上。
- 激活梯度、`router.load_*` 不抓（见上）。
- TPU 扰动臂（只改执行方式的 TPU 运行）已在日常回归里定义并标定（[`../../regression/README.md`](../../regression/README.md)「五、TPU 自噪声标定」）。
  - L4 用它们说明 TPU 不是挑出来的一个好样本（ALIGNMENT-REPORT 第 3 节）。
  - 第 1 步整步的噪声并集用的是另一组 7 份只改执行参数的状态读（`phase2-st1-on-state-xnoise.yaml` 开头）。
  - 草稿目录 [`phase2-tpu.yaml`](../../alignments/01-v5p-acceptance/catalogs/phase2-tpu.yaml) 没有接这些扰动臂，第 1 步前向检查仍是 `hard: false`，不进验收。
