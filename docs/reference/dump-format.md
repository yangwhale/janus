# 接口规范 · 抓数格式

`schema_version: 1`。所有比较都从这里读。

- 不符合本规范的产物，janus 拒绝读并报 `error`。

本页结构：目录 → `meta.json` → `metrics.jsonl` → 张量（文件、索引、规范名）→ 输入指纹与参数清单 → 要求。

## 目录

```
<root>/<run-name>/
  meta.json                         # 运行级元信息（必有）
  metrics.jsonl                     # 逐步标量（训练类运行必有）
  step_<N>/                         # 抓了张量的步，N 用本侧自己的步号
    index.json                      # 本步全部张量的索引（必有）
    <scope>.safetensors             # 张量文件，按 scope 分组
  data/rank<R>.jsonl                # 每 rank 每步每个 microbatch 的输入指纹（可选，数据检查要用）
  params_sha/rank<R>.<init|final>.json  # 每 rank 训练前 / 后持有的每个参数与浮点 buffer 的 sha256（可选，全量不抽样）
  state_sha/<init|final>.json       # 训练状态的 FP32 全量清单（可选，TPU 侧；日常回归 R07 / R15 / R18 用）
  hlo/train_step.*                  # 生产训练步的优化后 HLO 与编译器生效的开关（可选，TPU 侧；日常回归 R05 / R20 / R21 用）
  modules.txt                       # 实例化出来的模块树与参数清单（可选，「看产物不看配置」用）
  DONE                              # 写完最后一个文件后创建；没有它的目录视为未完成
```

- `<run-name>` 全模型唯一，登记在对齐实践的 `runs.yaml`。
  - 写法见 [`../dev/conventions.md`](../dev/conventions.md)「运行名」（`<前缀>-<区分符>`，如 `g1k-a`、`t-loss30`）。
- `<side>` 写在 `meta.json` 里，取值：
  - `golden-<n>`（GPU 参照）；
  - `tpu`（被测）；
  - `tpu-perturb-<n>`（只改执行方式的扰动臂）；
  - `gpu-perturb-<n>`。
- 产物写完即只读，不覆盖；重跑用新的 `<run-name>`。

## meta.json

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `schema_version` | int | 是 | 本规范版本，现为 1 |
| `run_name` | str | 是 | 同目录名 |
| `side` | str | 是 | 见上 |
| `framework` | str | 是 | `megatron` / `maxtext` |
| `model` | str | 是 | 如 `hunyuan3-4l` |
| `code` | obj | 是 | `{repo, commit}`；有多个仓库时为数组 |
| `image` | str | 是 | 镜像 `@sha256:` digest |
| `versions` | obj | 是 | 关键版本：jax / jaxlib / libtpu，或 torch / transformer-engine / cudnn / nccl |
| `hardware` | obj | 是 | `{accelerator, count, topology}` |
| `parallelism` | obj | 是 | `{dp, fsdp, tp, ep, cp, pp}`，没有的写 1 |
| `config` | obj | 是 | `{path, sha256}`：训练配置文件 |
| `checkpoint` | obj | 是 | `{path, manifest_sha256}`：初始权重；manifest 列出每个文件的 sha256 |
| `data` | obj | 是 | `{manifest, manifest_sha256, seed}` |
| `steps` | obj | 是 | `{first, last, convention}`，见注 1 |
| `determinism` | obj | 是 | 开了哪些确定性设置（环境变量与开关原样列出） |
| `dtypes` | obj | 是 | `{compute, weight, optimizer_states, grad_reduce}` |
| `hooks` | obj | 否 | 抓数钩子的版本与开关；未抓张量时省略 |
| `model_config` / `optimizer_config` / `scheduler_config` | obj | 否 | 运行时实际配置的关键项（`config_diff` 用），见注 2 |
| `fault` | obj | 否 | 故障注入运行才有：埋了什么错（只用于验证检查会报警） |
| `created_at` | str | 是 | ISO 8601，带时区 |

缺项怎么处理：janus 读写运行目录时只查必填的键在不在（`janus/store.py` 的 `META_REQUIRED`），缺键就报错、打不开；不查内容。`image` 只记了 tag 没有 digest、`versions` 少了某个库（如 libtpu 的 pip 版本号）都不报错。这种运行引用时把缺的那一项写成「推断」并给出依据（例：混元 3 的 libtpu 按构建号认作 0.0.46，见[环境「两个模型实测用的版本」](environment.md#两个模型实测用的版本)）。

注 1（`steps.convention`）：
- 写明本侧步号与 golden 步号的换算（如 `golden_step = step + 1`）；
- `janus run` 取被测的逐步序列时按它换算。

注 2（三份运行时配置）：调度器会覆盖优化器里的权重衰减，两份都要记。

## metrics.jsonl

一行一步，按步号递增：

```json
{"step": 0, "lm_loss": 2.0231120586395264, "lr": 0.0, "raw_grad_norm": 1.53125, "grad_norm": 1.0, "param_norm": 412.5}
```

- 必有 `step`。
- 其余字段用规范名：
  - 标量：`lm_loss`、`mtp_loss`、`z_loss`、`aux_loss`、`lr`、`wd`（实际生效的权重衰减）、`raw_grad_norm`、`grad_norm`、`param_norm`、`num_zeros`、`skipped_iter`；
  - 逐层：`router.bias_mean.<i>` / `router.bias_std.<i>` / `router.load_cv.<i>` / `router.load_max.<i>` / `router.load_min.<i>` 等。
  - 框架原生名不同时，在 `models/<模型>/metrics_map.yaml` 里映射；
  - 混元 3 的 GPU 抓数回调直接写规范名，不需要映射文件。
- 浮点写成能往返还原的十进制（Python `repr` / `float.hex` 等价精度），**不得截断**。
  - 日志里只打几位小数的值不能当数据源。
- 非有限值写成字符串 `"nan"`、`"inf"`、`"-inf"`。

## 张量

### 文件

- 用 safetensors 存，**保持原始 dtype**（BF16 就存 BF16），不先升 FP32。
- 一个 `<scope>.safetensors` 放一组张量（如 `layers.3`、`grads.layers.3`、`router`），控制单文件大小与文件数。
- 默认存**全局张量**（各 rank 拼好后的完整张量）。
  - 规模太大存不下时按 rank 分片存，文件名 `<scope>.rank_<R>.safetensors`；
  - 在索引里写分片方式。

### index.json

```json
{
  "schema_version": 1,
  "step": 0,
  "tensors": [
    {"name": "layers.3.mlp.router.logits", "kind": "activation", "point": "forward",
     "file": "layers.3.safetensors", "dtype": "float32", "shape": [8192, 192],
     "layout": "global", "sha256": "…"}
  ]
}
```

| 字段 | 说明 |
|---|---|
| `name` | 规范名（见下） |
| `kind` | `activation` / `grad` / `param` / `opt_state` / `router` / `tokens` / `other` |
| `point` | `forward` / `backward` / `before_update` / `after_update` |
| `dtype`、`shape` | 全局形状；分片存时另给 `shard: {rank, index, global_shape}` |
| `layout` | `global` 或 `sharded` |
| `sha256` | 该张量原始字节的 sha256 |

### 规范名

`<作用域>.<模块>.<点>`，作用域与层号从 0 开始，和框架原生名脱钩：

#### 各类张量

- 激活：
  - `embed.out`、`layers.<i>.in`；
  - `layers.<i>.attn.{qkv,q_norm,k_norm,core,proj}.out`、`layers.<i>.attn.out`、`layers.<i>.mlp_norm.out`；
  - `layers.<i>.mlp.{fc1,fc2}.out`（稠密层）、`layers.<i>.mlp.{experts,shared}.out`（MoE 层）；
  - `layers.<i>.mlp.out`、`layers.<i>.out`、`final_norm.out`、`logits`。
- 路由：`layers.<i>.mlp.router.{logits,probs,topk_idx,counts}`（counts = 本 rank 每个专家收到的 token 数）
- 激活的梯度（反向）：`<激活名>.grad`
- 参数：
  - `param.<参数名>`（更新前，原 dtype）；
  - `main_param.<参数名>` / `main_param_after.<参数名>`（FP32 主权重更新前 / 后）；
  - `grad.<参数名>`（进优化器前、已归约的梯度）；
  - `exp_avg.<参数名>` / `exp_avg_sq.<参数名>`（更新后的 Adam 两个动量）；
  - `momentum_buffer.<参数名>`（更新后的 Muon 动量缓冲；Megatron 一侧用 Muon 时才有，走 Muon 的参数没有 `exp_avg*`）；
  - `buffer.<名>` / `buffer_after.<名>`（浮点 buffer，如 expert_bias，更新前 / 后）。
- 输入与 loss：
  - `input.{tokens,labels,loss_mask,position_ids}`（rank 0 第一个 microbatch）；
  - `loss.per_token`（未归约的逐 token loss）。

#### 可选观测点

- 可选观测点（GPU 抓数回调的 `observe`，默认不开）：
  - 运行目录另写 `observe_step<步>.json` 记模块类型与开关实际值。
  - `attn`：`layers.<i>.attn.core.{q,k,v}`（核心注意力输入，RoPE 之后，`[s, b, 头, 每头维度]`）及其 `.grad`
  - `moe`（MoE 层内各阶段，rank 0 视角）：
    - `layers.<i>.mlp.moe.permuted` / `permuted_probs`（分发前置换后的 token 与概率，按专家号排）；
    - `routing_map`、`unpermute_index`、`input_splits`、`output_splits`、`tokens_from_rank_per_local_expert`（置换与 all-to-all 元数据）；
    - `layers.<i>.mlp.experts.{in,tokens_per_expert,probs}`（all-to-all 之后按本地专家排好的专家输入、每专家 token 数、专家侧概率）；
    - `layers.<i>.mlp.experts.{fc1,act}.out`（GEMM1 输出、激活输出 = GEMM2 输入；GEMM2 输出就是 `experts.out`）；
    - `layers.<i>.mlp.moe.combined`（combine 回本 rank、仍按置换顺序）；
    - `layers.<i>.mlp.moe.unpermuted`（unpermute 之后、加共享专家之前）；
    - 专家输入、GEMM1、激活另有 `.grad`。
  - `expw`：本 rank 全部本地专家权重 `param.<参数名>`（全量抓数默认只留每层前两个专家）
  - `gemm`：不写张量，写 `gemm_inventory.json`（第一个抓数步每次 GEMM 调用的输入 / 输出 dtype 与调用处）

#### 名字映射


框架原生名到规范名的映射、以及转置 / 拆分 / 拼接关系，写在 `models/<模型>/model/name_map.yaml`，两侧各一份。

现状：

- GPU 侧激活的映射在 `janus/dump/names.py`。
- 参数类张量（`param.*`、`grad.*` 等）用 Megatron 参数名（如 `grad.decoder.layers.<i>....`）。
- 混元 3 的 `models/hunyuan3/model/name_map.yaml` 一份写全三侧（HF、Megatron、MaxText）与规范名。
  - 解释器是 `janus/dump/name_map.py`；
  - TPU 侧抓数按它把参数类张量换成 Megatron 名字与布局、参数清单按 GPU 的 EP rank 分组；
  - 所以两侧运行目录可以直接按名字比。

## 输入指纹与参数清单

### 输入指纹 data/

- `data/rank<R>.jsonl` 每行：`{"step", "micro", "tokens", "labels", "loss_mask", "position_ids", "nonpad"}`。
  - 四个字段是 sha256，`nonpad` 是 loss_mask 之和。
  - 按 `<步>.r<rank>.m<microbatch>` 配对比较，另比每步全局 `nonpad` 总和。

### 参数清单 params_sha/

- `params_sha/rank<R>.<init|final>.json`：`{参数名: {sha256, shape, dtype}}`。
  - 浮点 buffer 以 `buffer.` 开头一并列入。
  - 比加载后权重、DP 副本一致性都用它，不抽样。
  - TPU 侧为了和 GPU 的 BF16 参数同名同义，参数是 FP32 主权重转成 BF16 后再算的。

### 训练状态清单 state_sha/

- `state_sha/<init|final>.json`：训练前 / 后整个训练状态的清单 `{状态路径: {sha256, shape, dtype, kind}}`。
  - 来源：TPU 侧 `train_hy3.py --state-sha`，`janus/dump/maxtext.py` 的 `write_state_sha`。
  - 状态路径是 MaxText 原生的（`params/model/…`、`MoEBiasVar/model/…`、`OptVariable/optimizer/opt_state/0/{mu,nu}/…`、计数器）。
  - `kind` 是 `param` / `bias` / `mu` / `nu` / `count`。
  - **按存储的 dtype 取原始字节算，不转 BF16、不换名字和布局**。
    - FP32 主权重上不到半个 BF16 格的差，BF16 清单看不见；
    - 日常回归的 R07 / R15 必须用这一份。
  - 不分 rank（每个叶子是全局数组）。

### 编译产物 hlo/

- `hlo/`：生产训练步 `jit_train_step` 的编译产物。
  - 来源：TPU 侧 `train_hy3.py --hlo-dir`，配合 `XLA_FLAGS=--xla_dump_to=…`。
  - `train_step.after_optimizations.txt`：优化后的 HLO，也就是实际执行的那个可执行文件；
  - `train_step.flagfile` 与 `train_step.debug_options`：编译器实际生效的开关。
  - `meta.hlo` 记来源与各文件的 sha256。
  - 同名模块编了不止一次时文件按 `train_step.<i>.*` 编号，`meta.hlo.error` 写明，回归门因此报 `error`。

### meta.json 里日常回归另读的字段

- `meta.json` 里日常回归另读的字段（TPU 侧）：
  - `flags.at_start` / `flags.after_compile`：运行进程启动时、第一步编译执行完各取一份，内容是相关环境变量全量与 `jax.config` 的全部标量取值；
  - `maxtext_config_full`：MaxText 展开后的完整配置；
  - `restarts`：中途重启次数，单进程训练恒为 0。

## 要求

- **抓数不改数值**：同一配置下开钩子与不开钩子，loss 与指定张量逐位相同（`determinism` 检查）。
- **完整性**：`index.json` 里每个张量的 sha256 必须能复算。
  - 缺文件或校验和不符，读入时报 `error`。
  - 同一步里张量名必须唯一：`index.json` 有重名，读入就报 `error`（不静默取第一个）。
  - `RunWriter` 在不同 scope 里写同名张量也直接报错。
- 两侧都按本规范写，janus 不读框架原生的 dump。
