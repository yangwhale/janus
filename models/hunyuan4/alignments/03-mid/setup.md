[janus](../../../../README.md) › [混元 4](../../README.md) › [mid](README.md) ｜ 设置

# 混元 4 · 03 · mid（hy4-mid）· 设置

只写和 T1（[`../01-mini/setup.md`](../01-mini/setup.md)）不一样的地方。

| 项 | TPU（被测） | GPU（golden） |
|---|---|---|
| 模型 | MaxText `hy4-mid`（集成分支；与 `hy4-mini` 只差宽度、头数、q_lora、稠密 FFN、专家宽度） | 同一 HF 目录 |
| 初始权重 | `hf/hy4-mid-seed0`（`model/make_tiny_hf.py --config mid --layers 7 --mtp 1 --seed 0`，权重矩阵标准差 0.006）转换的 MaxText 检查点 `maxtext/hy4-mid-maxtext/0/items` | `hf/hy4-mid-seed0` |
| 配置 | [`hy4-mid-t1.yml`](../../frameworks/maxtext/hy4-mid-t1.yml)：`hy4-mini-t1.yml` 的全部开关，只换 `model_name`；`STAGE=mid` | 参数同 g2 / g3，只换 HF 目录 |
| 数据 | 回放 `gm-a` 的 batch 与 loss_mask | `tokens.npy`（与 T1 同一份） |

路径都在 `gs://<POC 桶>/janus/hunyuan4/` 下。TPU 命令：`YML=hy4-mid-t1.yml STAGE=mid GOLDEN=<gm-a 运行目录> HF_DIR=<…/hf/hy4-mid-seed0> bash run.sh tm-a 300 --dump-steps none`。复现 `tm-a` 要用 janus `fd796f6` 的 yml 与 MaxText `469dbb85`：之后加的 M39（MaxText 代码）与 M40、M41（yml 开关）都会改变数值。

## GPU golden 的作业规格

程序与 Pod 环境同 T2（[`../02-real-weight/setup.md`](../02-real-weight/setup.md)「GPU golden 的作业规格」），两边共用的附加参数 `C` 是：

```text
C=--ep=4,--optimizer=muon,--opt-kv=muon_scale_mode=shape_scaling,--qk-clip,--qk-clip-threshold=100,--wd=0,--lr=1e-5,--adam-eps=1e-10,--adam-beta2=0.95,--attn-scores-fp32=on,--head-chunk=1024,--indexer-loss=0.1,--eod-mask=on,--per-token-loss=on,--muon-exclude-ihc=ihc+head,--muon-gate=per_head,--bias-rate=1e-3,--clip=1.0,--seq=4096
```

相对 T2 只换：`HFDIR=hy4-mid-seed0`、`HF_FILES="config.json model.safetensors.index.json tensor_sha256.json model.safetensors"`，层数 7、带 `--mtp`、`--qk-clip-mtp=on`；可以带 `--dump-all-experts`（16 个专家）。

```text
gm-a:7:300:--mtp,$C,--qk-clip-mtp=on,--recompute,--save-batches,--side=golden-gm-a      （gm-b … gm-h 同，不带 --save-batches）
gms-a:7:1:--mtp,$C,--qk-clip-mtp=on,--dump-steps=1,--full,--dump-all-experts,--save-batches,--side=golden-gms-a   （gms-b、gms-c 同，不带 --save-batches）
```

跑完放到 `gs://<POC 桶>/janus/hunyuan4/golden-mid/<运行名>/`，写 `DONE`。

显存：v5p 编译期每设备参数 + 临时量 63.16 GB；B200 每卡约 86.7 GiB（估算）。见 [`../02-real-weight/results/hy4-t2-aot-memory.json`](../02-real-weight/results/hy4-t2-aot-memory.json)。
