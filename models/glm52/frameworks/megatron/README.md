[janus](../../../../README.md) › [GLM-5.2](../../README.md) ｜ GPU 侧（Megatron-Bridge）

# GLM-5.2 · GPU 侧（Megatron-Bridge）

和[混元 3 的 GPU 侧](../../../hunyuan3/frameworks/megatron/README.md)对称：B200 × 8 单机，Megatron-Bridge 的 GLM-5 桥 + Megatron-Core，按 janus 抓数格式写运行目录。训练程序、截层与往返核对、对 HF 的 FP32 逐层等价检查、Pod 流程随 G0–G3 加进本目录（[`PLAN.md`](../../PLAN.md) 第 2 节）。
