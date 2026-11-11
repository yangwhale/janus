[janus](../../README.md) › [GLM-5.2](README.md) ｜ GPU golden 登记

# GLM-5.2 · GPU golden 登记

每一份 GPU golden 是怎么来的、用的哪份代码、跑出来的基本事实。设置的理由见 [`PLAN.md`](PLAN.md) 第 3 节，比较协议见 [`PROTOCOL.md`](PROTOCOL.md)，程序与跑法见 [`frameworks/megatron/README.md`](frameworks/megatron/README.md)。

**现状**：还没有 golden。G0–G3（截层与往返、对 HF 的 FP32 逐层等价、B200 冒烟、确定性检查）完成并把学习率登记进协议之后，才跑阶段 1 的 `g1-a … g1-h` 与 `g1s-a … g1s-c`。
