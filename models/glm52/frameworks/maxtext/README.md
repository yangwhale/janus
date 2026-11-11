[janus](../../../../README.md) › [GLM-5.2](../../README.md) ｜ TPU 侧（MaxText）

# GLM-5.2 · TPU 侧（MaxText）

和[混元 3 的 TPU 侧](../../../hunyuan3/frameworks/maxtext/README.md)对称：v5p 8 片，上游 MaxText 的 DeepSeek 解码器接入代码 + GLM-5.2 模型配置。走的不是 maxtext 的混元 4 集成分支，那条分支上的对齐开关这里没有。现在只放编译期显存分析用的配置；训练、数据回放与抓数等 GPU 参照跑通后加。
