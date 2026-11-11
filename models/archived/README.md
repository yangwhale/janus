[janus](../../README.md) › 归档

# 归档的模型实践

现在没有归档的实践。

| 文件夹 | 状态 |
|---|---|
| （空） | 混元 4 于 2026-11-07 归档、同日恢复到 [`models/hunyuan4/`](../hunyuan4/README.md)（项目负责人改为反向方案，见 [`docs/dev/CHANGES-PATHS.md`](../../docs/dev/CHANGES-PATHS.md)） |

GLM-5.2（[`models/glm52/`](../glm52/README.md)）暂停但没有归档，是否归档项目负责人还没定。它的 GPU golden 已经完整：阶段 1 的 8 份 300 步 golden、3 份第 1 步全量抓数和 28 对噪声带，登记在 GLM-5.2 的开发分支 `feat/69-glm52-alignment` 上。

归档的规矩：文件夹整个移进 `models/archived/<模型>/`，内部结构不变；结果文件原样保留（证据不改），文中的仓库内路径按新位置改；在 `docs/dev/CHANGES-PATHS.md` 记一笔。
