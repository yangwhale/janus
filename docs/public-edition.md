[janus](../README.md) › 关于公开版

# 关于公开版

这个仓库是 janus 的公开版，由开发仓库定期同步生成。公开版不带开发历史：每个提交对应开发仓库的一次同步，提交说明里写着对应的开发仓库提交号。

## 文中提到、但没有公开的东西

- **MaxText 的修改。** 文中的 `maxtext` 指我们修改过的 MaxText，包括 `hunyuan` 分支（混元 3）和混元 4 集成分支。这些修改目前没有公开。
  - 文档和配置里写的开关名（如 `dense_fp32_accumulation`、`rmsnorm_single_rounding`、`embedding_grad_fp32`）、文件路径与提交号（如 `hunyuan@0622774`）都指这份修改后的代码，原样保留作出处。
  - 结果文件 `meta.json`、`results.json`、`runs.yaml` 里 `code`、`commits` 记的 maxtext 提交号也一样。
  - 用上游 MaxText 跑不出这些开关。方法、判法、尺子与上游无关，照样可以用。
- **janus 的开发历史。** 结果文件和文档里记的 janus 提交号指开发仓库的提交，公开版里查不到，保留作出处。
- **问题跟踪记录。** 文中「一次实际排查」「项目里的排查」等说法，指开发过程中的问题跟踪记录，没有公开。相关的技术结论都已写进正文。

## 换成你自己的环境

- 桶名一律写成 `gs://<POC 桶>/…` 占位，换成你自己的存储桶。
- 集群的 kubectl 上下文、命名空间、节点池、队列等提交设置不在仓库里，放在仓库外的站点配置（`JANUS_SITE_PROFILE`），照 `models/*/frameworks/maxtext/site-profile.example.sh` 写一份，见[准备 TPU 环境](guide/03-tpu-setup.md)。
- GPU 一侧的 Pod / JobSet 模板里，节点池等写成 `<…>` 的位置换成你自己的值。

## 许可

见 [`LICENSE`](../LICENSE)。
