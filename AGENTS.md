# AGENTS.md —— 给在本仓库里干活的 AI 编码助手

人类读者可以跳过这份文件，开发流程见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

1. 先读 [`README.md`](README.md) 和 [`CONTRIBUTING.md`](CONTRIBUTING.md)。改检查的判法、结果格式、命令行之前，先读 `docs/reference/` 下对应的规格。
2. 数字要有来源。写进文档的数字标明出自哪个结果文件的哪条检查、或哪次运行。推出来的写「推断」并给推导，查不到的写「未确认」。不编数字、运行名、文件路径。
3. 结果里的数字不手写：`report.md` 从 `results.json` 渲染，总览与逐级大表用 `make dashboard` 重画。
4. 已提交的 `results/`、`figures/`、`faults/`、`cases/` 是证据，不改、不删。发现错了，追加更正并写明原因。
5. 门槛不能比噪声严。比较工具上线前，先喂一份故意改坏的数据，确认会报警。
6. 不往仓库里写密钥、令牌、真实桶名（写成 `gs://<POC 桶>/…` 占位）、本机绝对路径、任何公司的内部链接。`tests/docs/test_repo_hygiene.py` 会拦住常见的写法。
7. 提交前跑 `make check`。提交信息第一行写 `<type>: <简述>`，正文用 `git commit -F <文件>` 给出，不写在命令行里。
