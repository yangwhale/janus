# tests

```bash
pip install -e '.[test]'
make test            # 等于 python -m pytest tests
make docs-check      # 只跑文档与仓库卫生（tests/docs/）
```

CI（`.github/workflows/tests.yml`）在每个 PR 上跑 `make lint` 与 `make test`，Python 3.10 与 3.12 各一遍。全部只要 CPU：
要 torch / jax 的测试在没装时自动跳过。

## 目录照包的结构

| 目录 | 测什么 |
|---|---|
| `tests/test_<模块>.py` | `janus/` 顶层模块：`cli`、`run`、`catalog`、`validate`、`store`、`metrics`、`dtypes`、`replay`、`demo`、`data_layout`；`test_run_full.py` 是整条链的端到端；`test_audit_fixes.py` 是复现审计记录的问题，每条一个回归测试；`test_compat.py` 是旧导入路径的转发；`test_imports.py` 守「引擎不导入 torch / jax / matplotlib」 |
| `tests/checks/` | `janus/checks/` 每个模块一份：`test_base.py`、`test_tensor.py`、`test_discrete.py`、`test_meta.py`、`test_series.py`、`test_extra.py`、`test_ops.py` |
| `tests/dump/` | `janus/dump/`：`test_names.py`、`test_name_map.py`、`test_runinfo.py`、`test_maxtext.py`（不依赖 jax 的部分）、`test_megatron.py`（没有 torch 时跳过） |
| `tests/analysis/` | `janus/analysis/`：`test_ops.py`（CPU 参照）、`test_drift.py`、`test_loss.py`、`test_router.py`、`test_serial.py`、`test_curves.py`、`test_offsets.py`、`test_fault_matrix.py` |
| `tests/report/` | `janus/report/`：`test_tables.py`（逐级大表）、`test_figures.py`、`test_html.py`（门槛总表；整份 HTML 报告由 `test_run_full.py` 端到端跑） |
| `tests/models/<模型>/` | 模型层脚本：混元 3 的日常回归目录与 `regress.py`（`test_regression.py`）、原生加载器对拍（`test_native_loader_parity.py`） |
| `tests/docs/` | 文档断链（`test_links.py`）、「某文件「某节」」引用的节还在（`test_section_refs.py`）、仓库卫生（`test_repo_hygiene.py`：内部链接、真实桶名、模块说明） |

共用的造数据函数在 `tests/helpers.py`（`from helpers import make, catalog, …`），不要从别的测试模块导入。
pytest 用 importlib 模式（`pyproject.toml`），子目录不需要 `__init__.py`，不同目录里可以有同名的测试文件。

## 规矩

- **比较工具的测试必须包含「故意改坏的数据会被发现」**（工具先自证，见 `CONTRIBUTING.md`）。
- 新加的检查类型、指标、子命令都要有测试；修 bug 先写一个能复现它的测试。
- 测试只用 `tmp_path` 里现造的小数据，不读运行目录、不联网。仓库里已提交的配置（目录、`name_map.yaml`、分组 YAML）可以读。
