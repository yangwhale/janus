# janus 的常用命令。全部只要 CPU；`make help` 列出目标。
# 约定：CI（.github/workflows/tests.yml）跑的就是 `make lint` 与 `make test`，本地过了 CI 就过。

PYTHON ?= python3
RUFF   ?= $(PYTHON) -m ruff
PYTEST ?= $(PYTHON) -m pytest

# 强制 ruff format 的路径。老代码按自己的对齐注释风格写，不批量重排（理由见
# docs/dev/design/0001-repo-standards.md「lint 与格式」）；新模块、整体重写过的模块加进来。
FMT_PATHS ?= janus/__init__.py janus/__main__.py janus/cli.py janus/demo.py janus/validate.py \
             janus/analysis/curves.py janus/analysis/offsets.py janus/analysis/fault_matrix.py \
             janus/report/__init__.py janus/report/gates.py janus/report/inventory.py janus/report/overview.py janus/report_html.py janus/alignment.py janus/figures.py \
             tests/test_cli.py tests/test_committed_results.py tests/test_compat.py tests/test_reference_stats.py tests/test_demo.py tests/test_imports.py tests/test_validate.py tests/test_replay.py tests/test_dtypes.py \
             tests/analysis/test_drift.py tests/analysis/test_loss.py tests/analysis/test_serial.py tests/analysis/test_router.py tests/report/test_figures.py tests/report/test_gates.py tests/report/test_html.py tests/report/test_inventory.py tests/dump/test_runinfo.py tests/docs/test_links.py tests/docs/test_repo_hygiene.py tests/docs/test_section_refs.py tests/models/hunyuan3/test_tpu_submit.py \
             models/hunyuan3/analysis/pair_summary.py models/hunyuan3/analysis/remat_router_check.py models/hunyuan3/analysis/dgrad_accum_ablation.py \
             models/hunyuan3/frameworks/megatron/routing_replay.py tools/migrate_catalog_content_hash.py \
             models/hunyuan4/analysis/sizing.py models/hunyuan4/analysis/optim_replay_summary.py models/hunyuan4/analysis/l1_mini_site_compare.py models/hunyuan4/analysis/switch_isolation.py models/hunyuan4/analysis/l2_switch_compare.py models/hunyuan4/analysis/weight_override_refs.py models/hunyuan4/analysis/sink_probe_compare.py models/hunyuan4/analysis/hlo_probe_compare.py models/hunyuan4/analysis/w_combo_compare.py models/hunyuan4/analysis/forced_replay_compare.py models/hunyuan4/analysis/routing_agreement.py models/hunyuan4/analysis/l4_vs_all_goldens.py models/hunyuan4/frameworks/maxtext/build_forced_routing.py tests/models/hunyuan4/test_forced_routing.py models/hunyuan4/dashboard/build.py tests/models/hunyuan4/test_sizing.py \
             models/glm52/analysis/param_counts.py \
             models/hunyuan4/frameworks/megatron/hy4_mcore.py models/hunyuan4/frameworks/megatron/hy4_weights.py models/hunyuan4/frameworks/megatron/hy4_bridge.py \
             models/hunyuan4/frameworks/megatron/equiv_hf.py models/hunyuan4/frameworks/megatron/bridge_check.py models/hunyuan4/frameworks/megatron/train_hy4mg.py models/hunyuan4/frameworks/megatron/hy4_capture.py models/hunyuan4/analysis/fp64_l1_gpu.py models/hunyuan4/analysis/qk_clip_nonpositive_repro.py models/hunyuan4/analysis/sigmoid_silu_probe.py models/hunyuan4/analysis/ihc_sum_order_probe.py models/hunyuan4/analysis/rmsnorm_dgamma_probe.py \
             models/hunyuan4/analysis/attn_scores_fp32_unit.py models/hunyuan4/analysis/ihc_param_grad_conditioning.py models/hunyuan4/analysis/step1_train_grad_file_check.py models/hunyuan4/analysis/tpu_grad_file_holes.py \
             models/hunyuan3/analysis/rebaseline_sensitivity.py models/hunyuan3/analysis/tpu_vs_rc_loss.py \
             models/hunyuan4/frameworks/megatron/indexer_kd_check.py models/hunyuan4/frameworks/megatron/export_hf_grads.py \
             models/hunyuan4/frameworks/megatron/step1_ref.py models/hunyuan4/analysis/step1_vs_fp64.py \
             models/hunyuan4/frameworks/megatron/layer_ref.py models/hunyuan4/analysis/l2_vs_fp64.py models/hunyuan4/analysis/l1_kd_qkclip_vs_fp64.py models/hunyuan4/analysis/step1_inject_vs_fp64.py models/hunyuan4/analysis/gpu_scale_compare.py models/hunyuan4/analysis/noise_leave_one_run.py models/hunyuan4/alignments/05-gpu-scale/figures/plot.py models/hunyuan4/analysis/step1_loss_vs_fp64.py \
             models/hunyuan4/analysis/step1_base2_emulation.py models/hunyuan4/frameworks/maxtext/splash_base2_probe.py \
             models/hunyuan4/analysis/emb_scatter_sort_probe.py models/hunyuan4/analysis/step1_base2_ab_vs_fp64.py models/hunyuan4/analysis/rw3_base2_replay.py models/hunyuan4/analysis/mini_base2_loo.py models/hunyuan4/analysis/exec_spread_300.py models/hunyuan4/analysis/export_train_grads.py models/hunyuan4/analysis/derive_max_logit.py \
             models/hunyuan4/analysis/kd_input_rounding.py models/hunyuan4/analysis/kd_bf16_vs_fp64.py models/hunyuan4/analysis/kd_gpu_dump_vs_fp64.py models/hunyuan4/analysis/kd_cpu16_vs_gpu_dump.py models/hunyuan4/analysis/kd_noise_scaling.py models/hunyuan4/analysis/kpe_round_emul.py models/hunyuan4/analysis/kd_gpu_bias_summary.py \
             models/hunyuan4/analysis/mla_rope16_emul.py models/hunyuan4/analysis/round_sites_emul.py models/hunyuan4/analysis/main_loss_sites_summary.py models/hunyuan4/analysis/same_state_grads.py models/hunyuan4/analysis/emb_grad_row_groups.py \
             models/hunyuan4/alignments/04-tpu-self/hlo_screen.py models/hunyuan4/frameworks/maxtext/aot_hlo.py tests/models/hunyuan4/test_hlo_screen.py models/hunyuan4/analysis/refresh_program_fingerprint.py \
             models/hunyuan4/alignments/04-tpu-self/hlo_screen_libtpu.py models/hunyuan4/analysis/llo_bf16_alu.py models/hunyuan4/analysis/qkclip_accum_compare.py models/hunyuan4/analysis/prodrecipe_settings_check.py models/hunyuan4/analysis/gpu_meta_versions_check.py models/hunyuan4/analysis/step1_metrics_vs_ref.py models/hunyuan4/analysis/max_logit_summary.py models/hunyuan4/analysis/exec_spread_drop_items.py models/hunyuan4/analysis/meta_settings_diff.py models/hunyuan4/analysis/prodrecipe_ruler_compare.py models/hunyuan4/analysis/hc_head_dgrad_fusion.py models/hunyuan4/analysis/jit_miscompile_offline_scan.py models/hunyuan4/analysis/tpu_self_step1_rejudge_n6.py

.PHONY: help install dev test test-fast lint fmt docs-check check dist dashboard dashboard-check clean

help:            ## 列出目标
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-11s %s\n", $$1, $$2}'

install:         ## 装核心包（numpy、ml_dtypes、pyyaml）
	$(PYTHON) -m pip install -e .

dev:             ## 装开发依赖：报告画图、pytest、ruff
	$(PYTHON) -m pip install -e '.[dev]'

test:            ## 全部单元测试（含断链与仓库卫生检查），不要 GPU / TPU
	$(PYTEST) tests

test-fast:       ## 第一个失败就停
	$(PYTEST) -x --ff tests

lint:            ## ruff check 全仓库 + FMT_PATHS 的格式检查
	$(RUFF) check .
	$(RUFF) format --check $(FMT_PATHS)

fmt:             ## 自动修：ruff check --fix 全仓库，ruff format FMT_PATHS
	$(RUFF) check --fix .
	$(RUFF) format $(FMT_PATHS)

docs-check:      ## 文档：相对链接都指到存在的文件、#锚点指到真实标题；「文件「节」」引用还在；没有内部链接与真实桶名
	$(PYTEST) tests/docs

check: lint test ## 提 PR 前跑这一个

dist:            ## 打 wheel 到 dist/（CI 的 package job 装它、在仓库外跑 demo）
	$(PYTHON) -m pip wheel --no-deps -w dist .

dashboard:       ## 重画混元 3 的检查清单与跨实践总览、混元 4 各尺寸的检查清单与总览、各实践 README 里的逐级大表与配图（要 matplotlib）
	$(PYTHON) models/hunyuan3/dashboard/build.py
	$(PYTHON) models/hunyuan4/dashboard/build.py

dashboard-check: ## 重画后 .md / .html 与提交的一致（CI 的 generated job）。PNG 随机器字体变，不比
	$(PYTHON) models/hunyuan3/dashboard/build.py >/dev/null
	$(PYTHON) models/hunyuan4/dashboard/build.py >/dev/null
	git diff --stat --exit-code -- ':(glob)models/*/dashboard/*.md' ':(glob)models/*/dashboard/*.html' ':(glob)models/*/alignments/*/README.md' ':(glob)models/*/CHECKS.md' \
	  ':(glob)models/*/alignments/*/CHECKS.md' ':(glob)models/*/alignments/*/dashboard/*.md' ':(glob)models/*/alignments/*/dashboard/*.html' ':(glob)models/*/alignments/*/*/README.md'

clean:           ## 删缓存与构建产物
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
