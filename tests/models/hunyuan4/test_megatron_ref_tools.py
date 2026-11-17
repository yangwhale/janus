"""混元 4 Megatron 一侧两个参照工具的接线，CPU 上跑，不要 Megatron-Core。

- 全局参数表：`global_plan` 按 hc_mult 只给本 config 有的末层写法（有 iHC：hc_head + RMSNorm；hc_mult 1：普通
  RMSNorm）；`equiv_hf.grad_compare` 逐参数比梯度时只比这几项，不会去找模型里没有的名字。
- 单层回放（`layer_ref.py`）交给 `step1_ref.build` 的参数取自 `step1_ref` 的命令行默认值：`build`（含扩展的
  `step1_config`）读到的每一项都在里面。
"""

import ast
import inspect
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_MG = Path(__file__).resolve().parents[3] / "models" / "hunyuan4" / "frameworks" / "megatron"
sys.path.insert(0, str(_MG))

import equiv_hf as E  # noqa: E402
import hy4_weights as W  # noqa: E402
import layer_ref as LR  # noqa: E402
import step1_ref as R  # noqa: E402

_IHC_FINAL = {
    "decoder.final_layernorm.norm.weight",
    "decoder.final_layernorm.hc_fn",
    "decoder.final_layernorm.hc_base",
    "decoder.final_layernorm.hc_scale",
}
_COMMON = {"embedding.word_embeddings.weight", "output_layer.weight"}


# ------------------------------------------------------------------ 全局参数表


def test_global_plan_ihc():
    plan = W.global_plan(4)
    assert {m for m, _ in plan} == _COMMON | _IHC_FINAL
    assert dict(plan)["decoder.final_layernorm.norm.weight"] == "model.norm.weight"


def test_global_plan_plain_final_norm():
    plan = W.global_plan(1)
    assert {m for m, _ in plan} == _COMMON | {"decoder.final_layernorm.weight"}
    assert dict(plan)["decoder.final_layernorm.weight"] == "model.norm.weight"


class _Params:
    """只有 named_parameters() 的替身：名字带点，直接给 {名字: 参数}。"""

    def __init__(self, names, seed):
        g = torch.Generator().manual_seed(seed)
        self.p = {}
        for n in names:
            t = torch.nn.Parameter(torch.zeros(3, 2))
            t.grad = torch.randn(3, 2, generator=g)
            self.p[n] = t

    def named_parameters(self):
        return self.p.items()


@pytest.mark.parametrize("hc_mult", [4, 1])
def test_grad_compare_globals_follow_hc_mult(hc_mult):
    plan = W.global_plan(hc_mult)
    mm = _Params([m for m, _ in plan], seed=0)
    hm = _Params(sorted({E.ckpt_to_hf(h) for _, h in plan}), seed=0)
    for m, h in plan:  # 两边梯度相同：比出来全是 0
        hm.p[E.ckpt_to_hf(h)].grad = mm.p[m].grad.clone()
    mcfg = SimpleNamespace(num_layers=0, hc_mult=hc_mult, num_moe_experts=0)
    rows = E.grad_compare(hm, mm, mcfg)
    assert {r["param"] for r in rows} == {m for m, _ in plan}
    assert all(r["rel_l2"] == 0.0 for r in rows)


# ------------------------------------------------------------------ 单层回放交给 step1_ref.build 的参数


def _attrs_read(fn, var="a"):
    """函数体里 `<var>.<名字>` 读到的名字。"""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return {
        n.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == var
    }


def test_build_reads_only_parser_dests():
    import hy4_ext

    dests = set(vars(R.make_parser().parse_args(["--hf", "x", "--out", ""])))
    read = _attrs_read(R.build)
    for f in hy4_ext.hooks("step1_config"):
        read |= _attrs_read(f)
    assert read and read <= dests, sorted(read - dests)


@pytest.mark.parametrize("use_dsa", [True, False])
def test_layer_ref_args_cover_step1_defaults(use_dsa):
    hf_cfg = {"use_dsa": use_dsa}
    args = LR.step1_build_args("hf-dir", 3e-4, hf_cfg)
    defaults = vars(R.make_parser().parse_args(["--hf", "x", "--out", ""]))
    assert set(defaults) <= set(vars(args)), sorted(set(defaults) - set(vars(args)))
    # 单层回放定的几项
    assert (args.layers, args.mtp, args.head_chunk, args.log_max_logit, args.expert_store) == (0, None, 1024, False, "")
    assert args.aux_loss == 3e-4
    # 其余跟 step1_ref 的默认（原来手写的参数里缺了这两项，build 一读就报 AttributeError）
    assert args.rope_theta is None
    assert args.qk_clip_max_logit == defaults["qk_clip_max_logit"]
