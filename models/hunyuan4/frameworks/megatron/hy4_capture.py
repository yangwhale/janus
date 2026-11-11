"""混元 4 GPU 一侧的逐算子 / 逐层抓数（L1 / L2）：第 N 步 rank 0 第 0 个 microbatch，Megatron-Core + TE（BF16）。

抓的是算子的输入与输出（前向）和每层输入 / 输出的梯度（反向），名字与布局照 HF 一侧 golden 的抓数
（`hf.model.layers.<L>.<子模块>.<量>`，激活 [b, s, …]，4 路流 [b, s, n, H]），这样 `analysis/fp64_ops.py`
能直接拿来在 FP64 里重算同一个算子（同一份输入），TPU 一侧的 `frameworks/maxtext/op_replay.py` 也认同一套名字。

写出（运行目录下 `l1/`）：
- `step_<N>_fwd.safetensors`：前向张量（原 dtype）；
- `step_<N>_bwd.safetensors`：梯度（`<名>.grad`）；
- `step_<N>_index.json`：每个张量的名字、形状、dtype、来自哪个 Megatron 模块与哪个口子（输入 / 输出 / 梯度）、布局；
- `step_<N>_grad_micro0.safetensors`（`grad_micro0=True` 时）：第 0 个 microbatch 反向做完时本 rank 各参数的 main_grad
  （Megatron 名字，FP32），即这一行单独的参数梯度，给逐层回放比参数梯度用。
iHC 两处混合的输出（分支输入）另抓梯度 `<层>.attn_hc.out.grad` / `.ffn_hc.out.grad`。

只在不开整层重算时用（重算会让前向钩子在反向里再触发一次）。格式说明另见 `README.md`「逐算子与逐层抓数」。
"""

from __future__ import annotations

import json
import os
import re

import torch

LP = r"^(decoder\.layers\.(\d+)|mtp\.layers\.0\.mtp_model_layer)"


def _hf_prefix(mname: str) -> tuple[str, str] | None:
    """Megatron 模块名 → (HF 前缀, 层内后缀)。"""
    m = re.match(LP + r"(?:\.(.*))?$", mname)
    if not m:
        return None
    pre = f"hf.model.layers.{m.group(2)}" if m.group(2) is not None else "hf.model.mtp_layers.0"
    return pre, m.group(3) or ""


def _first(x):
    return x[0] if isinstance(x, (tuple, list)) else x


def _sb_to_bs(t: torch.Tensor) -> torch.Tensor:
    """Megatron 的 [s, b, …] → HF 的 [b, s, …]。"""
    return t.transpose(0, 1) if t.dim() >= 2 else t


class Hy4Capture:
    def __init__(self, model_core, config, rundir: str, step: int, grad_micro0: bool = False):
        self.core, self.cfg, self.step = model_core, config, step
        self.grad_micro0 = grad_micro0  # 另存第 0 个 microbatch 之后各参数的 main_grad（逐层回放的参数梯度对照）
        self.bwd0_seen, self.grads0 = False, None
        self.out = os.path.join(rundir, "l1")
        self.fwd: dict[str, torch.Tensor] = {}
        self.bwd: dict[str, torch.Tensor] = {}
        self.meta: dict[str, dict] = {}
        self.active = False
        self.handles = []
        self.head_calls = 0
        self._install()

    # ---- 记录
    def _put(self, name, t, src, port, layout, bwd=False):
        # 前向只在第 0 个 microbatch 内记；反向钩子挂在那次的张量上，由 active_bwd 控制（在 _grad_hook 里判）
        if (not bwd and not self.active) or t is None or not isinstance(t, torch.Tensor):
            return
        store = self.bwd if bwd else self.fwd
        if name in store:  # 每个名字只记第一次（第 0 个 microbatch）
            return
        store[name] = t.detach().to("cpu", copy=True).contiguous()
        self.meta[name] = {
            "src": src,
            "port": port,
            "layout": layout,
            "shape": list(store[name].shape),
            "dtype": str(store[name].dtype).replace("torch.", ""),
        }

    def _act(self, name, t, src, port, stream=False):
        """激活：[s, b, …] → [b, s, …]；4 路流 [s, b, n·H] → [b, s, n, H]。"""
        if not isinstance(t, torch.Tensor):
            return
        n, H = self.cfg.hc_mult, self.cfg.hidden_size
        if stream and t.dim() == 3 and t.shape[-1] == n * H:
            t = t.view(t.shape[0], t.shape[1], n, H)
            self._put(name, _sb_to_bs(t), src, port, "b,s,n,H")
        else:
            self._put(name, _sb_to_bs(t), src, port, "b,s,…")

    def _snap_main_grads(self):
        """本 rank 每个参数的 main_grad（FP32）此刻的值：只含第 0 个 microbatch 的贡献（梯度累积还没走到第 1 个，DP 归约
        在最后一个 microbatch 之后才做）。路由专家只有本 rank 的。"""
        self.grads0 = {}
        for n, prm in self.core.named_parameters():
            g = getattr(prm, "main_grad", None)
            if g is not None:
                self.grads0[n] = g.detach().to("cpu", dtype=torch.float32, copy=True).contiguous()

    def _grad_hook(self, name, src, stream=False):
        def hook(g):
            if name == "hf.model.embed_tokens.out" and self.active_bwd:
                self.bwd0_seen = True
            if self.active_bwd:
                n, H = self.cfg.hc_mult, self.cfg.hidden_size
                gg = g.view(g.shape[0], g.shape[1], n, H) if stream and g.dim() == 3 and g.shape[-1] == n * H else g
                self._put(name + ".grad", _sb_to_bs(gg), src, "grad", "b,s,n,H" if gg.dim() == 4 else "b,s,…", bwd=True)

        return hook

    # ---- 钩子
    def _install(self):
        self.active_bwd = False
        reg = self.handles.append
        for mname, mod in self.core.named_modules():
            if mname == "embedding":

                def emb_pre(mod, args, kwargs, mname=mname):
                    ids = kwargs.get("input_ids", args[0] if args else None)
                    self._put("input.tokens", ids, mname, "input", "b,s")
                    if self.grad_micro0 and self.bwd0_seen and self.grads0 is None:
                        self._snap_main_grads()  # 第 1 个 microbatch 的前向开始：第 0 个的反向已经全部做完

                def emb_fwd(mod, args, kwargs, out, mname=mname):
                    self._act("hf.model.embed_tokens.out", out, mname, "output")
                    if self.active and out.requires_grad:
                        out.register_hook(self._grad_hook("hf.model.embed_tokens.out", mname))

                reg(mod.register_forward_pre_hook(emb_pre, with_kwargs=True))
                reg(mod.register_forward_hook(emb_fwd, with_kwargs=True))
                continue
            if mname == "decoder.final_layernorm.norm":
                reg(
                    mod.register_forward_pre_hook(
                        lambda m, a, n=mname: self._act("hf.model.hc_head.out", a[0], n, "input")
                    )
                )
                reg(mod.register_forward_hook(lambda m, a, o, n=mname: self._act("hf.model.norm.out", o, n, "output")))
                continue
            if mname == "output_layer":

                def head_fwd(mod, args, out, mname=mname):
                    if not self.active:
                        return
                    self.head_calls += 1
                    # 带 MTP 时输出头先被 MTP 调一次、再被主干调一次（Megatron-Core GPTModel 的顺序）
                    tag = (
                        "hf.lm_head.out"
                        if (not self.cfg.mtp_num_layers or self.head_calls == 2)
                        else "hf.mtp.lm_head.out"
                    )
                    self._act(tag, _first(out), mname, "output")

                reg(mod.register_forward_hook(head_fwd))
                continue
            mm = re.match(r"^mtp\.layers\.0\.(enorm|hnorm|eh_proj|final_layernorm)$", mname)
            if mm:
                hn = f"hf.model.mtp_layers.0.{mm.group(1)}"
                reg(
                    mod.register_forward_pre_hook(
                        lambda m, a, hn=hn, n=mname: self._act(hn + ".in", a[0] if a else None, n, "input")
                    )
                )

                which_ = mm.group(1)

                def mtp_fwd(m, a, o, hn=hn, n=mname, which=which_):
                    o = _first(o)
                    self._act(hn + ".out", o, n, "output")
                    if which == "eh_proj" and self.active and isinstance(o, torch.Tensor) and o.requires_grad:
                        o.register_hook(self._grad_hook(hn + ".out", n))  # = MTP 层输入的梯度

                reg(mod.register_forward_hook(mtp_fwd))
                continue
            hp = _hf_prefix(mname)
            if hp is None:
                continue
            pre, sub = hp
            if sub == "":  # 整层：输入 / 输出（4 路流）与两者的梯度

                def layer_pre(m, args, kwargs, pre=pre, n=mname):
                    x = kwargs.get("hidden_states", args[0] if args else None)
                    self._act(pre + ".in0", x, n, "input", stream=True)
                    if self.active and isinstance(x, torch.Tensor) and x.requires_grad:
                        x.register_hook(self._grad_hook(pre + ".in0", n, stream=True))

                def layer_fwd(m, args, kwargs, out, pre=pre, n=mname):
                    o = _first(out)
                    self._act(pre + ".out0", o, n, "output", stream=True)
                    if self.active and isinstance(o, torch.Tensor) and o.requires_grad:
                        o.register_hook(self._grad_hook(pre + ".out0", n, stream=True))

                reg(mod.register_forward_pre_hook(layer_pre, with_kwargs=True))
                reg(mod.register_forward_hook(layer_fwd, with_kwargs=True))
            elif sub in ("attn_hc", "ffn_hc"):

                def hc_fwd(m, a, o, pre=pre, sub=sub, n=mname):
                    self._act(f"{pre}.{sub}.in", a[0], n, "input", stream=True)
                    self._act(f"{pre}.{sub}.post", o[0], n, "output")
                    self._act(f"{pre}.{sub}.out", o[1], n, "output")
                    if self.active and isinstance(o[1], torch.Tensor) and o[1].requires_grad:
                        o[1].register_hook(self._grad_hook(f"{pre}.{sub}.out", n))  # 分支输入（iHC 混合输出）的梯度

                reg(mod.register_forward_hook(hc_fwd))
            elif sub in ("input_layernorm", "pre_mlp_layernorm"):
                hn = "input_layernorm" if sub == "input_layernorm" else "post_attention_layernorm"
                reg(
                    mod.register_forward_hook(
                        lambda m, a, o, pre=pre, hn=hn, n=mname: self._act(f"{pre}.{hn}.out", o, n, "output")
                    )
                )
            elif sub == "self_attention":
                reg(
                    mod.register_forward_hook(
                        lambda m, a, o, pre=pre, n=mname: self._act(f"{pre}.self_attn.out", _first(o), n, "output")
                    )
                )
            elif sub == "self_attention.linear_gate":
                reg(
                    mod.register_forward_hook(
                        lambda m, a, o, pre=pre, n=mname: self._act(
                            f"{pre}.self_attn.gate_proj.out", _first(o), n, "output"
                        )
                    )
                )
            elif sub == "self_attention.core_attention":
                mod.capture = True  # last_topk

                def core_pre(m, args, kwargs, pre=pre, n=mname):
                    names = ("query", "key", "value", "attention_mask", "x", "qr")
                    for i, nm in enumerate(names):
                        t = kwargs.get(nm, args[i] if i < len(args) else None)
                        if nm in ("query", "key", "value", "x", "qr"):
                            self._act(f"{pre}.self_attn.core.{nm}", t, n, "input")

                def core_fwd(m, args, kwargs, out, pre=pre, n=mname):
                    self._act(f"{pre}.self_attn.core.out", _first(out), n, "output")
                    if self.active and getattr(m, "last_topk", None) is not None:
                        self._put(f"{pre}.self_attn.topk", m.last_topk, n, "output", "b,s,k")

                reg(mod.register_forward_pre_hook(core_pre, with_kwargs=True))
                reg(mod.register_forward_hook(core_fwd, with_kwargs=True))
            elif sub == "mlp":
                reg(
                    mod.register_forward_hook(
                        lambda m, a, o, pre=pre, n=mname: self._act(f"{pre}.mlp.out", _first(o), n, "output")
                    )
                )
            elif sub in ("mlp.linear_fc1", "mlp.shared_experts.linear_fc1", "mlp.experts.linear_fc1"):
                tag = {
                    "mlp.linear_fc1": "mlp.fc1.out",
                    "mlp.shared_experts.linear_fc1": "mlp.shared_experts.fc1.out",
                    "mlp.experts.linear_fc1": "mlp.experts.fc1.out",
                }[sub]
                reg(
                    mod.register_forward_hook(
                        lambda m, a, o, pre=pre, tag=tag, n=mname: self._put(
                            f"{pre}.{tag}", _first(o), n, "output", "tokens,…" if "experts." in tag else "s,b,…"
                        )
                    )
                )
            elif sub in ("mlp.linear_fc2", "mlp.shared_experts.linear_fc2", "mlp.experts.linear_fc2"):
                tag = {
                    "mlp.linear_fc2": "mlp.fc2.in",
                    "mlp.shared_experts.linear_fc2": "mlp.shared_experts.fc2.in",
                    "mlp.experts.linear_fc2": "mlp.experts.fc2.in",
                }[sub]
                reg(
                    mod.register_forward_pre_hook(
                        lambda m, a, pre=pre, tag=tag, n=mname: self._put(
                            f"{pre}.{tag}", a[0], n, "input", "tokens,…" if "experts." in tag else "s,b,…"
                        )
                    )
                )
            elif sub == "mlp.shared_experts":

                def shared_fwd(m, a, o, pre=pre, n=mname):  # 钩子必须返回 None，否则会替换模块输出
                    self._put(f"{pre}.mlp.shared_experts.in", a[0], n, "input", "s,b,…")
                    self._put(f"{pre}.mlp.shared_experts.out", _first(o), n, "output", "s,b,…")

                reg(mod.register_forward_hook(shared_fwd))
            elif sub == "mlp.router":
                orig = mod.gating

                def gating(x, orig=orig, pre=pre, n=mname):
                    out = orig(x)
                    self._put(f"{pre}.mlp.gate.in", x, n, "input", "tokens,H")
                    self._put(f"{pre}.mlp.gate.logits", out, n, "output", "tokens,E")
                    return out

                mod.gating = gating

                def router_fwd(m, a, o, pre=pre, n=mname):
                    self._put(f"{pre}.mlp.gate.probs", o[0], n, "output", "tokens,E（选中的专家处为路由权重，其余 0）")
                    self._put(f"{pre}.mlp.gate.routing_map", o[1], n, "output", "tokens,E（bool）")
                    eb = getattr(m, "expert_bias", None)
                    self._put(f"{pre}.mlp.gate.expert_bias", eb, n, "buffer", "E")

                reg(mod.register_forward_hook(router_fwd))
            elif sub == "mlp.experts":

                def ex_pre(m, args, kwargs, pre=pre, n=mname):
                    names = ("permuted_local_hidden_states", "tokens_per_expert", "permuted_probs")
                    for i, nm in enumerate(names):
                        t = kwargs.get(nm, args[i] if i < len(args) else None)
                        self._put(f"{pre}.mlp.experts.{nm}", t, n, "input", "tokens,…")

                reg(mod.register_forward_pre_hook(ex_pre, with_kwargs=True))
                reg(
                    mod.register_forward_hook(
                        lambda m, a, o, pre=pre, n=mname: self._put(
                            f"{pre}.mlp.experts.out",
                            _first(o),
                            n,
                            "output",
                            "tokens,H（本 rank 收到的、按专家排好的 token）",
                        )
                    )
                )

    # ---- 开关与写出
    def begin(self):
        self.active, self.active_bwd = True, True
        self.bwd0_seen, self.grads0 = False, None

    def end_forward(self):
        self.active = False  # 只抓第 0 个 microbatch 的前向；反向的钩子挂在那一次的张量上

    def finish(self, extra: dict | None = None):
        from safetensors.torch import save_file

        self.active_bwd = False
        for h in self.handles:
            h.remove()
        os.makedirs(self.out, exist_ok=True)
        save_file(self.fwd, os.path.join(self.out, f"step_{self.step}_fwd.safetensors"))
        save_file(self.bwd, os.path.join(self.out, f"step_{self.step}_bwd.safetensors"))
        if self.grads0:
            save_file(self.grads0, os.path.join(self.out, f"step_{self.step}_grad_micro0.safetensors"))
        with open(os.path.join(self.out, f"step_{self.step}_index.json"), "w") as f:
            json.dump(
                {
                    "step": self.step,
                    "rank": 0,
                    "micro": 0,
                    "names": "HF golden 抓数的命名（见模块说明）",
                    **(extra or {}),
                    "tensors": self.meta,
                },
                f,
                indent=1,
                ensure_ascii=False,
                sort_keys=True,
            )


class Hy4ChoiceCapture:
    """每个 rank、每个 microbatch 的离散选择（`--choices-capture-step N`）：MoE 路由与 DSA top-k。

    名字与 `Hy4Capture` 相同（`hf.model.layers.<L>.self_attn.topk`、`.mlp.gate.{routing_map,probs,expert_bias}`，
    MTP 层是 `hf.model.mtp_layers.0.*`），另存 `input.tokens`，所以 `step1_ref.py --inject-l1` 可以直接读。
    每个 microbatch 写一个文件 `choices/step_<N>_row_<R>.safetensors`，R = 第 N 步全局批里的行号
    （microbatch m、DP rank r → R = m·DP + r；tokens.npy 的行 = (N−1)·GBS + R），另写每个 rank 的 index json。
    开整层重算时，反向里的重算前向会再触发钩子；每个 microbatch 每个名字只记第一次（首次前向）的值。
    可以抓多步：`begin_step(N)` … `end_step()`，每步各写一套文件。
    """

    def __init__(self, model_core, rundir: str, step: int, dp: int, dp_rank: int, gbs: int):
        self.step, self.dp, self.dp_rank, self.gbs = step, dp, dp_rank, gbs
        self.out = os.path.join(rundir, "choices")
        self.cur: dict[str, torch.Tensor] | None = None
        self.row = None
        self.index: dict[str, dict] = {}
        self.handles = []
        for mname, mod in model_core.named_modules():
            hp = _hf_prefix(mname)
            if hp is None:
                continue
            pre, sub = hp
            if sub == "self_attention.core_attention":
                mod.capture = True

                def core_fwd(m, a, o, pre=pre):
                    k = f"{pre}.self_attn.topk"
                    if self.cur is None or getattr(m, "last_topk", None) is None:
                        return
                    if k in self.cur:  # 重算时的第二次前向：只核对与首次是否相同
                        self._recheck(k, m.last_topk)
                        return
                    self.cur[k] = m.last_topk.detach().to("cpu", copy=True).contiguous()

                self.handles.append(mod.register_forward_hook(core_fwd))
            elif sub == "mlp.router":

                def router_fwd(m, a, o, pre=pre):
                    if self.cur is None:
                        return
                    if f"{pre}.mlp.gate.routing_map" in self.cur:  # 重算时的第二次前向：只核对与首次是否相同
                        self._recheck(f"{pre}.mlp.gate.routing_map", o[1])
                        return
                    self.cur[f"{pre}.mlp.gate.probs"] = o[0].detach().to("cpu", copy=True).contiguous()
                    self.cur[f"{pre}.mlp.gate.routing_map"] = o[1].detach().to("cpu", copy=True).contiguous()
                    eb = getattr(m, "expert_bias", None)
                    if eb is not None:
                        self.cur[f"{pre}.mlp.gate.expert_bias"] = eb.detach().to("cpu", copy=True).contiguous()

                self.handles.append(mod.register_forward_hook(router_fwd))

    def _recheck(self, key, t):
        same = torch.equal(self.cur[key], t.detach().to("cpu"))
        self.recheck.setdefault(key, [0, 0])
        self.recheck[key][0] += 1
        self.recheck[key][1] += int(not same)

    def begin_micro(self, m: int, tokens: torch.Tensor):
        self._flush()
        self.row = m * self.dp + self.dp_rank
        self.cur = {"input.tokens": tokens.detach().to("cpu", copy=True).contiguous()}
        self.recheck = {}
        self.micro = m

    def _flush(self):
        from safetensors.torch import save_file

        if self.cur is None:
            return
        os.makedirs(self.out, exist_ok=True)
        fn = f"step_{self.step}_row_{self.row:02d}.safetensors"
        save_file(self.cur, os.path.join(self.out, fn))
        self.index[fn] = {
            "row_in_step": self.row,
            "tokens_npy_row": (self.step - 1) * self.gbs + self.row,
            "rank": self.dp_rank,
            "micro": self.micro,
            # 开重算时：每个名字 [重算次数, 与首次前向不同的次数]
            "recompute_recheck": dict(getattr(self, "recheck", {})),
            "tensors": {
                k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")} for k, v in self.cur.items()
            },
        }
        self.cur = None

    def begin_step(self, step: int):
        self.step, self.index, self.cur = step, {}, None

    def end_step(self, extra: dict | None = None):
        self.finish(extra, remove=False)

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def finish(self, extra: dict | None = None, remove: bool = True):
        self._flush()
        if remove:
            self.close()
        os.makedirs(self.out, exist_ok=True)
        with open(os.path.join(self.out, f"step_{self.step}_rank{self.dp_rank}_index.json"), "w") as f:
            json.dump(
                {
                    "step": self.step,
                    "rank": self.dp_rank,
                    "layout": "topk [b, s, k]（-1 表示空位）；routing_map / probs [b·s, E]（Megatron 的 token 顺序，b = 1）",
                    **(extra or {}),
                    "files": self.index,
                },
                f,
                indent=1,
                ensure_ascii=False,
                sort_keys=True,
            )
