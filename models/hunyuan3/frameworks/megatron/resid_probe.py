"""残差流探针（GPU 侧，与 TPU 侧 frameworks/maxtext/train_hy3.py 的 --resid-probe-every 同名同口径）。

每 N 步（以及第 1 步），在这一步的参数更新之后，用当前参数在一条固定序列上另跑一次前向（torch.no_grad + eval，
不进训练图、不改训练轨迹），记：
  resid_mean.decoder.dense_layers.{in,out}.0      第 0 层（稠密层）的输入 / 输出在 token 上的均值向量（长度 = 隐藏维）
  resid_mean.decoder.moe_layers.{in,out}.<i>      第 i+1 层（MoE 层）的输入 / 输出（TPU 侧扫描层的第 i 层）
  resid_mean.decoder.decoder_norm.out             最后归一化输出
  probe_logit_mean / probe_lse_mean               logits 在 token × 词表上的均值、logsumexp 在 token 上的均值
  final_norm_scale                                最后归一化的 scale（FP32 主权重）
  tok_rms.<同上各名>                               同一张量第 0 行每个 token 沿隐藏维的 RMS（长度 = 序列长）
固定序列：golden 第 1 步 rank 0 第一个 microbatch（与 TPU 侧相同；tokens 指纹见 data/rank0.jsonl 第 1 行），
由 --resid-probe-seq 给的 .npz（tokens、position_ids，形状 [1, S]）读入；8 个 rank 喂同一条（EP 的 all-to-all 要所有 rank 一起前向），
只记 rank 0 的值。

副作用的处理：路由的 expert_bias 计数只在 torch.is_grad_enabled() 时累加（megatron/core/transformer/moe/router.py），
no_grad 下不动；JanusDump 的抓数钩子在探针期间关掉（capturing=False），路由负载计数（dump.counts）探针前后恢复；
探针不经过 forward_step，不进数据指纹。
"""

import hashlib

import numpy as np
import torch


# GPU 层号 → TPU 侧名字
def _layer_name(i):
    return ("decoder.dense_layers", 0) if i == 0 else ("decoder.moe_layers", i - 1)


class ResidProbe:
    def __init__(self, every: int, seq_path: str):
        self.every = every
        z = np.load(seq_path)
        self.tokens = torch.from_numpy(np.ascontiguousarray(z["tokens"]).astype(np.int64))
        self.position_ids = torch.from_numpy(np.ascontiguousarray(z["position_ids"]).astype(np.int64))
        self.tokens_sha256 = hashlib.sha256(np.ascontiguousarray(z["tokens"]).astype(np.int64).tobytes()).hexdigest()
        self.dump = None              # train_hy3.py 设成 JanusDump，值写进它这一步的 step_vals

    def _gpt(self, context):
        m = context.model[0]
        while not hasattr(m, "decoder") and hasattr(m, "module"):
            m = m.module
        return m

    def _final_norm_fp32(self, context, gpt):
        w = gpt.decoder.final_layernorm.weight
        opt = context.optimizer
        for sub in getattr(opt, "chained_optimizers", None) or [opt]:
            f16, f32 = getattr(sub, "float16_groups", None), getattr(sub, "fp32_from_float16_groups", None)
            if f16 is None:
                continue
            for ga, gb in zip(f16, f32):
                for a, mp in zip(ga, gb):
                    if a is w:
                        return mp.detach().double()
        return w.detach().double()

    @torch.no_grad()
    def on_train_step_end(self, context):
        step = context.state.train_state.step + 1
        if not (step == 1 or step % self.every == 0):
            return
        gpt = self._gpt(context)
        dev = torch.cuda.current_device()
        acts, rms, handles = {}, {}, []

        def hook(key, which):
            def fn(mod, args, kwargs, out):
                if which == "in":
                    t = args[0] if args else kwargs["hidden_states"]     # TransformerBlock 按关键字传 hidden_states
                else:
                    t = out[0] if isinstance(out, (tuple, list)) else out
                x = t.detach()[:, 0].double()                            # [S, B, H] → 第 0 行 [S, H]
                acts[key] = x.mean(dim=0)                                 # 在 token 上求均值 → [H]
                rms[key] = x.square().mean(dim=-1).sqrt()                 # 每个 token 沿隐藏维的 RMS → [S]
            return fn

        for i, layer in enumerate(gpt.decoder.layers):
            base, li = _layer_name(i)
            handles.append(layer.register_forward_hook(hook(f"{base}.in.{li}", "in"), with_kwargs=True))
            handles.append(layer.register_forward_hook(hook(f"{base}.out.{li}", "out"), with_kwargs=True))
        handles.append(gpt.decoder.final_layernorm.register_forward_hook(hook("decoder.decoder_norm.out", "out"),
                                                                          with_kwargs=True))
        d = self.dump
        saved = None
        if d is not None:
            saved = (d.capturing, {k: v.clone() for k, v in d.counts.items()})
            d.capturing = False
        was_training = [m.training for m in context.model]
        for m in context.model:
            m.eval()
        try:
            logits = gpt(self.tokens.to(dev), self.position_ids.to(dev), None)      # labels=None → [B, S, V]
        finally:
            for h in handles:
                h.remove()
            for m, t in zip(context.model, was_training):
                m.train(t)
            if d is not None:
                d.capturing, d.counts = saved[0], saved[1]
        if torch.distributed.get_rank() != 0 or d is None:
            return
        l0 = logits[0].float()
        vals = {"probe_logit_mean": float(l0.sum(dtype=torch.float64) / l0.numel()),
                "probe_lse_mean": float(torch.logsumexp(l0, dim=-1).double().mean())}
        for k, v in acts.items():
            vals[f"resid_mean.{k}"] = [float(x) for x in v.cpu().tolist()]
        for k, v in rms.items():
            vals[f"tok_rms.{k}"] = [float(x) for x in v.cpu().tolist()]
        vals["final_norm_scale"] = [float(x) for x in self._final_norm_fp32(context, gpt).cpu().tolist()]
        d.step_vals.update(vals)
