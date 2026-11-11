"""因果性（真实模块，整网）：单卡上用 Megatron-Bridge 搭出与训练同一套模块的混元 3 截 4 层模型（EP1，全部 192 个专家在一张卡上），
用第 1 步的真实输入，只改位置 ≥ p 的 token，loss 只算位置 < p，比位置 < p 的各层输出、逐 token loss、激活梯度与参数梯度。
规格见 docs/dev/design/0004-l1-gpu-kernels.md「四、合成 poison 与因果性」。

判据：位置 < p 的前向输出、逐 token loss、激活梯度必须逐位不变（因果注意力 + 逐 token 的 MLP / 路由）；
参数梯度分稠密与专家两类：未来位置的输出梯度精确为 0，所以数学上都不变；稠密层的 wgrad 形状不变、逐位应不变；
专家的 wgrad 每个专家收到的 token 数随未来 token 的路由变化，GEMM 形状变了，只报不判。
另有两条：同一输入跑两次必须逐位相同（不成立则比较无意义）；第一个 MoE 层里过去位置输出不同的 token，
数一数有多少路由到了「token 数变了、且 ≤ 32」的专家（小 M 时 GEMM 逐行结果随同组行数变，见 l1_kernels.py 的 gemm_rows）。

用法（Pod 里，一张卡；GPU 共享时包 flock）：
    source /data/env.sh; export PYTHONPATH=/tmp/janus-g61:$PYTHONPATH
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
    CUDA_VISIBLE_DEVICES=0 flock /data/gpu.lock python models/hunyuan3/frameworks/megatron/causal_model.py \
        --run /data/janus-runs2/g61-obs --hf /data/hy3-4l --out /data/janus-g61/causal-model
"""
from __future__ import annotations

import argparse
import datetime
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from l1_ops import OPS, SEED, STEP, N, environment, write_outputs  # noqa: E402

from janus.checks import Threshold  # noqa: E402
from janus.dump.names import canonical  # noqa: E402
from janus.store import Run  # noqa: E402


def determinism() -> dict:
    """与训练时 Megatron 确定性模式同一套设置：环境变量（apply_determinism_env）、torch 确定性算法、Triton 固定调优表。
    单卡脚本不走训练入口，这些要自己调；不调的话 MoE 里的 scatter / index_add 类算子不确定，同一输入跑两次都不逐位。"""
    from megatron.core import tuning
    from megatron.training.determinism import apply_determinism_env
    apply_determinism_env(os.environ)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    tuning.install(deterministic=True)
    return {k: os.environ.get(k) for k in ("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "CUBLAS_WORKSPACE_CONFIG", "NCCL_ALGO",
                                           "TRITON_CACHE_AUTOTUNING")} | {"torch_deterministic": torch.are_deterministic_algorithms_enabled()}


def build(hf: str):
    from megatron.bridge import AutoBridge
    pr = AutoBridge.from_hf_pretrained(hf, trust_remote_code=True).to_megatron_provider(load_weights=True)
    pr.tensor_model_parallel_size = pr.pipeline_model_parallel_size = pr.context_parallel_size = 1
    pr.expert_model_parallel_size = 1
    pr.sequence_parallel = False
    pr.seq_length = 4096
    pr.mtp_num_layers = None
    pr.moe_token_dispatcher_type = "alltoall"
    pr.deterministic_mode = True
    pr.cross_entropy_loss_fusion = False
    pr.gradient_accumulation_fusion = False          # 没有 DDP 外壳就没有 main_grad，wgrad 写进 .grad
    pr.finalize()
    model = pr.provide_distributed_model(wrap_with_ddp=False)[0]
    model.train()
    return model


class Probe:
    """按 janus 规范名抓各子模块前向输出（只留位置 < p 的行）和它们的梯度。"""

    def __init__(self, model):
        self.buf, self.handles = {}, []
        inner = model
        while hasattr(inner, "module") and not hasattr(inner, "decoder"):
            inner = inner.module
        for name, mod in inner.named_modules():
            c = canonical(name)
            if c and not c.endswith("experts.out"):           # 专家分发缓冲区的行序随路由变，不在 token 空间
                self.handles.append(mod.register_forward_hook(self._hook(c)))
            m = re.match(r"^decoder\.layers\.(\d+)\.mlp\.router$", name)
            if m:                                            # 路由选中矩阵：用来数每个专家收到几个 token
                self.handles.append(mod.register_forward_hook(
                    lambda mod, i, o, L=int(m.group(1)): self.maps.__setitem__(L, o[1].detach().cpu().numpy().astype(bool))))  # noqa: B008  默认参数绑定当轮层号
        self.maps = {}
        self.inner = inner

    def _hook(self, c):
        def fn(mod, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not isinstance(t, torch.Tensor):
                return
            self.buf[c] = t
            if t.requires_grad:
                t.register_hook(lambda g, c=c: self.buf.__setitem__(f"{c}.grad", g))
        return fn


def one(model, probe, tokens, labels, pos, p):
    model.zero_grad(set_to_none=True)
    probe.buf, probe.maps = {}, {}
    loss = model(input_ids=tokens, position_ids=pos, attention_mask=None, labels=labels)     # [b, s] 逐 token loss
    mask = torch.zeros_like(loss)
    mask[:, :p] = 1
    ((loss * mask).sum() / mask.sum()).backward()
    acts = {}
    for k, v in probe.buf.items():
        a = N(v)
        acts[k] = a.reshape(a.shape[0], -1)[:p] if a.shape[0] == tokens.shape[1] else a      # [s, b, ...] → 位置 < p
    acts["loss.per_token"] = N(loss.float())[:, :p]
    grads = {n: N(w.grad) for n, w in probe.inner.named_parameters() if w.grad is not None}
    acts["_maps"] = dict(probe.maps)
    return acts, grads


def attribute(base_a, alt_a, p, small=32):
    """第一个 MoE 层：过去位置里输出不同的 token，是不是都路由到了「token 数变了、且很少（≤ small）」的专家。
    小 M 时 BF16 GEMM 的逐行结果会随同组其他行变（l1_kernels.py 的 L1.kernel.gemm_row_invariance），这是唯一被怀疑的来源。"""
    L = min(base_a["_maps"])
    mb, ma = base_a["_maps"][L], alt_a["_maps"][L]
    cb, ca = mb.sum(0), ma.sum(0)
    o_b, o_a = base_a[f"layers.{L}.mlp.out"], alt_a[f"layers.{L}.mlp.out"]
    diff = np.nonzero((o_b.view(np.int16) != o_a.view(np.int16)).any(axis=1))[0]
    changed = cb != ca
    smallset = changed & (np.minimum(cb, ca) <= small)
    expl = [t for t in diff if (mb[t] & smallset).any()]
    return {"name": f"p{p}.layers.{L}", "layer": int(L), "p": p, "past_routing_equal": bool((mb[:p] == ma[:p]).all()),
            "experts_count_changed": int(changed.sum()), "experts_changed_and_small": int(smallset.sum()),
            "small_threshold": small, "past_tokens_out_differ": int(len(diff)), "explained_by_small_changed_expert": int(len(expl)),
            "min_count_changed_experts": int(np.minimum(cb, ca)[changed].min()) if changed.any() else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="取第 1 步 rank 0 第一个 microbatch 的输入（input.tokens / labels / position_ids）")
    ap.add_argument("--hf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    det = determinism()
    torch.cuda.set_device(0)
    env = environment()
    env["determinism"] = det
    run = Run.open(a.run)
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    t0 = time.time()
    model = build(a.hf)
    env["model_build_seconds"] = round(time.time() - t0, 1)
    env["experts_type"] = type(model.module.decoder.layers[1].mlp.experts).__name__ if hasattr(model, "module") else None
    probe = Probe(model)
    dev = torch.device("cuda", 0)
    tok = torch.from_numpy(run.tensor(STEP, "input.tokens").astype(np.int64)).to(dev)
    lab = torch.from_numpy(run.tensor(STEP, "input.labels").astype(np.int64)).to(dev)
    pos = torch.from_numpy(run.tensor(STEP, "input.position_ids").astype(np.int64)).to(dev)
    rng = np.random.default_rng(SEED)
    import json
    vocab = int(json.loads((Path(a.hf) / "config.json").read_text())["vocab_size"])
    fwd, agrad, dense, experts, repeat, attrib = [], [], [], [], [], []
    for p in (2048, 3001):
        base_a, base_g = one(model, probe, tok, lab, pos, p)
        if p == 2048:
            # 先证同一输入跑两次逐位相同：不成立的话，下面「改未来 token」的比较没有意义
            rep_a, rep_g = one(model, probe, tok, lab, pos, p)
            repeat += [dict(name=f"repeat.{k}", out=rep_a[k], bitref=base_a[k]) for k in sorted(base_a) if k != "_maps"]
            repeat += [dict(name=f"repeat.grad.{k}", out=rep_g[k], bitref=base_g[k]) for k in sorted(base_g)]
            del rep_a, rep_g
        tok2, lab2 = tok.clone(), lab.clone()
        alt = torch.from_numpy(rng.integers(0, vocab, tok.shape[1] - p)).to(dev)
        tok2[:, p:] = alt
        lab2[:, p:] = torch.roll(alt, -1)
        alt_a, alt_g = one(model, probe, tok2, lab2, pos, p)
        ex = {"p": p, "p_mod128": p % 128, "changed_tokens": int((tok2 != tok).sum())}
        attrib.append(attribute(base_a, alt_a, p))
        base_a.pop("_maps"), alt_a.pop("_maps")
        for k in sorted(base_a):
            row = dict(name=f"p{p}.{k}", out=alt_a[k], bitref=base_a[k], extra=ex)
            (agrad if k.endswith(".grad") else fwd).append(row)
        for k in sorted(base_g):
            row = dict(name=f"p{p}.grad.{k}", out=alt_g[k], bitref=base_g[k],
                       extra=dict(ex, rel_diff=float(np.linalg.norm((alt_g[k].astype(np.float64) - base_g[k].astype(np.float64)).ravel())
                                                     / max(np.linalg.norm(base_g[k].astype(np.float64).ravel()), 1e-300))))
            (experts if re.search(r"\.experts\.", k) else dense).append(row)
        torch.cuda.empty_cache()
    checks = []
    for cid, cases, th, hard in (("L2.causal.model.repeat", repeat, Threshold("exact"), True),
                                 ("L2.causal.model.fwd", fwd, Threshold("exact"), True),
                                 ("L2.causal.model.act_grad", agrad, Threshold("exact"), True),
                                 ("L2.causal.model.param_grad.dense", dense, Threshold("exact"), True),
                                 ("L2.causal.model.param_grad.experts", experts, None, False)):
        r = OPS(cid, "L2", cases, th, hard=hard)
        r.evidence = [{"path": str(run.path), "sha256": run.meta_sha256()}]
        checks.append(r)
        print(f"{r.status:<6} {cid:<36} {r.message}", flush=True)
    from janus.checks.base import PASS, CheckResult
    r = CheckResult("L2.causal.model.attribution", "op_replay", "L2", False, status=PASS, details=attrib,
                    metrics={"past_tokens_out_differ": sum(x["past_tokens_out_differ"] for x in attrib),
                             "explained": sum(x["explained_by_small_changed_expert"] for x in attrib)},
                    message="诊断：第一个 MoE 层里过去位置输出不同的 token，有多少路由到了 token 数变了且 ≤ 32 的专家")
    r.evidence = [{"path": str(run.path), "sha256": run.meta_sha256()}]
    checks.append(r)
    print(f"{r.status:<6} {r.id:<36} {r.metrics}", flush=True)
    return write_outputs(Path(a.out), checks, [run], env, Path(__file__), "phase1-causal-model", started,
                         columns=["bit_match_frac", "rel_diff", "p", "p_mod128", "changed_tokens", "past_routing_equal",
                                  "experts_count_changed", "experts_changed_and_small", "past_tokens_out_differ",
                                  "explained_by_small_changed_expert", "min_count_changed_experts"],
                         ref_line="参照：同一模型、未改未来 token 的同一次前向 / 反向（逐位）",
                         col_note="列：bit_match_frac 改了未来 token 之后对不改时的逐位相同率（只取位置 < p）；rel_diff 参数梯度的相对 L2 差。")


if __name__ == "__main__":
    sys.exit(main())
