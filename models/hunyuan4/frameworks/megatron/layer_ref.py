"""L2 逐层隔离回放的 CPU FP64 参照：GPU golden 同一套 Megatron-Core 代码，单层、全程 FP64。

和 TPU 一侧的 `frameworks/maxtext/layer_replay.py` 同一套输入（GPU 逐层抓数 `hy4_capture.py`，如 `g7-l1-a`）：
- 第 k 层（主干）的输入 = GPU 抓到的 `layers.k.in0`（4 路流；第 0 层取 `attn_hc.in`，即展开后的 4 路），前向得层输出；
- 余切 = GPU 抓到的 `layers.k.out0.grad`，反向得层输入的梯度（第 0 层是 4 路之和，同 GPU 的 `in0.grad`）；
- IndexCache 共享层（第 2–4 层用第 1 层、第 6 层用第 5 层的 top-k）：top-k 由块首层的索引器在 GPU 给的块首层输入上算
  （按层号顺序跑，先跑块首层），同 TPU 一侧的做法；
- MTP 层：`mtp_model_layer` 的输入 / 输出 `mtp_layers.0.in0` / `out0`（一路）。
FP64 的写法同 `step1_ref.py`（ATen 一层改写低精度、`--audit` 逐算子核类型）。输出每层的输出与输入梯度（FP64）；
`--param-grads` 另存这一层参数的梯度（只对这一行，同一个余切；索引器 KD 不在单层里，索引器参数的梯度不完整）。

    python layer_ref.py --hf <HF 目录> --gpu-l1 <step_1_fwd.safetensors> --gpu-l1-bwd <step_1_bwd.safetensors> \\
        --out <layer_ref.safetensors> [--layers 0,1,2,3,4,5,6,mtp0] [--threads 40] [--audit]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf", required=True)
    p.add_argument("--gpu-l1", required=True)
    p.add_argument("--gpu-l1-bwd", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", default="0,1,2,3,4,5,6,mtp0")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--audit", action="store_true")
    p.add_argument(
        "--param-grads",
        action="store_true",
        help="另存这一层参数的梯度（同一个余切），名字 <层>.param.<Megatron 层内参数名>",
    )
    a = p.parse_args()

    import step1_ref as R

    if a.threads:
        torch.set_num_threads(a.threads)
    dt = torch.float64
    torch.set_default_dtype(dt)
    force = R.ForceDtype(dt)
    force.__enter__()
    R.cpu_megatron_setup()
    import hy4_weights as W

    for nm in ("get", "get_slice"):
        setattr(W.Safetensors, nm, R._unforced(getattr(W.Safetensors, nm)))
    R.relax_mask_dtype(dt)
    from safetensors import safe_open
    from safetensors.torch import save_file

    args = SimpleNamespace(hf=a.hf, layers=0, mtp=None, indexer_loss=0.1, head_chunk=1024, log_max_logit=False)
    hf, cfg, model, L, mtp, _ = R.build(args, dt)
    model.eval()  # 单层回放只看层输出与输入梯度；索引器 KD 只更新索引器参数，不影响这两个量
    fwd = R._unforced(lambda: safe_open(a.gpu_l1, "pt"))()
    bwd = R._unforced(lambda: safe_open(a.gpu_l1_bwd, "pt"))()

    def get(f, name):
        return R._unforced(lambda: f.get_tensor(name))().to(dt)

    audit = R.DtypeAudit(dt) if a.audit else None
    out, rec = {}, {}
    layers = [x for x in a.layers.split(",") if x]
    order = sorted(layers, key=lambda x: (x == "mtp0", int(x) if x != "mtp0" else 0))
    for k in order:
        t0 = time.time()
        if k == "mtp0":
            pre = "hf.model.mtp_layers.0"
            x = get(fwd, pre + ".in0")[0].unsqueeze(1)  # [S, 1, H]
            g = get(bwd, pre + ".out0.grad")[0].unsqueeze(1)
            layer = model.mtp.layers[0].mtp_model_layer
        else:
            pre = f"hf.model.layers.{k}"
            xi = get(fwd, pre + (".attn_hc.in" if k == "0" else ".in0"))  # [1, S, n, H]
            x = xi[0].reshape(xi.shape[1], 1, -1)  # [S, 1, n·H]
            gi = get(bwd, pre + ".out0.grad")
            g = gi[0].reshape(gi.shape[1], 1, -1)
            layer = model.decoder.layers[int(k)]
        S = x.shape[0]
        x = x.detach().requires_grad_(True)
        if audit is not None:
            audit.__enter__()
        pos = torch.arange(S).unsqueeze(0)
        if k == "mtp0":
            y, _ = layer(hidden_states=x, attention_mask=None)
        else:
            y, _ = layer(x, attention_mask=None, position_ids=pos)
        y.backward(g)
        if audit is not None:
            audit.__exit__(None, None, None)
        gx = x.grad
        if k == "mtp0":
            out[f"{k}.out"], out[f"{k}.in_grad"] = y.detach()[:, 0], gx[:, 0]
        else:
            H = cfg.hidden_size
            out[f"{k}.out"] = y.detach()[:, 0].reshape(S, -1, H)
            ig = gx[:, 0].reshape(S, -1, H)
            out[f"{k}.in_grad"] = ig.sum(1) if k == "0" else ig  # 第 0 层：4 路之和（GPU 的 in0.grad 是展开前的一路）
        if a.param_grads:
            for n, prm in layer.named_parameters():
                if prm.grad is not None:
                    out[f"{k}.param.{n}"] = prm.grad.detach().clone()
        model.zero_grad(set_to_none=True)
        rec[k] = round(time.time() - t0, 1)
        print("LAYER", k, rec[k], flush=True)
    R._unforced(lambda: save_file({k: v.contiguous() for k, v in out.items()}, a.out))()
    meta = {
        "what": "L2 逐层回放的 CPU FP64 参照（GPU golden 同一套 Megatron-Core 代码，单层，全程 FP64）",
        "hf": os.path.basename(os.path.normpath(a.hf)),
        "gpu_l1": os.path.basename(a.gpu_l1),
        "seconds": rec,
        "dtype_rewrites": force.rewrites,
        "audit": None if audit is None else {"ops": audit.n_ops, "non_target_float_outputs": audit.bad},
    }
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("DONE", json.dumps(meta["audit"]), flush=True)


if __name__ == "__main__":
    main()
