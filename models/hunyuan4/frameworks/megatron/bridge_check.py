"""核 Megatron-Bridge 桥（hy4_bridge.py）：Bridge 载入的每个参数与 hy4_weights.load_hf_into_mcore 载入的逐位相同。

单进程、一张卡（TP = EP = 1），NeMo 26.10 镜像（自带 Megatron-Bridge 0.7.0 与 transformers 5.17.0）：
    python bridge_check.py --hf <截好的 4 层 HF 目录> --layers 2 --work /data/bridge-hf --json bridge_check.json
`--work` 下建一个截层的 HF 目录（config 改层数、去 MTP，张量文件用符号链接）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def truncated_dir(src: str, dst: str, n: int) -> str:
    os.makedirs(dst, exist_ok=True)
    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["num_hidden_layers"] = n
    for k in ("indexer_types", "layer_types", "mlp_layer_types"):
        cfg[k] = cfg[k][:n]
    cfg["num_nextn_predict_layers"] = 0
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=1)
    for f in os.listdir(src):
        if f != "config.json" and not os.path.exists(os.path.join(dst, f)):
            os.symlink(os.path.join(src, f), os.path.join(dst, f))
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--work", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    torch.distributed.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    ps.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(1234)
    import hy4_bridge  # noqa: F401  注册桥
    import hy4_mcore as H
    from hy4_weights import load_hf_into_mcore
    from megatron.bridge import AutoBridge

    d = truncated_dir(a.hf, a.work, a.layers)
    res = {"layers": a.layers}
    bridge = AutoBridge.from_hf_pretrained(d)
    provider = bridge.to_megatron_provider(load_weights=False)
    res["provider"] = type(provider).__name__
    provider.finalize()
    from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked

    # 同 Float16Module：标了 keep_in_fp32 的（输出头、iHC、sinks、k_norm、路由偏置）留 FP32，其余转 BF16
    m_bridge = convert_module_to_dtype_except_fp32_marked(provider.provide().cuda(), torch.bfloat16)
    bridge.load_hf_weights([m_bridge], d)

    hf = json.load(open(os.path.join(d, "config.json")))
    cfg = H.Hy4Config(
        **H.hy4_config_kwargs(hf),
        params_dtype=torch.bfloat16,
        bf16=True,
        transformer_impl="transformer_engine",
        moe_grouped_gemm=True,
        perform_initialization=False,
    )
    m_ref = H.build_model(cfg, use_te=True, vocab_size=hf["vocab_size"], max_seq=4096).cuda()
    load_hf_into_mcore(m_ref, cfg, d)
    pb = dict(m_bridge.named_parameters())
    pb.update({k: v for k, v in m_bridge.named_buffers() if k.endswith("expert_bias")})
    pr = dict(m_ref.named_parameters())
    pr.update({k: v for k, v in m_ref.named_buffers() if k.endswith("expert_bias")})
    only_b, only_r = sorted(set(pb) - set(pr)), sorted(set(pr) - set(pb))
    bad = []
    for k in sorted(set(pb) & set(pr)):
        x, y = pb[k].detach(), pr[k].detach()
        if x.shape != y.shape or not torch.equal(x.float(), y.float()):
            bad.append({"param": k, "bridge": [str(x.dtype), list(x.shape)], "ref": [str(y.dtype), list(y.shape)]})
    res.update(
        {
            "params_compared": len(set(pb) & set(pr)),
            "only_in_bridge": only_b[:20],
            "only_in_ref": only_r[:20],
            "mismatch": bad[:20],
            "pass": not bad and not only_b and not only_r,
        }
    )
    print("BRIDGE_CHECK", json.dumps(res), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
