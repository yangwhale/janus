"""TPU 抓到的离散选择（`train_hy4.py --capture-choices` 的 choices_captured.safetensors）→ 逐行的灌选择文件。

输出格式同 GPU 抓数里给 `step1_ref.py --inject-dir` / `cpu_bf16_gpu_forward.py --inject-dir` 用的
`step_<N>_row_<R>.safetensors`：
- `input.tokens`：这一行的输入（[S]，int64），读入时核对是不是 tokens.npy 的同一行；
- `hf.model.layers.<i>.self_attn.topk` / `hf.model.mtp_layers.0.self_attn.topk`：DSA top-k 位置（[1, S, K]，int64，空位 -1）；
  共享层的槽位在抓数时已经沿用块首层，这里照写（灌入的一方只读自带索引器的层）；
- `hf.model.layers.<i>.mlp.gate.routing_map` / `hf.model.mtp_layers.0.mlp.gate.routing_map`：路由专家（[S, E]，bool）。
抓数文件的行序已经是 GPU 的行序 R（全局批第 R 行 = tokens.npy 第 (N−1)·16 + R 行）。

    python choices_to_inject.py --captured <choices_captured.safetensors> --hf-config <config.json> --step 1 --out <目录>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--captured", required=True)
    p.add_argument("--hf-config", required=True)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cfg = json.load(open(a.hf_config))
    L = cfg["num_hidden_layers"]
    E = cfg["n_routed_experts"]
    moe = [i for i, t in enumerate(cfg["mlp_layer_types"][:L]) if t == "sparse"]
    f = safe_open(a.captured, "np")
    ids = f.get_tensor("forced_routed_experts")  # [rows, S, slots, k]
    topk = f.get_tensor("forced_indexer_topk")  # [rows, S, slots, K]
    toks = f.get_tensor("tokens")
    rows, S, slots, k = ids.shape
    assert slots == L + 1, (slots, L)
    os.makedirs(a.out, exist_ok=True)
    names = [(f"hf.model.layers.{i}", i) for i in range(L)] + [("hf.model.mtp_layers.0", L)]
    for r in range(rows):
        out = {"input.tokens": torch.from_numpy(toks[r].astype(np.int64))}
        for hn, slot in names:
            out[f"{hn}.self_attn.topk"] = torch.from_numpy(topk[r, :, slot].astype(np.int64))[None]
            if slot in moe or slot == L:
                idx = ids[r, :, slot]
                assert (idx >= 0).all() and (idx < E).all(), (r, slot)
                rm = np.zeros((S, E), bool)
                np.put_along_axis(rm, idx.astype(np.int64), True, axis=1)
                assert (rm.sum(1) == k).all(), (r, slot)
                out[f"{hn}.mlp.gate.routing_map"] = torch.from_numpy(rm)
        save_file(out, os.path.join(a.out, f"step_{a.step}_row_{r:02d}.safetensors"))
    print("WROTE", rows, "rows to", a.out)


if __name__ == "__main__":
    main()
