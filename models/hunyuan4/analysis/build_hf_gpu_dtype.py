"""GPU 某一步的 FP32 主权重（HF 名字）→ HF 目录，每个张量的 dtype 按 GPU 模型里参数的实际 dtype。

GPU 留 FP32 的（输出头、索引器 k_norm、iHC / hc_head、sink、路由偏置）存 FP32 主权重原值，其余存 bf16(主权重)。
不能照 HF 基准目录的 dtype 舍：基准里 lm_head、索引器 k_norm、MTP 层 sink 与 MTP 路由偏置是 BF16，但 GPU 模型把它们
留在 FP32，舍成 BF16 后重建的前向与原运行不同（第 8 步 lm_loss 差 1.2e-4，results/t1-rebuild-state-dtype.json）。
GPU 的载入检查对 FP32 参数收 BF16 值照样通过，拦不住。

    python models/hunyuan4/analysis/build_hf_gpu_dtype.py --params <主权重.safetensors（HF 名字，FP32，路由偏置已换成该步的）> \\
        --base <基准 HF 目录> --out <输出目录>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil


def keep_fp32(n: str) -> bool:
    return (n == "lm_head.weight" or ".indexer.k_norm." in n or "learnable_sink_param" in n
            or "e_score_correction_bias" in n or ".hc_" in n or "hc_head" in n)


def main():
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    src = safe_open(a.params, "pt")
    base = safe_open(os.path.join(a.base, "model.safetensors"), "pt")
    tens, n32, changed = {}, 0, []
    for n in base.keys():
        x = src.get_tensor(n).to(torch.float32)
        if keep_fp32(n):
            tens[n] = x.contiguous()
            n32 += 1
            if base.get_tensor(n).dtype == torch.bfloat16:
                changed.append(n)
        else:
            tens[n] = x.to(torch.bfloat16).contiguous()
    save_file(tens, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    for f in ("config.json", "model.safetensors.index.json"):
        shutil.copy(os.path.join(a.base, f), a.out)
    man = {}
    with open(os.path.join(a.out, "model.safetensors"), "rb") as fh:  # sha256 按原始字节，与 hy4_weights.tensor_sha_ok 同法
        hl = int.from_bytes(fh.read(8), "little")
        hdr = json.loads(fh.read(hl))
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            b0, b1 = meta["data_offsets"]
            fh.seek(8 + hl + b0)
            man[name] = hashlib.sha256(fh.read(b1 - b0)).hexdigest()
    json.dump(dict(sorted(man.items())), open(os.path.join(a.out, "tensor_sha256.json"), "w"), indent=1)
    print("FP32", n32, "BF16", len(tens) - n32, "基准里是 BF16、这里存 FP32：", changed)


if __name__ == "__main__":
    main()
