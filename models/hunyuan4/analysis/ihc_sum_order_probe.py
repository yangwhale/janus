"""B200 上 iHC 四路流加权和 torch.sum(pre.unsqueeze(-1) * x, dim=2) 的实际相加顺序（单卡、几十秒）。

    NVIDIA_TF32_OVERRIDE=0 python3 ihc_sum_order_probe.py      # 训练镜像里跑，打印一行 PROBE_JSON {...}

写法与 hy4_mcore.Hy4HyperConnection 相同（pre FP32、x BF16 或 FP32，乘积 FP32 各舍一次，沿长度 4 的维求和），
结果与 numpy 按几种顺序重算的结果逐位比：依次相加、两种配对、倒序、FMA 链。
"""

import json

import numpy as np
import torch

torch.manual_seed(0)
res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}


def variants(p, x):
    t = (p[..., None] * x).astype(np.float32)  # tk = fl(pk·xk)，FP32 乘、各舍一次
    t0, t1, t2, t3 = (t[:, :, k, :] for k in range(4))
    f = np.float32
    seq = ((t0 + t1) + t2) + t3
    p02 = (t0 + t2) + (t1 + t3)
    p01 = (t0 + t1) + (t2 + t3)
    rev = ((t3 + t2) + t1) + t0
    # FMA 写法：acc = fma(p_k, x_k, acc)，用 FP64 算 p·x + acc 再舍入成 FP32（p·x 在 FP64 里是精确的）
    acc = (p[:, :, 0, None].astype(np.float64) * x[:, :, 0, :]).astype(f)
    for k in range(1, 4):
        acc = (p[:, :, k, None].astype(np.float64) * x[:, :, k, :].astype(np.float64) + acc.astype(np.float64)).astype(
            f
        )
    return {
        "seq ((t0+t1)+t2)+t3": seq,
        "pair (t0+t2)+(t1+t3)": p02,
        "pair (t0+t1)+(t2+t3)": p01,
        "rev ((t3+t2)+t1)+t0": rev,
        "fma chain": acc,
    }


for name, mk in (
    ("bf16_stream_like_code", lambda: (torch.randn(4096, 1, 4, 1024) * 3).to(torch.bfloat16)),
    ("fp32_random", lambda: torch.randn(4096, 1, 4, 1024) * 3),
):
    x = mk()
    pre = torch.sigmoid(torch.randn(4096, 1, 4) * 2) + 1e-6
    out = (
        torch.sum(pre.cuda().unsqueeze(-1) * x.cuda(), dim=2).cpu().numpy()
    )  # 与 hy4_mcore.Hy4HyperConnection 同一写法
    xn = x.float().numpy()
    pn = pre.numpy()
    r = {}
    for k, v in variants(pn, xn).items():
        r[k] = {"bitwise_equal_frac": float((v == out).mean()), "n": int(out.size)}
    res[name] = r
# 同一写法再跑一次，看是否确定
x = (torch.randn(4096, 1, 4, 1024) * 3).to(torch.bfloat16).cuda()
pre = (torch.sigmoid(torch.randn(4096, 1, 4) * 2) + 1e-6).cuda()
a = torch.sum(pre.unsqueeze(-1) * x, dim=2)
b = torch.sum(pre.unsqueeze(-1) * x, dim=2)
res["rerun_bitwise_identical"] = bool(torch.equal(a, b))
print("PROBE_JSON " + json.dumps(res))
