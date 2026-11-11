"""B200 上 FP32 log / exp 的实测 ULP（对 FP64 正确舍入），与 TPU 探针（frameworks/maxtext/log_ulp_probe.py）同一批输入。

    NVIDIA_TF32_OVERRIDE=0 python3 gpu_log_probe.py [--kd elem-kd.npy] --out probe.json     # 训练镜像里、单卡、一两分钟

- log：torch.log（FP32，CUDA；PyTorch eager 的逐元素 log，GPU golden 里 log_softmax / KD 的 log 走同一类算子），
  输入 = analysis/log_probe_inputs.make_inputs()（[0.5, 4) 全部 FP32、对数均匀两段），可选 KD 的真实目标分布；
- exp：torch.exp（FP32），输入 unif[-87, 88] 与 unif[-10, 10] 各 2^22（default_rng(4321)），(-1, -0.5] 全部 FP32；
- 另记 torch.log_softmax 不在这里比（LSE 的误差由 exp、求和、log 三步合成，TPU 侧的拆解见 tpu-elementwise-ulp.json）。
ULP = |out − FP64 正确舍入到 FP32| 的 FP32 位模式有序距离（与 TPU 探针同一个函数）。打印一行 PROBE_JSON 并写 --out。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from log_probe_inputs import make_inputs  # noqa: E402


def ulp_dist(out: np.ndarray, ref64: np.ndarray) -> np.ndarray:
    cr = ref64.astype(np.float32)

    def key(a):
        i = a.view(np.int32).astype(np.int64)
        return np.where(i < 0, -(i & 0x7FFFFFFF), i)
    return np.abs(key(np.ascontiguousarray(out, np.float32)) - key(cr))


def stats(out, ref):
    keep = np.isfinite(ref) & (np.abs(ref) < 3.4e38) & ((ref == 0) | (np.abs(ref) >= 1.1754944e-38))  # 结果是次正规数的不比
    d = ulp_dist(out[keep], ref[keep])
    return {"n": int(d.size), "ulp_max": int(d.max()), "ulp_p999": float(np.quantile(d, 0.999)), "ulp_mean": float(d.mean()),
            "cr_match_frac": float((d == 0).mean()), "n_gt1": int((d > 1).sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kd", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = os.environ.get("PROBE_DEVICE", "cuda")
    res = {"gpu": torch.cuda.get_device_name() if dev == "cuda" else dev, "torch": torch.__version__, "cuda": torch.version.cuda, "log": {}, "exp": {}}
    X = make_inputs()
    if a.kd:
        X["real.kd_target"] = np.load(a.kd).astype(np.float32)
    res["inputs_sha256"] = {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in X.items()}
    for k, x in X.items():
        out = torch.log(torch.from_numpy(x).to(dev)).cpu().numpy()
        res["log"][k] = stats(out, np.log(x.astype(np.float64)))
    rng = np.random.default_rng(4321)
    lo, hi = np.float32(-1.0).view(np.uint32), np.float32(-0.5).view(np.uint32)
    E = {"unif[-87,88]": rng.uniform(-87, 88, 1 << 22).astype(np.float32), "unif[-10,10]": rng.uniform(-10, 10, 1 << 22).astype(np.float32),
         "all(-1,-0.5]": np.arange(hi, lo, dtype=np.uint32).view(np.float32)}
    for k, x in E.items():
        out = torch.exp(torch.from_numpy(x).to(dev)).cpu().numpy()
        res["exp"][k] = stats(out, np.exp(x.astype(np.float64)))
    open(a.out, "w").write(json.dumps(res, ensure_ascii=False, indent=1))
    print("PROBE_JSON " + json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
