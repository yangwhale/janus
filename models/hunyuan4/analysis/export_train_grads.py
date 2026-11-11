"""从 TPU 全量抓数（`--dump-steps 1 --full` 的 step_1/）反推训练第 1 步实际用的梯度，写成 HF 名字的 safetensors。

反推同 `tpu_self_compare.train_grad`：Adam 参数 g = m / (1 − β1)，Muon 参数 g = buf / (1 − β)，再乘回裁剪系数
max(1, raw_grad_norm)（第 1 步 raw_grad_norm 取运行目录 metrics.jsonl 第 1 步）。读张量时逐个核 sha256。
输出给 `step1_vs_fp64.py --tpu 名字=<输出>` 用（L3 正式判定用训练步梯度，PROTOCOL 第 5 节 2026-11-09）。

    python models/hunyuan4/analysis/export_train_grads.py --run <运行目录> --out <safetensors>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tpu_self_compare as tsc  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    with open(os.path.join(a.run, "metrics.jsonl")) as f:
        m1 = next(json.loads(line) for line in f if json.loads(line)["step"] == 1)
    d = tsc.Dump(a.run)
    names = sorted(n for (k, n) in d.idx if k == "exp_avg")
    out = {n: torch.from_numpy(np.ascontiguousarray(tsc.train_grad(d, n, m1["raw_grad_norm"]))).float() for n in names}
    save_file(
        out,
        a.out,
        metadata={"run": os.path.basename(os.path.normpath(a.run)), "raw_grad_norm": str(m1["raw_grad_norm"])},
    )
    print(len(out), "tensors, raw_grad_norm", m1["raw_grad_norm"])


if __name__ == "__main__":
    main()
