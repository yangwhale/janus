"""把融合的 q/k/v 投影权重梯度按 KV 组拆成 q / k / v 三块，各算离 FP64 的 rel_L2（第 3 节）。

布局：linear_qkv.weight 是 [10240, 4096]，行按 8 个 KV 组交错，每组 8 个 q 头、1 个 k 头、1 个 v 头，每头 128 行
（models/hunyuan3/model/name_map.yaml）。梯度由一阶动量反推，同 cp_step1_fp64.py。

  python qkv_split.py --fp64 <fp64_step 输出目录> --run 名=<运行目录> … --out step1-qkv-split.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from janus.store import Run  # noqa: E402


def grad(run, n):
    oc = run.meta.get("optimizer_config") or {}
    clip, gn = float(oc.get("clip_grad") or 0), float(run.series("grad_norm")[1])
    c = 1.0 if (clip <= 0 or gn < clip) else clip / gn
    return run.tensor(1, "exp_avg." + n[len("grad."):]).astype(np.float64) / (1 - float(oc["adam_beta1"])) / c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp64", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    f = safe_open(str(Path(a.fp64) / "grads.safetensors"), "numpy")
    runs = {k: Run.open(v) for k, v in (x.split("=", 1) for x in a.run)}
    parts = {"q": slice(0, 8), "k": slice(8, 9), "v": slice(9, 10)}
    res = {}
    for L in range(4):
        n = f"grad.decoder.layers.{L}.self_attention.linear_qkv.weight"
        ref = f.get_tensor(n).astype(np.float64).reshape(8, 10, 128, -1)
        res[f"layer{L}"] = {}
        for k, run in runs.items():
            x = grad(run, n).reshape(ref.shape)
            res[f"layer{L}"][k] = {p: float(np.linalg.norm(x[:, s] - ref[:, s]) / np.linalg.norm(ref[:, s])) for p, s in parts.items()}
        print(L, {k: {p: f"{v:.2e}" for p, v in d.items()} for k, d in res[f"layer{L}"].items()}, flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
