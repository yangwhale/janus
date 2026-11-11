"""权重替换后的单算子回放（op_replay_mini.py --sweep sweeps/weight_switches.json --param-override）：输出头、路由 logits、
索引器 weights_proj 三个纯矩阵乘，TPU 输出各对两种 FP64 参照的 rel_L2——
- FP64，权重取替换上的 FP32 主权重（GPU g2o-a 第 8 步 main_param_after）；
- FP64，同一份权重先舍成 BF16（GPU 前向用的是 BF16 的模型权重时，算的就是这个）。
输入是 GPU 第 1 步抓到的算子输入（都在 BF16 网格上）。数字只从 TPU 输出与这两份权重算，不比 GPU 输出（GPU 用的是第 1 步权重）。

  python models/hunyuan4/analysis/weight_override_refs.py --run <tw-sw-ovr 目录> --gpu-fwd <step_1_fwd.safetensors> \\
      --weights <g2o-a.s8.main_param_after.hf.safetensors> --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ml_dtypes
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frameworks" / "maxtext"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--gpu-fwd", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--groups", default="base,M13,M27,M8")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    import op_replay_mini as m

    from janus import metrics as M

    keep = lambda k: k == "lm_head.weight" or k.endswith(("weights_proj.weight", "mlp.gate.weight"))  # noqa: E731
    P = {}
    for k, v in m.load_st(a.weights).items():
        for pre in ("main_param_after.", "main_param.", "param."):
            k = k[len(pre) :] if k.startswith(pre) else k
        if keep(k):
            P[k] = v
    Pb = {k: np.asarray(v, np.float32).astype(ml_dtypes.bfloat16).astype(np.float32) for k, v in P.items()}
    G = m.load_st(a.gpu_fwd)
    out = {}
    for n in a.groups.split(","):
        T = m.load_st(str(Path(a.run) / f"op_replay_mini.{n}.safetensors"))
        rec = json.loads((Path(a.run) / f"op_replay_mini.{n}.json").read_text())
        for op, t in T.items():
            r32, rb = m.override_fp64(op, G, P), m.override_fp64(op, G, Pb)
            if r32 is None:
                continue
            t = np.asarray(t, np.float64).reshape(-1)[: r32.size]
            out.setdefault(op, {})[n] = {
                "overrides": rec.get("overrides"),
                "vs_fp64_fp32_weights": M.rel_l2(t, r32.reshape(-1)),
                "vs_fp64_bf16_weights": M.rel_l2(t, rb.reshape(-1)),
            }
    res = {
        "what": "权重替换后的单算子回放：TPU 输出对两种 FP64 参照（FP32 主权重 / 同一份权重舍成 BF16）的 rel_L2",
        "weights": "GPU g2o-a 第 8 步的 FP32 主权重（main_param_after）",
        "inputs": "GPU g7-l1-a 第 1 步抓到的算子输入（BF16 网格上）",
        "ops": out,
    }
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    for op, v in out.items():
        print(op, {k: f"{x['vs_fp64_fp32_weights']:.2e} / {x['vs_fp64_bf16_weights']:.2e}" for k, x in v.items()})


if __name__ == "__main__":
    main()
