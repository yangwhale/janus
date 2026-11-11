"""L2 逐层隔离回放对 CPU FP64 参照（`frameworks/megatron/layer_ref.py`），替换原来的 TPU FP32 + HIGHEST 参照。

同一套输入（GPU 逐层抓数，如 `g7-l1-a`：每层输入与输出的余切）下，每层两个量：层输出、层输入的梯度。
- GPU：抓数里的 `out0`、`in0.grad`（GPU 整网第 1 步里这一层真实的输出与梯度，输入就是喂给回放的那份）；
- TPU：`layer_replay.py --save` 存下的 `tpu.<层>.out` / `tpu.<层>.in_grad`；
- 旧参照：同一文件的 `ref.<层>.*`（TPU 同一程序的 FP32 + HIGHEST，被测一侧产出）。
R = rel_L2(TPU, FP64) ÷ rel_L2(GPU, FP64)，≤ 1.2 过（PROTOCOL 第 5 节，L2）。

    python models/hunyuan4/analysis/l2_vs_fp64.py --fp64 <layer_ref.safetensors> --tpu tw-l2-g=<layer_replay.safetensors> \\
        --gpu-l1 <step_1_fwd.safetensors> --gpu-l1-bwd <step_1_bwd.safetensors> --out <json>
"""

from __future__ import annotations

import argparse
import json

import torch
from safetensors import safe_open

R_LIMIT = 1.2


def rel(a, b):
    a, b = a.to(torch.float64).reshape(b.shape), b.to(torch.float64)
    return float((a - b).norm() / b.norm())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp64", required=True)
    p.add_argument("--tpu", required=True, help="名字=layer_replay.safetensors")
    p.add_argument("--gpu-l1", required=True)
    p.add_argument("--gpu-l1-bwd", required=True)
    p.add_argument("--note", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    ref = safe_open(a.fp64, "pt")
    tname, tpath = a.tpu.split("=", 1)
    t = safe_open(tpath, "pt")
    gf, gb = safe_open(a.gpu_l1, "pt"), safe_open(a.gpu_l1_bwd, "pt")
    meta = json.load(open(a.fp64.rsplit(".", 1)[0] + ".json"))
    rows = []
    for k in sorted({x.split(".")[0] for x in ref.keys()}, key=lambda x: (x == "mtp0", x)):
        pre = "hf.model.mtp_layers.0" if k == "mtp0" else f"hf.model.layers.{k}"
        r = {"layer": k}
        for q, gname, g in (("out", pre + ".out0", gf), ("in_grad", pre + ".in0.grad", gb)):
            f64 = ref.get_tensor(f"{k}.{q}")
            gv = g.get_tensor(gname)[0]
            r[f"{q}.gpu_vs_fp64"] = rel(gv, f64)
            r[f"{q}.tpu_vs_fp64"] = rel(t.get_tensor(f"tpu.{k}.{q}"), f64)
            r[f"{q}.old_ref_vs_fp64"] = rel(t.get_tensor(f"ref.{k}.{q}"), f64)
            r[f"{q}.R"] = r[f"{q}.tpu_vs_fp64"] / r[f"{q}.gpu_vs_fp64"]
            gr = rel(gv, t.get_tensor(f"ref.{k}.{q}").reshape(f64.shape))
            r[f"{q}.old_R"] = rel(t.get_tensor(f"tpu.{k}.{q}"), t.get_tensor(f"ref.{k}.{q}")) / gr
        rows.append(r)
    worst = max(((r["layer"], q, r[f"{q}.R"]) for r in rows for q in ("out", "in_grad")), key=lambda x: x[2])
    out = {
        "what": f"L2 逐层隔离回放：TPU（{tname}）与 GPU（逐层抓数）各对 CPU FP64 单层重算",
        "reference": {
            "kind": "CPU FP64，GPU golden 同一套 Megatron-Core 代码（frameworks/megatron/layer_ref.py）",
            **{k: meta[k] for k in ("seconds", "audit", "dtype_rewrites")},
        },
        "rule": "R = rel_L2(TPU, FP64) ÷ rel_L2(GPU, FP64) ≤ 1.2 过；逐层逐量取最差",
        "worst": worst,
        "verdict": "pass" if worst[2] <= R_LIMIT else "fail",
        "rows": rows,
        "note": a.note,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for r in rows:
        print(
            r["layer"],
            *(
                f"{q}: GPU {r[q + '.gpu_vs_fp64']:.2e} TPU {r[q + '.tpu_vs_fp64']:.2e} R {r[q + '.R']:.2f} (旧 {r[q + '.old_R']:.2f}, 旧参照离 FP64 {r[q + '.old_ref_vs_fp64']:.1e})"
                for q in ("out", "in_grad")
            ),
        )
    print("WORST", worst, out["verdict"])


if __name__ == "__main__":
    main()
