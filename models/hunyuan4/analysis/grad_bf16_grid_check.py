"""诊断：训练第 1 步实际用的梯度（`export_train_grads.py` 由动量反推）里，哪些张量整个落在 BF16 格点上。

为什么：梯度累积 1 时，TPU 训练步不经过 FP32 的累积函数；如果某些参数的梯度在设备上以 BF16 归约、或在 BF16 里产出后才交给
FP32 优化器，反推出的梯度就整个落在 BF16 格点上（反推只多 FP32 的几次舍入，相对 ≤ 1e-6 量级）。累积 2 时两份微批在 FP32 里相加，
即使每份都舍过 BF16，和也不在格点上，所以这一条只在累积 1 时看得出来。
逐张量：非零元素里 |g − bf16(g)| ≤ tol·|g| 的比例（tol 默认 1e-5，远大于反推的 FP32 舍入、远小于一次 BF16 舍入 2⁻⁹）；> 0.99 记「在格点上」。
大张量只取前 --max-elems 个元素。

    python grad_bf16_grid_check.py --run 名字=<export_train_grads.py 的 safetensors> [--run …] --out <json>
"""

from __future__ import annotations

import argparse
import json

import torch
from safetensors import safe_open


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, help="名字=safetensors，可给多次")
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--max-elems", type=int, default=2_000_000)
    p.add_argument("--note", action="append", default=[], help="名字=说明（梯度累积数、配置等），可给多次")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    notes = dict(x.split("=", 1) for x in a.note)
    out = {"what": __doc__.splitlines()[0], "tol": a.tol, "max_elems": a.max_elems, "runs": {}}
    for spec in a.run:
        name, path = spec.split("=", 1)
        f = safe_open(path, "pt")
        rows = []
        for n in sorted(f.keys()):
            g = f.get_tensor(n).double().ravel()[: a.max_elems]
            nz = g != 0
            if not bool(nz.any()):
                continue
            b = g.to(torch.bfloat16).double()
            frac = float(((g - b).abs() <= a.tol * g.abs())[nz].double().mean())
            rows.append({"tensor": n, "frac_on_bf16_grid": frac, "on_grid": frac > 0.99})
        on = [r["tensor"] for r in rows if r["on_grid"]]
        out["runs"][name] = {
            "note": notes.get(name, ""),
            "n_tensors": len(rows),
            "n_on_grid": len(on),
            "on_grid": on,
            "rows": rows,
        }
        print(f"{name}: {len(on)} / {len(rows)} 个张量整个落在 BF16 格点上")
    with open(a.out, "w") as fo:
        json.dump(out, fo, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
