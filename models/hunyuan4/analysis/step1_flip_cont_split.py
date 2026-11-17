"""第 1 步梯度对 FP64 的误差拆成「离散选择不同」与「连续舍入」两部分，按张量逐份给 rel_L2 与沿真值方向的分量。

对一个被测 X（TPU、GPU、CPU 上 GPU 写法的代理）：
- 自选误差 = X − FP64 自选（`step1_ref.py`）；
- 连续误差 = X − FP64 灌 X 自己的选择（`step1_ref.py --inject-dir`；X 自己也用这份选择时才是纯连续误差）；
- 翻转部分 = FP64 灌 X 的选择 − FP64 自选（离散选择不同本身带来的梯度差，与舍入无关）。
三者满足 自选误差 = 连续误差 + 翻转部分（逐元素恒等）。每一项给 rel = ‖·‖ ÷ ‖FP64 自选‖ 与
par = (·)·g ÷ ‖g‖²（g 为 FP64 自选；正 = 沿真值方向多出）。

    python step1_flip_cont_split.py --self <FP64 自选 grads_hf> \\
        --case 名字=<被测梯度>=<FP64 灌同一份选择的 grads_hf>   # 连续误差可算
        --case 名字=<被测梯度>                                  # 只有自选误差
        --tensors <张量名,...> --out <json>
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from safetensors import safe_open


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self", dest="self_path", required=True)
    p.add_argument("--case", nargs="+", required=True)
    p.add_argument("--tensors", required=True)
    p.add_argument("--note", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    F = safe_open(a.self_path, "np")
    cases = []
    for c in a.case:
        parts = c.split("=")
        cases.append((parts[0], safe_open(parts[1], "np"), safe_open(parts[2], "np") if len(parts) > 2 else None))
    out = {"what": "第 1 步梯度对 FP64 的误差拆成离散选择（翻转）与连续舍入两部分", "rule": __doc__.split("\n\n")[1],
           "note": a.note, "inputs": {c.split("=")[0]: c.split("=")[1:] for c in a.case}, "rows": {}}
    for k in a.tensors.split(","):
        g = F.get_tensor(k).astype(np.float64).ravel()
        n2 = float(g @ g)

        def rp(e):
            return {"rel": float(np.linalg.norm(e) / np.sqrt(n2)), "par": float(e @ g / n2)}

        row = {}
        for name, X, I in cases:
            x = X.get_tensor(k).astype(np.float64).ravel()
            r = {"self": rp(x - g)}
            if I is not None:
                i = I.get_tensor(k).astype(np.float64).ravel()
                r["cont"], r["flip"] = rp(x - i), rp(i - g)
            row[name] = r
        out["rows"][k] = row
        print(k)
        for name, r in row.items():
            s = f"   {name:14s} self rel {r['self']['rel']:.3e} par {r['self']['par']:+.3e}"
            if "cont" in r:
                s += f" | cont rel {r['cont']['rel']:.3e} par {r['cont']['par']:+.3e} | flip rel {r['flip']['rel']:.3e} par {r['flip']['par']:+.3e}"
            print(s)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
