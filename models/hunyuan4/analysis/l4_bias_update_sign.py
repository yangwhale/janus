"""路由偏置更新方向的同状态对拍：同一个参数状态、同一批 token 上，TPU、GPU 各份、CPU FP64 各自选专家，
各算一次本步的路由偏置更新方向，看 TPU 是否比 GPU 更常和 FP64 不一致。

为什么看方向：两边的偏置更新都是 b_e ← b_e + rate · sign(平均负载 − 专家 e 的负载)（rate 1e-3，Megatron-Core 原版写法，
MaxText 照它；PROTOCOL 第 1.2 节），负载 = 全局批里选中专家 e 的 token 数。偏置标准差的逐步差只能来自这一步的
sign 向量不同。同一状态下的 sign 向量谁离 FP64 远，就是这一步偏置更新谁不准。

输入与路由一致率相同（routing_agreement.py）：第 1 步从初始权重、第 8、30、100 步从 GPU g2w-b 该步重建的状态出发，
三方各自选（TPU --capture-choices；GPU 每行的 routing_map；FP64 step1_ref.py 的 choices.safetensors）；
行按 token 逐个相等对上 FP64 的行序。

  python l4_bias_update_sign.py --case 8 --fp64 <fp64 目录> --gpu s8-a=<目录> … --tpu tw-ra8=<紧凑文件> … --out <json>
（--case 可以给多次，参数写成 --case 8:<fp64>:<gpu,…>:<tpu>）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routing_agreement import Compact, GpuDir  # noqa: E402

LAYERS = ["1", "2", "3", "4", "5", "6", "mtp0"]
E = 16


def fp64_rows(d):
    h = safe_open(os.path.join(d, "choices.safetensors"), "np")
    return h


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", action="append", required=True, help="步:FP64 目录:GPU 名=目录,…:TPU 名=紧凑文件:tokens.npy")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out, tpu_names = [], set()
    for case in a.case:
        step, f64, gpus, tpu, toks = case.split(":")
        toks = np.load(toks)
        rows = 16
        seq = None
        h64 = fp64_rows(f64)
        srcs = {}
        for spec in gpus.split(","):
            n, d = spec.split("=")
            srcs[n] = ("gpu", GpuDir(d, 1))
        tn, tf = tpu.split("=")
        tpu_names.add(tn)
        srcs[tn] = ("tpu", Compact(tf, 7, E))
        # FP64 第 r 行的 token = tokens.npy 第 r 行的前 seq 个
        for _kind, s in srcs.values():
            seq = len(s.tokens[0])
            break
        f64_tokens = [toks[r][:seq] for r in range(rows)]
        row_map = {}
        for n, (_kind, s) in srcs.items():
            m = []
            for r in range(rows):
                hit = [i for i, t in enumerate(s.tokens) if np.array_equal(np.asarray(t).reshape(-1), f64_tokens[r])]
                if len(hit) != 1:
                    raise SystemExit(f"{n}：FP64 第 {r} 行对不上（{len(hit)} 个匹配）")
                m.append(hit[0])
            row_map[n] = m
        for L in LAYERS:
            ld = {}
            ld["fp64"] = np.zeros(E, np.int64)
            for r in range(rows):
                idx = h64.get_tensor(f"r{r}.route.{L}.idx").astype(np.int64)
                ld["fp64"] += np.bincount(idx.reshape(-1), minlength=E)
            for n, (_kind, s) in srcs.items():
                ld[n] = np.zeros(E, np.int64)
                for r in range(rows):
                    ld[n] += s.route(row_map[n][r], L).sum(0)
            sign = {k: np.sign(v.mean() - v) for k, v in ld.items()}
            row = {"step": int(step), "layer": L, "loads": {k: v.tolist() for k, v in ld.items()},
                   "sign_differ_vs_fp64": {k: int((sign[k] != sign["fp64"]).sum()) for k in ld if k != "fp64"},
                   "load_l1_vs_fp64": {k: int(np.abs(ld[k] - ld["fp64"]).sum()) for k in ld if k != "fp64"}}
            out.append(row)
            print(step, L, row["sign_differ_vs_fp64"], row["load_l1_vs_fp64"])
    agg = {}
    for r in out:
        t = [v for k, v in r["sign_differ_vs_fp64"].items() if k in tpu_names][0]
        g = [v for k, v in r["sign_differ_vs_fp64"].items() if k not in tpu_names]
        lt = [v for k, v in r["load_l1_vs_fp64"].items() if k in tpu_names][0]
        lg = [v for k, v in r["load_l1_vs_fp64"].items() if k not in tpu_names]
        agg.setdefault(r["layer"], {"tpu_sign_differ": 0, "gpu_sign_differ_each": None, "tpu_load_l1": 0, "gpu_load_l1_each": None})
        A = agg[r["layer"]]
        A["tpu_sign_differ"] += t
        A["gpu_sign_differ_each"] = [x + y for x, y in zip(A["gpu_sign_differ_each"] or [0] * len(g), g)]
        A["tpu_load_l1"] += lt
        A["gpu_load_l1_each"] = [x + y for x, y in zip(A["gpu_load_l1_each"] or [0] * len(lg), lg)]
    res = {"what": "路由偏置更新方向（sign(平均负载 − 负载)）同状态对拍：TPU、GPU 各份各自选专家，对 CPU FP64 自选", "rows": out,
           "per_layer_sum_over_steps": agg}
    open(a.out, "w").write(json.dumps(res, ensure_ascii=False, indent=1))
    for L, A in agg.items():
        print(L, A)


if __name__ == "__main__":
    main()
