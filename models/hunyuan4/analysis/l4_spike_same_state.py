"""grad_norm 尖峰段的同状态单步：GPU 按 g3 的设置重跑到尖峰（g3x-a，第 849、900 步存 FP32 主权重、梯度、路由偏置），
从这两个状态出发 TPU、GPU（同状态重跑两次）、CPU FP64 各走一步（不更新），比 grad_norm 与本步路由偏置的更新方向。

输入：GPU 原运行的 metrics.jsonl（该步 grad_norm）、同状态 GPU 重跑的 metrics.jsonl、TPU 诊断运行的 diag_step1.json
（grad_norm 在主机上按 FP64 求和）、FP64 的 summary.json；偏置方向来自 l4_bias_update_sign.py 的输出。
rel = (x − FP64) / FP64；R = |rel(TPU)| ÷ GPU 各份 |rel| 的中位数。

  python l4_spike_same_state.py --spec <json> --bias-sign <l4_bias_update_sign 的输出> --out <json>
  spec：[{step, gpu_orig_metrics, gpu_reruns: {名: metrics.jsonl}, tpu_diag: diag_step1.json, tpu_run, fp64_summary}]
"""
from __future__ import annotations

import argparse
import json
import statistics as st


def gn_at(metrics, step=None):
    for line in open(metrics):
        r = json.loads(line)
        if step is None or r["step"] == step:
            return r["grad_norm"]
    raise KeyError(step)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", required=True)
    p.add_argument("--bias-sign", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    bs = json.load(open(a.bias_sign))
    rows = []
    for s in json.load(open(a.spec)):
        k = s["step"]
        f = json.load(open(s["fp64_summary"]))["grad_norm"]
        g = {"orig": gn_at(s["gpu_orig_metrics"], k), **{n: gn_at(m) for n, m in s["gpu_reruns"].items()}}
        t = json.load(open(s["tpu_diag"]))["grad_norm"]
        rg = {n: (v - f) / f for n, v in g.items()}
        rt = (t - f) / f
        med = st.median(abs(x) for x in rg.values())
        b = [r for r in bs["rows"] if r["step"] == k]
        tn = s["tpu_run"]
        sign_t = sum(r["sign_differ_vs_fp64"][tn] for r in b)
        sign_g = {n: sum(r["sign_differ_vs_fp64"][n] for r in b) for n in s["gpu_reruns"]}
        l1_t = sum(r["load_l1_vs_fp64"][tn] for r in b)
        l1_g = {n: sum(r["load_l1_vs_fp64"][n] for r in b) for n in s["gpu_reruns"]}
        rows.append({"step": k, "fp64_grad_norm": f, "gpu_grad_norm": g, "tpu_grad_norm": t, "gpu_rel": rg, "tpu_rel": rt,
                     "R_grad_norm": abs(rt) / med, "tpu_run": tn,
                     "bias_sign_differ_vs_fp64": {"tpu": sign_t, "gpu": sign_g, "layers": len(b), "experts_per_layer": 16},
                     "load_l1_vs_fp64_sum": {"tpu": l1_t, "gpu": l1_g},
                     "per_layer": [{"layer": r["layer"], "sign_differ": r["sign_differ_vs_fp64"], "load_l1": r["load_l1_vs_fp64"]} for r in b]})
        print(rows[-1]["step"], f"FP64 {f:.6g}", {n: f"{v:+.3e}" for n, v in rg.items()}, f"TPU {rt:+.3e}", f"R {rows[-1]['R_grad_norm']:.3f}",
              "sign", sign_t, sign_g, "L1", l1_t, l1_g)
    out = {"what": "grad_norm 尖峰段（第 849、900 步）同状态单步：TPU、GPU、CPU FP64 从同一状态各走一步（不更新）",
           "state": "GPU g3x-a（g3 的设置重跑，这两步 grad_norm 11.41、10.81，与 g3-a 的 11.41、10.96 同为尖峰）该步更新前的 FP32 主权重与路由偏置；"
                    "GPU 重跑与 FP64 用按 GPU 参数实际 dtype 重建的 HF 目录（analysis/build_hf_gpu_dtype.py），TPU 用 --param-override / --bias-override",
           "gpu_choices_note": "GPU 原运行抓的路由选择上传不全（849 步缺 1 行、900 步缺 2 行），偏置更新方向只用两次同状态重跑",
           "definition": "rel = (x − FP64) / FP64；R = |rel(TPU)| ÷ GPU 各份 |rel| 的中位数；偏置更新方向 = 每层 16 个专家 sign(平均负载 − 负载)",
           "rows": rows}
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
