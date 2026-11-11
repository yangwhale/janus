"""单层参数梯度对 CPU FP64（定位 L3 第 1 步超线张量的舍入点）：同一层、同一行（GPU rank 0 第 0 个 microbatch）、同一个余切。

三方：
- GPU：逐算子抓数（hy4_capture.py，`--l1-capture-step 1 --l1-grad-micro0`）的 step_1_grad_micro0.safetensors：第 0 个
  microbatch 反向做完时 rank 0 各参数的 main_grad（Megatron 名字；路由专家只有本 rank 的 0–3 号）。整网反向里这一层收到
  的输入与余切就是抓数里的 in0 与 out0.grad；
- TPU：frameworks/maxtext/layer_replay.py --param-grads，吃同一份 in0 与 out0.grad（<out>.param_grads.safetensors，键 tpu.<HF 名>）；
- FP64：frameworks/megatron/layer_ref.py --param-grads（键 <层>.param.<层内 Megatron 名>，路由专家 16 个）。
R = rel_L2(TPU, FP64) ÷ rel_L2(GPU, FP64)。索引器参数的 KD 梯度不在单层里（GPU 的 main_grad 有、单层回放没有），不比。

  python l2_param_grads.py --gpu <step_1_grad_micro0.safetensors> --tpu <param_grads.safetensors> --fp64 <layer_ref.safetensors> \
      --hf-config <config.json> --layers 3 --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frameworks", "megatron"))


def npy(t):
    return t.double().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", required=True)
    p.add_argument("--tpu", required=True)
    p.add_argument("--fp64", required=True)
    p.add_argument("--hf-config", required=True)
    p.add_argument("--layers", default="3")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from export_hf_grads import mcore_to_hf

    cfg = json.load(open(a.hf_config))
    L = cfg["num_hidden_layers"]
    table = mcore_to_hf(SimpleNamespace(num_layers=L, hy4_indexer_types=cfg["indexer_types"][:L],
                                        moe_layer_freq=[int(t == "sparse") for t in cfg["mlp_layer_types"][:L]],
                                        mtp_num_layers=int(cfg.get("num_nextn_predict_layers", 0)),
                                        num_moe_experts=cfg["n_routed_experts"]))
    G = safe_open(a.gpu, "pt")
    T = safe_open(a.tpu, "pt")
    F = safe_open(a.fp64, "pt")
    gk, tk, fk = set(G.keys()), set(T.keys()), set(F.keys())

    def mc_to_hf(full_name, x):
        """Megatron 全名的一个张量 → {HF 名: 数组}（fc1 拆 gate / up；专家给 (HF 名, 专家号)）。"""
        m = re.match(r"^(.*mlp\.experts)\.(?:local_experts\.(\d+)\.(linear_fc[12])\.weight|(linear_fc[12])\.weight(\d+))$", full_name)
        if m:
            e = int(m.group(2) if m.group(2) is not None else m.group(5))
            which = m.group(3) or m.group(4)
            h, _tf = table[f"{m.group(1)}.{which}"]
            return {(h, e): x}
        if full_name not in table:
            return {}
        h, tf = table[full_name]
        if tf == "cat":
            n = x.shape[0] // 2
            return {(h[0], None): x[:n], (h[1], None): x[n:]}
        return {(h, None): x}

    rows = []
    for k in a.layers.split(","):
        pre = f"decoder.layers.{k}."
        gpu = {}
        for n in gk:
            if n.startswith(pre):
                gpu.update(mc_to_hf(n, npy(G.get_tensor(n))))
        f64 = {}
        for n in fk:
            if n.startswith(f"{k}.param."):
                f64.update(mc_to_hf(pre + n[len(f"{k}.param."):], npy(F.get_tensor(n))))
        hpre = f"model.layers.{k}."
        tpu = {n[4:]: npy(T.get_tensor(n)) for n in tk if n.startswith("tpu." + hpre)}
        for (h, e), gx in sorted(gpu.items(), key=lambda t: (t[0][0], -1 if t[0][1] is None else t[0][1])):
            if ".indexer." in h:
                continue  # KD 梯度不在单层里
            fx = f64.get((h, e))
            tx = tpu.get(h)
            if fx is None or tx is None:
                continue
            if e is not None:
                tx = tx[e]
            fn = float(np.linalg.norm(fx))
            eg, et = float(np.linalg.norm(gx - fx)) / fn, float(np.linalg.norm(tx - fx)) / fn
            rows.append({"layer": k, "tensor": h + (f"[{e}]" if e is not None else ""), "gpu_vs_fp64": eg, "tpu_vs_fp64": et,
                         "R": et / eg if eg else None, "tpu_vs_gpu": float(np.linalg.norm(tx - gx)) / float(np.linalg.norm(gx))})
    rows.sort(key=lambda r: -(r["R"] or 0))
    out = {"what": "单层参数梯度（同一行、同一份输入与余切）对 CPU FP64：GPU 第 0 个 microbatch 的 main_grad、TPU 单层回放",
           "rule": "R = rel_L2(TPU, FP64) ÷ rel_L2(GPU, FP64)", "rows": rows}
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    for r in rows[:25]:
        print(f"{r['tensor']:55s} GPU {r['gpu_vs_fp64']:.3e} TPU {r['tpu_vs_fp64']:.3e} R {r['R']:.3f}")


if __name__ == "__main__":
    main()
