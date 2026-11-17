"""低阈值 QK-Clip 第 1 步逐头最大 logit 超线的定责（诊断，不改判档）：最大那一对 (q, k) 的输入是不是被离散选择（路由、DSA）翻转改了。

输入（都是同一份第 1 步数据、同一份初始权重）：
- `--fp64-topk`：`qkclip_maxlogit_topk.py` 在 CPU FP64 前向（自选路由与 DSA）里记的逐层逐头前 K 大（值与 行 / query / key 位置）；
- `--fp64-inj-topk`：同一脚本、同一前向，但全局批每一行都灌 TPU 自选的路由与 DSA top-k（`step1_ref.py --inject-dir`）；
- `--fp64-summary` / `--fp64-inj-summary`：两次前向的 `summary.json`（`max_logit` 逐层逐头）；
- `--fp64-choices`：FP64 自选前向存的路由（`choices.safetensors`，`r<行>.route.<层>.idx / scores`）；
- `--tpu-choices`：TPU `--diag-step1 --capture-choices` 抓到的自选选择（`choices_captured.safetensors`）；
- `--tpu` / `--gpu`：被测与 GPU 的 metrics.jsonl（第 1 步 `qk.max_logit.<层>`）；
- `--pairs`：`qkclip_maxlogit_topk.py --pairs` 存的各层各头最大一对的 query / key 向量（FP64），算条件数
  κ = Σ|q_i k_i| ÷ |Σ q_i k_i| 与 q、k 各舍一次 BF16 对这一对的影响。
输出：每层 R（对 FP64 自选；另报对「FP64 灌 TPU 选择」的相对差，分母同样取 GPU 两份中位数，只作诊断）；指定层、头的前几名与
被换掉的那一对两个 token 在前几层 MoE 的路由（FP64 对 TPU、FP64 第 4、5 名分差）。

    python qkclip_flip_blame.py --layer 4 --head 7 --fp64-topk … --fp64-inj-topk … --fp64-summary … --fp64-inj-summary … \\
        --fp64-choices … --tpu-choices … --tpu tc-qk01=<metrics> --gpu g6-qk-a=<…> --gpu g6-qk-b=<…> --pairs … --out <json>
"""

from __future__ import annotations

import argparse
import json
import statistics as st

import numpy as np

LAYERS = ["0", "1", "2", "3", "4", "5", "6", "mtp0"]


def step1(path):
    for line in open(path):
        r = json.loads(line)
        if r.get("step") == 1:
            return r
    raise SystemExit(f"{path} 没有第 1 步")


def rl(x, f):
    x, f = np.asarray(x, np.float64), np.asarray(f, np.float64)
    return float(np.linalg.norm(x - f) / np.linalg.norm(f))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in (
        "fp64-topk",
        "fp64-inj-topk",
        "fp64-summary",
        "fp64-inj-summary",
        "fp64-choices",
        "tpu-choices",
        "tpu",
        "out",
    ):
        p.add_argument("--" + k, required=True)
    p.add_argument("--gpu", action="append", required=True)
    p.add_argument("--pairs", default="")
    p.add_argument("--layer", default="4")
    p.add_argument("--head", type=int, default=7)
    p.add_argument("--moe-layers", default="1,2,3")
    a = p.parse_args()
    from safetensors import safe_open

    tname, tpath = a.tpu.split("=", 1)
    T = step1(tpath)
    G = {g.split("=", 1)[0]: step1(g.split("=", 1)[1]) for g in a.gpu}
    fs = json.load(open(a.fp64_summary))["max_logit"]
    fi = json.load(open(a.fp64_inj_summary))["max_logit"]
    per_layer = []
    for L in LAYERS:
        t = T[f"qk.max_logit.{L}"]
        gm = st.median(rl(g[f"qk.max_logit.{L}"], fs[L]) for g in G.values())
        per_layer.append(
            {
                "layer": L,
                "gpu_median_vs_fp64": gm,
                "tpu_vs_fp64": rl(t, fs[L]),
                "R": rl(t, fs[L]) / gm,
                "tpu_vs_fp64_tpu_choices": rl(t, fi[L]),
                "R_vs_fp64_tpu_choices": rl(t, fi[L]) / gm,
                "fp64_tpu_choices_vs_fp64": rl(fi[L], fs[L]),
                "per_head_rel_tpu": [float(x) for x in (np.array(t) - np.array(fs[L])) / np.array(fs[L])],
                "per_head_rel_gpu": {
                    n: [float(x) for x in (np.array(g[f"qk.max_logit.{L}"]) - np.array(fs[L])) / np.array(fs[L])]
                    for n, g in G.items()
                },
                "per_head_rel_tpu_vs_fp64_tpu_choices": [
                    float(x) for x in (np.array(t) - np.array(fi[L])) / np.array(fi[L])
                ],
            }
        )
    ln = str(int(a.layer) + 1)  # qkclip_maxlogit_topk 记的是 Megatron 的 layer_number（1 起）
    top_self = json.load(open(a.fp64_topk))["layers"][ln]["heads"][str(a.head)]
    top_inj = json.load(open(a.fp64_inj_topk))["layers"][ln]["heads"][str(a.head)]
    lost = top_self[0]
    fc = safe_open(a.fp64_choices, "pt")
    tc = safe_open(a.tpu_choices, "pt")
    RE, SC = tc.get_slice("forced_routed_experts"), tc.get_slice("router_scores")
    routes = []
    for tok in (lost["q"], lost["k"]):
        for L in [int(x) for x in a.moe_layers.split(",")]:
            fsc = fc.get_tensor(f"r{lost['row']}.route.{L}.scores").numpy().reshape(-1, 16)[tok].astype(np.float64)
            fo = np.argsort(-fsc)
            fidx = sorted(int(i) for i in fc.get_tensor(f"r{lost['row']}.route.{L}.idx").numpy().reshape(4096, -1)[tok])
            tidx = sorted(int(i) for i in RE[lost["row"] : lost["row"] + 1].numpy()[0][tok, L])
            tsc = SC[lost["row"] : lost["row"] + 1].numpy()[0][tok, L].astype(np.float64)
            topk = len(fidx)
            routes.append(
                {
                    "token": tok,
                    "moe_layer": L,
                    "fp64_experts": fidx,
                    "tpu_experts": tidx,
                    "flip": fidx != tidx,
                    "fp64_gap_k_k1": float(fsc[fo[topk - 1]] - fsc[fo[topk]]),
                    "fp64_gap_k_k1_rel": float((fsc[fo[topk - 1]] - fsc[fo[topk]]) / fsc[fo[topk - 1]]),
                    "fp64_scores_top6": [[int(i), float(fsc[i])] for i in fo[:6]],
                    "tpu_scores_top6": [[int(i), float(tsc[i])] for i in np.argsort(-tsc)[:6]],
                }
            )
    kappa = []
    if a.pairs:
        import torch

        pf = safe_open(a.pairs, "pt")
        for key in sorted(k for k in pf.keys() if k.endswith(".q")):
            tag = key[:-2]
            q, k, s = pf.get_tensor(tag + ".q"), pf.get_tensor(tag + ".k"), float(pf.get_tensor(tag + ".scale")[0])
            v = float((q * k).sum()) * s
            bf = lambda x: x.to(torch.bfloat16).double()  # noqa: E731
            kappa.append(
                {
                    "pair": tag,
                    "logit": v,
                    "kappa": float((q * k).abs().sum() / (q * k).sum().abs()),
                    "bf16_qk_rel": float((bf(q) * bf(k)).sum()) * s / v - 1,
                }
            )
    tv = T[f"qk.max_logit.{a.layer}"][a.head]
    out = {
        "what": "低阈值 QK-Clip 第 1 步逐头最大 logit 超线的定责（诊断，不改判档）",
        "inputs": {"tpu": tname, "gpu": list(G), "tau": 0.1},
        "per_layer": per_layer,
        "target": {
            "layer": a.layer,
            "head": a.head,
            "tpu": tv,
            "fp64_top": top_self[:4],
            "fp64_tpu_choices_top": top_inj[:4],
            "lost_pair": lost,
            "routes_of_lost_pair_tokens": routes,
        },
        "pair_kappa": {
            "n": len(kappa),
            "kappa_max": max((x["kappa"] for x in kappa), default=None),
            "kappa_median": st.median(x["kappa"] for x in kappa) if kappa else None,
            "bf16_qk_rel_absmax": max((abs(x["bf16_qk_rel"]) for x in kappa), default=None),
            "rows": kappa,
        },
    }
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    w = [x for x in per_layer if x["layer"] == a.layer][0]
    print(
        f"第 {a.layer} 层：R {w['R']:.2f}（对 FP64 自选），对「FP64 灌 TPU 选择」{w['R_vs_fp64_tpu_choices']:.2f}；"
        f"头 {a.head}：TPU {tv:.6f}，FP64 自选第一 {top_self[0]['val']:.6f}（行 {lost['row']} q{lost['q']} k{lost['k']}），"
        f"第二 {top_self[1]['val']:.6f}；灌 TPU 选择后第一 {top_inj[0]['val']:.6f}"
    )
    for r in routes:
        if r["flip"]:
            print(
                f"  token {r['token']} 第 {r['moe_layer']} 层路由翻转：FP64 {r['fp64_experts']} → TPU {r['tpu_experts']}，FP64 第 k / k+1 名相对分差 {r['fp64_gap_k_k1_rel']:.2e}"
            )


if __name__ == "__main__":
    main()
