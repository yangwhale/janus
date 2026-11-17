#!/usr/bin/env python3
"""实践 04 阶梯实验 (i)：第 1 步每个梯度张量对 CPU FP64，各运行各自离 FP64 多远，并和同级的纯数据并行基线、FSDP 参照比。

梯度由第 1 步的一阶动量反推（同 models/hunyuan3/analysis/cp_step1_fp64.py）：g = m₁ / (1 − β1) / 裁剪系数，裁剪系数 = clip / grad_norm
（grad_norm ≥ clip 时，否则 1）；真值是 fp64_step.py --names-from 的输出（同一份 BF16 初始权重、回放数据第 1 步的全局 batch）。
对每个张量 e = rel_l2(运行, FP64)；整网与各组按 FP64 梯度能量合起来。比值 R_base = e / e(基线)、R_ref = e / e(FSDP 参照)，
列出 R > 阈值（默认 1.2）的张量。

  python ladder_step1.py --fp64 <fp64_step 输出> --run L0-dp8=<运行目录> --run L0-ag2=… … \\
      --pair L0-ag2:L0-dp8:L0-fsdp8 …   # 被测:基线:参照；不给 --pair 时只报各自离 FP64
      [--thresh 1.2] --out x.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4]))
from janus.store import Run  # noqa: E402

GROUPS = [("word_embeddings", "embedding"), ("output_layer", "output"), ("experts.local_experts", "routed_expert"),
          (".experts.", "routed_expert"), ("router", "router"), ("shared_experts", "shared_expert"),
          ("linear_qkv.layer_norm", "input_norm"), ("q_layernorm", "qk_norm"), ("k_layernorm", "qk_norm"),
          ("pre_mlp_layernorm", "mlp_norm"), ("final_layernorm", "final_norm"), ("linear_qkv", "attn_qkv"),
          ("linear_proj", "attn_out"), ("mlp.linear_fc", "dense_mlp")]


def group(n: str) -> str:
    return next((g for k, g in GROUPS if k in n), "other")


def coeff(run: Run):
    oc = run.meta.get("optimizer_config") or {}
    clip, gn = float(oc.get("clip_grad") or 0.0), float(run.series("grad_norm")[1])
    return (1.0 if (clip <= 0 or gn < clip) else clip / gn), float(oc["adam_beta1"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp64", required=True)
    ap.add_argument("--run", action="append", default=[], help="名=运行目录")
    ap.add_argument("--pair", action="append", default=[], help="被测:基线[:参照]")
    ap.add_argument("--thresh", type=float, default=1.2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d = Path(a.fp64)
    fmeta = json.loads((d / "meta.json").read_text())
    f = safe_open(str(d / "grads.safetensors"), "numpy")
    fkeys = set(f.keys())
    runs = {k: Run.open(v) for k, v in (x.split("=", 1) for x in a.run)}
    names = None
    for r in runs.values():
        s = {"grad." + n[len("exp_avg."):] for n in r.names(1, "exp_avg.*")} & fkeys
        names = s if names is None else names & s
    names = sorted(names or [])
    loss64 = float(np.mean(fmeta["loss_per_micro"]))
    counts = fmeta.get("counts", {})

    def expert_tokens(name):
        import re
        m = re.search(r"layers\.(\d+)\.mlp\.experts\.linear_fc\d(?:\.weight)?(\d+)$", name)
        return counts.get(m.group(1), [None, None])[int(m.group(2))] if m else None
    out = {"fp64": {"dir": d.name, "loss_mean": loss64}, "n_tensors": len(names), "thresh": a.thresh, "runs": {}, "pairs": {}}
    per = {k: {} for k in runs}
    acc = {k: {"all": [0.0, 0.0]} for k in runs}
    cs = {k: coeff(r) for k, r in runs.items()}
    for n in names:
        ref = f.get_tensor(n).astype(np.float64)
        en = float(np.sum(ref * ref))
        g = group(n)
        for k, r in runs.items():
            c, b1 = cs[k]
            x = r.tensor(1, "exp_avg." + n[len("grad."):]).astype(np.float64).reshape(ref.shape) / (1.0 - b1) / c
            dd = float(np.sum((x - ref) ** 2))
            per[k][n] = float(np.sqrt(dd / en)) if en else float("nan")
            for key in ("all", g):
                acc[k].setdefault(key, [0.0, 0.0])
                acc[k][key][0] += dd
                acc[k][key][1] += en
    for k, r in runs.items():
        m1 = r.metrics().get(1, {})
        l1 = m1.get("lm_loss", m1.get("loss"))
        out["runs"][k] = {"dir": r.root.name if hasattr(r, "root") else k, "grad_norm1": float(r.series("grad_norm")[1]),
                          "loss1": l1, "loss1_minus_fp64": (l1 - loss64) if l1 is not None else None,
                          "pooled": float(np.sqrt(acc[k]["all"][0] / acc[k]["all"][1])),
                          "groups": {g: float(np.sqrt(v[0] / v[1])) for g, v in acc[k].items() if g != "all"},
                          "tensors": per[k]}
    for p in a.pair:
        t, b, *rr = p.split(":")
        rec = {"base": b, "ref": rr[0] if rr else None}
        for tag, other in (("base", b), ("ref", rr[0] if rr else None)):
            if other is None:
                continue
            R = {n: per[t][n] / per[other][n] for n in names if per[other][n] > 0}
            rec[f"R_pooled_{tag}"] = out["runs"][t]["pooled"] / out["runs"][other]["pooled"]
            rec[f"R_groups_{tag}"] = {g: out["runs"][t]["groups"][g] / out["runs"][other]["groups"][g]
                                      for g in out["runs"][t]["groups"] if out["runs"][other]["groups"].get(g)}
            over = sorted([[n, float(r_), per[t][n], per[other][n]] for n, r_ in R.items() if r_ > a.thresh], key=lambda z: -z[1])
            # 路由专家（只抓了 0、1 号）单列：第 1 步选中它们的 token 少的层，一个 token 换了专家梯度就变很多（近似平局路由翻转），
            # 和舍入精度无关；FP64 参照 meta 的 counts 给每层两个专家的 token 数
            rec[f"over_{tag}"] = [z for z in over if group(z[0]) != "routed_expert"]
            rec[f"over_{tag}_routed"] = [z + [expert_tokens(z[0])] for z in over if group(z[0]) == "routed_expert"]
            rec[f"R_max_{tag}"] = max(R.items(), key=lambda z: z[1]) if R else None
        out["pairs"][t] = rec
        ob, orf = rec.get("over_base", []), rec.get("over_ref", [])
        print(f"{t:10s} pooled {out['runs'][t]['pooled']:.3e} R_base {rec.get('R_pooled_base', float('nan')):.3f} "
              f"R_ref {rec.get('R_pooled_ref', float('nan')):.3f}  >{a.thresh}x base: {len(ob)} {[z[0].split('.')[-2] for z in ob[:4]]} "
              f"ref: {len(orf)}  emb {out['runs'][t]['groups'].get('embedding', float('nan')):.2e}", flush=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
