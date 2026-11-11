"""M40 / M41 耦合的确定性查法：GPU g2w-a 第 k 步的整网状态（FP32 主权重 + expert_bias）进 TPU，4 种开关组合
（on = 两个都开，off = 都关，m40 = 只开 M40，m41 = 只开 M41）各算第 k 步的梯度（train_hy4.py --diag-step1 --first-step k
--param-override / --bias-override），逐张量对 GPU 同一步的梯度（g2w-a，裁剪前，export_hf_grads.py 换成 HF 名字）比 rel_L2。
每步给：4 格各自对 GPU 的中位数 / 最大，以及「只开 M41」与另外三格相差最大的张量。没有 FP64 参照（只有第 1 步有）。

  python models/hunyuan4/analysis/w_combo_compare.py --runs <上一级目录> --gpu-grads <目录，含 s<k>.grad.hf.safetensors> \\
      --steps 2,8,30,100 --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COMBOS = ("on", "off", "m40", "m41")


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True)
    p.add_argument("--gpu-grads", required=True)
    p.add_argument("--steps", default="2,8,30,100")
    p.add_argument("--prefix", default="tw-w")
    p.add_argument("--combos", default=",".join(COMBOS), help="逗号分隔的组合名（运行目录 <prefix><k>-<组合>）")
    p.add_argument("--focus", default="learnable_sink_param", help="另逐个列出 HF 名含这个子串的张量")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    from safetensors.numpy import load_file

    res = {"what": __doc__.split("\n\n")[0], "steps": {}}
    for k in [int(x) for x in a.steps.split(",")]:
        G = load_file(f"{a.gpu_grads}/s{k}.grad.hf.safetensors")
        T = {}
        combos = a.combos.split(",")
        for c in combos:
            f = Path(a.runs) / f"{a.prefix}{k}-{c}" / "diag_grads.safetensors"
            if f.exists():
                T[c] = load_file(str(f))
        if not T:
            continue
        names = sorted(set(G).intersection(*[set(t) for t in T.values()]))
        per = {}
        for h in names:
            per[h] = {c: rel(T[c][h], G[h]) for c in T}
        summ = {
            c: {
                "median": float(np.median([per[h][c] for h in names])),
                "max": max(per[h][c] for h in names),
                "argmax": max(names, key=lambda h: per[h][c]),
            }
            for c in T
        }
        div = []
        if "m41" in T and len(T) == 4:
            for h in names:
                others = [per[h][c] for c in ("on", "off", "m40")]
                div.append((per[h]["m41"] - max(others), h))
            div.sort(reverse=True)
        cs = [c for c in combos if c in T]
        pair = {
            f"{x}_vs_{y}": float(np.median([rel(T[x][h], T[y][h]) for h in names]))
            for i, x in enumerate(cs)
            for y in cs[i + 1 :]
        }
        focus = {h: per[h] for h in names if a.focus and a.focus in h}
        res["steps"][str(k)] = {
            "n_tensors": len(names),
            "metric": "第 k 步参数梯度 · TPU 对 GPU（g2w-a 同一步）的 rel_L2",
            "summary": summ,
            "tpu_pairwise_median_rel": pair,
            "m41_minus_max_others_top": [{"tensor": h, "excess": e, **per[h]} for e, h in div[:10]],
            "focus": focus,
        }
        print(k, {c: round(v["median"], 5) for c, v in summ.items()}, "top m41 excess:", div[:2])
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
