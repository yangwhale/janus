"""第 1 步少数张量的探针：TPU 运行（从动量反推的训练梯度）对 CPU FP64 参照的 rel_L2、前几个元素的值，加第 1 步的
indexer_loss、raw_grad_norm。给 TPU 自比做消融时快速看「某几个张量回没回来」，不做判档（判档用 tpu_self_compare.py）。

    python models/hunyuan4/analysis/step1_probe.py --root <运行目录根> --fp64 <FP64 参照目录> \
        --match 'learnable_sink_param|indexer\\.k_norm\\.weight|enorm\\.weight|hc_head' --out <json> <运行>...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import tpu_self_compare as tsc  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--fp64", required=True, help="FP64 参照目录（含 grads_hf.safetensors、summary.json）")
    p.add_argument("--match", required=True, help="张量名的正则")
    p.add_argument("--head", type=int, default=4, help="记下每个张量前几个元素")
    p.add_argument("--out", required=True)
    p.add_argument("runs", nargs="+")
    a = p.parse_args(argv)
    from safetensors import safe_open

    ref_f = safe_open(os.path.join(a.fp64, "grads_hf.safetensors"), "pt")
    summ = json.load(open(os.path.join(a.fp64, "summary.json")))
    pat = re.compile(a.match)
    out = {
        "what": "第 1 步训练梯度（从动量反推）对 CPU FP64 参照；不判档",
        "match": a.match,
        "fp64": {
            "indexer_loss": summ.get("indexer_loss"),
            "indexer_loss_per_layer": summ.get("indexer_loss_per_layer"),
            "grad_norm": summ.get("grad_norm"),
            "tensors": {},
        },
        "runs": {},
    }
    for r in a.runs:
        d = tsc.Dump(os.path.join(a.root, r))
        m = json.loads(open(os.path.join(a.root, r, "metrics.jsonl")).readline())
        rec = {"indexer_loss": m.get("indexer_loss"), "raw_grad_norm": m.get("raw_grad_norm"), "tensors": {}}
        # MaxText log_indexer_loss_per_layer 打开时的逐层 KD（产出层按收集顺序编号）；GPU / FP64 的逐层值见 summary.json
        rec.update({k: v for k, v in m.items() if k.startswith("indexer_loss_layer.")})
        for kind, n in sorted(d.idx):
            if kind != "exp_avg" or not pat.search(n) or n not in ref_f.keys():
                continue
            ref = ref_f.get_tensor(n).double().numpy()
            g = tsc.train_grad(d, n, m["raw_grad_norm"])
            out["fp64"]["tensors"].setdefault(n, {"head": ref.ravel()[: a.head].tolist()})
            rec["tensors"][n] = {
                "rel_l2_vs_fp64": float(np.linalg.norm(g - ref) / np.linalg.norm(ref)),
                "head": g.ravel()[: a.head].tolist(),
            }
        out["runs"][r] = rec
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
