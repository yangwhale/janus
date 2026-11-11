"""第 1 步梯度：FP64 自选离散选择 vs 灌 GPU 同一次运行的选择，两种 FP64 参照下的 R 对比（离散翻转定责）。

FP64 自选（`step1_ref.py`）：DSA top-k 与路由由 FP64 自己选，GPU、TPU 相对它的翻转都算各自的误差。
FP64 灌选择（`step1_ref.py --inject-dir`）：全局批每一行都用 GPU 同一次运行（如 `g8-chg-a`）抓到的选择，于是这次 GPU 运行
相对它只剩连续误差。GPU 一侧必须是同一次运行的梯度（GPU 重跑时离散选择也会变）。

逐张量：
- rel_L2(GPU, 自选)、rel_L2(GPU, 灌选择)：两者之差就是 GPU 的翻转带来的那部分；
- rel_L2(TPU, 自选)、rel_L2(TPU, 灌选择)：TPU 用的是自己的选择，灌选择那一份对 TPU 不公平（TPU 的翻转被算两次），只作对照；
- rel_L2(自选, 灌选择)：离散选择不同本身带来的梯度差（分母 ‖自选‖）；
- R_self = rel_L2(TPU, 自选) ÷ rel_L2(GPU, 自选)，R_inject = rel_L2(TPU, 灌选择) ÷ rel_L2(GPU, 灌选择)；
- R_cont_upper = rel_L2(TPU, 自选) ÷ rel_L2(GPU, 灌选择)：TPU 的全部误差（连续 + 自己的翻转）对 GPU 的纯连续误差。
  翻转与连续两部分近似正交时（推断），TPU 的纯连续误差 ≤ TPU 的全部误差，所以它是「只比连续误差的 R」的上界：
  上界 ≤ 1.2 就能把这个张量从 TPU 连续误差里排除；> 1.2 时分不开（TPU 自己的翻转没有隔离）。

另给 `--gpu-extra 名字=<HF 梯度>=<它自己的 FP64 灌选择目录>`（可多份，GPU 独立重跑，各自抓了选择）时，逐张量再算
连续误差的 GPU 噪声尺子：每份 GPU 对「灌它自己选择」的 FP64 的 rel_L2（gpu_cont_each，含 --gpu 那份），
R_cont = rel_L2(TPU, 灌选择) ÷ 中位数(gpu_cont_each)（同 PROTOCOL 第 5 节 GPU 取各份中位数）；TPU 一侧要用灌了
--gpu 那份选择的运行（三方选择相同，TPU 的 rel_L2 才是纯连续误差）。

    python models/hunyuan4/analysis/step1_inject_vs_fp64.py --self <fp64 目录> --inject <fp64 灌选择目录> \\
        --gpu g8-chg-a=<HF 梯度> --tpu t1zdiag2=<diag_grads> --focus <张量名,...> --out <json> \\
        [--gpu-extra g8-chg-b=<HF 梯度>=<fp64 灌 b 选择目录> g8-chg-c=...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1_vs_fp64 import module_of  # noqa: E402


def rel(a, b, den):
    return float((a - b).norm() / den)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self", dest="self_dir", required=True)
    p.add_argument("--inject", required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--tpu", required=True)
    p.add_argument("--focus", default="")
    p.add_argument("--gpu-extra", nargs="*", default=[], help="名字=<HF 梯度>=<它自己的 FP64 灌选择目录>")
    p.add_argument("--note", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    fs = safe_open(os.path.join(a.self_dir, "grads_hf.safetensors"), "pt")
    fi = safe_open(os.path.join(a.inject, "grads_hf.safetensors"), "pt")
    gname, gpath = a.gpu.split("=", 1)
    tname, tpath = a.tpu.split("=", 1)
    g, t = safe_open(gpath, "pt"), safe_open(tpath, "pt")
    extra = []
    for spec in a.gpu_extra:
        en, egp, eid = spec.split("=", 2)
        extra.append((en, safe_open(egp, "pt"), safe_open(os.path.join(eid, "grads_hf.safetensors"), "pt")))
    rows = []
    for h in sorted(g.keys()):
        s = fs.get_tensor(h).double()
        i = fi.get_tensor(h).double()
        gv = g.get_tensor(h).double()
        tv = t.get_tensor(h).double()
        assert s.shape == i.shape == gv.shape == tv.shape, (h, s.shape, gv.shape, tv.shape)
        ds, di = float(s.norm()), float(i.norm())
        r = {
            "tensor": h,
            "module": module_of(h),
            "gpu_vs_self": rel(gv, s, ds),
            "gpu_vs_inject": rel(gv, i, di),
            "tpu_vs_self": rel(tv, s, ds),
            "tpu_vs_inject": rel(tv, i, di),
            "self_vs_inject": rel(i, s, ds),
        }
        r["R_self"] = r["tpu_vs_self"] / r["gpu_vs_self"]
        r["R_inject"] = r["tpu_vs_inject"] / r["gpu_vs_inject"]
        r["R_cont_upper"] = r["tpu_vs_self"] / r["gpu_vs_inject"]
        if extra:
            ce = [r["gpu_vs_inject"]]
            for _, eg, ei in extra:
                er = ei.get_tensor(h).double()
                ce.append(rel(eg.get_tensor(h).double(), er, float(er.norm())))
            r["gpu_cont_each"] = ce
            r["gpu_cont_median"] = float(np.median(ce))
            r["R_cont"] = r["tpu_vs_inject"] / r["gpu_cont_median"]
            r["tpu_le_gpu_cont_max"] = r["tpu_vs_inject"] <= max(ce)
        r["gpu_flip_share"] = (
            1 - (r["gpu_vs_inject"] / r["gpu_vs_self"]) ** 2
        )  # GPU 误差平方里翻转占的份额（推断：两部分近似正交）
        rows.append(r)
    med = {
        k: float(np.median([r[k] for r in rows]))
        for k in (
            "gpu_vs_self",
            "gpu_vs_inject",
            "tpu_vs_self",
            "tpu_vs_inject",
            "self_vs_inject",
            "R_self",
            "R_inject",
            "R_cont_upper",
            "gpu_flip_share",
        )
    }
    if extra:
        for k in ("gpu_cont_median", "R_cont"):
            med[k] = float(np.median([r[k] for r in rows]))
    focus = [x for x in a.focus.split(",") if x]
    out = {
        "what": f"第 1 步梯度：FP64 自选选择 vs 灌 GPU（{gname}，同一次运行）的选择，两种参照下 GPU 与 TPU（{tname}）的 rel_L2 与 R",
        "self_summary": {
            k: json.load(open(os.path.join(a.self_dir, "summary.json")))[k]
            for k in ("lm_loss", "mtp_loss", "grad_norm")
        },
        "inject_summary": {
            k: json.load(open(os.path.join(a.inject, "summary.json")))[k]
            for k in ("lm_loss", "mtp_loss", "grad_norm", "audit")
        },
        "n_tensors": len(rows),
        "median": med,
        "n_R_gt_1.2": {
            "self": sum(r["R_self"] > 1.2 for r in rows),
            "inject": sum(r["R_inject"] > 1.2 for r in rows),
            "cont_upper": sum(r["R_cont_upper"] > 1.2 for r in rows),
            **({"cont": sum(r["R_cont"] > 1.2 for r in rows)} if extra else {}),
        },
        "gpu_extra": [x[0] for x in extra],
        "R_cont_gt_1.2": [
            {k: r[k] for k in ("tensor", "module", "tpu_vs_inject", "gpu_cont_each", "R_cont", "tpu_le_gpu_cont_max")}
            for r in sorted(rows, key=lambda r: -r.get("R_cont", 0))
            if extra and r["R_cont"] > 1.2
        ],
        "by_module": {
            m: {
                "n": len(rs),
                "gpu_vs_self": float(np.median([r["gpu_vs_self"] for r in rs])),
                "gpu_vs_inject": float(np.median([r["gpu_vs_inject"] for r in rs])),
                "R_self": float(np.median([r["R_self"] for r in rs])),
                "R_inject": float(np.median([r["R_inject"] for r in rs])),
                "R_cont_upper": float(np.median([r["R_cont_upper"] for r in rs])),
                **({"R_cont": float(np.median([r["R_cont"] for r in rs]))} if extra else {}),
            }
            for m in sorted({r["module"] for r in rows})
            if (rs := [r for r in rows if r["module"] == m])
        },
        "focus": [r for r in rows if r["tensor"] in focus],
        "note": a.note,
        "rows": rows,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(med, ensure_ascii=False))
    for r in out["focus"]:
        print(
            r["tensor"],
            *(
                f"{k} {r[k]:.3g}"
                for k in (
                    "gpu_vs_self",
                    "gpu_vs_inject",
                    "tpu_vs_self",
                    "tpu_vs_inject",
                    "R_self",
                    "R_inject",
                    "R_cont_upper",
                    *(("gpu_cont_median", "R_cont") if extra else ()),
                )
            ),
        )


if __name__ == "__main__":
    main()
