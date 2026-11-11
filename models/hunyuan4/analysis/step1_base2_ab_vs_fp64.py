"""第 1 步 TPU base2 开 / 关两份（只差 `sa_use_base2_exp`）对 CPU FP64 参照：主 loss、MTP loss 与逐张量梯度。

输入：
- `--ref <目录>`：`frameworks/megatron/step1_ref.py` 的整网反向输出（`summary.json` + `grads_hf.safetensors`；
  路由专家只留了少数几个，TPU 一侧也没存路由专家，逐张量比较不含路由专家）；
- `--tpu 名字=<运行目录>`：TPU `--diag-step1` 的运行目录（`diag_step1.json` + `diag_grads.safetensors`）。
  先核文件：每个张量的平方和对设备上算的（`per_tensor.sq`），对不上的记为落盘损坏、不进统计。
  `gradient_accumulation_steps=1` 时 `--diag-step1` 的梯度比训练步的小一个全局有效 token 数 N 的倍数（梯度累积函数
  再除一次 N；截层 D3：诊断 grad_norm × N 与训练步 raw_grad_norm 差 1.3e-5）。这里按 FP64 参照的 grad_norm 判：
  TPU grad_norm × N 落在参照的 1% 以内就把 TPU 梯度乘 N 再比，乘没乘记在输出的 tpu_scale；
- `--ref-emul <目录>`（可选）：同一脚本加 `--emulate-base2 full` 的 FP64 整网反向（FP64 里完整模拟 base2）。给了就另算
  base2 开那份对它的距离（扣掉 base2 之后还剩多少）、模拟本身把 FP64 梯度移了多少，以及 TPU 开 − 关的差与
  模拟移动的夹角余弦（TPU 开、关逐位相同的张量没有夹角，记 NaN、不进中位数）；分母仍是 ‖FP64‖；
- `--gpu 名字=<HF 名字的梯度 safetensors>`（可选，可多份，`export_hf_grads.py` 的输出）：给了就逐张量另算每份 GPU 对 FP64
  的 rel_L2、GPU 两两之间（同一分母）、R = rel(TPU, FP64) ÷ 中位数 rel(GPU, FP64)（两份 TPU 各一个），按模块汇总 R 的中位数与
  R > 1.2 的个数（同 step1_vs_fp64.py 的口径；GPU 文件里没有的张量不进这一部分）；
- `--loss 名字=<metrics.jsonl>`：TPU 训练第 1 步的 `lm_loss` / `mtp_loss_mt`（同一配置的训练运行）；
  `--gpu-loss 名字=lm,mtp`：GPU 第 1 步（只有标量）。

输出：
- loss：每份对 FP64 的相对差，GPU 取中位数作 R 的分母；
- 逐张量：rel_L2(TPU_x, FP64)（分母 ‖FP64‖），两份 TPU 之间的 rel_L2（同一分母），
  比值 rel(关) ÷ rel(开)，误差向量夹角余弦；按模块、按层汇总中位数。

    python step1_base2_ab_vs_fp64.py --ref <fp64 目录> --tpu on=<b2s1-ad> off=<b2s1-bd> \\
        --loss on=<b2s1-at/metrics.jsonl> off=<b2s1-bt/metrics.jsonl> --gpu-loss rw1k-a=12.01,0.95 … --out <json>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics

import numpy as np
import torch
from safetensors import safe_open
from step1_vs_fp64 import FILE_SQ_TOL, module_of


def layer_of(name):
    m = re.search(r"\.(layers|mtp_layers)\.(\d+)\.", name)
    if not m:
        return "其他"
    return f"mtp{m.group(2)}" if m.group(1) == "mtp_layers" else f"L{m.group(2)}"


def step1(path):
    for line in open(path):
        r = json.loads(line)
        if r.get("step") == 1:
            return r
    raise SystemExit(f"{path} 没有第 1 步")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--tpu", nargs=2, required=True, help="on=<目录> off=<目录>（base2 开 / 关）")
    ap.add_argument("--ref-emul", default="")
    ap.add_argument("--gpu", nargs="*", default=[], help="名字=export_hf_grads.py 的 safetensors（可多份）")
    ap.add_argument("--loss", nargs="*", default=[], help="名字=<metrics.jsonl>")
    ap.add_argument("--gpu-loss", nargs="*", default=[], help="名字=lm_loss,mtp_loss_mt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    refsum = json.load(open(os.path.join(a.ref, "summary.json")))
    out = {
        "what": "第 1 步 TPU base2 开 / 关（只差 sa_use_base2_exp）对 CPU FP64 参照：loss 与逐张量梯度",
        "reference": {
            k: refsum.get(k)
            for k in ("dtype", "lm_loss", "mtp_loss", "global_tokens", "rows", "code", "audit", "kept_experts")
        },
    }
    ref_lm = refsum["lm_loss"]
    # MTP 的合成口径同 step1_loss_vs_fp64.py / step1_base2_emulation.py：0.1 × Σ(每行 MTP 均值 × token 数) ÷ Σ token 数
    # （summary.json 的 mtp_loss 是逐行均值的算术平均，差 1e-6 量级，不用它）
    mic = refsum.get("micro") or []
    ref_mtp = (
        0.1 * sum(r["mtp_mean"] * r["n_tokens"] for r in mic) / sum(r["n_tokens"] for r in mic)
        if mic and all("mtp_mean" in r for r in mic)
        else None
    )

    # ---- loss
    loss = {"fp64": {"lm_loss": ref_lm, "mtp_loss_mt": ref_mtp}, "runs": {}}
    for spec in a.gpu_loss:
        n, v = spec.split("=", 1)
        lm, mtp = (float(x) for x in v.split(","))
        loss["runs"][n] = {"side": "gpu", "lm_loss": lm, "mtp_loss_mt": mtp}
    for spec in a.loss:
        n, p = spec.split("=", 1)
        r = step1(p)
        loss["runs"][n] = {"side": "tpu", "lm_loss": r["lm_loss"], "mtp_loss_mt": r.get("mtp_loss_mt")}
    for r in loss["runs"].values():
        r["lm_loss.rel_vs_fp64"] = r["lm_loss"] / ref_lm - 1
        if r.get("mtp_loss_mt") is not None and ref_mtp:
            r["mtp_loss_mt.rel_vs_fp64"] = r["mtp_loss_mt"] / ref_mtp - 1
    gpu = [r for r in loss["runs"].values() if r["side"] == "gpu"]
    if gpu:
        for k in ("lm_loss", "mtp_loss_mt"):
            vals = [r[f"{k}.rel_vs_fp64"] for r in gpu if f"{k}.rel_vs_fp64" in r]
            med = statistics.median(vals)
            loss[f"gpu_median.{k}"] = med
            medabs = statistics.median([abs(x) for x in vals])
            loss[f"gpu_median_abs.{k}"] = medabs
            for r in loss["runs"].values():
                if r["side"] == "tpu" and f"{k}.rel_vs_fp64" in r:
                    r[f"{k}.R_vs_gpu_median_abs"] = abs(r[f"{k}.rel_vs_fp64"]) / medabs
    out["loss"] = loss

    # ---- 逐张量
    ref = safe_open(os.path.join(a.ref, "grads_hf.safetensors"), "pt")
    names = {}
    files = {}
    diags = {}
    for spec in a.tpu:
        n, d = spec.split("=", 1)
        names[n] = d
        files[n] = safe_open(os.path.join(d, "diag_grads.safetensors"), "pt")
        diags[n] = json.load(open(os.path.join(d, "diag_step1.json")))
    (n_on, n_off) = list(names)
    N = float(refsum["global_tokens"])
    scale = {}
    for n in names:
        k = refsum["grad_norm"] / diags[n]["grad_norm"]
        scale[n] = N if abs(k / N - 1) < 1e-2 else 1.0
    out["tpu_scale"] = {
        n: {"factor": scale[n], "ref_over_tpu_grad_norm": refsum["grad_norm"] / diags[n]["grad_norm"]} for n in names
    }
    emul = safe_open(os.path.join(a.ref_emul, "grads_hf.safetensors"), "pt") if a.ref_emul else None
    if emul is not None:
        es = json.load(open(os.path.join(a.ref_emul, "summary.json")))
        out["reference_emul"] = {k: es.get(k) for k in ("lm_loss", "grad_norm", "emulate_base2", "audit")}
    gfiles = {k: safe_open(v, "pt") for k, v in (x.split("=", 1) for x in a.gpu)}
    common = sorted(set(ref.keys()) & set(files[n_on].keys()) & set(files[n_off].keys()))
    rows, corrupt = [], []
    for h in common:
        f64 = ref.get_tensor(h).to(torch.float64)
        den = float(f64.norm())
        xs = {}
        bad = False
        for n in names:
            x = files[n].get_tensor(h).to(torch.float64)
            sq_dev = diags[n]["per_tensor"][h]["sq"]
            sq_file = float(x.square().sum())
            if abs(sq_file - sq_dev) > FILE_SQ_TOL * max(sq_dev, 1e-300):
                corrupt.append({"tensor": h, "run": n, "sq_file": sq_file, "sq_device": sq_dev})
                bad = True
            xs[n] = x * scale[n]
        if bad or den == 0:
            continue
        e_on, e_off = xs[n_on] - f64, xs[n_off] - f64
        r_on, r_off = float(e_on.norm()) / den, float(e_off.norm()) / den
        nn = float(e_on.norm()) * float(e_off.norm())
        extra = {}
        if emul is not None:
            em = emul.get_tensor(h).to(torch.float64)
            shift_emul = em - f64
            shift_tpu = xs[n_on] - xs[n_off]
            mm = float(shift_emul.norm()) * float(shift_tpu.norm())
            extra = {
                f"{n_on}_vs_emul": float((xs[n_on] - em).norm()) / den,
                "emul_vs_ref": float(shift_emul.norm()) / den,
                "shift_cos_tpu_vs_emul": float((shift_emul * shift_tpu).sum()) / mm if mm > 0 else float("nan"),
            }
        if gfiles and all(h in gf.keys() for gf in gfiles.values()):
            gs = {k: gf.get_tensor(h).to(torch.float64) for k, gf in gfiles.items()}
            if all(tuple(g.shape) == tuple(f64.shape) for g in gs.values()):
                ge = {k: float((g - f64).norm()) / den for k, g in gs.items()}
                gk = list(gs)
                gpair = [float((gs[i] - gs[j]).norm()) / den for x, i in enumerate(gk) for j in gk[x + 1 :]]
                gmed = float(np.median(list(ge.values())))
                extra.update(
                    {
                        "gpu_vs_ref_each": ge,
                        "gpu_vs_ref": gmed,
                        "gpu_noise_max": max(gpair) if gpair else float("nan"),
                        f"R_{n_on}": r_on / gmed if gmed > 0 else float("nan"),
                        f"R_{n_off}": r_off / gmed if gmed > 0 else float("nan"),
                    }
                )
        rows.append(
            {
                **extra,
                "tensor": h,
                "module": module_of(h),
                "layer": layer_of(h),
                "numel": int(f64.numel()),
                "ref_norm": den,
                f"{n_on}_vs_ref": r_on,
                f"{n_off}_vs_ref": r_off,
                f"{n_on}_vs_{n_off}": float((xs[n_on] - xs[n_off]).norm()) / den,
                "ratio_off_on": r_off / r_on if r_on > 0 else float("nan"),
                "err_cos": float((e_on * e_off).sum()) / nn if nn > 0 else float("nan"),
            }
        )

    def summ(rs):
        return {
            "n": len(rs),
            f"{n_on}_vs_ref_median": float(np.median([r[f"{n_on}_vs_ref"] for r in rs])),
            f"{n_off}_vs_ref_median": float(np.median([r[f"{n_off}_vs_ref"] for r in rs])),
            f"{n_on}_vs_{n_off}_median": float(np.median([r[f"{n_on}_vs_{n_off}"] for r in rs])),
            "ratio_off_on_median": float(np.median([r["ratio_off_on"] for r in rs])),
            "n_off_closer": sum(r[f"{n_off}_vs_ref"] < r[f"{n_on}_vs_ref"] for r in rs),
            **(
                {
                    f"{n_on}_vs_emul_median": float(np.median([r[f"{n_on}_vs_emul"] for r in rs])),
                    "emul_vs_ref_median": float(np.median([r["emul_vs_ref"] for r in rs])),
                    "shift_cos_tpu_vs_emul_median": float(np.nanmedian([r["shift_cos_tpu_vs_emul"] for r in rs])),
                    "n_shift_cos_undefined": int(sum(np.isnan(r["shift_cos_tpu_vs_emul"]) for r in rs)),
                    f"n_{n_on}_closer_to_emul": sum(r[f"{n_on}_vs_emul"] < r[f"{n_on}_vs_ref"] for r in rs),
                }
                if rs and f"{n_on}_vs_emul" in rs[0]
                else {}
            ),
            **(
                {
                    "n_with_gpu": len(rg),
                    "gpu_vs_ref_median": float(np.median([r["gpu_vs_ref"] for r in rg])),
                    "gpu_noise_max_median": float(np.median([r["gpu_noise_max"] for r in rg])),
                    f"R_{n_on}_median": float(np.median([r[f"R_{n_on}"] for r in rg])),
                    f"R_{n_off}_median": float(np.median([r[f"R_{n_off}"] for r in rg])),
                    f"n_R_{n_on}_gt_1.2": sum(r[f"R_{n_on}"] > 1.2 for r in rg),
                    f"n_R_{n_off}_gt_1.2": sum(r[f"R_{n_off}"] > 1.2 for r in rg),
                }
                if (rg := [r for r in rs if "gpu_vs_ref" in r])
                else {}
            ),
        }

    out["tensors"] = {
        "n_common": len(common),
        "tpu_file_corrupt_not_counted": corrupt,
        "all": summ(rows),
        "by_module": {m: summ([r for r in rows if r["module"] == m]) for m in sorted({r["module"] for r in rows})},
        "by_layer": {m: summ([r for r in rows if r["layer"] == m]) for m in sorted({r["layer"] for r in rows})},
        "largest_shift": sorted(rows, key=lambda r: -abs(np.log(r["ratio_off_on"])))[:15],
        "rows": rows,
    }
    out["tpu_grad_norm_scaled"] = {n: diags[n]["grad_norm"] * scale[n] for n in names}
    out["ref_grad_norm"] = refsum["grad_norm"]
    out["tpu_diag_loss_total"] = {n: diags[n]["loss"] for n in names}
    out["tpu_runs"] = {n: os.path.basename(os.path.normpath(d)) for n, d in names.items()}
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    s = out["tensors"]["all"]
    print(
        json.dumps(
            {"loss": {n: {k: v for k, v in r.items() if "rel" in k} for n, r in loss["runs"].items()}},
            ensure_ascii=False,
            indent=1,
        )
    )
    print(json.dumps(s, ensure_ascii=False))
    for m, v in out["tensors"]["by_module"].items():
        print(m, json.dumps(v, ensure_ascii=False))


if __name__ == "__main__":
    main()
