"""从 Adam 一阶动量反推每步的参数梯度，写成一个派生运行目录（零扰动的梯度测量）。

TPU 上往训练步里加输出（抓梯度的钩子）会改变 XLA 融合，抓到的梯度只是诊断值（frameworks/maxtext/README.md
「上 TPU 才能核对的九项：结果与现状」第 2 条）。训练步之间从状态读出的 Adam 动量不经过加钩子的程序，来自生产图本身。AdamW 的一阶动量是
    m_k = β1·m_{k−1} + (1 − β1)·ĝ_k        （ĝ = 按全局范数裁剪后的梯度，m_0 = 0）
所以
    ĝ_k = (m_k − β1·m_{k−1}) / (1 − β1)，   g_k = ĝ_k / c_k，
c_k 是第 k 步的裁剪系数：optax.clip_by_global_norm 是 clip / ‖g‖（‖g‖ ≥ clip 时，否则 1），‖g‖ 取同一步 metrics 的 grad_norm
（MaxText 的 raw_grad_norm，裁剪之前）。g_k 与 GPU 的 main_grad（已归约、裁剪之前）同义。

精度：m 是 FP32。第 1 步 ĝ = m / 0.1，只多一次 FP32 舍入（相对 6e-8）；之后几步要减掉 β1·m_{k−1}，
相对误差约 ulp(m)/|0.1·ĝ|（推断：|m| 与 |ĝ| 同量级时约 1e-6），都远小于跨平台的梯度差（1e-3 量级）。
反推的梯度只能用在动量有记录的参数上（全量抓数：非专家参数全量 + 每层专家 0、1）。

二阶动量不参与反推，留给核对：v_k = β2·v_{k−1} + (1 − β2)·ĝ_k² 由 janus 的 optimizer_step 在 FP64 里重放，
和状态里读出的 v 比（L1.optim.adamw.*）。

派生目录：
  - step_<k>/ 里原运行的状态类张量（param / main_param / main_param_after / exp_avg / exp_avg_sq / buffer / buffer_after）
    原样硬链接过来，sha256 不变；原运行里加钩子抓到的 grad.* 与激活不带过来；
  - 新写 grad.*（FP32），scope 名 grad_m.<…>；
  - meta.json 抄原运行，改 run_name，加 derived = {from, method, beta1, clip, steps}；metrics.jsonl、params_sha、state_sha、
    modules.txt、data 链接过来。

  python grads_from_moments.py --src <运行根>/t-a --out <运行根>/t-a-gm [--steps 1]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from janus.store import Run, write_safetensors  # noqa: E402

STATE_HEADS = ("param", "main_param", "main_param_after", "exp_avg", "exp_avg_sq", "buffer", "buffer_after")


def clip_coeff_optax(grad_norm: float, clip: float) -> float:
    """optax.clip_by_global_norm：‖g‖ < clip 时不变，否则乘 clip / ‖g‖（没有 eps）。"""
    if not clip or clip <= 0:
        return 1.0
    return 1.0 if grad_norm < clip else clip / grad_norm


def recover(m_k: np.ndarray, m_prev: np.ndarray | None, beta1: float, coeff: float) -> np.ndarray:
    """ĝ_k = (m_k − β1·m_{k−1}) / (1 − β1)，再除以裁剪系数；在 FP64 里算，存 FP32。"""
    m = m_k.astype(np.float64)
    if m_prev is not None:
        m = m - beta1 * m_prev.astype(np.float64)
    return (m / (1.0 - beta1) / coeff).astype(np.float32)


def _link(src: Path, dst: Path):
    if src.exists() and not dst.exists():
        os.symlink(src.resolve(), dst)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="原运行目录（有抓数步的 exp_avg.*）")
    p.add_argument("--out", required=True, help="派生运行目录（不能已存在）")
    p.add_argument("--steps", default=None, help="逗号分隔；默认原运行里所有有 exp_avg.* 的步。第 k > 1 步要 k − 1 步也有动量")
    a = p.parse_args(argv)
    src, out = Run.open(a.src), Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 已存在")
    oc = src.meta.get("optimizer_config") or {}
    beta1, clip = float(oc["adam_beta1"]), float(oc.get("clip_grad") or 0.0)
    gnorm = src.series("grad_norm")
    have = sorted(int(d.name[5:]) for d in src.path.glob("step_*") if (d / "index.json").exists()
                  and src.names(int(d.name[5:]), "exp_avg.*"))
    steps = [int(x) for x in a.steps.split(",")] if a.steps else have
    out.mkdir(parents=True)
    for f in ("metrics.jsonl", "modules.txt", "params_sha", "state_sha", "data"):
        _link(src.path / f, out / f)
    info = {}
    for k in steps:
        if k not in have:
            raise SystemExit(f"第 {k} 步没有 exp_avg.*")
        prev = k - 1 if k > 1 else None
        if prev is not None and prev not in have:
            raise SystemExit(f"第 {k} 步要第 {prev} 步的 exp_avg.*（反推要上一步的一阶动量）")
        coeff = clip_coeff_optax(float(gnorm[k]), clip)
        d = out / f"step_{k}"
        d.mkdir()
        idx = json.loads((src.path / f"step_{k}" / "index.json").read_text())
        keep = [e for e in idx["tensors"] if e["name"].split(".")[0] in STATE_HEADS]
        for f in sorted({e["file"] for e in keep}):
            os.link(src.path / f"step_{k}" / f, d / f)
        grads: dict[str, dict[str, np.ndarray]] = {}
        for n in src.names(k, "exp_avg.*"):
            pname = n[len("exp_avg."):]
            m_prev = src.tensor(prev, n) if prev is not None else None
            g = recover(src.tensor(k, n), m_prev, beta1, coeff)
            scope = "grad_m." + (pname.split(".")[2] if pname.startswith("decoder.layers.") else "other")
            grads.setdefault(scope, {})[f"grad.{pname}"] = g
        new = []
        for scope, ts in sorted(grads.items()):
            for name, ent in write_safetensors(d / f"{scope}.safetensors", ts).items():
                new.append({"name": name, "kind": "grad", "point": "backward", "file": f"{scope}.safetensors",
                            "dtype": ent["dtype"], "shape": ent["shape"], "layout": "global", "sha256": ent["sha256"]})
        (d / "index.json").write_text(json.dumps({"schema_version": 1, "step": k,
                                                  "tensors": sorted(keep + new, key=lambda e: e["name"])}, indent=1))
        info[str(k)] = {"clip_coeff": coeff, "grad_norm": float(gnorm[k]), "grads": len(new), "prev_step": prev}
        print(f"第 {k} 步：裁剪系数 {coeff:.9g}（grad_norm {gnorm[k]:.9g}），反推 {len(new)} 个梯度", flush=True)
    meta = dict(src.meta)
    meta["run_name"] = out.name
    meta["derived"] = {"from": {"run": src.name, "meta_sha256": src.meta_sha256()},
                       "method": "grad = (m_k − β1·m_{k−1}) / (1 − β1) / 裁剪系数（optax clip_by_global_norm：clip / grad_norm）；"
                                 "m 来自训练步之间的状态读；原运行里加钩子抓到的 grad.* 与激活不带过来",
                       "beta1": beta1, "clip": clip, "steps": info,
                       "script": "models/hunyuan3/analysis/grads_from_moments.py"}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    (out / "DONE").write_text("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
