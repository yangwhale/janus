"""mini 1000 步：TPU 主 loss 对 GPU 16 份逐步均值的偏差随步数怎么变（窗口与 25 步分块），MTP loss 同口径。

做法：
- GPU 尺子：`g3-a` … `g3-p` 16 份逐步均值作基线；每份 GPU 对其余 15 份的同一量（留一法）给出窗口的标准差，作「一份 GPU
  运行会偏多少」的尺子；
- TPU：开、关 base2 各 4 份（同 `mini_base2_loo.py`），逐步相对差 (TPU − GPU 均值) ÷ GPU 均值，按窗口与 25 步分块取平均，
  记 8 份的均值与 8 份里偏高的份数；
- 另记分块偏差与几个量的相关系数（步数、GPU 均值 loss 的下降量、GPU param_norm、1 / grad_norm），回答「跟哪个量同步」；
  学习率是常数 1e-5，不参与。

另外 `--variants <运行名…>`：任意多份 TPU 1000 步运行（TPU 自比的各臂、本次的诊断开关运行），每份记主 loss 第 2–1000 步与
第 751–1000 步、第 51–100 步的偏差，MTP 第 751–1000 步的偏差，第 75 步 param_norm 对 GPU 均值的差（早期那段 TPU 参数范数
落后的标志）。

    python mini_main_loss_drift.py --root <含 g3-* 与各 TPU 运行的根目录> --out <json> [--variants ts-… mlh-…]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

GPU = [f"g3-{c}" for c in "abcdefghijklmnop"]
TPU = {
    "on": ["t1zw-a", "ts-c-nolhs-1k", "tz-on-vm32b", "tz-on-vm64"],
    "off": ["t1zz-a", "tz-off-nolhs", "tz-off-vm32", "tz-off-vm64"],
}
WINDOWS = [(1, 1), (2, 10), (11, 50), (51, 100), (101, 200), (201, 300), (301, 500), (501, 750), (751, 1000), (2, 1000)]
BLOCK = 25
STEPS = 1000


def series(root: str, run: str, field: str) -> np.ndarray:
    d = {}
    with open(os.path.join(root, run, "metrics.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get(field) is not None:
                d[r["step"]] = r[field]
    return np.array([d[s] for s in range(1, STEPS + 1)], float)


def table(root: str, field: str) -> dict:
    g = np.array([series(root, x, field) for x in GPU])
    gm = g.mean(0)
    runs = TPU["on"] + TPU["off"]
    t = np.array([series(root, x, field) for x in runs])
    rel_t = (t - gm) / gm
    loo = np.array([(g[i] - np.delete(g, i, 0).mean(0)) / gm for i in range(len(GPU))])

    def win(x, a, b):
        return x[..., a - 1 : b].mean(-1)

    windows = []
    for a, b in WINDOWS:
        wt = win(rel_t, a, b)
        sd = float(np.std(win(loo, a, b), ddof=1))
        windows.append({
            "steps": f"{a}-{b}", "tpu_mean": float(wt.mean()), "tpu_runs": dict(zip(runs, map(float, wt))),
            "n_tpu_above_0": int((wt > 0).sum()), "gpu_loo_sd": sd, "z_tpu_mean": float(wt.mean() / sd),
        })
    blocks = []
    for i in range(STEPS // BLOCK):
        a, b = i * BLOCK + 1, (i + 1) * BLOCK
        wt = win(rel_t, a, b)
        blocks.append({"steps": f"{a}-{b}", "tpu_mean": float(wt.mean()), "n_tpu_above_0": int((wt > 0).sum()),
                       "gpu_loo_sd": float(np.std(win(loo, a, b), ddof=1)),
                       "tpu_mean_abs": float(win(rel_t * gm, a, b).mean())})
    return {"windows": windows, "blocks": blocks, "rel_t_mean": rel_t.mean(0), "gm": gm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="*", default=[])
    a = ap.parse_args()
    out = {"what": "mini 1000 步 TPU 8 份（base2 开、关各 4）主 loss 与 MTP loss 对 GPU 16 份逐步均值的相对差：窗口、25 步分块、"
                   "与训练进度的相关", "gpu": GPU, "tpu": TPU}
    lm = table(a.root, "lm_loss")
    out["lm_loss"] = {k: lm[k] for k in ("windows", "blocks")}
    out["mtp_loss_mt"] = {k: v for k, v in table(a.root, "mtp_loss_mt").items() if k in ("windows", "blocks")}
    y = np.array([b["tpu_mean"] for b in lm["blocks"]])
    nb = len(y)

    def blk(x):
        return np.array([x[i * BLOCK : (i + 1) * BLOCK].mean() for i in range(nb)])

    feats = {
        "step": blk(np.arange(1, STEPS + 1, dtype=float)),
        "gpu_loss_drop": blk(lm["gm"][0] - lm["gm"]),
        "gpu_param_norm": blk(np.array([series(a.root, x, "param_norm_fp64") for x in GPU]).mean(0)),
        "inv_gpu_grad_norm": blk(1 / np.array([series(a.root, x, "grad_norm") for x in GPU]).mean(0)),
    }
    out["block_corr"] = {
        k: {"all": float(np.corrcoef(v, y)[0, 1]), "from_501": float(np.corrcoef(v[20:], y[20:])[0, 1])}
        for k, v in feats.items()
    }
    if a.variants:
        gl = np.array([series(a.root, x, "lm_loss") for x in GPU]).mean(0)
        gm_ = np.array([series(a.root, x, "mtp_loss_mt") for x in GPU]).mean(0)
        gp = np.array([series(a.root, x, "param_norm_fp64") for x in GPU]).mean(0)
        var = {}
        for r in a.variants:
            lm_ = (series(a.root, r, "lm_loss") - gl) / gl
            mt_ = (series(a.root, r, "mtp_loss_mt") - gm_) / gm_
            pn = series(a.root, r, "param_norm_exact")
            var[r] = {"lm_2_1000": float(lm_[1:].mean()), "lm_51_100": float(lm_[50:100].mean()),
                      "lm_751_1000": float(lm_[750:].mean()), "mtp_751_1000": float(mt_[750:].mean()),
                      "param_norm_step75_minus_gpu": float(pn[74] - gp[74])}
        out["variants"] = var
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for w in out["lm_loss"]["windows"]:
        print("lm ", w["steps"], f"{w['tpu_mean']:+.2e}", w["n_tpu_above_0"], f"sd {w['gpu_loo_sd']:.2e}")
    for w in out["mtp_loss_mt"]["windows"]:
        print("mtp", w["steps"], f"{w['tpu_mean']:+.2e}", w["n_tpu_above_0"], f"sd {w['gpu_loo_sd']:.2e}")
    print(json.dumps(out["block_corr"]))


if __name__ == "__main__":
    main()
