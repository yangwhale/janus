"""生产配方臂（P1）的设置核对：两边每步记下的学习率是否同一条 cosine 调度，路由偏置零均值是否真的生效
（零均值更新下，每层路由偏置的均值应一直等于第 0 步的初值，只剩浮点舍入）。

    python models/hunyuan4/analysis/prodrecipe_settings_check.py --root <运行目录根> \\
        --tpu p1t-a --gpu p1g-a --gpu p1g-b --contrast g3-a --out <json>

`--contrast` 给一份零均值关的现行配置运行作对照（它的偏置均值会漂）。YaRN 不在逐步指标里：GPU 看 meta.json 的
`config.rope_yarn`，TPU 看 meta.json 的 `maxtext_config_full.rope_factor`。
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def load(root, run):
    with open(os.path.join(root, run, "metrics.jsonl")) as f:
        rows = [json.loads(x) for x in f if x.strip()]
    with open(os.path.join(root, run, "meta.json")) as f:
        meta = json.load(f)
    return rows, meta


def bias_drift(rows):
    keys = sorted(k for k in rows[0] if k.startswith("router.bias_mean."))
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r], dtype=np.float64)
        out[k.split(".")[-1]] = {"first": float(v[0]), "max_minus_min": float(v.max() - v.min())}
    return {"per_layer": out, "max_drift": max(x["max_minus_min"] for x in out.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--tpu", required=True)
    p.add_argument("--gpu", action="append", required=True)
    p.add_argument("--contrast", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    t_rows, t_meta = load(a.root, a.tpu)
    res = {
        "schema_version": 1,
        "stage": "t1pr-settings",
        "model": "hunyuan4-mini",
        "inputs": {"tpu": a.tpu, "gpu": a.gpu},
    }
    mc = t_meta.get("maxtext_config_full", {})
    res["tpu_config"] = {
        k: mc.get(k)
        for k in (
            "learning_rate",
            "lr_schedule_type",
            "warmup_steps_fraction",
            "learning_rate_final_fraction",
            "learning_rate_schedule_steps",
            "rope_type",
            "rope_factor",
            "max_position_embeddings",
            "original_max_position_embeddings",
            "beta_fast",
            "beta_slow",
            "rope_attention_scaling",
            "routed_bias_zero_mean_update",
            "routed_bias_update_rate",
            "adam_weight_decay",
            "muon_weight_decay",
        )
    }
    t_lr = np.array([r["lr"] for r in t_rows], dtype=np.float64)
    res["tpu_lr"] = {
        "step1": float(t_lr[0]),
        "step201": float(t_lr[200]),
        "step1000": float(t_lr[-1]),
        "max": float(t_lr.max()),
    }
    res["tpu_bias_mean"] = bias_drift(t_rows)
    res["gpu"] = {}
    for g in a.gpu:
        g_rows, g_meta = load(a.root, g)
        g_lr = np.array([r["lr"] for r in g_rows], dtype=np.float64)
        n = min(len(g_lr), len(t_lr))
        nz = t_lr[:n] != 0
        rel = np.abs(g_lr[:n] - t_lr[:n])[nz] / np.abs(t_lr[:n][nz])
        cfg = g_meta.get("config", {})
        res["gpu"][g] = {
            "config": {
                k: cfg.get(k)
                for k in ("lr", "lr_schedule", "rope_yarn", "routing_bias_zero_mean", "weight_decay_effective")
            },
            "lr_vs_tpu": {
                "steps": int(n),
                "bitwise_equal": int((g_lr[:n] == t_lr[:n]).sum()),
                "max_rel": float(rel.max()) if rel.size else 0.0,
                "both_zero_at_step1": bool(g_lr[0] == 0 and t_lr[0] == 0),
            },
            "bias_mean": bias_drift(g_rows),
        }
    if a.contrast:
        c_rows, _ = load(a.root, a.contrast)
        res["contrast_zero_mean_off"] = {"run": a.contrast, "bias_mean": bias_drift(c_rows)}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
