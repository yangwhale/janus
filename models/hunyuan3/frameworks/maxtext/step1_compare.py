"""第 1 步逐层对比：被测（TPU）的激活 / 路由 / 激活梯度 对 golden（GPU）同名张量。只依赖 numpy 与 janus。

  python step1_compare.py --test <TPU 运行目录或 layer_replay 的输出目录> --golden <golden 运行目录> --out <json> [--label 诊断值]

每个张量给 rel_l2、cosine、max_abs、ULP 分布（BF16 才有意义）；路由给 top-8 集合一致比例，翻转 token 的
「第 8 / 第 9 名分差」（topk_margin，按 golden 一侧加了偏置的分数 sigmoid(logits) + expert_bias 算）和全部 token 的分差分位数。

「BF16 末位量级」怎么判（推断，写清楚好让人改）：一次 BF16 舍入的相对误差上限是 2⁻⁹（半个 ULP），
对一个分布不太偏的张量，两次独立舍入叠加后 rel_l2 一般落在 1e-3 量级；这里取 rel_l2 ≤ 2⁻⁸ ≈ 3.9e-3
作为「末位量级」的判据，ULP 分布原样给出但不进判据：v5p t-a 第 1 步第 0 层的 qkv（输入逐位相同、只隔一次
矩阵乘）就只有 93% 的元素在 1 ULP 内、最大差上千 ULP（接近 0 的元素），ULP 判据会把单次矩阵乘的平台差也算成「超过」。
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))

from janus import metrics as M  # noqa: E402
from janus.store import Run  # noqa: E402

BF16_LEVEL_REL_L2 = 2.0 ** -8


def compare_tensor(name, x, ref) -> dict:
    row = {"name": name, "shape": list(np.shape(ref)), "dtype": str(np.asarray(ref).dtype)}
    if np.shape(x) != np.shape(ref):
        row["error"] = f"形状不同 {np.shape(x)} vs {np.shape(ref)}"
        return row
    row.update(rel_l2=M.rel_l2(x, ref), cosine=M.cosine(x, ref), max_abs=M.max_abs(x, ref)[0])
    if "bfloat16" in row["dtype"]:
        u = M.ulp_summary(x, ref)
        # ulp_buckets 的桶是 0 / 1 / 2-4 / 5-8 / 9-16 / 17+
        row.update(ulp_le0=u.buckets.get("0", 0.0), ulp_le1=u.buckets.get("0", 0.0) + u.buckets.get("1", 0.0),
                   ulp_le4=u.buckets.get("0", 0.0) + u.buckets.get("1", 0.0) + u.buckets.get("2-4", 0.0),
                   ulp_p99=u.p99, ulp_max=u.max)
        row["bf16_level"] = bool(row["rel_l2"] <= BF16_LEVEL_REL_L2)
    else:
        row["bf16_level"] = bool(row["rel_l2"] <= BF16_LEVEL_REL_L2)
    return row


def golden_biased_scores(g: Run, step: int, layer: int, rows=None):
    """golden 一侧加了偏置的路由分数：sigmoid(router.logits) + expert_bias（更新前），FP64。"""
    lg = g.tensor(step, f"layers.{layer}.mlp.router.logits").astype(np.float64)
    lg = lg.reshape(-1, lg.shape[-1])
    b = g.tensor(step, f"buffer.decoder.layers.{layer}.mlp.router.expert_bias").astype(np.float64)
    s = 1.0 / (1.0 + np.exp(-lg)) + b
    return s if rows is None else s[rows]


def compare_router(name, idx, g: Run, step: int, layer: int, k: int, tokens: int) -> dict:
    ref = g.tensor(step, name)[:tokens]
    idx = np.asarray(idx)[:tokens]
    a = M.topk_agreement(idx, ref)
    row = {"name": name, "tokens": int(a.n), "set_frac": a.set_frac, "flipped": int(len(a.rows_differ))}
    try:
        scores = golden_biased_scores(g, step, layer)[:tokens]
        allm = M.topk_margin(scores, k)
        row["margin_all_p50"] = float(np.percentile(allm, 50))
        row["margin_all_p01"] = float(np.percentile(allm, 1))
        if len(a.rows_differ):
            fm = M.topk_margin(scores, k, rows=a.rows_differ)
            row["margin_flipped_max"] = float(fm.max())
            row["margin_flipped_p50"] = float(np.percentile(fm, 50))
            # 翻转 token 的分差落在全部 token 分差分布的第几百分位（越小越像「近并列」）
            row["margin_flipped_pctile_max"] = float((allm <= fm.max()).mean() * 100)
            # 每个翻转 token 换掉了几个专家
            sa, sb = np.sort(idx[a.rows_differ], 1), np.sort(ref[a.rows_differ], 1)
            row["experts_swapped_per_flip_max"] = int(max(len(set(x) - set(y)) for x, y in zip(sa, sb)))
    except Exception as e:  # noqa: BLE001  golden 缺 logits / bias 时只给一致比例
        row["margin_error"] = str(e)
    return row


def compare_dict(got: dict, g: Run, step: int, tokens: int | None = None, k: int = 8, names: str = "*") -> list[dict]:
    rows = []
    have = set(g.names(step))
    for name, x in sorted(got.items()):
        if not fnmatch.fnmatchcase(name, names) or name not in have:
            continue
        if name.endswith(".counts"):
            continue
        ref = g.tensor(step, name)
        T = tokens or ref.shape[0]
        if name.endswith(".topk_idx"):
            layer = int(name.split(".")[1])
            rows.append(compare_router(name, x, g, step, layer, k, T))
            continue
        rows.append(compare_tensor(name, np.asarray(x)[:T], ref[:T]))
    return rows


def first_beyond_bf16(rows: list[dict]) -> str | None:
    """按数据流顺序找第一个不在 BF16 末位量级的连续值张量。"""
    order = ["embed.out"]
    for i in range(4):
        L = f"layers.{i}"
        order += [f"{L}.in", f"{L}.attn.qkv.out", f"{L}.attn.q_norm.out", f"{L}.attn.k_norm.out", f"{L}.attn.core.out",
                  f"{L}.attn.proj.out", f"{L}.mlp_norm.out", f"{L}.mlp.router.logits", f"{L}.mlp.fc1.out", f"{L}.mlp.fc2.out",
                  f"{L}.mlp.shared.out", f"{L}.mlp.out", f"{L}.out"]
    order += ["final_norm.out", "logits"]
    by = {r["name"]: r for r in rows if "rel_l2" in r}
    for n in order:
        if n in by and not by[n]["bf16_level"]:
            return n
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", required=True)
    p.add_argument("--golden", required=True)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--names", default="*")
    a = p.parse_args(argv)
    t, g = Run.open(a.test), Run.open(a.golden)
    acts = {n: t.tensor(a.step, n) for n in t.names(a.step)
            if n.split(".")[0] not in ("grad", "param", "main_param", "main_param_after", "exp_avg", "exp_avg_sq",
                                       "buffer", "buffer_after", "input")}
    rows = compare_dict(acts, g, a.step, names=a.names)
    out = {"label": a.label, "test": str(a.test), "golden": str(a.golden), "step": a.step,
           "bf16_level_rule": {"rel_l2_max": BF16_LEVEL_REL_L2, "status": "推断"},
           "first_beyond_bf16": first_beyond_bf16(rows), "rows": rows}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    for r in rows:
        print("  ".join(f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items() if k != "shape"))
    print("第一个超过 BF16 末位量级的：", out["first_beyond_bf16"])


if __name__ == "__main__":
    main()
