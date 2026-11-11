"""整网串跑找台阶：按执行顺序列出第 1 步每个前向观测点的「逐位相同元素比例」与 rel_L2，
找第一个从逐位相同变成不同、或 rel_L2 突然跳一个量级的点。不设绝对门槛，看台阶。"""
from __future__ import annotations

import re

import numpy as np

from janus import metrics as M

_ORDER = ["in", "attn.qkv.out", "attn.q_norm.out", "attn.k_norm.out", "attn.core.out", "attn.proj.out", "attn.out",
          "mlp_norm.out", "mlp.router.logits", "mlp.router.probs", "mlp.router.topk_idx", "mlp.fc1.out", "mlp.fc2.out",
          "mlp.experts.out", "mlp.shared.out", "mlp.out", "out"]


def execution_order(names: list[str]) -> list[str]:
    layers = sorted({int(m.group(1)) for n in names if (m := re.match(r"^layers\.(\d+)\.", n))})
    seq = ["input.tokens", "embed.out"]
    for i in layers:
        seq += [f"layers.{i}.{p}" for p in _ORDER]
    seq += ["final_norm.out", "logits", "loss.per_token"]
    have = set(names)
    return [n for n in seq if n in have]


def table(test, ref, step: int = 1) -> list[dict]:
    rows, first = [], None
    for n in execution_order(ref.names(step)):
        a, b = test.tensor(step, n), ref.tensor(step, n)
        if a.shape != b.shape or a.dtype != b.dtype:
            rows.append({"name": n, "same_frac": 0.0, "note": f"dtype / 形状不同 {a.dtype}{a.shape} vs {b.dtype}{b.shape}"})
            first = first or n
            continue
        same = float(np.mean(a.view(np.uint8).reshape(a.shape + (-1,)).__eq__(b.view(np.uint8).reshape(b.shape + (-1,))).all(axis=-1))) if a.size else 1.0
        row = {"name": n, "same_frac": same}
        if not np.issubdtype(a.dtype, np.integer):
            st = M.pair_stats(a, b)
            if st is not None:
                row["rel_l2"] = st["rel_l2"]
        if same < 1.0 and first is None:
            first = n
            row["note"] = "第一个不再逐位相同的点"
        rows.append(row)
    return rows


def table_html(rows) -> str:
    import html
    h = "<table><tr><th>执行顺序</th><th>观测点</th><th>逐位相同元素比例</th><th>rel_L2</th><th>备注</th></tr>"
    for i, r in enumerate(rows):
        style = ' style="background:#fff5f5"' if r["same_frac"] < 1.0 else ""
        h += (f"<tr{style}><td>{i}</td><td>{html.escape(r['name'])}</td><td>{r['same_frac']:.6f}</td>"
              f"<td>{r.get('rel_l2', float('nan')):.3e}</td><td>{html.escape(r.get('note', ''))}</td></tr>")
    return h + ("</table><p class=note>串跑时误差会逐层累积；找的是第一个从逐位相同变成不同、或 rel_L2 跳一个量级的点，"
                "那里就是要查的模块。要把误差隔离到单层，用逐层注入（每层输入换成参照的层输入）。</p>")
