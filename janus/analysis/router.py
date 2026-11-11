"""专家路由怎么对照：同一份权重（第 1 步）上逐层比选中的专家，再看多步统计量。

逐层（第 1 步）：
- 选中集合一致率 set_frac、顺序一致率 order_frac：被测与参照对每个 token 选的 top-k 专家集合是否相同；
- 不一致的 token 上，参照自己的第 k 名与第 k+1 名分差（margin）：分差落在舍入量级就是「几乎打平、被舍入翻转」，
  分差大说明打分本身算错了（路由权重、偏置、输入、精度设置）；
- 对 FP64 重算：用同一次运行的路由输入、权重、expert_bias 在 FP64 里重选，翻转率是这套实现自己的地板；
- logits / probs 的 rel_L2；每个专家收到的 token 数（负载），两边逐专家比。
多步：每层 expert_bias 的均值 / 标准差、负载变异系数逐步曲线（比 loss 更早偏），expert_bias 逐步变化量。
"""
from __future__ import annotations

import numpy as np

from janus import metrics as M


def _scores(run, step, layer):
    """参照侧选专家用的分数：sigmoid(logits) + expert_bias（与选择规则一致）。"""
    logits = run.tensor(step, f"layers.{layer}.mlp.router.logits").astype(np.float64)
    logits = logits.reshape(-1, logits.shape[-1])
    s = 1.0 / (1.0 + np.exp(-logits))
    try:
        s = s + run.tensor(step, f"buffer.decoder.layers.{layer}.mlp.router.expert_bias").astype(np.float64)
    except Exception:  # noqa: BLE001  没抓 bias 时只用 sigmoid 分数
        pass
    return s


def layer_table(test, ref, step: int = 1) -> list[dict]:
    rows = []
    for n in ref.names(step, "layers.*.mlp.router.topk_idx"):
        layer = n.split(".")[1]
        ti, ri = test.tensor(step, n), ref.tensor(step, n)
        ag = M.topk_agreement(ti, ri)
        row = {"layer": int(layer), "tokens": int(ag.n), "set_frac": ag.set_frac, "order_frac": ag.order_frac,
               "flipped_tokens": int(ag.n - round(ag.set_frac * ag.n))}
        try:
            sc = _scores(ref, step, layer)
            k = ri.shape[-1]
            bad = np.asarray(ag.rows_differ, dtype=int) if len(ag.rows_differ) else np.zeros(0, int)
            all_m = M.topk_margin(sc, k)
            row["margin_p50_all"] = float(np.median(all_m))
            if bad.size:
                mb = all_m[bad]
                row["margin_p50_flipped"] = float(np.median(mb))
                row["margin_max_flipped"] = float(mb.max())
            row["_margins_all"] = all_m
            row["_margins_flipped"] = all_m[bad] if bad.size else np.zeros(0)
        except Exception:  # noqa: BLE001
            pass
        for t in ("logits", "probs"):
            try:
                row[f"{t}_rel_l2"] = M.rel_l2(test.tensor(step, f"layers.{layer}.mlp.router.{t}"),
                                              ref.tensor(step, f"layers.{layer}.mlp.router.{t}"))
            except Exception:  # noqa: BLE001
                pass
        try:
            ct = test.tensor(step, f"layers.{layer}.mlp.router.counts").astype(np.int64)
            cr = ref.tensor(step, f"layers.{layer}.mlp.router.counts").astype(np.int64)
            row["load_l1_diff"] = int(np.abs(ct - cr).sum())
            row["load_cv_test"], row["load_cv_ref"] = float(ct.std() / ct.mean()), float(cr.std() / cr.mean())
            row["_counts"] = (ct, cr)
        except Exception:  # noqa: BLE001
            pass
        rows.append(row)
    return sorted(rows, key=lambda r: r["layer"])


def figures(test, ref, rows, fig_to_img) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = []
    if not rows:
        return out
    # 1) 不一致 token 的分差 vs 全体分差
    fig, axes = plt.subplots(1, len(rows), figsize=(3.4 * len(rows), 2.8), squeeze=False)
    for ax, r in zip(axes[0], rows):
        ma = r.get("_margins_all")
        if ma is None:
            continue
        bins = np.logspace(-8, 0, 50)
        ax.hist(np.clip(ma, 1e-8, 1), bins=bins, color="0.7", label="全部 token")
        mf = r.get("_margins_flipped")
        if mf is not None and len(mf):
            ax.hist(np.clip(mf, 1e-8, 1), bins=bins, color="#cf222e", label=f"选中集合不同（{len(mf)}）")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"第 {r['layer']} 层：第 k 与 k+1 名分差", fontsize=9)
        ax.legend(fontsize=6)
    out.append(fig_to_img(fig))
    # 2) 逐专家负载：被测 vs 参照
    have = [r for r in rows if "_counts" in r]
    if have:
        fig, axes = plt.subplots(1, len(have), figsize=(3.4 * len(have), 2.8), squeeze=False)
        for ax, r in zip(axes[0], have):
            ct, cr = r["_counts"]
            ax.scatter(cr, ct, s=5)
            lim = max(ct.max(), cr.max()) * 1.05
            ax.plot([0, lim], [0, lim], color="0.5", lw=0.6)
            ax.set_title(f"第 {r['layer']} 层逐专家 token 数（rank 0）", fontsize=9)
            ax.set_xlabel("参照", fontsize=8)
            ax.set_ylabel("被测", fontsize=8)
        out.append(fig_to_img(fig))
    # 3) 多步统计曲线
    try:
        m0 = ref.metrics()[min(ref.metrics())]
    except Exception:  # noqa: BLE001
        return out
    for prefix, title in (("router.bias_std.", "expert_bias 标准差"), ("router.bias_mean.", "expert_bias 均值"),
                          ("router.load_cv.", "负载变异系数（rank 0）")):
        layers = sorted({k[len(prefix):] for k in m0 if k.startswith(prefix)}, key=int)
        if not layers:
            continue
        fig, axes = plt.subplots(1, len(layers), figsize=(3.4 * len(layers), 2.6), squeeze=False)
        for ax, ly in zip(axes[0], layers):
            rs, ts = ref.series(prefix + ly), test.series(prefix + ly)
            ax.plot(list(rs), list(rs.values()), lw=1, label="参照")
            ax.plot(list(ts), list(ts.values()), lw=1, color="k", ls="--", label="被测")
            ax.set_title(f"第 {ly} 层 {title}", fontsize=9)
        axes[0][0].legend(fontsize=6)
        out.append(fig_to_img(fig))
    return out


def table_html(rows) -> str:
    import html
    cols = [("layer", "层"), ("set_frac", "选中集合一致"), ("order_frac", "顺序一致"), ("flipped_tokens", "翻转 token 数"),
            ("margin_p50_all", "全体分差中位数"), ("margin_p50_flipped", "翻转 token 分差中位数"),
            ("margin_max_flipped", "翻转 token 最大分差"), ("logits_rel_l2", "logits rel_L2"), ("probs_rel_l2", "probs rel_L2"),
            ("load_l1_diff", "负载逐专家差之和"), ("load_cv_test", "负载 CV 被测"), ("load_cv_ref", "负载 CV 参照")]
    h = "<table><tr>" + "".join(f"<th>{c[1]}</th>" for c in cols) + "</tr>"
    for r in rows:
        h += "<tr>" + "".join(f"<td>{_fmt(r.get(k))}</td>" for k, _ in cols) + "</tr>"
    return h + "</table>" + html.escape("") + (
        "<p class=note>怎么读：翻转 token 的分差如果和全体分差的下沿重合（落在舍入量级），是舍入把几乎打平的专家换了位置，"
        "属于数值噪声；如果翻转 token 的分差明显大，或 logits rel_L2 远大于激活的 rel_L2，先查路由权重、expert_bias、"
        "路由精度设置（应为 FP32）。负载差只看 rank 0 本地 token。</p>")


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)
