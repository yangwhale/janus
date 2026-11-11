"""loss 怎么对照：逐步相对差 dₜ 的几种看法。

- 叠加曲线：被测与每份 golden 的 loss；
- dₜ =（被测 − golden）/ |golden| 逐步曲线，噪声参照对（同设置两两）画成灰线；
- 累积带符号均值 mean(d₁..dₜ) 与 ±2 倍标准误（mean_se，分批均值与一阶自相关取大），噪声参照对画同一条曲线：
  两条轨迹分叉后 dₜ 不平稳（越往后越大），标准误会低估，带子离开 0 不等于系统偏移——**以灰线（同设置噪声对）为尺子**，
  被测的曲线落在灰线范围内就是噪声量级；
- 分段表：每段的带符号均值、MARE、最大单步及步号（不只看前 100 步，后段可能翻转）；
- dₜ 直方图：被测对 golden 与噪声参照对放在一起，形状宽了是噪声大，整体平移是偏移。
"""
from __future__ import annotations

import numpy as np

from janus import metrics as M


def rel_diff(test: dict, ref: dict) -> tuple[list[int], np.ndarray]:
    st = sorted(set(test) & set(ref))
    return st, np.array([(test[s] - ref[s]) / abs(ref[s]) for s in st], dtype=np.float64)


def segments(test: dict, ref: dict, edges: list[int]) -> list[dict]:
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        st, d = rel_diff({s: v for s, v in test.items() if a <= s < b}, ref)
        if not len(d):
            continue
        i = int(np.argmax(np.abs(d)))
        out.append({"window": f"{a}:{b}", "steps": len(d), "signed_mean": float(d.mean()), "se": M.mean_se(d),
                    "mare": float(np.abs(d).mean()), "max_step": float(abs(d[i])), "max_step_at": st[i]})
    return out


def figures(field, test, goldens, noise, fig_to_img) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = test.series(field)
    if not ts or not goldens:
        return []
    out = []
    fig, ax = plt.subplots(figsize=(9, 3))
    for g in goldens:
        gs = g.series(field)
        ax.plot(list(gs), list(gs.values()), lw=1, alpha=0.7, label=f"golden {g.name}")
    ax.plot(list(ts), list(ts.values()), lw=1.3, color="k", label=f"被测 {test.name}")
    ax.set_title(f"{field} 逐步")
    ax.set_xlabel("step")
    ax.legend(fontsize=7)
    out.append(fig_to_img(fig))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.2))
    for a, b in noise:
        st, d = rel_diff(a.series(field), b.series(field))
        a1.plot(st, d, color="0.65", lw=0.7)
        cm = np.cumsum(d) / np.arange(1, len(d) + 1)
        a2.plot(st, cm, color="0.65", lw=0.7)
    a1.plot([], [], color="0.65", label="噪声参照对")
    for g in goldens:
        st, d = rel_diff(ts, g.series(field))
        a1.plot(st, d, lw=1.1, label=f"被测 − {g.name}")
        cm = np.cumsum(d) / np.arange(1, len(d) + 1)
        se = np.array([M.mean_se(d[: i + 1]) for i in range(len(d))])
        line, = a2.plot(st, cm, lw=1.2, label=f"被测 − {g.name}")
        a2.fill_between(st, cm - 2 * se, cm + 2 * se, color=line.get_color(), alpha=0.15)
    for ax in (a1, a2):
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("step")
        ax.legend(fontsize=6)
    a1.set_title("逐步相对差 d_t", fontsize=9)
    a2.set_title("累积带符号均值 ±2·标准误；灰线是噪声参照对的同一条曲线", fontsize=9)
    out.append(fig_to_img(fig))

    fig, ax = plt.subplots(figsize=(6, 2.8))
    nd = np.concatenate([rel_diff(a.series(field), b.series(field))[1] for a, b in noise]) if noise else np.zeros(0)
    td = np.concatenate([rel_diff(ts, g.series(field))[1] for g in goldens])
    lim = float(np.percentile(np.abs(np.concatenate([nd, td])), 99.5)) if len(nd) + len(td) else 1.0
    lim = lim if lim > 0 else 1e-12          # 全部为 0（逐位相同）时给一个极小的范围，避免直方图除零
    bins = np.linspace(-lim, lim, 60)
    if len(nd):
        ax.hist(nd, bins=bins, density=True, color="0.7", label="噪声参照对")
    ax.hist(td, bins=bins, density=True, histtype="step", color="k", lw=1.2, label="被测对各 golden")
    ax.set_title("d_t 分布", fontsize=9)
    ax.legend(fontsize=7)
    out.append(fig_to_img(fig))
    return out


def segment_table_html(field, test, goldens, edges) -> str:
    rows = []
    for g in goldens:
        for r in segments(test.series(field), g.series(field), edges):
            z = r["signed_mean"] / r["se"] if r["se"] else float("nan")
            rows.append(f"<tr><td>{g.name}</td><td>{r['window']}</td><td>{r['steps']}</td><td>{r['signed_mean']:.3e}</td>"
                        f"<td>{r['se']:.2e}</td><td>{z:.2f}</td><td>{r['mare']:.3e}</td><td>{r['max_step']:.3e}</td>"
                        f"<td>{r['max_step_at']}</td></tr>")
    return ("<table><tr><th>对哪份 golden</th><th>窗口</th><th>步数</th><th>带符号均值</th><th>标准误</th><th>均值 / 标准误</th>"
            "<th>MARE</th><th>最大单步</th><th>步号</th></tr>" + "".join(rows) + "</table>"
            "<p class=note>|均值 / 标准误| 小于 2：这一段分辨不出偏移；分段结论不一致时以全窗口和最后一段为准，别只看前 100 步。</p>")
