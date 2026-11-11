"""HTML 报告：四级门槛总表、曲线图、逐张量图、全部检查明细。图里的数都从运行目录与 results.json 读，不手写。"""
from __future__ import annotations

import base64
import html
import io
import json

LEVEL_ORDER = ["L0", "L1", "L2", "L3", "L4", "regress"]
LEVEL_NAME = {"L0": "L0 地基（数据 / 权重 / 配置）", "L1": "L1 算子 / 优化器", "L2": "L2 模块 / 逐层",
              "L3": "L3 第 1 步整网", "L4": "L4 多步 loss 与统计量", "regress": "回归门"}
COLOR = {"pass": "#1a7f37", "fail": "#cf222e", "inconclusive": "#9a6700", "error": "#8250df"}


FONT_CANDIDATES = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/data/NotoSansCJK-Regular.ttc",
                   "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]


def setup_fonts():
    """图里有中文：找一个 CJK 字体注册给 matplotlib（环境变量 JANUS_CJK_FONT 优先），找不到就退回默认字体（中文显示成方块）。"""
    import os

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams
    for path in [os.environ.get("JANUS_CJK_FONT")] + FONT_CANDIDATES:
        if path and os.path.exists(path):
            font_manager.fontManager.addfont(path)
            rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            rcParams["axes.unicode_minus"] = False
            return path
    return None


def _fig_to_img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"/>'


def _gate_table(results: dict) -> str:
    rows = []
    for lv in LEVEL_ORDER:
        cs = [c for c in results["checks"] if c["level"] == lv]
        if not cs:
            continue
        cnt = {k: sum(1 for c in cs if c["status"] == k) for k in COLOR}
        hard_bad = [c for c in cs if c.get("hard", True) and c["status"] in ("fail", "error")]
        inc = [c for c in cs if c.get("hard", True) and c["status"] == "inconclusive"]
        verdict = "fail" if hard_bad else ("inconclusive" if inc else "pass")
        rows.append(f"<tr><td>{LEVEL_NAME[lv]}</td><td>{len(cs)}</td>"
                    + "".join(f"<td>{cnt[k]}</td>" for k in COLOR)
                    + f'<td style="color:{COLOR[verdict]};font-weight:bold">{verdict}</td>'
                    + f"<td>{'、'.join(html.escape(c['id']) for c in hard_bad[:6])}</td></tr>")
    return ("<table><tr><th>级别</th><th>检查数</th><th>pass</th><th>fail</th><th>inconclusive</th><th>error</th>"
            "<th>本级结论</th><th>拦下本级的硬检查</th></tr>" + "".join(rows) + "</table>"
            "<p class=note>本级结论：本级任一硬检查 fail / error 即 fail；否则有 inconclusive 即 inconclusive。"
            "按级别从低往高看：低一级不过，高一级的结论不可信。</p>")


def _tensor_figs(results: dict, details: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = []
    for c in results["checks"]:
        if c["type"] not in ("tensor_tolerance", "optimizer_step", "bitwise_equal") or c["id"] not in details:
            continue
        rows = details[c["id"]]
        key = "rel_l2" if any("rel_l2" in r for r in rows) else ("differ_frac" if any("differ_frac" in r for r in rows) else None)
        if key is None:
            continue
        rows = [r for r in rows if isinstance(r.get(key), (int, float))][:80]
        if len(rows) < 2:
            continue
        fig, ax = plt.subplots(figsize=(9, max(2.2, 0.16 * len(rows))))
        vals = [max(float(r[key]), 1e-12) for r in rows]
        ax.barh(range(len(rows)), vals, color=COLOR.get(c["status"], "0.5"))
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r.get("name", "?")[-60:] for r in rows], fontsize=6)
        ax.invert_yaxis()
        ax.set_xscale("log")
        up = (((c.get("threshold") or {}).get("noise") or {}).get("upper") or {}).get(key)
        if up:
            ax.axvline(up, color="0.3", ls="--", lw=1)
            ax.axvline(2 * up, color="0.3", ls=":", lw=1)
        ax.set_title(f"{c['id']}：逐张量 {key}（虚线 = 噪声上限 N，点线 = 2N）", fontsize=9)
        out.append(_fig_to_img(fig))
    return out


def render_html(cat, results: dict, runner, out_dir) -> str:
    setup_fonts()
    details = {}
    for c in results["checks"]:
        if c.get("details"):
            try:
                details[c["id"]] = [json.loads(l) for l in (out_dir / c["details"]).read_text().splitlines() if l.strip()]
            except OSError:
                pass
    s = results["summary"]
    parts = [f"<h1>janus 报告：{html.escape(cat.data['model'])} / {html.escape(cat.data['stage'])}</h1>",
             f"<p>{html.escape(cat.data.get('description', ''))}</p>",
             f"<p>被测 <b>{html.escape(cat.runs['test'])}</b>；golden：{html.escape(', '.join(cat.runs.get('golden', [])))}；"
             f"噪声参照：{html.escape('; '.join('/'.join(p) for p in cat.runs.get('noise') or []) or '无')}</p>",
             f"<p>pass {s['pass']}、fail {s['fail']}、inconclusive {s['inconclusive']}、error {s['error']}"
             f"{'；<b style=color:#cf222e>有硬检查失败</b>' if s['hard_fail'] else ''}</p>",
             "<h2>四级门槛</h2>", _gate_table(results)]
    from janus.analysis import loss as LA
    from janus.analysis import router as RA
    from janus.store import StoreError
    try:
        test = runner.run(cat.runs["test"])
        goldens = [runner.run(g) for g in cat.runs.get("golden", [])]
        noise = [(runner.run(a), runner.run(b)) for a, b in cat.runs.get("noise") or []]
    except StoreError as e:                   # 运行目录不合规时检查已全部报 error；报告照出，只是不画图
        test = None
        parts.append(f"<p class=note>运行目录读不出来，曲线和逐层图都不画：{html.escape(str(e))}</p>")
    if test is not None:
        try:
            last = max(test.series("lm_loss"))
            edges = sorted({1, *[e for e in (11, 51, 101, 201, 501, 1001) if e < last + 1], last + 1})
            parts += ["<h2>loss 对照</h2>"] + LA.figures("lm_loss", test, goldens, noise, _fig_to_img)
            parts.append(LA.segment_table_html("lm_loss", test, goldens, edges))
            parts += ["<h3>grad_norm（裁剪前）</h3>"] + LA.figures("grad_norm", test, goldens, noise, _fig_to_img)[:2]
        except Exception as e:  # noqa: BLE001  画图失败不影响结论
            parts.append(f"<p class=note>loss 图没画出来：{html.escape(str(e))}</p>")
        try:
            from janus.analysis import serial as SA
            step = min(int(c["select"]["step"]) for c in cat.checks if (c.get("select") or {}).get("kind") == "tensor")
            if goldens:
                parts += [f"<h2>整网串跑找台阶（第 {step} 步，对 {goldens[0].name}）</h2>", SA.table_html(SA.table(test, goldens[0], step))]
        except Exception as e:  # noqa: BLE001
            parts.append(f"<p class=note>串跑表没出来：{html.escape(str(e))}</p>")
        try:
            step = min(int(c["select"]["step"]) for c in cat.checks if (c.get("select") or {}).get("kind") == "tensor")
            if goldens and test.names(step, "layers.*.mlp.router.topk_idx"):
                rows = RA.layer_table(test, goldens[0], step)
                parts += [f"<h2>专家路由（第 {step} 步，对 {goldens[0].name}）</h2>", RA.table_html(rows)]
                parts += RA.figures(test, goldens[0], rows, _fig_to_img)
        except Exception as e:  # noqa: BLE001
            parts.append(f"<p class=note>路由图没画出来：{html.escape(str(e))}</p>")
    parts.append("<h2>逐张量</h2>")
    parts += _tensor_figs(results, details)
    parts += ["<h2>全部检查</h2>", "<table><tr><th>检查</th><th>类型</th><th>级别</th><th>硬</th><th>状态</th><th>关键指标</th><th>说明</th></tr>"]
    for c in results["checks"]:
        m = c.get("metrics", {})
        key = ", ".join(f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}" for k, v in list(m.items())[:5])
        parts.append(f"<tr><td>{html.escape(c['id'])}</td><td>{c['type']}</td><td>{c['level']}</td>"
                     f"<td>{'是' if c.get('hard', True) else '否'}</td>"
                     f'<td style="color:{COLOR.get(c["status"], "#000")};font-weight:bold">{c["status"]}</td>'
                     f"<td><code>{html.escape(key)}</code></td><td>{html.escape(str(c.get('message', '')))}</td></tr>")
    parts.append("</table><p class=note>数字全部来自 results.json 与运行目录，本报告不手写数字。</p>")
    style = ("<style>body{font-family:-apple-system,'Noto Sans CJK SC',sans-serif;max-width:1100px;margin:auto;padding:1em}"
             "table{border-collapse:collapse;font-size:12px;margin:.5em 0}td,th{border:1px solid #ddd;padding:3px 6px}"
             "th{background:#f6f8fa}img{max-width:100%;display:block;margin:.6em 0}.note{color:#666;font-size:12px}"
             "code{font-size:11px}</style>")
    return f"<!doctype html><html><head><meta charset=utf-8><title>janus {html.escape(cat.data['stage'])}</title>{style}</head><body>" \
        + "\n".join(parts) + "</body></html>"
