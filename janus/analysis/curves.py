"""逐步曲线：多次运行按步对标的时序图与统计，不判 pass / fail，只把轨迹摊开给人看。

两种用法：

- **被测对 golden**（``compare``）：golden 两两的差是噪声参照（灰），被测对每份 golden 的差是红线。
  和 ``loss_series`` / ``multi_golden`` 一样，d_t = (被测 − golden) / |golden|。
- **按设置分组互比**（``groups``）：同一种设置的运行放一组，组内两两的差是噪声；两组之间的差作对照；
  另可按设置给出各组之间的持续偏移（``offsets``：带符号均值与 均值 / 标准误）。分组写在一份 YAML 里，
  格式见 ``GroupSpec``；混元 3 第一阶段的分组是 ``models/hunyuan3/analysis/phase1-curves.yaml``。

每个字段出三张图：``*-overlay``（叠加）、``*-pairdiff``（逐步相对差与 |d_t| 包络，看误差随步数怎么长）、
``*-mean``（滑动窗口与累积带符号均值，看是否一直往一边漂）。要 matplotlib（``pip install -e '.[report]'``）。

命令行：``janus curves compare …`` 与 ``janus curves groups …``，``--help`` 看参数。
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WIN = 50  # 默认滑动窗口步数


# ---------------------------------------------------------------- 读数与基本量


def load_file(path, field, steps=None) -> dict[int, float]:
    """读 metrics.jsonl 的一个字段：{步号: 值}。steps = (a, b) 时只取 a ≤ 步 < b。"""
    out = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if field in r and isinstance(r[field], (int, float)):
            st = int(r["step"])
            if steps is None or steps[0] <= st < steps[1]:
                out[st] = float(r[field])
    return out


def load(d, name, field):
    """读 <目录>/<运行名>.metrics.jsonl 的一个字段。"""
    return load_file(Path(d) / f"{name}.metrics.jsonl", field)


def rel(a, b):
    """两份逐步值共有的步上的相对差 (a − b) / |b|：返回 (步号数组, 差数组)。"""
    st = sorted(set(a) & set(b))
    return np.array(st), np.array([(a[s] - b[s]) / abs(b[s]) for s in st])


def rolling(x, w):
    """长度 w 的滑动均值；序列比窗口短时返回空数组。"""
    if len(x) < w:
        return np.array([])
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from janus.report.html import setup_fonts

    setup_fonts()
    return plt


# ---------------------------------------------------------------- 被测对 golden


def compare(test, goldens, out, fields=("lm_loss", "grad_norm"), prefix="L4", steps=None, win=None):
    """被测对多份 golden 的逐步曲线。test = (名, metrics.jsonl)，goldens = [(名, metrics.jsonl), ...]。

    每个字段出三张：<prefix>-<字段>-overlay / -pairdiff（逐步差与包络）/ -mean（滑动与累积带符号均值）。
    返回 {字段: [图路径]}；被测或 golden 没有这个字段时跳过。"""
    plt = _plt()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tname, tpath = test
    made = {}
    for fld in fields:
        T = load_file(tpath, fld, steps)
        G = {n: load_file(p, fld, steps) for n, p in goldens}
        G = {n: {s: v for s, v in g.items() if s in T} for n, g in G.items()}
        if not T or not any(G.values()):
            continue
        n_steps = len(T)
        w = win or max(3, min(WIN, n_steps // 3))
        pairs = list(itertools.combinations([n for n in G if G[n]], 2))
        # 1) 叠加
        fig, ax = plt.subplots(figsize=(10, 3.2))
        for g in G.values():
            ax.plot(list(g), list(g.values()), lw=0.9, alpha=0.7, color="tab:blue")
        ax.plot(list(T), list(T.values()), lw=1.2, color="tab:red", label=f"被测 {tname}")
        ax.plot([], [], color="tab:blue", label=f"golden {len(G)} 次")
        ax.set_title(f"{fld}：被测与 golden 叠加（{n_steps} 步）")
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
        f1 = out / f"{prefix}-{fld}-overlay.png"
        fig.savefig(f1, dpi=110, bbox_inches="tight")
        plt.close(fig)
        # 2) 逐步相对差 + 包络
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.4))
        env_n = env_t = None
        for a, b in pairs:
            st, d = rel(G[a], G[b])
            a1.plot(st, d, lw=0.6, alpha=0.6, color="0.45")
            env_n = np.abs(d) if env_n is None else np.maximum(env_n, np.abs(d))
            st_n = st
        for g in G.values():
            st, d = rel(T, g)
            a1.plot(st, d, lw=0.9, alpha=0.85, color="tab:red")
            env_t = np.abs(d) if env_t is None else np.maximum(env_t, np.abs(d))
            st_t = st
        a1.plot([], [], color="0.45", label=f"golden 两两（噪声参照，{len(pairs)} 对）")
        a1.plot([], [], color="tab:red", label=f"被测 − golden（{len(G)} 条）")
        a1.axhline(0, color="k", lw=0.5)
        a1.set_title(f"{fld}：逐步相对差 d_t（分母 golden）", fontsize=10)
        a1.set_xlabel("step")
        a1.legend(fontsize=7)
        if env_n is not None:
            a2.semilogy(st_n, np.maximum(env_n, 1e-12), lw=1.0, color="0.3", label="噪声包络：golden 两两 |d_t| 最大")
        a2.semilogy(st_t, np.maximum(env_t, 1e-12), lw=1.2, color="tab:red", label="被测对各 golden |d_t| 最大")
        a2.set_title(f"{fld}：|d_t| 包络（看误差随步数怎么长、是否走平）", fontsize=10)
        a2.set_xlabel("step")
        a2.legend(fontsize=7)
        f2 = out / f"{prefix}-{fld}-pairdiff.png"
        fig.savefig(f2, dpi=110, bbox_inches="tight")
        plt.close(fig)
        # 3) 滑动窗口带符号均值 + 累积带符号均值
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.4))
        for series, color, lw in [(rel(G[a], G[b]), "0.45", 0.6) for a, b in pairs] + [
            (rel(T, g), "tab:red", 1.0) for g in G.values()
        ]:
            st, d = series
            r = rolling(d, w)
            if len(r):
                a1.plot(st[w - 1 :], r, lw=lw, color=color)
            a2.plot(st, np.cumsum(d) / np.arange(1, len(d) + 1), lw=lw, color=color)
        a1.plot([], [], color="0.45", label="golden 两两")
        a1.plot([], [], color="tab:red", label="被测 − golden")
        a1.set_title(f"{fld}：{w} 步滑动窗口带符号均值", fontsize=10)
        a2.set_title(f"{fld}：累积带符号均值 mean(d_1..d_t)（看是否一直往一边漂）", fontsize=10)
        for ax in (a1, a2):
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("step")
        a1.legend(fontsize=7)
        f3 = out / f"{prefix}-{fld}-mean.png"
        fig.savefig(f3, dpi=110, bbox_inches="tight")
        plt.close(fig)
        made[fld] = [f1, f2, f3]
    return made


# ---------------------------------------------------------------- 按设置分组互比


@dataclass
class GroupSpec:
    """按设置分组互比的配置（YAML 里同名的键）。

    groups          组名 → 运行名列表。同一种设置的运行放一组：组内两两的差才是噪声。
    noise_groups    哪几组的组内两两当噪声参照（叠加图里画蓝色，算 summary 的统计）。
    contrast        [组 A, 组 B]：A×B 的跨组对作对照，画红线（没有时退回 highlight × 噪声组）。
    contrast_label  跨组对照在图例里的名字，如「分布式优化器 关 − 开」。
    highlight       另色（红）叠加的组，如「确定性」。
    fields          [[字段, 图题], …]。
    ref_line        lm_loss 图上画的 ±参考线（示意，不用于判定）；ref_label 是图例文字。
    window          滑动窗口步数。
    settings        运行名 → {设置名: 取值}，给 offsets 用；setting_keys 定设置名的顺序。
    offset_from     offsets 从第几步起算（第 1 步前向确定，不算）；offset_batches 是分批均值的段数。
    """

    groups: dict[str, list[str]]
    noise_groups: list[str]
    contrast: list[str] | None = None
    contrast_label: str = "跨组对照"
    highlight: list[str] = field(default_factory=list)
    base_label: str = "默认设置"
    highlight_label: str = "确定性"
    fields: list[list[str]] = field(default_factory=lambda: [["lm_loss", "lm_loss"]])
    ref_line: float | None = None
    ref_label: str | None = None
    window: int = WIN
    settings: dict[str, dict[str, str]] = field(default_factory=dict)
    setting_keys: list[str] = field(default_factory=list)
    offset_from: int = 2
    offset_batches: int = 20

    @classmethod
    def from_yaml(cls, path) -> GroupSpec:
        import yaml

        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"{path}：不认识的键 {sorted(unknown)}")
        spec = cls(**d)
        missing = [g for g in [*spec.noise_groups, *(spec.contrast or []), *spec.highlight] if g not in spec.groups]
        if missing:
            raise ValueError(f"{path}：这些组没有在 groups 里定义：{missing}")
        return spec


def groups(src, out, spec: GroupSpec) -> dict:
    """按设置分组互比：每个字段出三张图，写 curves-summary.json，返回 summary。"""
    plt = _plt()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    noise_groups = [spec.groups[g] for g in spec.noise_groups]
    default = [n for g in noise_groups for n in g]
    det = [n for g in spec.highlight for n in spec.groups[g]]
    ca, cb = (spec.groups[spec.contrast[0]], spec.groups[spec.contrast[1]]) if spec.contrast else ([], [])
    win = spec.window
    summary = {}
    for fld, title in spec.fields:
        S = {n: load(src, n, fld) for n in dict.fromkeys(default + det + ca + cb)}
        dflt = [n for n in default if S[n]]  # 老运行没记 param_norm 等字段的跳过
        dets = [n for n in det if S[n]]
        pairs = [p for g in noise_groups for p in itertools.combinations([n for n in g if S[n]], 2)]  # 同设置两两
        mixed = [(a, b) for a in ca for b in cb if S[a] and S[b]]  # 跨设置
        mixed = mixed or [(d, x) for d in dets for x in dflt]
        n_steps = max((len(S[n]) for n in dflt + dets), default=0)
        # 1) 叠加
        fig, ax = plt.subplots(figsize=(10, 3.2))
        for n in dflt:
            ax.plot(list(S[n]), list(S[n].values()), lw=0.8, alpha=0.6, color="tab:blue")
        for n in dets:
            ax.plot(list(S[n]), list(S[n].values()), lw=0.8, alpha=0.9, color="tab:red")
        ax.plot([], [], color="tab:blue", label=f"{spec.base_label} {len(dflt)} 次")
        ax.plot([], [], color="tab:red", label=f"{spec.highlight_label} {len(dets)} 次")
        ax.set_title(f"{title}：{len(dflt) + len(dets)} 次 {n_steps} 步运行叠加")
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
        fig.savefig(out / f"{fld}-overlay.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        # 2) 两两逐步相对差 + 包络
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.4))
        env = None
        for a, b in pairs:
            st, d = rel(S[a], S[b])
            a1.plot(st, d, lw=0.4, alpha=0.35, color="0.4")
            env = np.abs(d) if env is None else np.maximum(env, np.abs(d))
        for a, b in mixed[:3]:
            st, d = rel(S[a], S[b])
            a1.plot(st, d, lw=0.6, alpha=0.8, color="tab:red")
        a1.plot([], [], color="0.4", label=f"同设置两两（{len(pairs)} 对）")
        a1.plot([], [], color="tab:red", label=f"{spec.contrast_label}（示 3 对）")
        a1.axhline(0, color="k", lw=0.5)
        a1.set_title(f"{title}：两两逐步相对差 d_t", fontsize=10)
        a1.set_xlabel("step")
        a1.legend(fontsize=7)
        a2.semilogy(st, np.maximum(env, 1e-12), lw=0.8, color="0.2", label="同设置两两 |d_t| 最大值")
        a2.semilogy(
            st,
            np.maximum(np.convolve(env, np.ones(win) / win, mode="same"), 1e-12),
            lw=1.4,
            color="tab:blue",
            label=f"{win} 步滑动平均",
        )
        a2.set_title(f"{title}：|d_t| 包络（误差随步数怎么长）", fontsize=10)
        a2.set_xlabel("step")
        a2.legend(fontsize=7)
        fig.savefig(out / f"{fld}-pairdiff.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        # 3) 滑动窗口带符号均值 + 累积带符号均值，参考线 ±ref_line
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.4))
        stats = []
        for a, b in pairs:
            st, d = rel(S[a], S[b])
            r = rolling(d, win)
            a1.plot(st[win - 1 :], r, lw=0.5, alpha=0.5, color="0.4")
            cm = np.cumsum(d) / np.arange(1, len(d) + 1)
            a2.plot(st, cm, lw=0.6, alpha=0.6, color="0.4")
            stats.append(
                {
                    "pair": f"{a}/{b}",
                    "signed_mean": float(d[1:].mean()),
                    "mare": float(np.abs(d[1:]).mean()),
                    "max_step": float(np.abs(d[1:]).max()),
                    "max_roll": float(np.abs(r).max()) if len(r) else None,
                    "max_cum": float(np.abs(cm[1:]).max()),
                }
            )
        cross = []
        for a, b in mixed:
            st, d = rel(S[a], S[b])
            cross.append(
                {
                    "pair": f"{a}/{b}",
                    "signed_mean": float(d[1:].mean()),
                    "max_cum_late": float(np.abs((np.cumsum(d) / np.arange(1, len(d) + 1))[win:]).max()),
                }
            )
        for a, b in mixed[:3]:
            st, d = rel(S[a], S[b])
            a1.plot(st[win - 1 :], rolling(d, win), lw=0.7, color="tab:red")
            a2.plot(st, np.cumsum(d) / np.arange(1, len(d) + 1), lw=0.8, color="tab:red")
        if fld == "lm_loss" and spec.ref_line:
            for ax in (a1, a2):
                ax.axhline(spec.ref_line, color="tab:green", ls="--", lw=1)
                ax.axhline(-spec.ref_line, color="tab:green", ls="--", lw=1)
            a1.plot([], [], color="tab:green", ls="--", label=spec.ref_label or f"参考线 ±{spec.ref_line:g}（示意）")
        a1.plot([], [], color="0.4", label="同设置两两")
        a1.plot([], [], color="tab:red", label=spec.contrast_label)
        a1.set_title(f"{title}：{win} 步滑动窗口带符号均值", fontsize=10)
        a2.set_title(f"{title}：累积带符号均值 mean(d_1..d_t)", fontsize=10)
        for ax in (a1, a2):
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("step")
        a1.legend(fontsize=7)
        fig.savefig(out / f"{fld}-mean.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        summary[fld] = {
            "pairs": len(pairs),
            "signed_mean_abs_max": max(abs(s["signed_mean"]) for s in stats),
            "mare_max": max(s["mare"] for s in stats),
            "max_step_max": max(s["max_step"] for s in stats),
            "rolling_abs_max": max(s["max_roll"] for s in stats if s["max_roll"] is not None),
            "cumulative_abs_max": max(s["max_cum"] for s in stats),
            "per_pair": stats,
            "cross_pairs": cross,  # 跨设置对照
        }
    (out / "curves-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    for f, s in summary.items():
        print(
            f,
            {
                k: (round(v, 9) if isinstance(v, float) else v)
                for k, v in s.items()
                if k not in ("per_pair", "cross_pairs")
            },
        )
    return summary


def offsets(src, out, spec: GroupSpec) -> dict:
    """按设置分组的持续偏移：每两组之间 lm_loss 的带符号均值与 均值 / 标准误（分批均值），写 offsets.json。"""
    keys = spec.setting_keys or sorted({k for v in spec.settings.values() for k in v})
    S = {n: load(src, n, "lm_loss") for n in spec.settings}
    by = {}
    for n, v in spec.settings.items():
        by.setdefault(tuple(v[k] for k in keys), []).append(n)
    nb = spec.offset_batches

    def pair(a, b):
        st, d = rel(S[a], S[b])
        d = d[st >= spec.offset_from]
        m = len(d) // nb
        se = d[: m * nb].reshape(nb, m).mean(axis=1).std(ddof=1) / np.sqrt(nb)
        return float(d.mean()), float(d.mean() / se) if se else None

    rows = []
    order = sorted(by)
    for i, ka in enumerate(order):
        for kb in order[i:]:
            ps = [(a, b) for a in by[ka] for b in by[kb] if a != b and (ka != kb or a < b)]
            if not ps:
                continue
            vals = [pair(a, b) for a, b in ps]
            rows.append(
                {
                    "a": dict(zip(keys, ka)),
                    "b": dict(zip(keys, kb)),
                    "pairs": len(ps),
                    "signed_mean": [v[0] for v in vals],
                    "z": [v[1] for v in vals],
                }
            )
    last = max((max(s) for s in S.values() if s), default=spec.offset_from)
    res = {"window": f"{spec.offset_from}:{last + 1}", "rows": rows}
    Path(out).mkdir(parents=True, exist_ok=True)
    Path(out, "offsets.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for r in rows:
        zs = [abs(z) for z in r["z"] if z is not None]
        print(
            r["a"],
            "对",
            r["b"],
            f"{r['pairs']} 对",
            f"均值 {min(r['signed_mean']):.1e} ~ {max(r['signed_mean']):.1e}",
            f"|z| 最大 {max(zs):.1f}" if zs else "|z| 无",
        )
    return res


# ---------------------------------------------------------------- 命令行


def _steps(s):
    return tuple(int(x) for x in s.split(":")) if s else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="janus curves", description="多次运行按步对标的时序图与统计（不判 pass / fail）。"
    )
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("compare", help="被测对多份 golden：golden 两两是噪声参照")
    c.add_argument("--test", required=True, help="名=metrics.jsonl")
    c.add_argument(
        "--golden", nargs="+", action="extend", required=True, help="名=metrics.jsonl …（可以一次给几个，也可以重复写）"
    )
    c.add_argument("--out", required=True, type=Path, help="输出目录（每个字段三张 PNG）")
    c.add_argument("--prefix", default="L4", help="图文件名前缀（默认 L4）")
    c.add_argument(
        "--fields",
        nargs="+",
        default=["lm_loss", "grad_norm"],
        help="metrics.jsonl 里要画的字段（默认 lm_loss grad_norm）",
    )
    c.add_argument("--steps", default=None, help="窗口 a:b（含 a 不含 b），和目录里的 window 写法一样")
    c.add_argument("--window", type=int, default=None, help="滑动窗口步数（默认 min(50, 步数 / 3)，至少 3）")
    g = sub.add_parser("groups", help="按设置分组互比；分组写在 YAML 里，键的含义见本模块 GroupSpec 的说明")
    g.add_argument("series", type=Path, help="目录，里面每个运行一份 <运行名>.metrics.jsonl")
    g.add_argument("out", type=Path, help="输出目录")
    g.add_argument(
        "--config", required=True, type=Path, help="分组配置 YAML（例：models/hunyuan3/analysis/phase1-curves.yaml）"
    )
    g.add_argument("--no-offsets", action="store_true", help="不算按设置分组的持续偏移（offsets.json）")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    if a.mode == "compare":
        made = compare(
            tuple(a.test.split("=", 1)),
            [tuple(x.split("=", 1)) for x in a.golden],
            a.out,
            a.fields,
            a.prefix,
            _steps(a.steps),
            a.window,
        )
        for f, figs in made.items():
            print(f, [str(p) for p in figs])
        return 0
    spec = GroupSpec.from_yaml(a.config)
    groups(a.series, a.out, spec)
    if spec.settings and not a.no_offsets:
        offsets(a.series, a.out, spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
