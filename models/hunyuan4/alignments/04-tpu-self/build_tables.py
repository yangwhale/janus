"""TPU 自比：按 arms.yaml 与 results/ 生成 MATRIX.md 的逐类结果表与「TPU 脾气表」、README 按「对 TPU 精度有没有影响」分的几堆表。

两份文件里 `<!-- BEGIN:<名字> -->` 与 `<!-- END:<名字> -->` 之间的内容由这里重写，其余手写部分不动。
数字全部取自 results/<运行>-vs-<golden>.json（tpu_self_compare.py 的输出），不手写。

    python models/hunyuan4/alignments/04-tpu-self/build_tables.py [--check]
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
GOLDEN = "ts-g-a"
# 结果文件里的档名 → 文档里的写法。2026-11-10 前的结果文件写「在刻度内」「超出刻度」（历史证据，不改），这里读成新叫法
BAND = {"逐位相同": "逐位相同", "在噪声内": "在噪声内", "在刻度内": "在噪声内", "略超": "略超", "超": "超",
        "超出噪声上限": "超出噪声上限", "超出刻度": "超出噪声上限"}
# 曲线检查 → 给人看的名字（领头的那条）
CHECK_NAME = {"C.loss": "主 loss", "C.loss.tail": "主 loss（第 101 步起）", "C.mtp_loss": "MTP loss", "C.indexer_loss": "索引器 loss",
              "C.grad_norm": "梯度范数", "C.param_norm": "参数范数", "C.router.bias_mean": "路由偏置均值",
              "C.router.bias_std": "路由偏置标准差"}
KEY_NAME = {"signed_mean": "带符号均值", "mare": "MARE", "max_step": "最大单步"}
# 「和 GPU 重跑比」一列的三种说法（事后解读，不改判档）
SAME_AS_RERUN = "和 GPU 重跑分不开"
APART = "分得开"


def loo_max(steps):
    """GPU 重跑留一法的最差 r 的最大值（analysis/gpu_loo_r.py 的输出）；没有就 None。"""
    p = HERE / "results" / f"gpu-rerun-loo-r-{steps}.json"
    return json.loads(p.read_text())["r_max_max"] if p.exists() else None


def reading(c, d, steps=300):
    """事后解读（不改判档）：略超或超、但曲线最差 r 不超过 GPU 重跑留一法的最大值、且第 1 步没有定不了责的张量 → 和 GPU 重跑分不开。"""
    if c["band"] in ("逐位相同", "在噪声内"):
        return "—"
    lm = loo_max(steps)
    r = d["verdict"].get("curves_r_max")
    un = (d.get("step1_grad") or {}).get("n_unattributed") or 0
    if lm is None or r is None:
        return "—"
    if un:
        return APART
    if r > lm:
        return APART
    # 贴着 GPU 重跑的最大值（差 5% 以内）单独标出：最大值只是有限个组合里的一个样本
    return "临界：贴着 GPU 重跑的最大值" if r > 0.95 * lm else SAME_AS_RERUN


@functools.lru_cache(maxsize=1)
def screen_results():
    """跑前离线筛（hlo_screen.py）的结果：{臂编号: 记录}；没有结果文件就空。"""
    p = HERE / "results" / "hlo-screen.json"
    return {r["arm"]: r for r in json.loads(p.read_text())["arms"]} if p.exists() else {}


SCREEN_SHORT = {"程序相同": "相同", "程序不同": "不同", "无法离线判断": "判不了"}


def screen_cell(aid):
    r = screen_results().get(aid)
    return "—" if r is None or not r.get("verdict") else SCREEN_SHORT[r["verdict"]]


def result(arm):
    run, vs = arm.get("run"), arm.get("vs", GOLDEN)
    if not run:
        return None
    p = HERE / "results" / f"{run}-vs-{'golden' if vs == GOLDEN else vs}.json"
    return json.loads(p.read_text()) if p.exists() else None


def e(x, nd=2):
    return "—" if x is None else f"{x:.{nd}e}"


def cells(d):
    """一份结果 → 档、r、第 1 步相对差中位数、对 FP64 比值、步时、显存峰值。"""
    v = d["verdict"]
    band = v.get("band") or v["label"]
    band = BAND.get(band, band)
    r = v.get("curves_r_max")
    if d["curves"].get("skipped"):
        r = None
    s1 = d.get("step1_grad") or {}
    fp = (s1.get("fp64") or {}).get("ratio_median")
    st = d["step_time_median_from_11"]
    pk = (d.get("peak_hbm_bytes_max_local_device") or {}).get("test")
    worst = max(d["curves"].get("checks") or [{"r": 0}], key=lambda c: c["r"])
    lead = "—" if not worst.get("id") or worst["r"] <= 0 else f"{CHECK_NAME.get(worst['id'].replace('L4.', 'C.'), worst['id'])}·{KEY_NAME.get(worst['worst_key'], worst['worst_key'])}"
    return {"band": band, "lead": lead, "r": "—" if r is None else f"{r:.2f}", "d": e(s1.get("d_median")) if s1 else "—",
            "fp": "—" if fp is None else f"{fp:.3f}", "st": "—" if not st.get("test") else f"{st['test']:.3f}",
            "st_rel": None if not (st.get("test") and st.get("golden")) else st["test"] / st["golden"] - 1,
            "pk": e(pk), "hlo": d["hlo"].get("program_equal")}


LIBTPU_DEFAULT = "0.0.46"  # golden 镜像自带；arms.yaml 里不写 libtpu 的臂都是它


def libtpu_of(a):
    return a.get("libtpu", LIBTPU_DEFAULT)


def matrix_table(cls, arms):
    rows = ["| 编号 | 运行 | libtpu | 改什么 | 预期 | 档 | 曲线最差 r | 领头 | 和 GPU 重跑比（事后解读） | 第 1 步梯度相对差中位数 | 对 FP64 | 步时 s | 显存峰值 B | 程序相同 | 离线筛 | 出处 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        if a["id"][0] != cls:
            continue
        d = result(a)
        run = f"`{a['run']}`" + (f" 对 `{a['vs']}`" if a.get("vs") else "") if a.get("run") else "—"
        if d is None:
            rows.append(f"| {a['id']} | {run} | {libtpu_of(a)} | {a['change']} | {a['expect']} | {a.get('note', '待跑')} |  |  |  |  |  |  |  |  | {screen_cell(a['id'])} |  |")
            continue
        c = cells(d)
        src = f"[结果](results/{Path(a['run']).name}-vs-{'golden' if a.get('vs', GOLDEN) == GOLDEN else a['vs']}.json)"
        hlo = {True: "是", False: "否", None: "—"}[c["hlo"]]
        rd = reading(c, d, 1000 if a["id"].startswith("L") else 300)
        rows.append(f"| {a['id']} | {run} | {libtpu_of(a)} | {a['change']} | {a['expect']} | {c['band']} | {c['r']} | {c['lead']} | {rd} | {c['d']} | {c['fp']} | "
                    f"{c['st']} | {c['pk']} | {hlo} | {screen_cell(a['id'])} | {src} |")
    return rows


def screen_table(arms):
    """跑前离线筛的回测：每个筛过的臂，预判对 TPU 上的实际（程序相同否、档）。
    预判「程序相同」而实际不逐位，是筛选本身错了，直接报错退出，不出表（不许靠改表掩盖）。"""
    sr = screen_results()
    rows = ["| 编号 | 改什么 | 离线筛 | 理由 | CPU 编的与 TPU 存的程序相同 | TPU 上程序相同 | 实际档 | 符不符 |", "|---|---|---|---|---|---|---|---|"]
    n = Counter()
    comp = {}   # 编译按规格去重（同一个对照被很多臂共用，只编一次）
    for a in arms:
        r = sr.get(a["id"])
        if r is None or not r.get("verdict"):
            continue
        d = result(a)
        band = cells(d)["band"] if d else "未出结果"
        # TPU 上两次运行存下的程序是否相同：用 hlo_screen.py 的规范化（含重编号）比的结果；没有再退回结果文件里的程序指纹
        tpu_same = r.get("tpu_program_equal", None if d is None else d["hlo"].get("program_equal"))
        v = r["verdict"]
        if v == "程序相同":
            ok = ("符" if tpu_same is not False else "符（但 TPU 上存下的两个程序不同，要查）") if band == "逐位相同" else None
            if d is not None and ok is None:
                raise SystemExit(f"{a['id']} 离线筛判「程序相同」，TPU 上实际是「{band}」：筛选有误，先查原因")
            ok = ok or "—（未出结果）"
        elif v == "程序不同":
            ok = ("不符（TPU 上是同一个程序）" if tpu_same else "符（程序不同，结果仍逐位：多跑一次，不漏）" if band == "逐位相同" else "符")
        else:
            ok = "—（不判）"
        n[v] += 1
        n[ok.split("（")[0]] += 1
        cvt = r.get("cpu_vs_tpu") or {}
        cv = "—" if not cvt else ("是" if all(cvt.values()) else "否：" + "、".join(k for k, x in cvt.items() if not x))
        why = r["reason"] if v != "无法离线判断" else r["reason"].split("：")[0]
        rows.append(f"| {a['id']} | {a['plain']} | {SCREEN_SHORT[v]} | {why} | {cv} | {({True: '是', False: '否', None: '—'})[tpu_same]} | {band} | {ok} |")
        for c in (r.get("compile") or {}).values():
            if c.get("seconds"):
                comp[c["hash"]] = c
    total = n["程序相同"] + n["程序不同"] + n["无法离线判断"]
    head = [f"筛了 {total} 个臂：判程序相同 {n['程序相同']} 个、程序不同 {n['程序不同']} 个、无法离线判断 {n['无法离线判断']} 个；"
            f"对上 TPU 实际的「符」{n['符']} 个、「不符」{n['不符']} 个。"]
    if comp:
        secs = sorted(c["seconds"] for c in comp.values())
        wall = sorted(c["wall_seconds"] for c in comp.values())
        rss = [c["max_rss_gib"] for c in comp.values()]
        head.append(f"一共编了 {len(comp)} 份不同的规格，每份一次：XLA 编译 {secs[0]:.0f}–{secs[-1]:.0f} s，连追踪与建抽象状态整次 "
                    f"{wall[0]:.0f}–{wall[-1]:.0f} s（中位数 {wall[len(wall) // 2]:.0f} s）；单进程峰值内存 {min(rss):.1f}–{max(rss):.1f} GiB。")
    return head + [""] + rows


def temper_table(arms, classes):
    rows = ["| 类 | 改动 | 场景 | 档 | 300 步曲线最差 r | 1000 步 | 差得最多的曲线·统计量 | 和 GPU 重跑比 | 多花的算力 | 要怎么办 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    chips = {x["run"]: x.get("chips", 8) for x in arms if x.get("run")}
    for a in arms:
        if a["id"].startswith("L"):   # 长跑臂并进原臂那一行的「1000 步」一列
            continue
        d = result(a)
        if d is None:
            continue
        c = cells(d)
        long = "—"
        if a.get("long"):
            dl = result({"run": a["long"], "vs": a.get("long_vs", "ts-g-a-1000")})
            if dl is None:
                long = "在跑"
            else:
                cl = cells(dl)
                rd1k = reading(cl, dl, 1000)
                long = f"{cl['band']} {cl['r']}"
                if cl["band"] not in ("逐位相同", "在噪声内"):
                    long += f"（{rd1k}）"
        # 算力代价按芯片·秒：步时 × 芯片数 ÷（对照的步时 × 对照的芯片数）− 1；芯片数不是 8 的臂在 arms.yaml 里写 chips
        # 对照不是 golden 的臂（如对某个只改了连带项的对照比），按对照自己的芯片数折算
        base_chips = chips.get(a.get("vs", GOLDEN), 8)
        cost = "—" if c["st_rel"] is None else f"{(1 + c['st_rel']) * a.get('chips', 8) / base_chips - 1:+.1%}"
        fix = a.get("fix", "—")
        if fix == "—" and c["band"] in ("逐位相同", "在噪声内"):
            fix = "不用管"
        lead = c["lead"] if c["band"] not in ("逐位相同",) else "—"
        rd = reading(c, d, 1000 if a["id"].startswith("L") else 300)
        rows.append(f"| {classes[a['id'][0]]} | {a['plain']} | {a['scenario']} | {c['band']} | {c['r']} | {long} | {lead} | {rd} | {cost} | {fix} |")
    return rows


def summary_table(arms, classes):
    """按类汇总：多少次实验逐位相同、多少和 GPU 重跑分不开、多少分得开（列出是哪几个）、多少未出结果（300 步；长跑类按 1000 步）。

    每一行四个数加起来必须等于实验数（arms.yaml 里登记的该类条数），不等就报错退出，--check 也会跑到这里。
    """
    rows = ["| 类 | 实验数 | 逐位相同 | 和 GPU 重跑分不开 | 分得开 | 未出结果 | 分得开的是哪几个 |", "|---|---|---|---|---|---|---|"]
    for k, name in classes.items():
        n = same = indist = missing = 0
        dist = []
        for a in arms:
            if a["id"][0] != k:
                continue
            n += 1
            d = result(a)
            if d is None:
                missing += 1
                continue
            c = cells(d)
            if c["band"] == "逐位相同":
                same += 1
            elif c["band"] == "在噪声内" or reading(c, d, 1000 if k == "L" else 300) != APART:
                indist += 1
            else:
                un = (d.get("step1_grad") or {}).get("n_unattributed") or 0
                lm = loo_max(1000 if k == "L" else 300)
                why = "第 1 步有梯度离精确值更远" if (un and (d["verdict"].get("curves_r_max") or 0) <= (lm or 0)) else f"r {c['r']}"
                dist.append(f"{a['plain']}（{why}）")
        if not n:
            continue
        if same + indist + len(dist) + missing != n:
            raise SystemExit(f"汇总表「{name}」对不上：逐位相同 {same} + 分不开 {indist} + 分得开 {len(dist)} + 未出结果 {missing} ≠ 实验数 {n}")
        rows.append(f"| {name} | {n} | {same} | {indist} | {len(dist)} | {missing} | {'；'.join(dist) or '—'} |")
    return rows


# ---------- README：按「对 TPU 精度有没有影响」分堆 ----------
# 每一种改动（不算 L 类：L 是把某个改动跑到 1000 步，并进原改动的「1000 步」一列）归进下面一堆，全部由 arms.yaml 与 results/ 算出：
#   missing 未出结果：没有结果文件
#   broken  故意改坏：S 类，或 arms.yaml 里 kind: broken
#   issue   真有影响：300 步或 1000 步「和 GPU 重跑比」是分得开；必须在 arms.yaml 里写 issue 归到某一件
#   same    逐位相同
#   blind   动了精度、但量不出来：kind: precision，且和 GPU 重跑分不开
#   noise   其余：变了，但跟 GPU 重跑一样大
HEAPS = ("same", "noise", "issue", "blind", "broken", "missing")
GROUP_BY_CLASS = {"A": "重跑与环境", "B": "重跑与环境", "G": "重跑与环境", "H": "重跑与环境", "C": "编译开关", "D": "代码写法", "E": "代码写法", "F": "版本"}
GROUP_ORDER = ("重跑与环境", "代码写法", "编译开关", "版本")


def long_result(a):
    return result({"run": a["long"], "vs": a.get("long_vs", "ts-g-a-1000")}) if a.get("long") else None


def apart(d, steps):
    c = cells(d)
    return c["band"] not in ("逐位相同", "在噪声内") and reading(c, d, steps) == APART


def heap_of(a):
    d = result(a)
    if d is None:
        return "missing"
    if a["id"][0] == "S" or a.get("kind") == "broken":
        return "broken"
    dl = long_result(a)
    if apart(d, 300) or (dl is not None and apart(dl, 1000)):
        if not a.get("issue"):
            raise SystemExit(f"{a['id']} 和 GPU 重跑分得开，但 arms.yaml 里没写 issue（归到「真有影响」的哪一件）")
        return "issue"
    if cells(d)["band"] == "逐位相同":
        return "same"
    return "blind" if a.get("kind") == "precision" else "noise"


def chip_cost(a, d, chips):
    """芯片·秒对它自己的对照的变化（同 temper_table）；没有步时就 None。"""
    st = cells(d)["st_rel"]
    if st is None:
        return None
    return (1 + st) * a.get("chips", 8) / chips.get(a.get("vs", GOLDEN), 8) - 1


def r_long(a):
    dl = long_result(a)
    return "—" if dl is None else cells(dl)["r"]


def worst_step1(d):
    """第 1 步定不了责的张量里离精确值最远的那个：(张量名, 被测离 FP64 ÷ max(golden, GPU) 离 FP64)。"""
    un = (d.get("step1_grad") or {}).get("unattributed") or []
    if not un:
        return None
    w = max(un, key=lambda x: x.get("r64") or 0)
    return w["tensor"], w.get("r64")


def heaps(arms):
    longs = {a["long"] for a in arms if a.get("long")}
    for a in arms:
        if a["id"].startswith("L") and a.get("run") not in longs:
            raise SystemExit(f"{a['id']} 的运行 {a.get('run')} 没有挂到任何改动的 long 上")
    out = {h: [] for h in HEAPS}
    for a in arms:
        if not a["id"].startswith("L"):
            out[heap_of(a)].append(a)
    n = sum(1 for a in arms if not a["id"].startswith("L"))
    if sum(len(v) for v in out.values()) != n:
        raise SystemExit(f"分堆对不上：{ {h: len(v) for h, v in out.items()} } 加起来 ≠ 改动总数 {n}")
    return out, n


def overview_lines(arms, spec):
    # V 类是同一批改动在 libtpu 0.0.52 上的重测，不算新的改动：主计数不含它，另起一句
    hv, nv = heaps([a for a in arms if a["id"].startswith("V")])
    arms = [a for a in arms if not a["id"].startswith("V")]
    h, n = heaps(arms)
    k = len({a["issue"] for a in h["issue"]})
    nl = sum(1 for a in arms if a["id"].startswith("L"))
    tail = (
        f"另有 {nv} 次是同一批改动在 libtpu 0.0.52 上的重测（V 类，[`libtpu-052.md`](libtpu-052.md)）：逐位相同 {len(hv['same'])} 次，"
        f"跟 GPU 重跑一样大 {len(hv['noise'])} 次，真有影响 {len(hv['issue'])} 次；下面各表里标「libtpu 0.0.52」的就是它们。"
        if nv
        else ""
    )
    return [f"一共 **{n} 种改动**（其中 {nl} 种另外跑到 1000 步）：逐位相同 {len(h['same'])} 种；变了、但跟 GPU 重跑一样大 {len(h['noise'])} 种；"
            f"真有影响 {len(h['issue'])} 种，归成 {k} 件；动了精度、但这把尺子量不出来 {len(h['blind'])} 种；"
            f"故意改坏、用来证明工具抓得到的 {len(h['broken'])} 种；登记了、没出结果的 {len(h['missing'])} 种。"
            f"{len(h['same'])} + {len(h['noise'])} + {len(h['issue'])} + {len(h['blind'])} + {len(h['broken'])} + {len(h['missing'])} = {n}。" + tail]


def same_table(arms):
    h, _ = heaps(arms)
    rows = ["| 分组 | 改动 | 场景 |", "|---|---|---|"]
    by = {g: [] for g in GROUP_ORDER}
    for a in h["same"]:
        by[a.get("group") or GROUP_BY_CLASS[a["id"][0]]].append(a)
    for g in GROUP_ORDER:
        for a in by[g]:
            rows.append(f"| {g} | {a['plain']} | {a['scenario']} |")
    return rows


def noise_table(arms, classes):
    h, _ = heaps(arms)
    chips = {x["run"]: x.get("chips", 8) for x in arms if x.get("run")}
    rows = ["| 类 | 改动 | 场景 | 300 步 r | 1000 步 r | 多花的算力 |", "|---|---|---|---|---|---|"]
    for a in h["noise"]:
        d = result(a)
        c = chip_cost(a, d, chips)
        rows.append(f"| {classes[a['id'][0]]} | {a['plain']} | {a['scenario']} | {cells(d)['r']} | {r_long(a)} | {'—' if c is None else f'{c:+.1%}'} |")
    return rows


def issue_table(arms, spec):
    """真有影响：每件一行摘要，标题链到 impacts/ 下的详解。"""
    h, _ = heaps(arms)
    by_id = {a["id"]: a for a in arms}
    rows = ["| 件 | 涉及的实验 · 曲线最差 r | 第 1 步变远的张量 · R（离 FP64 ÷ max(golden, GPU 中位数)，第一次实验） | 能不能忍 | 怎么修 | 修后 r |", "|---|---|---|---|---|---|"]
    issues = spec.get("issues") or {}
    keys = []
    for a in h["issue"]:
        if a["issue"] not in issues:
            raise SystemExit(f"{a['id']} 的 issue「{a['issue']}」在 arms.yaml 的 issues 里没有登记")
        if a["issue"] not in keys:
            keys.append(a["issue"])
    for key in keys:
        it = issues[key]
        if not (HERE / it["file"]).exists():
            raise SystemExit(f"issue「{key}」的详解文件 {it['file']} 不存在")
        mine = [a for a in h["issue"] if a["issue"] == key]
        size, worst = [], None
        for a in mine:
            d = result(a)
            t = f"{a['plain']} {cells(d)['r']}"
            if a.get("long") and r_long(a) != "—":
                t += f"（1000 步 {r_long(a)}）"
            size.append(t)
            w = worst_step1(d)
            if w and worst is None:   # 取这一件里排在最前、第 1 步有变远张量的那次（第一行是这件事的代表，后面可能有已修的 bug）
                worst = w
        after = []
        for fx in it.get("fix_arms") or []:
            if fx["arm"] == "golden":
                after.append("默认即 golden，逐位相同")
                continue
            f = by_id[fx["arm"]]
            t = f"{f['plain']} {cells(result(f))['r']}"
            if f.get("long") and r_long(f) != "—":
                t += f"（1000 步 {r_long(f)}）"
            after.append(t)
        w = "—" if worst is None else f"`{worst[0]}`，R {worst[1]:.2f}"
        rows.append(f"| [{it['title']}]({it['file']}) | {'；'.join(size)} | {w} | {it['verdict']} | {it['fix']} | {'；'.join(after) or '—'} |")
    return rows


def broken_table(arms):
    h, _ = heaps(arms)
    rows = ["| 改了什么 | 300 步 r | 差得最多的曲线·统计量 | 和 GPU 重跑比 |", "|---|---|---|---|"]
    for a in h["broken"]:
        d = result(a)
        c = cells(d)
        rows.append(f"| {a['plain']} | {c['r']} | {c['lead']} | {reading(c, d, 300)} |")
    return rows


def blind_table(arms):
    h, _ = heaps(arms)
    rows = ["| 改了什么 | 300 步 r | 1000 步 r |", "|---|---|---|"]
    for a in h["blind"]:
        rows.append(f"| {a['plain']} | {cells(result(a))['r']} | {r_long(a)} |")
    lost = [a["plain"] for a in h["broken"] if not apart(result(a), 300)]
    if lost:
        rows += ["", f"故意埋的 bug 里，{'、'.join(lost)} 也和 GPU 重跑分不开。"]
    return rows


def splice(text, name, lines):
    pat = re.compile(rf"(<!-- BEGIN:{name} -->\n).*?(<!-- END:{name} -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"缺标记 {name}")
    return pat.sub(lambda m: m.group(1) + "\n".join(lines) + "\n" + m.group(2), text)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="只核对生成结果与文件一致，不写")
    a = p.parse_args(argv)
    spec = yaml.safe_load((HERE / "arms.yaml").read_text())
    arms, classes = spec["arms"], spec["classes"]
    changed = []
    matrix_parts = [(f"matrix-{k}", matrix_table(k, arms)) for k in classes]
    matrix_parts += [("temper", temper_table(arms, classes)), ("summary", summary_table(arms, classes)), ("screen", screen_table(arms))]
    readme_parts = [("overview", overview_lines(arms, spec)), ("same", same_table(arms)), ("noise", noise_table(arms, classes)),
                    ("issues", issue_table(arms, spec)), ("broken", broken_table(arms)), ("blind", blind_table(arms))]
    for fname, parts in (("MATRIX.md", matrix_parts), ("README.md", readme_parts)):
        f = HERE / fname
        old = f.read_text()
        new = old
        for name, lines in parts:
            new = splice(new, name, lines)
        if new != old:
            changed.append(fname)
            if not a.check:
                f.write_text(new)
    if a.check and changed:
        print("需要重新生成：" + "、".join(changed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
