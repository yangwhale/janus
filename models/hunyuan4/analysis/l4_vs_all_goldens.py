"""诊断：L4 每一代 TPU 对全部 8 份 GPU golden 各算一遍，报 8 个 r 的中位数与最大值，旁列 GPU 两两 28 对的留一 r
（中位数与最大值）与 GPU 自检。只报不判：L4 的正式判定只对单一 golden（PROTOCOL 第 3 节，变更记录 2026-11-09）。

    python models/hunyuan4/analysis/l4_vs_all_goldens.py --root <运行目录根> --work <工作目录> \\
        --tpu t1zw=<正式目录文件 t1zw-l4.yaml> [--tpu ...] --out <json>

- `--tpu <代号>=<目录文件>`：正式比较的目录文件（golden 是 <前缀>-a）；脚本把 golden / baseline 依次换成 <前缀>-a…h、
  噪声 28 对不变（所以 N 不变），在 `--work` 下各跑一遍 `janus run`；r = |判定量| ÷ N，与正式判定同一口径。
  也可以给已跑好的目录模板（含 `{g}`，换成 a…h）。
- GPU 留一 r：一对 GPU 的判定量 ÷ 其余 27 对的最大值（一对不能拿包含自己的上限来判），取正式结果里的 `noise_loo`；
  没有 `noise_loo` 的检查（路由偏置统计）由脚本另跑 28 个「一对 GPU 当被测、其余 27 对当噪声」的目录补上。
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import yaml

KEYS = ("signed_mean", "mare", "max_step")
GOLDENS = "abcdefgh"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _r(check: dict) -> dict:
    up, m = check["threshold"]["noise"]["upper"], check["metrics"]
    return {k: abs(m.get(k, m.get(k + "_max"))) / up[k] for k in KEYS}


def gpu_self_check(out, pairs_named):
    """GPU 自检：把每份 GPU 当被测，用它与其余 7 份的留一 r 算中位数 / 最大（与 TPU 对 8 份的中位数 / 最大并看）。"""
    res = {}
    for cid, pairs in pairs_named.items():
        for k in KEYS:
            per = {}
            for x in out["goldens"]:
                vals = [v[k] for p, v in pairs.items() if x in p.split("~") and v.get(k) is not None]
                if vals:
                    per[x] = {"median": statistics.median(vals), "max": max(vals)}
            res.setdefault(cid, {})[k] = per
    return res


def _janus(cat: dict, root: str, path: str, out: str) -> str:
    if not os.path.exists(os.path.join(out, "results.json")):
        with open(path, "w") as f:
            yaml.safe_dump(cat, f, allow_unicode=True, sort_keys=False)
        r = subprocess.run([sys.executable, "-m", "janus", "run", path, "--root", root, "--out", out], cwd=REPO,
                           capture_output=True, text=True)  # fmt: skip
        if not os.path.exists(os.path.join(out, "results.json")):
            raise SystemExit(f"janus run 失败：{path}\n{r.stderr[-2000:]}")
    return out


def _swap(cat: dict, old: str, new: str) -> dict:
    """golden 与各检查 params 里的 baseline 换成另一份 GPU；噪声对不动。"""
    txt = yaml.safe_dump(cat, allow_unicode=True, sort_keys=False)
    c = yaml.safe_load(txt)
    c["runs"]["golden"] = [new if g == old else g for g in c["runs"]["golden"]]
    for k in c["checks"]:
        if (k.get("params") or {}).get("baseline") == old:
            k["params"]["baseline"] = new
    return c


def run_catalogs(gens: dict, root: str, work: str):
    """目录文件 → 每代 8 份结果目录模板；另返回补 GPU 留一 r 的目录模板（只含没有 noise_loo 的检查）与 GPU 名字。"""
    tpls, prefix, first = {}, None, None
    for gen, src in gens.items():
        if not src.endswith((".yaml", ".yml")):
            tpls[gen] = src
            continue
        os.makedirs(work, exist_ok=True)
        cat = yaml.safe_load(open(src))
        base = cat["runs"]["golden"][0]
        p = base.rsplit("-", 1)[0]
        prefix = prefix or p
        first = first or cat
        jobs = []
        for g in GOLDENS:
            name = f"{gen}-{p}-{g}"
            jobs.append(
                (_swap(cat, base, f"{p}-{g}"), root, os.path.join(work, name + ".yaml"), os.path.join(work, name))
            )
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(lambda j: _janus(*j), jobs))
        tpls[gen] = os.path.join(work, f"{gen}-{p}-{{g}}")
    return tpls, prefix, first


def gpu_loo_dirs(cat: dict, prefix: str, root: str, work: str, need: set) -> dict:
    runs = [f"{prefix}-{g}" for g in GOLDENS]
    pairs = list(itertools.combinations(runs, 2))
    out, jobs = {}, []
    for x, y in pairs:
        c = yaml.safe_load(yaml.safe_dump(cat, allow_unicode=True, sort_keys=False))
        c["stage"] = f"gpu-{x}-{y}"
        c["runs"] = {**c["runs"], "test": y, "golden": [x], "noise": [list(q) for q in pairs if q != (x, y)]}
        c["checks"] = [k for k in c["checks"] if k["id"] in need]
        for k in c["checks"]:
            if (k.get("params") or {}).get("baseline"):
                k["params"]["baseline"] = x
        name = f"gpu-{x}-{y}"
        out[(x, y)] = os.path.join(work, name)
        jobs.append((c, root, os.path.join(work, name + ".yaml"), out[(x, y)]))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: _janus(*j), jobs))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tpu", action="append", required=True, help="代号=目录文件（.yaml）或已跑好的目录模板（含 {g}）")
    p.add_argument("--root", default="", help="运行目录根（给目录文件时必填）")
    p.add_argument("--work", default="", help="生成的目录文件与 janus 结果放这里（给目录文件时必填）")
    p.add_argument("--gpu-loo", default="", help="已跑好的 GPU 留一目录模板（含 {x}、{y}，换成两份 GPU 的全名）")
    p.add_argument("--prefix", default="g3", help="只给目录模板时 GPU 名字的前缀")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    gens = dict(s.split("=", 1) for s in a.tpu)
    tpls, prefix, first = run_catalogs(gens, a.root, a.work)
    prefix = prefix or a.prefix
    names = [f"{prefix}-{g}" for g in GOLDENS]
    out = {"what": __doc__.split("\n")[0], "goldens": names, "generations": {}, "gpu_loo": {}}
    gen0 = next(iter(gens))
    pairs_named: dict = {}
    for gen, tpl in tpls.items():
        per = {}
        for g in GOLDENS:
            d = json.load(open(os.path.join(tpl.format(g=g), "results.json")))
            for c in d["checks"]:
                if "noise" not in (c.get("threshold") or {}):
                    continue
                row = per.setdefault(c["id"], {"r": {k: [] for k in KEYS}, "status": []})
                for k, v in _r(c).items():
                    row["r"][k].append(v)
                row["status"].append(c["status"])
                if gen == gen0 and g == "a":
                    loo = c["metrics"].get("noise_loo")
                    if loo:
                        out["gpu_loo"][c["id"]] = {k: [x[k] for x in loo if x.get(k) is not None] for k in KEYS}
                        pairs_named.setdefault(c["id"], {}).update({x["pair"]: {k: x[k] for k in KEYS} for x in loo})
        out["generations"][gen] = {
            cid: {
                "r_vs_each": {k: dict(zip(names, v["r"][k])) for k in KEYS},
                "r_median": {k: statistics.median(v["r"][k]) for k in KEYS},
                "r_max": {k: max(v["r"][k]) for k in KEYS},
                "r_vs_a": {k: v["r"][k][0] for k in KEYS},
                "status_vs_each": dict(zip(GOLDENS, v["status"])),
            }
            for cid, v in per.items()
        }
    need = {cid for cid in out["generations"][gen0] if cid not in out["gpu_loo"]}
    dirs = {}
    if need and first is not None:
        dirs = gpu_loo_dirs(first, prefix, a.root, a.work, need)
    elif need and a.gpu_loo:
        dirs = {(x, y): a.gpu_loo.format(x=x, y=y) for x, y in itertools.combinations(names, 2)}
    for (x, y), dd in dirs.items():
        for c in json.load(open(os.path.join(dd, "results.json")))["checks"]:
            if c["id"] in out["gpu_loo"] or c["id"] not in need:
                continue
            rr = _r(c)
            acc = out.setdefault("_gpu_loo_extra", {}).setdefault(c["id"], {k: [] for k in KEYS})
            for k, v in rr.items():
                acc[k].append(v)
            pairs_named.setdefault(c["id"], {})[f"{x}~{y}"] = rr
    out["gpu_loo"].update(out.pop("_gpu_loo_extra", {}))
    out["gpu_loo_summary"] = {
        cid: {
            "median": {k: statistics.median(v[k]) for k in KEYS},
            "max": {k: max(v[k]) for k in KEYS},
            "n": len(v[KEYS[0]]),
        }
        for cid, v in out["gpu_loo"].items()
    }
    out["gpu_self_check"] = gpu_self_check(out, pairs_named)
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for cid, s in out["gpu_loo_summary"].items():
        for k in KEYS:
            line = f"{cid:20s} {k:11s} GPU28 {s['median'][k]:.2f}/{s['max'][k]:.2f} |"
            for gen in tpls:
                gsum = out["generations"][gen].get(cid)
                if gsum:
                    line += f" {gen} {gsum['r_median'][k]:.2f}/{gsum['r_max'][k]:.2f} (a {gsum['r_vs_a'][k]:.2f}) |"
            print(line)


if __name__ == "__main__":
    main()
