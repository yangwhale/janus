"""从 exec_spread_300.py 已写的结果里去掉某些项，重算「升得最多的一项」的分布；另记每份运行与执行方式无关的设置差。

exec-spread-300*.json 的两组里，各有一份早一代运行（开组 `t1zw-a`、关组 `t1zz-a`）没有 M50
（`indexer_kd_generalized_kl`，只改索引器蒸馏 loss 的记录值，梯度不变），MaxText 提交也与另 3 份不同。
含这份运行的有序对里，蒸馏 loss 一项的升幅混进了记录写法的差。这里只读结果文件里已存的每份运行各项 r，
去掉指定的项（默认 `L4.indexer_loss`）后按原脚本同一规则（同组有序对 D_ij = max_项 (r_j − r_i)）重算；
不需要运行目录。给了 `--meta` 时，从各运行的 meta.json 读 M50、base2、MaxText 提交、libtpu_pip，连同 sha256 一起写进结果。

    python exec_spread_drop_items.py --in <exec-spread-300.json> <exec-spread-300-g3.json> \\
        [--drop L4.indexer_loss] [--meta <运行名>=<meta.json> …] --out <json>
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os

import numpy as np


def spread(res: dict, drop: tuple[str, ...]) -> dict:
    r = res["r"]
    pairs = []
    for g, members in res["groups"].items():
        for i, j in itertools.permutations(members, 2):
            best, item = -np.inf, None
            for k, v in r[i].items():
                if k.split("·")[0] in drop or k not in r[j]:
                    continue
                d = r[j][k] - v
                if d > best:
                    best, item = d, k
            pairs.append({"group": g, "from": i, "to": j, "max_rise": float(best), "item": item})
    d = np.array([p["max_rise"] for p in pairs])
    pairs.sort(key=lambda p: -p["max_rise"])
    return {
        "max_rise_over_ordered_pairs": {
            "n": len(d),
            "min": float(d.min()),
            "median": float(np.median(d)),
            "p90": float(np.quantile(d, 0.9)),
            "max": float(d.max()),
        },
        "top_pairs": pairs[:5],
    }


def meta_info(path: str) -> dict:
    raw = open(path, "rb").read()
    m = json.loads(raw)
    cfg = m.get("maxtext_config_full") or {}
    code = m.get("code") or []
    mt = [c["commit"][:8] for c in code if isinstance(c, dict) and c.get("repo", "").endswith("maxtext")]
    return {
        "meta_sha256": hashlib.sha256(raw).hexdigest(),
        "maxtext_commit": mt[0] if mt else None,
        "indexer_kd_generalized_kl": cfg.get("indexer_kd_generalized_kl"),
        "sa_use_base2_exp": cfg.get("sa_use_base2_exp"),
        "libtpu_pip": (m.get("versions") or {}).get("libtpu_pip"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inputs", nargs="+", required=True)
    p.add_argument("--drop", nargs="+", default=["L4.indexer_loss"])
    p.add_argument("--meta", nargs="*", default=[])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    drop = tuple(a.drop)
    out = {
        "what": "exec-spread-300 的「升得最多的一项」去掉指定项后重算（只读已存的各项 r，规则同 exec_spread_300.py）",
        "dropped_items": list(drop),
        "results": {},
    }
    for path in a.inputs:
        res = json.load(open(path))
        name = os.path.basename(path)
        out["results"][name] = {
            "ruler": res["ruler"],
            "groups": res["groups"],
            "input_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "all_items": spread(res, ()),
            "without_dropped": spread(res, drop),
        }
    if a.meta:
        out["run_settings"] = {s.split("=", 1)[0]: meta_info(s.split("=", 1)[1]) for s in a.meta}
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for n, v in out["results"].items():
        x, y = v["all_items"]["max_rise_over_ordered_pairs"], v["without_dropped"]["max_rise_over_ordered_pairs"]
        print(
            f"{n}: 全部项 中位 {x['median']:.2f} p90 {x['p90']:.2f} max {x['max']:.2f}；去掉后 中位 {y['median']:.2f} p90 {y['p90']:.2f} max {y['max']:.2f}"
        )


if __name__ == "__main__":
    main()
