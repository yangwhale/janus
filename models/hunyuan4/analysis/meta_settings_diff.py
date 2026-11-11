"""几份运行的 meta.json 逐字段对一份参照：配置（除 load_check）、代码、镜像、并行、版本、数据、模型、步数。

用途：命令行没有入库的 GPU 运行（如 `g2-*`、`g2s-*`），用 meta.json 证明它们与有完整命令行的参照（如 `g3-a`，
完整附加参数在 05-gpu-scale/setup.md 的 `G3`）只差哪几项。只比 meta.json 记下的字段；meta 没记的参数比不到。

    python meta_settings_diff.py --ref g3-a=<meta.json> --run g2-a=<meta.json> … --out <json>
"""

from __future__ import annotations

import argparse
import hashlib
import json

FIELDS = ("config", "code", "image", "parallelism", "versions", "data", "model", "steps")
SKIP = {"config.load_check"}  # 每份运行载入自检的读数，不是设置


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if key in SKIP:
            continue
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def load(path: str) -> tuple[dict, str]:
    raw = open(path, "rb").read()
    m = json.loads(raw)
    sel = {k: m[k] for k in FIELDS if k in m}
    return flat(sel), hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="名字=meta.json")
    p.add_argument("--run", nargs="+", required=True, help="名字=meta.json")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rn, rp = a.ref.split("=", 1)
    ref, rsha = load(rp)
    diffs, shas = {}, {rn: rsha}
    for spec in a.run:
        n, path = spec.split("=", 1)
        cur, shas[n] = load(path)
        keys = sorted(set(ref) | set(cur))
        diffs[n] = [{"field": k, rn: ref.get(k), n: cur.get(k)} for k in keys if ref.get(k) != cur.get(k)]
    out = {
        "what": "meta.json 逐字段对参照（只比 meta 记下的字段）",
        "reference": rn,
        "compared": [f for f in FIELDS] + ["（config.load_check 除外）"],
        "meta_sha256": shas,
        "differences": diffs,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for n, d in diffs.items():
        print(n, [x["field"] for x in d])


if __name__ == "__main__":
    main()
