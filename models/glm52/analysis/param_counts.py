"""GLM-5.2 的精确参数量：只读 Hugging Face checkpoint 各分片的 safetensors 头（HTTP Range），按层累加。

    python models/glm52/analysis/param_counts.py --json models/glm52/results/param-counts.json

固定 revision。输出全模型总数、非层张量（嵌入、输出头、最后的 norm）、若干层的参数量、各截层方案的合计，
以及哪些层带索引器权重（config 的 indexer_types 为 full 的层）。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import struct
import urllib.request

REV = "cf457fa734ab149ffef225f80893eb38c6ff5cdc"
BASE = f"https://huggingface.co/zai-org/GLM-5.2/resolve/{REV}/"


def rng(f, a, b):
    r = urllib.request.Request(BASE + f, headers={"Range": f"bytes={a}-{b}"})
    return urllib.request.urlopen(r, timeout=120).read()


def header(f):
    n = struct.unpack("<Q", rng(f, 0, 7))[0]
    h = json.loads(rng(f, 8, 7 + n))
    h.pop("__metadata__", None)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    cfg = json.loads(urllib.request.urlopen(BASE + "config.json").read())
    idx = json.loads(urllib.request.urlopen(BASE + "model.safetensors.index.json").read())["weight_map"]
    shapes = {}
    with cf.ThreadPoolExecutor(16) as ex:
        for h in ex.map(header, sorted(set(idx.values()))):
            shapes.update({k: v["shape"] for k, v in h.items()})

    def layer(i):
        return sum(math.prod(v) for k, v in shapes.items() if k.startswith(f"model.layers.{i}."))

    non_layer = sum(math.prod(v) for k, v in shapes.items() if not k.startswith("model.layers."))
    lay = {i: layer(i) for i in (0, 1, 2, 3, 6, 7, cfg["num_hidden_layers"])}
    d = {
        "hf_revision": REV,
        "tensors": len(shapes),
        "total": sum(math.prod(v) for v in shapes.values()),
        "non_layer": non_layer,
        "layer": lay,
        "stage1": non_layer + lay[0] + lay[6],
        "stage1_mtp": non_layer + lay[0] + lay[6] + lay[cfg["num_hidden_layers"]],
        "stage1_l7": non_layer + lay[0] + lay[6] + lay[7],
        "indexer_full_layers": [i for i, t in enumerate(cfg["indexer_types"]) if t == "full"],
        "indexer_tensors_per_layer": {
            i: sum(1 for k in shapes if k.startswith(f"model.layers.{i}.self_attn.indexer."))
            for i in (0, 1, 2, 3, 6, 7)
        },
    }
    txt = json.dumps(d, indent=1)
    if a.json:
        open(a.json, "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
