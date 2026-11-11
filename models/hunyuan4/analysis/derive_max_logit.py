"""从开了 `qk_clip_log_per_head` 的 TPU 运行派生「每步全局最大注意力 logit」，写成一个新运行目录给 L4 QK-Clip 曲线用。

`max_attention_logit` = 各层（含 MTP）各头 `qk.max_logit.<层>` 的最大值，与 GPU 的 `max_attention_logit` 同义
（全局最大缩放 q·k，不含 sink；见 `catalogs/t1mz-l4-qkclip.yaml`）。新目录的 metrics.jsonl 是原文件每行加这一个字段，
其余文件（meta.json、DONE 等）照原目录软链。

    python models/hunyuan4/analysis/derive_max_logit.py --run <运行目录> --out <新运行目录>
"""

from __future__ import annotations

import argparse
import json
import os


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    src = os.path.abspath(a.run)
    for name in os.listdir(src):
        if name != "metrics.jsonl" and not os.path.exists(os.path.join(a.out, name)):
            os.symlink(os.path.join(src, name), os.path.join(a.out, name))
    n = 0
    with open(os.path.join(src, "metrics.jsonl")) as f, open(os.path.join(a.out, "metrics.jsonl"), "w") as g:
        for line in f:
            r = json.loads(line)
            heads = [v for k, v in r.items() if k.startswith("qk.max_logit.")]
            if heads:
                r["max_attention_logit"] = max(max(h) for h in heads)
                n += 1
            g.write(json.dumps(r) + "\n")
    print(n, "steps with max_attention_logit")


if __name__ == "__main__":
    main()
