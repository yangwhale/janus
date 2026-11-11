"""TPU 自比第 1 步：把 GPU 参照从 3 份（g2s-a … c）换成 6 份（再加 g2s-d … f）重判，只用已入库的数。

自比结果文件（04-tpu-self/results/*-vs-*.json 的 step1_rows）逐张量存了被测对 golden 的差 d、被测与 golden 各自离 FP64 的距离；
01-mini 的 6 份结果（results/t1zo-step1-train-grad-vs-fp64-n6.json 的 rows）逐张量存了 6 份 GPU 两两差的最大值（分母 ‖GPU‖，
gpu_noise_max）与 6 份 GPU 离 FP64 的中位数（gpu_vs_ref）。GPU 一侧与 TPU 无关，换掉这两样就能按 tpu_self_compare.py 的同一判法重判：
- 尺子：d ≤ max(GPU 两两最大差, 2^-20) 记在噪声内；
- 定责：超出尺子的，R = 被测离 FP64 ÷ max(golden 离 FP64, GPU 离 FP64 的中位数) ≤ 1.2 记换了顺序，否则 iHC 单列、其余未定责。
先用 3 份的数复算一遍，和结果文件里存的判定逐张量对上，再出 6 份的判定。只报判定有变化的张量；判档、曲线不重算。

    python models/hunyuan4/analysis/tpu_self_step1_rejudge_n6.py \
        --out models/hunyuan4/alignments/04-tpu-self/results/step1-n6-rejudge.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N6 = ROOT / "alignments/01-mini/results/t1zo-step1-train-grad-vs-fp64-n6.json"
RESULTS = ROOT / "alignments/04-tpu-self/results"
FLOOR = 2.0**-20
R64_LIMIT = 1.2
IHC = re.compile(r"\.hc_(attn|mlp)_layer\.|^model\.hc_head\.")
STORED = {"scale": "scale", "attributed": "attributed", "ihc_accepted": "ihc_accepted", "unattributed": "unattributed"}


def judge(row, scale, med):
    over = row["d"] / max(scale, FLOOR)
    ref = max(row["golden_vs_fp64"], med)
    r64 = row["test_vs_fp64"] / ref if ref else (1.0 if row["test_vs_fp64"] == 0 else math.inf)
    if over <= 1:
        v = "scale"
    elif r64 <= R64_LIMIT:
        v = "attributed"
    else:
        v = "ihc_accepted" if IHC.search(row["tensor"]) else "unattributed"
    return v, r64, over


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    n6 = {r["tensor"]: r for r in json.loads(N6.read_text())["rows"]}
    runs, n_files = [], 0
    for f in sorted(RESULTS.glob("*-vs-*.json")):
        d = json.loads(f.read_text())
        rows = d.get("step1_rows") if isinstance(d, dict) else None
        if not rows or "r64" not in rows[0]:
            continue
        n_files += 1
        changed, cnt = [], {"3": {}, "6": {}}
        for r in rows:
            g = n6[r["tensor"]]
            v3, r3, o3 = judge(r, r["scale"], r["gpu_vs_fp64"])
            if v3 != STORED[r["verdict"]]:
                raise SystemExit(f"{f.name} {r['tensor']}：3 份复算 {v3} 与结果文件 {r['verdict']} 对不上")
            v6, r6, o6 = judge(r, g["gpu_noise_max"], g["gpu_vs_ref"])
            cnt["3"][v3] = cnt["3"].get(v3, 0) + 1
            cnt["6"][v6] = cnt["6"].get(v6, 0) + 1
            if v6 != v3:
                changed.append(
                    {
                        "tensor": r["tensor"],
                        "n3": v3,
                        "n6": v6,
                        "r64_n3": r3,
                        "r64_n6": r6,
                        "over_scale_n3": o3,
                        "over_scale_n6": o6,
                    }
                )
        if changed:
            v = d.get("verdict", {})
            runs.append(
                {
                    "file": f.name,
                    "band": v.get("band"),
                    "curves_r_max": v.get("curves_r_max"),
                    "n_unattributed": [cnt["3"].get("unattributed", 0), cnt["6"].get("unattributed", 0)],
                    "n_ihc_accepted": [cnt["3"].get("ihc_accepted", 0), cnt["6"].get("ihc_accepted", 0)],
                    "changed": changed,
                }
            )
    out = {
        "what": "TPU 自比第 1 步按 6 份 GPU 参照重判（只换尺子与中位数分母，判档、曲线不重算）",
        "how": "python models/hunyuan4/analysis/tpu_self_step1_rejudge_n6.py --out <本文件>",
        "inputs": {"n6": str(N6.relative_to(ROOT)), "results": str(RESULTS.relative_to(ROOT))},
        "n_files": n_files,
        "n3_recheck": "3 份复算与各结果文件存的判定逐张量相同",
        "unattributed_changed": [x for x in runs if x["n_unattributed"][0] != x["n_unattributed"][1]],
        "runs": runs,
    }
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(f"{n_files} 个结果文件，{len(runs)} 个有判定变化，未定责个数变了的 {len(out['unattributed_changed'])} 个")


if __name__ == "__main__":
    main()
