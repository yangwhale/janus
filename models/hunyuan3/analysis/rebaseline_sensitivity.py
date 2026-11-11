"""换一份当基线：正式第一层 L4 各量以 rcc-a..j 每一份当基线各判一遍，看 r 实际挪动多少。

正式比较只对 rcc-a（比较协议事先指定）。任意两份 golden 之间的差不超过 N，所以换基线 r 最多挪动 1；
这里把实际挪动量逐份算出来。做法：读正式第一层的四份目录（v5p / v7x × 当前配置 / EP = 8），
只把 runs.golden 与 multi_golden 的 params.baseline 换成另一份 rcc，噪声对（rcc-a..j 两两 45 对）、
窗口、分母、被测列一概不动，用 janus 的 Runner 原样执行 L4 各条检查。基线取 rcc-a 时与已提交的
results.json 逐位相同（脚本核对 janus 的 r1、r2 与状态）。

用法（仓库根目录）：
  python models/hunyuan3/analysis/rebaseline_sensitivity.py --root <运行根> \
      --json models/hunyuan3/alignments/01-v5p-acceptance/results/case01-rebaseline-rcc.json
运行根下要有 rcc-a..j 与四个被测运行（t-rmap-on-1000、t-ep8-1000、t-v7x-on-1000、t-v7x-ep8-1000）的
meta.json / metrics.jsonl；每份 meta.json 的 sha256 先和已提交 results.json 里记的核对，对不上就退出。
"""

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))  # 仓库根：没 pip install 也能 import janus

from janus.catalog import display_path, load  # noqa: E402
from janus.run import Runner  # noqa: E402

A = "models/hunyuan3/alignments"
CONFIGS = {
    "v5p_fsdp8": (
        f"{A}/01-v5p-acceptance/catalogs/phase2-acceptance-rcc-on.yaml",
        f"{A}/01-v5p-acceptance/results/phase2-acceptance-rcc-on.results.json",
    ),
    "v5p_ep8": (
        f"{A}/01-v5p-acceptance/catalogs/phase2-acceptance-rcc-ep8.yaml",
        f"{A}/01-v5p-acceptance/results/phase2-acceptance-rcc-ep8.results.json",
    ),
    "v7x_fsdp8": (
        f"{A}/02-v7x-acceptance/catalogs/phase2-v7x-acceptance-rcc.yaml",
        f"{A}/02-v7x-acceptance/results/phase2-v7x-acceptance-rcc.results.json",
    ),
    "v7x_ep8": (
        f"{A}/02-v7x-acceptance/catalogs/phase2-v7x-acceptance-rcc-ep8.yaml",
        f"{A}/02-v7x-acceptance/results/phase2-v7x-acceptance-rcc-ep8.results.json",
    ),
}
BASELINES = [f"rcc-{c}" for c in "abcdefghij"]
JUDGED = [
    "L4.loss",
    "L4.loss.tail",
    "L4.grad_norm",
    "L4.param_norm",
]  # 正式判定的四个 L4 量（lr 是逐位核对，与基线无关）
REPORT_ONLY = ["L4.grad_norm.golden_den", "L4.param_norm.raw"]


def _row(res) -> dict:
    """一条检查的读数：三个判定量各自的 r（= 被测值 / N，N 已含 signed_mean 的标准误下限）与最差值 r_worst。
    janus 的 r1 只是 MARE 那一项、r2 是带符号均值那一项，这里三项都列。signed_mean 带符号
    （multi_golden 的 metrics 只记绝对值，带符号的从逐基线明细里取）。"""
    d = res.to_json()
    m = d.get("metrics") or {}
    up = ((d.get("threshold") or {}).get("noise") or {}).get("upper") or {}
    if d["type"] == "multi_golden":
        signed = res.details[0]["signed_mean"]
        vals = {k: m.get(f"{k}_max") for k in ("signed_mean", "mare", "max_step")}
    else:
        signed = m.get("signed_mean")
        vals = {"signed_mean": abs(signed), "mare": m.get("mare"), "max_step": m.get("max_step")}
    r = {f"r_{k}": vals[k] / up[k] for k in vals if vals.get(k) is not None and up.get(k)}
    return {
        "status": d["status"],
        "r_worst": max(r.values()),
        **r,
        "signed_mean": signed,
        "mare": vals["mare"],
        "max_step": vals["max_step"],
        "janus_r1": d.get("r1"),
        "janus_r2": d.get("r2"),
    }


def run_config(root: Path, cat_path: str, committed: dict) -> dict:
    base = load(ROOT / cat_path)
    base.data["runs"]["root"] = str(root)
    # 先核 meta.json 的 sha256
    runner0 = Runner(base)
    for name, info in committed["runs"].items():
        got = runner0.run(name).meta_sha256()
        if got != info["meta_sha256"]:
            raise SystemExit(f"{name}：meta.json sha256 {got} 与已提交结果里记的 {info['meta_sha256']} 不一致")
    ids = JUDGED + REPORT_ONLY
    out = {"test": base.runs["test"], "baselines": {}}
    for b in BASELINES:
        cat = copy.deepcopy(base)
        cat.data["runs"]["golden"] = [b]
        for c in cat.checks:
            if c["type"] == "multi_golden":
                c.setdefault("params", {})["baseline"] = b
        runner = Runner(cat)
        rows = {}
        for c in cat.checks:
            if c["id"] in ids:
                rows[c["id"]] = _row(runner.execute(c))
        out["baselines"][b] = rows
    # 基线 rcc-a 必须复现已提交的 r1、r2 与状态
    ref = {c["id"]: c for c in committed["checks"]}
    for i in ids:
        got = out["baselines"]["rcc-a"][i]
        want = (ref[i].get("r1"), ref[i].get("r2"), ref[i]["status"])
        if (got["janus_r1"], got["janus_r2"], got["status"]) != want:
            raise SystemExit(f"{cat_path} {i}：基线 rcc-a 的 (r1, r2, 状态) {got} 与已提交的 {want} 不同")
    summ = {}
    for i in ids:
        rs = [out["baselines"][b][i]["r_worst"] for b in BASELINES]
        sms = [out["baselines"][b][i]["signed_mean"] for b in BASELINES]
        ra = out["baselines"]["rcc-a"][i]["r_worst"]
        worst_b = BASELINES[rs.index(max(rs))]
        summ[i] = {
            "r_worst_rcc_a": ra,
            "r_worst_min": min(rs),
            "r_worst_max": max(rs),
            "r_worst_max_baseline": worst_b,
            "r_worst_max_shift_vs_rcc_a": max(abs(r - ra) for r in rs),
            "non_pass": {
                b: out["baselines"][b][i]["status"] for b in BASELINES if out["baselines"][b][i]["status"] != "pass"
            },
            "signed_mean_min": min(sms),
            "signed_mean_max": max(sms),
            "statuses": sorted({out["baselines"][b][i]["status"] for b in BASELINES}),
            "n_pass": sum(out["baselines"][b][i]["status"] == "pass" for b in BASELINES),
        }
    out["summary"] = summ
    out["noise_upper"] = {
        c["id"]: (c.get("threshold") or {}).get("noise", {}).get("upper") for c in committed["checks"] if c["id"] in ids
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, type=Path, help="运行根目录（rcc-a..j 与四个被测运行）")
    ap.add_argument("--json", type=Path, help="写结果 JSON")
    args = ap.parse_args(argv)
    res = {
        "what": (
            "正式第一层 L4 以 rcc-a..j 每一份当基线各判一遍。只换 runs.golden 与 multi_golden 的 baseline，"
            "噪声对（rcc-a..j 两两 45 对）、窗口、分母（grad_norm 两边均值）、被测列（param_norm 读 param_norm_from_groups）"
            "与正式目录相同；基线 rcc-a 的 janus r1（MARE 比值）、r2（带符号均值比值）与状态和已提交 results.json 逐位相同"
            "（脚本核对过）。r_signed_mean / r_mare / r_max_step = 各判定量 ÷ N（N 取 45 对最大值，loss 两项的带符号均值"
            "另有标准误下限），r_worst 取三者最大，≤ 1 过、> 2 不过、其间待定；signed_mean 是对该基线的带符号均值。L4.param_norm.raw（被测读原始 param_norm）"
            "与 L4.grad_norm.golden_den（以 golden 为分母）只报，与案例 01 注 31 当时算 rcc-a..e 的口径相同的是 "
            "L4.param_norm.raw 与 L4.loss.tail 的 signed_mean"
        ),
        "script": display_path(Path(__file__)),
        "baselines": BASELINES,
        "configs": {},
    }
    for key, (cat_path, res_path) in CONFIGS.items():
        committed = json.loads((ROOT / res_path).read_text())
        r = run_config(args.root, cat_path, committed)
        r["catalog"] = cat_path
        r["committed_results"] = res_path
        r["runs_meta_sha256"] = {n: i["meta_sha256"] for n, i in committed["runs"].items()}
        res["configs"][key] = r
        print(f"== {key}（被测 {r['test']}）")
        for i, s in r["summary"].items():
            print(
                f"  {i:<26} r rcc-a {s['r_worst_rcc_a']:.3f}  范围 {s['r_worst_min']:.3f}–{s['r_worst_max']:.3f}"
                f"（最差基线 {s['r_worst_max_baseline']}，最大挪动 {s['r_worst_max_shift_vs_rcc_a']:.3f}）"
                f"  signed {s['signed_mean_min']:+.3e}…{s['signed_mean_max']:+.3e}  pass {s['n_pass']}/10"
            )
    if args.json:
        args.json.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
        print(f"→ {display_path(args.json)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
