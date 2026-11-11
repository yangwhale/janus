"""TPU 当前对齐配置对 GPU 生产 golden rc 的 loss 偏移：直接由逐步序列算，并和「两段相加」的推导值对照。

量的定义与 01 results/case01-tpu-vs-rc.json 相同：d_t = (TPU_t − mean_rc_t) / mean_rc_t，mean_rc_t 是同一步
rc-a..e 五份 lm_loss 的均值；报第 1–1000 步与第 301–1000 步 d_t 的带符号均值，另报第 1000 步 param_norm 的
绝对差 TPU − mean(rc)。

推导值（对齐报告第 3 节「对 GPU 生产 golden rc」）：TPU 对 rcc-a 的 301–1000 步 loss 带符号均值
（(TPU − rcc-a)/rcc-a 的均值）加上 rcc-a 对 rc 均值的同一量。这里两段也用同一批序列现算，三者并列。

用法（仓库根目录）：
  python models/hunyuan3/analysis/tpu_vs_rc_loss.py --tpu v5p=<metrics.jsonl> v7x=<metrics.jsonl> \
      --rcc-a <rcc-a/metrics.jsonl> [--rc <rc-a..e 的 metrics.jsonl …>] [--json <输出.json>]
--rc 默认读 03 results/series/ 里入库的 rc-a..e。
"""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from janus.catalog import display_path  # noqa: E402

RC_DEFAULT = [
    ROOT / f"models/hunyuan3/alignments/03-gpu-remat-golden/results/series/rc-{c}.metrics.jsonl" for c in "abcde"
]
WINDOWS = {"steps_1_1000": (1, 1000), "steps_301_1000": (301, 1000)}


def series(path: Path, field: str) -> dict[int, float]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if field in r and r[field] is not None:
                out[int(r["step"])] = float(r[field])
    return out


def _meta_sha(metrics_path: Path):
    """运行目录里 metrics.jsonl 旁边的 meta.json 的 sha256（入库的 series/ 文件旁边没有，记 None）。"""
    m = metrics_path.parent / "meta.json"
    return hashlib.sha256(m.read_bytes()).hexdigest() if metrics_path.name == "metrics.jsonl" and m.is_file() else None


def rel_signed_mean(test: dict, ref: dict, lo: int, hi: int) -> float:
    steps = [s for s in range(lo, hi + 1)]
    missing = [s for s in steps if s not in test or s not in ref]
    if missing:
        raise SystemExit(f"第 {missing[:5]} 步缺数据")
    t = np.array([test[s] for s in steps], dtype=np.float64)
    r = np.array([ref[s] for s in steps], dtype=np.float64)
    return float(np.mean((t - r) / np.abs(r)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tpu", nargs="+", required=True, help="<标签>=<metrics.jsonl>")
    ap.add_argument("--rcc-a", required=True, type=Path)
    ap.add_argument("--rc", nargs=5, type=Path, default=RC_DEFAULT)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    rc_loss = [series(p, "lm_loss") for p in args.rc]
    rc_pn = [series(p, "param_norm") for p in args.rc]
    common = sorted(set.intersection(*(set(s) for s in rc_loss)))
    mean_rc = {s: float(np.mean([x[s] for x in rc_loss])) for s in common}
    mean_rc_pn1000 = statistics.mean([x[1000] for x in rc_pn])  # 与 case01-tpu-vs-rc.json 同一种求均值（逐位一致）
    rcca = series(args.rcc_a, "lm_loss")
    rcca_vs_rc = {k: rel_signed_mean(rcca, mean_rc, *w) for k, w in WINDOWS.items()}

    out = {
        "what": (
            "TPU 当前对齐配置对 GPU 生产 golden rc-a..e：loss 逐步相对差 (TPU − mean(rc)) / mean(rc) 的带符号均值"
            "（mean(rc) 是同一步 5 份的均值），第 1000 步 param_norm 绝对差 TPU − mean(rc)（TPU 记的原始 param_norm 列）。"
            "inferred_301_1000 = 对齐报告第 3 节的推导：(TPU − rcc-a)/rcc-a 的 301–1000 步均值 + (rcc-a − mean(rc))/mean(rc) 的"
            "301–1000 步均值，两段都由同一批序列现算；direct_minus_inferred = 直接值 − 推导值"
        ),
        "script": display_path(Path(__file__)),
        "inputs": {
            "rc": [display_path(p) for p in args.rc],
            "rcc_a": args.rcc_a.parent.name,
            "rcc_a_meta_sha256": _meta_sha(args.rcc_a),
        },
        "rcc_a_vs_rc_loss_rel_signed_mean": rcca_vs_rc,
        "tpu": {},
    }
    for item in args.tpu:
        label, p = item.split("=", 1)
        p = Path(p)
        loss = series(p, "lm_loss")
        pn = series(p, "param_norm")
        direct = {k: rel_signed_mean(loss, mean_rc, *w) for k, w in WINDOWS.items()}
        vs_rcca = rel_signed_mean(loss, rcca, 301, 1000)
        inferred = vs_rcca + rcca_vs_rc["steps_301_1000"]
        out["tpu"][label] = {
            "run": p.parent.name if p.name == "metrics.jsonl" else p.name.split(".")[0],
            "meta_sha256": _meta_sha(p),
            "loss_rel_signed_mean": direct,
            "param_norm_diff_1000": pn[1000] - mean_rc_pn1000,
            "vs_rcc_a_loss_rel_signed_mean_301_1000": vs_rcca,
            "inferred_301_1000": inferred,
            "direct_minus_inferred_301_1000": direct["steps_301_1000"] - inferred,
        }
        print(
            f"{label}: 301–1000 直接 {direct['steps_301_1000']:+.6e}  推导 {inferred:+.6e}"
            f"（对 rcc-a {vs_rcca:+.6e} + rcc-a 对 rc {rcca_vs_rc['steps_301_1000']:+.6e}）"
            f"  差 {direct['steps_301_1000'] - inferred:+.2e}；1–1000 {direct['steps_1_1000']:+.6e}；"
            f"第 1000 步 param_norm 差 {pn[1000] - mean_rc_pn1000:+.6e}"
        )
    if args.json:
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
        print(f"→ {display_path(args.json)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
