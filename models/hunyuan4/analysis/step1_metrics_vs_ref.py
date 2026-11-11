"""各运行 metrics.jsonl 第 1 步的主 loss 对一份已提交的 CPU FP64 参照（结果文件里的 reference.lm_loss）：
相对差 (x − FP64) ÷ FP64，另记 GPU 各份之间的极差。参照已经在结果文件里，这里不重算参照。

    python models/hunyuan4/analysis/step1_metrics_vs_ref.py --ref <结果文件.json> \\
        --gpu gm-a=<运行目录> … --tpu tm-b=<运行目录> … --out <输出.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _step1(run: Path) -> tuple[dict, str]:
    raw = (run / "metrics.jsonl").read_bytes()
    for line in raw.decode().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("step") == 1:
                return r, hashlib.sha256((run / "meta.json").read_bytes()).hexdigest()
    raise SystemExit(f"{run} 没有第 1 步")


def _runs(specs: list[str], fp64: float) -> dict:
    out = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        r, meta_sha = _step1(Path(path))
        out[name] = {
            "lm_loss": r["lm_loss"],
            "loss": r.get("loss"),
            "rel_vs_fp64": (r["lm_loss"] - fp64) / fp64,
            "meta_sha256": meta_sha,
        }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--gpu", nargs="+", required=True, help="名字=运行目录")
    p.add_argument("--tpu", nargs="+", required=True, help="名字=运行目录")
    p.add_argument("--note", default="")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)

    fp64 = json.loads(a.ref.read_text())["reference"]["lm_loss"]
    gpu = _runs(a.gpu, fp64)
    tpu = _runs(a.tpu, fp64)
    g = [v["lm_loss"] for v in gpu.values()]
    out = {
        "what": "metrics.jsonl 第 1 步主 loss 对 CPU FP64 参照的相对差 (x − FP64) ÷ FP64",
        "reference": {"file": a.ref.as_posix(), "lm_loss": fp64},
        "gpu": gpu,
        "gpu_spread_rel": (max(g) - min(g)) / fp64,
        "tpu": tpu,
        "note": a.note,
    }
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: round(v["rel_vs_fp64"], 10) for k, v in {**gpu, **tpu}.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
