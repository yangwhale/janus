"""核 GPU golden 的软件版本与 T1 的 g3-* 是否逐字相同：逐份读 meta.json 的 image 与 versions，
对 g3-a 比字段，并记每份 meta.json 的 sha256（可与已提交结果文件里记的对上）。meta.json 不入库，
只入库这份比对输出。

    python models/hunyuan4/analysis/gpu_meta_versions_check.py \\
        --ref <g3-* 所在根> --group gb3=<根> --group gd3=<根> --group gm=<根> --group rw1k=<根> \\
        --out models/hunyuan4/results/gpu-meta-versions-vs-g3.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

LETTERS = "abcdefgh"


def _load(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _diff(a: dict, b: dict) -> dict:
    keys = sorted(set(a) | set(b))
    return {k: {"ref": a.get(k), "run": b.get(k)} for k in keys if a.get(k) != b.get(k)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", type=Path, required=True, help="g3-a … g3-h 所在的根目录")
    p.add_argument("--group", action="append", default=[], help="名字=根目录，读 <根>/<名字>-a … -h/meta.json")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)

    ref, ref_sha = _load(a.ref / "g3-a" / "meta.json")
    out = {
        "what": "GPU golden 各份 meta.json 的 image、versions 对 T1 g3-a 的逐字段比对（字段 diff 为空即逐字相同）",
        "reference": {
            "run": "g3-a",
            "meta_sha256": ref_sha,
            "image": ref.get("image"),
            "versions": ref.get("versions"),
        },
        "groups": {},
    }
    groups = [("g3", a.ref)] + [(g.split("=", 1)[0], Path(g.split("=", 1)[1])) for g in a.group]
    all_same = True
    for name, root in groups:
        rows = []
        for c in LETTERS:
            run = f"{name}-{c}"
            m, sha = _load(root / run / "meta.json")
            img = m.get("image") == ref.get("image")
            dv = _diff(ref.get("versions") or {}, m.get("versions") or {})
            all_same &= img and not dv
            rows.append({"run": run, "meta_sha256": sha, "image_same": img, "versions_diff": dv})
        out["groups"][name] = rows
    out["all_identical"] = all_same
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(f"all_identical={all_same} → {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
