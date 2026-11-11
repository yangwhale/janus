"""没存 HLO 的运行：按它的 meta.json 在 CPU 上为同一 TPU 拓扑重编训练步，再用 hlo_mof_dus_scan.py 找 #71 的特征（案例 16）。

编译走 TPU 自比离线筛选的同一条路径（alignments/04-tpu-self/hlo_screen.py 的 spec_from_meta + compile_spec：同一镜像、
同一 MaxText 提交、同一 yml（按 sha256 核对）、同一组覆盖参数与 LIBTPU_INIT_ARGS，frameworks/maxtext/aot_hlo.py 编一次，0 张卡）。

「离线编的程序和上卡的是不是同一份」：被扫的运行自己没存 HLO，没法直接比。--calib 给一组存了 HLO 的运行，同样从 meta.json
推规格、在本机重编，把 CPU 编出的程序与运行目录 hlo/ 里 TPU 上实际存下的按 hlo_screen.canon_lines 规范化后逐行比，
说明这条路径在同一镜像、同一 libtpu 构建上能编出上卡的同一份程序。被扫运行与校准运行的规格差别也写进结果（program_delta）。

  python models/hunyuan4/analysis/jit_miscompile_offline_scan.py --root <运行目录根> --runs tz-off-rep,x5-prod \
      --calib ts-g-a --cache <编译缓存> --maxtext-repo <maxtext 克隆> \
      --out models/hunyuan4/results/jit-miscompile-offline-scan.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "alignments" / "04-tpu-self"))
import hlo_mof_dus_scan as scan  # noqa: E402
import hlo_screen as hs  # noqa: E402


def compiled(spec: dict, a) -> dict:
    c = hs.compile_spec(spec, a.cache, a.maxtext_repo, a.runner, a.mem)
    rec = {
        "spec_hash": c["hash"],
        "compile_seconds": c["aot"].get("compile_seconds"),
        "max_rss_gib": c["aot"].get("max_rss_gib"),
        "versions": c["aot"].get("versions"),
        "device_kind": c["aot"].get("device_kind"),
        "devices": c["aot"].get("devices"),
        "mesh": c["aot"].get("mesh"),
        "error": c["aot"].get("error"),
    }
    return rec, (c["hlo"].read_text() if c["hlo"] else None)


def scan_rec(text: str) -> dict:
    r = scan.scan_text(text)
    return {
        "mof_with_dus": r["mof_with_dus"],
        "with_71_signature": r["with_71_signature"],
        "mof_dus_targets": sorted({f"{h['target'].split('{')[0]} {' '.join(h['target_tiles'])}" for h in r["hits"]}),
        "hits": [
            {k: h[k] for k in ("fusion", "dus", "target_tiles", "update_tiles", "n_outputs", "readers_of_target")}
            | {"target": h["target"].split("{")[0], "update": (h["update"] or "").split("{")[0]}
            for h in r["hits"]
        ],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--runs", required=True, help="要扫的运行（逗号分隔），没存 HLO")
    p.add_argument("--calib", default="", help="存了 HLO 的运行（逗号分隔），校准离线路径")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--maxtext-repo", type=Path, required=True)
    p.add_argument("--runner", choices=("docker", "local"), default="docker")
    p.add_argument("--mem", default="250g")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    out = {
        "tool": "models/hunyuan4/analysis/jit_miscompile_offline_scan.py",
        "scanner": "models/hunyuan4/analysis/hlo_mof_dus_scan.py",
        "note": "CPU 上按运行的 meta.json 重编训练步（hlo_screen.compile_spec），扫 #71 特征；calib 是存了 HLO 的运行，"
        "cpu_vs_tpu_program_equal = CPU 重编的程序与 TPU 上存下的规范化后逐行相同",
        "calib": [],
        "runs": [],
    }
    specs = {}
    for r in [x for x in a.calib.split(",") if x]:
        spec = hs.spec_from_meta(a.root / r)
        specs[r] = spec
        rec, text = compiled(spec, a)
        fs = sorted((a.root / r / "hlo").glob("*after_optimizations.txt"))
        tpu = fs[0].read_text() if len(fs) == 1 else None
        rec.update(run=r, spec=spec)
        if text is not None and tpu is not None:
            cl, tl = hs.canon_lines(text), hs.canon_lines(tpu)
            rec.update(
                cpu_vs_tpu_program_equal=cl == tl,
                program_fp={"cpu": hs.program_fp(cl), "tpu": hs.program_fp(tl)},
                scan_cpu=scan_rec(text),
                scan_tpu=scan_rec(tpu),
            )
            for k in ("scan_cpu", "scan_tpu"):
                rec[k].pop("hits")
        out["calib"].append(rec)
        print(f"[offline_scan] calib {r}: cpu_vs_tpu={rec.get('cpu_vs_tpu_program_equal')}", flush=True)
    for r in a.runs.split(","):
        spec = hs.spec_from_meta(a.root / r)
        rec, text = compiled(spec, a)
        rec.update(run=r, spec=spec)
        rec["delta_vs_calib"] = {
            c: hs.program_delta(spec, s) if hs.canon_spec(spec) != hs.canon_spec(s) else {} for c, s in specs.items()
        }
        rec["has_tpu_hlo"] = (a.root / r / "hlo").is_dir()
        if text is not None:
            rec["program_fp_cpu"] = hs.program_fp(hs.canon_lines(text))
            rec["scan"] = scan_rec(text)
        out["runs"].append(rec)
        s = rec.get("scan") or {}
        print(
            f"[offline_scan] {r}: 多输出 fusion 含 DUS {s.get('mof_with_dus')}，#71 特征 {s.get('with_71_signature')}"
            f"{'（编不出来：' + rec['error'][:200] + '）' if rec.get('error') else ''}",
            flush=True,
        )
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
