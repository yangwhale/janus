"""权重转换往返核对：HF → Megatron → HF 后，逐张量 sha256 对原件，数清张量数，查 dtype 清单。

用法：python ckpt_roundtrip.py <原 HF 目录> <往返后 HF 目录> <输出目录>
产出 <输出目录>/results.json（同 result-schema）与 report.md。逐张量按原始字节算 sha256（BF16 不升精度），不抽样。
注意：往返对上只证明「这两个方向互逆」，证明不了映射表本身对（两边错成同一种的情况查不出）。
"""
import hashlib
import json
import sys
from pathlib import Path

from safetensors import safe_open

from janus.checks import REGISTRY


def manifest(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                t = sf.get_tensor(k).contiguous()
                raw = t.view(-1).view(dtype=__import__("torch").uint8).numpy().tobytes() if t.numel() else b""
                out[k] = {"sha256": hashlib.sha256(raw).hexdigest(), "dtype": str(t.dtype).replace("torch.", ""),
                          "shape": list(t.shape)}
    return out


def main(src, dst, out):
    src, dst, out = Path(src), Path(dst), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ma, mb = manifest(src), manifest(dst)
    checks = []
    r = REGISTRY["checkpoint_equal"]("L0.ckpt.roundtrip.bitwise", "L0", manifests=({"hf": mb}, {"hf": ma}))
    checks.append(r.to_json())
    cov = REGISTRY["coverage"]("L0.ckpt.roundtrip.coverage", "L0", sorted(ma), sorted(mb))
    checks.append(cov.to_json())
    bad_dtype = sorted(k for k in ma if k in mb and ma[k]["dtype"] != mb[k]["dtype"])
    f32 = sorted(k for k, v in ma.items() if v["dtype"] == "float32")
    checks.append({"id": "L0.ckpt.dtype_manifest", "type": "artifact_check", "level": "L0", "hard": True,
                   "status": "pass" if not bad_dtype else "fail",
                   "metrics": {"tensors": len(ma), "dtype_changed": len(bad_dtype), "fp32_in_source": len(f32)},
                   "message": "dtype 与原件一致" if not bad_dtype else f"dtype 变了：{bad_dtype[:5]}",
                   "details_inline": {"fp32_tensors": f32[:50]}})
    summary = {k: sum(1 for c in checks if c["status"] == k) for k in ("pass", "fail", "inconclusive", "error")}
    res = {"schema_version": 1, "stage": "phase1-ckpt-roundtrip", "source": str(src), "roundtrip": str(dst),
           "summary": summary, "checks": checks}
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    lines = ["# 权重转换往返（HF → Megatron → HF）", "", f"原件 `{src}`，往返件 `{dst}`；张量 {len(ma)} 个（原件）/ {len(mb)} 个（往返）。", "",
             "| 检查 | 状态 | 说明 |", "|---|---|---|"]
    lines += [f"| `{c['id']}` | {c['status']} | {c.get('message', '')} |" for c in checks]
    lines += ["", "往返逐位相同只证明两个方向互逆；映射表本身靠逐层比（加载后的权重对原件）。"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
