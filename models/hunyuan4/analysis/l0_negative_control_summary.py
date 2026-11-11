"""把 l0_negative_control.sh 四步的产物合成一份结果 json（L0 负对照：不去交错的检查点必须判不一致）。

过 = 三道核对（往返逐位、RoPE 去交错、对 GPU 初始参数清单）都判不一致，且不一致的张量正好是带 RoPE 列的那些
（多了说明核对在别处也报错、负对照不干净；少了说明有带 RoPE 列的张量没被查出来）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True)
    p.add_argument("--hf", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    w = Path(a.work)
    hf_sha = json.loads((Path(a.hf) / "tensor_sha256.json").read_text())
    neg = json.loads((w / "runs" / "hy4-mini-maxtext-ckpt-neg" / "params_sha" / "rank0.init.json").read_text())
    rt_differ = sorted(k for k, v in neg.items() if v["sha256"] != hf_sha.get(k))
    rope = json.loads((w / "rope" / "results.json").read_text())["checks"][0]
    rope_rows = rope["details_inline"]["rows"]
    rope_set = sorted({r["hf"] for r in rope_rows})
    rope_failed = sorted({r["hf"] for r in rope_rows if not r["ok"]})
    iv = json.loads((w / "init_vs_gpu.json").read_text())["checks"][1]
    iv_rank0_differ = sorted(r["hf"] for r in iv["details_inline"]["rank0"] if not r["equal"])
    checks = []

    def add(cid, detected, specific, msg, metrics):
        checks.append({"id": cid, "type": "poison_selftest", "level": "L0", "hard": True,
                       "status": "pass" if detected and specific else "fail", "threshold": {"kind": "exact"},
                       "metrics": metrics, "message": msg})

    add("L0.neg.no_deinterleave.roundtrip", bool(rt_differ), rt_differ == rope_set,
        f"往返逐位：{len(rt_differ)} / {len(neg)} 个张量对 HF 原件不同，正好是带 RoPE 列的 {len(rope_set)} 个：{rt_differ == rope_set}",
        {"tensors": len(neg), "differ": len(rt_differ), "rope_tensors": len(rope_set), "differ_eq_rope_set": rt_differ == rope_set})
    add("L0.neg.no_deinterleave.rope", rope["status"] == "fail", rope_failed == rope_set,
        f"RoPE 去交错核对：{len(rope_failed)} / {len(rope_rows)} 条判不一致（检查状态 {rope['status']}）",
        {"rows": len(rope_rows), "failed": len(rope_failed)})
    want_iv = sorted(set(r["hf"] for r in iv["details_inline"]["rank0"]
                         if any(x in r["hf"].split("+")[0] for x in rope_set)))
    add("L0.neg.no_deinterleave.init_vs_gpu", iv["status"] == "fail", iv_rank0_differ == want_iv,
        f"对 GPU 初始参数清单：rank 0 上 {len(iv_rank0_differ)} 个参数不同，都来自带 RoPE 列的 HF 张量：{iv_rank0_differ == want_iv}；"
        f"8 个 rank 共不同 {iv['metrics']['differ']} 项",
        {"rank0_differ": len(iv_rank0_differ), "all_ranks_differ": iv["metrics"]["differ"], "differ_from_rope_tensors_only": iv_rank0_differ == want_iv})
    res = {"schema_version": 1, "stage": "hy4-mini-l0-negative-control", "model": "hunyuan4-mini",
           "description": "同一个 HF 目录不做 RoPE 列去交错（checkpoint_rope_interleave=false）转换，L0 三道核对都必须判不一致，且只在带 RoPE 列的张量上",
           "checks": checks, "rope_tensors": rope_set, "roundtrip_differ": rt_differ}
    res["summary"] = {k: sum(c["status"] == k for c in checks) for k in ("pass", "fail", "inconclusive", "error")}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for c in checks:
        print(c["status"], c["id"], c["message"])


if __name__ == "__main__":
    main()
