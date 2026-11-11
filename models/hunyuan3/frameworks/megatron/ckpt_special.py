"""特殊值存 / 恢复（GPU 侧）：存档前往参数里埋 −0.0、次正规数，走一次 Megatron 存档 / 恢复，逐位核对。

两次运行（run_check.sh，确定性、不开分布式优化器，见 docs/dev/design/0004-l1-gpu-kernels.md「五、特殊值存 / 恢复」）：
    run_check.sh g61-sv-save 2 det --no-dist-opt --dump-steps "" --save /data/ckpt-g61-sv --special '<参数名正则>'
    run_check.sh g61-sv-load 3 det --no-dist-opt --dump-steps "" --resume /data/ckpt-g61-sv --special '<同一正则>'
第一次在第 2 步结束、存档之前（框架在 on_train_step_end 回调之后才存档）往命中的 BF16 参数、FP32 主权重、Adam 两个动量、
FP32 buffer 的前 8 个元素写入特殊值，并把这些张量全量写进 special/rank<r>.saved.safetensors；第二次加载后、开训前把同一批张量写进
special/rank<r>.loaded.safetensors。本脚本逐 rank 逐张量比两份文件（不用 GPU）：

    python models/hunyuan3/frameworks/megatron/ckpt_special.py --save-run /data/janus-runs2/g61-sv-save --load-run /data/janus-runs2/g61-sv-load \
        --out /data/janus-g61/ckpt-special
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from janus.checks import REGISTRY, Threshold  # noqa: E402
from janus.store import Run, read_safetensors_header, read_safetensors_tensor  # noqa: E402

OPS = REGISTRY["op_replay"]
# 与 janus/dump/megatron.py 的 JanusDump.SPECIAL_F32 相同：−0.0、+0.0、±1e-40、±2^-149、最大次正规、最小正规
SPECIAL_BITS = np.array([0x80000000, 0x00000000, 0x000116C2, 0x800116C2, 0x00000001, 0x80000001, 0x007FFFFF, 0x00800000],
                        dtype=np.uint32)


def read_safetensors(path: Path) -> dict:
    hdr, _ = read_safetensors_header(path)
    return {n: read_safetensors_tensor(path, n)[0] for n in hdr if n != "__metadata__"}


def expected_head(key: str, dtype) -> np.ndarray:
    v = SPECIAL_BITS.view(np.float32).copy()
    if key.startswith("exp_avg_sq."):
        v = np.abs(v)
    if dtype == np.float32:
        return v
    import torch
    return torch.from_numpy(v).to(torch.bfloat16).view(torch.int16).numpy().view(dtype)    # 与埋值时同一条转换（RNE）


def classify(a: np.ndarray) -> dict:
    a64 = a.astype(np.float64)
    tiny = np.finfo(np.float32).tiny
    return {"neg_zero": int(np.sum((a64 == 0) & np.signbit(a64))), "subnormal": int(np.sum((a64 != 0) & (np.abs(a64) < tiny)))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-run", required=True)
    ap.add_argument("--load-run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from l1_ops import write_outputs
    rs, rl = Run.open(a.save_run), Run.open(a.load_run)
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    full, head, poked = [], [], []
    for f in sorted((rs.path / "special").glob("rank*.saved.safetensors")):
        rank = f.name.split(".")[0]
        saved = read_safetensors(f)
        loaded = read_safetensors(rl.path / "special" / f"{rank}.loaded.safetensors")
        for key in sorted(saved):
            s = saved[key]
            if key not in loaded:
                full.append(dict(name=f"{rank}.{key}", out=np.zeros(1, s.dtype), bitref=np.ones(1, s.dtype), extra={"missing_after_load": True}))
                continue
            ld = loaded[key]
            n = min(len(SPECIAL_BITS), s.size)
            exp = expected_head(key, s.dtype)[:n]
            full.append(dict(name=f"{rank}.{key}", out=ld, bitref=s,
                             extra={"dtype": str(s.dtype), "numel": int(s.size), **{f"saved_{k}": v for k, v in classify(s).items()},
                                    **{f"loaded_{k}": v for k, v in classify(ld).items()}}))
            head.append(dict(name=f"{rank}.{key}[:8]", out=ld.ravel()[:n], bitref=exp,
                             extra={"loaded_bits": [f"0x{int(x):0{8 if s.dtype == np.float32 else 4}x}"
                                                    for x in ld.ravel()[:n].view(np.uint32 if s.dtype == np.float32 else np.uint16)]}))
            poked.append(dict(name=f"{rank}.{key}[:8]", out=s.ravel()[:n], bitref=exp))
    checks = [OPS("CKPT.special.poked", "L0", poked, Threshold("exact")),
              OPS("CKPT.special.full", "L0", full, Threshold("exact")),
              OPS("CKPT.special.values", "L0", head, Threshold("exact"))]
    for c in checks:
        c.evidence = [{"path": str(r.path), "sha256": r.meta_sha256()} for r in (rs, rl)]
        print(f"{c.status:<6} {c.id:<24} {c.message}")
    env = {"save_run_args": rs.meta.get("config", {}).get("args"), "load_run_args": rl.meta.get("config", {}).get("args"),
           "special_bits_fp32": [f"0x{int(x):08x}" for x in SPECIAL_BITS]}
    return write_outputs(Path(a.out), checks, [rs, rl], env, Path(__file__), "phase1-ckpt-special", started,
                         columns=["bit_match_frac", "dtype", "numel", "saved_neg_zero", "saved_subnormal", "loaded_neg_zero",
                                  "loaded_subnormal", "loaded_bits", "missing_after_load"],
                         ref_line="参照：存档前埋值后的同一批张量（逐位）",
                         col_note="列：bit_match_frac 恢复后对存档前（或对期望埋入的位模式）的逐位相同率；*_neg_zero / *_subnormal 是张量里 −0.0 与次正规数的个数。")


if __name__ == "__main__":
    sys.exit(main())
