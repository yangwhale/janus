"""FP32 log 的 ULP（TPU）：现有写法与一个候选的精确写法，在同一批确定的输入上对 FP64 正确舍入比。

输入（log_probe_inputs，GPU 一侧的探针 analysis/gpu_log_probe.py 用同一个生成函数，两边各记输入的 sha256 互相核）：
  all[0.5,4)：[0.5, 4) 里全部 FP32（3 × 2^23 个，穷举，含 1 附近输出最小、ULP 最难的一段）；
  logu[1e-37,1e37]：对数均匀 2^22 个（numpy default_rng(1234)）；
  logu[1,4096)：对数均匀 2^22 个（同一个 rng 接着取）：log-sum-exp 里 log 的输入域（减去行最大值后的指数和 ≥ 1）；
  real.kd_target（可选 --inputs）：GPU golden 第 1 步索引器 KD 的目标分布，模型里 log 的真实输入。
写法：
  default：jnp.log（XLA 默认近似）；highest：lax.log(accuracy=HIGHEST)（MaxText transcendental_accuracy_highest 现在用的）；
  maxtext_freebsd：MaxText accurate_math.log_freebsd（开关 transcendental_log_freebsd，M51；装的 MaxText 有它才量）；
  fdlibm：FreeBSD msun e_logf.c 的 float 算法（x = 2^k · (1 + f)，s = f / (2 + f)，log(1 + f) = f − s·(f − R(s²))，
          k·ln2 拆成高低两段；作者注明误差 < 1 ULP），全部 FP32 运算，除法用一步修正的精确除法（MaxText muon_mcore_compat._div
          的同一写法：TPU 默认 FP32 除法最多 2 ULP）。
结果 JSON：每个写法、每批输入的 ulp_max / ulp_p999 / ulp_mean / 正确舍入比例。

  SCRIPT=log_ulp_probe.py bash run.sh <运行名> 0 [--inputs gs://…/elem-inputs.npz]      # TPU
  JAX_PLATFORMS=cpu python log_ulp_probe.py --out /tmp/x.json --cpu                       # CPU 自检（CPU 的 log 本来就准）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from log_probe_inputs import make_inputs  # noqa: E402


def ulp_dist(out: np.ndarray, ref64: np.ndarray) -> np.ndarray:
    """|out − FP64 正确舍入到 FP32| 的 ULP 数（按 FP32 位模式的有序整数距离，跨 0 也对）。"""
    cr = ref64.astype(np.float32)

    def key(a):
        i = a.view(np.int32).astype(np.int64)
        return np.where(i < 0, -(i & 0x7FFFFFFF), i)
    return np.abs(key(np.ascontiguousarray(out, np.float32)) - key(cr))


def stats(out, x):
    ref = np.log(x.astype(np.float64))
    d = ulp_dist(out, ref)
    return {"n": int(d.size), "ulp_max": int(d.max()), "ulp_p999": float(np.quantile(d, 0.999)), "ulp_mean": float(d.mean()),
            "cr_match_frac": float((d == 0).mean()), "n_gt1": int((d > 1).sum())}


def fdlibm_logf(x):
    """FreeBSD msun e_logf.c（__ieee754_logf）的 float 算法移植；x > 0 的正规数与次正规数，0 / 负数 / inf / nan 交给 jnp.log。"""
    import jax.numpy as jnp
    from jax import lax

    ln2_hi, ln2_lo, two25 = np.float32(6.9313812256e-01), np.float32(9.0580006145e-06), np.float32(3.355443200e07)
    Lg1, Lg2, Lg3, Lg4 = (np.float32(v) for v in (6.6666668653e-01, 4.0000000596e-01, 2.8571429849e-01, 2.2222198546e-01))
    Lg5, Lg6, Lg7 = (np.float32(v) for v in (1.8183572590e-01, 1.5313838422e-01, 1.4798198640e-01))
    f32 = jnp.float32

    def recip(v):
        r = 1.0 / v
        return r * (2.0 - v * r)

    def div(a, b):
        r = recip(b)
        q = a * r
        return q + r * (a - q * b)

    x = x.astype(f32)
    ix = lax.bitcast_convert_type(x, jnp.int32)
    sub = ix < 0x00800000
    xs = jnp.where(sub, x * two25, x)
    k = jnp.where(sub, -25, 0)
    ix = lax.bitcast_convert_type(xs, jnp.int32)
    k = k + ((ix >> 23) - 127)
    ix = ix & 0x007FFFFF
    i = (ix + (0x95F64 << 3)) & 0x800000
    xn = lax.bitcast_convert_type(ix | (i ^ 0x3F800000), f32)
    k = k + (i >> 23)
    f = xn - f32(1.0)
    dk = k.astype(f32)
    # |f| < 2^-20
    tiny = (0x007FFFFF & (15 + ix)) < 16
    Rt = f * f * (f32(0.5) - f32(0.33333333333333333) * f)
    tiny_v = jnp.where(f == 0, dk * ln2_hi + dk * ln2_lo,
                       jnp.where(k == 0, f - Rt, dk * ln2_hi - ((Rt - dk * ln2_lo) - f)))
    tiny_v = jnp.where((f == 0) & (k == 0), f32(0.0), tiny_v)
    s = div(f, f32(2.0) + f)
    z = s * s
    i2 = ix - (0x6147A << 3)
    w = z * z
    j = (0x6B851 << 3) - ix
    t1 = w * (Lg2 + w * (Lg4 + w * Lg6))
    t2 = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7)))
    i2 = i2 | j
    R = t2 + t1
    hfsq = f32(0.5) * f * f
    a1 = jnp.where(k == 0, f - (hfsq - s * (hfsq + R)), dk * ln2_hi - ((hfsq - (s * (hfsq + R) + dk * ln2_lo)) - f))
    a2 = jnp.where(k == 0, f - s * (f - R), dk * ln2_hi - ((s * (f - R) - dk * ln2_lo) - f))
    main = jnp.where(i2 > 0, a1, a2)
    v = jnp.where(tiny, tiny_v, main)
    ok = (x > 0) & jnp.isfinite(x)
    return jnp.where(ok, v, jnp.log(x))


def fetch(path):
    if not path or not path.startswith("gs://"):
        return path
    loc = Path(tempfile.mkdtemp()) / Path(path).name
    subprocess.run(["gcloud", "storage", "cp", "-q", path, str(loc)], check=True)
    return str(loc)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--inputs", default=None, help="elem-inputs.npz（取 kd_target）")
    p.add_argument("--cpu", action="store_true")
    a, _ = p.parse_known_args()
    import jax
    import jax.numpy as jnp
    from jax import lax
    if not a.cpu:
        jax.distributed.initialize()
    dev = jax.local_devices()[0]
    X = make_inputs()
    if a.inputs:
        X["real.kd_target"] = np.load(fetch(a.inputs))["kd_target"].astype(np.float32)
    impl = {"default": jnp.log, "highest": lambda v: lax.log(v, accuracy=lax.AccuracyMode.HIGHEST), "fdlibm": fdlibm_logf}
    try:  # MaxText 开关 transcendental_log_freebsd（M51）实际用的实现
        from maxtext.utils import accurate_math as am
        if hasattr(am, "log_freebsd"):
            impl["maxtext_freebsd"] = am.log_freebsd
    except ImportError:
        pass
    res = {"what": "FP32 log 的 ULP（对 FP64 正确舍入）", "device_kind": dev.device_kind, "jax": jax.__version__,
           "inputs_sha256": {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in X.items()},
           "results": {}}
    for name, f in impl.items():
        fj = jax.jit(f)
        res["results"][name] = {}
        for k, x in X.items():
            out = np.asarray(fj(jax.device_put(x, dev)))
            res["results"][name][k] = stats(out, x)
            print(name, k, res["results"][name][k], flush=True)
    if jax.process_index() == 0:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
        (Path(a.out).parent / "DONE").write_text("ok\n")


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "tpu,cpu")
    main()
