"""CPU 冒烟：不占 TPU，把混元 4 TPU 侧整条链路在 8 个假 CPU 设备上跑一遍（hy4-tiny 维度、结构不变）。

  JAX_PLATFORMS=cpu python cpu_smoke.py --work /tmp/hy4-cpu-smoke [--stages A,B,C,D] [--steps 3]

做的事（每个阶段一遍）：
1. make_tiny_hf.py 造小维度的「截 4 层 + MTP」HF 目录（随机权重，名字与 dtype 分布同发布的检查点），
   make_fake_golden.py 造合成 golden（8 rank × 2 microbatch × N 步，和混元 4 的 GPU golden 一样只存 tokens / labels）；
2. convert_hf_to_maxtext.py --stage 转换；verify_ckpt.py maxtext 反向逐张量对 sha256、verify_ckpt.py rope 核去交错；
   负对照：不去交错转换的检查点，按去交错的读法核，必须 FAIL（PROTOCOL.md 第 5 节「未打补丁的读法必须被判不一致」）；
3. train_hy4.py --cpu-small 跑 N 步、第 1 步全量抓数；再以「indexer 不冻、权重衰减 0」跑一次：indexer 的 FP32 主权重
   逐位不变且第 1 步 Adam 动量全 0 ⇔ indexer 梯度恒为 0（只冻结时梯度范数不受 indexer 影响的证据）；
4. 读运行目录：数据指纹对合成 golden、参数清单 init 对 HF 原件 sha256（HF 名字换算对）、indexer 参数 init = final、学习率恒定。
结果写 <work>/smoke.json；最后一行打印 OK / FAIL。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[1] / "model"
REPO = HERE.parents[3]
TINY = ["model_name=hy4-tiny", "checkpoint_rope_interleave=true"]


def run(cmd, log):
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=f"{REPO}:{os.environ.get('PYTHONPATH', '')}")
    with open(log, "w") as f:
        r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=env)
    tail = Path(log).read_text().strip().splitlines()[-1:] or [""]
    return r.returncode, tail[0]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default="/tmp/hy4-cpu-smoke")
    p.add_argument("--stages", default="A,B,C,D")
    p.add_argument("--steps", type=int, default=3)
    a = p.parse_args(argv)
    w = Path(a.work)
    w.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    hf, golden = w / "hf", w / "golden"
    res = {"checks": []}

    def check(name, ok, detail=""):
        res["checks"].append({"id": name, "ok": bool(ok), "detail": detail})
        print(f"{'✓' if ok else '✗'} {name} {detail}", flush=True)

    rc, t = run([py, MODEL / "make_tiny_hf.py", "--out", hf], w / "make_hf.log")
    check("make_tiny_hf", rc == 0, t)
    rc, t = run([py, HERE / "make_fake_golden.py", "--out", golden, "--steps", a.steps, "--seq", 128, "--vocab", 1024,
                "--fields", "tokens,labels"],
                w / "make_golden.log")
    check("make_fake_golden", rc == 0, t)
    rc, t = run([py, MODEL / "verify_ckpt.py", "names", "--hf", hf], w / "names.log")
    check("names_shapes", rc == 0, t)
    # HF 原件按 GPU golden 清单的写法（名字、FP32、rank 0 的 FSDP2 分片）算一遍，对 TPU 运行的初始参数清单
    sys.path.insert(0, str(MODEL))
    sys.path.insert(0, str(REPO))
    import hashlib

    from hy4_map import gpu_name, gpu_rank_pieces

    from janus.store import read_safetensors_tensor as _rt
    wm = json.loads((hf / "model.safetensors.index.json").read_text())["weight_map"]
    hf32 = {}
    for n, f in wm.items():
        x = np.ascontiguousarray(_rt(hf / f, n)[0].astype(np.float32))
        hf32[gpu_name(n)] = hashlib.sha256(np.ascontiguousarray(gpu_rank_pieces(gpu_name(n), x)[0]).tobytes()).hexdigest()

    for st in a.stages.split(","):
        ck = w / f"ckpt-{st}"
        rc, t = run([py, MODEL / "convert_hf_to_maxtext.py", "--hf", hf, "--stage", st, "--out", ck, *TINY], w / f"convert-{st}.log")
        check(f"{st}.convert", rc == 0, t)
        rc, t = run([py, MODEL / "verify_ckpt.py", "maxtext", "--ckpt", ck / "0" / "items", "--hf", hf, "--stage", st, *TINY],
                    w / f"verify-{st}.log")
        check(f"{st}.roundtrip_bitwise", rc == 0, t)
        rc, t = run([py, MODEL / "verify_ckpt.py", "rope", "--ckpt", ck / "0" / "items", "--hf", hf, "--stage", st, *TINY],
                    w / f"rope-{st}.log")
        check(f"{st}.rope_deinterleave", rc == 0, t)
        if st == "A":
            neg = w / "ckpt-A-no-deinterleave"
            run([py, MODEL / "convert_hf_to_maxtext.py", "--hf", hf, "--stage", st, "--out", neg, "model_name=hy4-tiny",
                 "checkpoint_rope_interleave=false"], w / "convert-neg.log")
            rc, t = run([py, MODEL / "verify_ckpt.py", "maxtext", "--ckpt", neg / "0" / "items", "--hf", hf, "--stage", st, *TINY],
                        w / "verify-neg.log")
            check("A.negative_control_unpatched_rope_fails", rc != 0, t)
        rd = w / f"run-{st}"
        rc, t = run([py, HERE / "train_hy4.py", "--stage", st, "--cpu-small", "--load", ck / "0" / "items", "--golden", golden,
                     "--hf-config", hf / "config.json", "--hf-manifest", hf / "tensor_sha256.json", "--steps", a.steps,
                     "--janus-run", rd, "--dump-steps", "1", "--full", "--state-sha"], w / f"train-{st}.log")
        check(f"{st}.train", rc == 0 and (rd / "DONE").exists(), t)
        if rc:
            continue
        init = json.loads((rd / "params_sha" / "rank0.init.json").read_text())
        final = json.loads((rd / "params_sha" / "rank0.final.json").read_text())
        bad = [k for k in init if init[k]["sha256"] != hf32[k]]
        check(f"{st}.params_sha_init_equals_hf", not bad and init,
              f"{len(init)} 个张量（GPU 名字、FP32、rank 0 的分片），不同 {bad[:3]}")
        idx = [k for k in init if ".indexer." in k]
        moved = [k for k in idx if init[k]["sha256"] != final[k]["sha256"]]
        check(f"{st}.indexer_frozen_no_decay", idx and not moved, f"indexer 张量 {len(idx)} 个，变了 {moved[:3]}")
        fp_ok = all((golden / "data" / f"rank{r}.jsonl").read_text() == (rd / "data" / f"rank{r}.jsonl").read_text()
                    for r in range(8))
        check(f"{st}.data_fingerprints", fp_ok)
        m = [json.loads(x) for x in (rd / "metrics.jsonl").read_text().splitlines()]
        check(f"{st}.lr_constant_finite_loss", all(x["lr"] == 1e-5 for x in m) and all(np.isfinite(x["lm_loss"]) for x in m),
              f"lm_loss {[round(x['lm_loss'], 4) for x in m]}")
        if st in ("A", "D"):
            # indexer 不冻、权重衰减 0：indexer 梯度若恒为 0，FP32 主权重逐位不变、第 1 步 Adam 动量全 0
            rg = w / f"run-{st}-indexer-trainable"
            rc, t = run([py, HERE / "train_hy4.py", "--stage", st, "--cpu-small", "--load", ck / "0" / "items",
                         "--golden", golden, "--hf-config", hf / "config.json", "--steps", 2, "--janus-run", rg,
                         "--dump-steps", "1", "--full", "--state-sha", "--no-param-sha", "trainable_parameters_mask=[]",
                         "adam_weight_decay=0.0"], w / f"train-{st}-ig.log")
            if rc:
                check(f"{st}.indexer_grad_zero", False, t)
                continue
            si = json.loads((rg / "state_sha" / "init.json").read_text())
            sf = json.loads((rg / "state_sha" / "final.json").read_text())
            ik = [k for k in si if k.startswith("params/") and "/indexer/" in k]
            same = [k for k in ik if si[k]["sha256"] == sf[k]["sha256"]]
            sys.path.insert(0, str(REPO))
            from janus.store import read_safetensors_header, read_safetensors_tensor
            mx = 0.0
            for f in (rg / "step_1").glob("exp_avg*.safetensors"):
                hdr, _ = read_safetensors_header(f)
                for k in hdr:
                    if ".indexer." in k:
                        mx = max(mx, float(np.abs(read_safetensors_tensor(f, k)[0]).max()))
            g1 = json.loads((rg / "metrics.jsonl").read_text().splitlines()[0])["grad_norm"]
            check(f"{st}.indexer_grad_zero", ik and len(same) == len(ik) and mx == 0.0 and g1 == m[0]["grad_norm"],
                  f"indexer 主权重逐位不变 {len(same)}/{len(ik)}；第 1 步动量 max|m|,|v| = {mx}；"
                  f"第 1 步 grad_norm 冻 / 不冻 {m[0]['grad_norm']!r} / {g1!r}")
    ok = all(c["ok"] for c in res["checks"])
    res["ok"] = ok
    (w / "smoke.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
