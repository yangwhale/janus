"""CPU 冒烟：不占 TPU，把 TPU 侧整条链路在 8 个假 CPU 设备上跑一遍（缩小维度、结构不变）。

  python cpu_smoke.py --work /tmp/hy3-cpu-smoke [--steps 3]

做的事：
1. 按 name_map 造一个小维度的假 HF 目录（随机 BF16 权重、FP32 expert_bias，名字与真模型一致）和一个假 golden
   （8 rank × 2 microbatch × N 步的 data/batches 与 data/rank*.jsonl，格式同 GPU 抓数）；
2. convert_hf_to_maxtext.py 转成 MaxText 检查点，verify_ckpt.py maxtext 反向逐张量对 sha256；
   再用 janus 的正式检查（models/hunyuan3/alignments/01-v5p-acceptance/catalogs/phase2-ckpt-convert.yaml）比一遍，并喂一份故意改坏的检查点
   （一个专家权重翻一位、expert_bias 存成 BF16），确认 checkpoint_equal 与 dtype_manifest 都报出来；
3. train_hy3.py --cpu-small 跑 N 步，第 1 步全量抓数（含加钩子 / 不加钩子逐位自检）；再跑一次 --no-hooks 当对照；
4. 用 janus 读两份运行目录：数据指纹对假 golden、两次运行逐步 loss 逐位相同（钩子不改数值）、
   第 1 步 FP64 重放一步 AdamW（optimizer_step，核对权重衰减作用范围、裁剪、eps）、参数清单对 HF 权重拼出的 Megatron 布局。
最后一行打印 OK / FAIL。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
MODEL = HERE.parents[1] / "model"   # 跨平台共用：name_map、转换、逐张量核对
sys.path.insert(0, str(REPO))

from janus.dump.name_map import NameMap  # noqa: E402
from janus.store import write_safetensors  # noqa: E402

# 与 train_hy3.CPU_SMALL 一致
SMALL = dict(hidden=256, ffn=512, moe_ffn=128, experts=16, topk=4, heads=8, kv_heads=2, head_dim=32, vocab=1024)
SEQ = 128


def small_map() -> NameMap:
    nm = NameMap.load(MODEL / "name_map.yaml")
    return NameMap({**nm.spec, "dims": {**nm.dims, **SMALL},
                    "megatron": {**nm.spec["megatron"], "local_experts": SMALL["experts"] // nm.ep}})


def make_hf(d: Path, seed=0):
    import ml_dtypes
    rng = np.random.default_rng(seed)
    nm = small_map()
    tensors, manifest = {}, {}
    for ent in nm.entries:
        for h, shp in zip(ent.hf, nm.hf_shapes(ent)):
            if h.endswith("expert_bias"):
                a = (rng.standard_normal(shp) * 1e-2).astype(np.float32)
            elif "norm" in h:
                a = (1 + 0.1 * rng.standard_normal(shp)).astype(ml_dtypes.bfloat16)
            else:
                a = (rng.standard_normal(shp) * 0.05).astype(ml_dtypes.bfloat16)
            tensors[h] = a
            manifest[h] = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    d.mkdir(parents=True, exist_ok=True)
    write_safetensors(d / "model-00001.safetensors", tensors)
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model-00001.safetensors" for k in tensors}}))
    (d / "tensor_sha256.json").write_text(json.dumps(manifest, indent=1))


def make_golden(d: Path, steps: int, ranks=8, micro=2, seed=1):
    rng = np.random.default_rng(seed)
    (d / "data" / "batches").mkdir(parents=True, exist_ok=True)
    for r in range(ranks):
        off = 0
        with open(d / "data" / "batches" / f"rank{r}.bin", "wb") as fb, \
                open(d / "data" / "batches" / f"rank{r}.jsonl", "w") as fi, open(d / "data" / f"rank{r}.jsonl", "w") as ff:
            for s in range(1, steps + 1):
                for m in range(micro):
                    sample = rng.integers(0, SMALL["vocab"], SEQ + 1)
                    fields = {"tokens": sample[:-1].reshape(1, SEQ).astype(np.int64),
                              "labels": sample[1:].reshape(1, SEQ).astype(np.int64),
                              "loss_mask": np.ones((1, SEQ), np.float32),
                              "position_ids": np.arange(SEQ, dtype=np.int64).reshape(1, SEQ)}
                    rec, fp = {"step": s, "micro": m, "fields": {}}, {"step": s, "micro": m}
                    for k, a in fields.items():
                        raw = a.tobytes()
                        h = hashlib.sha256(raw).hexdigest()
                        rec["fields"][k] = {"offset": off, "dtype": str(a.dtype), "shape": list(a.shape), "sha256": h}
                        fp[k] = h
                        fb.write(raw)
                        off += len(raw)
                    fp["nonpad"] = int(fields["loss_mask"].sum())
                    fi.write(json.dumps(rec) + "\n")
                    ff.write(json.dumps(fp) + "\n")
    (d / "meta.json").write_text(json.dumps({"run_name": "fake-golden", "data": {"manifest": "fake", "manifest_sha256": "0"}}))


def poison_checkpoint(src_items: Path, dst: Path):
    """故意改坏的检查点：MoE 第 2 层（栈内 1）专家 3 的 wo 翻一位，三层 expert_bias 都存成 BF16。"""
    import ml_dtypes
    import orbax.checkpoint as ocp
    from maxtext.checkpoint_conversion.utils.utils import save_weights_to_checkpoint
    tree = ocp.PyTreeCheckpointer().restore(str(src_items))
    root = tree.get("params", tree)
    wo = np.array(root["params"]["decoder"]["moe_layers"]["Hunyuan3MoeBlock_0"]["MoeBlock_0"]["wo"], np.float32)
    wo.view(np.uint32)[3, 1, 0, 0] ^= 1 << 16
    root["params"]["decoder"]["moe_layers"]["Hunyuan3MoeBlock_0"]["MoeBlock_0"]["wo"] = wo
    g = root["MoEBiasVar"]["decoder"]["moe_layers"]["Hunyuan3MoeBlock_0"]["MoeBlock_0"]["gate"]
    g["bias"] = np.asarray(g["bias"], np.float32).astype(ml_dtypes.bfloat16)
    save_weights_to_checkpoint(str(dst), root, device_count=1, use_ocdbt=True, use_zarr3=True)


def convert_selftest(w: Path, hf: Path, ckpt: Path, nm_small: Path, env: dict):
    """L0 权重转换的正式检查 + 故意改坏的自证。"""
    py, runs = sys.executable, w / "runs-convert"
    runs.mkdir()
    run([py, MODEL / "verify_ckpt.py", "--name-map", nm_small, "hf-manifest", "--hf", hf, "--janus-run", runs / "hy3-4l-hf"])
    run([py, MODEL / "verify_ckpt.py", "--name-map", nm_small, "maxtext", "--ckpt", ckpt / "0" / "items", "--hf", hf,
         "--janus-run", runs / "hy3-4l-maxtext-ckpt"], env)
    poison_checkpoint(ckpt / "0" / "items", w / "ckpt-poison")
    r = subprocess.run([py, MODEL / "verify_ckpt.py", "--name-map", nm_small, "maxtext", "--ckpt", w / "ckpt-poison" / "0" / "items",
                        "--hf", hf, "--janus-run", runs / "hy3-4l-maxtext-ckpt-poison"], env={**os.environ, **env},
                       text=True, capture_output=True)
    if r.returncode != 1:
        raise SystemExit(f"改坏的检查点 verify_ckpt 应当返回 1，实际 {r.returncode}")
    cat = HERE.parents[1] / "alignments" / "01-v5p-acceptance" / "catalogs" / "phase2-ckpt-convert.yaml"
    out = {}
    for test in ("hy3-4l-maxtext-ckpt", "hy3-4l-maxtext-ckpt-poison"):
        subprocess.run([py, "-m", "janus", "run", cat, "--root", runs, "--test", test, "--out", w / f"results-{test}"], cwd=REPO,
                       text=True, capture_output=True)
        res = json.loads((w / f"results-{test}" / "results.json").read_text())
        out[test] = {c["id"]: (c["status"], c["metrics"]) for c in res["checks"]}
        print(f"[janus] 转换检查 {test}：" + "；".join(f"{k} {v[0]} {v[1]}" for k, v in out[test].items()), flush=True)
    good, bad = out["hy3-4l-maxtext-ckpt"], out["hy3-4l-maxtext-ckpt-poison"]
    ok = all(v[0] == "pass" for v in good.values())
    ok &= bad["L0.ckpt.convert.bitwise"][0] == "fail" and bad["L0.ckpt.convert.bitwise"][1]["differ"] == 4
    ok &= bad["L0.ckpt.convert.dtype"][0] == "fail" and bad["L0.ckpt.convert.dtype"][1]["wrong_dtype"] == 3
    if not ok:
        raise SystemExit("转换检查的自证不通过")
    print("[janus] 转换检查自证：好检查点全过；改坏的检查点 4 个张量 sha256 不同（1 个专家权重 + 3 个 expert_bias）、3 个 dtype 不符，都报出来了", flush=True)


def run(cmd, env=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    r = subprocess.run([str(c) for c in cmd], env={**os.environ, **(env or {})}, text=True, capture_output=True)
    keep = [ln for ln in (r.stdout + r.stderr).splitlines()
            if ln.startswith(("[janus]", "OK", "FAIL", "name_map", "rank", "从 MaxText", "MaxText 变量", "搬运", "已存", "Traceback"))
            or "Error" in ln]
    print("\n".join(keep[-40:]), flush=True)
    if r.returncode:
        print((r.stdout + r.stderr)[-6000:])
        raise SystemExit(f"命令失败（{r.returncode}）")
    return r


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default="/tmp/hy3-cpu-smoke")
    p.add_argument("--steps", type=int, default=3)
    a = p.parse_args(argv)
    w = Path(a.work)
    if w.exists():
        raise SystemExit(f"{w} 已存在，换一个目录")
    hf, golden, ckpt = w / "hf", w / "golden", w / "ckpt"
    make_hf(hf)
    make_golden(golden, a.steps)
    small = ["base_emb_dim=256", "base_mlp_dim=512", "base_moe_mlp_dim=128", "num_experts=16", "num_experts_per_tok=4",
             "base_num_query_heads=8", "base_num_kv_heads=2", "head_dim=32", "vocab_size=1024", f"max_target_length={SEQ}"]
    env = {"JAX_PLATFORMS": "cpu"}
    py = sys.executable
    run([py, MODEL / "convert_hf_to_maxtext.py", "--hf", hf, "--out", ckpt, *small], env)
    nm_small = w / "name_map.small.yaml"
    import yaml
    nm_small.write_text(yaml.safe_dump(small_map().spec, allow_unicode=True))
    run([py, MODEL / "verify_ckpt.py", "--name-map", nm_small, "maxtext", "--ckpt", ckpt / "0" / "items", "--hf", hf], env)
    convert_selftest(w, hf, ckpt, nm_small, env)
    common = ["--load", ckpt / "0" / "items", "--golden", golden, "--hf-manifest", hf / "tensor_sha256.json",
              "--steps", a.steps, "--cpu-small"]
    # hy3-4layers.yml 默认 GA 2，加钩子的训练步只支持 GA 1：冒烟要核钩子与梯度，显式用每设备 2 行、不累积
    ga1 = ["per_device_batch_size=2", "gradient_accumulation_steps=1"]
    run([py, HERE / "train_hy3.py", *common, "--janus-run", w / "runs" / "t-hooks", "--dump-steps", "1", "--full", *ga1], env)
    run([py, HERE / "train_hy3.py", *common, "--janus-run", w / "runs" / "t-nohooks", "--no-hooks", "--dump-steps", "1",
         *ga1], env)
    r = run([py, HERE / "smoke_checks.py", w, "--name-map", nm_small])
    print("OK" if "OK" in r.stdout.splitlines()[-1] else "FAIL")


if __name__ == "__main__":
    main()
