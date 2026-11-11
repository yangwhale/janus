"""cpu_smoke.py 的第 4 步：用 janus 自己的目录与检查读 CPU 冒烟的两份运行目录。单独可跑：

  python smoke_checks.py /tmp/hy3-cpu-smoke --name-map /tmp/hy3-cpu-smoke/name_map.small.yaml

把假 golden 补成合规的运行目录、把 HF 权重按 name_map 拼成 Megatron 布局的参数清单（另一个参照运行 hf-ref），
写一份小目录 catalog.yaml，然后 `python -m janus run`。最后一行打印 OK / FAIL。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

from janus.dump.name_map import NameMap  # noqa: E402
from janus.store import read_safetensors_tensor  # noqa: E402

META_STUB = {"schema_version": 1, "framework": "megatron", "model": "hunyuan3-4l", "code": {}, "image": "n/a",
             "versions": {}, "hardware": {}, "parallelism": {}, "config": {}, "checkpoint": {}, "data": {},
             "steps": {"first": 1, "last": 1, "convention": "golden_step = step"}, "determinism": {}, "dtypes": {},
             "created_at": "1970-01-01T00:00:00+00:00"}


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("work")
    p.add_argument("--name-map", required=True)
    a = p.parse_args(argv)
    w = Path(a.work)
    runs = w / "runs"
    nm = NameMap.load(a.name_map)

    # 假 golden → 合规运行目录（只有输入指纹与空的逐步标量）
    g = runs / "fake-golden"
    if not g.exists():
        shutil.copytree(w / "golden" / "data", g / "data", ignore=shutil.ignore_patterns("batches"))
        (g / "meta.json").write_text(json.dumps({**META_STUB, "run_name": "fake-golden", "side": "golden-fake"}))
        (g / "metrics.jsonl").write_text("")
        (g / "DONE").write_text("")

    # HF 权重按 name_map 拼成 Megatron 布局的参数清单（GPU EP8 的持有关系）
    h = runs / "hf-ref"
    if not h.exists():
        hf = w / "hf"
        wm = json.loads((hf / "model.safetensors.index.json").read_text())["weight_map"]
        man = {r: {} for r in range(nm.ep)}
        for ent in nm.entries:
            arr = nm.to_megatron(ent, [read_safetensors_tensor(hf / wm[x], x)[0] for x in ent.hf])
            name = ("buffer." if ent.buffer else "") + nm.megatron_name(ent)
            rec = {"sha256": sha(arr), "shape": list(arr.shape), "dtype": str(arr.dtype)}
            for r in man:
                if nm.expert_owner(ent) in (None, r):
                    man[r][name] = rec
        (h / "params_sha").mkdir(parents=True)
        for r, d in man.items():
            (h / "params_sha" / f"rank{r}.init.json").write_text(json.dumps(d))
        (h / "meta.json").write_text(json.dumps({**META_STUB, "run_name": "hf-ref", "side": "golden-hf"}))
        (h / "metrics.jsonl").write_text("")
        (h / "DONE").write_text("")

    steps = json.loads((runs / "t-hooks" / "meta.json").read_text())["steps"]["last"]
    cat = f"""schema_version: 1
model: hunyuan3-4l
stage: tpu-cpu-smoke
runs:
  test: t-hooks
  golden: [fake-golden]
  root: {runs}
checks:
  - id: L0.data.fingerprint
    type: data_fingerprint
    level: L0
    select: {{kind: tokens, steps: "1:{steps + 1}"}}
  - id: hooks.loss
    type: determinism
    level: L0
    compare: lkg
    select: {{kind: metrics, fields: [lm_loss, grad_norm, grad_norm_clipped, param_norm, router.bias_mean.1, router.bias_std.3], step: 1, name: "buffer*"}}
  - id: hooks.final_weights
    type: checkpoint_equal
    level: L0
    compare: lkg
    select: {{kind: checkpoint, which: final, manifest: true}}
  - id: L1.optim.adamw.replay
    type: optimizer_step
    level: L1
    select: {{kind: tensor, step: 1, name: "*"}}
    metrics: [rel_l2, max_abs]
    threshold: {{rel_l2: 5.0e-3}}
    source: "沿用 models/hunyuan3/alignments/00-gpu-noise-and-drills/catalogs/phase1-full.yaml 的 L1.optim.adamw.replay：FP64 重放的更新量舍入到 FP32 后自身误差最大 1.79e-3"
"""
    (w / "catalog.yaml").write_text(cat.replace("runs:\n", "runs:\n  lkg: t-nohooks\n"))
    (w / "catalog-init.yaml").write_text(f"""schema_version: 1
model: hunyuan3-4l
stage: tpu-cpu-smoke-init
runs:
  test: t-hooks
  golden: [hf-ref]
  root: {runs}
checks:
  - id: L0.weights.init
    type: checkpoint_equal
    level: L0
    select: {{kind: checkpoint, which: init, manifest: true}}
""")
    ok = True
    for c, out in ((w / "catalog.yaml", w / "results"), (w / "catalog-init.yaml", w / "results-init")):
        r = subprocess.run([sys.executable, "-m", "janus", "run", str(c), "--out", str(out), "--verbose"],
                           cwd=REPO, text=True, capture_output=True)
        print(r.stdout[-4000:], r.stderr[-2000:])
        res = json.loads((out / "results.json").read_text())
        want = ["L0.weights.init"] if "init" in out.name else ["L0.data.fingerprint", "hooks.loss", "hooks.final_weights",
                                                             "L1.optim.adamw.replay"]
        st = {x["id"]: x["status"] for x in res["checks"]}
        bad = {k: st.get(k) for k in want if st.get(k) != "pass"}
        if bad:
            print("未通过：", bad)
            ok = False
    # 权重衰减的作用范围：逐参数分别按 wd=0 与 wd=0.033 做 FP64 重放，看哪一个更贴近 TPU 实际的更新量；
    # 应当是一维参数（norm 权重）贴 0、其余贴 0.033（与 Megatron 的默认规则一致）
    from janus.replay import adamw_step, clip_coeff
    from janus.store import Run
    t = Run.open(runs / "t-hooks")
    lr, gn = t.series("lr")[1], t.series("grad_norm")[1]
    coeff = clip_coeff(gn, 1.0)
    wrong = []
    for n in t.names(1, "main_param.*"):
        pn = n[len("main_param."):]
        p0, g1 = t.tensor(1, n), t.tensor(1, "grad." + pn)
        real = t.tensor(1, "main_param_after." + pn).astype(np.float64) - p0
        err = {}
        for wd in (0.0, 0.033):
            new, _, _ = adamw_step(p0, g1, lr=lr, beta1=0.9, beta2=0.95, eps=1e-8, wd=wd, clip_coeff=coeff)
            err[wd] = float(np.linalg.norm(real - (new - p0)))
        best = min(err, key=err.get)
        want = 0.0 if p0.ndim == 1 else 0.033
        if best != want:
            wrong.append((pn, err))
    print(f"权重衰减作用范围：{len(t.names(1, 'main_param.*'))} 个参数里与 Megatron 规则不符 {len(wrong)} 个 {wrong[:3]}")
    ok &= not wrong

    # 激活与梯度的名字、形状和 golden 的约定一致（第 1 步）
    idx = json.loads((runs / "t-hooks" / "step_1" / "index.json").read_text())["tensors"]
    names = {t["name"]: t for t in idx}
    d = nm.dims
    expect = {"embed.out": [128, 1, d["hidden"]], "layers.0.attn.qkv.out": [128, 1, (d["heads"] + 2 * d["kv_heads"]) * d["head_dim"]],
              "layers.0.mlp.fc1.out": [128, 1, 2 * d["ffn"]], "layers.1.mlp.router.topk_idx": [128, d["topk"]],
              "layers.1.mlp.router.logits": [128, 1, d["experts"]], "logits": [128, 1, d["vocab"]], "loss.per_token": [1, 128],
              "grad.decoder.layers.0.self_attention.linear_qkv.weight": [(d["heads"] + 2 * d["kv_heads"]) * d["head_dim"], d["hidden"]],
              "exp_avg.decoder.layers.1.mlp.experts.linear_fc1.weight0": [2 * d["moe_ffn"], d["hidden"]]}
    for n, shp in expect.items():
        if n not in names or names[n]["shape"] != shp:
            print("名字或形状不对：", n, names.get(n, {}).get("shape"), "期望", shp)
            ok = False
    print(f"第 1 步写了 {len(names)} 个张量")
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
