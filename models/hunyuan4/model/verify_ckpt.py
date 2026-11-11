"""混元 4 的 L0 逐张量核对：HF 原件清单、MaxText 检查点反向取回、RoPE 去交错置换。全量不抽样。

  # 0) HF 目录的张量名、形状对形状表（make_tiny_hf.hf_shapes，训练时换 HF 名字用它），并数参数量
  python verify_ckpt.py names --hf /data/hy4-4l --out <结果目录>

  # 1) 把 HF 原件中属于这个阶段的张量写成 janus 运行目录（tensor_sha256.json 的 sha256 + safetensors 头里的 dtype / 形状），
  #    并逐张量重算原始字节的 sha256 对 tensor_sha256.json（确认下载、截层没坏）
  python verify_ckpt.py hf-manifest --hf /data/hy4-4l --stage A --janus-run <运行根>/hy4-a-hf

  # 2) 读 MaxText 检查点，按分支映射反向取回每个 HF 张量（拆栈、反转置、拼回 pre/post 与 gate/up、RoPE 列逆置换；
  #    FP32 → BF16 只在每个值都是 BF16 精确值时做），逐张量算 sha256，对 tensor_sha256.json
  JAX_PLATFORMS=cpu python verify_ckpt.py maxtext --ckpt <转换输出>/0/items --hf /data/hy4-4l --stage A \
      --janus-run <运行根>/hy4-a-maxtext-ckpt

  # 3) RoPE 去交错单独核：对每个带 RoPE 列的 HF 张量，正向置换后的 rope 段 = 原件的偶数列接奇数列，逆置换逐位回到原件；
  #    再核 MaxText 检查点里的对应叶子与「原件按置换搬过去」逐位相同。结果写 results.json
  JAX_PLATFORMS=cpu python verify_ckpt.py rope --ckpt <转换输出>/0/items --hf /data/hy4-4l --stage A --out <结果目录>

  # 4) 跨侧：MaxText 检查点 → GPU golden 的参数名与 FP32 主权重清单，和 GPU golden 的初始参数清单逐项比
  JAX_PLATFORMS=cpu python verify_ckpt.py gpu-manifest --ckpt <转换输出>/0/items --hf /data/hy4-4l --stage A \
      --janus-run <运行根>/hy4-a-maxtext-gpunames

1) 和 2) 的两个运行目录用 janus 的正式检查比（checkpoint_equal 清单模式 + dtype_manifest）：
  python -m janus run models/hunyuan4/alignments/<实践>/catalogs/l0-ckpt-convert.yaml --root <运行根>

每个子命令最后打印一行 `OK` 或 `FAIL`，退出码 0 / 1。
反向取回对上原件只证明「正反两个方向互逆」；映射本身对不对（两边错成同一种查不出）靠逐层前向比。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))           # 仓库根：import janus
sys.path.insert(0, str(HERE))

from hy4_map import (  # noqa: E402
    STAGES,
    assemble_hf,
    deinterleave_perm,
    hf_dtype,
    hf_kind,
    iter_hf_sources,
    stack_axis,
    stage_tensor_names,
)

from janus.store import read_safetensors_header, read_safetensors_tensor  # noqa: E402

MODEL = "hunyuan4-trunc"


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class HF:
    """HF 目录：按名字读张量（保持原 dtype 的位模式）。"""

    def __init__(self, d):
        self.d = Path(d)
        self.wm = json.loads((self.d / "model.safetensors.index.json").read_text())["weight_map"]
        self.manifest = json.loads((self.d / "tensor_sha256.json").read_text())
        self.config = json.loads((self.d / "config.json").read_text())
        self._hdr = {}

    def header(self, name):
        f = self.wm[name]
        if f not in self._hdr:
            self._hdr[f] = read_safetensors_header(self.d / f)[0]
        return self._hdr[f][name]

    def get(self, name) -> np.ndarray:
        return read_safetensors_tensor(self.d / self.wm[name], name)[0]

    def raw(self, name) -> bytes:
        return read_safetensors_tensor(self.d / self.wm[name], name)[1]


def _norm_st_dtype(st: str) -> str:
    return {"BF16": "bfloat16", "F32": "float32", "F16": "float16", "F64": "float64"}.get(st, st)


def write_manifest_run(path, side: str, framework: str, manifest: dict, extra: dict):
    """参数清单写成合规的 janus 运行目录（docs/reference/dump-format.md）：只有 meta.json、params_sha/rank0.init.json、DONE。"""
    import datetime

    from janus.store import RunWriter
    meta = {"schema_version": 1, "run_name": Path(path).name, "side": side, "framework": framework, "model": MODEL,
            "code": {"repo": "janus", "commit": extra.pop("janus_commit", "unknown")}, "image": "n/a（CPU）",
            "versions": extra.pop("versions", {}), "hardware": {"accelerator": "cpu", "count": 1, "topology": "-"},
            "parallelism": {"dp": 1, "fsdp": 1, "tp": 1, "ep": 1, "cp": 1, "pp": 1},
            "config": {"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "checkpoint": extra.pop("checkpoint"), "data": {"manifest": "n/a", "manifest_sha256": None, "seed": None},
            "steps": {"first": 0, "last": 0, "convention": "golden_step = step"}, "determinism": {}, "dtypes": {},
            "created_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds"),
            **extra}
    w = RunWriter(path, meta)
    (Path(path) / "params_sha").mkdir(exist_ok=True)
    (Path(path) / "params_sha" / "rank0.init.json").write_text(json.dumps(manifest, indent=0, sort_keys=True))
    w.done()
    print(f"已写运行目录 {path}（{len(manifest)} 项）")


def _janus_commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 —— 不在 git 检出里也能跑
        return "unknown"


def cmd_names(a) -> bool:
    """HF 目录的张量名、形状对 make_tiny_hf.hf_shapes，dtype 对 hy4_map.hf_dtype（训练时 hf_view 用这两张表）。"""
    from make_tiny_hf import hf_shapes
    hf = HF(a.hf)
    layers, mtp = STAGES["4L"]
    want = hf_shapes(hf.config, hf.config["num_hidden_layers"], hf.config.get("num_nextn_predict_layers", 0))
    have = set(hf.wm)
    only_f, only_h = sorted(set(want) - have), sorted(have - set(want))
    bad = [(n, hf.header(n)["shape"], list(want[n])) for n in sorted(set(want) & have)
           if list(hf.header(n)["shape"]) != list(want[n])]
    dtypes = {}
    for n in sorted(have):
        dtypes.setdefault(_norm_st_dtype(hf.header(n)["dtype"]), []).append(n)
    bad += [(n, _norm_st_dtype(hf.header(n)["dtype"]), hf_dtype(n)) for n in sorted(set(want) & have)
            if _norm_st_dtype(hf.header(n)["dtype"]) != hf_dtype(n)]
    print(f"形状表 {len(want)} 个，HF 目录 {len(have)} 个；只在形状表 {only_f[:5]}；只在 HF {only_h[:5]}；形状 / dtype 不符 {bad[:5]}")
    print("dtype 分布：" + "；".join(f"{k} {len(v)} 个（例 {v[0]}）" for k, v in sorted(dtypes.items())))
    nparam = sum(int(np.prod(hf.header(n)["shape"])) for n in have)
    print(f"参数量（HF 头）：{nparam:,}")
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        ok = not (only_f or only_h or bad)
        res = {"schema_version": 1, "stage": a.name, "model": MODEL, "hf": str(a.hf),
               "summary": {"pass": int(ok), "fail": int(not ok), "inconclusive": 0, "error": 0},
               "checks": [{"id": "L0.ckpt.names_shapes", "type": "artifact_check", "level": "L0", "hard": True,
                           "status": "pass" if ok else "fail",
                           "metrics": {"tensors_hf": len(have), "tensors_table": len(want), "params_hf_headers": nparam,
                                       "shape_mismatch": len(bad)},
                           "message": "HF 张量名、形状、dtype 逐个等于形状表与 hy4_map.hf_dtype" if ok else "名字、形状或 dtype 对不上",
                           "details_inline": {"only_table": only_f, "only_hf": only_h, "shape_mismatch": bad,
                                              "dtypes": {k: len(v) for k, v in dtypes.items()},
                                              "fp32_tensors": dtypes.get("float32", [])}}]}
        (Path(a.out) / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return not (only_f or only_h or bad)


def cmd_hf_manifest(a) -> bool:
    hf = HF(a.hf)
    layers, mtp = STAGES[a.stage]
    names = stage_tensor_names(hf.wm, layers, mtp)
    man, bad = {}, []
    for name in names:
        hd = hf.header(name)
        h = hashlib.sha256(hf.raw(name)).hexdigest()
        if h != hf.manifest.get(name):
            bad.append(name)
        man[name] = {"sha256": hf.manifest[name], "shape": list(hd["shape"]), "dtype": _norm_st_dtype(hd["dtype"])}
    print(f"阶段 {a.stage}：HF 张量 {len(names)} 个；重算 sha256 与 tensor_sha256.json 不同 {len(bad)} 个 {bad[:5]}")
    msha = hashlib.sha256((Path(a.hf) / "tensor_sha256.json").read_bytes()).hexdigest()
    write_manifest_run(a.janus_run, "golden-hf", "hf", man,
                       {"checkpoint": {"path": str(a.hf), "manifest_sha256": msha}, "stage": a.stage,
                        "janus_commit": _janus_commit()})
    return not bad


def _restore(ckpt: str):
    import orbax.checkpoint as ocp
    tree = ocp.PyTreeCheckpointer().restore(str(ckpt))
    return tree


def _leaf(tree, slash_path: str):
    """检查点树里按 '<集合>/<路径>' 取叶子；兼容外面多包一层 'params' 的存法。"""
    for root in (tree, tree.get("params", {}) if isinstance(tree, dict) else {}):
        cur = root
        try:
            for k in slash_path.split("/"):
                cur = cur[k]
            return np.asarray(cur)
        except (KeyError, TypeError):
            continue
    raise KeyError(slash_path)


def _mt_setup(a):
    sys.path.insert(0, str(HERE.parent / "frameworks" / "maxtext"))
    from convert_hf_to_maxtext import build
    return build(Path(a.hf), a.stage, a.maxtext_args or [], strict_params=not a.maxtext_args)


def roundtrip(a, hf: HF):
    """读检查点，反向取回本阶段每个 HF 张量。返回 ({hf 名: 数组（存的 dtype）}, {hf 名: 记录}, config)。"""
    config, abstract, mapping, _, hf_cfg, slash_of = _mt_setup(a)
    rope = int(hf_cfg["qk_rope_head_dim"])
    dein = bool(getattr(config, "checkpoint_rope_interleave", False))
    tree = _restore(a.ckpt)
    pieces: dict[str, dict] = {}
    where: dict[str, list] = {}
    for mt_key in abstract:
        full = _leaf(tree, slash_of[mt_key])
        for k, i, h in iter_hf_sources({mt_key: mapping[mt_key]}):
            if i is None:
                y = full
            else:
                ax = stack_axis(mt_key, full.ndim, config.param_scan_axis)
                y = np.take(full, i, axis=ax)
            pieces.setdefault(h, {})[k] = y
            where.setdefault(h, []).append({"maxtext": slash_of[mt_key], "layer_slot": i})
    out = {}
    for h, pc in pieces.items():
        out[h] = assemble_hf(pc, tuple(hf.header(h)["shape"]), rope, dein)
    return out, where, config, dein


def cmd_maxtext(a) -> bool:
    import ml_dtypes

    hf = HF(a.hf)
    layers, mtp = STAGES[a.stage]
    want = stage_tensor_names(hf.wm, layers, mtp)
    got, where, config, dein = roundtrip(a, hf)
    differ, man = [], {}
    for name in want:
        if name not in got:
            differ.append(name + "（没取回）")
            continue
        x = got[name]
        stored = str(x.dtype)
        wdt = hf.header(name)["dtype"]
        if wdt == "BF16" and x.dtype == np.float32:
            y = x.astype(ml_dtypes.bfloat16)
            if np.array_equal(y.astype(np.float32), x):
                x = y
            else:
                differ.append(name + "（FP32 值不是 BF16 精确值）")
        h = sha(x)
        man[name] = {"sha256": h, "shape": list(x.shape), "dtype": str(x.dtype), "stored_dtype": stored,
                     "maxtext": where[name]}
        if h != hf.manifest[name]:
            differ.append(name)
    extra = sorted(set(got) - set(want))
    print(f"阶段 {a.stage}（checkpoint_rope_interleave={dein}）：从 MaxText 检查点取回 {len(got)} 个 HF 张量；"
          f"本阶段应有 {len(want)} 个；与 tensor_sha256.json 不同 {len(differ)} 个 {differ[:5]}；多出 {extra[:5]}")
    if a.janus_run:
        import jax
        import orbax.checkpoint as ocp
        msha = hashlib.sha256((Path(a.hf) / "tensor_sha256.json").read_bytes()).hexdigest()
        write_manifest_run(a.janus_run, "tpu", "maxtext", man,
                           {"checkpoint": {"path": str(a.ckpt), "manifest_sha256": msha}, "stage": a.stage,
                            "checkpoint_rope_interleave": dein, "janus_commit": _janus_commit(),
                            "versions": {"jax": jax.__version__, "orbax_checkpoint": getattr(ocp, "__version__", None)}})
    return not differ and not extra


def cmd_gpu_manifest(a) -> bool:
    """MaxText 检查点 → GPU golden 的参数清单写法（hy4_map.gpu_name 的名字、FP32 主权重、按 hy4_map.gpu_rank_pieces 分 rank：
    GPU 每个 rank 记本地 FSDP2 分片的 sha256），写成运行目录 params_sha/rank0..7.init.json。和 GPU golden 的
    params_sha/rank*.init.json 用 checkpoint_equal 逐项比，就是「两边从同一份权重出发」的跨侧 L0 核对。"""
    from hy4_map import gpu_name, gpu_rank_pieces
    hf = HF(a.hf)
    got, where, config, dein = roundtrip(a, hf)
    mans = [dict() for _ in range(8)]
    for name, x in sorted(got.items()):
        if name.endswith("e_score_correction_bias"):
            continue        # 路由偏置在 GPU 上是 buffer，不在 GPU 的参数清单里（GPU modules.txt 只列 nn.Parameter）
        gn = gpu_name(name)
        y = np.ascontiguousarray(x, np.float32)
        for r, piece in enumerate(gpu_rank_pieces(gn, y)):
            mans[r][gn] = {"sha256": sha(np.ascontiguousarray(piece)), "shape": list(y.shape), "dtype": "float32"}
    msha = hashlib.sha256((Path(a.hf) / "tensor_sha256.json").read_bytes()).hexdigest()
    write_manifest_run(a.janus_run, "tpu", "maxtext", mans[0],
                       {"checkpoint": {"path": str(a.ckpt), "manifest_sha256": msha}, "stage": a.stage,
                        "checkpoint_rope_interleave": dein, "janus_commit": _janus_commit(),
                        "names": "GPU golden 的参数名（hy4_map.gpu_name）、FP32 主权重、按 GPU 的 FSDP2 分片分 rank（hy4_map.gpu_rank_pieces）"})
    for r in range(1, 8):
        (Path(a.janus_run) / "params_sha" / f"rank{r}.init.json").write_text(json.dumps(mans[r], indent=0, sort_keys=True))
    print(f"阶段 {a.stage}：GPU 写法的 FP32 清单 {len(mans[0])} 项 × 8 个 rank → {a.janus_run}")
    return True


def cmd_rope(a) -> bool:
    """RoPE 去交错的三条逐位核对，写 results.json（janus result-schema 的 artifact_check）。"""
    hf = HF(a.hf)
    config, abstract, mapping, _, hf_cfg, slash_of = _mt_setup(a)
    rope = int(hf_cfg["qk_rope_head_dim"])
    tree = _restore(a.ckpt)
    rows = []
    n_ok = 0
    for mt_key in abstract:
        kind = hf_kind(mt_key)
        if kind not in ("kernel_rope", "rope_only"):
            continue
        full = _leaf(tree, slash_of[mt_key])
        for _k, i, h in iter_hf_sources({mt_key: mapping[mt_key]}):
            x = hf.get(h).astype(np.float32)                    # HF [out, in] 或 1 维（k_norm）
            # HF 的输出维按「每头 [nope, rope]」排；kernel 的输出维在第 0 维，1 维张量就是它本身
            xo = x if x.ndim == 1 else x.T                      # → [..., out]
            y = full if i is None else np.take(full, i, axis=stack_axis(mt_key, full.ndim, config.param_scan_axis))
            if kind == "kernel_rope":
                d_last = y.shape[-1]
                y2 = y.reshape(-1, d_last)
                xo2 = xo.reshape(y2.shape[0], -1, d_last) if xo.ndim > 1 else xo.reshape(1, -1, d_last)
                xo2 = xo2.reshape(y2.shape[0], -1)
            else:
                d_last = y.shape[-1]
                y2 = y.reshape(1, d_last)
                xo2 = xo.reshape(1, d_last)
            # ① 正向：rope 段 = 原件 rope 段的偶数位接奇数位；非 rope 段原样
            xr = xo2.reshape(xo2.shape[0], -1, d_last)
            expect = np.concatenate([xr[..., : d_last - rope], xr[..., d_last - rope:][..., 0::2],
                                     xr[..., d_last - rope:][..., 1::2]], axis=-1).reshape(y2.shape)
            fwd = np.array_equal(expect, y2)
            # ② 逆置换逐位回到原件
            perm = deinterleave_perm(d_last, rope)
            inv = np.array_equal(y2.reshape(xr.shape)[..., np.argsort(perm)], xr)
            # ③ 置换本身是双射（逆的逆是自己）
            bij = np.array_equal(np.sort(perm), np.arange(d_last)) and np.array_equal(perm[np.argsort(perm)], np.arange(d_last))
            ok = fwd and inv and bij
            n_ok += ok
            rows.append({"hf": h, "maxtext": slash_of[mt_key], "layer_slot": i, "kind": kind, "head_dim": d_last,
                         "rope": rope, "forward_even_odd": fwd, "inverse_bitwise": inv, "bijection": bij, "ok": ok})
    status = "pass" if rows and n_ok == len(rows) else "fail"
    res = {"schema_version": 1, "stage": f"hy4-{a.stage}-l0-rope", "model": MODEL, "hf": str(a.hf), "ckpt": str(a.ckpt),
           "summary": {"pass": int(status == "pass"), "fail": int(status == "fail"), "inconclusive": 0, "error": 0},
           "checks": [{"id": "L0.ckpt.rope_deinterleave.bitwise", "type": "artifact_check", "level": "L0", "hard": True,
                       "status": status, "metrics": {"tensors": len(rows), "ok": n_ok},
                       "message": f"{n_ok}/{len(rows)} 个带 RoPE 列的张量：正向 = 偶数位接奇数位、逆置换逐位回到原件",
                       "details_inline": {"rows": rows}}]}
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(f"RoPE 去交错：{n_ok}/{len(rows)} 个张量三条全过")
    return status == "pass"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("names")
    s.add_argument("--hf", required=True)
    s.add_argument("--out", default=None, help="结果目录（写 results.json）")
    s.add_argument("--name", default="hy4-l0-names", help="结果里的 stage 名（每个尺寸一份）")
    s = sub.add_parser("hf-manifest")
    s.add_argument("--hf", required=True)
    s.add_argument("--stage", required=True, choices=sorted(STAGES))
    s.add_argument("--janus-run", required=True)
    for name in ("maxtext", "rope", "gpu-manifest"):
        s = sub.add_parser(name)
        s.add_argument("--ckpt", required=True, help="MaxText 参数检查点（…/0/items）")
        s.add_argument("--hf", required=True)
        s.add_argument("--stage", required=True, choices=sorted(STAGES))
        if name in ("maxtext", "gpu-manifest"):
            s.add_argument("--janus-run", default=None, required=name == "gpu-manifest", help="把还原出的逐张量清单写成 janus 运行目录")
        else:
            s.add_argument("--out", default=None, help="结果目录（写 results.json）")
        s.add_argument("maxtext_args", nargs="*", help="与转换时相同的额外 MaxText 覆盖")
    a = p.parse_args(argv)
    ok = {"names": cmd_names, "hf-manifest": cmd_hf_manifest, "maxtext": cmd_maxtext, "rope": cmd_rope, "gpu-manifest": cmd_gpu_manifest}[a.cmd](a)
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
