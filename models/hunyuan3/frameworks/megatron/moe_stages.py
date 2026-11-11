"""MoE 层内分阶段对 FP64（GPU 侧，规格见 docs/dev/design/0004-l1-gpu-kernels.md「三、MoE 层内分阶段」）：router logits → scores → top-k → probs → permute → 专家 GEMM1 → 激活 → GEMM2
→（combine 回本 rank）→ unpermute → 加共享专家，每一阶段的训练实值对 FP64 算 rel_L2，找误差从哪一步跳上去。

输入：带 `--observe moe,expw` 的确定性运行（g61-obs）第 1 步 rank 0 第一个 microbatch；其余 rank 上的专家权重从 HF 原件读
（先按 params_sha 逐个核对和训练时加载的逐位相同）。两种 FP64 参照：
- 链式（chain）：从 MoE 层输入（`mlp_norm.out`，BF16）开始全程 FP64，中间不舍入——累积误差；
- 本阶段（local）：只把这一阶段换成 FP64，输入取训练抓下的上一阶段 BF16 / FP32 实值——这一步自己引入的误差。
数据搬运（permute、all-to-all 前后、combine）另做逐位检查。

用法（Pod 里，一张卡；GPU 共享时包 flock）：
    source /data/env.sh; export PYTHONPATH=/tmp/janus-g61:$PYTHONPATH
    CUDA_VISIBLE_DEVICES=0 flock /data/gpu.lock python models/hunyuan3/frameworks/megatron/moe_stages.py \
        --run /data/janus-runs2/g61-obs --hf /data/hy3-4l --out /data/janus-g61/moe-stages
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from l1_ops import DEV, OPS, STEP, N, T, environment, write_outputs  # noqa: E402

from janus.analysis import ops as O  # noqa: E402
from janus.checks import Threshold  # noqa: E402
from janus.store import Run  # noqa: E402


def D(a) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a).astype(np.float64)).to(DEV)


def H(t: torch.Tensor) -> np.ndarray:
    return t.cpu().numpy()


class Stages:
    def __init__(self, run: Run, hf: str):
        self.run, self.hf = run, Path(hf)
        self.checks = []
        self.evidence = [{"path": str(run.path), "sha256": run.meta_sha256()}]
        mc = run.meta["model_config"]
        self.topk, self.scale = int(mc["moe_router_topk"]), float(mc["moe_router_topk_scaling_factor"])
        self.E = int(mc["num_moe_experts"])
        self.moe = sorted({int(n.split(".")[1]) for n in run.names(STEP, "layers.*.mlp.moe.permuted")})
        self.info = json.loads((run.path / f"observe_step{STEP}.json").read_text())
        self.sha = {}
        for p in sorted((run.path / "params_sha").glob("rank*.init.json")):
            r = int(p.name[4:].split(".")[0])
            self.sha[r] = json.loads(p.read_text())
        self.chain_rows, self.local_rows, self.move_rows, self.topk_rows, self.wcheck = [], [], [], [], []

    def t(self, name):
        return self.run.tensor(STEP, name)

    def p(self, layer, name):
        return self.run.tensor(STEP, f"param.decoder.layers.{layer}.{name}")

    def add(self, cid, cases, threshold=None, hard=True, params=None, level="L2"):
        t0 = time.time()
        r = OPS(cid, level, cases, threshold, hard=hard, params=params)
        r.evidence = list(self.evidence)
        self.checks.append(r)
        print(f"{r.status:<13} {cid:<34} {time.time() - t0:6.1f}s  {r.message[:100]}", flush=True)

    # ---- 全部 192 个专家的权重：HF 原件 → Megatron 布局（fc1 = [gate; up]，fc2 = down），按 params_sha 逐个核对
    def expert_weights(self, L):
        from safetensors import safe_open
        idx = json.loads((self.hf / "model.safetensors.index.json").read_text())["weight_map"]
        info = self.info[f"layers.{L}.mlp.moe"]
        nle = int(info["num_local_experts"])
        W1, W2 = [], []
        handles = {}

        def get(key):
            f = idx[key]
            if f not in handles:
                handles[f] = safe_open(str(self.hf / f), framework="pt")
            return N(handles[f].get_tensor(key))
        for e in range(self.E):
            pre = f"model.layers.{L}.mlp.experts.{e}"
            w1 = np.concatenate([get(f"{pre}.gate_proj.weight"), get(f"{pre}.up_proj.weight")], axis=0)
            w2 = get(f"{pre}.down_proj.weight")
            r, i = divmod(e, nle)
            for nm, w in (("linear_fc1", w1), ("linear_fc2", w2)):
                key = f"decoder.layers.{L}.mlp.experts.{nm}.weight{i}"
                want = self.sha.get(r, {}).get(key, {}).get("sha256")
                got = hashlib.sha256(np.ascontiguousarray(w).view(np.int16).tobytes()).hexdigest()
                self.wcheck.append({"name": f"layers.{L}.experts.{e}.{nm}", "layer": L, "expert": e, "param": nm, "rank": r,
                                    "match": want == got})
            W1.append(w1)
            W2.append(w2)
        return W1, W2

    def layer(self, L):
        info = self.info[f"layers.{L}.mlp.moe"]
        nle = int(info["num_local_experts"])
        x = self.t(f"layers.{L}.mlp_norm.out")
        S, Hd = x.shape[0] * x.shape[1], x.shape[-1]
        x = x.reshape(S, Hd)
        wr = self.p(L, "mlp.router.weight")
        bias = self.t(f"buffer.decoder.layers.{L}.mlp.router.expert_bias").astype(np.float64)
        lg = self.t(f"layers.{L}.mlp.router.logits").reshape(S, -1)
        probs = self.t(f"layers.{L}.mlp.router.probs").reshape(S, -1)
        topk = self.t(f"layers.{L}.mlp.router.topk_idx")
        rmap = self.t(f"layers.{L}.mlp.moe.routing_map").astype(bool)
        perm = self.t(f"layers.{L}.mlp.moe.permuted")
        perm_p = self.t(f"layers.{L}.mlp.moe.permuted_probs")
        comb = self.t(f"layers.{L}.mlp.moe.combined")
        unperm = self.t(f"layers.{L}.mlp.moe.unpermuted").reshape(S, Hd)
        shared = self.t(f"layers.{L}.mlp.shared.out").reshape(S, Hd)
        out = self.t(f"layers.{L}.mlp.out").reshape(S, Hd)
        xin = self.t(f"layers.{L}.mlp.experts.in")
        tpe = self.t(f"layers.{L}.mlp.experts.tokens_per_expert").astype(np.int64)
        cnt = self.t(f"layers.{L}.mlp.moe.tokens_from_rank_per_local_expert").reshape(-1, nle)   # [源 rank, 本地专家]
        y1, act, y2 = (self.t(f"layers.{L}.mlp.experts.{n}") for n in ("fc1.out", "act.out", "out"))
        eprobs = self.t(f"layers.{L}.mlp.experts.probs").astype(np.float64)
        W1, W2 = self.expert_weights(L)

        # 本 rank 置换顺序：按专家号升序，同一专家内按 token 号升序（Megatron permute 的约定，下面逐位核对）
        e_of, t_of = [], []
        for e in range(self.E):
            ts = np.nonzero(rmap[:, e])[0]
            e_of.append(np.full(len(ts), e))
            t_of.append(ts)
        e_of, t_of = np.concatenate(e_of), np.concatenate(t_of)
        # ---- FP64 链式
        xd = D(x)
        lg64 = xd @ D(wr).t()
        sc64 = 1 / (1 + torch.exp(-lg64))
        sel64 = O.topk_stable(H(sc64) + bias, self.topk)
        p64, _ = O.router_probs_f64(H(lg64), bias, self.topk, self.scale, idx=sel64)
        P64 = np.zeros((S, self.E))
        np.put_along_axis(P64, sel64, p64, axis=-1)
        # 专家侧链式：对本 rank 每个 (token, 专家) 对，用对应专家权重全程 FP64（与 permute 顺序一致）
        g1_64 = torch.empty(len(t_of), W1[0].shape[0], dtype=torch.float64, device=DEV)
        g2_64 = torch.empty(len(t_of), Hd, dtype=torch.float64, device=DEV)
        a_64 = torch.empty(len(t_of), W2[0].shape[1], dtype=torch.float64, device=DEV)
        pp64 = D(P64[t_of, e_of])
        start = 0
        for e in range(self.E):
            n = int(rmap[:, e].sum())
            if n == 0:
                continue
            sl = slice(start, start + n)
            h1 = xd[t_of[sl]] @ D(W1[e]).t()
            g, u = h1[:, : h1.shape[1] // 2], h1[:, h1.shape[1] // 2:]
            a = g * torch.sigmoid(g) * u * pp64[sl, None]
            g1_64[sl], a_64[sl], g2_64[sl] = h1, a, a @ D(W2[e]).t()
            start += n
        un64 = torch.zeros(S, Hd, dtype=torch.float64, device=DEV).index_add_(0, torch.from_numpy(t_of).to(DEV), g2_64)
        ws1, ws2 = self.p(L, "mlp.shared_experts.linear_fc1.weight"), self.p(L, "mlp.shared_experts.linear_fc2.weight")
        hs = xd @ D(ws1).t()
        g, u = hs[:, : hs.shape[1] // 2], hs[:, hs.shape[1] // 2:]
        sh64 = (g * torch.sigmoid(g) * u) @ D(ws2).t()
        out64 = un64 + sh64

        # ---- 数据搬运逐位：permute、permuted_probs、all-to-all 后本 rank 自己发给本地专家的段、combine 回来的本地专家段
        mv = [dict(name=f"layers.{L}.permute.tokens", out=perm, bitref=x[t_of], extra={"rows": int(len(t_of))}),
              dict(name=f"layers.{L}.permute.probs", out=perm_p.astype(np.float32), bitref=probs[t_of, e_of].astype(np.float32))]
        off_e = np.concatenate([[0], np.cumsum(tpe)])
        src0 = []                                   # experts.in 里来自 rank 0 的行（按本地专家）
        for e in range(nle):
            base = off_e[e]                         # 每个本地专家内按源 rank 排：rank 0 在最前
            src0.append(np.arange(base, base + cnt[0, e]))
        src0 = np.concatenate(src0)
        own = np.nonzero(e_of < nle)[0]             # 本 rank 置换顺序里发给本 rank 本地专家（0..nle-1）的行
        mv.append(dict(name=f"layers.{L}.dispatch.rank0_to_local", out=xin[src0], bitref=perm[own], extra={"rows": int(len(own))}))
        mv.append(dict(name=f"layers.{L}.combine.local_back", out=comb[own], bitref=y2[src0], extra={"rows": int(len(own))}))
        self.move_rows += mv
        self.topk_rows.append(dict(name=f"layers.{L}.topk.vs_fp64_chain", out=np.sort(topk, -1).astype(np.int64),
                                   expect=np.sort(sel64, -1).astype(np.int64),
                                   extra={"boundary_tie_rows_fp64": int(O.boundary_ties(H(sc64) + bias, self.topk).size)}))

        # ---- 各阶段：链式与本阶段
        with torch.no_grad():
            sc_obs = N(torch.sigmoid(T(lg.astype(np.float32))))           # scores 没抓，用训练同一算子从抓下的 logits 重算
        g1_loc = torch.empty(xin.shape[0], W1[0].shape[0], dtype=torch.float64, device=DEV)
        g2_loc = torch.empty(xin.shape[0], Hd, dtype=torch.float64, device=DEV)
        xind, actd = D(xin), D(act)
        for e in range(nle):
            sl = slice(off_e[e], off_e[e + 1])
            g1_loc[sl] = xind[sl] @ D(W1[e]).t()
            g2_loc[sl] = actd[sl] @ D(W2[e]).t()
        c1 = y1.astype(np.float64)
        g, u = c1[:, : c1.shape[1] // 2], c1[:, c1.shape[1] // 2:]
        act_loc = O.silu_f64(g) * u * eprobs[:, None]
        own_rows = src0                                                      # 链式只对来自 rank 0 的专家输入行有 FP64 链
        g1c, ac, g2c = H(g1_64[own]), H(a_64[own]), H(g2_64)
        comb64_loc = torch.zeros(S, Hd, dtype=torch.float64, device=DEV).index_add_(0, torch.from_numpy(t_of).to(DEV), D(comb))
        rows = [
            ("router.logits", lg, H(lg64), H(lg64), "抓下的 logits（FP32）；本阶段输入 = MoE 层输入"),
            ("router.scores", sc_obs, H(sc64), O.sigmoid_f64(lg), "torch.sigmoid(抓下的 logits)（scores 没抓，重算；L1 已证 top-k 与训练逐位一致）"),
            ("router.probs", probs[np.arange(S)[:, None], np.sort(topk, -1)], P64[np.arange(S)[:, None], np.sort(topk, -1)],
             O.router_probs_f64(lg, bias, self.topk, self.scale, idx=np.sort(topk, -1))[0], "选中位置的概率（FP32）"),
            ("permute", perm, H(xd[t_of]), perm.astype(np.float64), "置换后 token（纯搬运）"),
            ("gemm1", y1[own_rows], g1c, H(g1_loc)[own_rows], "专家 GEMM1 输出（只取 rank 0 自己发来的行，才有链式参照）"),
            ("act", act[own_rows], ac, act_loc[own_rows], "silu(gate)·up·prob → BF16"),
            ("gemm2", comb, g2c, None, "GEMM2 输出（combine 回本 rank，全部 (token, 专家) 对）"),
            ("unpermute", unperm, H(un64), H(comb64_loc), "按 token 把 k 个专家输出加起来（加共享专家之前）"),
            ("shared", shared, H(sh64), H(sh64), "共享专家输出；本阶段输入 = MoE 层输入，所以与链式相同"),
            ("output", out, H(out64), unperm.astype(np.float64) + shared.astype(np.float64), "MoE 层输出 = unpermute + 共享专家（BF16 相加）"),
        ]
        # gemm2 本阶段：专家侧全部行（GEMM2 本身），链式：combine 回来的全部行
        for st, obs, chain, local, desc in rows:
            if st == "gemm2":
                self.local_rows.append(dict(name=f"layers.{L}.gemm2", out=y2, fp64=H(g2_loc),
                                            extra={"stage": st, "layer": L, "desc": "专家侧全部行：训练抓下的激活 · W2"}))
                c = dict(name=f"layers.{L}.gemm2", out=comb, fp64=chain, extra={"stage": st, "layer": L, "desc": desc})
            else:
                self.local_rows.append(dict(name=f"layers.{L}.{st}", out=obs, fp64=local, extra={"stage": st, "layer": L, "desc": desc}))
                c = dict(name=f"layers.{L}.{st}", out=obs, fp64=chain, extra={"stage": st, "layer": L, "desc": desc})
            self.chain_rows.append(c)
        torch.cuda.empty_cache()

    def finish(self):
        self.add("L2.moe.stages.chain", self.chain_rows, None, hard=False, params={"worst_by": ["rel_l2"]})
        self.add("L2.moe.stages.local", self.local_rows, None, hard=False, params={"worst_by": ["rel_l2"]})
        self.add("L2.moe.movement", self.move_rows, Threshold("exact"))
        self.add("L2.moe.topk.fp64", self.topk_rows, None, hard=False)
        ok = all(w["match"] for w in self.wcheck)
        from janus.checks.base import FAIL, PASS, CheckResult
        r = CheckResult("L2.moe.expert_weights", "op_replay", "L2", True)
        r.details = self.wcheck
        r.metrics = {"compared": len(self.wcheck), "mismatch": sum(1 for w in self.wcheck if not w["match"])}
        r.threshold = {"kind": "exact"}
        r.status = PASS if ok and self.wcheck else FAIL
        r.message = (f"HF 原件转成 Megatron 布局后 {len(self.wcheck)} 个专家权重与训练加载时的 sha256 全部相同" if r.status == PASS
                     else f"{r.metrics['mismatch']} 个专家权重与训练时不同")
        r.evidence = list(self.evidence)
        self.checks.insert(0, r)
        print(f"{r.status:<13} {r.id:<34}         {r.message}", flush=True)


COLS = ["stage", "rel_l2", "cr_match_frac", "ulp_p999", "bit_match_frac", "expect_match_frac", "rows_differ",
        "boundary_tie_rows_fp64", "rows", "match"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--hf", required=True, help="HF 原件目录（取其余 rank 上专家的权重）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)          # FP64 参照里的 index_add_ 默认用原子加，末位会随调度变
    env = environment()
    run = Run.open(a.run)
    run.enable_cache()
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    job = Stages(run, a.hf)
    for L in job.moe:
        t0 = time.time()
        job.layer(L)
        print(f"  layer {L} {time.time() - t0:.0f}s", flush=True)
    job.finish()
    return write_outputs(Path(a.out), job.checks, [run], env, Path(__file__), "phase1-moe-stages", started, columns=COLS,
                         ref_line="参照：FP64 链式（从 MoE 层输入全程 FP64）与 FP64 本阶段（输入取训练实值）",
                         col_note="列：stage 阶段；rel_l2 对 FP64（`L2.moe.stages.chain` 对链式、`.local` 对本阶段）；cr_match_frac 对 FP64 正确舍入到输出 dtype 的逐位相同率；"
                                  "bit_match_frac 搬运检查的逐位相同率；expect_match_frac top-k 选中集合与 FP64 链式一致的行比例。")


if __name__ == "__main__":
    sys.exit(main())
