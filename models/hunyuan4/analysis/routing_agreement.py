"""路由一致率（PROTOCOL 第 5 节，2026-11-09 起的正式检查）：TPU 自选的 MoE 路由与 DSA 索引器 top-k 离 CPU FP64 自选
有多远，与 GPU 离 FP64 的距离比（对真值口径，判定）；对 GPU 的翻转率与 GPU 两两的尺子照报不判。

量的定义（逐层）：
- 路由翻转率：一个 token 选中的专家集合（top-k，不计顺序）与对方不同即算翻转；翻转 token 数 ÷ token 数。
  另按分差分桶：分差 = CPU FP64 参照（`step1_ref.py` 的 choices.safetensors，加偏置后的分数）里第 k 名与第 k+1 名之差，
  同一个 token 在各对比较里落进同一个桶（分桶轴与被比较的两边都无关）。
- DSA 不重合率：自带索引器的层，每个 query 选中的位置集合里对方没选的比例（1 − |A∩B| ÷ |A|），对有选择的 query 取平均；
  另记「选中集合不完全相同」的 query 比例。
判定（逐层，再取最差层）：正式判定用对真值口径：R = TPU 对 CPU FP64 自选的量 ÷ GPU 各份对 FP64 的最大值，≤ 1.2 过、
≤ 2 待定、> 2 不过（1.2 这条线同 L3 的 R，待定一档是这里自己的；DSA 要 FP64 每行都存了 top-k：step1_ref.py --dsa-topk-all）。「对 GPU 两两」口径照报不判：尺子 = GPU 各对的最大值，
TPU ≤ 尺子 / ≤ 2 倍 / > 2 倍 记 过 / 待定 / 不过；尺子为 0 时 TPU 为 0 记过、不为 0 记不过。GPU 两两量的是同一套舍入的确定性。

输入（每份「选择」一个来源，行按 token 逐个相等对上第一份 GPU，不依赖文件里的行号）：
- GPU 抓数目录：每行一个 step_<N>_row_<R>.safetensors（`hf.model.layers.<L>.mlp.gate.routing_map`、
  `hf.model.layers.<L>.self_attn.topk`、`input.tokens`；MTP 层是 `hf.model.mtp_layers.<i>`）；
- 紧凑文件（`build_forced_routing.py` 的输出或 TPU `--capture-choices` 抓的，同一格式）：`forced_routed_experts`
  [行, 序列, 槽位, k]、可选 `forced_indexer_topk` [行, 序列, 槽位, topk]、`tokens` [行, 序列]；槽位 L = HF 第 L 层，
  MTP 层排在主干层之后。

    python models/hunyuan4/analysis/routing_agreement.py --step 1 \\
        --gpu g8-chg-a=<抓数目录> --gpu g8-chg-b=<抓数目录> --gpu g8-chg-c=<抓数目录> \\
        [--tpu tw-ra1=<紧凑文件>] --fp64 <FP64 参照目录> --tokens <tokens.npy> --gbs 16 \\
        --num-decoder-layers 7 --dsa-layers 0,1,5,mtp0 --out <json>
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re

import numpy as np
from safetensors import safe_open

BUCKETS = [0.0, 1e-4, 1e-3, 1e-2, float("inf")]


def bucket_names() -> list[str]:
    def f(x):
        return "∞" if x == float("inf") else f"{x:g}"

    return [f"[{f(a)}, {f(b)})" for a, b in itertools.pairwise(BUCKETS)]


def slot_name(slot: int, n_main: int) -> str:
    return str(slot) if slot < n_main else f"mtp{slot - n_main}"


class GpuDir:
    """GPU 抓数目录：每行一个文件，按需读。"""

    def __init__(self, path: str, step: int):
        self.files = sorted(
            os.path.join(path, f) for f in os.listdir(path) if re.fullmatch(rf"step_{step}_row_\d+\.safetensors", f)
        )
        if not self.files:
            raise SystemExit(f"{path}：没有 step_{step}_row_*.safetensors")
        self._h = [safe_open(f, "np") for f in self.files]
        self.tokens = [h.get_tensor("input.tokens").reshape(-1) for h in self._h]

    @staticmethod
    def _key(layer: str, what: str) -> str:
        if layer.startswith("mtp"):
            return f"hf.model.mtp_layers.{layer[3:]}.{what}"
        return f"hf.model.layers.{layer}.{what}"

    def route(self, row: int, layer: str):
        h = self._h[row]
        k = self._key(layer, "mlp.gate.routing_map")
        if k not in h.keys():
            return None
        rm = h.get_tensor(k).reshape(len(self.tokens[row]), -1)
        return rm.astype(bool)

    def dsa(self, row: int, layer: str):
        h = self._h[row]
        k = self._key(layer, "self_attn.topk")
        if k not in h.keys():
            return None
        return h.get_tensor(k).reshape(len(self.tokens[row]), -1)


class Compact:
    """紧凑文件（build_forced_routing 输出 / TPU --capture-choices）。"""

    def __init__(self, path: str, n_main: int, n_experts: int):
        self.h = safe_open(path, "np")
        self.tokens = list(self.h.get_tensor("tokens"))
        self.n_main, self.n_experts = n_main, n_experts
        self._r = self.h.get_tensor("forced_routed_experts")
        self._d = self.h.get_tensor("forced_indexer_topk") if "forced_indexer_topk" in self.h.keys() else None

    def _slot(self, layer: str) -> int:
        return self.n_main + int(layer[3:]) if layer.startswith("mtp") else int(layer)

    def route(self, row: int, layer: str):
        idx = self._r[row, :, self._slot(layer), :]
        if (idx < 0).all():
            return None
        rm = np.zeros((idx.shape[0], self.n_experts), bool)
        np.put_along_axis(rm, idx.astype(np.int64), True, axis=1)
        return rm

    def dsa(self, row: int, layer: str):
        if self._d is None:
            return None
        return self._d[row, :, self._slot(layer), :]


def topk_mask(topk: np.ndarray, seq: int) -> np.ndarray:
    """[seq, topk] 的位置下标（-1 = 空位）→ [seq, seq] 的布尔掩码。"""
    m = np.zeros((topk.shape[0], seq + 1), bool)
    t = np.where(topk >= 0, topk, seq).astype(np.int64)
    np.put_along_axis(m, t, True, axis=1)
    return m[:, :seq]


def bad_queries(topk: np.ndarray) -> np.ndarray:
    """完整性自检：第 q ≥ k 个 query（k = top-k 宽度）能看到的位置不少于 k 个，选中的 k 个位置必须互不相同且非空；
    取回时被零填充的块（整段 query 全是 0）会在这里被抓到。返回不合格 query 的下标。"""
    k = topk.shape[-1]
    t = np.sort(topk[k:], axis=-1)
    ok = (t[:, 0] >= 0) & (np.diff(t, axis=-1) > 0).all(-1)
    return np.where(~ok)[0] + k


def fp64_margins(fp64_dir: str, rows: list[int], layers: list[str], k: int, n_experts: int) -> tuple[dict, dict]:
    """FP64 参照每行每层：每个 token 的分差（加偏置后分数的第 k 名减第 k+1 名），与 FP64 自选的专家集合（布尔 [token, 专家]）。"""
    h = safe_open(os.path.join(fp64_dir, "choices.safetensors"), "np")
    marg, sel = {}, {}
    for r in rows:
        for L in layers:
            key = f"r{r}.route.{L}.scores"
            if key in h.keys():
                s = -np.sort(-h.get_tensor(key), axis=1)
                marg[(r, L)] = s[:, k - 1] - s[:, k]
                idx = h.get_tensor(f"r{r}.route.{L}.idx").astype(np.int64)
                m = np.zeros((idx.shape[0], n_experts), bool)
                np.put_along_axis(m, idx, True, axis=1)
                sel[(r, L)] = m
    return marg, sel


TRUTH_R = 1.2


def judge_truth(tpu_vs64: float, gpu_vs64: list[float]) -> tuple[str, float | None]:
    """正式判定（对真值）：R = TPU 对 FP64 的量 ÷ GPU 各份对 FP64 的最大值，≤ 1.2 过，≤ 2 待定，> 2 不过（1.2 这条线同 L3 的 R，待定一档是这里自己的）。"""
    g = max(gpu_vs64)
    if g == 0:
        return ("pass" if tpu_vs64 == 0 else "fail"), None
    r = tpu_vs64 / g
    return ("pass" if r <= TRUTH_R else "inconclusive" if r <= 2 else "fail"), r


def fp64_dsa(fp64_dir: str, row: int, layer: str):
    """FP64 参照这一行这一层的 DSA top-k（step1_ref.py --dsa-topk-all 或 --detail-rows 存的）；没有给 None。"""
    h = safe_open(os.path.join(fp64_dir, "choices.safetensors"), "np")
    key = f"r{row}.dsa.{layer}.topk"
    return h.get_tensor(key).reshape(-1, h.get_slice(key).get_shape()[-1]) if key in h.keys() else None


def judge(tpu_vals: list[float], ruler: float) -> tuple[str, float | None]:
    worst = max(tpu_vals)
    if ruler == 0:
        return ("pass" if worst == 0 else "fail"), None
    ratio = worst / ruler
    return ("pass" if ratio <= 1 else "inconclusive" if ratio <= 2 else "fail"), ratio


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--gpu", action="append", required=True, help="名字=GPU 抓数目录（≥ 2 份；第一份定行序）")
    p.add_argument("--tpu", action="append", default=[], help="名字=紧凑文件（TPU 自选，不灌）")
    p.add_argument("--fp64", default="", help="FP64 参照目录（含 choices.safetensors）；给了才分桶")
    p.add_argument("--tokens", default="", help="FP64 参照用的 tokens.npy（定 FP64 的行）")
    p.add_argument("--gbs", type=int, default=16)
    p.add_argument("--num-decoder-layers", type=int, required=True)
    p.add_argument("--num-experts", type=int, default=16)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--moe-layers", default="", help="逗号分隔；空 = 第一份 GPU 里有路由的层")
    p.add_argument("--dsa-layers", default="", help="自带索引器的层，逗号分隔（如 0,1,5,mtp0）；空 = 不比 DSA")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    gpus = {n: GpuDir(d, a.step) for n, d in (s.split("=", 1) for s in a.gpu)}
    if len(gpus) < 2:
        raise SystemExit("尺子要 ≥ 2 份 GPU")
    tpus = {n: Compact(f, a.num_decoder_layers, a.num_experts) for n, f in (s.split("=", 1) for s in a.tpu)}
    names = list(gpus) + list(tpus)
    srcs = {**gpus, **tpus}
    base = gpus[names[0]]
    n_rows = len(base.tokens)

    # 行对齐：按 token 逐个相等
    rowmap = {}
    for n, s in srcs.items():
        m = []
        for t in base.tokens:
            hit = [j for j, u in enumerate(s.tokens) if len(u) == len(t) and np.array_equal(u, t)]
            if len(hit) != 1:
                raise SystemExit(f"{n}：第一份 GPU 的某行在这里对上 {len(hit)} 行")
            m.append(hit[0])
        rowmap[n] = m
    fp64_rows = None
    if a.fp64:
        toks = np.load(a.tokens, mmap_mode="r")
        lo = (a.step - 1) * a.gbs
        fp64_rows = []
        for t in base.tokens:
            hit = [r for r in range(a.gbs) if np.array_equal(np.asarray(toks[lo + r]).reshape(-1)[: len(t)], t)]
            if len(hit) != 1:
                raise SystemExit(f"FP64：第一份 GPU 的某行对上 {len(hit)} 行")
            fp64_rows.append(hit[0])

    n_main = a.num_decoder_layers
    if a.moe_layers:
        moe = a.moe_layers.split(",")
    else:
        moe = [slot_name(s, n_main) for s in range(n_main + 4) if base.route(0, slot_name(s, n_main)) is not None]
    dsa_layers = [x for x in a.dsa_layers.split(",") if x]
    margins, fp64_sel = fp64_margins(a.fp64, sorted(set(fp64_rows)), moe, a.topk, a.num_experts) if a.fp64 else ({}, {})
    pairs = list(itertools.combinations(list(gpus), 2)) + [(t, g) for t in tpus for g in gpus]
    bnames = bucket_names()

    route_res, dsa_res = {}, {}
    for L in moe:
        flips = {f"{x}~{y}": 0 for x, y in pairs}
        n_tok = 0
        bflips = {f"{x}~{y}": np.zeros(len(bnames), np.int64) for x, y in pairs}
        bcount = np.zeros(len(bnames), np.int64)
        vs64 = dict.fromkeys(names, 0)
        for i in range(n_rows):
            maps = {n: srcs[n].route(rowmap[n][i], L) for n in names}
            if any(v is None for v in maps.values()):
                raise SystemExit(f"第 {L} 层：有一份没有路由")
            n_tok += maps[names[0]].shape[0]
            b = None
            if margins:
                mg = margins[(fp64_rows[i], L)]
                b = np.clip(np.searchsorted(BUCKETS, mg, side="right") - 1, 0, len(bnames) - 1)
                bcount += np.bincount(b, minlength=len(bnames))
                for n in names:
                    vs64[n] += int((maps[n] != fp64_sel[(fp64_rows[i], L)]).any(axis=1).sum())
            for x, y in pairs:
                d = (maps[x] != maps[y]).any(axis=1)
                flips[f"{x}~{y}"] += int(d.sum())
                if b is not None:
                    bflips[f"{x}~{y}"] += np.bincount(b[d], minlength=len(bnames))
        rate = {k: v / n_tok for k, v in flips.items()}
        gpu_pairs = [f"{x}~{y}" for x, y in itertools.combinations(list(gpus), 2)]
        ruler = max(rate[k] for k in gpu_pairs)
        res = {
            "n_tokens": n_tok,
            "flip_rate": rate,
            "ruler_gpu_max": ruler,
            "gpu_pair_range": [min(rate[k] for k in gpu_pairs), ruler],
        }
        for t in tpus:
            vals = [rate[f"{t}~{g}"] for g in gpus]
            st, ratio = judge(vals, ruler)
            res.setdefault("tpu", {})[t] = {"vs_each_gpu": dict(zip(gpus, vals)), "ratio_to_ruler": ratio, "status": st}
        if margins:
            res["vs_fp64_flip_rate"] = {n: v / n_tok for n, v in vs64.items()}
            for t in tpus:
                st, r = judge_truth(res["vs_fp64_flip_rate"][t], [res["vs_fp64_flip_rate"][g] for g in gpus])
                res["tpu"][t]["truth"] = {"R": r, "status": st}
            res["margin_buckets"] = {
                "edges": bnames,
                "count": bcount.tolist(),
                "flip_rate": {k: [float(f / c) if c else None for f, c in zip(v, bcount)] for k, v in bflips.items()},
            }
        route_res[L] = res

    for L in dsa_layers:
        acc = {f"{x}~{y}": [0.0, 0, 0] for x, y in pairs}  # 不重合率之和、不完全相同的 query 数、有选择的 query 数
        acc64 = {n: [0.0, 0] for n in names}
        have64 = bool(a.fp64)
        for i in range(n_rows):
            tk = {n: srcs[n].dsa(rowmap[n][i], L) for n in names}
            if any(v is None for v in tk.values()):
                raise SystemExit(f"DSA 第 {L} 层：有一份没有 top-k")
            for n, v in tk.items():
                bad = bad_queries(v)
                if len(bad):
                    raise SystemExit(
                        f"{n} 第 {rowmap[n][i]} 行 DSA 第 {L} 层有 {len(bad)} 个 query 的 top-k 不完整"
                        f"（query {bad.min()}–{bad.max()}；重复或空位，文件可能取回时损坏）"
                    )
            seq = tk[names[0]].shape[0]
            masks = {n: topk_mask(v, seq) for n, v in tk.items()}
            ref = fp64_dsa(a.fp64, fp64_rows[i], L) if have64 else None
            if have64 and ref is None:
                have64 = False  # FP64 没存这一行：对真值口径不判
            if ref is not None:
                m64 = topk_mask(ref, seq)
                for n in names:
                    na = masks[n].sum(1)
                    ok = na > 0
                    acc64[n][0] += float((1 - (masks[n] & m64).sum(1)[ok] / na[ok]).sum())
                    acc64[n][1] += int(ok.sum())
            for x, y in pairs:
                na = masks[x].sum(1)
                inter = (masks[x] & masks[y]).sum(1)
                ok = na > 0
                acc[f"{x}~{y}"][0] += float((1 - inter[ok] / na[ok]).sum())
                acc[f"{x}~{y}"][1] += int((masks[x][ok] != masks[y][ok]).any(1).sum())
                acc[f"{x}~{y}"][2] += int(ok.sum())
        nonov = {k: v[0] / v[2] for k, v in acc.items()}
        differ = {k: v[1] / v[2] for k, v in acc.items()}
        gpu_pairs = [f"{x}~{y}" for x, y in itertools.combinations(list(gpus), 2)]
        ruler = max(nonov[k] for k in gpu_pairs)
        res = {
            "n_queries": acc[gpu_pairs[0]][2],
            "nonoverlap_rate": nonov,
            "query_differ_frac": differ,
            "ruler_gpu_max": ruler,
            "gpu_pair_range": [min(nonov[k] for k in gpu_pairs), ruler],
        }
        for t in tpus:
            vals = [nonov[f"{t}~{g}"] for g in gpus]
            st, ratio = judge(vals, ruler)
            res.setdefault("tpu", {})[t] = {"vs_each_gpu": dict(zip(gpus, vals)), "ratio_to_ruler": ratio, "status": st}
        if have64:
            res["vs_fp64_nonoverlap_rate"] = {n: v[0] / v[1] for n, v in acc64.items()}
            for t in tpus:
                st, r = judge_truth(
                    res["vs_fp64_nonoverlap_rate"][t], [res["vs_fp64_nonoverlap_rate"][g] for g in gpus]
                )
                res["tpu"][t]["truth"] = {"R": r, "status": st}
        dsa_res[L] = res

    order = {"pass": 0, "inconclusive": 1, "fail": 2}
    overall, overall_gpu_pairs = {}, {}
    for t in tpus:
        rows_ = list(route_res.values()) + list(dsa_res.values())
        sts = [r["tpu"][t]["status"] for r in rows_]
        overall_gpu_pairs[t] = max(sts, key=order.get) if sts else None
        tr = [r["tpu"][t]["truth"]["status"] for r in rows_ if "truth" in r["tpu"][t]]
        missing = [L for L, r in list(route_res.items()) + list(dsa_res.items()) if "truth" not in r["tpu"][t]]
        overall[t] = {"status": max(tr, key=order.get) if tr else None, "layers_not_judged": missing}
    out = {
        "what": f"路由一致率（第 {a.step} 步）：MoE 路由翻转率与 DSA top-k 不重合率，逐层；正式判定对 CPU FP64（truth / overall），GPU 两两照报不判",
        "definitions": {
            "flip_rate": "专家集合（top-k，不计顺序）不同的 token ÷ token 数",
            "nonoverlap_rate": "每个 query 选中位置里对方没选的比例，对有选择的 query 平均",
            "margin": "CPU FP64 参照加偏置后分数的第 k 名 − 第 k+1 名（分桶轴，与被比较两边无关）",
            "vs_fp64_flip_rate": "各份对 CPU FP64 自选的翻转率（对真值口径的分子与分母）",
            "vs_fp64_nonoverlap_rate": "各份对 CPU FP64 自选的 DSA 不重合率（对真值口径的分子与分母）",
            "truth": "正式判定（PROTOCOL 第 5 节，2026-11-09 项目负责人定）：每层 R = TPU 对 CPU FP64 的翻转率（DSA 为不重合率）÷ GPU 各份对 FP64 的最大值，≤ 1.2 过，≤ 2 待定，> 2 不过；整体取最差层（overall）",
            "status": "只报不判：每层 TPU 对每份 GPU 的量 ÷ GPU 两两最大值，≤ 1 / ≤ 2 / > 2 记 过 / 待定 / 不过（overall_gpu_pairs）；GPU 两两量的是同一套舍入的确定性",
        },
        "step": a.step,
        "gpu": list(gpus),
        "tpu": list(tpus),
        "rows": n_rows,
        "routing": route_res,
        "dsa": dsa_res,
        "overall": overall,
        "overall_gpu_pairs": overall_gpu_pairs,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for L, r in route_res.items():
        print(
            f"route {L:>5}: GPU 两两 {r['gpu_pair_range'][0]:.4f}–{r['gpu_pair_range'][1]:.4f}",
            " ".join(
                f"{t} {max(v['vs_each_gpu'].values()):.4f} {v['status']} 真值 {v.get('truth')}"
                for t, v in r.get("tpu", {}).items()
            ),
        )
    for L, r in dsa_res.items():
        print(
            f"dsa   {L:>5}: GPU 两两 {r['gpu_pair_range'][0]:.4f}–{r['gpu_pair_range'][1]:.4f}",
            " ".join(
                f"{t} {max(v['vs_each_gpu'].values()):.4f} {v['status']} 真值 {v.get('truth')}"
                for t, v in r.get("tpu", {}).items()
            ),
        )
    print("overall", overall)


if __name__ == "__main__":
    main()
