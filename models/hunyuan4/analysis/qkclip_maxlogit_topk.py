"""诊断：CPU FP64 前向里记每层每头缩放后 q·k 的前 K 大（选中且因果，不含 sink，跨行），给 QK-Clip 逐头最大 logit 定责用。

为什么：低阈值 QK-Clip 第 1 步逐头最大 logit 的 R 是一个「最大值」统计量：只要最大的那一对 (q, k) 在被测一侧算低了
或没被选中，读数就掉到第二大的那一对。这里在 `step1_ref.py` 的 FP64 前向里替换 `hy4_mcore.sparse_sink_attention`，
逐层逐头记前 K 大的值与位置（行、query 位置、key 位置），以及最大那一对的 query 位置是否不少于索引器 top-k
（位置 < top-k 时因果范围内的 key 全被选中，不存在 DSA 选择翻转）。层号取调用方 `Hy4DSAttention.layer_number`
（1 起，MTP 层另记）。不改任何数值。

用法：参数原样转给 step1_ref.py（只前向、不加 --audit）：
    python qkclip_maxlogit_topk.py --k 8 --out-json <json> -- --hf ... --tokens ... --dtype float64 --out <目录> \\
        --forward-only --log-max-logit
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "frameworks", "megatron"))


def main():
    argv = sys.argv[1:]
    K, out_json, watch, pairs_file = 8, None, [], None
    while argv and argv[0] != "--":
        if argv[0] == "--k":
            K, argv = int(argv[1]), argv[2:]
        elif (
            argv[0] == "--pairs"
        ):  # JSON 文件：[{layer, head, row, q, k}, …]，存这些 (q, k) 的 query / key 向量（FP64，已含缩放前）
            pairs_file, argv = argv[1], argv[2:]
        elif (
            argv[0] == "--watch"
        ):  # 行:query:key[,…]：在每个 full 层记这一对的索引器分数、名次与第 top-k、top-k+1 名的分数
            watch, argv = [tuple(int(x) for x in w.split(":")) for w in argv[1].split(",")], argv[2:]
        elif argv[0] == "--out-json":
            out_json, argv = argv[1], argv[2:]
        else:
            raise SystemExit(f"不认识的参数 {argv[0]}")
    argv = argv[1:]
    assert out_json and "--audit" not in argv
    import hy4_mcore as H
    import step1_ref as S
    import torch

    rec = {}  # layer -> head -> list of (val, row, q, k)
    rows = [int(x) for x in argv[argv.index("--rows") + 1].split(",")] if "--rows" in argv else None
    cur = {"layer": None, "row": -1, "n": -1, "seen": set()}
    orig_fwd = H.Hy4DSAttention.forward

    def fwd(self, *a, **k):
        lname = str(self.layer_number) + ("mtp" if getattr(self, "force_full", False) else "")
        if lname in cur["seen"]:  # 同一行里同一层第二次出现 = 下一行开始
            cur["seen"] = set()
        if not cur["seen"]:
            cur["n"] += 1
            cur["row"] = rows[cur["n"]] if rows else cur["n"]
        cur["seen"].add(lname)
        cur["layer"] = lname
        try:
            return orig_fwd(self, *a, **k)
        finally:
            cur["layer"] = None

    H.Hy4DSAttention.forward = fwd
    orig = H.sparse_sink_attention

    pairs = json.load(open(pairs_file)) if pairs_file else []
    pair_vecs = {}

    def ssa(query, key, value, topk, sinks, scale, scores_fp32=False, return_max_logit=False):
        for pp in pairs:
            if pp["layer"] == cur["layer"] and pp["row"] == cur["row"]:
                tag = f"{pp['layer']}.{pp['head']}.{pp['row']}.{pp['q']}.{pp['k']}"
                pair_vecs[tag + ".q"] = query[pp["q"], 0, pp["head"]].detach().double().clone()
                pair_vecs[tag + ".k"] = key[pp["k"], 0, pp["head"]].detach().double().clone()
                pair_vecs[tag + ".scale"] = torch.tensor([float(scale)], dtype=torch.float64)
        with torch.no_grad():
            sq, b, h, _ = query.shape
            sk = key.shape[0]
            q = query.permute(1, 2, 0, 3).double()
            kk = key.permute(1, 2, 0, 3).double()
            sc = torch.matmul(q, kk.transpose(-1, -2)) * scale
            allowed = H.causal_sparse_mask(topk, sq, sk).unsqueeze(1)
            sc = sc.masked_fill(~allowed, float("-inf"))[0]  # [h, sq, sk]
            flat = sc.reshape(h, -1)
            v, idx = torch.topk(flat, K, dim=-1)
            L = rec.setdefault(cur["layer"], {})
            for hh in range(h):
                lst = L.setdefault(hh, [])
                for j in range(K):
                    qi, ki = divmod(int(idx[hh, j]), sk)
                    lst.append((float(v[hh, j]), cur["row"], qi, ki))
                lst.sort(key=lambda t: -t[0])
                del lst[K:]
            L.setdefault("_topk_width", int(topk.shape[-1]))
        return orig(query, key, value, topk, sinks, scale, scores_fp32, return_max_logit)

    H.sparse_sink_attention = ssa
    idx_rec = []
    if watch:
        from megatron.core.transformer.experimental_attention_variant.dsa import _compute_index_scores

        orig_topk = H.fused_qk_topk_naive

        def topk_w(q, k, weights, index_topk, mask=None, **kw):
            res = orig_topk(q, k, weights, index_topk, mask=mask, **kw)
            for row, qi, ki in watch:
                if row != cur["row"]:
                    continue
                with torch.no_grad():
                    sc = _compute_index_scores(
                        q.double(), weights.double(), k.double(), use_relu=kw.get("use_relu", True)
                    )[0, qi, : qi + 1]
                    srt, order = torch.sort(sc, descending=True)
                    rank = int((order == ki).nonzero()[0, 0]) + 1
                    sel = ki in set(res[1][0, qi].tolist())
                    idx_rec.append(
                        {
                            "layer": cur["layer"],
                            "row": row,
                            "q": qi,
                            "k": ki,
                            "score": float(sc[ki]),
                            "rank": rank,
                            "causal_keys": qi + 1,
                            "kth": float(srt[index_topk - 1]) if qi + 1 >= index_topk else None,
                            "kth_plus1": float(srt[index_topk]) if qi + 1 > index_topk else None,
                            "selected": bool(sel),
                        }
                    )
            return res

        H.fused_qk_topk_naive = topk_w
    sys.argv = ["step1_ref.py"] + argv
    try:
        S.main()
    finally:
        out = {"what": __doc__.splitlines()[0], "k": K, "layers": {}, "watch": idx_rec}
        for lname, L in rec.items():
            w = L.pop("_topk_width", None)
            out["layers"][lname] = {
                "topk_width": w,
                "heads": {
                    str(hh): [{"val": t[0], "row": t[1], "q": t[2], "k": t[3]} for t in lst] for hh, lst in L.items()
                },
            }
        with open(out_json, "w") as f:
            json.dump(out, f, indent=1)
        if pair_vecs:
            from safetensors.torch import save_file

            save_file(pair_vecs, os.path.splitext(out_json)[0] + ".pairs.safetensors")


if __name__ == "__main__":
    main()
