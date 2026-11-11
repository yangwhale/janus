"""TPU 原生数据加载器（MaxText Grain，grain_file_type=mmap_npy）对 GPU（Megatron GPTDataset）逐样本对拍。

读两样东西：
  - MaxText 侧：maxtext_dump.py 存下的每步全局批（<mt>/batches.bin + batches.jsonl + meta.json）；
  - GPU 侧：golden 运行 --save-batches 存下的每 rank 每步每个 microbatch 的整批输入（<run>/data/batches/rank<r>.*）、
    数据指纹（<run>/data/rank<r>.jsonl）和 meta.json；另读 Megatron 缓存的 doc / sample / shuffle 索引（.npy）。

逐项比（每项一条检查，写进 results.json，report.md 从它渲染）：
  L0.native.gpu_integrity     GPU 整批输入自身的 sha256 对得上它的索引和数据指纹（先证参照可信）
  L0.native.index             doc / sample / shuffle 三个索引：Megatron 缓存 vs MaxText 重建，文件与数组 sha256
  L0.native.stream            每步全局批作为一条样本流：MaxText 第 g 行 = Megatron 步内第 g 个样本（与怎么分给设备无关）
  L0.native.device_map.native MaxText 原样的「设备 d · microbatch m」对 GPU「rank d · microbatch m」
  L0.native.fingerprint       经 janus.data_layout 换序后，同上，四个字段逐元素 + sha256（硬检查）
  L0.native.nonpad_total      每步全局非 padding token 数
  L0.native.fields            字段级 dtype / shape 约定、注意力分段与位置的约定、pad / EOD 计数
  L0.native.readers           不同读取进程数（grain_worker_count）下逐步逐位相同
  L0.native.hosts             模拟多 host（每个 host 用 MaxText 自己的 host 切分建本地迭代器）：样本顺序怎么变、换序后是否仍对上
  L0.native.epochs            GPU 训练没跑到的样本数（跨 epoch、最后一个 epoch 单独打乱）：两边现建索引与边界样本
  L0.native.eod_cases         合成边界用例：EOD / 位置重置 / 注意力 / loss mask 在各开关组合下两边是否一致
  L0.native.pad_id            同上，数据里出现 pad id 时两边的处理
  L0.native.negative_controls 反例：故意把 MaxText 配错（默认的 reset_attention_mask、不设 split、默认种子），对拍要报不同并定位对
  L0.native.coverage          比到的单元数

用法：python native_loader_parity.py --mt <dump 目录> --gpu-run <golden 运行目录> --megatron-index-dir <缓存目录> \
          --out <输出目录> [--mt-variant <另一个 dump 目录> ...] [--mt-hosts <模拟多 host 的 dump 目录> ...] \
          [--epochs-ref <megatron 参照目录>:<maxtext 参照目录> ...] [--eod-ref <megatron 合成>:<maxtext 合成>] [--steps 1:1000]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from janus.checks import REGISTRY  # noqa: E402
from janus.data_layout import maxtext_loaded_slot, maxtext_perm_for_megatron, per_device  # noqa: E402

GPU_FIELDS = ("tokens", "labels", "loss_mask", "position_ids")
MT_FIELDS = ("inputs", "targets", "inputs_segmentation", "targets_segmentation", "inputs_position", "targets_position")
INDEX_KINDS = ("document_index", "sample_index", "shuffle_index")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- 读数据

class GpuBatches:
    """GPU 整批输入：get(step, rank, micro) → {字段: 数组}（dtype / shape 原样）。"""

    def __init__(self, run: Path, dp: int):
        self.run, self.dp, self.idx, self.mm = Path(run), dp, {}, {}
        for r in range(dp):
            b = self.run / "data" / "batches" / f"rank{r}.bin"
            self.mm[r] = np.memmap(b, dtype=np.uint8, mode="r")
            for line in open(self.run / "data" / "batches" / f"rank{r}.jsonl"):
                rec = json.loads(line)
                self.idx[(rec["step"], r, rec["micro"])] = rec["fields"]
        self.fp = {}
        for r in range(dp):
            fpath = self.run / "data" / f"rank{r}.jsonl"
            if fpath.exists():
                for line in open(fpath):
                    rec = json.loads(line)
                    self.fp[(rec["step"], r, rec["micro"])] = rec

    def steps(self):
        return sorted({k[0] for k in self.idx})

    def get(self, step, rank, micro) -> dict:
        out = {}
        for k, f in self.idx[(step, rank, micro)].items():
            n = int(np.prod(f["shape"])) * np.dtype(f["dtype"]).itemsize
            out[k] = np.frombuffer(self.mm[rank][f["offset"]:f["offset"] + n], dtype=f["dtype"]).reshape(f["shape"])
        return out


class MtBatches:
    """MaxText dump：get(step) → {字段: [gbs, L] 数组}。"""

    def __init__(self, d: Path):
        self.d = Path(d)
        self.meta = json.load(open(self.d / "meta.json"))
        self.mm = np.memmap(self.d / "batches.bin", dtype=np.uint8, mode="r")
        self.idx = {}
        for line in open(self.d / "batches.jsonl"):
            rec = json.loads(line)
            self.idx[rec["step"]] = rec["fields"]

    def steps(self):
        return sorted(self.idx)

    def get(self, step) -> dict:
        out = {}
        for k, f in self.idx[step].items():
            n = int(np.prod(f["shape"])) * np.dtype(f["dtype"]).itemsize
            out[k] = np.frombuffer(self.mm[f["offset"]:f["offset"] + n], dtype=f["dtype"]).reshape(f["shape"])
        return out


def to_gpu_fields(mt: dict, gpu_ref: dict | None = None) -> dict:
    """MaxText 一行（或几行）→ GPU 四个字段，dtype / shape 按 GPU 的来（GPU 存的是 [mbs, L]）。

    对应关系（maxtext 的 mmap_npy 管线 → Megatron GPTDataset.__getitem__）：
      inputs → tokens，targets → labels，targets_segmentation != 0 → loss_mask，inputs_position → position_ids
    """
    dt = {k: (gpu_ref[k].dtype if gpu_ref else d) for k, d in
          (("tokens", np.int64), ("labels", np.int64), ("loss_mask", np.float32), ("position_ids", np.int64))}
    return {"tokens": mt["inputs"].astype(dt["tokens"]), "labels": mt["targets"].astype(dt["labels"]),
            "loss_mask": (mt["targets_segmentation"] != 0).astype(dt["loss_mask"]),
            "position_ids": mt["inputs_position"].astype(dt["position_ids"])}


# ---------------------------------------------------------------- 定位：对不上时是哪一步的约定

def localize(mine: dict, ref: dict, step_ref_tokens: list, eod: int, pad: int | None) -> str:
    """一个单元对不上时，判断差异出在哪一步。mine / ref：GPU 四字段；step_ref_tokens：同一步 GPU 全部样本的 tokens。"""
    if not np.array_equal(mine["tokens"], ref["tokens"]):
        if any(np.array_equal(mine["tokens"], t) for t in step_ref_tokens):
            return "layout：样本在这一步里，但分给了别的设备 / microbatch（样本到设备的分配约定）"
        if mine["tokens"].shape == ref["tokens"].shape and np.array_equal(mine["tokens"][..., 1:], ref["tokens"][..., :-1]):
            return "split：样本切分错了一格（起点偏移）"
        return "index/shuffle：这一步根本没有这个样本（索引重建或打乱不同）"
    if not np.array_equal(mine["labels"], ref["labels"]):
        if np.array_equal(mine["labels"][..., :-1], ref["labels"][..., :-1]):
            return "label：只有最后一个 label 不同（seq+1 取 label 的约定：多读 1 个 token vs 末位补 pad）"
        return "label：labels 不是 tokens 左移一格（seq+1 约定不同）"
    if not np.array_equal(mine["loss_mask"], ref["loss_mask"]):
        diff = mine["loss_mask"] != ref["loss_mask"]
        lab = ref["labels"][diff]
        tok = ref["tokens"][diff]
        if pad is not None and (lab == pad).any():
            return "mask：label 为 pad id 的位置 loss_mask 不同（Megatron 把 label==pad 的位置屏蔽）"
        if (tok == eod).any():
            return "mask：EOD 位置的 loss_mask 不同（eod_mask_loss 约定）"
        return "mask：loss_mask 不同"
    if not np.array_equal(mine["position_ids"], ref["position_ids"]):
        return "position：position_ids 不同（EOD 后是否重置，reset_position_ids / reset_attention_mask）"
    return "dtype/shape：数值相同但 dtype 或 shape 不同"


# ---------------------------------------------------------------- 各项检查

def check_gpu_integrity(gpu: GpuBatches, steps, dp, ga):
    bad, n = [], 0
    for s in steps:
        for r in range(dp):
            for m in range(ga):
                key = (s, r, m)
                if key not in gpu.idx:
                    bad.append(f"{s}.r{r}.m{m}: 缺整批")
                    continue
                arrs = gpu.get(*key)
                for k in GPU_FIELDS:
                    h = sha(np.ascontiguousarray(arrs[k]).tobytes())
                    n += 1
                    if h != gpu.idx[key][k]["sha256"]:
                        bad.append(f"{s}.r{r}.m{m}.{k}: 与 batches 索引不符")
                    fp = gpu.fp.get(key)
                    if fp is not None and fp.get(k) != h:
                        bad.append(f"{s}.r{r}.m{m}.{k}: 与数据指纹不符")
    return {"id": "L0.native.gpu_integrity", "type": "artifact_check", "level": "L0", "hard": True,
            "status": "pass" if not bad else "error", "threshold": {"kind": "exact"},
            "metrics": {"arrays": n, "bad": len(bad), "fingerprint_units": len(gpu.fp)},
            "worst": [{"name": b} for b in bad[:10]],
            "message": "GPU 整批输入与其 sha256 索引、数据指纹全部一致" if not bad else f"GPU 参照本身有 {len(bad)} 处对不上，不能当参照"}


def find_megatron_index(cache_dir: Path, want: dict):
    """按 description 找 Megatron 缓存的索引三件套：数据前缀、样本数、种子、序列长度、split 都对上的那一组。"""
    hits = []
    for desc in sorted(Path(cache_dir).glob("*-GPTDataset-train-description.txt")):
        d = json.loads(desc.read_text())
        if all(d.get(k) == v for k, v in want.items()):
            hits.append((desc.name.split("-")[0], d))
    return hits


def index_manifest(paths: dict) -> dict:
    out = {}
    for kind, p in paths.items():
        a = np.load(p, mmap_mode="r")
        out[kind] = {"file": str(p), "sha256_file": sha_file(p), "dtype": str(a.dtype), "shape": list(a.shape),
                     "sha256_array": sha(np.ascontiguousarray(a).tobytes())}
    return out


def check_index(meg_dir: Path, want: dict, mt_npy: dict):
    hits = find_megatron_index(meg_dir, want)
    res = {"id": "L0.native.index", "type": "artifact_check", "level": "L0", "hard": True,
           "threshold": {"kind": "exact"}}
    if len(hits) != 1:
        res.update(status="error", message=f"Megatron 缓存里按 {want} 找到 {len(hits)} 组索引，要正好 1 组")
        return res, None
    h, desc = hits[0]
    meg = index_manifest({k: Path(meg_dir) / f"{h}-GPTDataset-train-{k}.npy" for k in INDEX_KINDS})
    mt = mt_npy
    rows, differ = [], []
    for k in INDEX_KINDS:
        a, b = meg[k], mt.get(k)
        same_file = b is not None and a["sha256_file"] == b["sha256_file"]
        same_arr = b is not None and a["sha256_array"] == b["sha256_array"]
        rows.append({"name": k, "megatron": a, "maxtext": b, "same_file": same_file, "same_array": same_arr})
        if not same_arr:
            differ.append(k)
    res.update(status="pass" if not differ else "fail",
               metrics={"megatron_hash": h, "num_samples": desc["num_samples"], "same_array": 3 - len(differ),
                        "same_file": sum(r["same_file"] for r in rows)},
               details_inline={"indices": rows, "megatron_description": desc},
               message="三个索引两边重建结果逐字节相同（文件 sha256 也相同）" if not differ
               else f"索引不同：{differ}（差异出在索引重建：文档打乱 / 样本切分 / 样本打乱）")
    return res, rows


def unit_shas(b: dict, perm, dp, mbs, ga, step, gpu_dtypes: dict, gpu_shapes: dict) -> dict:
    """MaxText 一步的全局批 → {「步.r<d>.m<m>.<字段>」: sha256}，按 GPU 的 dtype / shape 换算后算。"""
    bp = {k: v[perm] for k, v in b.items()}
    dev = {k: per_device(v, dp, mbs, ga) for k, v in bp.items()}
    out = {}
    for d in range(dp):
        for m in range(ga):
            mine = to_gpu_fields({k: dev[k][(d, m)] for k in MT_FIELDS})
            for k in GPU_FIELDS:
                x = np.ascontiguousarray(mine[k].astype(gpu_dtypes[k]).reshape(gpu_shapes[k]))
                out[f"{step}.r{d}.m{m}.{k}"] = sha(x.tobytes())
    return out


def check_hosts(dumps, gpu: GpuBatches, steps, dp, mbs, ga, gbs):
    rows, bad = [], []
    first = gpu.idx[(steps[0], 0, 0)]
    dts = {k: np.dtype(first[k]["dtype"]) for k in GPU_FIELDS}
    shp = {k: first[k]["shape"] for k in GPU_FIELDS}
    for dpath in dumps:
        vm = MtBatches(dpath)
        H = int(vm.meta.get("emulated_hosts") or vm.meta.get("process_count") or 1)
        perm1 = maxtext_perm_for_megatron(dp, mbs, ga)          # 单 host 的换序：多 host 时用它会错
        permH = maxtext_perm_for_megatron(dp, mbs, ga, H)
        slot_ok = n_units = same1 = sameH = 0
        for s in steps:
            b = vm.get(s)
            refs_tok = {(r, m): gpu.get(s, r, m)["tokens"] for r in range(dp) for m in range(ga)}
            for g in range(gbs):       # 读进来的第 g 行确实是步内第 maxtext_loaded_slot(g) 个样本
                t = maxtext_loaded_slot(g, gbs, H)
                slot_ok += int(np.array_equal(b["inputs"][g].astype(np.int64), refs_tok[((t // mbs) % dp, t // (dp * mbs))][t % mbs]))
            u1, uH = unit_shas(b, perm1, dp, mbs, ga, s, dts, shp), unit_shas(b, permH, dp, mbs, ga, s, dts, shp)
            for unit, h in uH.items():
                st, r, m, k = unit.split(".")
                ref = gpu.fp.get((s, int(r[1:]), int(m[1:])), {}).get(k) or gpu.idx[(s, int(r[1:]), int(m[1:]))][k]["sha256"]
                n_units += 1
                same1 += int(u1[unit] == ref)
                sameH += int(h == ref)
        rows.append({"dump": str(dpath), "hosts": H, "steps": len(steps), "rows_slot_ok": slot_ok, "rows": len(steps) * gbs,
                     "units": n_units, "same_with_single_host_perm": same1, "same_with_host_aware_perm": sameH})
        if sameH != n_units or slot_ok != len(steps) * gbs:
            bad.append(f"{H} host：换序后 {n_units - sameH} 个单元不同，{len(steps) * gbs - slot_ok} 行不在预期的流位置")
    msg = ("；".join(f"{r['hosts']} host：读进来的行按 host 交错（{r['rows_slot_ok']}/{r['rows']} 行符合 host 切分公式），"
                    f"用单 host 换序只对上 {r['same_with_single_host_perm']}/{r['units']}，按 host 数换序后 {r['same_with_host_aware_perm']}/{r['units']}"
                    for r in rows))
    return {"id": "L0.native.hosts", "type": "data_fingerprint", "level": "L0", "hard": False,
            "status": "pass" if not bad else "fail", "threshold": {"kind": "exact"},
            "metrics": {"variants": rows,
                        "emulation": "单进程里对每个 host 调 MaxText 的 get_datasets(host_index=h, host_count=H) + _mmap_pretrain_pipeline，"
                                     "本地批按 host 拼成全局批（假定 host h 的设备在 mesh 批维上连续）；没有跑真的多进程 / MultiHostDataLoadIterator"},
            "worst": [{"name": x} for x in bad], "message": msg if not bad else "；".join(bad)}


def check_epochs(pairs):
    rows, bad = [], []
    for meg_dir, mt_dir in pairs:
        M, T = json.load(open(Path(meg_dir) / "ref.json")), json.load(open(Path(mt_dir) / "ref.json"))
        ti = {}
        for f, v in T["index"].items():
            for k in INDEX_KINDS:
                if f.endswith(f"-{k}.npy"):
                    ti[k] = v
        same_idx = {k: ti.get(k, {}).get("sha256_file") == M["index"][k]["sha256_file"] for k in INDEX_KINDS}
        a, b = np.load(Path(meg_dir) / "samples.npz"), np.load(Path(mt_dir) / "samples.npz")
        diff = sorted(k for k in a.files if k not in b.files or not np.array_equal(a[k], b[k]))
        row = {"num_samples": M["num_samples"], "num_epochs": M["num_epochs"], "separate_final_epoch": M.get("separate_final_epoch"),
               "len_megatron": M["len"], "len_maxtext": T["len"], "index_same": same_idx,
               "index_sha256": {k: M["index"][k]["sha256_file"] for k in INDEX_KINDS},
               "cross_epoch": M["cross_epoch"], "samples_compared": len(M["picks"]), "arrays_differ": diff[:10]}
        rows.append(row)
        if not all(same_idx.values()) or diff or M["len"] != T["len"]:
            bad.append(f"{M['num_samples']} 样本：索引相同 {same_idx}，样本不同 {len(diff)} 个")
    msg = "；".join(f"{r['num_samples']} 个样本（{r['num_epochs']} 个 epoch"
                   + ("，最后一个 epoch 单独打乱" if r["separate_final_epoch"] else "")
                   + f"）：三个索引相同，{r['samples_compared']} 个样本（含跨 epoch 边界的 {len(r['cross_epoch'])} 个）逐位相同"
                   for r in rows)
    return {"id": "L0.native.epochs", "type": "data_fingerprint", "level": "L0", "hard": True,
            "status": "pass" if not bad else "fail", "threshold": {"kind": "exact"}, "metrics": {"cases": rows},
            "message": msg if not bad else "；".join(bad)}


def check_eod_cases(meg_dir, mt_dir):
    M = json.load(open(Path(meg_dir) / "ref.json"))
    a, b = np.load(Path(meg_dir) / "samples.npz"), np.load(Path(mt_dir) / "samples.npz")
    pad, keys = M["pad"], ("tokens", "labels", "loss_mask", "position_ids", "attention_mask")
    rows, eod_bad, pad_rows = [], [], []
    for c in M["combos"]:
        ram, rpi, eml = c["reset_attention_mask"], c["reset_position_ids"], c["eod_mask_loss"]
        mt_name = f"ram{int(ram)}_eml{int(eml)}"
        row = {"megatron": c["name"], "maxtext": mt_name if ram == rpi else None, "samples": c["len"]}
        if ram != rpi:
            # MaxText 只有 reset_attention_mask 一个开关同时管注意力和位置；拿两种设置各比一次，说明哪一项对不上
            row["maxtext"] = None
            row["note"] = "MaxText 表达不了：reset_attention_mask 同时决定注意力分段与位置重置"
            for alt in (f"ram{int(ram)}_eml{int(eml)}", f"ram{int(rpi)}_eml{int(eml)}"):
                no_pad = [i for i in range(c["len"]) if not ((b[f"{alt}.{i}.tokens"] == pad).any() or (b[f"{alt}.{i}.labels"] == pad).any())]
                row[f"differ_vs_{alt}"] = sorted({k for i in no_pad for k in keys      # 只看不含 pad id 的样本，免得混进 pad 的差异
                                                 if not np.array_equal(a[f"{c['name']}.{i}.{k}"],
                                                                       b[f"{alt}.{i}.{k}"].reshape(a[f"{c['name']}.{i}.{k}"].shape))})
            rows.append(row)
            continue
        clean = pads = 0
        for i in range(c["len"]):
            has_pad = bool((b[f"{mt_name}.{i}.tokens"] == pad).any() or (b[f"{mt_name}.{i}.labels"] == pad).any())
            diff = [k for k in keys if not np.array_equal(a[f"{c['name']}.{i}.{k}"],
                                                          b[f"{mt_name}.{i}.{k}"].reshape(a[f"{c['name']}.{i}.{k}"].shape))]
            if has_pad:
                pads += 1
                pad_rows.append({"combo": c["name"], "sample": i, "fields_differ": diff})
            elif diff:
                eod_bad.append(f"{c['name']} 样本 {i}：{diff}")
            else:
                clean += 1
        row.update({"samples_without_pad_same": clean, "samples_with_pad": pads})
        rows.append(row)
    n_repr = sum(1 for r in rows if r["maxtext"])
    eod = {"id": "L0.native.eod_cases", "type": "data_fingerprint", "level": "L0", "hard": True,
           "status": "pass" if not eod_bad else "fail", "threshold": {"kind": "exact"},
           "metrics": {"doc_lens": M["doc_lens"], "seq": M["seq"], "combos": rows,
                       "synthetic_idx_sha256": M["idx_sha256"], "synthetic_bin_sha256": M["bin_sha256"]},
           "worst": [{"name": x} for x in eod_bad[:10]],
           "message": (f"合成数据（{len(M['doc_lens'])} 篇文档，含 1 token 空文档、恰好 seq / seq+1 长、跨两个样本的长文档，序列 {M['seq']}，2 个 epoch）："
                       f"MaxText 能表达的 {n_repr} 种组合（reset_attention_mask = reset_position_ids）下，不含 pad id 的样本 "
                       f"tokens / labels / loss_mask / position_ids / 注意力 mask 全部逐位相同；"
                       f"reset_attention_mask ≠ reset_position_ids 的 {len(rows) - n_repr} 种组合 MaxText 表达不了（位置重置跟着注意力走）")
           if not eod_bad else "；".join(eod_bad)}
    pad_fields = sorted({f for r in pad_rows for f in r["fields_differ"]})
    padc = {"id": "L0.native.pad_id", "type": "data_fingerprint", "level": "L0", "hard": False,
            "status": "pass" if not any(r["fields_differ"] for r in pad_rows) else "fail", "threshold": {"kind": "exact"},
            "metrics": {"pad_id": pad, "cases": pad_rows},
            "message": ("含 pad id 的样本两边一致" if not pad_fields else
                        f"含 pad id 的样本在 {n_repr} 种组合下共 {len(pad_rows)} 处两边不同（字段 {pad_fields}）：Megatron GPTDataset 把 tokens / labels 里的 pad id "
                        f"换成 0、label 为 pad id 的位置 loss_mask 置 0，MaxText mmap_npy 原样保留、照常算 loss。"
                        f"只在数据里出现 pad id 时触发（见 L0.native.fields 的计数）")}
    return eod, padc


def bin_dtype(bin_path: str):
    """Megatron .idx 头：9 字节魔数 + 8 字节版本 + 1 字节 dtype 代码（megatron/core/datasets/indexed_dataset.py 的 DType）。"""
    codes = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float64, 7: np.float32, 8: np.uint16}
    with open(str(bin_path)[:-4] + ".idx", "rb") as f:
        head = f.read(18)
    if head[:9] != b"MMIDIDX\x00\x00":
        raise ValueError(f"不是 Megatron .idx：{head[:9]!r}")
    return codes[head[17]]


def check_negative(controls, gpu: GpuBatches, meg_index_dir, want, dp, mbs, ga, gbs, eod, pad):
    """controls：[(名字, 只存了前几步的 dump 目录)]。每个反例都应当被发现（索引不同或样本不同），并说清差在哪一步。"""
    rows, missed = [], []
    hits = find_megatron_index(meg_index_dir, want)
    meg = index_manifest({k: Path(meg_index_dir) / f"{hits[0][0]}-GPTDataset-train-{k}.npy" for k in INDEX_KINDS}) if len(hits) == 1 else {}
    for name, d in controls:
        vm = MtBatches(d)
        H = int(vm.meta.get("emulated_hosts") or vm.meta.get("process_count") or 1)
        perm = maxtext_perm_for_megatron(dp, mbs, ga, H)
        idx_same = {}
        for f, v in vm.meta["index_files"].items():
            for k in INDEX_KINDS:
                if f.endswith(f"-{k}.npy") and k in meg:
                    idx_same[k] = v["sha256_file"] == meg[k]["sha256_file"]
        where, units, same = {}, 0, 0
        for s in vm.steps():
            b = vm.get(s)
            refs = {(r, m): gpu.get(s, r, m) for r in range(dp) for m in range(ga)}
            dev = {k: per_device(v[perm], dp, mbs, ga) for k, v in b.items()}
            for (r, m), ref in refs.items():
                mine = to_gpu_fields({k: dev[k][(r, m)] for k in MT_FIELDS}, ref)
                mine = {k: v.reshape(ref[k].shape) for k, v in mine.items()}
                units += 1
                if all(np.array_equal(mine[k], ref[k]) for k in GPU_FIELDS):
                    same += 1
                else:
                    why = localize(mine, ref, [x["tokens"] for x in refs.values()], eod, pad)
                    where[why] = where.get(why, 0) + 1
        eff = vm.meta["effective_config"]
        row = {"name": name, "steps": len(vm.steps()), "units": units, "same": same, "index_same": idx_same,
               "localized": where, "config": {k: eff.get(k) for k in ("reset_attention_mask", "packing_max_segments_per_sample",
                                                                      "mmap_npy_split", "data_shuffle_seed")}}
        rows.append(row)
        if same == units and all(idx_same.values()):
            missed.append(name)
    return {"id": "L0.native.negative_controls", "type": "poison_selftest", "level": "L0", "hard": False,
            "status": "pass" if not missed else "fail", "threshold": {"kind": "exact"}, "metrics": {"controls": rows},
            "message": ("；".join(f"{r['name']}：{r['units'] - r['same']}/{r['units']} 个单元不同"
                                 + ("，索引不同" if not all(r["index_same"].values()) else "")
                                 + "（" + "、".join(k.split("：")[0] for k in r["localized"]) + "）" for r in rows))
            if not missed else f"反例没被发现：{missed}"}


def run(mt_dir, gpu_run, meg_index_dir, out, variants=(), steps=None, pad_id=None, hosts=(), epochs=(), eod_ref=None,
        full_pad_count=None, negatives=()):
    t0 = datetime.datetime.now().astimezone()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    gmeta = json.load(open(Path(gpu_run) / "meta.json"))
    args = gmeta["config"]["args"]
    dp, mbs, gbs, seq = gmeta["parallelism"]["dp"], args["mbs"], args["gbs"], args["seq"]
    ga = gbs // (dp * mbs)
    mt = MtBatches(mt_dir)
    eff = mt.meta["effective_config"]
    eod = int(eff["mmap_eod_id"])
    pad = pad_id
    gpu = GpuBatches(gpu_run, dp)
    all_steps = sorted(set(gpu.steps()) & set(mt.steps()))
    if steps:
        lo, hi = (int(x) for x in steps.split(":"))
        all_steps = [s for s in all_steps if lo <= s <= hi]
    expect_steps = list(range(1, args["steps"] + 1))
    checks = []

    # 0. 参照可信
    checks.append(check_gpu_integrity(gpu, all_steps, dp, ga))

    # 1. 索引
    want = {"dataset_path": args["data"], "num_samples": args["steps"] * gbs, "random_seed": args["seed"],
            "sequence_length": seq}
    mt_npy = {}
    for f, v in mt.meta["index_files"].items():
        for k in INDEX_KINDS:
            if f.endswith(f"-{k}.npy"):
                mt_npy[k] = {"file": f, **v}
    idx_res, _ = check_index(Path(meg_index_dir), want, mt_npy)
    checks.append(idx_res)

    # 2–5. 逐步
    H = int(mt.meta.get("emulated_hosts") or mt.meta.get("process_count") or 1)
    perm = maxtext_perm_for_megatron(dp, mbs, ga, H)
    stream_bad, native_same, native_total = [], 0, 0
    test_fp, ref_fp, elem_bad, where = {}, {}, [], {}
    nonpad_bad, nonpad_rows = [], []
    eod_count = pad_count_mt = pad_count_gpu = lm_zero_gpu = 0
    seg_not_one = pos_mismatch = tpos_mismatch = 0
    dtype_seen = {}
    for s in all_steps:
        b = mt.get(s)
        refs = {(r, m): gpu.get(s, r, m) for r in range(dp) for m in range(ga)}
        # 步内样本流：Megatron 步内第 i 个 = rank (i // mbs) % dp、microbatch i // (dp*mbs)
        stream_ref = [refs[((i // mbs) % dp, i // (dp * mbs))]["tokens"][i % mbs] for i in range(gbs)]
        step_tokens = [refs[k]["tokens"] for k in refs]
        for g in range(gbs):
            if not np.array_equal(b["inputs"][g].astype(np.int64), stream_ref[maxtext_loaded_slot(g, gbs, H)]):
                stream_bad.append(f"{s}.row{g}")
        # 原样布局
        native = per_device(b["inputs"], dp, mbs, ga)
        for (d, m), x in native.items():
            native_total += 1
            native_same += int(np.array_equal(x.astype(np.int64), refs[(d, m)]["tokens"]))
        # 换序之后
        bp = {k: v[perm] for k, v in b.items()}
        dev = {k: per_device(v, dp, mbs, ga) for k, v in bp.items()}
        np_mt = 0
        for d in range(dp):
            for m in range(ga):
                ref = refs[(d, m)]
                mine = to_gpu_fields({k: dev[k][(d, m)] for k in MT_FIELDS}, ref)
                for k in GPU_FIELDS:
                    mine[k] = mine[k].reshape(ref[k].shape)
                    unit = f"{s}.r{d}.m{m}.{k}"
                    test_fp[unit] = sha(np.ascontiguousarray(mine[k]).tobytes())
                    ref_fp[unit] = gpu.fp.get((s, d, m), {}).get(k) or gpu.idx[(s, d, m)][k]["sha256"]
                    dtype_seen.setdefault(k, [str(ref[k].dtype), list(ref[k].shape)])
                bad = [k for k in GPU_FIELDS if not np.array_equal(mine[k], refs[(d, m)][k])]
                if bad:
                    elem_bad.append(f"{s}.r{d}.m{m}:{','.join(bad)}")
                    where.setdefault(localize(mine, ref, step_tokens, eod, pad), []).append(f"{s}.r{d}.m{m}")
                np_mt += int(mine["loss_mask"].sum())
                eod_count += int((ref["tokens"] == eod).sum())
                lm_zero_gpu += int((ref["loss_mask"] == 0).sum())
                if pad is not None:
                    pad_count_gpu += int((ref["tokens"] == pad).sum() + (ref["labels"] == pad).sum())
        if pad is not None:
            pad_count_mt += int((b["inputs"] == pad).sum() + (b["targets"] == pad).sum())
        seg_not_one += int((b["inputs_segmentation"] != 1).sum())
        pos_mismatch += int((b["inputs_position"] != np.arange(seq, dtype=b["inputs_position"].dtype)).sum())
        tpos_mismatch += int((b["targets_position"] != b["inputs_position"]).sum())
        np_gpu = sum(int(gpu.fp[(s, r, m)]["nonpad"]) if (s, r, m) in gpu.fp and "nonpad" in gpu.fp[(s, r, m)]
                     else int(refs[(r, m)]["loss_mask"].sum()) for r in range(dp) for m in range(ga))
        nonpad_rows.append((s, np_mt, np_gpu))
        if np_mt != np_gpu:
            nonpad_bad.append(f"{s}: {np_mt} vs {np_gpu}")
    mt_dtypes = {k: [mt.idx[all_steps[0]][k]["dtype"], mt.idx[all_steps[0]][k]["shape"]] for k in MT_FIELDS}

    checks.append({"id": "L0.native.stream", "type": "data_fingerprint", "level": "L0", "hard": True,
                   "status": "pass" if not stream_bad else "fail", "threshold": {"kind": "exact"},
                   "metrics": {"steps": len(all_steps), "rows": len(all_steps) * gbs, "differ": len(stream_bad)},
                   "worst": [{"name": x} for x in stream_bad[:10]],
                   "message": f"每步 {gbs} 个样本按顺序与 Megatron 的样本流逐位相同" if not stream_bad
                   else f"{len(stream_bad)} 行与 Megatron 样本流不同（差异在打乱 / 样本切分，与设备分配无关）"})
    checks.append({"id": "L0.native.device_map.native", "type": "data_fingerprint", "level": "L0", "hard": False,
                   "status": "pass" if native_same == native_total else "fail", "threshold": {"kind": "exact"},
                   "metrics": {"units": native_total, "same": native_same, "dp": dp, "mbs": mbs, "ga": ga},
                   "message": "MaxText 原样的设备 / microbatch 布局与 Megatron rank / microbatch 一致" if native_same == native_total
                   else (f"原样布局只有 {native_same} / {native_total} 个（设备 · microbatch）单元与 GPU 同号 rank 相同："
                         f"MaxText 梯度累积按 reshape(gbs//ga, ga) 交错取行，Megatron 每个 microbatch 取连续 dp×mbs 个；"
                         f"样本本身没错（见 L0.native.stream），差在分配约定")})

    fp_res = REGISTRY["data_fingerprint"]("L0.native.fingerprint", "L0", test_fp, ref_fp, fields=list(GPU_FIELDS)).to_json()
    fp_res["metrics"].update({"elementwise_differ": len(elem_bad), "adapter": "janus.data_layout.maxtext_perm_for_megatron",
                              "perm": perm.tolist()})
    if elem_bad and fp_res["status"] == "pass":
        fp_res["status"] = "fail"
    if where:
        fp_res["details_inline"] = {"localized": {k: {"count": len(v), "first": v[:5]} for k, v in where.items()}}
        fp_res["message"] = "；".join(f"{k}（{len(v)} 处）" for k, v in where.items())
    else:
        fp_res["message"] = (f"换序后 {len(all_steps)} 步 × {dp} 个设备 × {ga} 个 microbatch 的 tokens / labels / loss_mask / "
                             f"position_ids 全部逐元素相同，sha256 与 GPU 数据指纹相同")
    checks.append(fp_res)

    checks.append({"id": "L0.native.nonpad_total", "type": "data_fingerprint", "level": "L0", "hard": True,
                   "status": "pass" if not nonpad_bad else "fail", "threshold": {"kind": "exact"},
                   "metrics": {"steps": len(nonpad_rows), "differ": len(nonpad_bad),
                               "min": min(r[1] for r in nonpad_rows), "max": max(r[1] for r in nonpad_rows),
                               "expected_full": gbs * seq},
                   "worst": [{"name": x} for x in nonpad_bad[:10]],
                   "message": "每步全局非 padding token 数逐步相等" if not nonpad_bad else f"{len(nonpad_bad)} 步不等"})

    conv_bad = []
    if seg_not_one:
        conv_bad.append(f"inputs_segmentation 有 {seg_not_one} 处不是 1（GPU reset_attention_mask=False 时整条序列是一个因果段）")
    if pos_mismatch:
        conv_bad.append(f"inputs_position 有 {pos_mismatch} 处不是 0..L-1（GPU reset_position_ids=False）")
    if tpos_mismatch:
        conv_bad.append(f"targets_position 与 inputs_position 有 {tpos_mismatch} 处不同")
    checks.append({"id": "L0.native.fields", "type": "artifact_check", "level": "L0", "hard": True,
                   "status": "pass" if not conv_bad else "fail", "threshold": {"kind": "exact"},
                   "metrics": {"eod_tokens_in_compared_inputs": eod_count, "gpu_loss_mask_zeros": lm_zero_gpu,
                               "pad_id": pad, "pad_occurrences_gpu": pad_count_gpu if pad is not None else None,
                               "pad_occurrences_maxtext": pad_count_mt if pad is not None else None,
                               "full_bin": full_pad_count,
                               "inputs_segmentation_not_1": seg_not_one, "inputs_position_not_arange": pos_mismatch,
                               "targets_position_ne_inputs_position": tpos_mismatch},
                   "details_inline": {"dtype_shape": {
                       "gpu": dtype_seen, "maxtext_global_batch": mt_dtypes,
                       "mapping": {"tokens": "inputs", "labels": "targets", "loss_mask": "targets_segmentation != 0",
                                   "position_ids": "inputs_position",
                                   "attention": "inputs_segmentation（全 1 = 整条序列一个因果段 = GPU reset_attention_mask=False）"}}},
                   "message": ("字段约定一致：MaxText 用 int32、按全局批 [gbs, L] 存，GPU 用 int64 / float32、按 [mbs, L] 存；"
                               "按 inputs→tokens、targets→labels、targets_segmentation≠0→loss_mask、inputs_position→position_ids "
                               "换算后数值逐位相同；inputs_segmentation 全 1（整条序列一个因果段）、位置 0..L-1，与 GPU 配置一致")
                   if not conv_bad else "；".join(conv_bad)})

    # 读取进程数
    var_rows, var_bad = [], []
    for v in variants:
        vm = MtBatches(v)
        n_same = n = 0
        for s in all_steps:
            if s not in vm.idx:
                var_bad.append(f"{Path(v).name}: 缺第 {s} 步")
                continue
            n += 1
            n_same += int(all(vm.idx[s][k]["sha256"] == mt.idx[s][k]["sha256"] for k in MT_FIELDS))
        var_rows.append({"dump": str(v), "grain_worker_count": vm.meta["effective_config"]["grain_worker_count"],
                         "steps": n, "same": n_same})
        if n_same != n:
            var_bad.append(f"{Path(v).name}: {n - n_same} 步不同")
    if variants:
        checks.append({"id": "L0.native.readers", "type": "determinism", "level": "L0", "hard": True,
                       "status": "pass" if not var_bad else "fail", "threshold": {"kind": "exact"},
                       "metrics": {"base_grain_worker_count": eff["grain_worker_count"], "variants": var_rows},
                       "worst": [{"name": x} for x in var_bad[:10]],
                       "message": ("grain_worker_count " + " / ".join(str(r["grain_worker_count"]) for r in var_rows)
                                   + f" 与 {eff['grain_worker_count']} 逐步逐位相同") if not var_bad else "；".join(var_bad)})

    if hosts:
        checks.append(check_hosts(hosts, gpu, all_steps, dp, mbs, ga, gbs))
    if epochs:
        checks.append(check_epochs(epochs))
    if eod_ref:
        checks.extend(check_eod_cases(*eod_ref))
    if negatives:
        checks.append(check_negative(negatives, gpu, Path(meg_index_dir), want, dp, mbs, ga, gbs, eod, pad))

    exp_units = [f"{s}.r{d}.m{m}.{k}" for s in expect_steps for d in range(dp) for m in range(ga) for k in GPU_FIELDS]
    cov = REGISTRY["coverage"]("L0.native.coverage", "L0", exp_units, sorted(test_fp)).to_json()
    checks.append(cov)

    summary = {k: sum(1 for c in checks if c["status"] == k) for k in ("pass", "fail", "inconclusive", "error")}
    summary["hard_fail"] = any(c["status"] in ("fail", "error") and c.get("hard") for c in checks)
    t1 = datetime.datetime.now().astimezone()
    res = {"schema_version": 1, "stage": "phase2-native-loader",
           "runs": {"gpu": {"path": str(gpu_run), "meta_sha256": sha_file(Path(gpu_run) / "meta.json")},
                    "maxtext": {"path": str(mt_dir), "meta_sha256": sha_file(Path(mt_dir) / "meta.json")}},
           "maxtext": {"commit": mt.meta["maxtext"]["commit"], "versions": mt.meta["versions"],
                       "overrides": mt.meta["overrides"], "mesh": mt.meta["mesh"], "rows_of_device": mt.meta["rows_of_device"]},
           "gpu": {"dp": dp, "mbs": mbs, "gbs": gbs, "ga": ga, "seq": seq, "seed": args["seed"], "data": args["data"],
                   "data_idx_sha256": gmeta["data"]["manifest_sha256"]},
           "data_idx_sha256_maxtext": mt.meta["data"]["idx_sha256"],
           "steps": {"first": all_steps[0], "last": all_steps[-1], "count": len(all_steps)},
           "started_at": t0.isoformat(timespec="seconds"), "finished_at": t1.isoformat(timespec="seconds"),
           "summary": summary, "checks": checks}
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    (out / "report.md").write_text(render(res))
    print(json.dumps(summary, ensure_ascii=False))
    return res


def render(res: dict) -> str:
    g, m = res["gpu"], res["maxtext"]
    c = {x["id"]: x for x in res["checks"]}
    lines = ["# TPU 原生数据加载器对拍（MaxText Grain mmap_npy ↔ Megatron GPTDataset）", "",
             f"GPU 参照 `{res['runs']['gpu']['path']}`（DP {g['dp']}、mbs {g['mbs']}、全局 batch {g['gbs']}、"
             f"每 rank {g['ga']} 个 microbatch、序列 {g['seq']}、种子 {g['seed']}）；"
             f"MaxText `{m['commit'][:12]}`，CPU 上 {len(m['rows_of_device'])} 个设备；"
             f"第 {res['steps']['first']}–{res['steps']['last']} 步，共 {res['steps']['count']} 步。", "",
             f"数据 `{g['data']}`：`.idx` sha256 两边都是 `{g['data_idx_sha256'][:16]}…`"
             + ("" if g["data_idx_sha256"] == res["data_idx_sha256_maxtext"] else "（**不同**）") + "。", "",
             "| 检查 | 状态 | 说明 |", "|---|---|---|"]
    lines += [f"| `{x['id']}` | {x['status']} | {x.get('message', '')} |" for x in res["checks"]]
    if "L0.native.index" in c and "details_inline" in c["L0.native.index"]:
        lines += ["", "## 索引 sha256（文件）", "", "| 索引 | dtype / shape | Megatron | MaxText | 相同 |", "|---|---|---|---|---|"]
        for r in c["L0.native.index"]["details_inline"]["indices"]:
            a, b = r["megatron"], r["maxtext"] or {}
            lines.append(f"| {r['name']} | {a['dtype']} {a['shape']} | `{a['sha256_file'][:16]}…` | "
                         f"`{b.get('sha256_file', '—')[:16]}…` | {'是' if r['same_file'] else '否'} |")
    fpc = c.get("L0.native.fingerprint", {})
    perm = fpc.get("metrics", {}).get("perm")
    if perm:
        lines += ["", "## 设备与 rank 的对应", "",
                  f"MaxText 原样：设备 d 的第 m 个 microbatch 是全局批第 d×ga+m 行；Megatron：rank d 的第 m 个 microbatch 是步内第 m×dp+d 个样本。"
                  f"换序 `batch[perm]`，perm = `{perm}`（`janus.data_layout.maxtext_perm_for_megatron`）。"]
    fl = c.get("L0.native.fields", {}).get("metrics", {})
    if fl:
        lines += ["", "## 边界与约定计数", "",
                  f"- 比到的输入里 EOD 共 {fl['eod_tokens_in_compared_inputs']} 个（跨文档边界确实被覆盖）；"
                  f"GPU loss_mask 为 0 的位置 {fl['gpu_loss_mask_zeros']} 个",
                  f"- pad id {fl['pad_id']} 在比到的样本里出现：GPU {fl['pad_occurrences_gpu']} 次、MaxText {fl['pad_occurrences_maxtext']} 次"
                  + (f"；整份 .bin（{fl['full_bin']['documents']} 篇文档、{fl['full_bin']['tokens']} 个 token、EOD {fl['full_bin']['eod']} 个）里 "
                     f"{fl['full_bin']['pad']} 次" if fl.get("full_bin") else "")
                  + "（两边对 pad id 的处理不同，见 `L0.native.pad_id`；出现次数为 0 时这条差异不触发）"]
    h = c.get("L0.native.hosts", {}).get("metrics", {}).get("variants")
    if h:
        lines += ["", "## 多 host（模拟）", "", "| host 数 | 行符合 host 切分公式 | 单 host 换序对上 | 按 host 数换序对上 |", "|---|---|---|---|"]
        lines += [f"| {r['hosts']} | {r['rows_slot_ok']} / {r['rows']} | {r['same_with_single_host_perm']} / {r['units']} | "
                  f"{r['same_with_host_aware_perm']} / {r['units']} |" for r in h]
        lines += ["", c["L0.native.hosts"]["metrics"]["emulation"] + "。"]
    e = c.get("L0.native.epochs", {}).get("metrics", {}).get("cases")
    if e:
        lines += ["", "## 跨 epoch（两边各自现建索引）", "",
                  "| 样本数 | epoch | 最后一个 epoch 单独打乱 | 样本总数 | 三个索引相同 | 比的样本 | 跨 epoch 边界的流下标 |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {r['num_samples']} | {r['num_epochs']} | {'是' if r['separate_final_epoch'] else '否'} | {r['len_megatron']} | "
                  f"{'是' if all(r['index_same'].values()) else '否'} | {r['samples_compared']} | "
                  f"{', '.join(str(x['stream_index']) for x in r['cross_epoch']) or '—'} |" for r in e]
    ec = c.get("L0.native.eod_cases", {}).get("metrics", {}).get("combos")
    if ec:
        lines += ["", "## 合成边界用例（EOD / 位置 / 注意力 / loss mask）", "",
                  f"文档长度 `{c['L0.native.eod_cases']['metrics']['doc_lens']}`（每篇末尾是 EOD），序列 {c['L0.native.eod_cases']['metrics']['seq']}。", "",
                  "| Megatron（ram / rpi / eml） | MaxText（ram / eml） | 不含 pad 的样本逐位相同 | 含 pad 的样本 | 说明 |", "|---|---|---|---|---|"]
        for r in ec:
            if r["maxtext"]:
                lines.append(f"| {r['megatron']} | {r['maxtext']} | {r['samples_without_pad_same']} / {r['samples'] - r['samples_with_pad']} | "
                             f"{r['samples_with_pad']} | |")
            else:
                alts = "；".join(f"不含 pad 的样本对 {k[10:]} 不同：{v}" for k, v in r.items() if k.startswith("differ_vs_"))
                lines.append(f"| {r['megatron']} | — | — | — | {r['note']}（{alts}） |")
    ng = c.get("L0.native.negative_controls", {}).get("metrics", {}).get("controls")
    if ng:
        lines += ["", "## 反例（故意配错，对拍必须报出来）", "", "| 反例 | 配置 | 前几步单元相同 | 索引相同 | 定位 |", "|---|---|---|---|---|"]
        for r in ng:
            cfg = ", ".join(f"{k}={v!r}" for k, v in r["config"].items())
            lines.append(f"| {r['name']} | {cfg} | {r['same']} / {r['units']} | "
                         f"{'是' if all(r['index_same'].values()) else '否'} | "
                         + "；".join(f"{k}（{v}）" for k, v in r["localized"].items()) + " |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mt", required=True)
    p.add_argument("--gpu-run", required=True)
    p.add_argument("--megatron-index-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mt-variant", action="append", default=[])
    p.add_argument("--steps", default=None)
    p.add_argument("--pad-id", type=int, default=None, help="分词器的 pad id（Megatron 会屏蔽 label 为它的位置），只用来计数")
    p.add_argument("--count-pad-in", default=None, help="数一下这个 .bin 全量里 pad id 出现几次（dtype 读同名 .idx 的头）")
    p.add_argument("--mt-hosts", action="append", default=[])
    p.add_argument("--epochs-ref", action="append", default=[], help="<megatron 参照目录>:<maxtext 参照目录>")
    p.add_argument("--eod-ref", default=None, help="<megatron 合成参照目录>:<maxtext 合成参照目录>")
    p.add_argument("--negative", action="append", default=[], help="<名字>=<反例 dump 目录>")
    a = p.parse_args()
    full = None
    if a.count_pad_in and a.pad_id is not None:
        allt = np.memmap(a.count_pad_in, dtype=bin_dtype(a.count_pad_in), mode="r")
        eod_id = json.load(open(Path(a.mt) / "meta.json"))["effective_config"]["mmap_eod_id"]
        full = {"tokens": int(allt.size), "pad": int((allt == a.pad_id).sum()), "eod": int((allt == int(eod_id)).sum()),
                # .idx 头第 18–26 字节是序列数；没按句切分（mmap_split_sentences=false 已校验）时就是文档数
                "documents": int(np.frombuffer(open(a.count_pad_in[:-4] + ".idx", "rb").read(26)[18:26], "<i8")[0]),
                "max_token_id": int(allt.max())}
    res = run(a.mt, a.gpu_run, a.megatron_index_dir, a.out, a.mt_variant, a.steps, a.pad_id, a.mt_hosts,
              [tuple(x.split(":")) for x in a.epochs_ref], tuple(a.eod_ref.split(":")) if a.eod_ref else None, full,
              [tuple(x.split("=", 1)) for x in a.negative])
    sys.exit(1 if res["summary"]["hard_fail"] else 0)


if __name__ == "__main__":
    main()
