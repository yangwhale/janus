"""TPU 第 1 步梯度文件（`diag_grads.safetensors`）里那些「一段被写成 0」的张量：零段在哪、多长、有什么规律，
是 TPU 写盘时坏的还是取回本机时坏的。只读本机已有的运行目录，不上卡。

    python models/hunyuan4/analysis/tpu_grad_file_holes.py --run <运行目录> [--ref <同布局的干净运行目录>] \\
        [--fetch-log <取回日志>] [--gcs-uri <桶里同一文件>] [--committed <已提交的 step1 结果 JSON>] --out <json>

做的事：
1. 读 safetensors 文件头，得到每个张量在文件里的字节区间。
2. 用 `SEEK_HOLE` / `SEEK_DATA` 列出本机文件里的「空洞」（文件系统里从没写过、读出来是 0 的区间）。
   空洞只能出现在本机写文件的那一侧：TPU 写盘写的是 Pod 里的文件，本机的空洞说明取回时这段字节没有到达本机。
   空洞以文件系统块（4 KiB）为单位；每个空洞往前逐字节找到最后一个非零字节，给出字节级的零段起点。
3. 每个零段：起止偏移、长度、对 `--block`（默认 256 MiB，`run.sh` 并行分块取回的块大小）取模、落在第几块、
   覆盖了哪些张量的哪些元素。
4. 顺序读一遍整个文件：每个张量的平方和对设备上算的（`diag_step1.json` 的 `per_tensor.sq`），同时算整文件的
   sha256 与 crc32c。给了 `--gcs-uri` 就取桶里同一文件的 crc32c 比（只记是否相同，不记桶名）。
5. 给了 `--ref`（同一模型、同一批数据、文件布局逐字节相同的另一份干净运行）就读它在这些零段上的平方和，
   和本文件的平方和缺口（设备上算的 − 文件里的）比：两者量级相同，说明缺的就是零段里本该有的梯度。
6. 给了 `--fetch-log`，数这次运行取回期间 `kubectl exec` 流异常中断（websocket close 1006）的次数。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess

import numpy as np

CHUNK = 64 << 20
FILE_SQ_TOL = 1e-10  # 与 step1_vs_fp64.py 相同：相对差超过它记为损坏


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    base = 8 + n
    out = []
    for name, v in h.items():
        if v["dtype"] != "F32":
            raise SystemExit(f"{name} 不是 F32（{v['dtype']}），脚本只按 F32 读")
        s, e = v["data_offsets"]
        out.append({"tensor": name, "shape": v["shape"], "start": base + s, "end": base + e})
    out.sort(key=lambda t: t["start"])
    return out, base


def holes(path):
    fd = os.open(path, os.O_RDONLY)
    size = os.fstat(fd).st_size
    res, off = [], 0
    try:
        while off < size:
            h = os.lseek(fd, off, os.SEEK_HOLE)
            if h >= size:
                break
            try:
                d = os.lseek(fd, h, os.SEEK_DATA)
            except OSError:
                d = size
            res.append((h, d))
            off = d
    finally:
        os.close(fd)
    return res, size


def last_nonzero_before(f, pos, limit=1 << 20):
    """pos 之前最后一个非零字节的下一个位置（最多往前看 limit 字节）。"""
    lo = max(0, pos - limit)
    f.seek(lo)
    buf = np.frombuffer(f.read(pos - lo), np.uint8)
    nz = np.flatnonzero(buf)
    return lo + int(nz[-1]) + 1 if nz.size else lo


def pow2_align(x):
    return int(x & -x) if x else 0


def overlap(tensors, s, e):
    out = []
    for t in tensors:
        a, b = max(s, t["start"]), min(e, t["end"])
        if a < b:
            out.append(
                {
                    "tensor": t["tensor"],
                    "bytes": b - a,
                    "elem_first": (a - t["start"]) // 4,
                    "elem_last": (b - t["start"]) // 4 - 1,
                    "numel": (t["end"] - t["start"]) // 4,
                    "elems_frac": (b - a) / (t["end"] - t["start"]),
                }
            )
    return out


def sq_range(f, s, e):
    """[s, e) 字节区间（4 字节对齐到 F32 元素）的平方和。"""
    s4 = s - (s % 4)
    f.seek(s4)
    tot = 0.0
    while s4 < e:
        n = min(CHUNK, e - s4)
        n -= n % 4
        if n == 0:
            break
        x = np.frombuffer(f.read(n), np.float32).astype(np.float64)
        tot += float(np.dot(x, x))
        s4 += n
    return tot


def full_pass(path, tensors):
    """顺序读整个文件：每个张量的平方和、整文件 sha256 与 crc32c。"""
    import base64

    import google_crc32c

    sha, crc = hashlib.sha256(), google_crc32c.Checksum()
    sq = {t["tensor"]: 0.0 for t in tensors}
    with open(path, "rb") as f:
        pos = 0
        while True:
            buf = f.read(CHUNK)
            if not buf:
                break
            sha.update(buf)
            crc.update(buf)
            end = pos + len(buf)
            for t in tensors:
                a, b = max(pos, t["start"]), min(end, t["end"])
                if a < b:
                    x = np.frombuffer(buf[a - pos : b - pos], np.float32).astype(np.float64)
                    sq[t["tensor"]] += float(np.dot(x, x))
            pos = end
    return sq, sha.hexdigest(), base64.b64encode(crc.digest()).decode()


def fetch_errors(log, run):
    """取回日志里这次运行那一段的 kubectl exec 流中断次数（每次中断打一行 Copying stdout failed）。"""
    n, inside = 0, False
    for line in open(log, errors="replace"):
        if line.startswith("=== "):
            inside = line.split()[1] == run and " start " in line
            continue
        if inside and "Copying stdout failed" in line and "close 1006" in line:
            n += 1
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="运行目录（含 diag_grads.safetensors、diag_step1.json、meta.json）")
    p.add_argument("--ref", default="", help="文件布局逐字节相同的干净运行目录（读它在零段上的平方和）")
    p.add_argument("--block", type=int, default=256 << 20, help="取回时的分块大小（字节）")
    p.add_argument("--fetch-log", default="", help="run.sh 取回这次运行的日志")
    p.add_argument("--gcs-uri", default="", help="桶里同一文件；只比 crc32c，结果里不记桶名")
    p.add_argument("--committed", default="", help="已提交的 step1 结果 JSON（含 tpu_file_check），核对损坏清单")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    run = os.path.basename(os.path.normpath(a.run))
    path = os.path.join(a.run, "diag_grads.safetensors")
    tensors, base = header(path)
    hs, size = holes(path)
    dev = json.load(open(os.path.join(a.run, "diag_step1.json")))["per_tensor"]
    meta_sha = hashlib.sha256(open(os.path.join(a.run, "meta.json"), "rb").read()).hexdigest()

    ref_f = None
    if a.ref:
        rpath = os.path.join(a.ref, "diag_grads.safetensors")
        rt, _ = header(rpath)
        same_layout = [(t["tensor"], t["start"], t["end"]) for t in rt] == [
            (t["tensor"], t["start"], t["end"]) for t in tensors
        ]
        if not same_layout or os.path.getsize(rpath) != size:
            raise SystemExit("--ref 的文件布局与被查文件不同，不能逐字节对位")
        rholes, _ = holes(rpath)
        ref_f = open(rpath, "rb")

    segs = []
    with open(path, "rb") as f:
        for h, d in hs:
            z0 = last_nonzero_before(f, h)
            seg = {
                "hole_start": h,
                "hole_end": d,
                "zero_start": z0,
                "zero_end": d,
                "zero_len": d - z0,
                "zero_start_mod_block": z0 % a.block,
                "zero_end_mod_block": d % a.block,
                "end_is_block_boundary_or_eof": d % a.block == 0 or d == size,
                "block_index": z0 // a.block,
                "zero_start_pow2_align": pow2_align(z0),
                "in_header": z0 < base,
                "tensors": overlap(tensors, z0, d),
            }
            if ref_f is not None:
                seg["ref_sq_in_segment"] = sq_range(ref_f, z0, d)
            segs.append(seg)

    sq, sha, crc = full_pass(path, tensors)
    per = []
    touched = {o["tensor"] for s in segs for o in s["tensors"]}
    for t in tensors:
        n = t["tensor"]
        d_sq = dev[n]["sq"]
        row = {
            "tensor": n,
            "file_sq": sq[n],
            "device_sq": d_sq,
            "file_sq_rel": (sq[n] - d_sq) / d_sq if d_sq else 0.0,
            "touched_by_zero_segment": n in touched,
        }
        row["corrupt"] = abs(row["file_sq_rel"]) > FILE_SQ_TOL
        if ref_f is not None and n in touched:
            row["ref_sq_in_segments"] = sum(
                sq_range(ref_f, max(s["zero_start"], t["start"]), min(s["zero_end"], t["end"]))
                for s in segs
                if max(s["zero_start"], t["start"]) < min(s["zero_end"], t["end"])
            )
            row["sq_deficit"] = d_sq - sq[n]
            row["deficit_over_ref_segment_sq"] = (
                row["sq_deficit"] / row["ref_sq_in_segments"] if row["ref_sq_in_segments"] else None
            )
        per.append(row)

    out = {
        "what": __doc__.split("\n")[0],
        "run": run,
        "meta_sha256": meta_sha,
        "file": {"size": size, "header_end": base, "sha256": sha, "crc32c_b64": crc, "n_tensors": len(tensors)},
        "block_bytes": a.block,
        "holes_n": len(hs),
        "zero_segments": segs,
        "summary": {
            "segments": len(segs),
            "all_end_on_block_boundary": all(s["end_is_block_boundary_or_eof"] for s in segs),
            "zero_len_min": min((s["zero_len"] for s in segs), default=0),
            "zero_len_max": max((s["zero_len"] for s in segs), default=0),
            "zero_len_total": sum(s["zero_len"] for s in segs),
            "distinct_blocks": len({s["block_index"] for s in segs}),
            "nblocks": (size + a.block - 1) // a.block,
            "tensors_touched": sorted(touched),
            "tensors_corrupt_by_sq": sorted(r["tensor"] for r in per if r["corrupt"]),
            "touched_but_sq_matches": sorted(
                r["tensor"] for r in per if r["touched_by_zero_segment"] and not r["corrupt"]
            ),
            "corrupt_but_not_touched": sorted(
                r["tensor"] for r in per if r["corrupt"] and not r["touched_by_zero_segment"]
            ),
            "clean_max_abs_file_sq_rel": max((abs(r["file_sq_rel"]) for r in per if not r["corrupt"]), default=0.0),
        },
        "per_tensor": per,
    }
    if a.ref:
        out["ref"] = {
            "run": os.path.basename(os.path.normpath(a.ref)),
            "meta_sha256": hashlib.sha256(open(os.path.join(a.ref, "meta.json"), "rb").read()).hexdigest(),
            "holes_n": len(rholes),
            "same_layout": True,
        }
    if a.fetch_log:
        out["fetch_log_stream_aborts"] = fetch_errors(a.fetch_log, run)
    if a.gcs_uri:
        r = subprocess.run(
            ["gcloud", "storage", "objects", "describe", a.gcs_uri, "--format=json"], capture_output=True, text=True
        )
        if r.returncode == 0:
            obj = json.loads(r.stdout)
            out["bucket_copy"] = {
                "size": int(obj["size"]),
                "crc32c_b64": obj.get("crc32c_hash"),
                "same_as_local": int(obj["size"]) == size and obj.get("crc32c_hash") == crc,
            }
        else:
            out["bucket_copy"] = {"error": r.stderr.strip()[-300:]}
    if a.committed:
        c = json.load(open(a.committed))
        listed = sorted(x["tensor"] for x in c["tpu_file_check"]["corrupt"])
        rel = {x["tensor"]: x["file_sq_rel"] for x in c["tpu_file_check"]["corrupt"]}
        out["committed_check"] = {
            "file": os.path.relpath(a.committed, os.path.join(os.path.dirname(__file__), "..", "..", "..")),
            "listed_corrupt": listed,
            "same_set": listed == out["summary"]["tensors_corrupt_by_sq"],
            "max_abs_rel_diff_of_file_sq_rel": max(
                (abs(rel[r["tensor"]] - r["file_sq_rel"]) for r in per if r["tensor"] in rel), default=0.0
            ),
        }
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    s = out["summary"]
    print(
        f"{run}: 零段 {s['segments']} 个（长 {s['zero_len_min']}–{s['zero_len_max']} B，共 {s['zero_len_total']} B），"
        f"全部止于 {a.block} 字节块边界：{s['all_end_on_block_boundary']}；碰到 {len(s['tensors_touched'])} 个张量，"
        f"平方和对不上 {len(s['tensors_corrupt_by_sq'])} 个；碰到但对得上 {s['touched_but_sq_matches']}；"
        f"对不上但没碰到 {s['corrupt_but_not_touched']}"
    )


if __name__ == "__main__":
    main()
