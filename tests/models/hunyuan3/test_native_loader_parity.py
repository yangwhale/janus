"""原生加载器对拍脚本：用合成的 GPU 整批输入与 MaxText dump 跑一遍，并确认故意改坏的数据会被发现、且定位到对的环节。"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from helpers import ROOT

SPEC = importlib.util.spec_from_file_location(
    "native_loader_parity", ROOT / "models/hunyuan3/data/native_loader_parity.py")
nlp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nlp)

DP, MBS, GBS, SEQ, STEPS, EOD = 2, 1, 4, 8, 3, 99
GA = GBS // (DP * MBS)


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _stream(seed=0):
    """全局样本流：每个样本 SEQ+1 个 token，里面夹着 EOD。"""
    rng = np.random.RandomState(seed)
    raw = rng.randint(1, 50, size=(STEPS * GBS, SEQ + 1))
    raw[:, 3] = EOD
    return raw


def _write_npy(d, name, arr):
    np.save(d / name, arr)
    return d / name


def build(tmp: Path, mutate_mt=None, mutate_gpu=None, mt_index_differs=False):
    raw = _stream()
    gpu = tmp / "gpu"
    (gpu / "data" / "batches").mkdir(parents=True)
    meta = {"parallelism": {"dp": DP}, "config": {"args": {"mbs": MBS, "gbs": GBS, "seq": SEQ, "steps": STEPS, "seed": 1,
                                                           "data": "/x/ds"}},
            "data": {"manifest_sha256": "abc"}}
    (gpu / "meta.json").write_text(json.dumps(meta))
    for r in range(DP):
        fb = open(gpu / "data" / "batches" / f"rank{r}.bin", "wb")
        fj = open(gpu / "data" / "batches" / f"rank{r}.jsonl", "w")
        ff = open(gpu / "data" / f"rank{r}.jsonl", "w")
        for s in range(1, STEPS + 1):
            for m in range(GA):
                i = (s - 1) * GBS + m * DP * MBS + r * MBS          # Megatron 的分法
                f = {"tokens": raw[i, :-1].astype(np.int64)[None], "labels": raw[i, 1:].astype(np.int64)[None],
                     "loss_mask": np.ones((1, SEQ), np.float32), "position_ids": np.arange(SEQ, dtype=np.int64)[None]}
                if mutate_gpu:
                    mutate_gpu(s, r, m, f)
                rec, fp = {"step": s, "micro": m, "fields": {}}, {"step": s, "micro": m}
                for k, v in f.items():
                    rec["fields"][k] = {"offset": fb.tell(), "dtype": str(v.dtype), "shape": list(v.shape), "sha256": _sha(v)}
                    fp[k] = _sha(v)
                    fb.write(v.tobytes())
                fp["nonpad"] = int(f["loss_mask"].sum())
                fj.write(json.dumps(rec) + "\n")
                ff.write(json.dumps(fp) + "\n")
        fb.close(), fj.close(), ff.close()
    # Megatron 索引缓存
    cache = tmp / "cache"
    cache.mkdir()
    desc = {"class": "GPTDataset", "dataset_path": "/x/ds", "num_samples": STEPS * GBS, "random_seed": 1,
            "sequence_length": SEQ, "index_split": "train"}
    (cache / "h1-GPTDataset-train-description.txt").write_text(json.dumps(desc))
    idx = {"document_index": np.arange(5, dtype=np.int32), "sample_index": np.zeros((4, 2), np.int32),
           "shuffle_index": np.arange(3, dtype=np.uint32)}
    for k, v in idx.items():
        _write_npy(cache, f"h1-GPTDataset-train-{k}.npy", v)
    mt_idx = {}
    for k, v in idx.items():
        if mt_index_differs and k == "shuffle_index":
            v = v[::-1].copy()
        p = _write_npy(tmp, f"mt-{k}.npy", v)
        mt_idx[f"mthash-{k}.npy"] = {"sha256_file": nlp.sha_file(p), "sha256_array": _sha(v), "dtype": str(v.dtype),
                                     "shape": list(v.shape)}
    # MaxText dump：单 host，读进来的第 g 行 = 步内第 g 个样本
    mt = tmp / "mt"
    mt.mkdir()
    fb, fj = open(mt / "batches.bin", "wb"), open(mt / "batches.jsonl", "w")
    for s in range(1, STEPS + 1):
        rows = raw[(s - 1) * GBS: s * GBS]
        f = {"inputs": rows[:, :-1].astype(np.int32), "targets": rows[:, 1:].astype(np.int32),
             "inputs_segmentation": np.ones((GBS, SEQ), np.int32), "targets_segmentation": np.ones((GBS, SEQ), np.int32),
             "inputs_position": np.tile(np.arange(SEQ, dtype=np.int32), (GBS, 1)),
             "targets_position": np.tile(np.arange(SEQ, dtype=np.int32), (GBS, 1))}
        if mutate_mt:
            mutate_mt(s, f)
        rec = {"step": s, "fields": {}}
        for k in nlp.MT_FIELDS:
            v = np.ascontiguousarray(f[k])
            rec["fields"][k] = {"offset": fb.tell(), "dtype": str(v.dtype), "shape": list(v.shape), "sha256": _sha(v)}
            fb.write(v.tobytes())
        fj.write(json.dumps(rec) + "\n")
    fb.close(), fj.close()
    (mt / "meta.json").write_text(json.dumps({
        "maxtext": {"commit": "0" * 40}, "versions": {}, "overrides": [], "mesh": {}, "process_count": 1,
        "rows_of_device": {str(d): [d * 2, d * 2 + 1] for d in range(DP)},
        "effective_config": {"mmap_eod_id": EOD, "grain_worker_count": 1}, "data": {"idx_sha256": "abc"},
        "index_files": mt_idx}))
    return mt, gpu, cache


def _run(tmp, **kw):
    mt, gpu, cache = build(tmp, **kw)
    res = nlp.run(mt, gpu, cache, tmp / "out", pad_id=7)
    return {c["id"]: c for c in res["checks"]}, res


def test_clean_run_passes_and_native_layout_is_flagged(tmp_path):
    c, res = _run(tmp_path)
    assert not res["summary"]["hard_fail"]
    for k in ("L0.native.gpu_integrity", "L0.native.index", "L0.native.stream", "L0.native.fingerprint",
              "L0.native.nonpad_total", "L0.native.fields", "L0.native.coverage"):
        assert c[k]["status"] == "pass", (k, c[k])
    # GA=2 时 MaxText 原样的设备分配与 Megatron 不同：只有设备 0 的第 0 个和最后一个设备的第 1 个凑巧相同
    assert c["L0.native.device_map.native"]["status"] == "fail"
    assert c["L0.native.device_map.native"]["metrics"]["same"] == 2 * STEPS
    assert (tmp_path / "out" / "report.md").read_text().startswith("# TPU 原生数据加载器对拍")


def test_last_label_change_is_caught_and_localized(tmp_path):
    def bad(s, f):
        if s == 2:
            f["targets"][3, -1] += 1                    # 换序后落在设备 1、microbatch 1
    c, res = _run(tmp_path, mutate_mt=bad)
    assert res["summary"]["hard_fail"]
    fp = c["L0.native.fingerprint"]
    assert fp["status"] == "fail"
    assert any(k.startswith("label：只有最后一个") for k in fp["details_inline"]["localized"])


def test_row_order_change_is_caught_as_stream_and_layout(tmp_path):
    def swap(s, f):
        if s == 1:
            for k in f:
                f[k][[0, 1]] = f[k][[1, 0]]
    c, _ = _run(tmp_path, mutate_mt=swap)
    assert c["L0.native.stream"]["status"] == "fail"
    assert any(k.startswith("layout") for k in c["L0.native.fingerprint"]["details_inline"]["localized"])


def test_foreign_sample_is_localized_to_index_or_shuffle(tmp_path):
    def foreign(s, f):
        if s == 3:
            f["inputs"][0] = 1234
    c, _ = _run(tmp_path, mutate_mt=foreign)
    assert any(k.startswith("index/shuffle") for k in c["L0.native.fingerprint"]["details_inline"]["localized"])


def test_eod_loss_mask_change_is_localized(tmp_path):
    def eml(s, f):
        f["targets_segmentation"][f["inputs"] == EOD] = 0     # 相当于开了 eod_mask_loss
    c, _ = _run(tmp_path, mutate_mt=eml)
    loc = c["L0.native.fingerprint"]["details_inline"]["localized"]
    assert any(k.startswith("mask：EOD") for k in loc)
    assert c["L0.native.nonpad_total"]["status"] == "fail"


def test_position_reset_is_localized(tmp_path):
    def reset(s, f):
        f["inputs_position"][:, 4:] -= 4                       # EOD（第 3 位）之后位置从 0 重来
    c, _ = _run(tmp_path, mutate_mt=reset)
    assert any(k.startswith("position") for k in c["L0.native.fingerprint"]["details_inline"]["localized"])
    assert c["L0.native.fields"]["status"] == "fail"


def test_index_difference_fails(tmp_path):
    c, _ = _run(tmp_path, mt_index_differs=True)
    assert c["L0.native.index"]["status"] == "fail"


def test_corrupt_gpu_reference_is_error_not_pass(tmp_path):
    mt, gpu, cache = build(tmp_path)
    b = gpu / "data" / "batches" / "rank1.bin"
    raw = bytearray(b.read_bytes())
    raw[10] ^= 1
    b.write_bytes(bytes(raw))
    res = nlp.run(mt, gpu, cache, tmp_path / "out")
    c = {x["id"]: x for x in res["checks"]}
    assert c["L0.native.gpu_integrity"]["status"] == "error"
    assert res["summary"]["hard_fail"]


def test_partial_window_is_coverage_error(tmp_path):
    mt, gpu, cache = build(tmp_path)
    res = nlp.run(mt, gpu, cache, tmp_path / "out", steps="1:2")
    c = {x["id"]: x for x in res["checks"]}
    assert c["L0.native.coverage"]["status"] == "error"


def test_bin_dtype_reads_idx_header(tmp_path):
    (tmp_path / "a.idx").write_bytes(b"MMIDIDX\x00\x00" + (1).to_bytes(8, "little") + bytes([4]))
    assert nlp.bin_dtype(str(tmp_path / "a.bin")) == np.int32
    (tmp_path / "b.idx").write_bytes(b"NOTANIDX!" + bytes(9))
    with pytest.raises(ValueError):
        nlp.bin_dtype(str(tmp_path / "b.bin"))


def _ref(d: Path, ref: dict, arrs: dict):
    d.mkdir(parents=True)
    (d / "ref.json").write_text(json.dumps(ref))
    np.savez(d / "samples.npz", **arrs)


def test_check_epochs_catches_sample_and_index_difference(tmp_path):
    idx = {k: {"sha256_file": k * 2} for k in nlp.INDEX_KINDS}
    meg = {"num_samples": 10, "num_epochs": 2, "separate_final_epoch": False, "len": 12, "index": idx,
           "cross_epoch": [{"epoch_boundary": 1, "sample_id": 5, "stream_index": 7}], "picks": [0, 7]}
    arrs = {"0.tokens": np.arange(4), "7.tokens": np.arange(4) + 1}
    _ref(tmp_path / "m", meg, arrs)
    _ref(tmp_path / "t", {"len": 12, "index": {f"h-{k}.npy": v for k, v in idx.items()}}, arrs)
    assert nlp.check_epochs([(tmp_path / "m", tmp_path / "t")])["status"] == "pass"
    _ref(tmp_path / "t2", {"len": 12, "index": {f"h-{k}.npy": v for k, v in idx.items()}},
         {"0.tokens": np.arange(4), "7.tokens": np.arange(4)})
    assert nlp.check_epochs([(tmp_path / "m", tmp_path / "t2")])["status"] == "fail"
    bad_idx = {f"h-{k}.npy": {"sha256_file": "x"} for k in nlp.INDEX_KINDS}
    _ref(tmp_path / "t3", {"len": 12, "index": bad_idx}, arrs)
    assert nlp.check_epochs([(tmp_path / "m", tmp_path / "t3")])["status"] == "fail"


def test_check_eod_cases_separates_pad_from_eod_semantics(tmp_path):
    L, pad = 4, 7
    combos, marr, tarr = [], {}, {}
    base = {"tokens": np.array([1, 99, 2, 3]), "labels": np.array([99, 2, 3, 4]), "loss_mask": np.ones(L, np.float32),
            "position_ids": np.arange(L), "attention_mask": ~np.tril(np.ones((L, L), bool))}
    for ram in (0, 1):
        for rpi in (0, 1):
            for eml in (0, 1):
                name = f"ram{ram}_rpi{rpi}_eml{eml}"
                combos.append({"name": name, "reset_attention_mask": bool(ram), "reset_position_ids": bool(rpi),
                               "eod_mask_loss": bool(eml), "len": 2})
                for k, v in base.items():
                    marr[f"{name}.0.{k}"] = v
                    marr[f"{name}.1.{k}"] = v
                marr[f"{name}.1.labels"] = np.array([99, 0, 3, 4])           # Megatron：label 里的 pad 换成 0
                marr[f"{name}.1.loss_mask"] = np.array([1, 0, 1, 1], np.float32)
                if ram != rpi:
                    marr[f"{name}.0.position_ids"] = np.array([0, 1, 0, 1])
    for ram in (0, 1):
        for eml in (0, 1):
            name = f"ram{ram}_eml{eml}"
            for k, v in base.items():
                tarr[f"{name}.0.{k}"] = v
                tarr[f"{name}.1.{k}"] = v
            tarr[f"{name}.1.labels"] = np.array([99, pad, 3, 4])            # MaxText 原样保留
    ref = {"pad": pad, "combos": combos, "doc_lens": [2, 2], "seq": L, "idx_sha256": "i", "bin_sha256": "b"}
    _ref(tmp_path / "m", ref, marr)
    _ref(tmp_path / "t", {"combos": []}, tarr)
    eod, padc = nlp.check_eod_cases(tmp_path / "m", tmp_path / "t")
    assert eod["status"] == "pass"
    assert sum(1 for r in eod["metrics"]["combos"] if r["maxtext"] is None) == 4
    assert padc["status"] == "fail" and padc["hard"] is False
    # EOD 语义被改坏（比如 MaxText 侧多屏蔽了一个位置）时要报 fail
    tarr["ram0_eml0.0.loss_mask"] = np.array([1, 0, 1, 1], np.float32)
    (tmp_path / "t" / "samples.npz").unlink()
    np.savez(tmp_path / "t" / "samples.npz", **tarr)
    eod, _ = nlp.check_eod_cases(tmp_path / "m", tmp_path / "t")
    assert eod["status"] == "fail"


def test_negative_controls_must_be_detected(tmp_path):
    mt, gpu, cache = build(tmp_path / "a")

    def reset(s, f):
        f["inputs_position"][:, 4:] -= 4
    ctrl, _, _ = build(tmp_path / "b", mutate_mt=reset)
    res = nlp.run(mt, gpu, cache, tmp_path / "out", negatives=[("reset", ctrl), ("clean", mt)])
    neg = {x["id"]: x for x in res["checks"]}["L0.native.negative_controls"]
    # 干净的那个「反例」没被发现 → 整条 fail；被改坏的那个定位到 position
    assert neg["status"] == "fail" and "clean" in neg["message"]
    rows = {r["name"]: r for r in neg["metrics"]["controls"]}
    assert rows["reset"]["same"] == 0 and any(k.startswith("position") for k in rows["reset"]["localized"])
    res = nlp.run(mt, gpu, cache, tmp_path / "out2", negatives=[("reset", ctrl)])
    assert {x["id"]: x for x in res["checks"]}["L0.native.negative_controls"]["status"] == "pass"
