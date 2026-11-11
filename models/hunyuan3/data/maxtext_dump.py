"""用 MaxText 自己的 Grain 输入管线（grain_file_type=mmap_npy）读 Megatron 索引格式数据，逐步存下每步的整批输出。

只用 CPU：用 XLA 的假设备把 CPU 当成 N 个设备，建和 TPU 上同样的 mesh，走 MaxText 训练时同一个入口
（pyconfig → get_mesh_from_config → create_data_iterator），所以分片、host 切分、批次组装都是训练时的代码路径。

用法（在装了 MaxText 的 venv 里跑，见 models/hunyuan3/data/README.md）：
  python maxtext_dump.py --data /data/ds/fw_hy3_text_document --npy-dir /tmp/g62/npy --out /tmp/g62/mt \
      --steps 1000 [--devices 8] [--per-device-batch 1] [--ga 2] [--seq 4096] [--eod 120025] [--seed 1234] \
      [--split 9999,1,0] [--workers 1] [--set key=value ...]

产出 <out>/：
  batches.bin      每步的全局批，字段按 FIELDS 顺序，int32 原样
  batches.jsonl    每步一行：{step, fields: {名: {offset, dtype, shape, sha256}}}
  meta.json        MaxText commit、实际生效的配置、mesh、每个设备持有全局批的哪几行、索引文件 sha256、版本
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

FIELDS = ("inputs", "targets", "inputs_segmentation", "targets_segmentation", "inputs_position", "targets_position")


def sha_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def emulated_hosts_iterator(config, hosts):
    """照 make_grain_train_iterator 的默认分支，对每个 host 建本地迭代器；只换 host_index / host_count 和本地批大小。"""
    import functools

    import numpy as np
    from maxtext.input_pipeline import data_processing_utils
    from maxtext.input_pipeline import grain_data_processing as gdp
    gbs = config.global_batch_size_to_load
    assert gbs % hosts == 0
    data_processing_utils.get_local_batch_size = lambda _cfg: gbs // hosts   # jax.process_count() 在单进程里是 1
    dataset_config = gdp._build_dataset_config(config, num_samples=config.steps * gbs, seed=config.data_shuffle_seed,
                                               split_ratio=config.mmap_npy_split or None, split_index=0)
    get_ds_fn = functools.partial(
        gdp.get_datasets, config.grain_train_files, config.grain_file_type, shuffle=config.enable_data_shuffling,
        shuffle_seed=config.data_shuffle_seed, shuffle_buffer_size=config.grain_shuffle_buffer_size,
        num_epoch=config.num_epoch, grain_worker_count=config.grain_worker_count,
        grain_num_threads=config.grain_num_threads, grain_prefetch_buffer_size=config.grain_prefetch_buffer_size,
        grain_data_source_max_workers=config.grain_data_source_max_workers,
        grain_index_storage_option=getattr(config, "grain_index_storage_option", None),
        mixture_config_path=config.grain_train_mixture_config_path, elastic=config.grain_use_elastic_iterator,
        hf_access_token=getattr(config, "hf_access_token", None), dataset_config=dataset_config, split="train")
    its = []
    for h in range(hosts):
        ds = get_ds_fn(dataloading_host_index=h, dataloading_host_count=hosts)
        its.append(iter(gdp._mmap_pretrain_pipeline(ds, config, config.train_data_columns[0], config.grain_worker_count,
                                                    config.grain_per_worker_buffer_size)))
    while True:
        parts = [next(x) for x in its]
        yield {k: np.concatenate([p[k] for p in parts]) for k in FIELDS}


def emulated_rows(config, devices):
    per = config.global_batch_size_to_load // devices
    return {str(d): list(range(d * per, (d + 1) * per)) for d in range(devices)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Megatron .bin/.idx 前缀")
    p.add_argument("--npy-dir", required=True, help="MaxText 的索引缓存目录（没有就现建）")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--dump-steps", type=int, default=None, help="只存前多少步（默认 = steps）；steps 决定索引的样本数，不要为少存几步去改它")
    p.add_argument("--devices", type=int, default=8)
    p.add_argument("--per-device-batch", type=int, default=1)
    p.add_argument("--ga", type=int, default=2, help="gradient_accumulation_steps")
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--eod", type=int, default=120025)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--split", default="9999,1,0")
    p.add_argument("--workers", type=int, default=1, help="grain_worker_count")
    p.add_argument("--reset-attention-mask", action="store_true",
                   help="打开 MaxText 的 reset_attention_mask（MaxText 默认值；GPU 这边是关的，只在做反例时用）")
    p.add_argument("--packing-divisor", type=int, default=0, help="packing_max_segments_per_sample（MaxText 默认 25）")
    p.add_argument("--set", action="append", default=[], help="额外的 MaxText 配置覆盖 key=value")
    p.add_argument("--emulate-hosts", type=int, default=0,
                   help="在一个进程里模拟 H 个 host：每个 host 用 MaxText 自己的 get_datasets(host_index=h, host_count=H) "
                        "+ _mmap_pretrain_pipeline 建本地迭代器，本地批按 host 拼成全局批（假定 host h 的设备在批维上连续）")
    a = p.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={a.devices}"
    from absl import flags
    flags.FLAGS(sys.argv[:1])  # MaxText 训练入口走 absl app.run；Grain 多进程读 absl flag，不解析会报错
    import grain
    import jax
    import maxtext
    import numpy as np
    from maxtext.configs import pyconfig
    from maxtext.input_pipeline.input_pipeline_interface import create_data_iterator
    from maxtext.utils import maxtext_utils

    assert jax.default_backend() == "cpu", jax.default_backend()
    mt_root = os.path.dirname(os.path.dirname(os.path.dirname(maxtext.__file__)))
    base_yml = os.path.join(os.path.dirname(maxtext.__file__), "configs", "base.yml")
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.npy_dir, exist_ok=True)
    overrides = [
        "run_name=g62-native-loader", f"base_output_directory={a.out}/run",
        "dataset_type=grain", "grain_file_type=mmap_npy", f"grain_train_files={a.npy_dir}|{a.data}",
        f"max_target_length={a.seq}", f"mmap_eod_id={a.eod}", f"data_shuffle_seed={a.seed}",
        # GPU 侧（Megatron-Bridge 默认）：reset_attention_mask / reset_position_ids / eod_mask_loss 全是 False
        f"reset_attention_mask={a.reset_attention_mask}", "eod_mask_loss=False",
        f"packing_max_segments_per_sample={a.packing_divisor}",
        "mmap_split_sentences=False",
        f"per_device_batch_size={a.per_device_batch}", f"gradient_accumulation_steps={a.ga}", f"steps={a.steps}",
        f"grain_worker_count={a.workers}", "enable_checkpointing=False", "eval_interval=-1",
    ] + ([f"mmap_npy_split={a.split}"] if a.split else []) + a.set   # 空 split 不传（命令行的空值会被解析成 None），用 base.yml 的默认 ''
    config = pyconfig.initialize(["maxtext_dump", base_yml] + overrides)
    mesh = maxtext_utils.get_mesh_from_config(config)
    t0 = time.time()
    if a.emulate_hosts:
        it = emulated_hosts_iterator(config, a.emulate_hosts)
    else:
        it, _ = create_data_iterator(config, mesh)
    t_build = time.time() - t0

    n_dump = a.dump_steps or a.steps
    rows_of_device = None
    bin_path, jl_path = os.path.join(a.out, "batches.bin"), os.path.join(a.out, "batches.jsonl")
    off = 0
    t1 = time.time()
    with open(bin_path, "wb") as fb, open(jl_path, "w") as fj:
        for step in range(1, n_dump + 1):
            batch = next(it)
            rec = {"step": step, "fields": {}}
            for k in FIELDS:
                arr = batch[k]
                if rows_of_device is None and not a.emulate_hosts:
                    # 每个设备持有全局批的哪几行：直接读 jax.Array 的分片，不靠推断
                    rows_of_device = {}
                    for sh in arr.addressable_shards:
                        rs = sh.index[0]
                        rows_of_device[str(sh.device.id)] = list(range(*rs.indices(arr.shape[0])))
                x = np.ascontiguousarray(np.asarray(arr))
                raw = x.tobytes()
                fb.write(raw)
                rec["fields"][k] = {"offset": off, "dtype": str(x.dtype), "shape": list(x.shape),
                                    "sha256": hashlib.sha256(raw).hexdigest()}
                off += len(raw)
            fj.write(json.dumps(rec) + "\n")
            if step % 100 == 0:
                print(f"step {step} {time.time() - t1:.0f}s", flush=True)
    t_iter = time.time() - t1

    idx_files = {}
    for f in sorted(os.listdir(a.npy_dir)):
        fp = os.path.join(a.npy_dir, f)
        if f.endswith(".npy"):
            arr = np.load(fp, mmap_mode="r")
            idx_files[f] = {"sha256_file": sha_file(fp), "dtype": str(arr.dtype), "shape": list(arr.shape),
                            "sha256_array": hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()}
        elif f.endswith(".json"):
            idx_files[f] = {"sha256_file": sha_file(fp), "content": json.load(open(fp))}

    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", mt_root, *args], text=True).strip()
        except Exception as e:  # 装的是 wheel 而不是源码
            return f"unknown ({e.__class__.__name__})"

    eff = {k: getattr(config, k) for k in (
        "grain_file_type", "grain_train_files", "max_target_length", "mmap_eod_id", "data_shuffle_seed",
        "reset_attention_mask", "eod_mask_loss", "packing_max_segments_per_sample", "mmap_split_sentences",
        "mmap_npy_split", "per_device_batch_size", "gradient_accumulation_steps", "global_batch_size_to_load",
        "micro_batch_size_to_train_on", "steps", "grain_worker_count", "enable_data_shuffling", "mesh_axes",
        "data_sharding", "use_tunix_gradient_accumulation", "expansion_factor_real_data")}
    meta = {
        "tool": "models/hunyuan3/data/maxtext_dump.py",
        "maxtext": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
                    "root": mt_root},
        "versions": {"python": sys.version.split()[0], "jax": jax.__version__, "numpy": np.__version__,
                     "grain": getattr(grain, "__version__", "unknown")},
        "backend": jax.default_backend(), "process_count": jax.process_count(),
        "devices": [str(d) for d in jax.devices()],
        "mesh": {"axis_names": list(mesh.axis_names), "shape": dict(mesh.shape),
                 "device_ids": np.vectorize(lambda d: d.id)(mesh.devices).reshape(-1).tolist()},
        "overrides": overrides, "effective_config": eff,
        "data": {"prefix": a.data, "idx_sha256": sha_file(a.data + ".idx")},
        "rows_of_device": rows_of_device or emulated_rows(config, a.devices), "fields": list(FIELDS), "steps_dumped": n_dump,
        "emulated_hosts": a.emulate_hosts or None,
        "index_files": idx_files,
        "timing_s": {"build_iterator": round(t_build, 1), "iterate": round(t_iter, 1)},
    }
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1, default=str)
    print(json.dumps({"steps": n_dump, "build_s": meta["timing_s"]["build_iterator"],
                      "iter_s": meta["timing_s"]["iterate"], "index_files": sorted(idx_files)}))


if __name__ == "__main__":
    main()
