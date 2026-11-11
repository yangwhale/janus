"""GPU 训练镜像里实际加载的软件版本：在 GPU 上跑一次 BF16 / FP32 矩阵乘与 TE 线性层，再读进程里真正映射的 cuBLAS / cuDNN / NCCL / cudart 并调各库自己的版本函数。

要在训练的库环境里跑（同 pod_run.sh：LD_LIBRARY_PATH 以 /usr/lib/x86_64-linux-gnu 打头，否则节点挂进来的旧 NCCL 先被找到、torch 导入失败）：

    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=7 python3 version_probe.py

输出一行 `VERSION_PROBE {json}`。
"""

import ctypes
import json
import re
import subprocess

import torch

out = {}
out["torch"] = torch.__version__
out["torch_version_cuda"] = torch.version.cuda
out["cudnn"] = torch.backends.cudnn.version()
try:
    out["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
except Exception as e:  # noqa: BLE001
    out["nccl"] = f"error: {e}"
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
(x @ x).sum().item()
xf = x.float()
(xf @ xf).sum().item()
import transformer_engine  # noqa: E402
import transformer_engine.pytorch as te  # noqa: E402

out["transformer_engine"] = transformer_engine.__version__
lin = te.Linear(256, 256, params_dtype=torch.bfloat16).cuda()
lin(x).sum().item()
import megatron.core  # noqa: E402

out["megatron_core"] = megatron.core.__version__
libs = {}
for line in open("/proc/self/maps"):
    p = line.split()[-1]
    m = re.search(r"/(lib(cublasLt|cublas|cudart|nccl|cudnn)[^/]*\.so[^/]*)$", p)
    if m:
        libs.setdefault(m.group(2), set()).add(p)
out["loaded_libs"] = {k: sorted(v) for k, v in libs.items()}


def ver(path, fn):
    lib = ctypes.CDLL(path)
    if fn == "cublasLtGetVersion":
        f = lib.cublasLtGetVersion
        f.restype = ctypes.c_size_t
        return int(f())
    if fn == "cublasGetProperty":
        vals = []
        for prop in (0, 1, 2):  # MAJOR / MINOR / PATCH
            v = ctypes.c_int()
            lib.cublasGetProperty(prop, ctypes.byref(v))
            vals.append(v.value)
        return ".".join(map(str, vals))
    if fn == "cudaRuntimeGetVersion":
        v = ctypes.c_int()
        lib.cudaRuntimeGetVersion(ctypes.byref(v))
        return v.value
    if fn == "ncclGetVersion":
        v = ctypes.c_int()
        lib.ncclGetVersion(ctypes.byref(v))
        return v.value
    return None


for key, fn in (
    ("cublas", "cublasGetProperty"),
    ("cublasLt", "cublasLtGetVersion"),
    ("cudart", "cudaRuntimeGetVersion"),
    ("nccl", "ncclGetVersion"),
):
    for p in out["loaded_libs"].get(key, []):
        try:
            out.setdefault("lib_versions", {})[p] = ver(p, fn)
        except Exception as e:  # noqa: BLE001
            out.setdefault("lib_versions", {})[p] = f"error: {e}"
pip = subprocess.run(["python3", "-m", "pip", "list", "--format=json"], capture_output=True, text=True).stdout
try:
    out["pip_nvidia"] = {
        d["name"]: d["version"]
        for d in json.loads(pip)
        if re.search(r"cublas|cudnn|nccl|cuda-runtime|transformer.engine|^torch$|megatron", d["name"], re.I)
    }
except Exception as e:  # noqa: BLE001
    out["pip_nvidia"] = f"error: {e}"
out["nvidia_smi"] = (
    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True
    )
    .stdout.strip()
    .splitlines()[:1]
)
out["nvcc"] = subprocess.run("nvcc --version | tail -2", shell=True, capture_output=True, text=True).stdout.strip()
out["env"] = {
    k: v
    for k, v in __import__("os").environ.items()
    if re.match(
        r"(CUDA_VERSION|CUBLAS_VERSION|NCCL_VERSION|CUDNN_VERSION|TRANSFORMER_ENGINE_VERSION|"
        r"PYTORCH_VERSION|NVIDIA_PYTORCH_VERSION|NEMO_VERSION|NVIDIA_BUILD_ID|TRT_VERSION)$",
        k,
    )
}
print("VERSION_PROBE " + json.dumps(out, ensure_ascii=False, sort_keys=True), flush=True)
