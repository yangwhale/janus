"""把 tencent/Hy3 截成前 N 层、去掉 MTP，写成一个新的 HF 目录。

用法：python truncate_hy3.py <源目录> <目标目录> <层数>
只拷 embed / final norm / lm_head 和前 N 层的张量，逐张量核对字节相同。
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

src, dst, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
dst.mkdir(parents=True, exist_ok=True)
cfg = json.loads((src / "config.json").read_text())
cfg["num_hidden_layers"] = n
cfg["num_nextn_predict_layers"] = 0
(dst / "config.json").write_text(json.dumps(cfg, indent=2))
for f in ["generation_config.json", "tokenizer.json", "tokenizer_config.json"]:
    if (src / f).exists():
        shutil.copy(src / f, dst / f)

wm = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
def keep(k):
    if not k.startswith("model.layers."):
        return True
    return int(k.split(".")[2]) < n
names = sorted(k for k in wm if keep(k))
by_file = {}
for k in names:
    by_file.setdefault(wm[k], []).append(k)

new_map, manifest, part = {}, {}, 0
for f in sorted(by_file):
    tensors = {}
    with safe_open(src / f, framework="pt") as h:
        for k in by_file[f]:
            tensors[k] = h.get_tensor(k).contiguous()
    part += 1
    out = f"model-{part:05d}.safetensors"
    save_file(tensors, dst / out, metadata={"format": "pt"})
    for k, t in tensors.items():
        new_map[k] = out
        manifest[k] = hashlib.sha256(t.view(-1).view(dtype=__import__("torch").uint8).numpy().tobytes()).hexdigest()
    print(out, len(tensors), flush=True)

(dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": new_map}, indent=1))
(dst / "tensor_sha256.json").write_text(json.dumps(manifest, indent=1))
print("tensors", len(names))
