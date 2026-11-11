"""Megatron（torch_dist）→ HF。用法：python export_hy3.py <原 HF 目录（取配置与模型代码）> <Megatron 目录> <导出 HF 目录>"""
import sys

from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained(sys.argv[1], trust_remote_code=True)
bridge.export_ckpt(sys.argv[2], sys.argv[3], source_path=sys.argv[1])
print("EXPORT_DONE")
