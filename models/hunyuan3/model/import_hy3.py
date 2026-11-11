"""HF → Megatron（torch_dist）。用法：python import_hy3.py <HF 目录> <Megatron 目录>"""
import sys

from megatron.bridge import AutoBridge

AutoBridge.import_ckpt(sys.argv[1], sys.argv[2], trust_remote_code=True)
print("IMPORT_DONE")
