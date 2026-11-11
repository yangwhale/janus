"""dtype 名与位模式。BF16 用 ml_dtypes 表示，存取时保持原始位模式。"""
import ml_dtypes
import numpy as np

# safetensors 的 dtype 名 → numpy dtype
ST_TO_NP = {
    "F64": np.dtype(np.float64), "F32": np.dtype(np.float32), "F16": np.dtype(np.float16),
    "BF16": np.dtype(ml_dtypes.bfloat16),
    "I64": np.dtype(np.int64), "I32": np.dtype(np.int32), "I16": np.dtype(np.int16), "I8": np.dtype(np.int8),
    "U64": np.dtype(np.uint64), "U32": np.dtype(np.uint32), "U16": np.dtype(np.uint16), "U8": np.dtype(np.uint8),
    "BOOL": np.dtype(np.bool_),
}
NP_TO_ST = {v: k for k, v in ST_TO_NP.items()}

# 浮点类型的位宽与对应的无符号整数类型
_FLOAT_BITS = {
    np.dtype(np.float64): (64, np.uint64), np.dtype(np.float32): (32, np.uint32),
    np.dtype(np.float16): (16, np.uint16), np.dtype(ml_dtypes.bfloat16): (16, np.uint16),
}


def float_bits(dtype):
    """(位宽, 无符号整数 dtype)；不是支持的浮点类型时抛 TypeError。"""
    d = np.dtype(dtype)
    if d not in _FLOAT_BITS:
        raise TypeError(f"不支持的浮点类型: {d}")
    return _FLOAT_BITS[d]
