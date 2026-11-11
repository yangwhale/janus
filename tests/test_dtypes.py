"""dtypes.py：浮点类型 → (位宽, 同宽无符号整数)。ULP 与逐位比较都靠它按原始 dtype 取位模式。"""

import ml_dtypes
import numpy as np
import pytest

from janus.dtypes import float_bits


@pytest.mark.parametrize(
    "dtype,width,uint",
    [
        (np.float64, 64, np.uint64),
        (np.float32, 32, np.uint32),
        (np.float16, 16, np.uint16),
        (ml_dtypes.bfloat16, 16, np.uint16),
    ],
)
def test_float_bits(dtype, width, uint):
    w, u = float_bits(dtype)
    assert w == width and np.dtype(u) == np.dtype(uint)
    x = np.array([1.5, -0.0], dtype=dtype)
    assert x.view(u).dtype.itemsize * 8 == width  # 同宽，能原样 view


@pytest.mark.parametrize("dtype", [np.int32, np.uint8, np.bool_])
def test_non_float_rejected(dtype):
    with pytest.raises(TypeError, match="不支持的浮点类型"):
        float_bits(dtype)
