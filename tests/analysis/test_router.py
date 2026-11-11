"""analysis/router.py：第 1 步逐层比路由——选中集合一致率、翻转 token 的分差、logits 差、专家负载。"""

import numpy as np
import pytest

from janus.analysis import router


class FakeRun:
    def __init__(self, tensors):
        self.t = tensors

    def names(self, step, pattern=None):
        import fnmatch

        return [n for n in self.t if pattern is None or fnmatch.fnmatchcase(n, pattern)]

    def tensor(self, step, name):
        if name not in self.t:
            raise KeyError(name)
        return self.t[name]


def _run(logits, k=2):
    idx = np.argsort(-logits, axis=-1, kind="stable")[:, :k].astype(np.int32)
    counts = np.bincount(idx.ravel(), minlength=logits.shape[-1]).astype(np.int32)
    return {
        "layers.1.mlp.router.logits": logits.astype(np.float32),
        "layers.1.mlp.router.topk_idx": idx,
        "layers.1.mlp.router.counts": counts,
    }


def test_layer_table_counts_flips_and_margins():
    rng = np.random.default_rng(0)
    ref_logits = rng.standard_normal((64, 8))
    test_logits = ref_logits.copy()
    # 在第 0 个 token 上把第 2、3 名几乎打平再交换：只翻这一个 token
    order = np.argsort(-ref_logits[0])
    ref_logits[0, order[2]] = ref_logits[0, order[1]] - 1e-4
    test_logits[0] = ref_logits[0]
    test_logits[0, order[2]] = ref_logits[0, order[1]] + 1e-4
    rows = router.layer_table(FakeRun(_run(test_logits)), FakeRun(_run(ref_logits)))
    (row,) = rows
    assert row["layer"] == 1 and row["tokens"] == 64 and row["flipped_tokens"] == 1
    assert row["set_frac"] == pytest.approx(63 / 64)
    assert row["margin_p50_flipped"] < row["margin_p50_all"]  # 翻转的 token 是几乎打平的那个
    assert row["logits_rel_l2"] > 0 and row["load_l1_diff"] == 2
    html = router.table_html(rows)
    assert "选中集合一致" in html and "<td>1</td>" in html  # 表头与翻转数都在


def test_layer_table_identical_runs():
    rng = np.random.default_rng(1)
    t = _run(rng.standard_normal((16, 8)))
    (row,) = router.layer_table(FakeRun(t), FakeRun(t))
    assert row["set_frac"] == 1.0 and row["order_frac"] == 1.0 and row["flipped_tokens"] == 0
    assert row["logits_rel_l2"] == 0.0 and row["load_l1_diff"] == 0
