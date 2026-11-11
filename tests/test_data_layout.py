import numpy as np
import pytest

from janus.data_layout import (
    maxtext_loaded_slot,
    maxtext_micro_split,
    maxtext_perm_for_megatron,
    maxtext_row,
    megatron_slot,
    per_device,
)


def _megatron_view(stream, dp, mbs, ga):
    """照 MegatronPretrainingSampler 的切法：每个 microbatch 取 dp×mbs 个，rank r 拿第 r 段。"""
    return {(r, m): stream[m * dp * mbs + r * mbs: m * dp * mbs + (r + 1) * mbs] for r in range(dp) for m in range(ga)}


@pytest.mark.parametrize("dp,mbs,ga", [(8, 1, 2), (8, 2, 2), (4, 1, 4), (8, 1, 1), (2, 3, 2)])
def test_perm_maps_maxtext_devices_onto_megatron_ranks(dp, mbs, ga):
    gbs = dp * mbs * ga
    stream = np.arange(gbs)[:, None] * 10 + np.arange(3)          # 每行可辨认
    meg = _megatron_view(stream, dp, mbs, ga)
    perm = maxtext_perm_for_megatron(dp, mbs, ga)
    assert sorted(perm.tolist()) == list(range(gbs))             # 是置换，不丢不重
    got = per_device(stream[perm], dp, mbs, ga)
    for key in meg:
        np.testing.assert_array_equal(got[key], meg[key])


def test_native_maxtext_layout_differs_from_megatron_when_ga_gt_1():
    dp, mbs, ga = 8, 1, 2
    stream = np.arange(16)[:, None]
    meg = _megatron_view(stream, dp, mbs, ga)
    native = per_device(stream, dp, mbs, ga)
    same = [k for k in meg if np.array_equal(meg[k], native[k])]
    # 只有设备 0 的第 0 个和设备 7 的第 1 个凑巧相同
    assert same == [(0, 0), (7, 1)]
    assert native[(1, 0)][0, 0] == 2 and meg[(1, 0)][0, 0] == 1


def test_ga1_layouts_agree():
    stream = np.arange(8)[:, None]
    assert (maxtext_perm_for_megatron(8, 1, 1) == np.arange(8)).all()
    meg = _megatron_view(stream, 8, 1, 1)
    native = per_device(stream, 8, 1, 1)
    assert all(np.array_equal(meg[k], native[k]) for k in meg)


def test_micro_split_matches_maxtext_reshape_and_rejects_bad_ga():
    b = np.arange(6)
    np.testing.assert_array_equal(maxtext_micro_split(b, 2), [[0, 2, 4], [1, 3, 5]])
    with pytest.raises(ValueError):
        maxtext_micro_split(np.arange(5), 2)


def test_slot_formulas():
    assert megatron_slot(rank=3, micro=1, k=0, dp=8, mbs=1) == 11
    assert maxtext_row(device=3, micro=1, k=0, mbs=1, ga=2) == 7


@pytest.mark.parametrize("hosts", [1, 2, 4, 8])
def test_perm_with_hosts(hosts):
    """多 host：host h 读样本流的第 h::H 个、全局批按 host 拼接；换序后仍对上 Megatron。"""
    dp, mbs, ga = 8, 1, 2
    gbs = dp * mbs * ga
    stream = np.arange(gbs)[:, None]
    local = gbs // hosts
    loaded = np.concatenate([stream[h::hosts][:local] for h in range(hosts)])   # 照 MaxText 的 host 切法拼出全局批
    assert [maxtext_loaded_slot(g, gbs, hosts) for g in range(gbs)] == loaded[:, 0].tolist()
    perm = maxtext_perm_for_megatron(dp, mbs, ga, hosts)
    got = per_device(loaded[perm], dp, mbs, ga)
    meg = _megatron_view(stream, dp, mbs, ga)
    assert all(np.array_equal(got[k], meg[k]) for k in meg)
    if hosts > 1:   # 不换序时多 host 的全局批连样本顺序都变了
        assert not np.array_equal(loaded, stream)
