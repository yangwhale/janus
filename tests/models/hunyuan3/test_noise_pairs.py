"""混元 3 各检查项目录的噪声参照都是若干份运行两两配对的全部组合。

总览（dashboard/build.py）的说明里写了「k 对 = n 份两两」的换算（3 对 = 3 份 … 45 对 = 10 份），
这条测试守住那句话：哪个目录的 runs.noise 不是完全配对，就要改那句说明。
第 1 步整步的两侧执行噪声并集（GPU 一组 ∪ TPU 一组）是两组互不相交的运行各自完全配对再合起来，
按连通的组分别核对（同一句说明里写了 24 对 = 3 + 21）。
"""

from itertools import combinations

import yaml
from helpers import ROOT


def _catalogs():
    base = ROOT / "models" / "hunyuan3"
    yield from sorted(base.glob("alignments/*/catalogs/*.yaml"))
    yield from sorted(base.glob("regression/*.yaml"))


def _groups(pairs):
    """噪声对按共用运行连成的组（连通分量）。"""
    groups: list[set] = []
    for p in pairs:
        hit = [g for g in groups if g & p]
        merged = set(p).union(*hit)
        groups = [g for g in groups if not (g & p)] + [merged]
    return groups


def test_noise_pairs_are_complete_pairings():
    seen = 0
    for path in _catalogs():
        data = yaml.safe_load(path.read_text())
        runs = data.get("runs") if isinstance(data, dict) else None
        noise = runs.get("noise") if isinstance(runs, dict) else None
        if not isinstance(noise, list) or not noise:
            continue
        pairs = {frozenset(p) for p in noise}
        assert len(noise) == len(pairs), f"{path.relative_to(ROOT)}：噪声对有重复"
        for members in _groups(pairs):
            expect = {frozenset(c) for c in combinations(sorted(members), 2)}
            got = {p for p in pairs if p <= members}
            assert got == expect, f"{path.relative_to(ROOT)}：噪声对里 {len(members)} 份的那一组不是两两的全部组合"
        seen += 1
    assert seen > 0
