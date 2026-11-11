"""04-tpu-self/hlo_screen.py：程序规范化（去元数据、重编号、展开 mesh 写法的 replica_groups）、规格增量、离线判不了的类别。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models" / "hunyuan4" / "alignments" / "04-tpu-self"))

import hlo_screen as h  # noqa: E402

A = """\
HloModule jit_train_step, is_scheduled=true

%region_3.17 (psum.49: f32[], psum.50: f32[]) -> f32[] {
  %psum.49 = f32[] parameter(0)
  %psum.50 = f32[] parameter(1)
  ROOT %add.1403 = f32[] add(%psum.49, %psum.50), metadata={op_name="jit(train_step)/add" source_file="/deps/a.py" source_line=10}
}

ENTRY %main.9 (p.1: bf16[8,4]) -> f32[8,4] {
  %p.1 = bf16[8,4]{1,0} parameter(0)
  %convert.7 = f32[8,4]{1,0} convert(%p.1), metadata={op_name="x" stack_frame_id=12}
  ROOT %all-reduce.3 = f32[8,4]{1,0} all-reduce(%convert.7), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}}, use_global_device_ids=true, to_apply=%region_3.17
}

FileNames
1 "/deps/a.py"
"""

# 同一个程序：名字的序号全变了，元数据与源码位置表不同，replica_groups 用 mesh 写法
B = """\
HloModule jit_train_step, is_scheduled=true

%region_5.20 (psum.60: f32[], psum.61: f32[]) -> f32[] {
  %psum.60 = f32[] parameter(0)
  %psum.61 = f32[] parameter(1)
  ROOT %add.1441 = f32[] add(%psum.60, %psum.61), metadata={op_name="jit(train_step)/add" source_file="/deps/a.py" source_line=99}
}

ENTRY %main.11 (p.2: bf16[8,4]) -> f32[8,4] {
  %p.2 = bf16[8,4]{1,0} parameter(0)
  %convert.9 = f32[8,4]{1,0} convert(%p.2)
  ROOT %all-reduce.5 = f32[8,4]{1,0} all-reduce(%convert.9), channel_id=1, replica_groups=mesh['data'=1,'fsdp'=8] {'fsdp'}, use_global_device_ids=true, to_apply=%region_5.20
}

FileNames
1 "/deps/b.py"
2 "/deps/c.py"
"""


def test_same_program_after_canon():
    assert h.canon_lines(A) == h.canon_lines(B)
    assert h.compare_texts(A, B)["program_equal"]


def test_dtype_change_is_a_different_program():
    c = B.replace("ROOT %all-reduce.5 = f32[8,4]", "ROOT %all-reduce.5 = bf16[8,4]")
    r = h.compare_texts(c, A)
    assert not r["program_equal"]
    assert r["summary_diff"]["collectives"] == {
        "all-reduce bf16": {"test": 1, "base": 0},
        "all-reduce f32": {"test": 0, "base": 1},
    }


def test_mesh_replica_groups_expand_row_major():
    import re

    def ex(s):
        return h.MESH_RG.sub(h._expand_mesh_rg, s)

    assert ex("replica_groups=mesh['a'=2,'b'=4] {'b'}") == "replica_groups={{0,1,2,3},{4,5,6,7}}"
    assert ex("replica_groups=mesh['a'=2,'b'=4] {'a'}") == "replica_groups={{0,4},{1,5},{2,6},{3,7}}"
    assert ex("replica_groups=mesh['axis_0'=8] {'axis_0'}") == "replica_groups={{0,1,2,3,4,5,6,7}}"
    assert re.fullmatch(r"replica_groups=\{\{0,1\},\{2,3\}\}", ex("replica_groups=mesh['x'=2,'y'=2] {'y'}"))


def test_summary_counts_sorted_scatter_and_convert_pairs():
    t = """\
ENTRY %m (p: f32[4]) -> bf16[16,8] {
  %i = s32[4,1]{1,0} parameter(1)
  %u = bf16[4,8]{1,0} parameter(2)
  %z = bf16[16,8]{1,0} parameter(3)
  %s1 = bf16[16,8]{1,0} scatter(%z, %i, %u), update_window_dims={1}, indices_are_sorted=true, to_apply=%add
  %s2 = bf16[16,8]{1,0} scatter(%s1, %i, %u), update_window_dims={1}, to_apply=%add
  %c = f32[16,8]{1,0} convert(%s2)
  ROOT %ag = (f32[16,8]{1,0}, f32[16,8]{1,0}) all-gather-start(%c), dimensions={0}
  %agd = f32[16,8]{1,0} all-gather-done(%ag)
}
"""
    s = h.hlo_summary(t)
    assert s["scatter"] == {"scatter bf16": 1, "scatter bf16 sorted": 1}
    assert s["convert"] == {"bf16→f32": 1}
    assert s["collectives"] == {"all-gather f32": 1}


BASE = {
    "maxtext": "m0",
    "yml": "y.yml",
    "yml_rev": "r0",
    "yml_sha256": "s0",
    "stage": "mini",
    "platform": "v5p",
    "steps": 1000,
    "args": [],
    "xla_flags": [],
    "libtpu_args": ["--a=true"],
    "libtpu": "0.0.46 cl/1",
    "image": "img",
}


def test_delta_round_trip_and_libtpu_identity():
    d = {"args": ["ici_data_parallelism=8", "ici_fsdp_parallelism=1"], "steps": 300, "libtpu_args": ["--b=false"]}
    t = h.apply_delta(BASE, d)
    assert h.program_delta(t, BASE) == d
    # 同一个 libtpu 构建，写法不同不算差别
    assert h.program_delta(dict(BASE, libtpu="cl/1"), BASE) == {}
    assert h.canon_spec(dict(BASE, libtpu="cl/1")) == h.canon_spec(BASE)


def test_offline_na_categories():
    assert h.offline_na_reasons(BASE, BASE) == []
    assert h.offline_na_reasons(dict(BASE, libtpu="0.0.48 cl/2"), BASE)[0].startswith("换 libtpu")
    assert h.offline_na_reasons(dict(BASE, platform="v7x"), BASE)[0].startswith("换硬件代际")
    sc = h.apply_delta(BASE, {"libtpu_args": ["--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true"]})
    assert h.offline_na_reasons(sc, BASE)[0].startswith("SparseCore")
    sc2 = h.apply_delta(BASE, {"libtpu_args": ["--xla_sc_force_bf16_collective_reductions_to_f32=true"]})
    assert h.offline_na_reasons(sc2, BASE)
    # 同代换规模、普通编译开关都可以离线判
    assert (
        h.offline_na_reasons(
            h.apply_delta(BASE, {"platform": "v5p-16", "xla_flags": ["--xla_allow_excess_precision=false"]}), BASE
        )
        == []
    )


def test_every_registered_arm_resolves():
    arms = h.Arms()
    n = 0
    for a in arms.arms:
        if "screen" in a and not a["id"].startswith("L"):
            assert arms.spec_of_run(a["run"]) is not None, a["id"]
            assert arms.spec_of_run(a.get("vs", h.GOLDEN)) is not None, a["id"]
            n += 1
    assert n > 0
