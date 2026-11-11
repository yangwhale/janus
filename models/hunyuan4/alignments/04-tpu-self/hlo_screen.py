"""TPU 自比跑前的离线筛选：在 CPU 上把被测臂和它的对照各编一次训练步（0 张卡），比优化后的程序，判三档：

  程序相同     → 预判逐位相同，可以不上 TPU（TPU 同一程序重跑逐位相同，见 MATRIX「A 校准」）
  程序不同     → 要上 TPU 跑
  无法离线判断 → 编不出来，或改的东西在编译产物里看不出来（换 libtpu 版本、换硬件代际、SparseCore 卸载），要上 TPU 跑

程序怎么比：取 XLA 优化后的 HLO（after_optimizations），去掉 metadata（op_name、源码文件与行号）、stack_frame_id、
头部的源码位置表，指令与计算的名字按出现顺序重新编号（名字里的序号随 JAX 追踪时建了多少个同名中间量变，不影响程序），
replica_groups 的 mesh 写法展开成设备号列表（真卡与拓扑描述打印的写法不同，指的是同一组设备）。剩下的逐字相同才算程序相同。
另报 diff 摘要：各类集合通信按 dtype 的个数、scatter 个数与带 indices_are_sorted 的个数、convert 的 dtype 对、矩阵乘输出 dtype、
编译期显存（memory_analysis）。

规格（spec）：golden 的写在 arms.yaml 的 screen_base；每个臂在 arms.yaml 里写 screen（对它的对照 vs 改了什么，只写改了的项）：
  args        追加的 MaxText key=value（同名键后写的生效）
  xla_flags   追加的 XLA_FLAGS 开关
  libtpu_args 追加的 LIBTPU_INIT_ARGS 开关
  maxtext     换 MaxText 提交         steps    换步数（学习率表按它建）
  platform    v5p | v5p-16 | v7x | v7x-16          libtpu   换 libtpu 版本
  screen: {} 表示程序规格与对照完全相同（换机器、换集群、重跑），不用编。

子命令：
  run       按 arms.yaml 筛（--arms 逗号分隔，缺省 = 所有写了 screen 的臂），结果写 results/hlo-screen.json
  derive    从运行目录的 meta.json 推出某个臂的 screen 增量（登记已跑过的臂、核对 arms.yaml 用）
  compare   直接比两份优化后 HLO（例如 CPU 编的和 TPU 运行目录里存的）

  python models/hunyuan4/alignments/04-tpu-self/hlo_screen.py run --arms B1,C1 --maxtext-repo <maxtext 克隆> \
      --cache <缓存目录> [--root <运行目录根>]
  python models/hunyuan4/alignments/04-tpu-self/hlo_screen.py compare <a.txt> <b.txt>

编译在 golden 的镜像里做（docker；镜像 digest 在 screen_base.image），--runner local 改用当前 Python 环境（要装好 jax、libtpu、
maxtext，版本要与镜像一致，否则程序不可比）。每次编译单进程，峰值内存十几 GiB；容器限 --mem（默认 250g）。
"""

from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
GOLDEN = "ts-g-a"
OUT_JSON = HERE / "results" / "hlo-screen.json"
AOT_SCRIPT = "models/hunyuan4/frameworks/maxtext/aot_hlo.py"

SAME, DIFF, NA = "程序相同", "程序不同", "无法离线判断"
# 平台 → MaxText compile_topology（v5p 的名字按 TensorCore 数：v5p-16 = 8 片；v7x 每片两个 TensorCore，tpu7x-8 = 4 片 8 个设备）
TOPOLOGY = {"v5p": "v5p-16", "v5p-16": "v5p-32", "v7x": "tpu7x-8", "v7x-16": "tpu7x-16"}
GEN = {"v5p": "v5p", "v5p-16": "v5p", "v7x": "v7x", "v7x-16": "v7x"}
# 这些开关的作用在 TPU 后端或运行时（SparseCore 上执行集合通信），优化后 HLO 不一定反映，程序相同也不能推出逐位相同
OFFLINE_NA_FLAG = re.compile(r"sparse_core|(?:^|_)sc_")
LIST_KEYS = ("args", "xla_flags", "libtpu_args")
SCALAR_KEYS = ("maxtext", "yml", "yml_rev", "yml_sha256", "stage", "platform", "steps", "libtpu", "image")
DUMP_FLAG = re.compile(r"^--xla_dump")
# meta.config.overrides 里随运行变、和程序无关的键（train_hy4.py 每次都加）
RUN_KEYS = ("run_name", "base_output_directory", "steps", "learning_rate_schedule_steps", "load_parameters_path")


# ---------------- 规格 ----------------


def kv(items):
    """key=value / --flag=value 列表 → 有序字典（同名后写的生效）。没有等号的开关值记 None。"""
    out = {}
    for x in items:
        k, _, v = x.partition("=")
        out.pop(k, None)
        out[k] = v if "=" in x else None
    return out


def apply_delta(base: dict, delta: dict) -> dict:
    s = copy.deepcopy(base)
    for k, v in delta.items():
        if k in LIST_KEYS:
            s[k] = list(s.get(k) or []) + list(v)
        elif k in SCALAR_KEYS:
            s[k] = v
        else:
            raise SystemExit(f"screen 里不认识的键 {k}（只认 {LIST_KEYS + SCALAR_KEYS}）")
    return s


def libtpu_id(x):
    """libtpu 的身份按构建号（cl/…）认；写法可以是「0.0.48 cl/981334199」或只写构建号。"""
    m = re.search(r"cl/\d+", x or "")
    return m.group(0) if m else x


def canon_spec(s: dict) -> dict:
    """比较与缓存用：列表项按生效的键值比（顺序只在同名键之间有意义）；libtpu 按构建号比。"""
    c = {k: s.get(k) for k in SCALAR_KEYS}
    c["libtpu"] = libtpu_id(c["libtpu"])
    for k in LIST_KEYS:
        c[k] = sorted(kv(s.get(k) or []).items(), key=lambda t: t[0])
    return c


def spec_hash(s: dict) -> str:
    c = canon_spec(s)
    c.pop("yml", None)  # yml 内容由 yml_sha256 定，路径不进缓存键
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]


class Arms:
    def __init__(self, path=HERE / "arms.yaml"):
        y = yaml.safe_load(Path(path).read_text())
        self.base = y["screen_base"]
        self.arms = y["arms"]
        self.by_id = {a["id"]: a for a in self.arms}

    def spec_of_run(self, run: str, _seen=()) -> dict | None:
        """一个运行的程序规格：golden 取 screen_base；其余取「run 是它、写了 screen」的臂，在它对照的规格上加增量。
        同一个运行出现在几个臂里时，几种算法必须得到同一份规格。"""
        if run in (GOLDEN, self.base["run"]):
            return {k: v for k, v in self.base.items() if k != "run"}
        if run in _seen:
            raise SystemExit(f"screen 规格循环引用：{_seen + (run,)}")
        got = []
        for a in self.arms:
            if a.get("run") == run and "screen" in a and not a["id"].startswith("L"):
                b = self.spec_of_run(a.get("vs", GOLDEN), _seen + (run,))
                if b is not None:
                    got.append((a["id"], apply_delta(b, a["screen"] or {})))
        if not got:
            return None
        for aid, s in got[1:]:
            if canon_spec(s) != canon_spec(got[0][1]):
                raise SystemExit(f"运行 {run} 在 {got[0][0]} 与 {aid} 里推出的规格不同，arms.yaml 的 screen 有误")
        return got[0][1]


def spec_from_meta(run_dir: Path) -> dict:
    """运行目录 meta.json → 程序规格（与 screen_base 同一组键）。"""
    m = json.loads((run_dir / "meta.json").read_text())
    code = {c["repo"].split("/")[-1]: c["commit"] for c in m.get("code") or []}
    hw = m["hardware"]
    gen = "v7x" if "7" in hw["accelerator"] else "v5p"
    platform = gen if hw["count"] == 8 else f"{gen}-{hw['count']}"
    ov = list(m["config"]["overrides"])
    idx = next((i for i, x in enumerate(ov) if x.startswith("load_parameters_path=")), None)
    args = (
        ov[idx + 1 :]
        if idx is not None
        else [x for x in ov if x.split("=")[0] not in RUN_KEYS + ("base_num_decoder_layers", "mtp_num_layers")]
    )
    det = m.get("determinism") or {}
    v = m.get("versions") or {}
    cl = re.search(r"cl/\d+", v.get("libtpu") or "")
    pip = (v.get("libtpu_pip") or "").removeprefix("libtpu==")
    path = m["config"].get("path")  # 个别运行的 meta 是从运行中镜像补的，没有 yml 路径
    return {
        "maxtext": code.get("maxtext"),
        "yml": f"models/hunyuan4/frameworks/maxtext/{Path(path).name}" if path else None,
        "yml_rev": code.get("janus"),
        "yml_sha256": m["config"].get("sha256"),
        "stage": m["stage"],
        "platform": platform,
        "steps": (m["config"].get("args") or {}).get("steps") or (m.get("steps") or {}).get("last"),
        "args": args,
        "xla_flags": [f for f in (det.get("XLA_FLAGS") or "").split() if not DUMP_FLAG.match(f)],
        "libtpu_args": (det.get("LIBTPU_INIT_ARGS") or "").split(),
        "libtpu": " ".join(x for x in (pip, cl.group(0) if cl else "") if x) or None,
        "image": m.get("image"),
    }


def program_delta(test: dict, base: dict) -> dict:
    """两份规格里影响程序的差别（screen 增量的写法）。yml_rev 只在 yml 内容（sha256）变了时算差别。"""
    d = {}
    for k in ("maxtext", "stage", "platform", "steps", "image"):
        if test.get(k) != base.get(k):
            d[k] = test.get(k)
    if libtpu_id(test.get("libtpu")) != libtpu_id(base.get("libtpu")):
        d["libtpu"] = test.get("libtpu")
    if test.get("yml_sha256") != base.get("yml_sha256") or test.get("yml") != base.get("yml"):
        d.update({"yml": test["yml"], "yml_rev": test["yml_rev"], "yml_sha256": test["yml_sha256"]})
    for k in LIST_KEYS:
        t, b = kv(test.get(k) or []), kv(base.get(k) or [])
        if t != b:
            removed = [x for x in b if x not in t]
            if removed:
                raise SystemExit(f"{k} 里对照有、被测没有的项 {removed}：screen 只能追加，换个对照或手写")
            d[k] = [x if t[x] is None else f"{x}={t[x]}" for x in t if t[x] != b.get(x, object())]
    return d


def offline_na_reasons(test: dict, base: dict) -> list[str]:
    """改的东西在编译产物里看不出来：返回原因（空 = 可以离线判）。"""
    why = []
    if libtpu_id(test.get("libtpu")) != libtpu_id(base.get("libtpu")):
        why.append(
            f"换 libtpu 版本（{base.get('libtpu')} → {test.get('libtpu')}）：编译器后端与运行时一起换了，HLO 相同也不保证逐位"
        )
    if test.get("image") != base.get("image"):
        why.append("换镜像：库版本可能一起换了")
    if GEN[test["platform"]] != GEN[base["platform"]]:
        why.append(f"换硬件代际（{base['platform']} → {test['platform']}）：两边程序本来就不可比")
    for k in ("xla_flags", "libtpu_args"):
        t, b = kv(test.get(k) or []), kv(base.get(k) or [])
        for f in sorted(set(t) | set(b)):
            if t.get(f, "∅") != b.get(f, "∅") and OFFLINE_NA_FLAG.search(f):
                why.append(f"SparseCore 开关 {f}：集合通信卸载到 SparseCore 上执行，效果不一定反映在优化后 HLO 里")
    return why


# ---------------- HLO 规范化与摘要 ----------------

META = re.compile(r",?\s*metadata=\{[^{}]*\}")
STACK = re.compile(r",?\s*stack_frame_id=\d+")
TABLES = ("FileNames", "FunctionNames", "FileLocations", "StackFrames")
HDR_PARAM = re.compile(r"(\(|, )[\w.\-]+: ")
NAME = re.compile(r"%[\w.\-]+")
MESH_RG = re.compile(r"replica_groups=mesh\[([^\]]*)\] \{([^}]*)\}")
INSTR = re.compile(
    r"^\s*(?:ROOT )?(%[\w.\-]+) = (\((?:[^=]|/\*index=\d+\*/)*?\)|\S+) ([a-z][a-z0-9-]*)\(([^)]*)\)(.*)$"
)
DTYPE = re.compile(r"\b(pred|[subf]\d+|bf16|f8e\w+|c\d+|token)\[")
COLL = (
    "all-reduce",
    "all-gather",
    "reduce-scatter",
    "all-to-all",
    "ragged-all-to-all",
    "collective-permute",
    "collective-broadcast",
)


def _expand_mesh_rg(m) -> str:
    """mesh['a'=2,'b'=4] {'b'} → 显式设备号列表（按 mesh 的行优先编号；列出的轴在组内变化，其余轴分组）。"""
    axes = [(k.strip().strip("'"), int(v)) for k, v in (x.split("=") for x in m.group(1).split(","))]
    red = [x.strip().strip("'") for x in m.group(2).split(",") if x.strip()]
    names = [a for a, _ in axes]
    size = dict(axes)
    stride, s = {}, 1
    for a, n in reversed(axes):
        stride[a] = s
        s *= n

    def combos(keys):
        out = [0]
        for k in keys:
            out = [o + i * stride[k] for o in out for i in range(size[k])]
        return out

    keep = [a for a in names if a not in red]
    groups = [[g + r for r in combos(red)] for g in combos(keep)]
    return "replica_groups={" + ",".join("{" + ",".join(map(str, g)) + "}" for g in groups) + "}"


PARAM = re.compile(r"^\s*(?:ROOT )?(%[\w.\-]+) = .* parameter\((\d+)\)")


def _params_first(lines: list[str]) -> list[str]:
    """每个计算里的 parameter 指令按参数号排到最前、名字按参数号定。parameter 没有副作用，它在计算里打印的先后不影响程序；
    CPU 上编和 TPU 上编，归约子计算里 parameter(0)、parameter(1) 的先后偶尔相反。"""
    out, i, k = [], 0, 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        if not (line.endswith("{") and (line.startswith("%") or line.startswith("ENTRY "))):
            continue
        j = i
        while j < len(lines) and lines[j] != "}":
            j += 1
        body = lines[i:j]
        params = sorted(((int(m.group(2)), m.group(1), x) for x in body if (m := PARAM.match(x))), key=lambda t: t[0])
        ren = {old: f"%c{k}p{n}" for n, old, _ in params}
        rest = [x for x in body if not PARAM.match(x)]
        block = [x for _, _, x in params] + rest
        if ren:
            pat = re.compile("|".join(re.escape(o) + r"(?![\w.\-])" for o in ren))
            block = [pat.sub(lambda m: ren[m.group(0)], x) for x in block]
        out.extend(block)
        i, k = j, k + 1
    return out


def canon_lines(text: str) -> list[str]:
    t = STACK.sub("", META.sub("", text))
    out, skip = [], False
    for line in t.splitlines():
        s = line.strip()
        if s in TABLES:
            skip = True
            continue
        if skip and (not s or re.match(r"^\d+ ", s)):
            continue
        skip = False
        if line.endswith("{") and (line.startswith("%") or line.startswith("ENTRY ")):
            line = HDR_PARAM.sub(r"\1", line)
        out.append(MESH_RG.sub(_expand_mesh_rg, line))
    out = _params_first(out)
    ids: dict[str, str] = {}

    def sub(m):
        k = m.group(0)
        if k not in ids:
            ids[k] = f"%i{len(ids)}"
        return ids[k]

    return [NAME.sub(sub, x) for x in out]


def program_fp(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def operand_type(types: dict, operand: str) -> str:
    """操作数的类型；%x#k（元组的第 k 个元素）取元组类型里的第 k 项。"""
    name, _, k = operand.partition("#")
    ty = types.get(name, "")
    if not k or not ty.startswith("("):
        return ty
    parts, depth, cur = [], 0, ""
    for ch in ty[1:-1]:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts[int(k)] if int(k) < len(parts) else ""


def hlo_summary(text: str) -> dict:
    """优化后 HLO 的摘要：集合通信（按 opcode·dtype）、scatter（个数、标了 indices_are_sorted 的个数、dtype）、
    convert 的 dtype 对、dot / convolution 的输出 dtype、reduce-precision 个数。含融合体内部；异步的 -start 记进同名操作，-done 不数。"""
    types: dict[str, str] = {}
    rows = []
    for line in text.splitlines():
        m = INSTR.match(line)
        if m:
            name, ty, op, args, rest = m.groups()
            types[name] = ty
            rows.append((ty, op, args, rest))
    coll, scat, conv, dots, other = Counter(), Counter(), Counter(), Counter(), Counter()
    for ty, op, args, rest in rows:
        dts = "/".join(sorted(set(DTYPE.findall(ty)))) or "?"
        base = op.removesuffix("-start")
        if op.endswith("-done"):
            continue
        if base in COLL:
            coll[f"{base} {dts}"] += 1
        elif op == "scatter":
            scat[f"scatter {dts}" + (" sorted" if "indices_are_sorted=true" in rest else "")] += 1
        elif op == "convert":
            src = DTYPE.findall(operand_type(types, args.split(",")[0].strip()))
            conv[f"{src[0] if src else '?'}→{dts}"] += 1
        elif op in ("dot", "convolution"):
            dots[f"{op} {dts}"] += 1
        elif op in ("reduce-precision", "sort", "custom-call", "fusion"):
            other[op] += 1
    return {
        "collectives": dict(sorted(coll.items())),
        "scatter": dict(sorted(scat.items())),
        "convert": dict(sorted(conv.items())),
        "dot": dict(sorted(dots.items())),
        "other": dict(sorted(other.items())),
    }


def summary_diff(t: dict, b: dict) -> dict:
    out = {}
    for sec in t.keys() | b.keys():
        x, y = t.get(sec, {}), b.get(sec, {})
        d = {
            k: {"test": x.get(k, 0), "base": y.get(k, 0)}
            for k in sorted(x.keys() | y.keys())
            if x.get(k, 0) != y.get(k, 0)
        }
        if d:
            out[sec] = d
    return out


def first_hunks(a: list[str], b: list[str], n=3, width=240, max_lines=24) -> list[str]:
    """规范化后程序的头几处差别（给人看从哪开始不同）；最多 max_lines 行。"""
    out, k = [], 0
    for line in difflib.unified_diff(b, a, lineterm="", n=0):
        if line.startswith(("---", "+++")):
            continue
        if line.startswith("@@"):
            k += 1
            if k > n:
                break
        if len(out) >= max_lines:
            break
        out.append(line[:width])
    return out


def compare_texts(test_text: str, base_text: str) -> dict:
    a, b = canon_lines(test_text), canon_lines(base_text)
    same = a == b
    r = {
        "program_equal": same,
        "fp": {"test": program_fp(a), "base": program_fp(b)},
        "lines": {"test": len(a), "base": len(b)},
    }
    if not same:
        r["summary_diff"] = summary_diff(hlo_summary(test_text), hlo_summary(base_text))
        r["first_hunks"] = first_hunks(a, b)
    return r


# ---------------- 编译 ----------------


def maxtext_tree(repo: Path, commit: str, cache: Path) -> Path:
    d = cache / "maxtext" / commit
    if not (d / "src" / "maxtext" / "__init__.py").exists():
        d.mkdir(parents=True, exist_ok=True)
        tar = subprocess.run(
            ["git", "-C", str(repo), "archive", commit, "src/maxtext"], check=True, capture_output=True
        ).stdout
        subprocess.run(["tar", "-x", "-C", str(d)], input=tar, check=True)
    return d / "src" / "maxtext"


def yml_file(spec: dict, cache: Path) -> Path:
    data = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{spec['yml_rev']}:{spec['yml']}"], check=True, capture_output=True
    ).stdout
    sha = hashlib.sha256(data).hexdigest()
    if spec.get("yml_sha256") and sha != spec["yml_sha256"]:
        raise SystemExit(
            f"{spec['yml']}@{spec['yml_rev'][:10]} 的 sha256 {sha[:12]} ≠ 规格里的 {spec['yml_sha256'][:12]}"
        )
    p = cache / "yml" / sha / Path(spec["yml"]).name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def compile_spec(spec: dict, cache: Path, mt_repo: Path, runner: str, mem: str, log=print) -> dict:
    """编一份规格（有缓存就直接用），返回 {dir, aot, hlo（优化后 HLO 路径）}；编不出来时 hlo 为 None。"""
    h = spec_hash(spec)
    d = cache / "aot" / h
    done = d / "aot.json"
    if not (done.exists() and (d / "train_step.after_optimizations.txt").exists()):
        if d.exists():
            shutil.rmtree(d)
        (d / "xla").mkdir(parents=True)
        (d / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1))
        mt = maxtext_tree(mt_repo, spec["maxtext"], cache)
        yml = yml_file(spec, cache)
        xla = (
            " ".join(spec.get("xla_flags") or [])
            + " --xla_dump_to={X} --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train_step.*"
        )
        envs = {
            "JAX_PLATFORMS": "cpu",
            "ALLOW_MULTIPLE_LIBTPU_LOAD": "1",
            "TPU_SKIP_MDS_QUERY": "1",
            "LIBTPU_INIT_ARGS": " " + " ".join(spec.get("libtpu_args") or []),
        }
        args = [
            "--stage",
            spec["stage"],
            "--topology",
            TOPOLOGY[spec["platform"]],
            "--steps",
            str(spec["steps"]),
            *spec.get("args", []),
        ]
        t0 = time.time()
        if runner == "docker":
            envs.update(
                {
                    "PYTHONPATH": "/janus:/deps/src",
                    "HY4_YML": f"/yml/{yml.name}",
                    "XLA_FLAGS": xla.format(X="/out/xla").strip(),
                    "HOME": "/tmp",
                }
            )
            cmd = [
                "docker",
                "run",
                "--rm",
                f"--memory={mem}",
                "-u",
                f"{os.getuid()}:{os.getgid()}",
                "-v",
                f"{mt}:/deps/src/maxtext:ro",
                "-v",
                f"{REPO}:/janus:ro",
                "-v",
                f"{yml}:/yml/{yml.name}:ro",
                "-v",
                f"{d}:/out",
            ]
            for k, v in envs.items():
                cmd += ["-e", f"{k}={v}"]
            cmd += [
                "--entrypoint",
                "python3",
                spec["image"],
                f"/janus/{AOT_SCRIPT}",
                *args[:6],
                "--out",
                "/out",
                *args[6:],
            ]
        else:
            envs.update(
                {"PYTHONPATH": f"{REPO}:{mt.parent}", "HY4_YML": str(yml), "XLA_FLAGS": xla.format(X=d / "xla").strip()}
            )
            cmd = [sys.executable, str(REPO / AOT_SCRIPT), *args[:6], "--out", str(d), *args[6:]]
        log(
            f"[hlo_screen] 编译 {h}：{' '.join(spec.get('args') or [])} {' '.join(spec.get('xla_flags') or [])} "
            f"maxtext {spec['maxtext'][:8]} {spec['platform']}"
        )
        with open(d / "compile.log", "w") as f:
            rc = subprocess.run(
                cmd, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, **envs) if runner != "docker" else None
            ).returncode
        fs = sorted((d / "xla").glob("module_*.jit_train_step.*after_optimizations.txt"))
        keep = {
            "after_optimizations.txt": fs,
            "after_optimizations-memory-usage-report.txt": sorted(
                (d / "xla").glob("module_*.jit_train_step.*after_optimizations-memory-usage-report.txt")
            ),
        }
        aot = json.loads(done.read_text()) if done.exists() else {"error": f"aot_hlo.py 没写 aot.json（退出码 {rc}）"}
        aot["returncode"], aot["host_wall_seconds"] = rc, round(time.time() - t0, 1)
        if len(fs) > 1:
            aot["error"] = (aot.get("error") or "") + f" jit_train_step 编出 {len(fs)} 份"
        for suffix, src in keep.items():
            if len(src) == 1:
                shutil.move(str(src[0]), d / f"train_step.{suffix}")
        shutil.rmtree(d / "xla", ignore_errors=True)
        done.write_text(json.dumps(aot, ensure_ascii=False, indent=1))
    aot = json.loads(done.read_text())
    hlo = d / "train_step.after_optimizations.txt"
    return {"dir": str(d), "hash": h, "aot": aot, "hlo": hlo if hlo.exists() and not aot.get("error") else None}


def mem_info(aot: dict) -> dict | None:
    ma = aot.get("memory_analysis")
    if not ma:
        return None
    peak = (
        ma["argument_size_in_bytes"] + ma["output_size_in_bytes"] - ma["alias_size_in_bytes"] + ma["temp_size_in_bytes"]
    )
    return {
        "temp_bytes": ma["temp_size_in_bytes"],
        "argument_bytes": ma["argument_size_in_bytes"],
        "peak_estimate_bytes": peak,
    }


# ---------------- 判一个臂 ----------------


def screen_arm(arms: Arms, aid: str, cache: Path, mt_repo: Path, runner: str, mem: str, root: Path | None) -> dict:
    a = arms.by_id[aid]
    vs = a.get("vs", GOLDEN)
    rec = {"arm": aid, "run": a.get("run"), "vs": vs, "screen": a.get("screen")}
    test, base = arms.spec_of_run(a["run"]), arms.spec_of_run(vs)
    if test is None or base is None:
        rec.update(verdict=None, reason="arms.yaml 里没登记规格（screen）")
        return rec
    # 运行目录在手边时，核对 arms.yaml 推出的规格与实际跑的是同一个程序规格（防 screen 写错）
    if root is not None:
        for name, s in (("test", test), ("base", base)):
            r = a["run"] if name == "test" else vs
            md = root / r / "meta.json"
            if md.exists():
                m = spec_from_meta(md.parent)
                diff = program_delta(m, s) if canon_spec(m) != canon_spec(s) else {}
                try:
                    back = program_delta(s, m) if diff else {}
                except SystemExit as e:
                    back = {"error": str(e)}
                rec.setdefault("meta_check", {})[name] = (
                    "一致" if not diff else {"meta_minus_spec": diff, "spec_minus_meta": back}
                )
    # TPU 上两次运行实际存下的程序按同一套规范化比一次（回测的「实际」；比 tpu_self_compare 的程序指纹多做了重编号）
    tpu = {n: tpu_canon(root, r) for n, r in (("test", a["run"]), ("base", vs))} if root is not None else {}
    if tpu and all(x is not None for x in tpu.values()):
        rec["tpu_program_equal"] = tpu["test"] == tpu["base"]
    rec["delta"] = program_delta(test, base) if canon_spec(test) != canon_spec(base) else {}
    if not rec["delta"]:
        rec.update(verdict=SAME, reason="程序规格与对照完全相同（只换机器、集群或重跑），不用编")
        return rec
    na = offline_na_reasons(test, base)
    can_compile = not any(r.startswith(("换 libtpu", "换镜像", "换硬件")) for r in na)
    if can_compile:
        ct, cb = (compile_spec(s, cache, mt_repo, runner, mem) for s in (test, base))
        rec["compile"] = {
            k: {
                "hash": c["hash"],
                "seconds": c["aot"].get("compile_seconds"),
                "wall_seconds": c["aot"].get("wall_seconds"),
                "max_rss_gib": c["aot"].get("max_rss_gib"),
                "memory": mem_info(c["aot"]),
                "error": c["aot"].get("error"),
            }
            for k, c in (("test", ct), ("base", cb))
        }
        if ct["hlo"] is None or cb["hlo"] is None:
            rec.update(
                verdict=NA,
                reason="编不出来："
                + "；".join(
                    f"{k} {c['aot'].get('error', '')[:300]}"
                    for k, c in (("被测", ct), ("对照", cb))
                    if c["hlo"] is None
                ),
            )
            return rec
        rec["hlo"] = compare_texts(ct["hlo"].read_text(), cb["hlo"].read_text())
        if root is not None:  # 校准：CPU 编的和 TPU 运行实际存下的是不是同一个程序
            for name, c in (("test", ct), ("base", cb)):
                if tpu.get(name) is not None:
                    rec.setdefault("cpu_vs_tpu", {})[name] = tpu[name] == canon_lines(c["hlo"].read_text())
    if na:
        rec.update(verdict=NA, reason="；".join(na))
    else:
        rec.update(
            verdict=SAME if rec["hlo"]["program_equal"] else DIFF,
            reason="优化后程序逐字相同（去掉元数据、重编号后）" if rec["hlo"]["program_equal"] else "优化后程序不同",
        )
    return rec


_TPU_CANON: dict = {}


def tpu_canon(root: Path, run: str):
    """运行目录 hlo/ 里 TPU 上存下的训练步优化后 HLO，规范化后的行；没有或不止一份时 None。"""
    if run not in _TPU_CANON:
        d = root / run / "hlo"
        fs = sorted(d.glob("*after_optimizations.txt")) if d.is_dir() else []
        _TPU_CANON[run] = canon_lines(fs[0].read_text()) if len(fs) == 1 else None
        if len(_TPU_CANON) > 6:  # 每份几十 MB，只留最近几份
            _TPU_CANON.pop(next(iter(_TPU_CANON)))
    return _TPU_CANON[run]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument(
        "--arms", default=None, help="逗号分隔的臂编号；缺省 = arms.yaml 里写了 screen 的全部臂（不含 L 类）"
    )
    r.add_argument(
        "--maxtext-repo",
        type=Path,
        default=os.environ.get("MAXTEXT_REPO"),
        required=os.environ.get("MAXTEXT_REPO") is None,
        help="maxtext 的本地克隆（要有规格里用到的提交）",
    )
    r.add_argument("--cache", type=Path, required=True, help="编译产物缓存（每份规格一个目录，优化后 HLO 约 35 MB）")
    r.add_argument(
        "--root", type=Path, default=None, help="运行目录根（可选）：有就核对规格与 meta.json、CPU 编的与 TPU 存的 HLO"
    )
    r.add_argument("--runner", choices=("docker", "local"), default="docker")
    r.add_argument("--mem", default="250g", help="容器内存上限")
    r.add_argument("--out", type=Path, default=OUT_JSON, help="结果 JSON（同一臂覆盖旧记录，其余保留）")
    d = sub.add_parser("derive")
    d.add_argument("--root", type=Path, required=True)
    d.add_argument("--arms", required=True)
    c = sub.add_parser("compare")
    c.add_argument("test")
    c.add_argument("base")
    a = p.parse_args(argv)
    arms = Arms()
    if a.cmd == "compare":
        res = compare_texts(Path(a.test).read_text(), Path(a.base).read_text())
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0 if res["program_equal"] else 1
    if a.cmd == "derive":
        for aid in a.arms.split(","):
            x = arms.by_id[aid]
            vs = x.get("vs", GOLDEN)
            t = spec_from_meta(a.root / x["run"])
            b = spec_from_meta(a.root / vs) if (a.root / vs / "meta.json").exists() else arms.spec_of_run(vs)
            print(f"{aid}: screen: {json.dumps(program_delta(t, b), ensure_ascii=False)}")
        return 0
    ids = a.arms.split(",") if a.arms else [x["id"] for x in arms.arms if "screen" in x and not x["id"].startswith("L")]
    a.cache.mkdir(parents=True, exist_ok=True)
    old = json.loads(a.out.read_text()) if a.out.exists() else {}
    recs = {x["arm"]: x for x in old.get("arms", [])}
    for aid in ids:
        rec = screen_arm(arms, aid, a.cache, a.maxtext_repo, a.runner, a.mem, a.root)
        recs[aid] = rec
        print(f"[hlo_screen] {aid}: {rec['verdict']}（{rec['reason'][:120]}）", flush=True)
        order = [x["id"] for x in arms.arms]
        a.out.write_text(
            json.dumps(
                {
                    "note": "TPU 自比跑前的离线筛选（hlo_screen.py）：每个臂的预判、理由、规格增量、两边编译的耗时与显存、"
                    "程序比较与 diff 摘要；cpu_vs_tpu = CPU 编出的程序与该运行 TPU 上实际存下的是否相同",
                    "image": arms.base.get("image"),
                    "arms": sorted(recs.values(), key=lambda x: order.index(x["arm"]) if x["arm"] in order else 1e9),
                },
                ensure_ascii=False,
                indent=1,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
