"""检查项目录：读入与校验。格式见 docs/reference/catalog-schema.md（schema_version 1）。"""
from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from janus.checks import REGISTRY

LEVELS = {"L0", "L1", "L2", "L3", "L4", "regress"}
SELECT_KINDS = {"tensor", "metrics", "tokens", "checkpoint", "config", "artifact"}
# 这些类型的判据写死在类型里（逐位、计数规则、自证），目录里不用写 threshold
NO_THRESHOLD = {"coverage", "scalar_series", "config_diff", "data_fingerprint", "checkpoint_equal", "replica_consistency",
                "overlap_replay", "artifact_check", "health_signals", "determinism", "poison_selftest", "bitwise_equal",
                "dtype_manifest", "fake_fp32",       # 判据写死：dtype 规则；低 16 位全 0 比例（默认门槛自带 source）
                "ulp_distribution",   # ulp_distribution 不写门槛时只诊断
                "run_complete"}       # 判据写死：步数、重复、重启


# threshold: pending（待标定）：按「没有噪声参照」判，最多 inconclusive；只给判定里用 calibrated 规则的序列类检查
PENDING_OK = {"loss_series", "router_stats"}
# 不写 hard 时默认只报不拦的类型（规格写明「默认只诊断」的）；其余默认 hard: true
SOFT_BY_DEFAULT = {"ulp_distribution"}


class CatalogError(Exception):
    """目录文件不合规：janus run 直接退出码 2。"""


# 目录指纹（结果里的 catalog.content_sha256）只管「查什么、怎么判」，不管说明文字。
# content-v1：解析 YAML（注释随之丢掉）→ 按位置删掉下面这些纯描述键 → JSON（键排序、无空白、浮点按 repr）→ sha256。
# 名单之外的键一律算内容：runs、select、threshold、metrics、params（含 params 里的 source）、hard、compare、
# foreach、vars、include、row / role / expect（regress.py 读）……新加键默认算内容，确认只是说明文字才加进来。
HASH_ALGO = "content-v1"
DESCRIPTIVE_TOP_KEYS = frozenset({"description"})                                   # 顶层
DESCRIPTIVE_CHECK_KEYS = frozenset({"description", "notes", "source", "pending_note"})  # checks 的每一项


def _canon(obj):
    """转成 JSON 能逐字节复现的值：键转字符串，日期写 ISO 8601，其余类型报错（不靠 default=str 猜）。"""
    if isinstance(obj, dict):
        return {str(k): _canon(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon(x) for x in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    raise CatalogError(f"目录里有算不了指纹的值：{type(obj).__name__} {obj!r}")


def content_view(data) -> object:
    """目录（或 include 片段）解析后的内容，去掉纯描述键。不展开 include / vars / foreach：按文件原样算。"""
    if not isinstance(data, dict):
        return data
    out = {k: v for k, v in data.items() if k not in DESCRIPTIVE_TOP_KEYS}
    if isinstance(out.get("checks"), list):
        out["checks"] = [{k: v for k, v in c.items() if k not in DESCRIPTIVE_CHECK_KEYS} if isinstance(c, dict) else c
                         for c in out["checks"]]
    return out


def content_bytes(text: str) -> bytes:
    """content-v1 的规范序列化：JSON、键排序、无空白、UTF-8 原文；浮点用 Python 的 repr（最短往返表示），
    所以 1e-3 与 0.001 是同一个值，1 与 1.0 不是。"""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise CatalogError(f"YAML 解析失败：{e}") from e
    return json.dumps(_canon(content_view(data)), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=True).encode()


def content_sha256(src) -> str:
    """目录文件的内容指纹（content-v1）。src 是路径，或目录文件的文本 / 字节（迁移脚本拿 git 历史版本时用）。"""
    if isinstance(src, Path):
        src = src.read_bytes()
    if isinstance(src, bytes):
        src = src.decode()
    return hashlib.sha256(content_bytes(src)).hexdigest()


def display_path(p) -> str:
    """结果与报告里记的目录路径：在某个 git 仓库里就写成相对该仓库根的路径，免得把本机绝对路径（含用户名）写进入库文件。"""
    p = Path(p)
    if not p.is_absolute():
        return p.as_posix()
    for parent in p.resolve().parents:
        if (parent / ".git").exists():
            return p.resolve().relative_to(parent).as_posix()
    return p.as_posix()


def fingerprint(p: Path) -> dict:
    """结果里记的一份目录文件：路径（仓库相对，见 display_path）、内容指纹、原文件 sha256（只留作历史，核对用 content_sha256）。"""
    raw = Path(p).read_bytes()
    return {"path": display_path(p), "hash_algo": HASH_ALGO, "content_sha256": content_sha256(raw),
            "raw_sha256": hashlib.sha256(raw).hexdigest()}


def recorded_hash(entry: dict) -> tuple[str, str]:
    """结果里一份目录记录的（算法, 哈希）：content-v1 记的是 content_sha256；旧结果与脚本写的结果只有 sha256（原文件）。"""
    if entry.get("hash_algo") == HASH_ALGO:
        return HASH_ALGO, entry["content_sha256"]
    return "raw", entry.get("sha256") or entry.get("raw_sha256")


@dataclass
class Catalog:
    path: Path
    data: dict

    @property
    def runs(self) -> dict:
        return self.data["runs"]

    @property
    def checks(self) -> list[dict]:
        return self.data["checks"]


def _fill(obj, var, val):
    if isinstance(obj, str):
        return obj.replace("{" + var + "}", str(val))
    if isinstance(obj, list):
        return [_fill(x, var, val) for x in obj]
    if isinstance(obj, dict):
        return {k: _fill(v, var, val) for k, v in obj.items()}
    return obj


def expand(checks: list) -> list:
    """展开 foreach：{变量: [取值…]}，把条目里所有字符串的 {变量} 换成每个取值，一条变多条。可多个变量（笛卡尔积）。"""
    out = []
    for c in checks or []:
        fe = c.get("foreach")
        if not fe:
            out.append(c)
            continue
        items = [{k: v for k, v in c.items() if k != "foreach"}]
        for var, vals in fe.items():
            items = [_fill(it, var, v) for it in items for v in vals]
        out.extend(items)
    return out


def _subst(obj, vals: dict):
    """vars 替换：字符串恰好是「{变量}」时换成取值本身（保留数字类型），否则在字符串里按文本替换。"""
    if isinstance(obj, str):
        for k, v in vals.items():
            if obj == "{" + k + "}":
                return v
        for k, v in vals.items():
            obj = obj.replace("{" + k + "}", str(v))
        return obj
    if isinstance(obj, list):
        return [_subst(x, vals) for x in obj]
    if isinstance(obj, dict):
        return {k: _subst(v, vals) for k, v in obj.items()}
    return obj


def _read_yaml(p: Path):
    try:
        return yaml.safe_load(p.read_text())
    except yaml.YAMLError as e:
        raise CatalogError(f"{p}：YAML 解析失败：{e}") from e


def load(path) -> Catalog:
    """include: [片段文件…]（相对本文件）：片段里的 checks 排在本文件的 checks 前面；vars: {变量: 值}：合并后的 checks 里
    「{变量}」换成取值（几档共用同一批检查、只差步数与窗口时用，共用的部分只写一处）。片段的路径与指纹（见 fingerprint）记在 data["_includes"]。"""
    p = Path(path)
    data = _read_yaml(p)
    if isinstance(data, dict):
        checks, includes = [], []
        for inc in data.get("include") or []:
            ip = (p.parent / inc).resolve()
            if not ip.is_file():
                raise CatalogError(f"include 的文件不存在：{inc}")
            frag = _read_yaml(ip) or {}
            checks += list(frag.get("checks") or [])
            includes.append(fingerprint(ip))
        checks += list(data.get("checks") or [])
        if data.get("vars"):
            checks = _subst(checks, dict(data["vars"]))
        data["checks"] = expand(checks)
        for c in data["checks"]:                   # hard 的默认值在这里定一次，引擎与报告读到的都是显式的值
            if isinstance(c, dict) and "hard" not in c:
                c["hard"] = c.get("type") not in SOFT_BY_DEFAULT
        data["_includes"] = includes
    validate(data)
    return Catalog(p, data)


def validate(d: dict) -> None:
    """载入时的六条校验规则（docs/reference/catalog-schema.md「校验规则」第 1–6 条）；第 7–9 条在执行每条检查时查。"""
    errs = []
    if not isinstance(d, dict):
        raise CatalogError("顶层必须是映射")
    for k in ("schema_version", "model", "stage", "runs", "checks"):
        if k not in d:
            errs.append(f"缺顶层字段 {k}")
    if d.get("schema_version") != 1:
        errs.append(f"schema_version 必须是 1，现在是 {d.get('schema_version')}")
    runs = d.get("runs") or {}
    if "test" not in runs or "root" not in runs:
        errs.append("runs 必须有 test 与 root")
    noise = runs.get("noise") or []
    for pair in noise:
        if not (isinstance(pair, list) and len(pair) == 2):
            errs.append(f"runs.noise 的每一项必须是两个运行名：{pair}")
    seen = set()
    for i, c in enumerate(d.get("checks") or []):
        where = f"checks[{i}]（{c.get('id', '?')}）"
        for k in ("id", "type", "level", "select", "threshold"):
            if k not in c and not (k == "threshold" and c.get("type") in NO_THRESHOLD):
                errs.append(f"{where} 缺字段 {k}")
        if c.get("id") in seen:
            errs.append(f"{where} id 重复")
        seen.add(c.get("id"))
        if c.get("type") not in REGISTRY:
            errs.append(f"{where} 未知类型 {c.get('type')}")
        if c.get("level") not in LEVELS:
            errs.append(f"{where} level 不合法 {c.get('level')}")
        sel = c.get("select") or {}
        if sel.get("kind") not in SELECT_KINDS:
            errs.append(f"{where} select.kind 不合法 {sel.get('kind')}")
        th = c.get("threshold")
        if isinstance(th, dict) and not c.get("source"):
            errs.append(f"{where} 数值门槛必须写 source")
        if th == "pending" and c.get("type") not in PENDING_OK:
            errs.append(f"{where} threshold: pending（待标定）只能用在 {sorted(PENDING_OK)}")
        if th == "calibrated" and not noise and c.get("type") != "multi_golden":
            errs.append(f"{where} calibrated 门槛需要 runs.noise")
        if c.get("compare") == "lkg" and not runs.get("lkg"):
            errs.append(f"{where} compare: lkg 需要 runs.lkg")
    if errs:
        raise CatalogError("；".join(errs))
