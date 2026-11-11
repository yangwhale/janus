"""混元 3 · TPU 自噪声标定（标准第五节）：扰动臂过三关，再从「基线 + 过关的臂」两两成对算噪声参照。

  python calibrate.py --root <运行根> --out calibration-v5p.json \\
      --t1-base <T1 基线> --t1-base-pair <T1 基线的双跑另一份> --t0-base <T0 基线（LKG）> \\
      --arm <名>=<T0 运行 a>,<T0 运行 b>,<T1 运行>[,<允许变的配置键>…] … --golden g1k-a,g1k-b,g1k-c
  只跑了 T1 的臂写 --arm <名>=-,-,<T1 运行>（第一、二关记「没跑」，不进参照），--arm-note <名>=<说明> 记原因。

三关：
1. 臂自己确定：臂的 T0 双跑（a / b）逐位——用 T0 目录跑 janus，全部硬检查 pass。
2. 只改了执行：臂的 T0 a 对 T0 基线，R03、R04、R06、R07、R14、R18、R20 全过；R02 只有命令行覆盖不同、R05 只有臂声明的那几个键
   （LIBTPU_INIT_ARGS 或 MaxText 配置键）不同；编译器 flagfile 只差臂自己的开关行。
3. 差在舍入量级：臂的 T1 对 T1 基线，窗口 1:1001 的 loss 三个量（带符号均值取绝对值、MARE、最大单步）都不大于
   GPU 自己重跑的噪声上限（同窗口同分母）。上限取**至少 10 次**默认设置 GPU 运行两两组合（≥ 45 对）的最大值，
   `--golden` 少于 10 份时结论标为「暂定」：3 份 golden 只有 3 对，带符号均值的上限会被低估（实测：3 份 golden 两两 3 对是 3.14e-5，10 份运行两两 45 对是 5.39e-5，results/calibration-v5p.json）。
   第 1 步参数梯度那一条要抓张量，这一版没做（README 写明）。

噪声参照：过关的臂与基线两两成对（同一配置的双跑算不上一对，剔掉），T1 上逐序列（lm_loss、raw_grad_norm、param_norm、
每层 router.bias_*）算三个量，T0（20 步）同样算一份给 T0 的容差门用。另报留一法互判（noise_loo）。

写 calibration.yaml（--yaml）：每档一条，绑定平台、档位配置 sha256、训练配置（hy3-4layers.yml）sha256、镜像与依赖版本（标准第五节），带参照运行的位置
（--locations：运行名 → 占位路径的 JSON，不写桶名）、参照对、每个序列三个量的上限、留一法互判；对数 < 15 时
status 写「暂定」（标准第五节：≥ 3 对能判，≥ 15 对才是正式启用的下限）。--step-time-runs（T0）、--t1-step-time-runs（T1）给同一程序的 ≥ 5 次运行时，
另写 R22 步时阈值：各次第 3 步起步时中位数的极差 ÷ 中位数（标准 R22）。

  python calibrate.py … --yaml calibration.yaml --locations locs.json --date 2026-11-05 \\
      --step-time-runs reg-t0-lkg1-a,reg-t0-lkg1-b,reg-t0-r2,reg-t0-st1,reg-t0-st2
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from janus.checks.series import CALIBRATED_KEYS, _stats, noise_from_pairs  # noqa: E402
from janus.store import Run  # noqa: E402

T1_WINDOW, T0_WINDOW = (1, 1001), (1, 21)
MIN_GPU_RUNS = 10       # 第三关的 GPU 噪声上限至少要 10 次运行（45 对）
MIN_PAIRS, FORMAL_PAIRS = 3, 15     # 标准第五节：< 3 对最多 inconclusive；≥ 15 对是正式启用 calibrated 的下限
MIN_STEP_TIME_RUNS = 5              # 标准 R22：同一提交 ≥ 5 次运行的步时极差
STEP_TIME_SKIP = 2                  # 与 regress.py 相同：去掉前两步
FIELDS = ["lm_loss", "raw_grad_norm", "param_norm"]


def janus_t0(root: Path, test: str, lkg: str) -> dict:
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, "-m", "janus", "run", str(HERE / "catalogs" / "regression-t0.yaml"), "--root", str(root),
                        "--test", test, "--lkg", lkg, "--out", d], cwd=REPO, capture_output=True, text=True)
        res = json.loads((Path(d) / "results.json").read_text())
        details = {}
        for c in res["checks"]:
            f = Path(d) / f"details/{c['id']}.jsonl"
            if f.exists():
                details[c["id"]] = [json.loads(x) for x in f.read_text().splitlines()]
    return {"checks": {c["id"]: c for c in res["checks"]}, "details": details}


def gate1(root, a, b) -> dict:
    r = janus_t0(root, b, a)
    bad = [k for k, c in r["checks"].items() if c.get("hard", True) and c["status"] != "pass"]
    return {"pass": not bad, "not_pass": bad}


def gate2(root, arm_a, base, allowed: set) -> dict:
    r = janus_t0(root, arm_a, base)
    ch, det = r["checks"], r["details"]
    problems = []
    for k in ("R03.data_fingerprint", "R04.versions", "R06.must_equal", "R07.state_init", "R14.lr_wd", "R18.dtypes",
              "R20.artifact_counts", "R01.run_complete"):
        if ch[k]["status"] != "pass":
            problems.append(f"{k} {ch[k]['status']}：{ch[k].get('message', '')[:120]}")
    for row in det.get("R02.inputs", []):
        if not row.get("equal", True) and "config.overrides" not in row["name"]:
            problems.append(f"R02 {row['name']}")
    for row in det.get("R05.process_flags", []):
        for d in row.get("differ", []):
            if d["key"] not in allowed:
                problems.append(f"R05 {row['name']}：{d['key']}")
        if not row.get("equal", True) and "differ" not in row:
            problems.append(f"R05 {row['name']}")
    for row in det.get("R05.compiler_flagfile", []):
        line = row.get("line", "")
        if line and not any(k in line for k in allowed):
            problems.append(f"flagfile：{line[:100]}")
    return {"pass": not problems, "problems": problems,
            "r20": {d["name"]: d["count"] for d in det.get("R20.artifact_counts", [])}}


def series(run: Run, prefix_fields=True) -> dict:
    m = run.metrics()
    names = list(FIELDS) + sorted(k for k in m[min(m)] if k.startswith("router.bias_"))
    return {n: run.series(n) for n in names}


def pair_stats(sa: dict, sb: dict, window) -> dict:
    out = {}
    for n in sa:
        vals, raw = _stats(sa[n], sb[n], window, "golden", 1)
        out[n] = {k: vals[k] for k in CALIBRATED_KEYS} | {"max_step_at": raw["max_step_at"]}
    return out


def worst(stats: dict, names) -> dict:
    return {k: max(stats[n][k] for n in names) for k in CALIBRATED_KEYS}


def loo(noise: dict) -> list:
    """留一法互判（标准第五节「查参照齐不齐」）：每一对用其余各对的最大值当上限算 r。少于 3 对时只报。"""
    vals = {k: noise[k]["values"] for k in CALIBRATED_KEYS}
    n = len(vals["mare"])
    rows = []
    for i in range(n):
        row = {"pair": i}
        for k in CALIBRATED_KEYS:
            others = [v for j, v in enumerate(vals[k]) if j != i]
            row[k] = (vals[k][i] / max(others)) if others and max(others) > 0 else None
        rows.append(row)
    return rows


def step_time(root: Path, runs: list) -> dict:
    """R22 阈值：同一程序各次运行第 3 步起步时中位数的极差 ÷ 这几次中位数的中位数。"""
    med = {}
    for r in runs:
        st = Run.open(root / r).series("step_time")
        med[r] = statistics.median([v for s, v in sorted(st.items()) if s > STEP_TIME_SKIP])
    vals = list(med.values())
    mid = statistics.median(vals)
    return {"runs": med, "median_s": mid, "range_s": max(vals) - min(vals),
            "threshold": (max(vals) - min(vals)) / mid if len(vals) >= MIN_STEP_TIME_RUNS else None,
            "enough_runs": len(vals) >= MIN_STEP_TIME_RUNS}


def _binding(root: Path, base: str, tier: str) -> dict:
    m = Run.open(root / base).meta
    cfg = HERE / "tiers" / f"{tier.lower()}.yaml"
    # 训练配置 hy3-4layers.yml 的 sha256 取基线运行 meta 里记的（就是那次训练实际用的文件），regress.py 拿它对被测的同一字段
    return {"platform": "v5p", "tier_config_sha256": hashlib.sha256(cfg.read_bytes()).hexdigest(),
            "config_yml_sha256": (m.get("config") or {}).get("sha256"),
            "image": m.get("image"), "versions": m.get("versions")}


def arm_verdict(arm: dict) -> str:
    """一句话：进没进参照、卡在哪一关。第三关是准入，只有不大于 GPU 上限（1 倍）才进；1–2 倍记「未证实」，不是判它改了数学。"""
    if arm["admitted"]:
        return "进参照"
    if arm["bitwise_to_base"]:
        return "不进参照：与基线逐位（算不上一对）"
    if arm.get("bitwise_to"):
        return f"不进参照：T1 与已进参照的 {arm['bitwise_to']} 逐位（同一条轨迹，算不上新的一份）"
    if not arm["gate3"]["pass"]:
        r = arm["gate3"]["ratio_to_gpu_upper"]
        worst = max(arm["gate3"]["over_gpu_upper"], key=lambda k: r[k])
        tag = "未证实只改执行" if r[worst] <= 2 else "超出两倍"
        return f"不进参照：第三关 {worst} 是 GPU 上限的 {r[worst]:.2f} 倍（{tag}）"
    if arm["gate1"].get("not_run"):
        return f"不进参照：{arm.get('note') or '第一、二关没跑'}"
    if not arm["gate1"]["pass"]:
        return "不进参照：第一关（臂自己重跑不逐位）"
    return "不进参照：第二关（不只改了执行）"


def write_yaml(path: Path, out: dict, root: Path, a) -> None:
    import yaml

    locs = json.loads(Path(a.locations).read_text()) if a.locations else {}
    reg = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    reg.setdefault("schema", 1)
    entries = reg.setdefault("entries", [])
    for tier, key in (("T1", "tier_t1"), ("T0", "tier_t0")):
        t = out[key]
        n = t["n_pairs"]
        e = {
            "key": {"platform": "v5p", "tier": tier, "variant": "main"},
            "date": a.date,
            "status": "正式" if n >= FORMAL_PAIRS else ("暂定" if n >= MIN_PAIRS else "参照不足"),
            "binding": _binding(root, t["base"], tier),
            "base": t["base"],
            "runs": {r: locs.get(r, "") for r in t["runs"]},
            "pairs": t["pairs"], "n_pairs": n, "window": t["window"],
            "noise_upper": {s: {k: v["upper"] for k, v in d.items()} for s, d in t["noise"].items()},
            "noise_loo_lm_loss": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
                                  for r in t["noise_loo"]],
            "step_time_threshold": (t.get("step_time") or {}).get("threshold"),
        }
        if t.get("step_time"):
            e["step_time_runs"] = {r: round(v, 5) for r, v in t["step_time"]["runs"].items()}
        e["arms"] = {nm: arm_verdict(arm) for nm, arm in out["arms"].items()}
        entries.append(e)
    path.write_text(_YAML_HEAD + yaml.safe_dump(reg, allow_unicode=True, sort_keys=False, width=140))


_YAML_HEAD = """# 混元 3 · 日常回归 · TPU 执行噪声标定（标准第五节）。calibrate.py --yaml 追加条目，regress.py / run_regression.sh 读最新一条。
# 一条 = 平台 × 档位 × 并行变体。binding 里任何一样和被测不同（档位配置 sha256、训练配置 hy3-4layers.yml 的 sha256、镜像、依赖版本），这份参照作废，
# 用它的容差门降为 inconclusive，直到重标。runs 是参照运行的位置（占位写法，不写桶名），run_regression.sh 按它取回。
# status：参照对 < 3「参照不足」（最多 inconclusive）；3–14「暂定」（能判，纯噪声约每 n + 1 次有 1 次超上限）；≥ 15「正式」。
# 目录 catalogs/regression-t{0,1}.yaml 的 runs.noise 必须和这里最新一条的 pairs 相同（tests 核）。
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--t1-base", required=True)
    p.add_argument("--t1-base-pair", default=None)
    p.add_argument("--t0-base", required=True)
    p.add_argument("--arm", action="append", default=[])
    p.add_argument("--golden", required=True, help="GPU 默认设置运行，逗号分隔，至少 10 份（两两 ≥ 45 对）")
    p.add_argument("--arm-note", action="append", default=[], help="<臂名>=<说明>：写进输出（例如只跑了 T1 的臂为什么不再跑 T0）")
    p.add_argument("--yaml", default=None, help="写（追加）calibration.yaml 的条目，T0、T1 各一条")
    p.add_argument("--locations", default=None, help="JSON：参照运行名 → 运行目录的占位路径（gs://<POC 桶>/…）")
    p.add_argument("--date", default=None, help="标定日期（写进 calibration.yaml）")
    p.add_argument("--step-time-runs", default=None, help="同一程序的 T0 运行（≥ 5 份，逗号分隔）：算 R22 步时阈值")
    p.add_argument("--t1-step-time-runs", default=None, help="同一程序的 T1 运行（≥ 5 份）：T1 的 R22 步时阈值")
    a = p.parse_args(argv)
    root = Path(a.root)
    gold = [Run.open(root / g) for g in a.golden.split(",")]
    gnoise = noise_from_pairs([(x.series("lm_loss"), y.series("lm_loss")) for x, y in itertools.combinations(gold, 2)],
                              T1_WINDOW, "golden", 1, names=[f"{x.name}~{y.name}" for x, y in itertools.combinations(gold, 2)])
    gpu_upper = {k: max(gnoise[k]) for k in CALIBRATED_KEYS}
    provisional = len(gold) < MIN_GPU_RUNS
    base1 = Run.open(root / a.t1_base)
    out = {"platform": "v5p-8", "tier_t1": {"base": a.t1_base}, "tier_t0": {"base": a.t0_base},
           "gpu_noise_upper_lm_loss_1_1001": gpu_upper, "gpu_runs": [g.name for g in gold],
           "gpu_pairs": len(gnoise["mare"]), "gate3_provisional": provisional, "gpu_noise_loo": gnoise.get("loo"), "arms": {}}
    if a.t1_base_pair:
        st = pair_stats(series(base1), series(Run.open(root / a.t1_base_pair)), T1_WINDOW)
        out["tier_t1"]["base_pair_bitwise"] = all(v["mare"] == 0 and v["max_step"] == 0 for v in st.values())
    passed_t1, passed_t0 = [a.t1_base], [a.t0_base]
    notes = dict(x.split("=", 1) for x in a.arm_note)
    for spec in a.arm:
        name, rest = spec.split("=", 1)
        parts = rest.split(",")
        t0a, t0b, t1 = parts[:3]
        allowed = set(parts[3:]) | {"LIBTPU_INIT_ARGS"}
        if t0a == "-":          # 只跑了 T1（第三关或 T1 上的 R20 已经不过，不再跑 T0 双跑）：第一、二关记「没跑」
            g1 = g2 = {"pass": False, "not_run": True}
        else:
            g1 = gate1(root, t0a, t0b)
            g2 = gate2(root, t0a, a.t0_base, allowed)
        st = pair_stats(series(Run.open(root / t1)), series(base1), T1_WINDOW)
        loss = st["lm_loss"]
        g3_bad = {k: loss[k] for k in CALIBRATED_KEYS if loss[k] > gpu_upper[k]}
        bitwise = all(v["mare"] == 0 and v["max_step"] == 0 for v in st.values())
        # 和已经进参照的某个臂逐位（例如编译器把更大的片上内存上限截到同一个值）：同一条轨迹，算不上新的一份
        dup = next((r for r in passed_t1[1:] if all(v["mare"] == 0 and v["max_step"] == 0 for v in
                                                   pair_stats(series(Run.open(root / t1)), series(Run.open(root / r)), T1_WINDOW).values())), None)
        ok = g1["pass"] and g2["pass"] and not g3_bad and not bitwise and dup is None
        out["arms"][name] = {"t0": [t0a, t0b], "t1": t1, "gate1": g1, "gate2": g2, "note": notes.get(name, ""),
                             "gate3": {"pass": not g3_bad, "lm_loss_vs_base": loss, "over_gpu_upper": g3_bad,
                                       "ratio_to_gpu_upper": {k: loss[k] / gpu_upper[k] for k in CALIBRATED_KEYS}},
                             "bitwise_to_base": bitwise, "bitwise_to": dup, "admitted": ok}
        if ok:
            passed_t1.append(t1)
            passed_t0.append(t0a)
    for tier, runs, window in (("tier_t1", passed_t1, T1_WINDOW), ("tier_t0", passed_t0, T0_WINDOW)):
        rs = {r: series(Run.open(root / r)) for r in runs}
        pairs = list(itertools.combinations(runs, 2))
        per = {f"{x}~{y}": pair_stats(rs[x], rs[y], window) for x, y in pairs}
        names = list(next(iter(rs.values())))
        noise = {}
        for n in names + ["router.bias_*"]:
            sel = [m for m in names if m.startswith("router.bias_")] if n == "router.bias_*" else [n]
            vals = {pn: worst(ps, sel) for pn, ps in per.items()}
            noise[n] = {k: {"upper": max((v[k] for v in vals.values()), default=None), "values": [v[k] for v in vals.values()]}
                        for k in CALIBRATED_KEYS}
        out[tier].update({"runs": runs, "pairs": [list(x) for x in pairs], "n_pairs": len(pairs),
                          "window": f"{window[0]}:{window[1]}", "noise": noise})
    for tier in ("tier_t1", "tier_t0"):
        out[tier]["noise_loo"] = loo(out[tier]["noise"]["lm_loss"])
    if a.step_time_runs:
        out["tier_t0"]["step_time"] = step_time(root, a.step_time_runs.split(","))
    if a.t1_step_time_runs:
        out["tier_t1"]["step_time"] = step_time(root, a.t1_step_time_runs.split(","))
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if a.yaml:
        write_yaml(Path(a.yaml), out, root, a)
    for name, arm in out["arms"].items():
        g12 = "没跑" if arm["gate1"].get("not_run") else f"{arm['gate1']['pass']} gate2 {arm['gate2']['pass']}"
        print(f"{name}: gate1 {g12} gate3 {arm['gate3']['pass']} "
              f"bitwise {arm['bitwise_to_base']} → {'进参照' if arm['admitted'] else '剔除'}")
    print("T1 pairs", out["tier_t1"]["n_pairs"], {k: v["upper"] for k, v in out["tier_t1"]["noise"]["lm_loss"].items()})
    print("GPU upper", gpu_upper, f"（{len(gold)} 份运行、{len(gnoise['mare'])} 对" + ("，不足 10 份：第三关暂定）" if provisional else "）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
