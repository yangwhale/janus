"""对训练分支头跑数值门：确定性脚本（不调用任何模型或外部服务，只用 git、gh、kubectl）。

手动触发为主（`--job <名>`）；也可由站点的定时任务调起做轮询（可选，默认不开：站点配置顶层 enabled: false）。每次调起做一遍：

1. 读站点配置（`--config`，YAML，放在仓库外；字段见 `site.example.yaml`）。顶层 `enabled: false` 或存在 `pause_file` 就什么都不做。
2. 对每个 `enabled: true` 的作业：`git fetch` 被守的分支，分支头和状态文件里上次跑过的提交不同就跑一次：
   - `kind: hy3-t1`：混元 3 日常回归 T1（`models/hunyuan3/frameworks/maxtext/stage.sh` + `models/hunyuan3/regression/run_regression.sh`）。
3. 结果写进 `<log_dir>/<作业>/<日期>-<提交>/`（`report.md`、`result.json`、`run.log`），一行摘要追加到 `<log_dir>/history.jsonl`。
4. 判定是 fail 或 inconclusive、且作业 `comment: true` 时，把报告写成评论：分支头提交属于某个已合并的 PR 就评在 PR 下，
   否则评在提交下。pass 与工具出错不评论（工具出错只进日志，下一轮重试，同一提交最多 `max_attempts` 次）。

同一时间只跑一个实例（`<state_dir>/autorun.lock`）。一次只处理分支头：两次轮询之间合进来的几个提交并成一次，报告里列出范围。

    python tools/autorun/autorun.py --config <站点配置>            # 定时调起
    python tools/autorun/autorun.py --config <站点配置> --job <名> --dry-run   # 手动试跑一个作业：照跑，但不评论、不改状态
    python tools/autorun/autorun.py --config <站点配置> --status   # 看各作业上次跑到哪、结论是什么
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

JANUS = Path(__file__).resolve().parents[2]
EXIT_VERDICT = {0: "pass", 1: "fail", 2: "error", 3: "inconclusive"}
VERDICT_ZH = {"pass": "过", "fail": "不过", "inconclusive": "待判", "error": "工具出错"}


def expand(p) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p))))


def expand_str(v) -> str:
    """环境变量的值：只展开 ~ 与 $VAR，不当路径规整（gs://… 不能被压成 gs:/…）。"""
    return os.path.expandvars(os.path.expanduser(str(v)))


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def sh(cmd, check=True, **kw) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, text=True, capture_output=True, **kw)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} 失败（{r.returncode}）：{r.stderr[-1500:]}")
    return r


# ------------------------------------------------------------------------------------------------ 状态
def load_state(state_dir: Path, job: str) -> dict:
    p = state_dir / f"{job}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(state_dir: Path, job: str, st: dict) -> None:
    p = state_dir / f"{job}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(p)


# ------------------------------------------------------------------------------------------------ git / gh
def branch_head(clone: Path, branch: str) -> str:
    sh(["git", "-C", str(clone), "fetch", "-q", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}"])
    return sh(["git", "-C", str(clone), "rev-parse", f"origin/{branch}"]).stdout.strip()


def commit_range(clone: Path, old: str | None, new: str) -> list[str]:
    if not old:
        return [new]
    r = sh(["git", "-C", str(clone), "rev-list", "--first-parent", f"{old}..{new}"], check=False)
    return r.stdout.split() if r.returncode == 0 and r.stdout.strip() else [new]


def merged_pr(repo: str, sha: str) -> int | None:
    r = sh(
        [
            "gh",
            "api",
            f"repos/{repo}/commits/{sha}/pulls",
            "--jq",
            "[.[] | select(.merged_at != null) | .number] | first",
        ],
        check=False,
    )
    s = r.stdout.strip()
    return int(s) if r.returncode == 0 and s.isdigit() else None


def post_comment(repo: str, sha: str, body: str) -> str:
    bf = Path(os.environ.get("TMPDIR", "/tmp")) / f"janus-autorun-comment-{os.getpid()}.md"
    bf.write_text(body)
    try:
        pr = merged_pr(repo, sha)
        if pr:
            r = sh(["gh", "api", f"repos/{repo}/issues/{pr}/comments", "-F", f"body=@{bf}", "--jq", ".html_url"])
        else:
            r = sh(["gh", "api", f"repos/{repo}/commits/{sha}/comments", "-F", f"body=@{bf}", "--jq", ".html_url"])
        return r.stdout.strip()
    finally:
        bf.unlink(missing_ok=True)


# ------------------------------------------------------------------------------------------------ 作业
HY3_ROW = re.compile(r"^\| (R\d+) \| (fail|inconclusive|error) \|")


def hy3_comment_body(report_md: str) -> str:
    """混元 3 回归报告：首屏代码块 + 不是 pass 的行（表头一并带上）。"""
    lines = report_md.splitlines()
    head, i = [], 0
    if lines and lines[0].startswith("```"):
        i = lines.index("```", 1) + 1 if "```" in lines[1:] else len(lines)
        head = lines[:i]
    rows, keep = [], False
    for ln in lines[i:]:
        if ln.startswith("| 行 |") or ln.startswith("|---"):
            rows.append(ln)
            continue
        m = HY3_ROW.match(ln)
        if m:
            keep = True
            rows.append(ln)
        elif keep and ln.startswith("|  |"):
            rows.append(ln)
        else:
            keep = False
    body = head + ([""] + rows if len(rows) > 2 else [])
    return "\n".join(body) + "\n"


def run_hy3_t1(job: dict, head: str, out: Path, logf) -> tuple[str, str]:
    env = {**os.environ, **{k: expand_str(v) for k, v in (job.get("env") or {}).items()}}
    name = f"{job.get('run_prefix', 'auto-t1')}-{head[:8]}"
    stage = sh(
        ["bash", str(JANUS / "models/hunyuan3/frameworks/maxtext/stage.sh"), str(expand(job["clone"])), head],
        env=env,
        check=False,
    )
    logf.write(stage.stderr)
    if stage.returncode != 0:
        return "error", f"打包失败：{stage.stderr[-800:]}"
    for line in stage.stdout.splitlines():  # stage.sh 打印 export K=V …
        for kv in line.replace("export ", "").split():
            k, _, v = kv.partition("=")
            env[k] = v
    args = ["bash", str(JANUS / "models/hunyuan3/regression/run_regression.sh"), "T1", name]
    rc = subprocess.run(args, stdout=logf, stderr=subprocess.STDOUT, env=env).returncode
    rep = expand(env.get("LOCAL_RUNS", "~/janus-runs")) / "regression-results" / name / "report.md"
    report = rep.read_text() if rep.exists() else ""
    (out / "report.md").write_text(report)
    verdict = EXIT_VERDICT.get(rc, "error")
    return verdict, hy3_comment_body(report) if report else ""


KINDS = {"hy3-t1": run_hy3_t1}
TITLE = {"hy3-t1": "混元 3 · 日常回归 T1"}


def process(cfg: dict, job: dict, dry: bool) -> None:
    name, state_dir, log_dir = job["name"], expand(cfg["state_dir"]), expand(cfg["log_dir"])
    clone = expand(job["clone"])
    head = branch_head(clone, job["branch"])
    st = load_state(state_dir, name)
    if not dry and st.get("last_done") == head:
        return
    attempts = st.get("attempts", {}).get(head, 0)
    if not dry and attempts >= int(job.get("max_attempts", 2)):
        return
    commits = commit_range(clone, st.get("last_done"), head)
    out = log_dir / name / f"{time.strftime('%Y%m%d-%H%M')}-{head[:10]}"
    out.mkdir(parents=True, exist_ok=True)
    log(f"{name}：分支头 {head[:10]}（上次 {str(st.get('last_done'))[:10]}，{len(commits)} 个提交）→ {out}")
    t0 = time.time()
    with open(out / "run.log", "w") as logf:
        verdict, body = KINDS[job["kind"]](job, head, out, logf)
    secs = round(time.time() - t0)
    rec = {
        "job": name,
        "head": head,
        "commits": commits,
        "verdict": verdict,
        "seconds": secs,
        "out": str(out),
        "at": now(),
        "dry_run": dry,
        "comment": None,
    }
    if verdict in ("fail", "inconclusive") and job.get("comment") and not dry and body:
        extra = (
            f"\n本次覆盖 {len(commits)} 个合并提交（`{commits[-1][:10]}` … `{head[:10]}`）。\n"
            if len(commits) > 1
            else ""
        )
        try:
            rec["comment"] = post_comment(
                job["repo"], head, f"**{TITLE[job['kind']]}：{VERDICT_ZH[verdict]}**\n\n{body}{extra}"
            )
        except RuntimeError as e:
            rec["comment_error"] = str(e)
    (out / "result.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
    with open(log_dir / "history.jsonl", "a") as h:
        h.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"{name}：{verdict}（{secs} s）{rec.get('comment') or ''}")
    if dry:
        return
    if verdict == "error":
        st.setdefault("attempts", {})[head] = attempts + 1
    else:
        st.update(last_done=head, last_verdict=verdict, last_out=str(out), last_at=rec["at"], attempts={})
    save_state(state_dir, name, st)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--job", help="只处理这一个作业（不看它的 enabled）")
    ap.add_argument("--dry-run", action="store_true", help="照跑，但不评论、不改状态")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(expand(a.config).read_text())
    state_dir, log_dir = expand(cfg["state_dir"]), expand(cfg["log_dir"])
    state_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if a.status:
        for job in cfg.get("jobs", []):
            st = load_state(state_dir, job["name"])
            print(
                f"{job['name']:12s} enabled={job.get('enabled', False)!s:5s} 上次 {str(st.get('last_done'))[:10]} "
                f"{st.get('last_verdict', '-')} {st.get('last_at', '')} {st.get('last_out', '')}"
            )
        return 0
    if not a.job and (not cfg.get("enabled", False) or (cfg.get("pause_file") and expand(cfg["pause_file"]).exists())):
        return 0
    lock = open(state_dir / "autorun.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("上一轮还在跑，这轮跳过")
        return 0
    rc = 0
    for job in cfg.get("jobs", []):
        if a.job and job["name"] != a.job:
            continue
        if not a.job and not job.get("enabled", False):
            continue
        try:
            process(cfg, job, a.dry_run)
        except Exception as e:  # noqa: BLE001  一个作业出错不影响别的作业，进日志
            log(f"{job['name']}：轮询器出错 {type(e).__name__}: {e}")
            rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
