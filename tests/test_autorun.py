"""tools/autorun/autorun.py：分支头变了才跑、同一提交不重跑、出错有限次重试、fail / inconclusive 才评论、
dry-run 不评论不改状态、开关与暂停文件；混元 3 报告摘出评论正文。用本地 git 仓库当远端，不碰网络。"""

import json
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "autorun"))

import autorun as ar  # noqa: E402


def git(*a, cwd):
    subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True)


def make_remote(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    git("init", "-q", "-b", "main", cwd=remote)
    git("config", "user.email", "t@t", cwd=remote)
    git("config", "user.name", "t", cwd=remote)
    (remote / "a").write_text("1")
    git("add", "a", cwd=remote)
    git("commit", "-q", "-m", "1", cwd=remote)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(remote), str(clone)], check=True)
    return remote, clone


def commit(remote, n):
    (remote / "a").write_text(str(n))
    git("commit", "-q", "-am", str(n), cwd=remote)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=remote, capture_output=True, text=True).stdout.strip()


def setup(tmp_path, monkeypatch, verdicts):
    remote, clone = make_remote(tmp_path)
    calls, comments = [], []

    def fake(job, head, out, logf):
        calls.append(head)
        return verdicts.pop(0), f"差异 {head[:6]}"

    monkeypatch.setitem(ar.KINDS, "hy3-t1", fake)
    monkeypatch.setattr(
        ar, "post_comment", lambda repo, sha, body: comments.append((sha, body)) or "https://example/c/1"
    )
    cfg = {
        "enabled": True,
        "pause_file": str(tmp_path / "PAUSE"),
        "state_dir": str(tmp_path / "st"),
        "log_dir": str(tmp_path / "log"),
        "jobs": [
            {
                "name": "j",
                "enabled": True,
                "kind": "hy3-t1",
                "repo": "o/r",
                "branch": "main",
                "clone": str(clone),
                "comment": True,
                "max_attempts": 2,
            }
        ],
    }
    cp = tmp_path / "cfg.yaml"
    cp.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    return remote, cp, calls, comments


def test_runs_once_per_head_and_comments_on_fail(tmp_path, monkeypatch):
    remote, cp, calls, comments = setup(tmp_path, monkeypatch, ["pass", "fail"])
    assert ar.main(["--config", str(cp)]) == 0 and len(calls) == 1 and comments == []
    ar.main(["--config", str(cp)])
    assert len(calls) == 1  # 分支头没变，不重跑
    h2 = commit(remote, 2)
    ar.main(["--config", str(cp)])
    assert calls[-1] == h2 and comments[0][0] == h2 and "差异" in comments[0][1]
    st = json.loads((tmp_path / "st" / "j.json").read_text())
    assert st["last_done"] == h2 and st["last_verdict"] == "fail"
    hist = [json.loads(x) for x in (tmp_path / "log" / "history.jsonl").read_text().splitlines()]
    assert [h["verdict"] for h in hist] == ["pass", "fail"] and hist[-1]["comment"] == "https://example/c/1"


def test_error_retries_then_stops(tmp_path, monkeypatch):
    remote, cp, calls, comments = setup(tmp_path, monkeypatch, ["error", "error", "pass"])
    for _ in range(4):
        ar.main(["--config", str(cp)])
    assert len(calls) == 2 and comments == []  # max_attempts = 2，出错不评论


def test_switches_and_dry_run(tmp_path, monkeypatch):
    remote, cp, calls, comments = setup(tmp_path, monkeypatch, ["fail", "fail"])
    (tmp_path / "PAUSE").write_text("")
    ar.main(["--config", str(cp)])
    assert calls == []  # 暂停文件在：什么都不做
    ar.main(["--config", str(cp), "--job", "j", "--dry-run"])
    assert len(calls) == 1 and comments == [] and not (tmp_path / "st" / "j.json").exists()
    (tmp_path / "PAUSE").unlink()
    cfg = yaml.safe_load(cp.read_text())
    cfg["jobs"][0]["enabled"] = False
    cp.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    ar.main(["--config", str(cp)])
    assert len(calls) == 1  # 作业关着


def test_hy3_comment_body_keeps_header_and_bad_rows():
    rep = "\n".join(
        [
            "```",
            "[T1 · v5p] 数值 FAIL",
            "第一个不同：R08",
            "```",
            "",
            "- 档位配置 …",
            "",
            "| 行 | 状态 | 检查 | 关键数 | 说明 |",
            "|---|---|---|---|---|",
            "| R01 | pass | `R01.run_complete`（pass） | x | y |",
            "| R12 | fail | `R12.loss`（fail） | x | y |",
            "|  |  | `R12.loss_tolerance`（inconclusive，只报） | **1.03 倍尺子** | z |",
            "| R13 | pass | `R13.raw_grad_norm`（pass） | x | y |",
        ]
    )
    body = ar.hy3_comment_body(rep)
    assert body.startswith("```\n[T1 · v5p] 数值 FAIL") and "R12.loss`" in body and "1.03 倍尺子" in body
    assert "R01" not in body and "R13" not in body


def test_env_values_keep_urls():
    assert ar.expand_str("gs://<POC 桶>/x") == "gs://<POC 桶>/x"
    assert ar.expand_str("~/a").startswith("/")
