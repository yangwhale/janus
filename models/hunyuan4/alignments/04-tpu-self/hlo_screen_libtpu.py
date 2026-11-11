"""换 libtpu 版本时的离线筛选：用 hlo_screen.py 的编译与比较函数，把 arms.yaml 里登记了 screen 的臂（含 golden）在
旧、新两个 libtpu 下各编一遍（CPU 上为 TPU 拓扑编，0 张卡），回答两个问题：

  ① 每个臂对它的对照，「程序相同 / 不同」的判定在新版本下变没变；
  ② 同一份规格（golden 和各臂），新版本编出的程序和旧版本比变了什么（hlo_screen.hlo_summary 的摘要差、编译期显存）。

hlo_screen.py run 遇到换 libtpu 的臂会直接判「无法离线判断」、不编；这里是把每份规格的 image 与 libtpu 换成新版本，
两边都用同一个新版本编，所以判的是「新版本下改动前后程序是否相同」。注意：跨版本即使程序相同，也不能推出逐位相同，
后端代码生成一起换了；跨版本是否逐位只能上卡比。

新版本的镜像：在 golden 镜像上只换 libtpu（jax 不动），例如

  FROM <screen_base.image>
  RUN pip install --no-deps libtpu==0.0.52

  docker build -t janus-aot:libtpu-0.0.52 .
  python models/hunyuan4/alignments/04-tpu-self/hlo_screen_libtpu.py --image janus-aot:libtpu-0.0.52 \
      --libtpu 0.0.52 --maxtext-repo <maxtext 克隆> --cache <缓存目录> \
      [--extra extra.json] [--jobs 8] --out models/hunyuan4/alignments/04-tpu-self/results/hlo-screen-libtpu052.json

--extra：另外要编的规格，{名字: 对 screen_base 的增量}（写法同 arms.yaml 的 screen），例如 v7x 上的纯数据并行不累积。
每次编译单进程，峰值内存约 15 GiB；--jobs 个并行，先看 free -g。旧版本的编译与 hlo_screen.py run 共用缓存。
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hlo_screen as hs  # noqa: E402

EMB = re.compile(r"= bf16\[(\d+),(\d+)\]\S* scatter\(")


def emb_scatter(text: str, vocab: int) -> list[dict]:
    """词嵌入反向那次 BF16 scatter（输出形状 [词表, 列]）：每一处带不带 indices_are_sorted、在不在 SparseCore 上执行。"""
    lines = text.splitlines()
    out = []
    for i, ln in enumerate(lines):
        m = EMB.search(ln)
        if not m or int(m.group(1)) != vocab:
            continue
        j = i
        while j < len(lines) and not lines[j].startswith("}"):
            j += 1
        out.append(
            {
                "shape": [int(m.group(1)), int(m.group(2))],
                "sorted": "indices_are_sorted=true" in ln,
                "sparsecore": j < len(lines) and 'execution_thread="sparsecore"' in lines[j],
            }
        )
    return out


def analyse(arg):
    h, d, vocab = arg
    d = Path(d)
    f = d / "train_step.after_optimizations.txt"
    aot = json.loads((d / "aot.json").read_text()) if (d / "aot.json").exists() else {"error": "没编"}
    if not f.exists() or aot.get("error"):
        return h, {"error": (aot.get("error") or "没有优化后 HLO")[:300]}
    t = f.read_text()
    lines = hs.canon_lines(t)
    return h, {
        "fp": hs.program_fp(lines),
        "lines": len(lines),
        "summary": hs.hlo_summary(t),
        "emb_scatter": emb_scatter(t, vocab),
        "memory": hs.mem_info(aot),
        "compile_seconds": aot.get("compile_seconds"),
        "max_rss_gib": aot.get("max_rss_gib"),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, help="新版本的镜像（golden 镜像上只换 libtpu）")
    p.add_argument(
        "--libtpu", required=True, help="新版本，写法同 screen_base.libtpu，如「0.0.52」或「0.0.52 cl/<构建号>」"
    )
    p.add_argument("--maxtext-repo", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--extra", type=Path, default=None)
    p.add_argument("--arms", default=None, help="逗号分隔；缺省 = 写了 screen 的全部臂（不含 L 类）")
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--vocab", type=int, default=120832)
    p.add_argument("--mem", default="250g")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    arms = hs.Arms()
    old = {k: arms.base[k] for k in ("image", "libtpu")}

    def new(s):
        s = copy.deepcopy(s)
        s["image"], s["libtpu"] = a.image, a.libtpu
        return s

    ids = a.arms.split(",") if a.arms else [x["id"] for x in arms.arms if "screen" in x and not x["id"].startswith("L")]
    base = {k: v for k, v in arms.base.items() if k != "run"}
    items = {"golden": (base, None, None)}
    for aid in ids:
        x = arms.by_id[aid]
        vs = x.get("vs", hs.GOLDEN)
        t, b = arms.spec_of_run(x["run"]), arms.spec_of_run(vs)
        if t is not None and b is not None and (t.get("libtpu"), t.get("image")) != (b.get("libtpu"), b.get("image")):
            b = None  # 改的就是版本：两边都换成新版本后没有可比的对照，只记它自己新旧两版的程序
        items[aid] = (t, b, vs) if t is not None else None
    for name, delta in (json.loads(a.extra.read_text()) if a.extra else {}).items():
        items["extra:" + name] = (hs.apply_delta(base, delta), None, None)
    specs = {}
    for it in items.values():
        for s in (it or ())[:2]:
            if s is not None:
                for v in (s, new(s)):
                    specs[hs.spec_hash(v)] = v
    for c in {s["maxtext"] for s in specs.values()}:  # 先串行解出代码树，免得并行编译时抢着解同一个提交
        hs.maxtext_tree(a.maxtext_repo, c, a.cache)
    print(f"[hlo_screen_libtpu] {len(items)} 项，{len(specs)} 份规格（有缓存的不重编）", flush=True)
    dirs = {}

    def one(h):
        dirs[h] = hs.compile_spec(specs[h], a.cache, a.maxtext_repo, "docker", a.mem)["dir"]

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(one, sorted(specs)))
    with Pool(min(a.jobs * 2, 16)) as pool:
        info = dict(pool.map(analyse, [(h, dirs[h], a.vocab) for h in sorted(specs)]))

    def cmp(x, y):
        if x == y:
            return {"program_equal": True, "note": "规格相同，不用编"}
        p, q = info[x], info[y]
        if "error" in p or "error" in q:
            return {"program_equal": None, "error": [p.get("error"), q.get("error")]}
        r = {"program_equal": p["fp"] == q["fp"]}
        if not r["program_equal"]:
            r["summary_diff"] = hs.summary_diff(p["summary"], q["summary"])
        return r

    out = {
        "note": "换 libtpu 时的离线筛选（hlo_screen_libtpu.py，调 hlo_screen.py 的编译与比较）。vs_base：该臂对它的对照，"
        "在旧版本、新版本下各自的程序比较；new_vs_old：同一份规格新版本对旧版本的程序比较（summary_diff 里 test 是新版本）；"
        "emb_scatter：词嵌入反向那次 BF16 scatter 带不带 indices_are_sorted、在不在 SparseCore 上执行。"
        "跨版本程序相同也不能推出逐位相同",
        "old": old,
        "new": {"image": a.image, "libtpu": a.libtpu},
        "items": {},
    }
    for name, it in items.items():
        if it is None:
            out["items"][name] = None
            continue
        t, b, vs = it
        o, n = hs.spec_hash(t), hs.spec_hash(new(t))
        rec = {
            "run": arms.by_id[name]["run"] if name in arms.by_id else None,
            "vs": vs,
            "platform": t["platform"],
            "new_vs_old": cmp(n, o),
            "emb_scatter": {"old": info[o].get("emb_scatter"), "new": info[n].get("emb_scatter")},
            "memory": {"old": info[o].get("memory"), "new": info[n].get("memory")},
        }
        if b is not None:
            bo, bn = hs.spec_hash(b), hs.spec_hash(new(b))
            rec["offline_na"] = [r for r in hs.offline_na_reasons(t, b) if not r.startswith(("换 libtpu", "换镜像"))]
            rec["vs_base"] = {"old": cmp(o, bo), "new": cmp(n, bn)}
        out["items"][name] = rec
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    changed = [
        k
        for k, r in out["items"].items()
        if r and "vs_base" in r and r["vs_base"]["old"].get("program_equal") != r["vs_base"]["new"].get("program_equal")
    ]
    print(f"[hlo_screen_libtpu] 判定变了的臂：{changed or '无'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
