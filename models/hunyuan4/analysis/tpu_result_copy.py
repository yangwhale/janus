"""把 TPU 运行目录里检查脚本写的原始 JSON 收成结果文件：只把真实桶名换成 gs://<POC 桶>（仓库里不写真实桶名），另记来源运行名，数字不动。

python models/hunyuan4/analysis/tpu_result_copy.py <TPU 运行目录>/<脚本名>.json <输出 json> [--run <运行名>] [--note <说明>]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--run", default=None, help="来源 TPU 运行名（默认取源文件所在目录名）")
    p.add_argument("--note", default=None, help="附一段说明（读数的口径、注意事项），记在 note 字段")
    a = p.parse_args(argv)
    d = json.loads(Path(a.src).read_text())
    d = {"source_tpu_run": a.run or Path(a.src).parent.name, **({"note": a.note} if a.note else {}), **d}
    text = re.sub(r"gs://[^/\"]+", "gs://<POC 桶>", json.dumps(d, ensure_ascii=False, indent=1))
    Path(a.out).write_text(text + "\n")
    print(f"{a.src} → {a.out}")


if __name__ == "__main__":
    main()
