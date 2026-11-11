"""从 GCS 并行分段下载一个对象（Pod 里没有 gcloud；单流 curl 跨区只有几十 MB/s）。

    python3 pget.py <桶> <对象名> <本地路径> [--threads 16]
凭据取 GKE 元数据服务器的 token（Workload Identity）。
"""

import argparse
import concurrent.futures as cf
import json
import os
import time
import urllib.parse
import urllib.request


def token():
    r = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    return json.loads(urllib.request.urlopen(r, timeout=30).read())["access_token"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bucket")
    ap.add_argument("obj")
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    o = urllib.parse.quote(a.obj, safe="")
    base = f"https://storage.googleapis.com/storage/v1/b/{a.bucket}/o/{o}"
    tk = token()
    meta = json.loads(
        urllib.request.urlopen(
            urllib.request.Request(base, headers={"Authorization": f"Bearer {tk}"}), timeout=60
        ).read()
    )
    size = int(meta["size"])
    chunk = 64 << 20
    tmp = a.out + ".part"
    with open(tmp, "wb") as f:
        f.truncate(size)

    def get(off):
        end = min(size, off + chunk) - 1
        for i in range(8):
            try:
                r = urllib.request.Request(
                    base + "?alt=media", headers={"Authorization": f"Bearer {tk}", "Range": f"bytes={off}-{end}"}
                )
                data = urllib.request.urlopen(r, timeout=120).read()
                assert len(data) == end - off + 1
                with open(tmp, "r+b") as f:
                    f.seek(off)
                    f.write(data)
                return
            except Exception:  # noqa: BLE001 - 重试网络错误
                time.sleep(2 + 2 * i)
        raise RuntimeError(f"failed {a.obj} @ {off}")

    with cf.ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(get, range(0, size, chunk)))
    os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
