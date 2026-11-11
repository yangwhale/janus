"""把运行目录原样传到 GCS（逐文件，保留相对路径），传完逐文件核对大小。

用法：python upload_run.py <运行目录> gs://<桶>/<前缀>/<运行名>
凭据用默认凭据（GKE 上是 Workload Identity）。需要 google-cloud-storage。
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google.cloud import storage


def main(src, dst):
    src = Path(src)
    assert dst.startswith("gs://")
    bucket_name, _, prefix = dst[5:].partition("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    files = [p for p in sorted(src.rglob("*")) if p.is_file()]

    def up(p):
        blob = bucket.blob(f"{prefix}/{p.relative_to(src).as_posix()}")
        blob.chunk_size = 64 * 1024 * 1024
        blob.upload_from_filename(str(p))
        blob.reload()
        return p, blob.size == p.stat().st_size
    with ThreadPoolExecutor(16) as ex:
        bad = [str(p) for p, ok in ex.map(up, files) if not ok]
    total = sum(p.stat().st_size for p in files)
    print(f"{len(files)} 个文件、{total / 1e9:.2f} GB → {dst}；大小不符 {len(bad)} 个")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
