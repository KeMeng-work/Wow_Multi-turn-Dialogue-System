"""
Milvus flush + 等待索引完成脚本

用途：每次 producer 入库结束后运行一次，确保所有新数据从
      Growing Segment 晋升为 Sealed Segment 并完成索引构建。

运行：
  docker compose exec app python scripts/flush_milvus.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pymilvus import MilvusClient
from config.config_path import MILVUS_CONFIG, MILVUS_COLLECTION

POLL_INTERVAL = 5   # 轮询间隔（秒）
TIMEOUT       = 300 # 最长等待（秒），超时报警但不退出


def main():
    client = MilvusClient(**MILVUS_CONFIG)

    # ── 当前状态 ──────────────────────────────────────────────
    stats  = client.get_collection_stats(collection_name=MILVUS_COLLECTION)
    total  = int(stats["row_count"])
    print(f"集合：{MILVUS_COLLECTION}，当前总行数：{total}")

    # ── Flush：强制 Growing Segment → Sealed ─────────────────
    print("Flushing（封存增量 segment）...", flush=True)
    client.flush(collection_name=MILVUS_COLLECTION)
    print("Flush 完成，等待索引构建...", flush=True)

    # ── 轮询索引进度（vector + sparse 双字段）────────────────
    def get_indexed_rows(field: str) -> int:
        try:
            desc = client.describe_index(collection_name=MILVUS_COLLECTION, field_name=field)
            return desc.get("indexed_rows", 0)
        except Exception:
            return 0

    elapsed = 0
    while elapsed < TIMEOUT:
        stats          = client.get_collection_stats(collection_name=MILVUS_COLLECTION)
        total          = int(stats["row_count"])
        vec_indexed    = get_indexed_rows("vector")
        sparse_indexed = get_indexed_rows("sparse")

        print(
            f"  [{elapsed:3d}s] vector={vec_indexed}/{total}  "
            f"sparse={sparse_indexed}/{total}",
            flush=True,
        )

        if vec_indexed >= total and sparse_indexed >= total:
            print("✅ vector + sparse 索引均构建完成！")
            break

        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    else:
        print(f"⚠️  超过 {TIMEOUT}s 仍未完成，Milvus 后台会继续构建，可稍后再查。")

    # ── Reload：让查询节点加载最新索引 ───────────────────────
    print("Reloading collection...", flush=True)
    client.release_collection(collection_name=MILVUS_COLLECTION)
    client.load_collection(collection_name=MILVUS_COLLECTION)
    print("✅ 集合已重新加载，全量数据可高效检索。")

    # ── 最终状态汇报 ──────────────────────────────────────────
    stats = client.get_collection_stats(collection_name=MILVUS_COLLECTION)
    print(f"\n最终行数：{stats['row_count']}")


if __name__ == "__main__":
    main()
