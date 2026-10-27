"""
Milvus 知识库快照导出脚本

从生产 Milvus 集合中导出飞轮 Agent 1 所需的最小字段集，
输出为 data/evaluation/kb_snapshot.json。

只取 5 个字段，跳过向量字段（vector/sparse），控制文件大小。

运行：
  docker compose exec app python scripts/export_kb_snapshot.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.config_path import MILVUS_CONFIG, MILVUS_COLLECTION
from pymilvus import MilvusClient

OUT_PATH = ROOT / "data/evaluation/kb_snapshot.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

FIELDS = ["chunk_id", "document_id", "topic", "content", "wizard_eval_score", "dialog_id"]
BATCH  = 500   # 每次查询批次大小


def main():
    client = MilvusClient(**MILVUS_CONFIG)

    stats = client.get_collection_stats(collection_name=MILVUS_COLLECTION)
    total = int(stats["row_count"])
    print(f"集合：{MILVUS_COLLECTION}  总行数：{total}")

    all_records = []
    offset = 0

    while offset < total:
        batch = client.query(
            collection_name=MILVUS_COLLECTION,
            filter="",
            output_fields=FIELDS,
            limit=BATCH,
            offset=offset,
        )
        all_records.extend(batch)
        offset += len(batch)
        print(f"  已导出 {offset}/{total}", end="\r", flush=True)
        if len(batch) < BATCH:
            break

    print(f"\n导出完成，共 {len(all_records)} 条")

    # 过滤空 content
    before = len(all_records)
    all_records = [r for r in all_records if r.get("content", "").strip()]
    print(f"过滤空 content：跳过 {before - len(all_records)} 条，剩余 {len(all_records)} 条")

    # 按 document_id 去重：同一 dialog chunk 的多行 content 完全相同
    # 每行 vector 不同（单条 CS），但 content = 全部 CS 合并，Agent 1 只需唯一 content
    seen_doc_ids = set()
    deduped = []
    for r in all_records:
        doc_id = r.get("document_id") or r.get("chunk_id")
        if doc_id not in seen_doc_ids:
            seen_doc_ids.add(doc_id)
            deduped.append(r)
    print(f"按 document_id 去重：{len(all_records)} 行 → {len(deduped)} 条唯一 content 块")
    all_records = deduped

    # 按 dialog_id 排序，方便 multi-hop 配对时查找相邻 chunk
    all_records.sort(key=lambda r: (r.get("dialog_id", ""), r.get("chunk_id", "")))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    # 统计摘要
    topics     = {r["topic"] for r in all_records}
    dialog_ids = {r.get("dialog_id") for r in all_records}
    scores     = [r.get("wizard_eval_score", -1) for r in all_records]
    score_dist = {}
    for s in scores:
        score_dist[s] = score_dist.get(s, 0) + 1

    print(f"\n快照摘要")
    print(f"  总条数      : {len(all_records)}")
    print(f"  涉及话题数  : {len(topics)}")
    print(f"  涉及对话数  : {len(dialog_ids)}")
    print(f"  wizard_eval_score 分布: {dict(sorted(score_dist.items()))}")
    print(f"  输出文件    : {OUT_PATH}")


if __name__ == "__main__":
    main()
