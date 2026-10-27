"""
消息队列 - 生产者 - 知识图谱（Producer_KG）：
读取 scripts/kg_build.py 生成的待入库数据（data/kg/import/），分批推入 Redis Stream kg_pipeline，
由 consumer_neo4j.py 写入 Neo4j。与 knowledge_pipeline（Milvus）互不影响，不加载向量模型。

推送顺序：[reset] → constraints → entities → chunks → relations → mentions → verify
每条消息带 built_at / ontology_version / extractor，用于版本核对与入库历史（:KGVersion）。

运行：
    python3 src/core/redis-stream/producer_kg.py            # 增量覆盖写入（MERGE，幂等）
    python3 src/core/redis-stream/producer_kg.py --reset    # 先清空旧图谱再全量写入
    python3 src/core/redis-stream/producer_kg.py --dir <待入库数据目录>
"""

import json
import sys
from pathlib import Path

import redis

from config.paths import paths
from config.config_path import REDIS_CONFIG

# ========== 配置 ==========
REDIS_STREAM = "kg_pipeline"
BATCH_SIZE = 200   # 每条消息携带的行数（一条消息 = consumer 的一个事务）
IMPORT_DIR = paths.project_root / "data" / "kg" / "import"

# 文件 → (消息类型, manifest 中对应的计数字段)
FILES = [
    ("entities.jsonl", "entities", ["entities"]),
    ("chunks.jsonl", "chunks", ["chunks"]),
    ("relations.jsonl", "relations", ["relations_asserted", "relations_inferred"]),
    ("mentions.jsonl", "mentions", ["mentions"]),
]


def connect_redis():
    """连接 Redis，测试连通性"""
    r = redis.Redis(**REDIS_CONFIG)
    try:
        r.ping()
        print("Redis 连接成功")
        return r
    except redis.ConnectionError:
        print("无法连接到 Redis，请检查服务")
        raise


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_import(import_dir: Path):
    """读取待入库数据并与 manifest 核对行数，不一致说明文件不完整或不是同一次构建的产物"""
    manifest = json.loads((import_dir / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("shacl_conforms") or manifest.get("rejected"):
        raise ValueError(f"构建未通过校验（SHACL={manifest.get('shacl_conforms')}，拒绝 {manifest.get('rejected')}），"
                         f"请先修复再入库")
    data = {}
    for fname, kind, keys in FILES:
        rows = read_jsonl(import_dir / fname)
        expected = sum(manifest[k] for k in keys)
        if len(rows) != expected:
            raise ValueError(f"{fname} 有 {len(rows)} 行，manifest 记录 {expected}")
        data[kind] = rows
    statements = [s.strip() for s in (import_dir / "constraints.cypher").read_text(encoding="utf-8").split(";")
                  if s.strip()]
    return manifest, data, statements


def build_messages(manifest, data, statements, reset=False) -> list[dict]:
    meta = {"built_at": manifest["built_at"], "ontology_version": manifest["ontology_version"],
            "extractor": manifest["extractor"], "retry_count": "0"}
    msgs = []
    if reset:
        msgs.append({"kind": "reset", "payload": "null", **meta})
    msgs.append({"kind": "constraints", "payload": json.dumps(statements, ensure_ascii=False), **meta})
    for _, kind, _ in FILES:
        rows = data[kind]
        for i in range(0, len(rows), BATCH_SIZE):
            msgs.append({"kind": kind, "batch": str(i // BATCH_SIZE),
                         "payload": json.dumps(rows[i:i + BATCH_SIZE], ensure_ascii=False), **meta})
    expected = {"entities": manifest["entities"], "chunks": manifest["chunks"],
                "relations_asserted": manifest["relations_asserted"],
                "relations_inferred": manifest["relations_inferred"], "mentions": manifest["mentions"]}
    msgs.append({"kind": "verify", "payload": json.dumps(expected), **meta})
    return msgs


def main():
    reset = "--reset" in sys.argv
    import_dir = IMPORT_DIR
    if "--dir" in sys.argv:
        idx = sys.argv.index("--dir")
        if idx + 1 < len(sys.argv):
            import_dir = Path(sys.argv[idx + 1])

    manifest, data, statements = load_import(import_dir)
    print(f"待入库数据：本体 v{manifest['ontology_version']}，构建 {manifest['built_at']}；"
          f"实体 {manifest['entities']}、关系 {manifest['relations_asserted']}+{manifest['relations_inferred']}（推理）、"
          f"chunk {manifest['chunks']}、mentions {manifest['mentions']}")

    if reset:
        ans = input("--reset 会清空 Neo4j 中现有的全部实体与 chunk（入库历史 KGVersion 保留），继续？(y/N): ")
        if ans.strip().lower() != "y":
            print("已取消")
            sys.exit(1)

    r = connect_redis()
    msgs = build_messages(manifest, data, statements, reset)
    pipe = r.pipeline()
    for msg in msgs:
        pipe.xadd(REDIS_STREAM, msg, maxlen=1000, approximate=True)
    pipe.execute()
    print(f"已推送 {len(msgs)} 条消息到 {REDIS_STREAM}；查看 consumer_neo4j 日志中的「入库核对通过」确认完成")


if __name__ == "__main__":
    main()
