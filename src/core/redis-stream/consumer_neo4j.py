"""
消息队列 - 消费者 - Neo4j：（Consumer_Neo4j）
从 Redis Stream kg_pipeline 读取知识图谱数据（producer_kg.py 推送），写入 Neo4j。

消息类型（kind）按推送顺序：
    reset        清空旧图谱，保留 KGVersion 历史（仅 producer_kg.py --reset 时推送）
    constraints  执行 constraints.cypher：实体 / chunk 唯一约束、实体名全文索引
    entities     实体节点  (:Entity:<祖先类>:<类> {entity_id, ...})
    chunks       原文句子  (:Chunk {chunk_id, ...})，chunk_id 与 Milvus 一致
    relations    事实关系  (a)-[:<TYPE> {rel_id, evidence, source_chunks, ...}]->(b)
    mentions     (:Chunk)-[:MENTIONS]->(:Entity)，GraphRAG 回查原文用
    verify       与 manifest.json 核对节点数 / 关系数，记录 (:KGVersion)

写入幂等：节点按 entity_id / chunk_id、关系按 rel_id MERGE，属性整体覆盖为本次构建的值，重复消费结果不变。
关系 / mentions 的端点不存在时整批失败重试（不静默丢弃）；重试超过 MAX_RETRIES 次转入死信队列。
"""

import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime

import redis
from neo4j import GraphDatabase

from config.config_path import REDIS_CONFIG, NEO4J_CONFIG

# ========== 配置 ==========
REDIS_STREAM = "kg_pipeline"
REDIS_DLQ = "kg_pipeline_dlq"
CONSUMER_GROUP = "group_neo4j"
CONSUMER_NAME = f"consumer_neo4j_{os.getpid()}"
MAX_RETRIES = 3
BATCH_SIZE = 10
BLOCK_MS = 5000
REDIS_CONNECT_RETRIES = 5
REDIS_RETRY_DELAY = 2

NEO4J_URI      = NEO4J_CONFIG['url']
NEO4J_USER     = NEO4J_CONFIG.get('user', 'neo4j')
NEO4J_PASSWORD = NEO4J_CONFIG['password']
NEO4J_DATABASE = NEO4J_CONFIG.get('database', 'neo4j')

# 标签 / 关系类型无法参数化，只能拼进 Cypher：按本体命名规则校验，防注入
LABEL_RE = re.compile(r"[A-Z][A-Za-z]*")
REL_TYPE_RE = re.compile(r"[A-Z][A-Z_]*")
KINDS = ("reset", "constraints", "entities", "chunks", "relations", "mentions", "verify")
# 约束语句只允许这几种开头
DDL_RE = re.compile(r"CREATE (CONSTRAINT|INDEX|FULLTEXT INDEX) [a-z_]+ IF NOT EXISTS ", re.I)

# ========== 日志 ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IncompleteWrite(Exception):
    """端点尚未入库等原因导致本批只写入了一部分，需要重试"""


# ========== Redis 连接（带重试） ==========
def connect_redis():
    cfg = {
        **REDIS_CONFIG,
        'decode_responses': True,
        'socket_timeout': BLOCK_MS / 1000 + 10,   # 比阻塞窗口多 10s，避免误触发超时
    }
    for i in range(REDIS_CONNECT_RETRIES):
        try:
            r = redis.Redis(**cfg)
            r.ping()
            logger.info("Redis 连接成功")
            return r
        except redis.ConnectionError as e:
            logger.warning(f"Redis 连接失败 ({i+1}/{REDIS_CONNECT_RETRIES}): {e}")
            if i == REDIS_CONNECT_RETRIES - 1:
                raise
            time.sleep(REDIS_RETRY_DELAY)


# ========== 确保消费者组存在 ==========
def ensure_consumer_group(redis_client):
    # id='0'：consumer 晚于 producer 启动时也能读到已推送的消息（写入幂等，重复读取无副作用）
    try:
        redis_client.xgroup_create(REDIS_STREAM, CONSUMER_GROUP, id='0', mkstream=True)
        logger.info(f"消费者组 {CONSUMER_GROUP} 创建成功")
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e):
            logger.info(f"消费者组 {CONSUMER_GROUP} 已存在")
        else:
            raise


# ========== Neo4j 写入 ==========
class Neo4jClient:
    def __init__(self, uri, user, password, database):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        self.database = database

    def close(self):
        self.driver.close()

    def _write(self, query, **params):
        records, _, _ = self.driver.execute_query(query, parameters_=params, database_=self.database)
        return records

    # ---- 各类消息 ----
    def reset(self, _payload, _meta):
        """清空旧图谱（实体、chunk 及其关系），分批删除避免大事务；KGVersion 入库历史保留"""
        with self.driver.session(database=self.database) as session:
            for label in ("Entity", "Chunk"):
                session.run(f"MATCH (n:{label}) CALL (n) {{ DETACH DELETE n }} IN TRANSACTIONS OF 1000 ROWS").consume()
        logger.info("旧图谱已清空")

    def constraints(self, statements, _meta):
        for stmt in statements:
            if not DDL_RE.match(stmt):
                raise ValueError(f"拒绝执行非约束 / 索引语句: {stmt[:80]}")
            self._write(stmt)
        logger.info(f"约束与索引 {len(statements)} 条已创建 / 存在")

    def entities(self, rows, meta):
        by_labels = defaultdict(list)
        for row in rows:
            labels = row.pop("labels")
            bad = [l for l in labels if not LABEL_RE.fullmatch(l)]
            if bad or "Entity" not in labels:
                raise ValueError(f"实体 {row.get('entity_id')} 标签不合规: {labels}")
            row.update(meta)
            by_labels[tuple(labels)].append(row)
        n = 0
        for labels, group in by_labels.items():
            label_str = ":".join(labels)
            # SET n = row：属性整体替换为本次构建的值（上一版残留的属性一并清除）
            res = self._write(f"""
                UNWIND $rows AS row
                MERGE (n:Entity {{entity_id: row.entity_id}})
                SET n = row
                SET n:{label_str}
                RETURN count(n) AS c""", rows=group)
            n += res[0]["c"]
        return n

    def chunks(self, rows, meta):
        for row in rows:
            row.update(meta)
        res = self._write("""
            UNWIND $rows AS row
            MERGE (c:Chunk {chunk_id: row.chunk_id})
            SET c = row
            RETURN count(c) AS c""", rows=rows)
        return res[0]["c"]

    def relations(self, rows, meta):
        by_type = defaultdict(list)
        for row in rows:
            rtype = row.pop("type")
            if not REL_TYPE_RE.fullmatch(rtype):
                raise ValueError(f"关系 {row.get('rel_id')} 类型不合规: {rtype}")
            start, end = row.pop("start"), row.pop("end")
            row.update(meta)
            by_type[rtype].append({"start": start, "end": end, "props": row})
        n = 0
        for rtype, group in by_type.items():
            res = self._write(f"""
                UNWIND $rows AS row
                MATCH (a:Entity {{entity_id: row.start}})
                MATCH (b:Entity {{entity_id: row.end}})
                MERGE (a)-[r:{rtype} {{rel_id: row.props.rel_id}}]->(b)
                SET r = row.props
                RETURN count(r) AS c""", rows=group)
            n += res[0]["c"]
        if n < len(rows):
            raise IncompleteWrite(f"关系写入 {n}/{len(rows)}，部分端点实体尚未入库")
        return n

    def mentions(self, rows, _meta):
        res = self._write("""
            UNWIND $rows AS row
            MATCH (c:Chunk {chunk_id: row.chunk_id})
            MATCH (e:Entity {entity_id: row.entity_id})
            MERGE (c)-[m:MENTIONS]->(e)
            RETURN count(m) AS c""", rows=rows)
        n = res[0]["c"]
        if n < len(rows):
            raise IncompleteWrite(f"mentions 写入 {n}/{len(rows)}，部分 chunk / 实体尚未入库")
        return n

    def verify(self, expected, meta):
        """按本次构建版本统计，与 manifest 对比；不一致时抛出重试（可能还有批次在重试队列中）"""
        v = meta["kg_built_at"]
        actual = {
            "entities": self._write("MATCH (n:Entity {kg_built_at: $v}) RETURN count(n) AS c", v=v)[0]["c"],
            "chunks": self._write("MATCH (c:Chunk {kg_built_at: $v}) RETURN count(c) AS c", v=v)[0]["c"],
            "relations_asserted": self._write(
                "MATCH ()-[r {kg_built_at: $v, inferred: false}]->() RETURN count(r) AS c", v=v)[0]["c"],
            "relations_inferred": self._write(
                "MATCH ()-[r {kg_built_at: $v, inferred: true}]->() RETURN count(r) AS c", v=v)[0]["c"],
            "mentions": self._write("MATCH (:Chunk {kg_built_at: $v})-[m:MENTIONS]->() RETURN count(m) AS c", v=v)[0]["c"],
        }
        stale = self._write("MATCH (n:Entity) WHERE n.kg_built_at <> $v RETURN count(n) AS c", v=v)[0]["c"]
        diff = {k: (expected[k], actual[k]) for k in expected if expected[k] != actual[k]}
        if diff:
            raise IncompleteWrite(f"入库核对不一致（期望, 实际）: {diff}")
        self._write("""
            MERGE (k:KGVersion {built_at: $v})
            SET k += $props""", v=v, props={
            "ontology_version": meta["ontology_version"], "extractor": meta["kg_extractor"],
            "imported_at": datetime.now().isoformat(timespec="seconds"), **actual})
        logger.info(f"✅ 入库核对通过（本体 v{meta['ontology_version']}，构建 {v}）：{actual}")
        if stale:
            logger.warning(f"图中还有 {stale} 个旧版本实体（未用 --reset 重建），按需清理")
        return sum(actual.values())


# ========== 消息处理 ==========
def handle(neo4j_client, msg_data):
    kind = msg_data["kind"]
    if kind not in KINDS:
        raise ValueError(f"未知消息类型: {kind}")
    handler = getattr(neo4j_client, kind)
    payload = json.loads(msg_data["payload"])
    meta = {"kg_built_at": msg_data["built_at"], "ontology_version": msg_data["ontology_version"],
            "kg_extractor": msg_data["extractor"]}
    n = handler(payload, meta)
    if kind in ("entities", "chunks", "relations", "mentions"):
        logger.info(f"{kind} 批次 {msg_data.get('batch', '?')} 写入 {n} 条")


# ========== 主循环 ==========
def main():
    r = connect_redis()
    ensure_consumer_group(r)

    neo4j_client = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)
    logger.info("Neo4j 连接成功，消费者启动，等待消息...")

    while True:
        try:
            results = r.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername=CONSUMER_NAME,
                streams={REDIS_STREAM: '>'},
                count=BATCH_SIZE,
                block=BLOCK_MS
            )
            if not results:
                continue

            for stream, messages in results:
                for msg_id, msg_data in messages:
                    retry_count = int(msg_data.get("retry_count") or 0)
                    try:
                        handle(neo4j_client, msg_data)
                        r.xack(REDIS_STREAM, CONSUMER_GROUP, msg_id)
                        continue
                    except (ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as e:
                        # 消息本身不合规：重试也不会成功，直接进死信队列
                        logger.error(f"消息 {msg_id}（{msg_data.get('kind')}）不合规，移入死信队列: {e}")
                        r.xadd(REDIS_DLQ, {**msg_data, "error": str(e)[:500]})
                        r.xack(REDIS_STREAM, CONSUMER_GROUP, msg_id)
                        continue
                    except Exception as e:
                        logger.warning(f"消息 {msg_id}（{msg_data.get('kind')} 批次 {msg_data.get('batch', '?')}）写入失败: {e}")

                    new_retry = retry_count + 1
                    if new_retry > MAX_RETRIES:
                        logger.error(f"消息 {msg_id} 重试超过 {MAX_RETRIES} 次，移入死信队列")
                        r.xadd(REDIS_DLQ, msg_data)
                    else:
                        msg_data["retry_count"] = str(new_retry)
                        r.xadd(REDIS_STREAM, msg_data)
                        logger.info(f"消息 {msg_id} 重试 {new_retry} 次，已重新加入队列")
                    r.xack(REDIS_STREAM, CONSUMER_GROUP, msg_id)

        except redis.exceptions.TimeoutError:
            continue
        except KeyboardInterrupt:
            logger.info("消费者被手动停止")
            break
        except Exception as e:
            logger.exception(f"主循环异常: {e}")
            time.sleep(5)

    neo4j_client.close()


if __name__ == "__main__":
    main()
