"""
消息队列 - 消费者 - Milvus：（Consumer_Milvus）
消费者：从 Redis Stream 读取消息，写入 Milvus（upsert）。
包含死信队列处理：重试超过 MAX_RETRIES 的消息转入死信队列。
"""
import json
import time
import redis
from pymilvus import MilvusClient
import logging
import os
from config.config_path import REDIS_CONFIG, MILVUS_CONFIG, MILVUS_COLLECTION

# ========== 配置 ==========
REDIS_STREAM = "knowledge_pipeline"
REDIS_DLQ = "knowledge_pipeline_dlq"        # 死信队列名称
CONSUMER_GROUP = "group_milvus"
CONSUMER_NAME = f"consumer_{os.getpid()}"    # 每个实例唯一
MAX_RETRIES = 3                               # 最大重试次数
BATCH_SIZE = 10                                # 每次读取消息数量
BLOCK_MS = 5000                                # 没有消息时阻塞 5 秒


def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def connect_redis(decode=True):
    """连接 Redis。
    socket_timeout 必须大于 BLOCK_MS，否则阻塞等待消息时会误触发超时异常。
    """
    cfg = {
        **REDIS_CONFIG,
        'decode_responses': decode,
        'socket_timeout': BLOCK_MS / 1000 + 10,   # 比阻塞窗口多 10s 的余量
    }
    r = redis.Redis(**cfg)
    r.ping()
    return r


def ensure_consumer_group(redis_client, stream, group):
    """确保消费者组存在"""
    try:
        redis_client.xgroup_create(stream, group, id="$", mkstream=True)
        logger.info(f"消费者组 {group} 创建成功")
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e):
            logger.info(f"消费者组 {group} 已存在")
        else:
            raise


def connect_milvus():
    """连接 Milvus 并加载集合"""
    client = MilvusClient(**MILVUS_CONFIG)
    client.load_collection(MILVUS_COLLECTION)
    logger.info("Milvus 连接成功，集合已加载")
    return client


def deserialize_message(msg_data):
    processed = {}
    # 需要反序列化为 JSON 对象的字段（包括之前未处理的）
    json_fields = [
        "vector",
        "keywords_text",
        "ner_entities_text",
        "keywords_json",
        "ner_entities_json",
        "relations_json",
    ]

    for k, v in msg_data.items():
        if k in json_fields:
            try:
                processed[k] = json.loads(v) if isinstance(v, str) else v
            except:
                processed[k] = v
        elif k in [
            "chunk_seq",
            "content_length",
            "dialog_index",
            "turn_count",
            "wizard_eval_score",
            "total_retrieved_passages",
            "total_retrieved_topics",
            "retry_count",
        ]:
            processed[k] = int(v) if v else 0
        elif k in ["evidence_density"]:
            processed[k] = float(v) if v else 0.0
        elif k in ["has_checked_evidence"]:
            processed[k] = v in ("True", "true", True)
        else:
            processed[k] = v
    return processed


def upsert_to_milvus(milvus_client, collection, data):
    """执行 upsert 操作"""
    milvus_client.upsert(collection, [data])


def process_message(msg_data, milvus_client):
    """
    处理单条消息：将数据 upsert 到 Milvus。
    返回 True 表示成功，False 表示失败（需重试）。
    """
    chunk_id = msg_data.get("chunk_id")
    if not chunk_id:
        logger.error("消息缺少 chunk_id，丢弃")
        return True  # 无法处理，视为成功并移除

    # 检查是否为纯知识图谱消息（无向量）
    vector = msg_data.get("vector")
    content = msg_data.get("content")
    if not vector or not content:
        logger.info(f"消息 {chunk_id} 无向量或内容，跳过写入 Milvus")
        return True

    try:
        # 准备数据（直接使用消息中的字段）
        data = {
            "chunk_id": chunk_id,
            "vector": msg_data.get("vector"),
            "content": msg_data.get("content"),
            "source": msg_data.get("source"),
            "document_id": msg_data.get("document_id"),
            "chunk_seq": msg_data.get("chunk_seq"),
            "content_length": msg_data.get("content_length"),
            "content_hash": msg_data.get("content_hash"),
            "dialog_id": msg_data.get("dialog_id"),
            "dialog_index": msg_data.get("dialog_index"),
            "topic": msg_data.get("topic"),
            "turn_count": msg_data.get("turn_count"),
            "speaker_role": msg_data.get("speaker_role"),
            "wizard_eval_score": msg_data.get("wizard_eval_score"),
            "has_checked_evidence": msg_data.get("has_checked_evidence"),
            "total_retrieved_passages": msg_data.get("total_retrieved_passages"),
            "total_retrieved_topics": msg_data.get("total_retrieved_topics"),
            "evidence_density": msg_data.get("evidence_density"),
            "prev_chunk_id": msg_data.get("prev_chunk_id", ""),
            "next_chunk_id": msg_data.get("next_chunk_id", ""),
            "split_type": msg_data.get("split_type"),
            "keywords_text": msg_data.get("keywords_text"),
            "ner_entities_text": msg_data.get("ner_entities_text"),
            "keywords_json": msg_data.get("keywords_json"),
            "ner_entities_json": msg_data.get("ner_entities_json"),
            "relations_json": msg_data.get("relations_json"),
            "cs_text": msg_data.get("cs_text", ""),
        }
        upsert_to_milvus(milvus_client, MILVUS_COLLECTION, data)
        logger.info(f"Upsert 成功: {chunk_id}")
        return True
    except Exception as e:
        logger.error(f"处理消息失败: {e}")
        return False


def handle_dead_letter(redis_client, msg_data, msg_id, retry_count):
    """将消息移入死信队列，并从原队列 ACK"""
    logger.warning(f"消息 {msg_id} 重试超过 {MAX_RETRIES} 次，移入死信队列")
    msg_data["retry_count"] = retry_count + 1
    redis_client.xadd(REDIS_DLQ, msg_data)
    redis_client.xack(REDIS_STREAM, CONSUMER_GROUP, msg_id)


def main_loop(redis_client, milvus_client):
    """消费者主循环"""
    logger.info("消费者启动，等待消息...")
    while True:
        try:
            # 从消费者组读取消息
            results = redis_client.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername=CONSUMER_NAME,
                streams={REDIS_STREAM: ">"},
                count=BATCH_SIZE,
                block=BLOCK_MS,
            )

            if not results:
                continue

            # results 格式: [[stream_name, [(msg_id, msg_data), ...]]]
            for stream, messages in results:
                for msg_id, msg_data in messages:
                    # 反序列化
                    processed_data = deserialize_message(msg_data)
                    retry_count = processed_data.get("retry_count", 0)

                    # 处理消息
                    success = process_message(processed_data, milvus_client)

                    if success:
                        # 成功，ACK
                        redis_client.xack(REDIS_STREAM, CONSUMER_GROUP, msg_id)
                        logger.debug(f"ACK 消息 {msg_id}")
                    else:
                        # 失败：重试次数+1
                        new_retry = retry_count + 1
                        if new_retry > MAX_RETRIES:
                            # 超过阈值，移入死信队列
                            handle_dead_letter(
                                redis_client, processed_data, msg_id, retry_count
                            )
                        else:
                            # 未超限，重新添加消息并 ACK 旧消息，更新重试计数
                            logger.info(
                                f"消息 {msg_id} 重试 {new_retry} 次，重新加入队列"
                            )
                            processed_data["retry_count"] = new_retry
                            redis_client.xadd(REDIS_STREAM, processed_data)
                            redis_client.xack(REDIS_STREAM, CONSUMER_GROUP, msg_id)

        except redis.exceptions.TimeoutError:
            # xreadgroup 阻塞窗口内无消息，属于正常情况，静默继续
            continue
        except KeyboardInterrupt:
            logger.info("消费者被手动停止")
            break
        except Exception as e:
            logger.exception(f"主循环异常: {e}")
            time.sleep(5)


def main():
    global logger
    logger = setup_logging()

    # 连接 Redis
    r = connect_redis()
    ensure_consumer_group(r, REDIS_STREAM, CONSUMER_GROUP)

    # 连接 Milvus
    milvus_client = connect_milvus()

    # 进入主循环
    main_loop(r, milvus_client)


if __name__ == "__main__":
    main()
