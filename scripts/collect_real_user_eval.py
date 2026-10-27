"""
真实用户问题采集脚本

读取 data/evaluation/real_user_questions.json（人工编写的问题+ground_truth），
逐条调用 /chat_eval 接口获取 answer + contexts，
输出为 ragas_input 格式，可直接混入飞轮评估集。

使用方式：
  # 确保系统已启动（docker compose up 或 uvicorn）
  python scripts/collect_real_user_eval.py

  # 指定 API 地址（默认 localhost:8000）
  python scripts/collect_real_user_eval.py --host http://your_server:8000

  # 测试模式（只跑前 3 条）
  python scripts/collect_real_user_eval.py --test
"""

import json
import os
import time
import argparse
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
sys.stdout.reconfigure(encoding="utf-8")

# /chat_eval 仅限 ops / admin：从环境变量 WOW_OPS_TOKEN 读取 ops 账号令牌（服务关闭认证时可不设）
OPS_HEADERS = {"Authorization": f"Bearer {os.environ['WOW_OPS_TOKEN']}"} if os.getenv("WOW_OPS_TOKEN") else {}

INPUT_PATH  = ROOT / "data/evaluation/real_user_questions.json"
OUTPUT_PATH = ROOT / "data/evaluation/real_user_eval_collected.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",    default="http://localhost:8000")
    parser.add_argument("--test",    action="store_true", help="只跑前 3 条")
    parser.add_argument("--sleep",   type=float, default=2.0, help="每条间隔秒数")
    parser.add_argument("--session", default="eval_real_user", help="session_id 前缀")
    return parser.parse_args()


def call_chat_eval(host: str, question: str, session_id: str, timeout: int = 120) -> dict | None:
    try:
        resp = requests.post(
            f"{host}/chat_eval",
            json={"user_input": question, "session_id": session_id, "user_id": "eval"},
            headers=OPS_HEADERS,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  ⚠️  请求失败: {e}")
        return None


def main():
    args = parse_args()

    if not INPUT_PATH.exists():
        print(f"❌ 找不到输入文件: {INPUT_PATH}")
        print("   请先创建 data/evaluation/real_user_questions.json")
        sys.exit(1)

    with open(INPUT_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    if args.test:
        questions = questions[:3]
        print(f"🔬 测试模式：仅采集前 3 条\n")

    print(f"📋 待采集问题数: {len(questions)}")
    print(f"🌐 API 地址: {args.host}\n")

    # 断点续传
    collected = []
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            collected = json.load(f)
        done_ids = {r["id"] for r in collected}
        questions = [q for q in questions if q.get("id") not in done_ids]
        print(f"♻️  断点续传：已完成 {len(collected)} 条，剩余 {len(questions)} 条\n")

    start_id = max((r["id"] for r in collected), default=-1) + 1

    for idx, q in enumerate(questions):
        qid      = q.get("id", start_id + idx)
        question = q["question"]
        gt       = q.get("ground_truth", "")
        topic    = q.get("topic", "real_user")
        qtype    = q.get("question_type", "simple")

        print(f"[{idx+1:03d}/{len(questions)}] id={qid}  {question[:60]}")

        result = call_chat_eval(args.host, question, f"{args.session}_{qid}")

        if result is None:
            print("  ❌ 跳过（请求失败）\n")
            continue

        record = {
            "id":            qid,
            "chunk_id":      "real_user",
            "topic":         topic,
            "question_type": qtype,
            "question":      question,
            "ground_truth":  gt,
            "answer":        result["answer"],
            "contexts":      result["contexts"],
        }
        collected.append(record)

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(collected, f, ensure_ascii=False, indent=2)

        print(f"  ✅ contexts={len(result['contexts'])} 条  answer={len(result['answer'])} 字\n")

        if idx < len(questions) - 1:
            time.sleep(args.sleep)

    print(f"{'='*50}")
    print(f"采集完成，共 {len(collected)} 条")
    print(f"输出文件: {OUTPUT_PATH}")
    print(f"\n下一步：将 real_user_eval_collected.json 与飞轮数据合并后跑 RAGAS")


if __name__ == "__main__":
    main()
