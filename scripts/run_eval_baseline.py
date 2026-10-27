"""
离线评估基线采集脚本 v3

两种运行模式：

【基线模式（默认）】
  读取 questions_100.json + ground_truths_100.json，输出 eval_results_100.json。
  docker compose exec app python scripts/run_eval_baseline.py

【飞轮模式（--from-batch N）】
  读取 ragas_input_batch_N_deduped.json（冻结评估集），用更新后的系统重新作答，
  输出 ragas_input_batch_{N+1}.json，完整保留 chunk_id / question_type / ground_truth。
  用于 Round N+1 评估：相同问题集，新系统答案，结果可与 Round N 直接对比。
  docker compose exec app python scripts/run_eval_baseline.py --from-batch 1
  docker compose exec app python scripts/run_eval_baseline.py --from-batch 1 --rerun

共同特性：
  - 超时 read=240s，给 ReAct 5 步推理留足时间
  - 每题最多重试 3 次，指数退避（10s / 20s / 30s）
  - 空答案视为失败并触发重试
  - 断点续传：默认跳过已有有效答案的 ID
  - --rerun：强制重跑全部
  - --ids：只跑指定 ID（逗号分隔）
  - 结果按 ID 排序后写入
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8")

# ── 路径 ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

# ── 超时 / 重试配置 ───────────────────────────────────────────────────────────
CONNECT_TIMEOUT = 15
READ_TIMEOUT    = 240
MAX_RETRIES     = 3
RETRY_DELAYS    = [10, 20, 30]

# ── 参数 ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--api",        default="http://localhost:8000", help="API 基地址")
parser.add_argument("--delay",      type=float, default=2.0, help="每题成功后的间隔秒数（限速）")
parser.add_argument("--rerun",      action="store_true", help="忽略已有结果，强制重跑全部")
parser.add_argument("--ids",        default="", help="只跑指定 ID，逗号分隔（配合 --rerun 使用）")
parser.add_argument("--from-batch", type=int, default=None,
                    help="飞轮模式：从 ragas_input_batch_N_deduped.json 读题，输出 batch_{N+1}.json")
parser.add_argument("--user-id",    default="eval_r3_v1t_all",
                    help="评估用户名；每批次须用新用户名，避免读到其他批次积累的长期记忆")
args = parser.parse_args()

STREAM_URL = f"{args.api.rstrip('/')}/chat/stream"
USER_ID    = args.user_id

# ── 路径解析（基线 vs 飞轮）────────────────────────────────────────────────────
if args.from_batch is not None:
    _src_deduped = ROOT / f"data/evaluation/ragas_input_batch_{args.from_batch}_deduped.json"
    _src_raw     = ROOT / f"data/evaluation/ragas_input_batch_{args.from_batch}.json"
    if _src_deduped.exists():
        SRC_PATH = _src_deduped
    elif _src_raw.exists():
        print(f"⚠️  未找到去重版本，回退到原始文件 {_src_raw.name}")
        SRC_PATH = _src_raw
    else:
        print(f"❌ 找不到输入文件（{_src_deduped.name} 或 {_src_raw.name}）")
        sys.exit(1)
    # 文件名带上 user_id：每轮评测产出独立文件，避免"结果文件已存在 → 断点续传跳过全部题目"
    OUT_PATH  = ROOT / f"data/evaluation/ragas_input_batch_{args.from_batch + 1}_{args.user_id}.json"
    FLYWHEEL  = True
else:
    SRC_PATH  = None   # 基线模式用 Q_PATH + GT_PATH
    OUT_PATH  = ROOT / "data/evaluation/eval_results_100.json"
    FLYWHEEL  = False

Q_PATH  = ROOT / "data/evaluation/questions_100.json"
GT_PATH = ROOT / "data/evaluation/ground_truths_100.json"


def call_stream(question: str, idx: int) -> dict:
    """
    调用 /chat/stream，收集完整 answer + contexts。
    返回 {"answer": str, "contexts": [str], "ok": bool}
    ok=False 表示请求失败或 answer 为空（视为无效，需重试）。
    """
    session_id = f"eval_{idx}_{uuid.uuid4().hex[:8]}"
    payload = {
        "user_input": question,
        "session_id": session_id,
        "user_id":    USER_ID,
    }
    answer   = ""
    contexts = []
    error    = None
    done     = False
    tools    = []   # 实际调用的工具（stream 不推送 intent，据此判断走了 local / web_search / direct 路径）
    meta     = {}   # 检索分档元信息（refused / evidence_level / retrieval_outcomes）

    try:
        with requests.post(
            STREAM_URL,
            json=payload,
            stream=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()

            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data_str = raw_line[len("data:"):].strip()
                if not data_str:
                    continue
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type")
                if etype == "answer":
                    answer += event.get("content", "")
                elif etype == "contexts":
                    raw_ctx = event.get("data", [])
                    if isinstance(raw_ctx, list):
                        contexts.extend(str(c) for c in raw_ctx if c)
                elif etype == "retrieval_meta":
                    meta = {k: v for k, v in event.items() if k != "type"}
                elif etype == "tool_call" and event.get("status") == "start":
                    tools.append(event.get("name", ""))
                elif etype == "error":
                    error = f"{event.get('code')} {event.get('detail', '')}"
                elif etype == "done":
                    done = True

    except requests.exceptions.ConnectTimeout:
        print(f"    ⚠️  连接超时（{CONNECT_TIMEOUT}s），服务器未响应")
        return {"answer": "", "contexts": [], "ok": False}
    except requests.exceptions.ReadTimeout:
        print(f"    ⚠️  读取超时（{READ_TIMEOUT}s），推理耗时过长")
        return {"answer": "", "contexts": [], "ok": False}
    except requests.exceptions.HTTPError as e:
        print(f"    ⚠️  HTTP 错误: {e.response.status_code}")
        return {"answer": "", "contexts": [], "ok": False}
    except Exception as e:
        print(f"    ⚠️  请求异常: {type(e).__name__}: {e}")
        return {"answer": "", "contexts": [], "ok": False}

    # 服务端中途报错时可能已推送部分答案，须同时满足：无 error 事件且收到 done
    if error:
        print(f"    ⚠️  服务端错误事件: {error[:120]}")
        return {"answer": "", "contexts": [], "ok": False}
    if not done:
        print("    ⚠️  流未正常结束（未收到 done），答案可能不完整")
        return {"answer": "", "contexts": [], "ok": False}
    ok = bool(answer.strip())   # 空答案视为失败
    return {"answer": answer, "contexts": contexts, "tools_called": tools, "retrieval_meta": meta, "ok": ok}


def run_with_retry(question: str, idx: int) -> dict:
    """带指数退避重试的单题执行。"""
    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            wait = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
            print(f"    ⏳ 第 {attempt}/{MAX_RETRIES} 次重试，等待 {wait}s ...")
            time.sleep(wait)

        t0  = time.time()
        res = call_stream(question, idx)
        elapsed = time.time() - t0

        if res["ok"]:
            print(f"    ✅ 成功  answer={len(res['answer'])}字  "
                  f"contexts={len(res['contexts'])}块  工具={res['tools_called'] or '无'}  耗时={elapsed:.1f}s")
            return res
        else:
            status = "空答案" if not res["answer"] else "失败"
            print(f"    ❌ {status}（尝试 {attempt+1}/{MAX_RETRIES+1}）耗时={elapsed:.1f}s")

    print(f"    💀 全部 {MAX_RETRIES+1} 次尝试失败，记录空答案继续")
    return {"answer": "", "contexts": [], "ok": False}


def main():
    # ── 读取问题列表 ──────────────────────────────────────────────────────────
    if FLYWHEEL:
        with open(SRC_PATH, encoding="utf-8") as f:
            questions = json.load(f)
        # 飞轮模式：问题、GT、元数据字段全部来自 batch_N 文件
        gt_by_id   = {q["id"]: q for q in questions}
        mode_label = f"飞轮模式（from-batch={args.from_batch}）"
    else:
        with open(Q_PATH,  encoding="utf-8") as f:
            questions = json.load(f)
        with open(GT_PATH, encoding="utf-8") as f:
            ground_truths = json.load(f)
        gt_by_id   = {item["id"]: item for item in ground_truths}
        mode_label = "基线模式"

    total = len(questions)

    # ── 断点续传：加载已有结果 ────────────────────────────────────────────────
    if OUT_PATH.exists() and not args.rerun:
        with open(OUT_PATH, encoding="utf-8") as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results if r.get("answer", "").strip()}
        print(f"📂 已有结果 {len(results)} 条，有效 {len(done_ids)} 条，跳过这些 ID")
    else:
        results  = []
        done_ids = set()
        if args.rerun and OUT_PATH.exists():
            print("🔄 --rerun：忽略已有结果，全部重跑")

    need_run = [q for q in questions if q["id"] not in done_ids]

    # --ids 白名单过滤
    if args.ids.strip():
        id_whitelist = {int(x.strip()) for x in args.ids.split(",") if x.strip()}
        need_run = [q for q in need_run if q["id"] in id_whitelist]
        print(f"🎯 --ids 过滤：仅运行 {sorted(id_whitelist)} 共 {len(need_run)} 条")

    print(f"\n🚀 评估配置")
    print(f"   模式       : {mode_label}")
    print(f"   API        : {STREAM_URL}")
    print(f"   用户名     : {USER_ID}")
    print(f"   超时       : connect={CONNECT_TIMEOUT}s / read={READ_TIMEOUT}s")
    print(f"   重试       : 最多 {MAX_RETRIES} 次，退避 {RETRY_DELAYS}s")
    print(f"   请求间隔   : {args.delay}s")
    print(f"   总题数     : {total}  本次运行: {len(need_run)} 条")
    print(f"   输出文件   : {OUT_PATH.name}")
    print()

    # ── 主循环 ────────────────────────────────────────────────────────────────
    for q_idx, q_item in enumerate(need_run):
        idx      = q_item["id"]
        question = q_item["question"]
        topic    = q_item.get("topic", "")
        gt_item  = gt_by_id.get(idx, {})

        print(f"[{idx+1:03d}/{total}] ({q_idx+1}/{len(need_run)}) {topic}")
        print(f"  Q: {question[:90]}{'...' if len(question) > 90 else ''}")

        res = run_with_retry(question, idx)

        if FLYWHEEL:
            # 飞轮模式：继承所有元数据字段，只替换 answer 和 contexts
            record = {
                "id":            idx,
                "chunk_id":      q_item.get("chunk_id", ""),
                "topic":         topic,
                "question_type": q_item.get("question_type", ""),
                "question":      question,
                "ground_truth":  gt_item.get("ground_truth", ""),
                "answer":        res["answer"],
                "contexts":      res["contexts"],
                "tools_called":  res.get("tools_called", []),
                "retrieval_meta": res.get("retrieval_meta", {}),
            }
        else:
            record = {
                "id":           idx,
                "topic":        topic,
                "question":     question,
                "answer":       res["answer"],
                "contexts":     res["contexts"],
                "ground_truth": gt_item.get("ground_truth", ""),
            }

        results = [r for r in results if r["id"] != idx]
        results.append(record)

        results_sorted = sorted(results, key=lambda r: r["id"])
        tmp_path = OUT_PATH.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(results_sorted, f, ensure_ascii=False, indent=2)
        tmp_path.replace(OUT_PATH)   # 原子替换，中断时不会留下半写的结果文件

        if q_idx < len(need_run) - 1:
            time.sleep(args.delay)
        print()

    # ── 最终统计 ──────────────────────────────────────────────────────────────
    all_results    = sorted(results, key=lambda r: r["id"])
    empty_answers  = [r for r in all_results if not r.get("answer", "").strip()]
    empty_contexts = [r for r in all_results if not r.get("contexts")]
    valid_count    = len(all_results) - len(empty_answers)

    print("=" * 56)
    print("  评估采集完成")
    print("=" * 56)
    print(f"  总记录数      : {len(all_results)}")
    print(f"  有效答案      : {valid_count}")
    print(f"  空答案（失败）: {len(empty_answers)}")
    print(f"  无 contexts   : {len(empty_contexts)}")
    print(f"  输出文件      : {OUT_PATH}")

    if empty_answers:
        failed_ids = [r["id"] for r in empty_answers]
        print(f"\n  ⚠️  失败 ID: {failed_ids}")
        retry_cmd = (
            f"python scripts/run_eval_baseline.py --from-batch {args.from_batch} --user-id {USER_ID}"
            if FLYWHEEL else f"python scripts/run_eval_baseline.py --user-id {USER_ID}"
        )
        print(f"  补跑命令（自动跳过已有有效答案）：")
        print(f"  docker compose exec app {retry_cmd}")
    else:
        next_step = (
            f"python scripts/dedup_ragas_input.py --batch {args.from_batch + 1}"
            if FLYWHEEL else "python scripts/run_ragas_offline.py --batch 0"
        )
        print(f"\n  全部成功，下一步：{next_step}")


if __name__ == "__main__":
    main()
