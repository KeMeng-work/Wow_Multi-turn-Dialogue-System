"""
离线 RAGAS 评估脚本（健壮版）

特性：
  - 逐条评估 + 断点续传（每条结果立即持久化，崩溃后自动续跑）
  - --test 模式：仅评估前 3 条，验证 API 连通性与指标返回
  - --model：覆盖 judge LLM（默认 qwen-plus，qwen-max 也可）
  - --batch 0：读取 eval_results_100.json（原始基准模式）
  - --batch N：读取 ragas_input_batch_N_deduped.json（飞轮模式，需先跑 dedup_ragas_input.py）
  - --ar-prompt：AnswerRelevancy 提示词版本（计算逻辑仍为 ragas 原生，只替换提示词）
        zh_v1  = R2 使用的中文指令（"模糊、回避或不确定"，会把"答出 + 说明资料缺失"误判为回避）
        zh_v2  = 默认。回避判定对齐 ragas 原版语义（只有未给出任何实质信息才算回避）+ 中文示例
        adapt  = ragas 官方 adapt_prompts(language="chinese") 自动翻译原版提示词
  - --metrics：只计算部分指标（如 answer_relevancy），结果写入单独文件，不覆盖全量结果
  - --retry-none：只重评有指标为 None 的记录
  - 熔断：连续 3 条全部指标为 None（API 故障/额度耗尽）立即停止；全 None 记录不写入断点，下次自动重试

运行示例：
  # 先测试连通性（只跑 3 条）
  python scripts/run_ragas_offline.py --batch 1 --test

  # 全量评估
  python scripts/run_ragas_offline.py --batch 1

  # 使用 qwen-max（更贵）
  python scripts/run_ragas_offline.py --batch 1 --model qwen-max

  # 原始 100 条基准
  python scripts/run_ragas_offline.py --batch 0
"""

import sys
from types import ModuleType

# vertexai 空壳必须在所有其他 import 之前注入
# ragas 0.4.x 硬编码 import langchain_community.chat_models.vertexai
# 该模块在 langchain-community >=0.3 中已被移除，提前注入空壳让 ragas 顺利加载
class _AnyAttr(ModuleType):
    def __getattr__(self, name):
        return None

for _mod in (
    "langchain_community.chat_models.vertexai",
    "langchain_community.chat_models",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = _AnyAttr(_mod)

# ragas 相关 import 紧跟 shim 之后
from langchain_openai import ChatOpenAI
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, AnswerCorrectness  # noqa: deprecated in 0.5, still works in 0.4.x
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.metrics._answer_relevance import ResponseRelevanceInput, ResponseRelevanceOutput
import asyncio

import json
import math
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# override=True 确保 .env 中的值覆盖系统环境变量，防止旧 key 干扰
load_dotenv(ROOT / ".env", override=True)

from config.config_path import RAGAS_CONFIG, EMBED_CONFIG
from src.core.embedder.onnx_embedder import ONNXEmbedder

METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]
CIRCUIT_BREAK_AFTER = 3   # 连续全 None 条数达到此值即停止，避免 API 故障时白跑

# ── AnswerRelevancy 提示词版本（只替换提示词，打分公式仍为 ragas 原生）──────────────
AR_PROMPT_ZH_V1 = (
    "根据给定的答案生成一个问题，并判断答案是否具有承诺性。"
    "如果答案模糊、回避或不确定（例如“我不知道”、“不确定”），noncommittal 返回 1，否则返回 0。"
    "生成的问题必须使用中文。"
)
AR_PROMPT_ZH_V2 = (
    "根据给定的答案生成一个问题，并判断答案是否回避了问题。"
    "只有当答案没有给出问题所问的任何实质信息时（例如只说“我不知道”“无法回答”“没有找到相关资料”），noncommittal 返回 1；"
    "答案给出了实质信息，同时说明某些方面资料未记载，仍视为正面回答，noncommittal 返回 0。"
    "生成的问题必须使用中文。"
)
# 中文示例（不取自评测集，避免泄漏）：正面回答 / 正面回答 + 说明资料缺失 / 真正回避
AR_EXAMPLES_ZH_V2 = [
    (ResponseRelevanceInput(response="珠穆朗玛峰海拔约 8849 米，位于中国与尼泊尔边境。"),
     ResponseRelevanceOutput(question="珠穆朗玛峰的海拔是多少？", noncommittal=0)),
    (ResponseRelevanceInput(response="珠穆朗玛峰海拔约 8849 米。知识库未记载首次登顶的具体日期。"),
     ResponseRelevanceOutput(question="珠穆朗玛峰的海拔是多少？", noncommittal=0)),
    (ResponseRelevanceInput(response="我不了解 2030 年发布的那款手机有什么突破性功能，相关资料没有记载。"),
     ResponseRelevanceOutput(question="2030 年发布的那款手机有什么突破性功能？", noncommittal=1)),
]

# ── 参数解析 ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="离线 RAGAS 评估（健壮版）")
    parser.add_argument("--batch", type=int, default=0,
                        help="0=100条基准, N=飞轮 batch N（需先去重）")
    parser.add_argument("--model", default="qwen-plus",
                        help="judge LLM 模型名，默认 qwen-plus")
    parser.add_argument("--test", action="store_true",
                        help="只评估前 3 条，验证连通性")
    parser.add_argument("--ids", default="",
                        help="只跑指定 ID，逗号分隔，如 --ids 6,8,10,12,14")
    parser.add_argument("--ar-prompt", default="zh_v2", choices=["zh_v1", "zh_v2", "adapt"],
                        help="AnswerRelevancy 提示词版本，默认 zh_v2")
    parser.add_argument("--metrics", default="",
                        help="只计算指定指标（逗号分隔，如 answer_relevancy），结果写入单独文件")
    parser.add_argument("--retry-none", action="store_true",
                        help="只重评有指标为 None 的记录，结果覆盖原记录")
    return parser.parse_args()


# ── 路径解析 ──────────────────────────────────────────────────────────────────

def resolve_paths(args):
    if args.batch == 0:
        in_path  = ROOT / "data/evaluation/eval_results_100.json"
        out_path = ROOT / "data/evaluation/ragas_baseline_report.json"
    else:
        deduped = ROOT / f"data/evaluation/ragas_input_batch_{args.batch}_deduped.json"
        raw     = ROOT / f"data/evaluation/ragas_input_batch_{args.batch}.json"
        if not deduped.exists() and not raw.exists():
            print(f"❌ 找不到输入文件（{deduped.name} 或 {raw.name}）")
            print(f"   请先运行：python scripts/dedup_ragas_input.py --batch {args.batch}")
            sys.exit(1)
        if not deduped.exists():
            print(f"⚠️  未找到去重版本 {deduped.name}，回退到原始文件 {raw.name}")
            print(f"   建议先运行：python scripts/dedup_ragas_input.py --batch {args.batch}")
            in_path = raw
        else:
            in_path = deduped
        suffix   = "_test" if args.test else ""
        if args.metrics.strip():
            # 只算部分指标：单独文件，文件名带提示词版本，不覆盖全量结果
            short = "_".join(m.strip() for m in args.metrics.split(",") if m.strip())
            suffix += f"_{short}_{args.ar_prompt}"
        out_path = ROOT / f"data/evaluation/ragas_results_batch_{args.batch}{suffix}.json"
    return in_path, out_path


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_valid_samples(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    valid   = [r for r in data if r.get("answer") and r.get("contexts") and r.get("ground_truth")]
    skipped = len(data) - len(valid)
    print(f"📂 加载 {len(data)} 条，有效: {len(valid)} 条，跳过（缺字段）: {skipped} 条")
    return valid


# ── 组件初始化 ─────────────────────────────────────────────────────────────────

def configure_ar_prompt(ar_metric, version: str, ragas_llm):
    """替换 AnswerRelevancy 的提示词（计算逻辑不变）"""
    prompt = ar_metric.question_generation
    if version == "zh_v1":
        prompt.instruction = AR_PROMPT_ZH_V1
    elif version == "zh_v2":
        prompt.instruction = AR_PROMPT_ZH_V2
        prompt.examples = list(AR_EXAMPLES_ZH_V2)
    elif version == "adapt":
        adapted = asyncio.run(ar_metric.adapt_prompts(language="chinese", llm=ragas_llm, adapt_instruction=True))
        ar_metric.set_prompts(**adapted)
    else:
        raise ValueError(f"未知的 ar-prompt 版本: {version}")


def build_components(model_name: str, ar_prompt: str = "zh_v2", only: list = None):
    judge_llm = ChatOpenAI(
        model=model_name,
        api_key=RAGAS_CONFIG["api_key"],
        base_url=RAGAS_CONFIG["base_url"],
        temperature=0,
        max_tokens=2048,
        timeout=180,
        max_retries=1,   # 主要重试交给 RunConfig，避免两层重试叠加放大 token 消耗
    )
    embedder = ONNXEmbedder(
        model_path=EMBED_CONFIG["model_path"],
        onnx_file=EMBED_CONFIG["onnx_file"],
        max_length=EMBED_CONFIG["max_length"],
        batch_size=EMBED_CONFIG["batch_size"],
    )
    ragas_llm = LangchainLLMWrapper(judge_llm)
    ragas_emb = LangchainEmbeddingsWrapper(embedder)
    # AnswerRelevancy 默认用英文 prompt 生成问题，导致中文场景下跨语言相似度极低（AR≈0）
    # 覆盖 question_generation 指令，强制生成中文问题，恢复真实 AR 值
    ar_metric = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb)
    configure_ar_prompt(ar_metric, ar_prompt, ragas_llm)
    metrics = [
        Faithfulness(llm=ragas_llm),
        ar_metric,
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
        AnswerCorrectness(llm=ragas_llm, embeddings=ragas_emb),
    ]
    if only:
        metrics = [m for m in metrics if m.name in only]
        if not metrics:
            raise ValueError(f"--metrics 未匹配任何指标: {only}")
    run_cfg = RunConfig(
        timeout=180,     # 单次 LLM 调用最长等 3 分钟，qwen-plus 偶尔排队
        max_retries=3,   # 3 次重试覆盖瞬时限速，失败后才记 None
        max_wait=60,     # 重试间隔最长 60s，让 API 有时间恢复
        max_workers=3,   # 允许 3 个 metric 并发，约 42s/条，风险低
    )
    return metrics, run_cfg


# ── 断点续传：加载已完成的 id 集合 ────────────────────────────────────────────

def load_checkpoint(out_path: Path) -> list[dict]:
    if not out_path.exists():
        return []
    with open(out_path, encoding="utf-8") as f:
        data = json.load(f)
    # 兼容两种格式：list（逐条保存）或 {"details": [...]}（最终报告格式）
    if isinstance(data, list):
        return data
    return data.get("details", [])


# ── 保存单条结果（立即写入，防崩溃丢失）────────────────────────────────────────

def save_checkpoint(details: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)


# ── 保存最终报告（带 summary）──────────────────────────────────────────────────

def save_final_report(details: list[dict], out_path: Path, model_name: str, ar_prompt: str = "zh_v2"):
    summary = {}
    for col in METRIC_COLS:
        vals = [r[col] for r in details if r.get(col) is not None]
        summary[col] = round(sum(vals) / len(vals), 4) if vals else None
        summary[f"{col}_n"] = len(vals)   # 各指标实际有效条数（None 不计入平均）
    report = {
        "judge_model": model_name,
        "eval_method": "ragas_framework",
        "ar_prompt_version": ar_prompt,
        "total":       len(details),
        "summary":     summary,
        "details":     details,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return summary


# ── 单条评估（不抛异常，失败返回 None 分数）──────────────────────────────────────

def evaluate_one(rec: dict, metrics, run_cfg, cols=METRIC_COLS) -> dict:
    sample  = SingleTurnSample(
        user_input=rec["question"],
        response=rec["answer"],
        retrieved_contexts=rec["contexts"],
        reference=rec["ground_truth"],
    )
    dataset = EvaluationDataset(samples=[sample])

    row = {}
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            run_config=run_cfg,
            batch_size=1,
            raise_exceptions=False,
        )
        df  = result.to_pandas()
        row = df.iloc[0].to_dict() if len(df) > 0 else {}
    except Exception as e:
        print(f"  ⚠️  evaluate() 异常: {type(e).__name__}: {e}")

    scored = {
        "id":            rec.get("id"),
        "chunk_id":      rec.get("chunk_id", ""),
        "topic":         rec.get("topic", ""),
        "question_type": rec.get("question_type", ""),
    }
    for col in cols:
        v = row.get(col)
        scored[col] = None if (v is None or (isinstance(v, float) and math.isnan(v))) else round(float(v), 4)
    return scored


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    in_path, out_path = resolve_paths(args)

    print(f"\n{'='*56}")
    print(f"  RAGAS 离线评估")
    print(f"{'='*56}")
    print(f"  输入文件  : {in_path.name}")
    print(f"  输出文件  : {out_path.name}")
    print(f"  Judge LLM : {args.model}")
    print(f"  AR 提示词 : {args.ar_prompt}")
    if args.metrics.strip():
        print(f"  指标子集  : {args.metrics}")
    print(f"  模式      : {'测试（3条）' if args.test else '全量'}")
    print(f"{'='*56}\n")

    samples = load_valid_samples(in_path)
    if not samples:
        print("❌ 无有效样本，退出")
        return

    only = [m.strip() for m in args.metrics.split(",") if m.strip()] or None
    cols = only or METRIC_COLS

    # 断点续传：找出已完成的 id（全 None 记录视为未完成，自动重试）
    done_records = [r for r in load_checkpoint(out_path)
                    if not all(r.get(c) is None for c in cols)]
    if args.retry_none:
        retry_ids = {r["id"] for r in done_records if any(r.get(c) is None for c in cols)}
        done_records = [r for r in done_records if r["id"] not in retry_ids]
        print(f"🔁 --retry-none：重评 {len(retry_ids)} 条含 None 的记录\n")
    done_ids     = {r["id"] for r in done_records}
    pending      = [s for s in samples if s.get("id") not in done_ids]

    if done_ids:
        print(f"♻️  断点续传：已完成 {len(done_ids)} 条，剩余 {len(pending)} 条\n")

    if args.test:
        pending = pending[:3]
        print(f"🔬 测试模式：仅评估 {len(pending)} 条\n")

    if args.ids.strip():
        id_whitelist = {int(x.strip()) for x in args.ids.split(",") if x.strip()}
        pending = [s for s in pending if s.get("id") in id_whitelist]
        print(f"🎯 --ids 过滤：仅运行 {sorted(id_whitelist)}，共 {len(pending)} 条\n")

    if not pending:
        print("✅ 所有记录已完成，无需重跑")
        save_final_report(done_records, out_path, args.model, args.ar_prompt)
        return

    print(f"🔧 初始化 Judge LLM ({args.model}) + ONNX embedder ...")
    metrics, run_cfg = build_components(args.model, args.ar_prompt, only)
    print(f"✅ 组件就绪，开始逐条评估（共 {len(pending)} 条）\n")

    details    = list(done_records)
    nan_count  = 0
    consecutive_none = 0
    t_start    = time.time()

    for idx, rec in enumerate(pending):
        t0    = time.time()
        label = f"[{idx+1:03d}/{len(pending)}] id={rec.get('id')} {rec.get('topic','')[:28]}"
        print(label)

        scored = evaluate_one(rec, metrics, run_cfg, cols)
        elapsed = time.time() - t0

        # 打印本条结果
        f_val  = scored.get("faithfulness")
        ar_val = scored.get("answer_relevancy")
        cp_val = scored.get("context_precision")
        cr_val = scored.get("context_recall")
        ac_val = scored.get("answer_correctness")

        all_none = all(scored.get(c) is None for c in cols)
        if all_none:
            nan_count += 1
            consecutive_none += 1
            print(f"  ⚠️  全部指标为 None（API 失败？）耗时 {elapsed:.1f}s —— 不写入断点，下次自动重试")
            if consecutive_none >= CIRCUIT_BREAK_AFTER:
                print(f"\n🛑 熔断：连续 {consecutive_none} 条全部指标为 None，停止评估（检查 API Key / 额度 / 网络后重跑，已完成的记录会自动跳过）")
                break
        else:
            consecutive_none = 0
            print(f"  F={f_val}  AR={ar_val}  CP={cp_val}  CR={cr_val}  AC={ac_val}  ({elapsed:.1f}s)")
            # 立即持久化（断点保护）；--retry-none 时覆盖同 id 旧记录
            details = [r for r in details if r.get("id") != scored["id"]]
            details.append(scored)
            save_checkpoint(details, out_path)

        # 记录间主动冷却，避免短时间内 16+ 个调用触发 RPM 限速
        # 测试模式缩短间隔，全量模式给 API 足够喘息时间
        cooldown = 3 if args.test else 8
        if idx < len(pending) - 1:
            time.sleep(cooldown)

        # 测试模式：额外打印问题内容
        if args.test:
            print(f"  Q: {rec['question'][:80]}")
            print(f"  A: {rec['answer'][:80]}")
            print(f"  ctx 数: {len(rec.get('contexts', []))}")
        print()

    # ── 最终报告 ──────────────────────────────────────────────────────────────
    summary = save_final_report(details, out_path, args.model, args.ar_prompt)
    elapsed_total = time.time() - t_start

    print("=" * 56)
    print(f"  评估完成  耗时 {elapsed_total/60:.1f} 分钟")
    print("=" * 56)
    print(f"  样本总数          : {len(details)}")
    print(f"  全 None 记录      : {nan_count} 条（API 失败）")
    for col in cols:
        print(f"  {col:<20}: {summary[col]}  (n={summary[col + '_n']})")
    print("=" * 56)
    print(f"\n📄 报告：{out_path}")

    if args.test:
        print("\n🔬 测试通过后，运行全量评估：")
        print(f"   python scripts/run_ragas_offline.py --batch {args.batch} --model {args.model}")


if __name__ == "__main__":
    main()
