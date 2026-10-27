"""
Round N Badcase 提取脚本

从 RAGAS 评估结果中提取低分记录，合并原始 QA 内容，
输出结构化 badcase 文件供人工诊断和针对性优化使用。

判定标准（可通过参数调整）：
  - Faithfulness   < F_THRESHOLD  （严重幻觉）
  - Answer Correct < AC_THRESHOLD （答案偏差过大）
  - Faithfulness   is None        （评估失败，潜在边缘用例）

输出字段（每条 badcase）：
  id / chunk_id / topic / question_type
  question / ground_truth / answer / contexts（去重后）
  faithfulness / answer_relevancy / context_precision / context_recall / answer_correctness
  badcase_reason   — 命中了哪条判定标准（可多个）
  diagnosis_hint   — 基于分数模式的自动诊断提示

用法：
  python scripts/extract_badcases.py --batch 1
  python scripts/extract_badcases.py --batch 1 --f-threshold 0.5 --ac-threshold 0.5
"""

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent


# ── 诊断提示生成 ──────────────────────────────────────────────────────────────

def build_diagnosis_hint(rec: dict) -> str:
    """
    根据分数组合模式输出诊断方向，供人工审核时快速定位根因。
    模式来自 Round 1 分析：
      - F 极低 + AC 极低：答案完全依赖参数记忆，检索文档几乎未被使用
      - F 极低 + AC 高：答案碰巧正确但未引用检索文档（参数知识覆盖了 GT）
      - F 中低 + AC 低：部分幻觉 + 答案与 GT 差距大
      - F=None：LLM 评估调用失败，需人工确认是否为边缘用例
      - AR 低（< 0.5）：答案偏题，改写可能引入了语义漂移
      - CP/CR 低（< 0.5）：检索召回的文档与问题相关度差，是上游问题
    """
    f   = rec.get("faithfulness")
    ac  = rec.get("answer_correctness")
    ar  = rec.get("answer_relevancy")
    cp  = rec.get("context_precision")
    cr  = rec.get("context_recall")
    qt  = rec.get("question_type", "")

    hints = []

    if f is None:
        hints.append("F=None：RAGAS 声明提取失败，可能答案过短或含特殊格式，需人工核查")
    elif f < 0.1:
        if ac is not None and ac > 0.7:
            hints.append("F极低+AC高：答案内容碰巧覆盖GT但完全未引用检索文档，模型用参数记忆作答")
        else:
            hints.append("F极低：答案声明几乎全部来自参数记忆而非检索文档，典型幻觉")
    elif f < 0.3:
        hints.append("F低：大量声明无文档依据，reasoning题多见；检查检索质量或加强接地约束")

    if ac is None:
        hints.append("AC=None：答案过长或调用超时，需人工确认答案质量")
    elif ac < 0.3:
        hints.append("AC极低：答案与GT差异极大，可能答非所问或严重幻觉")
    elif ac < 0.4:
        hints.append("AC低：答案与GT声明重叠少，检查GT覆盖范围或答案是否偏题")

    if ar is not None and ar < 0.5:
        hints.append("AR低：答案偏离原始问题，疑似查询改写扩展了问题范围或答案发散")

    if cp is not None and cp < 0.5:
        hints.append("CP低：检索文档中大量内容与问题无关，检索召回质量差")

    if cr is not None and cr < 0.5:
        hints.append("CR低：GT核心声明未被检索文档覆盖，知识库可能缺少对应chunk")

    if qt == "reasoning":
        hints.append("Reasoning题：多步推断本身易引发幻觉，优先考虑是否需要分解子问题")

    return "；".join(hints) if hints else "分数组合暂无明确模式，需人工审核"


# ── Badcase 判定 ──────────────────────────────────────────────────────────────

def get_badcase_reasons(rec: dict, f_threshold: float, ac_threshold: float) -> list[str]:
    reasons = []
    f  = rec.get("faithfulness")
    ac = rec.get("answer_correctness")
    if f is None:
        reasons.append("faithfulness=None")
    elif f < f_threshold:
        reasons.append(f"faithfulness={f:.3f}<{f_threshold}")
    if ac is None:
        reasons.append("answer_correctness=None")
    elif ac < ac_threshold:
        reasons.append(f"answer_correctness={ac:.3f}<{ac_threshold}")
    return reasons


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="提取 RAGAS badcase，合并 QA 内容，输出诊断文件")
    parser.add_argument("--batch",        type=int,   default=1,   help="飞轮批次编号")
    parser.add_argument("--f-threshold",  type=float, default=0.3, help="Faithfulness 低分阈值（默认 0.3）")
    parser.add_argument("--ac-threshold", type=float, default=0.4, help="Answer Correctness 低分阈值（默认 0.4）")
    args = parser.parse_args()

    results_path = ROOT / f"data/evaluation/ragas_results_batch_{args.batch}.json"
    inputs_path  = ROOT / f"data/evaluation/ragas_input_batch_{args.batch}_deduped.json"
    out_path     = ROOT / f"data/evaluation/badcases_batch_{args.batch}.json"

    if not results_path.exists():
        print(f"❌ 找不到评估结果文件：{results_path.name}")
        sys.exit(1)
    if not inputs_path.exists():
        print(f"❌ 找不到输入文件：{inputs_path.name}（需先运行 dedup_ragas_input.py）")
        sys.exit(1)

    with open(results_path, encoding="utf-8") as f:
        results_data = json.load(f)
    with open(inputs_path, encoding="utf-8") as f:
        inputs_data = json.load(f)

    # 以 id 为 key 建索引
    inputs_by_id = {r["id"]: r for r in inputs_data}
    details      = results_data.get("details", results_data if isinstance(results_data, list) else [])

    badcases = []
    for rec in details:
        reasons = get_badcase_reasons(rec, args.f_threshold, args.ac_threshold)
        if not reasons:
            continue

        rec_id  = rec.get("id")
        qa_rec  = inputs_by_id.get(rec_id, {})

        badcase = {
            # 索引字段
            "id":            rec_id,
            "chunk_id":      rec.get("chunk_id", ""),
            "topic":         rec.get("topic", ""),
            "question_type": rec.get("question_type", ""),
            # 原始 QA 内容
            "question":      qa_rec.get("question", ""),
            "ground_truth":  qa_rec.get("ground_truth", ""),
            "answer":        qa_rec.get("answer", ""),
            "contexts":      qa_rec.get("contexts", []),
            # RAGAS 分数
            "faithfulness":        rec.get("faithfulness"),
            "answer_relevancy":    rec.get("answer_relevancy"),
            "context_precision":   rec.get("context_precision"),
            "context_recall":      rec.get("context_recall"),
            "answer_correctness":  rec.get("answer_correctness"),
            # 诊断
            "badcase_reason":   reasons,
            "diagnosis_hint":   build_diagnosis_hint(rec),
        }
        badcases.append(badcase)

    # 按 Faithfulness 升序排列（最严重在前）
    badcases.sort(key=lambda x: (x["faithfulness"] is None, x["faithfulness"] or 0))

    # 统计
    f_none  = sum(1 for b in badcases if b["faithfulness"] is None)
    f_low   = sum(1 for b in badcases if b["faithfulness"] is not None and b["faithfulness"] < args.f_threshold)
    ac_none = sum(1 for b in badcases if b["answer_correctness"] is None and "answer_correctness=None" in str(b["badcase_reason"]))
    ac_low  = sum(1 for b in badcases if b["answer_correctness"] is not None and b["answer_correctness"] < args.ac_threshold)
    reasoning_cnt = sum(1 for b in badcases if b["question_type"] == "reasoning")
    simple_cnt    = sum(1 for b in badcases if b["question_type"] == "simple")

    report = {
        "batch":         args.batch,
        "source_results": results_path.name,
        "thresholds": {
            "faithfulness":        args.f_threshold,
            "answer_correctness":  args.ac_threshold,
        },
        "summary": {
            "total_evaluated":    len(details),
            "total_badcases":     len(badcases),
            "badcase_rate":       round(len(badcases) / len(details), 4) if details else 0,
            "faithfulness_none":  f_none,
            "faithfulness_low":   f_low,
            "ac_none":            ac_none,
            "ac_low":             ac_low,
            "by_question_type":   {"reasoning": reasoning_cnt, "simple": simple_cnt},
        },
        "badcases": badcases,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*52}")
    print(f"  Badcase 提取完成")
    print(f"{'='*52}")
    print(f"  评估总数        : {len(details)} 条")
    print(f"  Badcase 总数    : {len(badcases)} 条（{len(badcases)/len(details)*100:.1f}%）")
    print(f"  F=None          : {f_none} 条")
    print(f"  F<{args.f_threshold}          : {f_low} 条")
    print(f"  AC<{args.ac_threshold}        : {ac_low} 条")
    print(f"  Reasoning 题    : {reasoning_cnt} 条")
    print(f"  Simple 题       : {simple_cnt} 条")
    print(f"{'='*52}")
    print(f"\n📄 输出：{out_path}")
    print(f"\n诊断提示分布（Top 5 根因）：")

    from collections import Counter
    hint_counter: Counter = Counter()
    for b in badcases:
        for hint in b["diagnosis_hint"].split("；"):
            first_word = hint.split("：")[0] if "：" in hint else hint[:8]
            hint_counter[first_word] += 1
    for hint, cnt in hint_counter.most_common(5):
        print(f"  [{cnt:3d}] {hint}")


if __name__ == "__main__":
    main()
