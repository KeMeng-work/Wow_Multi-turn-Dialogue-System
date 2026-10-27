"""
回答后处理：规则检测器 (Answer Rule Detectors)

对最终答案做零成本的规则检测，结果写入 Langfuse 打分用于监控，不改写答案
（流式输出下答案已发出，改写需要句级缓冲，见后续的生成后核验模块）。

检测项：
    - meta_phrase：描述检索过程或套话开头（"根据现有资料""检索结果显示""多处文档提及"等）
    - english_span：答案中出现英文句子片段（连续 ≥4 个英文单词，括号内的专有名词原文除外）
    - unverified_name：括号内的英文原名在检索文档中找不到（如编造的全名 "Henry Pileggi"）

Author: Ke Meng
Created: 2026-10-04
"""

import re
from typing import Dict, List, Any

_META_PATTERNS = (
    "根据现有资料", "根据现有信息", "检索结果", "多处文档", "文档中提及", "被两次提及",
    "根据检索", "从检索", "资料显示",
)
# 括号内的英文原名：中文全角或半角括号
_PAREN_EN = re.compile(r"[（(]([A-Za-z][A-Za-z0-9 .'&\-]{1,80})[）)]")
# 连续 ≥4 个英文单词
_EN_SPAN = re.compile(r"[A-Za-z]+(?:[ '\-][A-Za-z]+){3,}")


def detect(answer: str, contexts: List[str]) -> Dict[str, Any]:
    """返回各检测项是否命中（0/1）及命中明细，供 Langfuse 打分与离线统计。"""
    answer = answer or ""
    ctx = " ".join(contexts or [])
    ctx_lower = ctx.lower()

    meta_hits = [p for p in _META_PATTERNS if p in answer]

    paren_names = [m.group(1).strip() for m in _PAREN_EN.finditer(answer)]
    unverified = [n for n in paren_names if ctx and n.lower() not in ctx_lower]

    no_paren = _PAREN_EN.sub("", answer)
    en_spans = _EN_SPAN.findall(no_paren)

    return {
        "meta_phrase": int(bool(meta_hits)),
        "english_span": int(bool(en_spans)),
        "unverified_name": int(bool(unverified)),
        "details": {
            "meta_phrase": meta_hits,
            "english_span": en_spans[:5],
            "unverified_name": unverified[:5],
        },
    }
