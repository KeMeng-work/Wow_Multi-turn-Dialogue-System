"""
上下文组装 (Context Builder) —— 把会话状态 + 近期原文按预算组装给改写器 / Agent，纯逻辑、无 IO

设计文档：docs/short_term_memory_design.md §7

token 计数为保守估算：中日韩字符按 1 字 1 token、其余按 3 字符 1 token（Qwen 实际约 1.3–1.5 个汉字 / token、
英文约 4 字符 / token，估算偏大，等价于预留余量）。

turns：近期原文列表（时间正序），每项 {idx, user, assistant, refused, rewritten_zh, rewritten_en, topic_id}
"""

import math
import re
from typing import Dict, List, Optional, Tuple

from src.core.memory_short.session_state import (
    CONSTRAINT_TYPE_ZH, SCOPE_ZH, effective_constraints, pending_conflicts,
)

_CJK_RE = re.compile(r"[⺀-鿿가-힯豈-﫿＀-￯]")
_LIST_ITEM_RE = re.compile(r"^\s*(?:\d{1,2}[.、)）]|[-*•·]|（\d{1,2}）|\(\d{1,2}\)|[一二三四五六七八九十]{1,2}、)\s*")
_SENT_END_RE = re.compile(r"[。！？!?]|\.\s")

_WINDOW_FLOOR = 4   # Agent 上下文裁剪第 1 步保留的最少轮次（其中较早的 2 轮在第 2 步压缩）

REFUSED_MARK = "（该轮未能从知识库中找到资料，未作回答）"


def est_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    return cjk + math.ceil((len(text) - cjk) / 3)


def _clip(text: str, max_tokens: int) -> str:
    """按估算 token 截断（保留开头）"""
    if est_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if est_tokens(text[:mid]) <= max_tokens - 1:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + "…"


def _first_sentence(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    m = _SENT_END_RE.search(text)
    s = text[: m.end()] if m else text
    return s if len(s) <= max_chars else s[:max_chars] + "…"


def compress_answer(answer: str, max_tokens: int = 120) -> str:
    """
    回答按结构压缩：保留列表项编号与首句、每段首句，去掉展开说明；不按字数硬截断列表结构。
    "上一轮列了 5 点，第五点展开说说"时，改写器仍能看到第五点讲什么。
    """
    lines = [ln.strip() for ln in (answer or "").splitlines() if ln.strip()]
    parts, items = [], []
    for ln in lines:
        m = _LIST_ITEM_RE.match(ln)
        if m:
            body = ln[m.end():]
            items.append(body)
            parts.append(f"{len(items)}. {_first_sentence(body, 50)}")
        else:
            parts.append(_first_sentence(ln, 80))
    out = " / ".join(parts)
    if est_tokens(out) > max_tokens and items:
        # 超长时先保住全部列表项标题，再截段落（首句对正文取，编号里的"1. "不能当作句末）
        # 每项长度按条数自适应收缩，尽量让全部编号都留下（"第七点展开说说"需要知道第七点是什么）
        for chars in (25, 18, 12, 8):
            out = " / ".join(f"{i}. {_first_sentence(body, chars)}" for i, body in enumerate(items, 1))
            if est_tokens(out) <= max_tokens:
                break
    return _clip(out, max_tokens)


def _answer_text(turn: dict) -> str:
    return REFUSED_MARK if turn.get("refused") else (turn.get("assistant") or "")


def render_constraints(constraints: List[dict]) -> str:
    return "\n".join(f"- {CONSTRAINT_TYPE_ZH.get(c['type'], c['type'])}：{c['value']}（{SCOPE_ZH.get(c['scope'], c['scope'])}）"
                     for c in constraints)


def render_conflicts(conflicts: List[dict]) -> str:
    return "\n".join(f"- {CONSTRAINT_TYPE_ZH.get(c['type'], c['type'])}：原为「{c['old_value']}」，"
                     f"用户似乎想改为「{c['value']}」（{SCOPE_ZH.get(c['scope'], c['scope'])}）" for c in conflicts)


def _render_entities(state: dict, max_tokens: int) -> str:
    groups: Dict[str, List[str]] = {}
    for e in sorted(state.get("entities", []), key=lambda e: e["last_turn"], reverse=True):
        alias = f"（又称 {'、'.join(e['aliases'][:2])}）" if e.get("aliases") else ""
        groups.setdefault(e.get("type") or "other", []).append(f"{e['name']}{alias}[第{e['last_turn']}轮]")
    lines, used = [], 0
    for etype, names in groups.items():
        line = f"- {etype}：" + "；".join(names)
        if used + est_tokens(line) > max_tokens:
            break
        lines.append(line)
        used += est_tokens(line)
    return "\n".join(lines)


def _render_facts(state: dict, max_tokens: int) -> Tuple[str, int]:
    lines, used = [], 0
    for f in sorted(state.get("facts", []), key=lambda f: f["last_turn"], reverse=True):
        line = f"- ({f['s']}, {f['r']}, {f['o']}) [第{','.join(map(str, f['turns'][-2:]))}轮]"
        if used + est_tokens(line) > max_tokens:
            break
        lines.append(line)
        used += est_tokens(line)
    return "\n".join(lines), len(lines)


def _render_outlines(state: dict, exclude_turns: set, max_tokens: int) -> str:
    lines, used = [], 0
    for o in sorted(state.get("outlines", []), key=lambda o: o["turn"], reverse=True):
        if o["turn"] in exclude_turns:
            continue
        line = f"- 第{o['turn']}轮回答：" + " | ".join(o["items"])
        if used + est_tokens(line) > max_tokens:
            break
        lines.append(line)
        used += est_tokens(line)
    return "\n".join(lines)


def _render_topics(state: dict) -> str:
    cur = state.get("current_topic")
    rows = []
    for t in sorted(state.get("topics", []), key=lambda t: t["last_turn"], reverse=True)[:8]:
        mark = "（当前）" if t["id"] == cur else ""
        rows.append(f"- {t['id']}：{t['label']}{mark}")
    return "\n".join(rows)


# ==================== 改写器上下文 ====================

def build_rewriter_context(state: dict, turns: List[dict], cfg: dict) -> Tuple[str, dict]:
    """
    改写器的记忆部分（不含系统提示词与当前问题）。分段各自有上限，总量 ≈ rewriter_budget：
    会话级约束 → 待确认的要求变更 → 话题列表 → 实体表 → 事实三元组 → 早期回答提纲 → 近期轮次 → 当前话题的话题级约束
    """
    eff = effective_constraints(state)
    session_c = [c for c in eff if c["scope"] == "session"]
    topic_c = [c for c in eff if c["scope"] == "topic"]

    recent_lines, used, shown = [], 0, set()
    for t in reversed(turns):
        rewritten = t.get("rewritten_zh") or t.get("rewritten_en") or ""
        block = (f"[第{t['idx']}轮]\n用户：{t.get('user', '')}\n"
                 f"助手（摘要）：{compress_answer(_answer_text(t), 120)}"
                 + (f"\n该轮问题的完整表述：{rewritten}" if rewritten else ""))
        cost = est_tokens(block)
        if recent_lines and used + cost > cfg["rewriter_recent_budget"]:
            break
        recent_lines.insert(0, _clip(block, cfg["rewriter_recent_budget"]))
        used += cost
        shown.add(t["idx"])

    facts, _ = _render_facts(state, 150)
    sections = [
        ("会话级约束", render_constraints(session_c)),
        ("待确认的要求变更", render_conflicts(pending_conflicts(state))),
        ("话题列表", _render_topics(state)),
        ("实体表（按类型，越靠前越近提到）", _render_entities(state, 150)),
        ("已确认的事实", facts),
        ("更早回答的提纲", _render_outlines(state, shown, 150)),
        ("近期对话", "\n".join(recent_lines)),
        ("当前话题的约束", render_constraints(topic_c)),
    ]
    text = "\n".join(f"## {name}\n{body}" for name, body in sections if body) or "无对话历史。"
    return text, {"tokens": est_tokens(text), "recent_turns": len(recent_lines),
                  "recent_tokens": used, "constraints": len(eff)}


# ==================== Agent 上下文 ====================

def build_agent_context(state: dict, turns: List[dict], cfg: dict,
                        turn_constraints: Optional[List[dict]] = None) -> dict:
    """
    返回 {"memory": 系统提示词中的会话记忆段, "history": [(user, assistant)], "question_prefix": 当前问题前的约束提示,
          "stats": {...}}
    位置：会话级约束、待确认变更、当前话题摘要、事实 → 系统提示词；近期原文 → 消息历史；话题级 / 本问约束 → 紧贴当前问题。
    其他话题不注入 Agent（跨话题指代由改写器借话题列表解决，改写后的问题已独立完整）。
    超预算裁剪顺序（当前问题与生效约束永不裁剪）：
      1. 移出最早的一整轮原文（窗口保留至少 _WINDOW_FLOOR 轮，给第 2 步留出可压缩的较早轮次）
      2. 较早的助手回答改为结构化压缩（最近 2 轮保持原文）
      3. 删除最久未提及的事实
      兜底：继续移出较早轮次至只剩最近 2 轮 → 压缩最近 2 轮 → 移出最近轮次
    """
    budget = cfg["agent_budget"]
    eff = effective_constraints(state, turn_constraints)
    session_c = [c for c in eff if c["scope"] == "session"]
    near_c = [c for c in eff if c["scope"] in ("topic", "turn")]
    conflicts = pending_conflicts(state)

    cur = state.get("current_topic")
    summ = (state.get("summaries", {}).get(cur) or {}).get("text", "")
    n_facts = len(state.get("facts", []))

    watermark = state.get("summarized_until", 0)
    recent = [t for t in turns if t["idx"] > watermark]
    history = [[t.get("user", ""), _answer_text(t)] for t in recent]
    compressed = [False] * len(history)

    question_prefix = ""
    if near_c:
        question_prefix = "【本问需遵守的要求】\n" + render_constraints(near_c) + "\n\n"

    def memory_text(fact_limit: int) -> str:
        parts = []
        if session_c:
            parts.append("### 用户要求（必须遵守）\n" + render_constraints(session_c))
        if conflicts:
            parts.append("### 待确认的要求变更（回答前先向用户确认，确认前按原要求执行）\n" + render_conflicts(conflicts))
        if summ:
            parts.append("### 当前话题的早期对话摘要\n" + summ)
        if fact_limit > 0:
            facts, _ = _render_facts(state, 10_000)
            facts = "\n".join(facts.splitlines()[:fact_limit])
            if facts:
                parts.append("### 对话中已确认的事实（来自对话，不是知识库证据）\n" + facts)
        return "\n\n".join(parts)

    fact_limit = n_facts

    def total() -> int:
        return (est_tokens(memory_text(fact_limit)) + est_tokens(question_prefix)
                + sum(est_tokens(u) + est_tokens(a) for u, a in history))

    steps = []
    while total() > budget and len(history) > _WINDOW_FLOOR:         # 1
        history.pop(0)
        compressed.pop(0)
        steps.append("drop_turn")
    for i in range(len(history) - 2):                                # 2
        if total() <= budget:
            break
        if not compressed[i]:
            history[i][1] = compress_answer(history[i][1], 120)
            compressed[i] = True
            steps.append("compress_turn")
    while total() > budget and fact_limit > 0:                       # 3
        fact_limit -= 1
        steps.append("drop_fact")
    while total() > budget and len(history) > 2:                     # 兜底
        history.pop(0)
        compressed.pop(0)
        steps.append("drop_turn")
    for i in range(len(history)):
        if total() <= budget:
            break
        if not compressed[i]:
            history[i][1] = compress_answer(history[i][1], 200)
            compressed[i] = True
            steps.append("compress_recent")
    while total() > budget and history:
        history.pop(0)
        steps.append("drop_recent")

    memory = memory_text(fact_limit)
    return {
        "memory": memory or "暂无",
        "history": [(u, a) for u, a in history],
        "question_prefix": question_prefix,
        "stats": {"tokens": total(), "budget": budget, "history_turns": len(history),
                  "trim_steps": steps, "constraints": len(eff), "conflicts": len(conflicts),
                  "has_summary": bool(summ)},
    }


def unsummarized_tokens(state: dict, turns: List[dict]) -> int:
    """水位线之后的原文 token 数（压缩触发判断）"""
    w = state.get("summarized_until", 0)
    return sum(est_tokens(t.get("user", "")) + est_tokens(_answer_text(t)) for t in turns if t["idx"] > w)
