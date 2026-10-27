"""
短期记忆单元测试（本地可跑，不依赖 Redis / PG / vLLM）

覆盖：session_state（约束生命周期 / 话题 / 实体 / 事实 / 提纲）、context_builder（预算与裁剪）、
      改写器输出解析与降级、压缩器校验与调度、STMService 全流程（FakeMemory 模拟 Lua 脚本语义）
Redis Lua 脚本本身由 scripts/test_stm_redis.py 在服务器上对真实 Redis 测试。

运行：python src/core/memory_short/stm_test.py
"""

import asyncio
import copy
import os
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from config.config_path import STM_CONFIG
from src.core.memory_short import compactor as compactor_mod
from src.core.memory_short import stm_service as stm_mod
from src.core.memory_short.compactor import Compactor
from src.core.memory_short.context_builder import (
    REFUSED_MARK, build_agent_context, build_rewriter_context, compress_answer, est_tokens,
)
from src.core.memory_short.session_state import (
    apply_constraints, apply_entities, apply_facts, apply_rewrite, apply_topic, constraint_hash,
    effective_constraints, merge_turn_effects, new_state, outline_items, pending_conflicts,
)
from src.core.memory_short.stm_service import STMService
from src.core.query_rewrite.query_rewriter import QueryRewriter, _extract_json, _fallback

CFG = dict(STM_CONFIG)


def _rw(**kw) -> dict:
    base = {"en_query": "", "zh_query": "", "intent": "local", "question_type": "simple", "standalone": False,
            "topic": {"shift": "continue", "return_to": "", "label": ""}, "constraints_delta": [], "entities": [],
            "unclassified_directive": False, "ok": True}
    base.update(kw)
    return base


def _set(ctype, value, scope="topic", explicit=False):
    return {"op": "set", "type": ctype, "value": value, "scope": scope, "explicit": explicit}


def _state_with_topic(label="苹果公司") -> dict:
    st = new_state()
    apply_topic(st, {"shift": "continue", "label": label}, 1, label, CFG["max_topics"])
    return st


# ==================== session_state：约束 ====================

def test_constraint_must_come_from_user():
    st = _state_with_topic()
    tc, ev = apply_constraints(st, [_set("time_range", "近三年")], 1, "苹果公司的营收怎么样", "", 15, 3, 2)
    assert not st["constraints"] and ev["rejected"][0]["reason"] == "not_from_user"
    # 用户确认助手的提议：值出现在上一轮助手回答中
    tc, ev = apply_constraints(st, [_set("time_range", "近三年")], 2, "好的", "要不要只看近三年的数据？", 15, 3, 2)
    assert st["constraints"][0]["value"] == "近三年"
    # 非确认语句不能借用助手回答中的值
    st2 = _state_with_topic()
    apply_constraints(st2, [_set("region", "中国")], 2, "那欧洲呢", "中国市场份额最高", 15, 3, 2)
    assert not st2["constraints"]


def test_constraint_explicit_replace_vs_conflict():
    st = _state_with_topic()
    apply_constraints(st, [_set("time_range", "近三年")], 1, "只看近三年", "", 15, 3, 2)
    _, ev = apply_constraints(st, [_set("time_range", "近五年")], 2, "近五年的呢", "", 15, 3, 2)
    assert st["constraints"][0]["value"] == "近三年" and ev["conflict"] and pending_conflicts(st)
    # 冲突 2 轮未确认自动失效
    apply_constraints(st, [], 4, "继续", "", 15, 3, 2)
    assert not st["conflicts"]
    _, ev = apply_constraints(st, [_set("time_range", "近五年", explicit=True)], 5, "改成近五年", "", 15, 3, 2)
    assert st["constraints"][0]["value"] == "近五年" and st["constraints"][0]["history"][0]["value"] == "近三年"


def _with_conflict():
    st = _state_with_topic()
    apply_constraints(st, [_set("language", "中文", "session")], 1, "以后都用中文回答", "", 15, 3, 2)
    apply_constraints(st, [_set("language", "英文", "session")], 2, "英文的也行", "", 15, 3, 2)
    assert pending_conflicts(st) and st["constraints"][0]["value"] == "中文"
    return st


def test_conflict_confirmed_by_affirmation():
    ask = "您是想之后改用英文回答吗？确认前我先继续用中文。"
    # 只回"好"、改写器没给出约束：上一轮助手提到了新值 → 确认
    st = _with_conflict()
    _, ev = apply_constraints(st, [], 3, "好", ask, 15, 3, 2)
    c = st["constraints"][0]
    assert c["value"] == "英文" and c["history"][-1]["value"] == "中文" and not st["conflicts"]
    assert ev["set"][0].get("confirmed")
    # 改写器看到待确认变更后给出同值 set（explicit 也可能为 false）→ 确认，且不重新记冲突
    st = _with_conflict()
    _, ev = apply_constraints(st, [_set("language", "英文", "session")], 3, "好的，改吧", "", 15, 3, 2)
    assert st["constraints"][0]["value"] == "英文" and not st["conflicts"] and not ev["conflict"]
    assert len(st["constraints"]) == 1


def test_conflict_not_confirmed():
    ask = "您是想之后改用英文回答吗？"
    # 否认 → 不改，冲突按 TTL 失效、旧值保留
    st = _with_conflict()
    apply_constraints(st, [], 3, "不用，还是中文", ask, 15, 3, 2)
    assert st["constraints"][0]["value"] == "中文" and st["conflicts"]
    apply_constraints(st, [], 4, "丹麦的首都是哪里", "", 15, 3, 2)
    assert st["constraints"][0]["value"] == "中文" and not st["conflicts"]
    # 肯定词但与冲突无关（助手没提新值、改写器给的是别的值）→ 不确认
    st = _with_conflict()
    apply_constraints(st, [], 3, "好的", "哥本哈根是丹麦的首都。", 15, 3, 2)
    assert st["constraints"][0]["value"] == "中文" and st["conflicts"]
    # "是哪"不是肯定词开头
    st = _with_conflict()
    apply_constraints(st, [], 3, "是哪一年的英文资料", ask, 15, 3, 2)
    assert st["constraints"][0]["value"] == "中文"


def test_constraint_revoke_and_turn_scope():
    st = _state_with_topic()
    apply_constraints(st, [_set("format", "简短回答", "session")], 1, "以后都简短回答", "", 15, 3, 2)
    tc, _ = apply_constraints(st, [_set("language", "英文", "turn")], 2, "这次用英文回答", "", 15, 3, 2)
    assert tc == [{"type": "language", "value": "英文", "scope": "turn"}]
    assert all(c["type"] != "language" for c in st["constraints"])        # turn 级不入库
    _, ev = apply_constraints(st, [{"op": "revoke", "type": "format"}], 3, "不用简短了", "", 15, 3, 2)
    assert not st["constraints"] and ev["revoked"]


def test_constraint_precedence_and_hash():
    st = _state_with_topic()
    apply_constraints(st, [_set("format", "简短", "session")], 1, "以后简短", "", 15, 3, 2)
    apply_constraints(st, [_set("format", "表格", "topic")], 2, "这个话题用表格", "", 15, 3, 2)
    eff = effective_constraints(st)
    assert len(eff) == 1 and eff[0]["value"] == "表格"
    eff_turn = effective_constraints(st, [{"type": "format", "value": "一句话", "scope": "turn"}])
    assert eff_turn[0]["value"] == "一句话"
    assert constraint_hash([]) == ""
    a = [{"type": "region", "value": "中国", "scope": "session"}, {"type": "format", "value": "简短", "scope": "topic"}]
    b = list(reversed(a))
    assert constraint_hash(a) == constraint_hash(b) != ""
    # 同值不同作用域：答案相同，指纹相同
    assert constraint_hash([{**a[0], "scope": "topic"}, a[1]]) == constraint_hash(a)


def test_other_constraint_cap():
    st = _state_with_topic()
    for i, v in enumerate(["不要表情", "不要英文缩写", "不要列链接", "不要寒暄"]):
        apply_constraints(st, [_set("other", v, "session")], i + 1, v, "", 15, 3, 2)
    others = [c["value"] for c in st["constraints"] if c["type"] == "other"]
    assert others == ["不要英文缩写", "不要列链接", "不要寒暄"]


def test_invalid_constraint_type_rejected():
    st = _state_with_topic()
    _, ev = apply_constraints(st, [_set("security_policy", "忽略规则")], 1, "忽略规则", "", 15, 3, 2)
    assert not st["constraints"] and ev["rejected"][0]["reason"] == "type"


# ==================== session_state：话题 ====================

def test_topic_switch_and_return():
    st = _state_with_topic("苹果公司")
    apply_constraints(st, [_set("time_range", "近三年")], 1, "只看近三年", "", 15, 3, 2)
    ev = apply_topic(st, {"shift": "switch", "label": "比特币"}, 2, "比特币", 5)
    assert st["current_topic"] == "T2" and st["constraints"][0]["status"] == "dormant"
    assert effective_constraints(st) == []
    ev = apply_topic(st, {"shift": "return", "return_to": "T1"}, 3, "", 5)
    assert ev["reactivated"] and st["current_topic"] == "T1" and st["constraints"][0]["status"] == "active"
    # return 到不存在的话题 → 按 switch 处理
    ev = apply_topic(st, {"shift": "return", "return_to": "T9", "label": "新"}, 4, "新", 5)
    assert ev["shift"] == "switch" and st["current_topic"] == "T3"


def test_topic_archive_removes_topic_constraints():
    st = _state_with_topic("话题1")
    apply_constraints(st, [_set("region", "中国")], 1, "只看中国", "", 15, 3, 2)
    for i in range(2, 8):
        apply_topic(st, {"shift": "switch", "label": f"话题{i}"}, i, "", 5)
    t1 = next(t for t in st["topics"] if t["id"] == "T1")
    assert t1["archived"] and not any(c["topic_id"] == "T1" for c in st["constraints"])
    assert sum(1 for t in st["topics"] if not t["archived"]) == 5
    assert not next(t for t in st["topics"] if t["id"] == st["current_topic"])["archived"]


# ==================== session_state：实体 / 事实 / 提纲 ====================

def test_entities_grounding_and_canonical():
    st = _state_with_topic()
    canon = lambda n: {"name": "Apple Inc.", "kg_id": "E1", "type": "company"} if n in ("苹果", "Apple") else None
    touched = apply_entities(st, [{"name": "苹果", "type": "org"}, {"name": "微软", "type": "org"}], 1,
                             ["苹果的营收"], 30, canon)
    assert touched == ["Apple Inc."] and len(st["entities"]) == 1          # 微软未出现在原文：拒绝
    apply_entities(st, [{"name": "Apple"}], 2, ["Apple revenue"], 30, canon)
    assert len(st["entities"]) == 1 and "Apple" in st["entities"][0]["aliases"] + [st["entities"][0]["name"]]


def test_facts_overwrite_by_recency():
    st = new_state()
    apply_facts(st, [{"s": "苹果", "r": "CEO", "o": "库克", "turn": 3}], 20)
    apply_facts(st, [{"s": "苹果", "r": "CEO", "o": "乔布斯", "turn": 1}], 20)     # 旧轮次不覆盖
    assert st["facts"][0]["o"] == "库克"
    apply_facts(st, [{"s": "苹果", "r": "CEO", "o": "新人", "turn": 5}], 20)
    assert len(st["facts"]) == 1 and st["facts"][0]["o"] == "新人" and st["facts"][0]["turns"] == [3, 5]


def test_outline_items():
    assert outline_items("只有一段话。没有列表。") == []
    items = outline_items("原因如下：\n1. 供应链紧张。展开……\n2. 需求下降。\n3) 汇率影响")
    assert items == ["1. 供应链紧张。", "2. 需求下降。", "3. 汇率影响"]


def test_apply_rewrite_failed_only_topic_and_no_mutation():
    st = _state_with_topic()
    before = copy.deepcopy(st)
    rw = _fallback("只看近三年")
    rw["constraints_delta"] = [_set("time_range", "近三年")]               # 降级结果即便带了约束也不应用
    new, tc, ev = apply_rewrite(st, rw, 2, "只看近三年", "", CFG)
    assert st == before and not new["constraints"] and "constraints" not in ev


def test_merge_turn_effects_on_fresh_state():
    """CAS 冲突重放：并发请求先写入的约束保留，本轮变更叠加"""
    base = _state_with_topic()
    fresh, _, _ = apply_rewrite(base, _rw(constraints_delta=[_set("region", "中国", "session")]), 2, "只看中国", "", CFG)
    fresh["version"] = 5
    rw = _rw(constraints_delta=[_set("format", "简短", "session")])
    merged = merge_turn_effects(fresh, rw, 3, "回答简短", "", "1. 第一点\n2. 第二点", CFG)
    types = sorted(c["type"] for c in merged["constraints"])
    assert types == ["format", "region"] and merged["outlines"][0]["turn"] == 3 and merged["version"] == 5


# ==================== context_builder ====================

def _turn(idx, user, assistant, **kw):
    return {"idx": idx, "user": user, "assistant": assistant, "refused": False, "rewritten_zh": "", "rewritten_en": "",
            "topic_id": "T1", **kw}


def test_rewriter_context_empty_and_refused():
    text, stats = build_rewriter_context(new_state(), [], CFG)
    assert text == "无对话历史。" and stats["recent_turns"] == 0
    st = _state_with_topic()
    text, _ = build_rewriter_context(st, [_turn(1, "苹果的CEO是谁", "抱歉，知识库中没有相关资料。", refused=True)], CFG)
    assert REFUSED_MARK in text and "抱歉" not in text


def test_rewriter_context_shows_pending_conflict():
    st = _with_conflict()
    text, _ = build_rewriter_context(st, [_turn(2, "英文的也行", "好的。")], CFG)
    assert "## 待确认的要求变更" in text and "原为「中文」" in text and "改为「英文」" in text
    text, _ = build_rewriter_context(_state_with_topic(), [_turn(1, "问题", "回答")], CFG)
    assert "待确认" not in text


def test_rewriter_context_budget():
    st = _state_with_topic()
    turns = [_turn(i, f"问题{i}" * 20, "回答内容。" * 200, rewritten_zh=f"完整问题{i}") for i in range(1, 13)]
    text, stats = build_rewriter_context(st, turns, CFG)
    assert stats["recent_tokens"] <= CFG["rewriter_recent_budget"] and "[第12轮]" in text
    assert stats["tokens"] <= CFG["rewriter_budget"] + 100


def test_compress_answer_keeps_list_items():
    ans = "主要有五点：\n" + "\n".join(f"{i}. 第{i}点标题。这里是很长的展开说明" + "细节" * 40 for i in range(1, 6))
    out = compress_answer(ans, 120)
    assert est_tokens(out) <= 120 and "细节细节细节细节" not in out
    assert all(f"{i}. 第{i}点标题" in out for i in range(1, 6)), out          # 列表项内容保留，不只剩编号
    # 列表项无句号（逗号长句）：截断后仍保留各项开头
    ans = "\n".join(f"{i}. 第{i}代车型于{1950 + i * 7}年推出，" + "采用了全新的设计和更强的发动机，" * 6 for i in range(1, 8))
    out = compress_answer(ans, 120)
    assert est_tokens(out) <= 120 and all(f"{i}. 第{i}代车型" in out for i in range(1, 8)), out


def test_agent_context_trim_order_and_budget():
    st = _state_with_topic()
    long_ans = "\n".join(f"{i}. 要点{i}。" + "展开说明" * 60 for i in range(1, 4))
    # 4 轮长回答：第 1 步不移出（窗口下限 4），第 2 步压缩较早的 2 轮
    turns = [_turn(i, f"问题{i}", long_ans) for i in range(1, 5)]
    cfg = {**CFG, "agent_budget": 1700}
    ctx = build_agent_context(st, turns, cfg)
    assert ctx["stats"]["tokens"] <= 1700, ctx["stats"]
    assert "compress_turn" in ctx["stats"]["trim_steps"] and ctx["stats"]["history_turns"] == 4
    assert ctx["history"][-1][1] == long_ans                                 # 最近一轮保持原文
    # 12 轮：先移出最早的轮次
    turns = [_turn(i, f"问题{i}", long_ans) for i in range(1, 13)]
    ctx = build_agent_context(st, turns, cfg)
    assert ctx["stats"]["tokens"] <= 1700 and ctx["stats"]["trim_steps"][0] == "drop_turn"
    assert ctx["history"][-1][0] == "问题12"
    # 极端：预算极小也不超（兜底移出最近轮次）
    ctx = build_agent_context(st, turns, {**CFG, "agent_budget": 50})
    assert ctx["stats"]["tokens"] <= 50 or not ctx["history"]


def test_agent_context_watermark_and_constraint_placement():
    st = _state_with_topic()
    apply_constraints(st, [_set("format", "简短", "session")], 1, "以后简短", "", 15, 3, 2)
    apply_constraints(st, [_set("time_range", "近三年", "topic")], 1, "只看近三年", "", 15, 3, 2)
    st["summaries"]["T1"] = {"text": "用户在问苹果公司的营收。", "until_turn": 2}
    st["summarized_until"] = 2
    turns = [_turn(i, f"问题{i}", f"回答{i}") for i in range(1, 5)]
    ctx = build_agent_context(st, turns, CFG, [{"type": "language", "value": "英文", "scope": "turn"}])
    assert [u for u, _ in ctx["history"]] == ["问题3", "问题4"]
    assert "简短" in ctx["memory"] and "近三年" not in ctx["memory"] and "用户在问苹果公司" in ctx["memory"]
    assert "近三年" in ctx["question_prefix"] and "英文" in ctx["question_prefix"]
    assert build_agent_context(new_state(), [], CFG)["memory"] == "暂无"


def test_agent_context_excludes_other_topics():
    st = _state_with_topic("苹果公司")
    st["summaries"]["T1"] = {"text": "讨论了苹果公司的营收。", "until_turn": 2}
    apply_topic(st, {"shift": "switch", "label": "滑板运动"}, 3, "滑板运动", CFG["max_topics"])
    st["summaries"]["T2"] = {"text": "讨论了滑板运动的起源。", "until_turn": 3}
    ctx = build_agent_context(st, [_turn(3, "滑板起源于哪里", "加州。", topic_id="T2")], CFG)
    assert "滑板运动的起源" in ctx["memory"] and "苹果" not in ctx["memory"] and "其他话题" not in ctx["memory"]
    # 改写器仍能看到完整话题列表（用于回到旧话题）
    text, _ = build_rewriter_context(st, [], CFG)
    assert "T1：苹果公司" in text and "T2：滑板运动（当前）" in text


# ==================== 改写器 ====================

class _Breaker:
    def __init__(self):
        self.fail = self.ok = 0

    def before_call(self):
        pass

    def on_failure(self, e):
        self.fail += 1

    def on_success(self):
        self.ok += 1


def test_rewriter_parse_and_fallback():
    assert _extract_json('```json\n{"en": "a"}\n```') == {"en": "a"}
    assert _extract_json("not json") is None and _extract_json("[1, 2]") is None
    br = _Breaker()
    rw = QueryRewriter(stm_config=CFG, circuit_breaker=br, enable_warmup=False)
    view = {"state": new_state(), "turns": []}

    rw._call_vllm = lambda prompt: None
    out = rw.rewrite("它的市值是多少", "u:s", view)
    assert out["ok"] is False and out["fallback_reason"] == "call_failed" and out["en_query"] == "它的市值是多少"
    assert br.fail == 1

    rw._call_vllm = lambda prompt: ('{"en": "What is Apple market cap?", "zh": "苹果的市值是多少？", "intent": "bogus", '
                                    '"question_type": "multi_hop", "standalone": "yes", "topic": {"shift": "jump"}, '
                                    '"constraints": [1, {"op": "set"}], "entities": "x"}')
    out = rw.rewrite("它的市值是多少", "u:s", view)
    assert out["ok"] and out["intent"] == "local" and out["question_type"] == "multi_hop"
    assert out["standalone"] is False and out["topic"]["shift"] == "continue"
    assert out["constraints_delta"] == [{"op": "set"}] and out["entities"] == [] and br.ok == 1

    out = rw.rewrite("你好", "u:s", view)
    assert out["intent"] == "direct" and out.get("rule_direct")


def test_rewriter_prompt_contains_memory():
    rw = QueryRewriter(stm_config=CFG, enable_warmup=False)
    seen = {}
    rw._call_vllm = lambda prompt: seen.setdefault("p", prompt) and None
    st = _state_with_topic("苹果公司")
    rw.rewrite("它的CEO是谁", "u:s", {"state": st, "turns": [_turn(1, "苹果公司怎么样", "苹果公司是一家科技公司。")]})
    p = seen["p"]
    assert p.index("# 当前任务") < p.index("[第1轮]") < p.index("## 用户当前问题\n它的CEO是谁")


def test_vllm_schema_downgrade_only_when_retry_succeeds():
    import httpx
    from src.core import vllm_client as vc

    def run(handler):
        orig_client, orig_flag = vc._sync_client, vc._schema_unsupported
        vc._sync_client, vc._schema_unsupported = httpx.Client(transport=httpx.MockTransport(handler)), False
        try:
            out = vc.vllm_generate_sync("http://x/v1", "m", "p", json_schema={"title": "t", "type": "object"})
            return out, vc._schema_unsupported
        finally:
            vc._sync_client, vc._schema_unsupported = orig_client, orig_flag

    ok_body = {"choices": [{"message": {"content": "{}"}}]}
    # 超出上下文：两次都 400 → 不降级
    out, flag = run(lambda req: httpx.Response(400, json={"message": "maximum context length"}))
    assert out is None and flag is False
    # 不支持 json_schema：json_object 重试成功 → 降级
    out, flag = run(lambda req: httpx.Response(400 if b"json_schema" in req.content else 200, json=ok_body))
    assert out == "{}" and flag is True


# ==================== 压缩器 ====================

def _compactor(memory=None, breaker=None):
    return Compactor(memory, CFG, "http://x", "m", "k", lambda: None, lambda: None, asyncio.Semaphore(2), breaker)


def test_compactor_validation():
    c = _compactor()
    by_idx = {3: _turn(3, "库克是苹果的CEO吗", "是的，库克是苹果公司的CEO。", turn_id="tid3")}
    facts = c._validated_facts([{"s": "苹果", "r": "CEO", "o": "库克", "turn": 3},
                                {"s": "苹果", "r": "创始人", "o": "乔布斯", "turn": 3},
                                {"s": "苹果", "r": "CEO", "o": "库克", "turn": 9}], by_idx, "T1")
    assert len(facts) == 1 and facts[0]["turn_id"] == "tid3"
    st = _state_with_topic()
    apply_constraints(st, [_set("time_range", "近三年")], 1, "只看近三年", "", 15, 3, 2)
    by_idx = {2: _turn(2, "以后都用中文回答，时间看近五年", "好的")}
    n = c._add_missed_constraints(st, [{"type": "time_range", "value": "近五年", "scope": "topic", "turn": 2},
                                       {"type": "language", "value": "中文", "scope": "session", "turn": 2},
                                       {"type": "region", "value": "美国", "scope": "session", "turn": 2}], by_idx, "T1")
    assert n == 1 and {c_["type"] for c_ in st["constraints"]} == {"time_range", "language"}
    assert next(c_ for c_ in st["constraints"] if c_["type"] == "time_range")["value"] == "近三年"


def test_compactor_select_keeps_recent():
    c = _compactor()
    st = new_state()
    turns = [_turn(i, f"问题{i}", "回答" * 600) for i in range(1, 7)]
    sel = c._select(st, turns)
    assert sel and max(t["idx"] for t in sel) <= 4
    assert c._select(st, turns[:2]) == []


def test_compactor_triggers_by_turn_count():
    """短回答：token 阈值到不了，按轮数在 Redis 淘汰前触发，压缩后剩余 ≤ compact_keep_turns 轮"""
    c = _compactor()
    st = new_state()
    short = [_turn(i, f"问题{i}", "简短回答。") for i in range(1, 13)]
    trigger = CFG["max_raw_turns"] - 2
    assert not c.should_compact(st, short[:trigger - 1]) and c.should_compact(st, short[:trigger])
    sel = c._select(st, short[:trigger])
    assert len(short[:trigger]) - len(sel) == CFG["compact_keep_turns"]
    # 水位线之后的轮数才计入
    st["summarized_until"] = 4
    assert not c.should_compact(st, short[:trigger])


class FakeMemory:
    """模拟 redis_short_memory 的 Lua 语义：发号 INCR、追加 + 淘汰、CAS"""

    def __init__(self, max_raw=12):
        self.data = {}
        self.max_raw = max_raw
        self.fail_load = False
        self.on_cas = None

    def _s(self, scope):
        return self.data.setdefault(scope, {"state": None, "turns": {}, "seq": 0})

    def load(self, scope, reserve_turn=False):
        if self.fail_load:
            raise ConnectionError("redis down")
        s = self._s(scope)
        idx = 0
        if reserve_turn:
            s["seq"] += 1
            idx = s["seq"]
        return copy.deepcopy(s["state"]), [copy.deepcopy(s["turns"][i]) for i in sorted(s["turns"])], idx

    def append_turn(self, scope, idx, turn):
        s = self._s(scope)
        t = {**turn, "idx": idx}
        if t.get("refused"):
            t["assistant"] = ""
        s["turns"][idx] = t
        s["seq"] = max(s["seq"], idx)
        for i in sorted(s["turns"])[:-self.max_raw]:
            del s["turns"][i]
        return 0

    def cas_state(self, scope, expected_version, state):
        if self.on_cas:
            hook, self.on_cas = self.on_cas, None
            hook()
        s = self._s(scope)
        cur = (s["state"] or {}).get("version", 0)
        if cur != expected_version:
            return False
        st = copy.deepcopy(state)
        st["version"] = expected_version + 1
        s["state"] = st
        state["version"] = st["version"]
        return True

    def restore(self, scope, state, turns):
        for t in turns:
            self.append_turn(scope, t["idx"], t)
        if state is not None:
            self._s(scope)["state"] = copy.deepcopy(state)

    def clear(self, scope):
        self.data.pop(scope, None)


def test_compactor_schedule_and_cas_discard():
    async def run():
        mem = FakeMemory()
        scope = "u:s"
        st = _state_with_topic()
        mem.cas_state(scope, 0, st)
        for i in range(1, 7):
            mem.append_turn(scope, i, _turn(i, f"问题{i}", "回答内容" * 300))
        c = _compactor(mem)
        calls = []

        async def fake_llm(**kw):
            calls.append(1)
            await asyncio.sleep(0.01)
            return '{"summary": "用户连续询问了多个问题。", "facts": [], "constraints": []}'

        compactor_mod.vllm_generate_async = fake_llm
        assert c.schedule(scope, "u", "s") == "scheduled"
        assert c.schedule(scope, "u", "s") == "coalesced"
        await asyncio.gather(*list(c._tasks))
        state = mem.load(scope)[0]
        assert state["summaries"]["T1"]["text"].startswith("用户连续") and state["summarized_until"] >= 1
        # 合并的请求在结束后再跑一次：水位线已前移、无可压缩轮次，不再调用模型
        assert len(calls) == 1 and scope not in c._running and scope not in c._rerun

        # 压缩期间状态被前台改写 → CAS 失败，结果丢弃
        mem2 = FakeMemory()
        mem2.cas_state(scope, 0, _state_with_topic())
        for i in range(1, 7):
            mem2.append_turn(scope, i, _turn(i, f"问题{i}", "回答内容" * 300))
        c2 = _compactor(mem2)
        mem2.on_cas = lambda: mem2.cas_state(scope, 1, {**mem2.load(scope)[0]})
        c2.schedule(scope, "u", "s")
        await asyncio.gather(*list(c2._tasks))
        assert "T1" not in mem2.load(scope)[0]["summaries"]

        class Open:
            def get_state(self):
                class V:
                    value = "OPEN"
                return V()
        assert _compactor(mem, Open()).schedule(scope, "u", "s") == "skipped_breaker"

    orig = compactor_mod.vllm_generate_async
    try:
        asyncio.run(run())
    finally:
        compactor_mod.vllm_generate_async = orig


# ==================== STMService 全流程 ====================

def _svc(mem, compactor=None):
    return STMService(mem, CFG, compactor=compactor, get_pg_pool=lambda: None)


def test_service_roundtrip_and_refused():
    async def run():
        mem = FakeMemory()
        svc = _svc(mem)
        view = await svc.prepare("u", "s")
        assert view["source"] == "new" and view["idx"] == 1
        rw = _rw(zh_query="苹果公司近三年营收", constraints_delta=[_set("time_range", "近三年")],
                 topic={"shift": "continue", "label": "苹果营收"})
        applied = svc.apply(view, rw, "苹果公司近三年营收怎么样，只看近三年")
        assert svc.cache_variant(applied) != ""
        out = await svc.finalize(view, applied, rw, "苹果公司近三年营收怎么样，只看近三年", "1. 2023年……\n2. 2024年……",
                                 refused=False, cache_hit=False, user_id="u", session_id="s")
        assert out["stm_written"] and out["stm_cas_retries"] == 0

        view2 = await svc.prepare("u", "s")
        assert view2["source"] == "redis" and view2["idx"] == 2 and len(view2["turns"]) == 1
        assert view2["state"]["constraints"][0]["value"] == "近三年" and view2["state"]["outlines"]
        applied2 = svc.apply(view2, _rw(), "那利润呢")
        assert svc.cache_variant(applied2) == svc.cache_variant(applied)     # 话题约束延续
        await svc.finalize(view2, applied2, _rw(), "那利润呢", "抱歉，知识库中没有资料。", refused=True,
                           cache_hit=False, user_id="u", session_id="s")
        t2 = (await svc.prepare("u", "s"))["turns"][-1]
        assert t2["refused"] and t2["assistant"] == ""
        ctx = svc.agent_inputs(view2, applied2)
        assert "近三年" in ctx["question_prefix"]
        # 用户隔离：同一 session_id 不同用户
        assert (await svc.prepare("other", "s"))["turns"] == []

    asyncio.run(run())


def test_service_cas_conflict_replays():
    async def run():
        mem = FakeMemory()
        svc = _svc(mem)
        v1 = await svc.prepare("u", "s")
        v2 = await svc.prepare("u", "s")                       # 并发的两个请求读到同一版本
        a1 = svc.apply(v1, _rw(constraints_delta=[_set("region", "中国", "session")]), "只看中国")
        a2 = svc.apply(v2, _rw(constraints_delta=[_set("format", "简短", "session")]), "回答简短")
        await svc.finalize(v1, a1, _rw(constraints_delta=[_set("region", "中国", "session")]), "只看中国", "好",
                           False, False, "u", "s")
        out = await svc.finalize(v2, a2, _rw(constraints_delta=[_set("format", "简短", "session")]), "回答简短", "好",
                                 False, False, "u", "s")
        assert out["stm_cas_retries"] == 1
        st, turns, _ = mem.load("u:s")
        assert sorted(c["type"] for c in st["constraints"]) == ["format", "region"] and len(turns) == 2

    asyncio.run(run())


def test_service_degraded_and_pg_restore():
    async def run():
        mem = FakeMemory()
        mem.fail_load = True
        svc = _svc(mem)
        view = await svc.prepare("u", "s")
        assert view["source"] == "degraded"
        out = await svc.finalize(view, svc.apply(view, _rw(), "x"), _rw(), "x", "y", False, False, "u", "s")
        assert out["stm_written"] is False

        mem.fail_load = False
        pg_state = _state_with_topic()
        pg_state["version"] = 7
        pg_turns = [_turn(5, "问题5", "回答5"), _turn(6, "问题6", "回答6")]

        async def fake_restore(pool, user_id, session_id, limit):
            return copy.deepcopy(pg_state), copy.deepcopy(pg_turns)

        orig = stm_mod.load_for_restore
        stm_mod.load_for_restore = fake_restore
        try:
            view = await svc.prepare("u", "s2")
        finally:
            stm_mod.load_for_restore = orig
        assert view["source"] == "pg" and view["idx"] == 7 and len(view["turns"]) == 2
        out = await svc.finalize(view, svc.apply(view, _rw(), "继续"), _rw(), "继续", "好", False, False, "u", "s2")
        assert out["stm_written"] and mem.load("u:s2")[0]["version"] == 8

        await svc.clear("u", "s2")
        assert mem.load("u:s2")[1] == []

    asyncio.run(run())


def test_service_cache_variant_isolation():
    """不同约束 → 不同缓存键；无约束为空串（与旧缓存键兼容的通用答案）"""
    async def run():
        svc = _svc(FakeMemory())
        v = await svc.prepare("u", "s")
        plain = svc.apply(v, _rw(), "苹果公司的营收")
        assert svc.cache_variant(plain) == ""
        scoped = svc.apply(v, _rw(constraints_delta=[_set("time_range", "近三年", "turn")]), "苹果公司近三年的营收")
        assert svc.cache_variant(scoped) != ""

    asyncio.run(run())


def test_service_long_short_session_never_evicts_unsummarized():
    """20 轮短回答：Redis 只留 12 轮，每一轮在被淘汰之前都已并入摘要（水位线 ≥ 被淘汰的最大轮次）"""
    async def run():
        mem = FakeMemory(max_raw=CFG["max_raw_turns"])
        comp = _compactor(mem)
        svc = _svc(mem, comp)

        async def fake_llm(**kw):
            return '{"summary": "用户询问了丹麦的多个问题。", "facts": [], "constraints": []}'

        compactor_mod.vllm_generate_async = fake_llm
        compactions = 0
        for i in range(1, 21):
            view = await svc.prepare("u", "s")
            rw = _rw(zh_query=f"丹麦问题{i}", topic={"shift": "continue", "label": "丹麦"})
            applied = svc.apply(view, rw, f"丹麦问题{i}")
            out = await svc.finalize(view, applied, rw, f"丹麦问题{i}", "简短回答。", refused=False, cache_hit=False,
                                     user_id="u", session_id="s")
            compactions += out.get("stm_compaction") == "scheduled"
            await asyncio.gather(*list(comp._tasks))
            state, turns, _ = mem.load("u:s")
            evicted_max = min(t["idx"] for t in turns) - 1
            assert evicted_max <= state.get("summarized_until", 0), (i, evicted_max, state.get("summarized_until"))
            unsumm = [t for t in turns if t["idx"] > state.get("summarized_until", 0)]
            assert len(unsumm) < CFG["max_raw_turns"] - 1
        assert compactions >= 2 and state["summaries"]["T1"]["text"]
        assert svc.agent_inputs(view, applied)["stats"]["has_summary"]

        # 前端提示：压缩完成后的下一轮提示一次，之后不重复
        notices, scheduled = [], []
        for i in range(21, 33):
            view = await svc.prepare("u", "s")
            notices.append(svc.new_summary_until(view))
            rw = _rw(zh_query=f"丹麦问题{i}", topic={"shift": "continue", "label": "丹麦"})
            applied = svc.apply(view, rw, f"丹麦问题{i}")
            out = await svc.finalize(view, applied, rw, f"丹麦问题{i}", "简短回答。", refused=False, cache_hit=False,
                                     user_id="u", session_id="s")
            scheduled.append(out.get("stm_compaction") == "scheduled")
            await asyncio.gather(*list(comp._tasks))
        # 第 k 轮触发的压缩完成后，恰好在第 k+1 轮提示一次
        assert any(scheduled[:-1])
        assert [bool(n) for n in notices[1:]] == scheduled[:-1], (notices, scheduled)
        shown = [n for n in notices if n]
        assert len(shown) == len(set(shown)) and shown == sorted(shown), notices

    orig = compactor_mod.vllm_generate_async
    try:
        asyncio.run(run())
    finally:
        compactor_mod.vllm_generate_async = orig


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
