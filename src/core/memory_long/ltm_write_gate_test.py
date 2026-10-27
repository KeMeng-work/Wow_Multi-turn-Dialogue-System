"""
长期记忆写入门槛单元测试（本地可跑，不依赖 PG / vLLM；1.7B 输出用桩函数模拟）

覆盖：L0 前置门槛、L2 写入前校验（依据、提问记录、占位符、敏感信息）、extract_candidates 全流程
运行：python src/core/memory_long/ltm_write_gate_test.py
"""

import asyncio
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))


def _stub_missing(name: str, **attrs) -> None:
    """本地开发机未装 PG / ONNX 依赖时用空模块占位（容器内为真实依赖，不受影响）"""
    try:
        __import__(name)
    except ImportError:
        import types
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod


_stub_missing("asyncpg", Pool=object, Connection=object)
_stub_missing("pgvector")
_stub_missing("pgvector.asyncpg", register_vector=None)

from src.core.memory_long import long_term_memory as ltm_mod  # noqa: E402
from src.core.memory_long.long_term_memory import LongTermMemory, memory_reject_reason, should_extract


def _mem(content, content_en, evidence="", memory_type="fact"):
    return {"content": content, "content_en": content_en, "evidence": evidence, "memory_type": memory_type}


def test_gate_requires_self_reference_or_remember():
    for q in ("丹麦王国的地理位置在哪里？", "好时公司总部在哪里？", "钢琴是谁发明的？", "Who invented the piano?",
              "好的", "今天天气不错"):
        assert not should_extract(q), q
    for q in ("我是一名 Python 后端开发工程师", "作为一名高中老师，怎么讲光合作用", "I'm a data scientist",
              "记住：好时公司的总部在纽约。", "my favorite team is Arsenal"):
        assert should_extract(q), q


def test_question_log_rejected():
    u = "我想知道红葡萄酒的制作原料是什么"
    for c, e in (("用户询问红葡萄酒的制作原料", "The user asked about the ingredients of red wine"),
                 ("用户想知道红葡萄酒的原料", "The user wants to know the ingredients of red wine"),
                 ("用户想了解丹麦和瑞典的关系", "The user wants to learn about Denmark and Sweden"),
                 ("用户问钢琴是谁发明的", "The user asks who invented the piano"),
                 ("用户对红葡萄酒感兴趣", "The user inquired about red wine")):
        assert memory_reject_reason(_mem(c, e, u), u) == "question_log", c


def test_evidence_must_be_in_user_text_with_self_reference():
    u = "我是一名 Python 后端开发工程师"
    ok = _mem("用户是一名 Python 后端开发工程师", "The user is a Python backend developer", "我是一名Python后端开发工程师")
    assert memory_reject_reason(ok, u) is None                         # 忽略空白差异
    assert memory_reject_reason({**ok, "evidence": ""}, u) == "no_evidence"
    assert memory_reject_reason({**ok, "evidence": "我是一名Java工程师"}, u) == "no_evidence"   # 编造的依据
    # 外部世界的断言：依据在原话中，但所在句没有自称
    u2 = "记住：好时公司的总部在纽约。"
    claim = _mem("用户认为好时公司总部在纽约", "The user believes Hershey is headquartered in New York", "好时公司的总部在纽约")
    assert memory_reject_reason(claim, u2) == "no_evidence"
    # 提问与个人信息同句：依据只摘个人信息部分也能通过（所在句含自称）
    u3 = "我在做 AI 应用开发，RAG 系统一般怎么评测？"
    assert memory_reject_reason(_mem("用户在做 AI 应用开发", "The user is developing AI applications", "我在做 AI 应用开发"), u3) is None
    # 偏好的依据不含"我"但所在句含
    u4 = "我是 AI 工程师，平时喜欢看硬科幻电影"
    assert memory_reject_reason(_mem("用户喜欢硬科幻电影", "The user likes hard sci-fi movies", "平时喜欢看硬科幻电影",
                                     "preference"), u4) is None


def test_sensitive_requires_explicit_remember():
    u = "我最近确诊了焦虑症，有点担心"
    m = _mem("用户最近确诊了焦虑症", "The user was recently diagnosed with anxiety", "我最近确诊了焦虑症")
    assert memory_reject_reason(m, u) == "sensitive"
    u2 = "记住，我对花生过敏"
    m2 = _mem("用户对花生过敏", "The user is allergic to peanuts", "我对花生过敏")
    assert memory_reject_reason(m2, u2) is None
    assert memory_reject_reason(m2, "我对花生过敏") == "sensitive"


def test_placeholder_rejected_and_format():
    u = "我的手机号是[手机号_1]，记住"
    m = _mem("用户的手机号是[手机号_1]", "The user's phone number is [手机号_1]", "我的手机号是[手机号_1]")
    assert memory_reject_reason(m, u) == "pii"
    assert memory_reject_reason(_mem("好时公司总部在纽约", "Hershey is in New York", "x"), "x") == "format"
    # 清理脚本模式（无原话）：只看正文规则
    assert memory_reject_reason(_mem("用户询问丹麦王国的地理位置", "The user asked where Denmark is"), None) == "question_log"
    assert memory_reject_reason(_mem("用户是一名 Python 后端开发工程师", "The user is a Python developer"), None) is None


def test_extract_candidates_flow():
    calls = []

    async def fake_llm(**kw):
        calls.append(kw["prompt"])
        return json.dumps({"memories": [
            {"content": "用户询问钢琴的发明者", "content_en": "The user asked who invented the piano", "memory_type": "episodic",
             "importance_score": 0.5, "evidence": "我想知道钢琴是谁发明的"},
            {"content": "用户在做 AI 应用开发", "content_en": "The user is developing AI applications", "memory_type": "episodic",
             "importance_score": 0.8, "evidence": "我在做 AI 应用开发"},
        ]}, ensure_ascii=False)

    async def run():
        ltm = LongTermMemory(db_pool=None, http_client=None, embedder=None)
        out = await ltm.extract_candidates("钢琴是谁发明的？", "钢琴由克里斯托福里发明。")
        assert out == {"skipped": "no_self_reference", "memories": [], "rejected": []} and not calls   # 不调用 1.7B
        out = await ltm.extract_candidates("我在做 AI 应用开发，我想知道钢琴是谁发明的", "……")
        assert [m["content"] for m in out["memories"]] == ["用户在做 AI 应用开发"]
        assert out["rejected"] == [{"content": "用户询问钢琴的发明者", "reason": "question_log"}] and len(calls) == 1
        mems = await ltm._extract_memories("我在做 AI 应用开发，我想知道钢琴是谁发明的", "……")
        assert len(mems) == 1

    orig = ltm_mod.vllm_generate_async
    ltm_mod.vllm_generate_async = fake_llm
    try:
        asyncio.run(run())
    finally:
        ltm_mod.vllm_generate_async = orig


def test_extract_candidates_call_failure_and_bad_json():
    async def none_llm(**kw):
        return None

    async def bad_llm(**kw):
        return "不是 JSON"

    async def run():
        ltm = LongTermMemory(db_pool=None, http_client=None, embedder=None)
        ltm_mod.vllm_generate_async = none_llm
        assert (await ltm.extract_candidates("我是工程师", ""))["memories"] is None
        assert await ltm._extract_memories("我是工程师", "") is None           # 调用失败与"无需提取"区分
        ltm_mod.vllm_generate_async = bad_llm
        assert (await ltm.extract_candidates("我是工程师", ""))["memories"] == []

    orig = ltm_mod.vllm_generate_async
    try:
        asyncio.run(run())
    finally:
        ltm_mod.vllm_generate_async = orig


# ==================== 召回门（门 1） ====================

def test_recall_gate_parse_and_rule():
    from src.core.memory_long.recall_gate import parse_answer, rule_decide
    assert parse_answer("<think>\n\n</think>\n\ntrue") is True and parse_answer("False") is False
    assert parse_answer('"true"') is True and parse_answer("yes") is None and parse_answer(None) is None
    assert rule_decide("给我推荐几部电影") and rule_decide("我能吃吗") and rule_decide("周末计划怎么安排")
    assert not rule_decide("丹麦王国的首都在哪里？") and not rule_decide("《星际穿越》的导演是谁？")


def test_recall_gate_decide_and_fallback():
    from src.core.memory_long import recall_gate as rg_mod
    outputs = {}

    async def fake_llm(**kw):
        return outputs["raw"]

    async def run():
        gate = rg_mod.RecallGate(lambda: None, "http://x", "m", "k")
        outputs["raw"] = "<think>\n\n</think>\n\nfalse"
        assert await gate.decide("《星际穿越》的导演是谁？") == (False, "model")
        outputs["raw"] = "true"
        assert await gate.decide("给我推荐几部电影") == (True, "model")
        outputs["raw"] = None                                       # 调用失败 → 规则
        assert await gate.decide("给我推荐几部电影") == (True, "rule_fallback")
        assert await gate.decide("丹麦王国的首都在哪里？") == (False, "rule_fallback")
        outputs["raw"] = "{"                                        # 无法解析 → 规则
        assert await gate.decide("我能吃花生吗") == (True, "rule_fallback")

    orig = rg_mod.vllm_generate_async
    rg_mod.vllm_generate_async = fake_llm
    try:
        asyncio.run(run())
    finally:
        rg_mod.vllm_generate_async = orig


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
