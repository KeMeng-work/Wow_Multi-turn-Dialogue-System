"""
飞轮评估专用请求处理器。

仅供 /chat_eval 接口调用，与生产 /chat 路径完全隔离。
返回 answer + retrieved_contexts 供飞轮 RAGAS 离线评估。

与 handle_request_async 的区别：
  - 跳过高频缓存（必须走完整 RAG 才能拿到 contexts）
  - 跳过短期记忆读写（每条 eval 问题独立，无对话历史）
  - 跳过长期记忆写入（不污染生产用户记忆状态）
  - 跳过 PG 写入（离线评估不需要持久化）
  - 跳过限流（内部工具，不经过用户配额）
"""

from datetime import datetime

from langchain_core.messages import HumanMessage

from src.core.high_frequency_query_cache.circuit_breaker import CircuitBreakerOpen


async def handle_request_eval(
    user_input: str,
    session_id: str,
    user_id: str = "flywheel_eval",
    role: str = "ops",
) -> dict:
    """
    执行完整检索推理流程，返回 answer 和 retrieved_contexts。

    Returns:
        {"answer": str, "contexts": list[str], "intent": str}
    """
    # 延迟导入生产全局实例，避免循环导入
    from src.main import (
        agent_app,
        direct_chain,
        graph_prefetch,
        merge_graph_evidence,
        rewriter,
        circuit_breaker,
        long_term_memory,
        _format_long_term_memory,
        _answer_policy,
        POSTPROCESS_CONFIG,
    )

    from src.core.memory_short.redis_short_memory import stm_scope
    from src.core.security.context import set_context
    from src.core.security.pii import PIIVault
    set_context(user_id, session_id, role, PIIVault())   # 评估不需要跨请求还原：内存映射表，照常脱敏外发内容

    # ── 查询改写 ──────────────────────────────────────────────────
    rewritten = rewriter.rewrite(user_input, stm_scope(user_id, session_id))
    en_query = rewritten["en_query"]
    zh_query = rewritten["zh_query"]
    intent = rewritten.get("intent", "local")
    question_type = rewritten.get("question_type", "simple")  # 题型分类：simple / reasoning / multi_hop

    # ── 长期记忆（只读；flywheel_eval 用户无历史，通常返回空）────
    ltm_str = "暂无长期记忆"
    if long_term_memory is not None:
        memories = await long_term_memory.retrieve_relevant(
            user_id=user_id, query=zh_query, memory_type=None,
        )
        ltm_str = _format_long_term_memory(memories)

    # ── Agent 调用 ────────────────────────────────────────────────
    retrieved_contexts: list[str] = []
    final_answer = "抱歉，我无法回答这个问题。"

    try:
        circuit_breaker.before_call()

        if intent == "direct":
            response = await direct_chain.ainvoke({
                "messages": [HumanMessage(content=user_input)],
                "current_date": datetime.now().strftime("%Y年%m月%d日"),
                "long_term_memory": ltm_str,
            })
            final_answer = response.content.strip() or final_answer

        else:
            # 与线上一致：multi_hop 题先做知识图谱预取（KG_ENABLED / KG_PREFETCH 控制，评估对照用）
            graph_state, _ = await graph_prefetch(question_type, intent, [en_query, user_input, zh_query])
            result = await agent_app.ainvoke({
                **graph_state,
                "query": en_query,
                "user_input": user_input,
                "messages": [HumanMessage(content=user_input)],
                "step_count": 0,
                "max_steps": {"simple": 3, "reasoning": 5, "multi_hop": 7}.get(question_type, 5),
                "final_answer": None,
                "long_term_memory": ltm_str,
                "current_date": datetime.now().strftime("%Y年%m月%d日"),
                "intent": intent,
                "question_type": question_type,
            })
            final_answer = result.get("final_answer") or final_answer
            # 答案依据 = 检索段落 + 知识图谱证据原文
            retrieved_contexts = merge_graph_evidence(result.get("retrieved_contexts"), result.get("graph_contexts"))
            # 与线上一致的回答后处理：拒答时 final_answer 已是固定话术；全部低置信时追加缺失声明
            if _answer_policy(result)["append_notice"]:
                final_answer = f"{final_answer}\n\n{POSTPROCESS_CONFIG['low_evidence_notice']}"

        circuit_breaker.on_success()

    except CircuitBreakerOpen:
        raise
    except Exception as e:
        circuit_breaker.on_failure(e)
        raise

    return {"answer": final_answer, "contexts": retrieved_contexts, "intent": intent}
