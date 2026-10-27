"""
短期记忆编排 (STM Service) —— main.py 只通过本模块读写短期记忆

一轮请求的生命周期：
    prepare()      读取会话（Redis 一次往返 + 发号本轮 idx）；Redis 缺失时从 PG 同步恢复
    apply()        改写结果落到状态（内存中）：话题 / 约束 / 实体，本轮立即生效
    agent_inputs() 按 Agent 预算组装：系统提示词中的会话记忆段 + 近期原文消息 + 当前问题前的约束提示
    cache_variant()影响答案的约束指纹（缓存键的一部分）
    finalize()     回答之后：原子写入本轮 → 在最新状态上重放本轮变更并 CAS 写入（冲突重试）→ PG 快照 → 必要时提交压缩
所有对外方法都不抛异常：短期记忆故障只降级（无上下文 / 不写入），不影响回答。
设计文档：docs/short_term_memory_design.md
"""

import asyncio
import logging
import time
import uuid
from typing import Callable, Optional, Set

from src.core.memory_short.context_builder import build_agent_context, unsummarized_tokens
from src.core.memory_short.redis_short_memory import STM_METRICS, ShortTermMemory, stm_scope
from src.core.memory_short.session_state import (
    apply_rewrite, constraint_hash, effective_constraints, merge_turn_effects, new_state, pending_conflicts,
)
from src.core.memory_short.session_store_pg import delete_state, load_for_restore, save_state

logger = logging.getLogger(__name__)

_CAS_RETRIES = 3


class STMService:
    def __init__(self, memory: ShortTermMemory, cfg: dict, compactor=None,
                 get_pg_pool: Optional[Callable] = None, canonicalize: Optional[Callable] = None,
                 background_tasks: Optional[Set] = None):
        self.memory = memory
        self.cfg = cfg
        self.compactor = compactor
        self._pool = get_pg_pool or (lambda: None)
        self._canon = canonicalize
        self._bg = background_tasks if background_tasks is not None else set()

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    # ==================== 读取 ====================
    async def prepare(self, user_id: str, session_id: str) -> dict:
        """
        返回 view：{scope, state, turns, idx, turn_id, source: redis / pg / new / degraded, load_ms}
        degraded：Redis 不可用，本轮无上下文且不写短期记忆（PG 全量写入不受影响）
        """
        scope = stm_scope(user_id, session_id)
        t0 = time.perf_counter()
        view = {"scope": scope, "state": new_state(), "turns": [], "idx": 0,
                "turn_id": str(uuid.uuid4()), "source": "new"}
        try:
            state, turns, idx = await asyncio.to_thread(self.memory.load, scope, True)
        except Exception as e:
            logger.warning(f"短期记忆读取失败，本轮无上下文: {e}")
            STM_METRICS.incr("load_failed")
            view["source"] = "degraded"
            view["load_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return view

        if state is None and not turns:
            try:
                pg_state, pg_turns = await load_for_restore(self._pool(), user_id, session_id,
                                                            self.cfg["pg_restore_turns"])
            except Exception as e:
                logger.warning(f"从 PG 恢复会话失败，按新会话处理: {e}")
                pg_state, pg_turns = None, []
            if pg_state is not None or pg_turns:
                try:
                    await asyncio.to_thread(self.memory.restore, scope, pg_state, pg_turns)
                    state, turns, idx = await asyncio.to_thread(self.memory.load, scope, True)
                    view["source"] = "pg"
                    STM_METRICS.incr("pg_restored")
                    # 只有原文没有状态时不在此补建摘要：本轮 finalize 写入状态后按 should_compact 正常触发
                    logger.info(f"短期记忆从 PG 恢复：{len(pg_turns)} 轮，状态{'有' if pg_state else '无'}")
                except Exception as e:
                    # 写回失败：本轮用 PG 数据组装上下文，但不写短期记忆（发号与恢复的轮次序号可能重叠）
                    logger.warning(f"PG 恢复写回 Redis 失败，本轮只读使用 PG 数据: {e}")
                    STM_METRICS.incr("pg_restore_writeback_failed")
                    view.update(state=pg_state or new_state(), turns=pg_turns, idx=0, source="degraded")
                    view["load_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    return view
            else:
                view["source"] = "new"
        else:
            view["source"] = "redis"
        view.update(state=state or new_state(), turns=turns, idx=idx)
        view["load_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return view

    # ==================== 本轮生效 ====================
    def apply(self, view: dict, rw: dict, user_input: str) -> dict:
        """改写结果落到状态（内存）；返回 {state, turn_constraints, events}"""
        last_assistant = view["turns"][-1].get("assistant", "") if view["turns"] else ""
        state, turn_c, events = apply_rewrite(view["state"], rw, view["idx"] or len(view["turns"]) + 1,
                                              user_input, last_assistant, self.cfg, self._canon)
        return {"state": state, "turn_constraints": turn_c, "events": events}

    def agent_inputs(self, view: dict, applied: dict) -> dict:
        return build_agent_context(applied["state"], view["turns"], self.cfg, applied["turn_constraints"])

    @staticmethod
    def cache_variant(applied: dict) -> str:
        return constraint_hash(effective_constraints(applied["state"], applied["turn_constraints"]))

    @staticmethod
    def new_summary_until(view: dict) -> int:
        """上一轮之后后台压缩新形成了摘要时，返回摘要覆盖到的轮次（前端只在这一轮提示一次）；否则 0"""
        st = view.get("state") or {}
        until = int(st.get("summarized_until", 0))
        return until if until > int(st.get("summary_seen_until", 0)) else 0

    @staticmethod
    def has_pending_conflict(applied: dict) -> bool:
        return bool(pending_conflicts(applied["state"]))

    @staticmethod
    def pg_metadata(view: dict, applied: dict, rw: dict, refused: bool) -> dict:
        """写入 conversation_history.metadata.stm：PG 恢复与按话题重建摘要的依据"""
        return {"turn_idx": view["idx"], "turn_id": view["turn_id"],
                "topic_id": applied["state"].get("current_topic") or "",
                "rewritten_en": rw.get("en_query") or "", "rewritten_zh": rw.get("zh_query") or "",
                "refused": bool(refused)}

    # ==================== 写入 ====================
    async def finalize(self, view: dict, applied: dict, rw: dict, user_input: str, answer: str,
                       refused: bool, cache_hit: bool, user_id: str, session_id: str) -> dict:
        """回答之后写入本轮；返回观测字段。任何失败只计数，不抛出"""
        out = {"stm_written": False, "stm_cas_retries": 0, "stm_compaction": None}
        if view["source"] == "degraded" or not view["idx"]:
            return out
        scope = view["scope"]
        state_now = applied["state"]
        turn = {"turn_id": view["turn_id"], "user": user_input, "assistant": answer, "refused": refused,
                "rewritten_en": rw.get("en_query") or "", "rewritten_zh": rw.get("zh_query") or "",
                "topic_id": state_now.get("current_topic") or "", "cache_hit": cache_hit, "ts": round(time.time(), 3)}
        last_assistant = view["turns"][-1].get("assistant", "") if view["turns"] else ""
        try:
            await asyncio.to_thread(self.memory.append_turn, scope, view["idx"], turn)
            out["stm_written"] = True
        except Exception as e:
            STM_METRICS.incr("write_failed")
            logger.error(f"短期记忆写入失败（本轮不进入上下文）: {e}")
            return out

        # 在最新状态上重放本轮变更再 CAS：并发请求 / 后台压缩先写入时不会被本轮覆盖
        base = view["state"]
        new_state_ = None
        for attempt in range(_CAS_RETRIES):
            new_state_ = merge_turn_effects(base, rw, view["idx"], user_input, last_assistant, answer,
                                            self.cfg, self._canon)
            # 本轮读取时已看到的摘要水位：之后压缩新推进的部分留给下一轮提示
            new_state_["summary_seen_until"] = max(int(new_state_.get("summary_seen_until", 0)),
                                                   int((view["state"] or {}).get("summarized_until", 0)))
            try:
                ok = await asyncio.to_thread(self.memory.cas_state, scope, int(base.get("version", 0)), new_state_)
            except Exception as e:
                STM_METRICS.incr("state_write_failed")
                logger.error(f"会话状态写入失败: {e}")
                return out
            if ok:
                break
            STM_METRICS.incr("cas_conflict")
            out["stm_cas_retries"] = attempt + 1
            try:
                fresh, _, _ = await asyncio.to_thread(self.memory.load, scope)
            except Exception:
                return out
            base = fresh or new_state()
        else:
            STM_METRICS.incr("cas_gave_up")
            logger.warning(f"会话状态 CAS 重试 {_CAS_RETRIES} 次仍冲突，本轮状态变更放弃（原文已写入）: {scope}")
            return out

        self._spawn(save_state(self._pool(), user_id, session_id, new_state_))

        if self.compactor:
            turns_after = [t for t in view["turns"] if t["idx"] != view["idx"]] + [{**turn, "idx": view["idx"]}]
            reactivated = (applied["events"].get("topic") or {}).get("reactivated")
            cur = new_state_.get("current_topic")
            if reactivated and cur and cur not in new_state_.get("summaries", {}):
                out["stm_compaction"] = self.compactor.schedule(scope, user_id, session_id, rebuild_topic=cur)
            elif self.compactor.should_compact(new_state_, turns_after):
                out["stm_compaction"] = self.compactor.schedule(scope, user_id, session_id)
            out["stm_unsummarized_tokens"] = unsummarized_tokens(new_state_, turns_after)
        return out

    # ==================== 删除 ====================
    async def clear(self, user_id: str, session_id: str) -> None:
        scope = stm_scope(user_id, session_id)
        try:
            await asyncio.to_thread(self.memory.clear, scope)
        except Exception as e:
            logger.warning(f"清空短期记忆失败: {e}")
        try:
            await delete_state(self._pool(), user_id, session_id)
        except Exception as e:
            logger.warning(f"删除 PG 会话状态失败: {e}")
