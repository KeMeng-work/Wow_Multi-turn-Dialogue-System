"""
短期记忆异步压缩 (Compactor) —— 回答返回之后在后台执行，不在请求关键路径上

设计文档：docs/short_term_memory_design.md §9.2、§9.3
    触发      水位线之后的原文超过 Agent 预算的 compact_trigger_ratio；或未压缩轮数达到 max_raw_turns - 2
              （短回答时 token 阈值到不了，按轮数在 Redis 淘汰最早轮次之前压缩，留 2 轮余量给异步执行）
    内容      旧摘要 + 即将离开窗口的轮次 → 本地 4B 递归合并为新摘要（按话题）；抽取事实三元组；二次抽取遗漏的约束
    重建      某话题累计 rebuild_every 次压缩，或回到已归档话题时，直接用 PG 原文重新生成摘要，消除"摘要的摘要"累积失真
    校验      三元组主体与客体须出现在来源轮次原文中；约束值须出现在来源轮次的用户原话中
    竞态      启动时记下状态版本，写回时 CAS，版本已变化则丢弃本次结果，等下一次触发
    合并      同一会话同时只有一个压缩任务；运行期间再次触发只标记"结束后再跑一次"
    背压      待执行任务超过 compact_backlog_max 时丢弃新任务；4B 熔断打开时不提交
    降级      失败 / 超时 / 丢弃都不影响请求：上下文由 Agent 预算的硬截断兜底
并发：与长期记忆后台任务共用后台 LLM 信号量；前台改写与证据判定有独立舱壁。
"""

import asyncio
import copy
import json
import logging
import re
from typing import Callable, Dict, List, Optional, Set

from src.core.memory_short.context_builder import REFUSED_MARK, compress_answer, est_tokens, unsummarized_tokens, _clip
from src.core.memory_short.redis_short_memory import STM_METRICS
from src.core.memory_short.session_state import (
    CONSTRAINT_TYPES, _norm, apply_facts, ensure_schema,
)
from src.core.memory_short.session_store_pg import load_recent_turns, save_state
from src.core.vllm_client import vllm_generate_async

logger = logging.getLogger(__name__)

COMPACT_SCHEMA = {
    "title": "compact",
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "facts": {"type": "array", "items": {"type": "object", "properties": {
            "s": {"type": "string"}, "r": {"type": "string"}, "o": {"type": "string"}, "turn": {"type": "integer"}},
            "required": ["s", "r", "o", "turn"]}},
        "constraints": {"type": "array", "items": {"type": "object", "properties": {
            "type": {"type": "string", "enum": list(CONSTRAINT_TYPES)}, "value": {"type": "string"},
            "scope": {"type": "string", "enum": ["session", "topic"]}, "turn": {"type": "integer"}},
            "required": ["type", "value", "scope", "turn"]}},
    },
    "required": ["summary", "facts", "constraints"],
}

COMPACT_SYSTEM_PROMPT = (
    "你是对话记忆整理助手。输入是某个话题的已有摘要和一段对话（每轮标注 [第N轮]），输出一个 JSON 对象：\n"
    "1. summary：把已有摘要与这段对话合并成新的话题摘要，150 字以内，第三人称客观陈述：用户关心什么、问过哪些问题、"
    "得到了哪些关键结论、还有哪些未解决的问题。只写对话中出现过的内容，不推测、不补充。已有摘要为空时只总结这段对话。\n"
    "2. facts：对话中明确陈述的事实，三元组 (s 主体, r 关系, o 客体)，turn 为该事实出现的轮次编号。"
    "s 和 o 必须是对话原文中出现过的词语；只抽明确说出的关系，不做推断（例如原文说某人是\"CEO 和大股东\"，不能写成\"创立\"）。"
    "最多 8 条，没有则为空数组。\n"
    "3. constraints：只从\"用户：\"的发言中找出要求系统之后怎么回答的指示（时间范围 time_range、对象范围 object_scope、"
    "地域 region、语言 language、格式与长度 format、受众 audience、其他 other），必须带持续性意图；"
    "描述事实或只属于某次提问的查询条件都不是约束。value 从用户原话逐字摘录，scope 为 session（全程）或 topic（当前话题），"
    "turn 为所在轮次。没有则为空数组。\n"
    "只输出 JSON。"
)


def _answer(t: dict) -> str:
    return REFUSED_MARK if t.get("refused") else (t.get("assistant") or "")


def _turn_text(t: dict) -> str:
    return (t.get("user") or "") + "\n" + _answer(t)


def _int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


class Compactor:
    def __init__(self, memory, cfg: dict, base_url: str, model: str, api_key: str,
                 get_http_client: Callable, get_pg_pool: Callable, background_sem: asyncio.Semaphore,
                 circuit_breaker=None):
        self.memory = memory
        self.cfg = cfg
        self.base_url, self.model, self.api_key = base_url, model, api_key
        self._http = get_http_client
        self._pool = get_pg_pool
        self._sem = background_sem
        self._breaker = circuit_breaker      # 只读：熔断打开时不提交；后台失败不计入（不能因后台拖累前台改写）
        self._running: Set[str] = set()
        self._rerun: Dict[str, Optional[str]] = {}
        self._tasks: Set[asyncio.Task] = set()

    # ==================== 调度 ====================
    def _trigger_turns(self) -> int:
        return max(self.cfg["compact_keep_turns"] + 1, self.cfg["max_raw_turns"] - 2)

    def should_compact(self, state: dict, turns: List[dict]) -> bool:
        w = state.get("summarized_until", 0)
        if sum(1 for t in turns if t["idx"] > w) >= self._trigger_turns():
            return True
        return unsummarized_tokens(state, turns) > self.cfg["agent_budget"] * self.cfg["compact_trigger_ratio"]

    def _breaker_open(self) -> bool:
        if self._breaker is None:
            return False
        try:
            return self._breaker.get_state().value == "OPEN"
        except Exception:
            return False

    def schedule(self, scope: str, user_id: str, session_id: str, rebuild_topic: Optional[str] = None) -> str:
        """提交压缩任务；返回 scheduled / coalesced / dropped_backlog / skipped_breaker"""
        if self._breaker_open():
            STM_METRICS.incr("compact_skipped_breaker")
            return "skipped_breaker"
        if scope in self._running:
            self._rerun[scope] = rebuild_topic or self._rerun.get(scope)
            STM_METRICS.incr("compact_coalesced")
            return "coalesced"
        if len(self._tasks) >= self.cfg["compact_backlog_max"]:
            STM_METRICS.incr("compact_dropped_backlog")
            return "dropped_backlog"
        self._running.add(scope)
        task = asyncio.create_task(self._run(scope, user_id, session_id, rebuild_topic))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return "scheduled"

    async def _run(self, scope: str, user_id: str, session_id: str, rebuild_topic: Optional[str]) -> None:
        try:
            while True:
                try:
                    await self._compact_once(scope, user_id, session_id, rebuild_topic)
                except Exception as e:
                    STM_METRICS.incr("compact_failed")
                    logger.warning(f"短期记忆压缩失败（不影响请求，硬截断兜底）{scope}: {type(e).__name__}: {e}")
                if scope not in self._rerun:
                    break
                rebuild_topic = self._rerun.pop(scope)
        finally:
            self._running.discard(scope)
            self._rerun.pop(scope, None)

    # ==================== 执行 ====================
    def _select(self, state: dict, turns: List[dict]) -> List[dict]:
        """选出要并入摘要的最早若干轮：压缩后剩余原文 ≤ keep_ratio × 预算且 ≤ compact_keep_turns 轮，至少保留最近 2 轮原文"""
        w = state.get("summarized_until", 0)
        unsumm = [t for t in turns if t["idx"] > w]
        if len(unsumm) <= 2:
            return []
        keep_budget = self.cfg["agent_budget"] * self.cfg["compact_keep_ratio"]
        keep_turns = max(2, self.cfg["compact_keep_turns"])
        remaining = sum(est_tokens(_turn_text(t)) for t in unsumm)
        left = len(unsumm)
        chosen = []
        for t in unsumm[:-2]:
            if remaining <= keep_budget and left <= keep_turns:
                break
            chosen.append(t)
            remaining -= est_tokens(_turn_text(t))
            left -= 1
        return chosen

    async def _call_4b(self, summary: str, turns: List[dict]) -> Optional[dict]:
        convo = "\n".join(f"[第{t['idx']}轮]\n用户：{t.get('user', '')}\n助手：{compress_answer(_answer(t), 300)}"
                          for t in turns)
        prompt = f"## 已有摘要\n{summary or '（空）'}\n\n## 对话\n{convo}\n\n## 输出\n"
        raw = await vllm_generate_async(
            http_client=self._http(), base_url=self.base_url, model=self.model, prompt=prompt,
            system_prompt=COMPACT_SYSTEM_PROMPT, api_key=self.api_key, timeout=self.cfg["compact_timeout_s"],
            temperature=0.0, max_tokens=700, json_schema=COMPACT_SCHEMA)
        if not raw:
            return None
        try:
            obj = json.loads(raw)
        except ValueError:
            m = re.search(r"\{.*\}", raw, re.S)
            obj = json.loads(m.group(0)) if m else None
        return obj if isinstance(obj, dict) else None

    def _validated_facts(self, facts: list, by_idx: Dict[int, dict], topic_id: str) -> List[dict]:
        out = []
        for f in facts or []:
            if not isinstance(f, dict):
                continue
            t = by_idx.get(_int(f.get("turn")))
            if t is None:
                STM_METRICS.incr("compact_fact_rejected")
                continue
            src = _norm(_turn_text(t))
            if _norm(f.get("s")) and _norm(f.get("o")) and _norm(f["s"]) in src and _norm(f["o"]) in src:
                out.append({**f, "turn": t["idx"], "turn_id": t.get("turn_id"), "topic_id": topic_id})
            else:
                STM_METRICS.incr("compact_fact_rejected")
        return out

    @staticmethod
    def _add_missed_constraints(state: dict, items: list, by_idx: Dict[int, dict], topic_id: str) -> int:
        """二次抽取：只补"该作用范围内同类型完全没有记录"的约束，不覆盖、不产生冲突（较新的约束优先）"""
        n = 0
        for c in items or []:
            if not isinstance(c, dict) or c.get("type") not in CONSTRAINT_TYPES:
                continue
            t = by_idx.get(_int(c.get("turn")))
            value = str(c.get("value") or "").strip()[:60]
            if t is None or not value or _norm(value) not in _norm(t.get("user", "")):
                STM_METRICS.incr("compact_constraint_rejected")
                continue
            scope = "session" if c.get("scope") == "session" else "topic"
            bucket_topic = topic_id if scope == "topic" else None
            if any(x["type"] == c["type"] and x["scope"] == scope and (scope == "session" or x["topic_id"] == bucket_topic)
                   for x in state["constraints"]):
                continue
            status = "active" if scope == "session" or bucket_topic == state.get("current_topic") else "dormant"
            state["constraints"].append({"type": c["type"], "value": value, "scope": scope, "topic_id": bucket_topic,
                                         "status": status, "source_turn": t["idx"], "history": []})
            n += 1
        return n

    async def _compact_once(self, scope: str, user_id: str, session_id: str, rebuild_topic: Optional[str]) -> None:
        async with self._sem:
            state, turns, _ = await asyncio.to_thread(self.memory.load, scope)
            if state is None:
                return
            base_version = int(state.get("version", 0))
            st = ensure_schema(copy.deepcopy(state))
            selected = self._select(st, turns)
            groups: Dict[str, List[dict]] = {}
            for t in selected:
                groups.setdefault(t.get("topic_id") or st["current_topic"] or "T0", []).append(t)
            if rebuild_topic and rebuild_topic not in groups:
                groups[rebuild_topic] = []
            if not groups:
                return

            stats = {"facts": 0, "constraints": 0, "rebuilt": []}
            for topic_id, group in groups.items():
                n_comp = st["compactions"].get(topic_id, 0)
                rebuild = topic_id == rebuild_topic or n_comp + 1 >= self.cfg["rebuild_every"]
                if rebuild:
                    # 用原文重建：PG 中该话题的历史轮次 + Redis 中尚未写入 PG 的轮次，按 idx 去重
                    pg_turns = await load_recent_turns(self._pool(), user_id, session_id, 30, topic_id=topic_id)
                    src = {t["idx"]: t for t in pg_turns}
                    src.update({t["idx"]: t for t in group})
                    source_turns = [src[i] for i in sorted(src)]
                    old_summary = ""
                else:
                    source_turns = group
                    old_summary = (st["summaries"].get(topic_id) or {}).get("text", "")
                if not source_turns:
                    continue
                # 输入过长时只保留最近的轮次（4B 上下文 8192，压缩输入控制在约 3000 token）
                while len(source_turns) > 1 and sum(est_tokens(_turn_text(t)) for t in source_turns) > 3000:
                    source_turns = source_turns[1:]

                obj = await self._call_4b(old_summary, source_turns)
                if obj is None:
                    STM_METRICS.incr("compact_llm_failed")
                    return                                   # 本次放弃，水位线不前移，下次触发重试
                by_idx = {t["idx"]: t for t in source_turns}
                summary = _clip(str(obj.get("summary") or "").strip(), self.cfg["summary_budget"])
                if summary:
                    until = max([t["idx"] for t in source_turns] + [(st["summaries"].get(topic_id) or {}).get("until_turn", 0)])
                    st["summaries"][topic_id] = {"text": summary, "until_turn": until}
                stats["facts"] += apply_facts(st, self._validated_facts(obj.get("facts"), by_idx, topic_id),
                                              self.cfg["max_facts"])
                stats["constraints"] += self._add_missed_constraints(st, obj.get("constraints"), by_idx, topic_id)
                st["compactions"][topic_id] = 0 if rebuild else n_comp + 1
                if rebuild:
                    stats["rebuilt"].append(topic_id)

            if selected:
                st["summarized_until"] = max(st.get("summarized_until", 0), max(t["idx"] for t in selected))
            ok = await asyncio.to_thread(self.memory.cas_state, scope, base_version, st)
            if not ok:
                STM_METRICS.incr("compact_discarded_cas")
                logger.info(f"短期记忆压缩结果因状态版本变化被丢弃，等待下次触发: {scope}")
                return
            STM_METRICS.incr("compact_ok")
            logger.info(f"短期记忆压缩完成 {scope}: 并入 {len(selected)} 轮，事实 +{stats['facts']}，"
                        f"补抽约束 {stats['constraints']}，原文重建话题 {stats['rebuilt']}")
            await save_state(self._pool(), user_id, session_id, st)
