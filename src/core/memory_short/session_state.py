"""
会话状态 (Session State) —— 短期记忆的结构化部分，纯逻辑、无 IO

设计文档：docs/short_term_memory_design.md §5–§6、§8.1

状态是一个可 JSON 序列化的 dict，Redis（热）与 PG session_state（冷）存同一份：
    constraints   约束（锚定）：只来自用户发言；scope = session / topic（turn 级不入库）；status = active / dormant
    conflicts     意图模糊的同类型约束修改：不覆盖旧值，等待用户确认；N 轮未确认自动失效
    topics        话题：动态产生；最多保留 max_topics 个完整摘要，更早的归档为一句话
    summaries     按话题存储的摘要（已离开窗口部分的概述）
    entities      实体表：知识库已知实体归一为标准名（kg_id），其余为 4B 抽取
    facts         事实三元组 (s, r, o)，带来源轮次；同 (s, r) 新值覆盖旧值
    outlines      近几轮回答的提纲（编号 + 标题），解决"你前面列的第二个"这类指代
    summarized_until  已并入摘要的最大轮次（水位线）；Agent 上下文只放其后的原文

所有 apply_* 函数返回新状态（深拷贝），不修改入参，便于 CAS 冲突时在最新状态上重放。
"""

import copy
import hashlib
import json
import re
from typing import Callable, Dict, List, Optional, Tuple

SCHEMA_VERSION = 1

CONSTRAINT_TYPES = ("time_range", "object_scope", "region", "language", "format", "audience", "other")
CONSTRAINT_TYPE_ZH = {
    "time_range": "时间范围", "object_scope": "对象范围", "region": "地域", "language": "语言",
    "format": "格式与长度", "audience": "受众", "other": "其他要求",
}
SCOPES = ("session", "topic", "turn")
SCOPE_ZH = {"session": "整个会话", "topic": "当前话题", "turn": "仅本问"}

_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)
# 用户只是确认助手的提议（"要不要只看近三年？""好"）：约束值出现在上一轮助手回答中即可
_AFFIRM_RE = re.compile(r"^(好|好的|可以|行|嗯|对|是的|是|没问题|就这样|ok|okay|yes|sure)[!！。.~～]*$", re.I)
# 确认待定的约束变更：以肯定词开头（其后为结束或标点），如"好"、"好的，改吧"、"是的。那它的营收呢"
_CONFIRM_RE = re.compile(r"^(好的|好|可以|行|嗯|对|是的|是|没问题|就这样|改吧|确认|okay|ok|yes|sure)(?:$|[\s,，.。!！~～;；:：])", re.I)


def _norm(text: str) -> str:
    return _PUNCT_RE.sub("", (text or "").lower())


def new_state() -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "version": 0,
        "current_topic": None,
        "topic_seq": 0,
        "topics": [],              # [{id, label, last_turn, archived}]
        "constraints": [],         # [{type, value, scope, topic_id, status, source_turn, history}]
        "conflicts": [],           # [{type, value, old_value, scope, topic_id, turn}]
        "summaries": {},           # {topic_id: {text, until_turn}}
        "entities": [],            # [{name, type, aliases, last_turn, kg_id}]
        "facts": [],               # [{s, r, o, turns, turn_ids, last_turn, topic_id}]
        "outlines": [],            # [{turn, topic_id, items}]
        "summarized_until": 0,
        "summary_seen_until": 0,   # 前端已提示过的摘要水位（压缩后的下一轮提示一次）
        "compactions": {},        # {topic_id: 距上次从原文重建以来的压缩次数}
    }


def ensure_schema(state: Optional[dict]) -> dict:
    """补齐缺失字段（旧版本状态 / 部分损坏的状态）；非 dict 视为新会话"""
    base = new_state()
    if not isinstance(state, dict):
        return base
    for k, v in base.items():
        if k not in state or not isinstance(state[k], type(v)) and v is not None:
            state[k] = v
    return state


# ==================== 话题 ====================

def _topic(state: dict, topic_id: Optional[str]) -> Optional[dict]:
    return next((t for t in state["topics"] if t["id"] == topic_id), None)


def _new_topic(state: dict, label: str, turn: int) -> str:
    state["topic_seq"] += 1
    tid = f"T{state['topic_seq']}"
    state["topics"].append({"id": tid, "label": (label or "未命名话题")[:30], "last_turn": turn,
                            "archived": False})
    return tid


def _set_topic_constraints_status(state: dict, topic_id: Optional[str], src: str, dst: str) -> None:
    if not topic_id:
        return
    for c in state["constraints"]:
        if c["scope"] == "topic" and c["topic_id"] == topic_id and c["status"] == src:
            c["status"] = dst


def _first_sentence(text: str, limit: int) -> str:
    text = (text or "").strip()
    m = re.search(r"[。！？!?\n]|\.\s", text)
    s = text[: m.start() + 1] if m else text
    return s[:limit]


def _archive_excess_topics(state: dict, max_topics: int) -> List[str]:
    """超过 max_topics 的话题（按最近活跃排序）归档：删除摘要，其话题级约束撤销"""
    active = [t for t in state["topics"] if not t["archived"]]
    if len(active) <= max_topics:
        return []
    active.sort(key=lambda t: t["last_turn"], reverse=True)
    archived = []
    for t in active[max_topics:]:
        if t["id"] == state["current_topic"]:
            continue
        state["summaries"].pop(t["id"], None)
        t["archived"] = True
        state["constraints"] = [c for c in state["constraints"]
                                if not (c["scope"] == "topic" and c["topic_id"] == t["id"])]
        state["conflicts"] = [c for c in state["conflicts"] if c.get("topic_id") != t["id"]]
        archived.append(t["id"])
    # 归档话题只保留标签用于"回到很久前的话题"时从 PG 重建；最多留 20 个
    archived_all = [t for t in state["topics"] if t["archived"]]
    if len(archived_all) > 20:
        drop = {t["id"] for t in sorted(archived_all, key=lambda t: t["last_turn"])[: len(archived_all) - 20]}
        state["topics"] = [t for t in state["topics"] if t["id"] not in drop]
    return archived


def apply_topic(state: dict, topic: Optional[dict], turn: int, fallback_label: str, max_topics: int) -> dict:
    """
    处理改写器给出的话题判断；返回事件 {shift, topic_id, reactivated, archived}
    topic = {"shift": continue / switch / return, "return_to": "T1", "label": "..."}
    """
    topic = topic or {}
    shift = topic.get("shift") if topic.get("shift") in ("continue", "switch", "return") else "continue"
    label = (topic.get("label") or "").strip() or fallback_label
    prev = state["current_topic"]
    event = {"shift": shift, "topic_id": None, "reactivated": False, "archived": []}

    if prev is None or _topic(state, prev) is None:
        tid = _new_topic(state, label, turn)
        state["current_topic"] = tid
        event.update(shift="new", topic_id=tid)
        return event

    if shift == "return":
        target = _topic(state, topic.get("return_to"))
        if target is None or target["id"] == prev:
            shift = "continue" if target is not None else "switch"
            event["shift"] = shift
        else:
            _set_topic_constraints_status(state, prev, "active", "dormant")
            _set_topic_constraints_status(state, target["id"], "dormant", "active")
            target["last_turn"] = turn
            if target["archived"]:
                target["archived"] = False       # 摘要由压缩任务按需从 PG 原文重建
            state["current_topic"] = target["id"]
            event.update(topic_id=target["id"], reactivated=True)
            event["archived"] = _archive_excess_topics(state, max_topics)
            return event

    if shift == "switch":
        _set_topic_constraints_status(state, prev, "active", "dormant")
        tid = _new_topic(state, label, turn)
        state["current_topic"] = tid
        event["topic_id"] = tid
        event["archived"] = _archive_excess_topics(state, max_topics)
        return event

    _topic(state, prev)["last_turn"] = turn
    event["topic_id"] = prev
    return event


# ==================== 约束 ====================

def _grounded(value: str, user_text: str, last_assistant: str) -> bool:
    """约束只能来自用户发言：值须出现在用户原话中；或用户只是确认助手的提议，且值出现在上一轮助手回答中"""
    nv = _norm(value)
    if not nv:
        return False
    if nv in _norm(user_text):
        return True
    return bool(_AFFIRM_RE.match((user_text or "").strip())) and nv in _norm(last_assistant)


def _bucket_match(c: dict, ctype: str, scope: str, topic_id: Optional[str]) -> bool:
    return (c["type"] == ctype and c["scope"] == scope
            and (scope == "session" or c["topic_id"] == topic_id))


def _evict_constraints(state: dict, max_total: int, max_other: int) -> None:
    others = [c for c in state["constraints"] if c["type"] == "other"]
    if len(others) > max_other:
        drop = sorted(others, key=lambda c: c["source_turn"])[: len(others) - max_other]
        state["constraints"] = [c for c in state["constraints"] if c not in drop]
    if len(state["constraints"]) <= max_total:
        return
    last_active = {t["id"]: t["last_turn"] for t in state["topics"]}
    # 先淘汰最久未回访话题的休眠约束；会话级 active 永不淘汰
    dormant = sorted([c for c in state["constraints"] if c["status"] == "dormant"],
                     key=lambda c: (last_active.get(c["topic_id"], -1), c["source_turn"]))
    while len(state["constraints"]) > max_total and dormant:
        state["constraints"].remove(dormant.pop(0))


def _confirm_conflicts(state: dict, delta: List[dict], user_text: str, last_assistant: str, event: dict) -> None:
    """
    用户确认待定的约束变更 → 新值覆盖旧值。须以肯定词开头，且满足其一：
      改写器给出了同类型、同值的 set（改写器能看到待确认变更）；或上一轮助手回答中提到了新值（助手按要求先向用户确认）
    否认或不回应时不处理，冲突按 TTL 失效、旧值保留。
    """
    if not state["conflicts"] or not _CONFIRM_RE.match((user_text or "").strip()):
        return
    topic_id = state["current_topic"]
    for cf in list(pending_conflicts(state)):
        nv = _norm(cf["value"])
        proposed = any(isinstance(d, dict) and d.get("op", "set") == "set" and d.get("type") == cf["type"]
                       and _norm(str(d.get("value") or "")) == nv for d in delta or [])
        if not (proposed or nv in _norm(last_assistant)):
            continue
        bucket_topic = topic_id if cf["scope"] == "topic" else None
        existing = next((c for c in state["constraints"]
                         if c["status"] == "active" and _bucket_match(c, cf["type"], cf["scope"], bucket_topic)), None)
        state["conflicts"].remove(cf)
        if existing is None:          # 旧约束已不存在（如已撤销）：确认即新设
            state["constraints"].append({"type": cf["type"], "value": cf["value"], "scope": cf["scope"],
                                         "topic_id": bucket_topic, "status": "active", "source_turn": cf["turn"],
                                         "history": []})
        else:
            existing["history"] = (existing.get("history") or [])[-4:] + [{"value": existing["value"],
                                                                             "turn": existing["source_turn"]}]
            existing["value"], existing["source_turn"] = cf["value"], cf["turn"]
        event["set"].append({"type": cf["type"], "value": cf["value"], "scope": cf["scope"], "confirmed": True})


def apply_constraints(state: dict, delta: List[dict], turn: int, user_text: str, last_assistant: str,
                      max_total: int, max_other: int, conflict_ttl: int) -> Tuple[List[dict], dict]:
    """
    应用改写器给出的约束变更；返回 (本问有效的 turn 级约束, 事件)
    delta 项：{"op": set / revoke, "type", "value", "scope": session / topic / turn, "explicit": bool}
    """
    topic_id = state["current_topic"]
    turn_constraints: List[dict] = []
    event = {"set": [], "revoked": [], "conflict": [], "rejected": []}

    # 过期冲突：N 轮未被确认
    state["conflicts"] = [c for c in state["conflicts"] if turn - c["turn"] < conflict_ttl]
    _confirm_conflicts(state, delta, user_text, last_assistant, event)

    for d in delta or []:
        if not isinstance(d, dict):
            continue
        ctype, op = d.get("type"), d.get("op", "set")
        scope = d.get("scope") if d.get("scope") in SCOPES else "topic"      # 拿不准作用范围时按话题级
        value = str(d.get("value") or "").strip()[:60]
        if ctype not in CONSTRAINT_TYPES:
            event["rejected"].append({"reason": "type", "type": ctype})
            continue

        if op == "revoke":
            hit = [c for c in state["constraints"] if c["status"] == "active" and c["type"] == ctype
                   and (c["scope"] == "session" or c["topic_id"] == topic_id)
                   and (ctype != "other" or not value or _norm(value) in _norm(c["value"]))]
            for c in hit:
                state["constraints"].remove(c)
            state["conflicts"] = [c for c in state["conflicts"] if c["type"] != ctype]
            event["revoked"] += [{"type": c["type"], "value": c["value"]} for c in hit]
            continue

        if not _grounded(value, user_text, last_assistant):
            event["rejected"].append({"reason": "not_from_user", "type": ctype, "value": value})
            continue
        if scope == "turn":
            turn_constraints.append({"type": ctype, "value": value, "scope": "turn"})
            continue

        bucket_topic = topic_id if scope == "topic" else None
        if ctype == "other":
            if not any(_norm(c["value"]) == _norm(value) for c in state["constraints"]
                       if c["type"] == "other" and c["status"] == "active"):
                state["constraints"].append({"type": ctype, "value": value, "scope": scope, "topic_id": bucket_topic,
                                             "status": "active", "source_turn": turn, "history": []})
                event["set"].append({"type": ctype, "value": value, "scope": scope})
            continue

        existing = next((c for c in state["constraints"]
                         if c["status"] == "active" and _bucket_match(c, ctype, scope, bucket_topic)), None)
        if existing is None:
            state["constraints"].append({"type": ctype, "value": value, "scope": scope, "topic_id": bucket_topic,
                                         "status": "active", "source_turn": turn, "history": []})
            state["conflicts"] = [c for c in state["conflicts"] if not (c["type"] == ctype and c["scope"] == scope)]
            event["set"].append({"type": ctype, "value": value, "scope": scope})
        elif _norm(existing["value"]) == _norm(value):
            existing["source_turn"] = turn
        elif d.get("explicit"):
            existing["history"] = (existing.get("history") or [])[-4:] + [{"value": existing["value"],
                                                                             "turn": existing["source_turn"]}]
            existing["value"], existing["source_turn"] = value, turn
            state["conflicts"] = [c for c in state["conflicts"] if not (c["type"] == ctype and c["scope"] == scope)]
            event["set"].append({"type": ctype, "value": value, "scope": scope, "replaced": True})
        else:
            # 意图模糊：不覆盖，记冲突待确认
            state["conflicts"] = [c for c in state["conflicts"] if not (c["type"] == ctype and c["scope"] == scope)]
            state["conflicts"].append({"type": ctype, "value": value, "old_value": existing["value"],
                                       "scope": scope, "topic_id": bucket_topic, "turn": turn})
            event["conflict"].append({"type": ctype, "old": existing["value"], "new": value})

    _evict_constraints(state, max_total, max_other)
    return turn_constraints, event


def effective_constraints(state: dict, turn_constraints: Optional[List[dict]] = None) -> List[dict]:
    """
    本问实际生效的约束：会话级 + 当前话题的 active + turn 级；同类型优先级 turn > topic > session。
    "other" 类不互相覆盖。返回 [{type, value, scope}]，按类型稳定排序。
    """
    topic_id = state.get("current_topic")
    rank = {"session": 0, "topic": 1, "turn": 2}
    chosen: Dict[str, dict] = {}
    others: List[dict] = []
    pool = [c for c in state.get("constraints", []) if c["status"] == "active"
            and (c["scope"] == "session" or c["topic_id"] == topic_id)] + list(turn_constraints or [])
    for c in pool:
        item = {"type": c["type"], "value": c["value"], "scope": c["scope"]}
        if c["type"] == "other":
            if all(_norm(o["value"]) != _norm(c["value"]) for o in others):
                others.append(item)
        elif c["type"] not in chosen or rank[c["scope"]] >= rank[chosen[c["type"]]["scope"]]:
            chosen[c["type"]] = item
    order = {t: i for i, t in enumerate(CONSTRAINT_TYPES)}
    return sorted(chosen.values(), key=lambda c: order[c["type"]]) + others


def pending_conflicts(state: dict) -> List[dict]:
    topic_id = state.get("current_topic")
    return [c for c in state.get("conflicts", []) if c["scope"] == "session" or c.get("topic_id") == topic_id]


def constraint_hash(effective: List[dict]) -> str:
    """影响答案的约束指纹：加入缓存键，约束不同不会命中同一缓存；无约束时为空串"""
    if not effective:
        return ""
    canon = sorted((c["type"], _norm(c["value"])) for c in effective)
    return hashlib.sha1(json.dumps(canon, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


# ==================== 实体 ====================

def apply_entities(state: dict, entities: List[dict], turn: int, grounding_texts: List[str], max_entities: int,
                   canonicalize: Optional[Callable[[str], Optional[dict]]] = None) -> List[str]:
    """
    合并本轮实体。只接受出现在用户原话 / 本轮改写结果中，或实体表已有的名称（防止 4B 编造）。
    canonicalize(name) → {"name", "kg_id", "type"} 或 None：知识库已知实体归一为标准名。
    返回本轮被更新的实体名。
    """
    blob = _norm(" ".join(grounding_texts))
    known = {_norm(e["name"]): e for e in state["entities"]}
    for e in state["entities"]:
        for a in e.get("aliases") or []:
            known.setdefault(_norm(a), e)
    touched = []
    for item in entities or []:
        if not isinstance(item, dict):
            continue
        surface = str(item.get("name") or "").strip()[:60]
        n = _norm(surface)
        if not n or (n not in blob and n not in known):
            continue
        canon = canonicalize(surface) if canonicalize else None
        name = (canon or {}).get("name") or surface
        kg_id = (canon or {}).get("kg_id")
        etype = (canon or {}).get("type") or str(item.get("type") or "other")[:20]
        target = None
        if kg_id:
            target = next((e for e in state["entities"] if e.get("kg_id") == kg_id), None)
        target = target or known.get(_norm(name)) or known.get(n)
        if target is None:
            target = {"name": name, "type": etype, "aliases": [], "last_turn": turn, "kg_id": kg_id}
            state["entities"].append(target)
            known[_norm(name)] = target
        target["last_turn"] = turn
        if kg_id and not target.get("kg_id"):
            target["kg_id"] = kg_id
        if _norm(surface) != _norm(target["name"]) and surface not in target["aliases"]:
            target["aliases"] = (target["aliases"] + [surface])[-5:]
            known[n] = target
        touched.append(target["name"])
    if len(state["entities"]) > max_entities:
        state["entities"].sort(key=lambda e: e["last_turn"], reverse=True)
        del state["entities"][max_entities:]
    return touched


# ==================== 事实三元组 ====================

def apply_facts(state: dict, facts: List[dict], max_facts: int) -> int:
    """
    合并（已由调用方做过原文证据校验的）三元组；同 (s, r) 新值覆盖旧值（时效）。返回写入条数。
    fact 项：{"s", "r", "o", "turn", "turn_id", "topic_id"}
    """
    n = 0
    for f in facts or []:
        s, r, o = (str(f.get(k) or "").strip()[:60] for k in ("s", "r", "o"))
        if not (s and r and o):
            continue
        turn = int(f.get("turn") or 0)
        cur = next((x for x in state["facts"] if _norm(x["s"]) == _norm(s) and _norm(x["r"]) == _norm(r)), None)
        if cur is None:
            cur = {"s": s, "r": r, "o": o, "turns": [], "turn_ids": [], "last_turn": turn,
                   "topic_id": f.get("topic_id")}
            state["facts"].append(cur)
        elif turn < cur["last_turn"]:
            continue                      # 旧轮次的信息不覆盖更新的事实
        cur["o"], cur["last_turn"] = o, max(cur["last_turn"], turn)
        if turn and turn not in cur["turns"]:
            cur["turns"] = (cur["turns"] + [turn])[-5:]
        if f.get("turn_id") and f["turn_id"] not in cur["turn_ids"]:
            cur["turn_ids"] = (cur["turn_ids"] + [f["turn_id"]])[-5:]
        n += 1
    if len(state["facts"]) > max_facts:
        state["facts"].sort(key=lambda x: x["last_turn"], reverse=True)
        del state["facts"][max_facts:]
    return n


# ==================== 回答提纲 ====================

_LIST_ITEM_RE = re.compile(r"^\s*(?:\d{1,2}[.、)）]|[-*•·]|（\d{1,2}）|\(\d{1,2}\)|[一二三四五六七八九十]{1,2}、)\s*")


def outline_items(answer: str, max_items: int = 8, item_chars: int = 40) -> List[str]:
    """回答中的列表项（编号 + 首句）；无列表的回答返回空（只有列表才会被"第 N 个"指代）"""
    items = []
    for line in (answer or "").splitlines():
        m = _LIST_ITEM_RE.match(line)
        if not m:
            continue
        body = _first_sentence(line[m.end():], item_chars)
        if body:
            items.append(f"{len(items) + 1}. {body}")
        if len(items) >= max_items:
            break
    return items if len(items) >= 2 else []


def apply_outline(state: dict, turn: int, answer: str, max_outlines: int) -> bool:
    items = outline_items(answer)
    if not items:
        return False
    state["outlines"] = [o for o in state["outlines"] if o["turn"] != turn]
    state["outlines"].append({"turn": turn, "topic_id": state["current_topic"], "items": items})
    state["outlines"] = sorted(state["outlines"], key=lambda o: o["turn"])[-max_outlines:]
    return True


# ==================== 组合：改写结果落到状态 ====================

def apply_rewrite(state: dict, rw: dict, turn: int, user_text: str, last_assistant: str, cfg: dict,
                  canonicalize: Optional[Callable[[str], Optional[dict]]] = None) -> Tuple[dict, List[dict], dict]:
    """
    把改写器的结构化输出应用到状态（深拷贝后修改）。
    返回 (新状态, 本问 turn 级约束, 事件)。改写失败（rw["ok"] 为 False）时只推进话题，不改约束与实体。
    """
    st = ensure_schema(copy.deepcopy(state))
    events = {"topic": apply_topic(st, rw.get("topic"), turn, _first_sentence(user_text, 20), cfg["max_topics"])}
    turn_constraints: List[dict] = []
    if rw.get("ok"):
        turn_constraints, events["constraints"] = apply_constraints(
            st, rw.get("constraints_delta") or [], turn, user_text, last_assistant,
            cfg["max_constraints"], cfg["max_other_constraints"], cfg["conflict_ttl_turns"])
        events["entities"] = apply_entities(
            st, rw.get("entities") or [], turn,
            [user_text, rw.get("en_query") or "", rw.get("zh_query") or ""], cfg["max_entities"], canonicalize)
    return st, turn_constraints, events


def merge_turn_effects(fresh: dict, rw: dict, turn: int, user_text: str, last_assistant: str, answer: str,
                       cfg: dict, canonicalize: Optional[Callable[[str], Optional[dict]]] = None) -> dict:
    """
    在最新状态上重放本轮的全部变更（CAS 冲突重试时使用）：话题 + 约束 + 实体 + 回答提纲。
    重放是确定性的，可安全重复执行。
    """
    st, _, _ = apply_rewrite(fresh, rw, turn, user_text, last_assistant, cfg, canonicalize)
    apply_outline(st, turn, answer, cfg["max_outlines"])
    return st
