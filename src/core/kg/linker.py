"""
实体链接（Entity Linking）：把问题 / 工具参数中的名称对齐到知识图谱实体

标准三段式（mention detection → candidate generation → disambiguation）：
    1. 找提及：Aho-Corasick 自动机一次扫描匹配全部名称 / 别名，取最长且不重叠的匹配
         - 强名称 / 别名忽略大小写，按英文词边界匹配（容忍复数 s / es）
         - 弱别名（Apple、Mac、NeXT、metal、UK 等常见词或缩写）只按原大小写匹配
         - 中文名在中文文本中匹配（≥2 字）
    2. 候选：一个名称可能对应多个实体（同名），候选来自名称表
    3. 消歧：优先选与其他已链接实体在图上 3 跳内相连的候选，其次按 mention_count
       预取模式下，仅靠弱别名命中的实体必须与另一个已链接实体相连才采用（宁缺毋滥）
    工具模式（Agent / MCP 传入实体名）精确匹配不到时，用 Neo4j 全文索引 entity_names 兜底

词典从 Neo4j 加载；每隔 reload_interval_s 检查一次 :KGVersion，图谱重新入库后自动重载，无需重启服务。

Author: Ke Meng
Created: 2026-10-18
"""

import logging
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import ahocorasick

logger = logging.getLogger(__name__)

_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD_CHAR = re.compile(r"[A-Za-z0-9]")
_LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')


@dataclass
class Link:
    """一个链接结果：文本中的提及 → 图谱实体"""
    mention: str
    entity_id: str
    name: str
    name_zh: str
    cls: str
    match: str            # name / alias / weak / zh / fulltext

    def to_dict(self) -> dict:
        return {"mention": self.mention, "entity_id": self.entity_id, "name": self.name,
                "name_zh": self.name_zh, "class": self.cls, "match": self.match}


@dataclass
class _Lexicon:
    """一次加载的词典快照（重载时整体替换，读写无需加锁）"""
    version: Optional[str] = None
    entities: Dict[str, dict] = field(default_factory=dict)
    ci: Optional[ahocorasick.Automaton] = None     # 小写强名称 → [(entity_id, match)]
    cs: Optional[ahocorasick.Automaton] = None     # 原大小写弱别名 → [(entity_id, "weak")]
    zh: Optional[ahocorasick.Automaton] = None     # 中文名 → [(entity_id, "zh")]
    exact_ci: Dict[str, List[tuple]] = field(default_factory=dict)
    exact_cs: Dict[str, List[tuple]] = field(default_factory=dict)
    exact_zh: Dict[str, List[tuple]] = field(default_factory=dict)


def _clean(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def _boundary_ok(text: str, start: int, end: int) -> Optional[int]:
    """英文词边界检查（只看 ASCII 字母数字，中文紧邻英文名也算边界）；返回含复数后缀的结束位置，不满足返回 None"""
    if start > 0 and _WORD_CHAR.match(text[start - 1]):
        return None
    for suffix in ("", "s", "es"):
        e = end + len(suffix)
        if text[end:e].lower() == suffix and (e >= len(text) or not _WORD_CHAR.match(text[e])):
            return e
    return None


def _build(words: Dict[str, List[tuple]]) -> Optional[ahocorasick.Automaton]:
    if not words:
        return None
    a = ahocorasick.Automaton()
    for w, payload in words.items():
        a.add_word(w, (w, payload))
    a.make_automaton()
    return a


class EntityLinker:
    def __init__(self, run_read: Callable[..., List[dict]], reload_interval_s: int = 300,
                 fulltext_min_score: float = 1.0, max_hops: int = 3):
        """
        :param run_read: 只读查询函数 run_read(cypher, **params) -> List[dict]（由 RelationVerifier 提供，共用连接与超时）
        """
        self._run = run_read
        self.reload_interval_s = reload_interval_s
        self.fulltext_min_score = fulltext_min_score
        self.max_hops = max_hops
        self._lex = _Lexicon()
        self._checked_at = 0.0
        self._lock = threading.Lock()

    # ==================== 词典加载 ====================
    def _current_version(self) -> Optional[str]:
        rows = self._run("MATCH (k:KGVersion) RETURN max(k.built_at) AS v")
        return rows[0]["v"] if rows else None

    def _load(self, version: Optional[str]) -> _Lexicon:
        rows = self._run("""
            MATCH (n:Entity)
            RETURN n.entity_id AS id, n.name AS name, n.name_zh AS name_zh, n.class AS cls,
                   coalesce(n.aliases, []) AS aliases, coalesce(n.weak_aliases, []) AS weak_aliases,
                   coalesce(n.name_weak, false) AS name_weak, coalesce(n.mention_count, 0) AS mentions""")
        lex = _Lexicon(version=version)
        ci, cs, zh = defaultdict(list), defaultdict(list), defaultdict(list)
        for r in rows:
            eid = r["id"]
            lex.entities[eid] = r
            forms = [(r["name"], "name", r["name_weak"])] + [(a, "alias", False) for a in r["aliases"]] \
                + [(a, "weak", True) for a in r["weak_aliases"]]
            for form, kind, weak in forms:
                form = _clean(form or "")
                if not form:
                    continue
                if weak:
                    cs[form].append((eid, "weak"))
                else:
                    ci[form.lower()].append((eid, kind))
            if r["name_zh"] and len(r["name_zh"]) >= 2:
                zh[_clean(r["name_zh"])].append((eid, "zh"))
        lex.ci, lex.cs, lex.zh = _build(ci), _build(cs), _build(zh)
        lex.exact_ci, lex.exact_cs, lex.exact_zh = dict(ci), dict(cs), dict(zh)
        logger.info(f"📚 实体词典已加载：版本 {version}，实体 {len(rows)}，"
                    f"强名称 {len(ci)}、弱别名 {len(cs)}、中文名 {len(zh)}")
        return lex

    def ensure_loaded(self) -> bool:
        """首次调用加载；之后每隔 reload_interval_s 检查一次图谱版本，有新版本则重载。返回词典是否可用"""
        now = time.monotonic()
        if self._lex.entities and now - self._checked_at < self.reload_interval_s:
            return True
        with self._lock:
            if self._lex.entities and now - self._checked_at < self.reload_interval_s:
                return True
            try:
                version = self._current_version()
                if version != self._lex.version or not self._lex.entities:
                    self._lex = self._load(version)
            except Exception as e:
                logger.warning(f"实体词典加载 / 版本检查失败，沿用现有词典: {e}")
            self._checked_at = now
        return bool(self._lex.entities)

    @property
    def version(self) -> Optional[str]:
        return self._lex.version

    # ==================== 找提及 ====================
    @staticmethod
    def _scan(automaton, text: str, haystack: str, english: bool) -> List[tuple]:
        """返回 [(start, end, surface, candidates)]"""
        if automaton is None or not text:
            return []
        hits = []
        for end_idx, (word, cands) in automaton.iter(haystack):
            start = end_idx - len(word) + 1
            end = end_idx + 1
            if english:
                end = _boundary_ok(text, start, end)
                if end is None:
                    continue
            hits.append((start, end, text[start:end], cands))
        return hits

    @staticmethod
    def _longest_non_overlapping(hits: List[tuple]) -> List[tuple]:
        chosen, used = [], set()
        for h in sorted(hits, key=lambda h: (-(h[1] - h[0]), h[0])):
            span = set(range(h[0], h[1]))
            if span & used:
                continue
            used |= span
            chosen.append(h)
        return sorted(chosen, key=lambda h: h[0])

    def _mentions(self, lex: _Lexicon, texts: List[str]) -> List[tuple]:
        """在多段文本中找提及；返回 [(surface, candidates)]，候选为 [(entity_id, match)]"""
        out = []
        for text in texts:
            text = _clean(text or "")
            if not text:
                continue
            if _CJK.search(text):
                hits = self._scan(lex.zh, text, text, english=False)
                # 中文文本里夹带的英文名（如"NeXT 公司"）也要匹配
                hits += self._scan(lex.ci, text, text.lower(), english=True) if len(text.lower()) == len(text) else []
                hits += self._scan(lex.cs, text, text, english=True)
            else:
                lowered = text.lower()
                hits = self._scan(lex.ci, text, lowered, english=True) if len(lowered) == len(text) else []
                hits += self._scan(lex.cs, text, text, english=True)
            out += [(h[2], h[3]) for h in self._longest_non_overlapping(hits)]
        return out

    # ==================== 消歧 ====================
    def _connected(self, ids: List[str]) -> Set[frozenset]:
        """返回在图上 max_hops 跳内相连的实体对（不经过 MENTIONS / 原文 chunk）"""
        if len(ids) < 2:
            return set()
        rows = self._run(f"""
            MATCH (a:Entity), (b:Entity)
            WHERE a.entity_id IN $ids AND b.entity_id IN $ids AND a.entity_id < b.entity_id
            MATCH p = SHORTEST 1 (a)-[:!MENTIONS]-{{1,{int(self.max_hops)}}}(b)
            RETURN a.entity_id AS a, b.entity_id AS b""", ids=ids)
        return {frozenset((r["a"], r["b"])) for r in rows}

    def _resolve(self, lex: _Lexicon, mentions: List[tuple], mode: str) -> List[Link]:
        """mentions: [(surface, candidates)]；mode = prefetch / tool"""
        if not mentions:
            return []
        all_ids = sorted({eid for _, cands in mentions for eid, _ in cands})
        try:
            connected = self._connected(all_ids)
        except Exception as e:
            logger.warning(f"消歧连通性查询失败，退化为按提及次数选择: {e}")
            connected = set()

        def linked_to_others(eid, own_ids):
            return any(frozenset((eid, o)) in connected for o in all_ids if o not in own_ids)

        links, seen = [], set()
        for surface, cands in mentions:
            own = {eid for eid, _ in cands}
            best = max(cands, key=lambda c: (linked_to_others(c[0], own), lex.entities[c[0]]["mentions"]))
            eid, match = best
            if mode == "prefetch" and match == "weak" and not linked_to_others(eid, own):
                logger.info(f"🔗 弱别名 '{surface}' → {eid} 未与其他实体相连，预取中不采用")
                continue
            if eid in seen:
                continue
            seen.add(eid)
            e = lex.entities[eid]
            links.append(Link(surface, eid, e["name"], e["name_zh"] or "", e["cls"], match))
        return links

    # ==================== 对外接口 ====================
    def link_text(self, texts: List[str]) -> List[Link]:
        """预取模式：在问题文本（英文改写 + 中文原问题）中找出图谱实体"""
        if not self.ensure_loaded():
            return []
        lex = self._lex
        return self._resolve(lex, self._mentions(lex, texts), mode="prefetch")

    def link_names(self, names: List[str]) -> tuple:
        """
        工具模式：逐个对齐调用方给出的实体名。
        返回 (links, not_in_graph)：精确匹配 → 文本内匹配 → 全文索引兜底，均失败视为图谱未收录
        """
        if not self.ensure_loaded():
            return [], list(names)
        lex = self._lex
        mentions, missing = [], []
        for raw in names:
            name = _clean(raw or "")
            if not name:
                continue
            cands = lex.exact_cs.get(name) or lex.exact_ci.get(name.lower()) or lex.exact_zh.get(name)
            if cands:
                mentions.append((raw, cands))
                continue
            found = self._mentions(lex, [name])
            if found:
                mentions.append((raw, max(found, key=lambda m: len(m[0]))[1]))
                continue
            ft = self._fulltext(name)
            if ft:
                mentions.append((raw, [(ft, "fulltext")]))
            else:
                missing.append(raw)
        return self._resolve(lex, mentions, mode="tool"), missing

    def canonical(self, name: str) -> Optional[dict]:
        """
        短期记忆实体表归一用：名称（英文名 / 别名 / 中文名）精确匹配到图谱实体，返回 {name, kg_id, type}。
        只查已加载的内存词典，不访问 Neo4j（在请求路径上同步调用，不能阻塞）；词典未加载时返回 None。
        """
        lex = self._lex
        n = _clean(name or "")
        if not lex.entities or not n:
            return None
        cands = lex.exact_cs.get(n) or lex.exact_ci.get(n.lower()) or lex.exact_zh.get(n)
        if not cands:
            return None
        eid = max(cands, key=lambda c: lex.entities[c[0]]["mentions"])[0]
        e = lex.entities[eid]
        return {"name": e["name"], "kg_id": eid, "type": (e["cls"] or "other").lower()}

    def _fulltext(self, name: str) -> Optional[str]:
        q = _LUCENE_SPECIAL.sub(r"\\\1", name)
        try:
            rows = self._run("""
                CALL db.index.fulltext.queryNodes('entity_names', $q) YIELD node, score
                WHERE score >= $min_score
                RETURN node.entity_id AS id, score ORDER BY score DESC LIMIT 1""",
                             q=q, min_score=self.fulltext_min_score)
        except Exception as e:
            logger.warning(f"全文索引查询失败: {e}")
            return None
        if rows and rows[0]["id"] in self._lex.entities:
            logger.info(f"🔎 全文索引兜底：'{name}' → {rows[0]['id']}（score={rows[0]['score']:.2f}）")
            return rows[0]["id"]
        return None
