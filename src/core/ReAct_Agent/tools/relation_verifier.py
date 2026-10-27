"""
ReAct Agent 实体关系验证工具集 (Relation Verifier Tool)

在 Neo4j 知识图谱（ontology/wow_ontology.ttl 定义的本体，scripts/kg_build.py 构建、producer_kg.py 入库）上
查询实体之间的关系与多跳路径，返回带知识库原文证据的事实，供 Agent 作答与核验。

两种使用方式（共用同一实例与连接）：
    - Agent 工具 relation_verifier：推理中发现新的桥接实体、需要核实关系时由 Agent 调用
    - 代码预取 prefetch()：multi_hop 题在 Agent 开始前由 main.py 调用，结果写入系统提示词
    另有 /tools/verify（MCP verify_relations）直调 verify()

查询模式：
    - 1 个实体：邻域展开，返回直接相连的事实（按独立来源数排序）
    - ≥2 个实体：两两查 max_hops 跳内的最短路径（不经过 MENTIONS / 原文 chunk），
      途经枢纽实体的路径降权，每对最多 max_paths_per_pair 条；均无路径时退化为各自的邻域
    推理补出的父关系（inferred，如 directedBy → createdBy）不返回，避免重复

全部查询为参数化的只读 Cypher 模板（不让模型生成 Cypher），单次查询有超时；Neo4j 不可用时返回空结果，不影响主流程。


Author: Ke Meng
Created: 2026-01-20
Version: 2.0.0
Last Modified: 2026-10-18

变更记录：
    - 2.0.0 (2026-10-18): 按新图谱重构（本体 v1.2.0）：
                          实体对齐改为 EntityLinker（词典 + 消歧 + 全文索引兜底），支持中文名；
                          由"实体两两直连关系"改为"单实体邻域 / 多实体多跳路径"；
                          输出改为 linked / not_in_graph / facts / paths，事实带原文证据与来源 chunk_id；
                          去掉置信度参数（新图谱为标准答案抽取，用独立来源数 support 表示支撑度）；
                          新增 prefetch() 与 render()，供 multi_hop 预取；连接改用官方 neo4j 驱动只读会话。
    - 1.0.2 (2026-10-15): 新增 create_relation_verifier() 返回验证器实例，供 /tools/verify（MCP 服务）与 Agent 工具共用
    - 1.0.1 (2026-03-15): 工具说明移入 description；结构化输出增加关系类型、置信度、证据原文
    - 1.0.0 (2026-01-28): 初始版本

依赖：
    - neo4j: 官方驱动
    - langchain-core: 用于工具创建
"""

import itertools
import logging
import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from neo4j import GraphDatabase, Query, READ_ACCESS
from pydantic import BaseModel, Field

from src.core.kg.linker import EntityLinker, Link

logger = logging.getLogger(__name__)

GRAPH_NOTE = "知识图谱只覆盖部分知识库；not_in_graph 表示图谱未收录该实体，不代表实体之间没有关系"

TOOL_DESCRIPTION = (
    "查询知识图谱中实体之间的关系与多跳路径，返回带知识库原文证据的事实。"
    "适用场景：需要确认两个实体之间的关系；检索文档中出现了新的关键实体，需要查它与其他实体的关联（多跳推理中的桥接实体）。"
    "传 1 个实体返回它直接相连的事实；传 2 个及以上实体返回它们之间的关系路径。"
    "实体使用检索文档中的英文原名（也可用中文名）。返回的 evidence 是知识库原文，可作为答案依据；"
    "not_in_graph 表示图谱未收录该实体，不代表没有关系。"
)


# ==================== 输入 Schema ====================
class VerifyRelationsInput(BaseModel):
    """实体关系验证工具输入参数"""
    entities: List[str] = Field(description="实体名称列表（1–5 个），使用检索文档中的英文原名",
                                min_length=1, max_length=5)
    max_hops: int = Field(default=3, ge=1, le=3, description="查询实体之间路径时的最大跳数")


# ==================== 工具实现 ====================
class RelationVerifier:
    """知识图谱关系查询：实体对齐 → 邻域 / 路径查询 → 带证据的事实"""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j", config: Optional[dict] = None):
        cfg = config or {}
        self.database = database
        self.timeout_s = cfg.get("timeout_s", 2.0)
        self.max_hops = cfg.get("max_hops", 3)
        self.max_facts = cfg.get("max_facts", 15)
        self.max_paths_per_pair = cfg.get("max_paths_per_pair", 3)
        self.max_pairs = cfg.get("max_pairs", 10)
        self.hub_degree = cfg.get("hub_degree", 8)
        self.evidence_chars = cfg.get("evidence_chars", 300)
        self.driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=5)
        self.linker = EntityLinker(self._run, reload_interval_s=cfg.get("reload_interval_s", 300),
                                   fulltext_min_score=cfg.get("fulltext_min_score", 1.0), max_hops=self.max_hops)
        # 启动时预热词典；Neo4j 不可用时只记录，不阻断服务启动（首次调用时会再尝试）
        try:
            self.driver.verify_connectivity()
            if self.linker.ensure_loaded():
                logger.info("✅ Neo4j 连接成功，实体词典已就绪")
            else:
                logger.warning("⚠️ Neo4j 已连接但图谱为空，实体关系验证暂不可用")
        except Exception as e:
            logger.error(f"❌ Neo4j 连接失败，实体关系验证暂不可用: {e}")

    def close(self):
        self.driver.close()

    def _run(self, cypher: str, **params) -> List[dict]:
        """只读查询（带超时）"""
        with self.driver.session(database=self.database, default_access_mode=READ_ACCESS) as session:
            return session.run(Query(cypher, timeout=self.timeout_s), params).data()

    # ==================== 图查询 ====================
    @staticmethod
    def _rel_fields(v: str = "r") -> str:
        return f"""{{rel_id: {v}.rel_id, s: startNode({v}).entity_id, s_name: startNode({v}).name,
                     o: endNode({v}).entity_id, o_name: endNode({v}).name, predicate: {v}.predicate,
                     label_zh: {v}.label_zh, evidence: {v}.evidence, source_chunks: {v}.source_chunks,
                     support: coalesce({v}.support_unique, {v}.support, 1)}}"""

    _REL_FIELDS = _rel_fields.__func__()

    def _neighborhood(self, entity_id: str, limit: int) -> List[dict]:
        rows = self._run(f"""
            MATCH (a:Entity {{entity_id: $id}})-[r:!MENTIONS]-(:Entity)
            WHERE NOT coalesce(r.inferred, false)
            WITH r ORDER BY coalesce(r.support_unique, r.support, 1) DESC LIMIT $limit
            RETURN {self._REL_FIELDS} AS rel""", id=entity_id, limit=limit)
        return [r["rel"] for r in rows]

    @staticmethod
    def _relation_score(predicate: str, stems: set) -> int:
        """关系与问题的词面相关度：关系名按驼峰拆词取前 5 个字母，与问题词干比对（founder ↔ foundedBy）"""
        words = re.findall(r"[A-Z]?[a-z]+", predicate or "")
        return sum(1 for w in words if len(w) >= 4 and w.lower()[:5] in stems)

    def _two_hop(self, entity_id: str, question: str, limit: int) -> List[dict]:
        """
        单种子实体的第二跳：经一跳邻居再向外一步，按"关系是否与问题相关"挑选（Think-on-Graph 的关系选择，
        这里用零成本的词面匹配代替 LLM 打分）：
          - 只用第一跳尚未覆盖的问题词打分（"developed" 已由 iOS developedBy Apple 回答，第二跳看 "founder"）
          - 跳过兄弟事实（与第一跳同一关系指向同一中间实体，如 Mac developedBy Apple）
          - 每个中间实体最多取 3 条，避免枢纽实体刷屏
        """
        stems = {w[:5] for w in re.findall(r"[a-z]{4,}", (question or "").lower())}
        if not stems:
            return []
        rows = self._run(f"""
            MATCH (a:Entity {{entity_id: $id}})-[r1:!MENTIONS]-(p:Entity)-[r:!MENTIONS]-(c:Entity)
            WHERE c <> a AND NOT coalesce(r1.inferred, false) AND NOT coalesce(r.inferred, false)
              AND NOT (r.predicate = r1.predicate AND (startNode(r) = startNode(r1) OR endNode(r) = endNode(r1)))
            RETURN p.entity_id AS pivot, r1.predicate AS first, {self._REL_FIELDS} AS rel LIMIT 300""", id=entity_id)
        covered = {st for row in rows for st in stems if self._relation_score(row["first"], {st})}
        remaining = stems - covered
        if not remaining:
            return []
        scored = [(self._relation_score(row["rel"]["predicate"], remaining), row["pivot"], row["rel"]) for row in rows]
        picked, per_pivot = [], {}
        for score, pivot, rel in sorted(scored, key=lambda x: (-x[0], -x[2]["support"])):
            if score == 0 or len(picked) >= limit:
                break
            if per_pivot.get(pivot, 0) >= 3:
                continue
            per_pivot[pivot] = per_pivot.get(pivot, 0) + 1
            picked.append(rel)
        return picked

    def _paths(self, a: str, b: str, max_hops: int) -> List[List[dict]]:
        rows = self._run(f"""
            MATCH (a:Entity {{entity_id: $a}}), (b:Entity {{entity_id: $b}})
            MATCH p = ALL SHORTEST (a)-[r:!MENTIONS WHERE NOT coalesce(r.inferred, false)]-{{1,{int(max_hops)}}}(b)
            RETURN [n IN nodes(p)[1..-1] | n.entity_id] AS via,
                   [r IN relationships(p) | {self._REL_FIELDS}] AS rels
            LIMIT 20""", a=a, b=b)
        if not rows:
            return []
        via_ids = sorted({v for row in rows for v in row["via"]})
        degree = {}
        if via_ids:
            for d in self._run("""
                    MATCH (n:Entity) WHERE n.entity_id IN $ids
                    RETURN n.entity_id AS id, COUNT { MATCH (n)-[r]-(m:Entity) WHERE NOT coalesce(r.inferred, false)
                                                      RETURN DISTINCT m } AS deg""", ids=via_ids):
                degree[d["id"]] = d["deg"]
        # 排序：跳数少优先 → 途经枢纽少优先 → 证据支撑强优先（PathRAG：图检索的问题是冗余而不是不足，只留关键路径）
        ranked = sorted(rows, key=lambda row: (
            len(row["rels"]),
            sum(1 for v in row["via"] if degree.get(v, 0) >= self.hub_degree),
            -min(r["support"] for r in row["rels"]),
        ))
        return [row["rels"] for row in ranked[:self.max_paths_per_pair]]

    # ==================== 组装输出 ====================
    def _query(self, links: List[Link], max_hops: int, question: Optional[str] = None) -> Dict[str, Any]:
        facts: Dict[str, dict] = {}

        def add(rel: dict) -> int:
            if rel["rel_id"] not in facts:
                ev = (rel.get("evidence") or [""])[0]
                facts[rel["rel_id"]] = {
                    "id": len(facts) + 1, "subject": rel["s_name"], "relation": rel["predicate"],
                    "relation_zh": rel["label_zh"], "object": rel["o_name"],
                    "evidence": ev[:self.evidence_chars], "source_chunks": (rel.get("source_chunks") or [])[:3],
                    "support": rel["support"],
                }
            return facts[rel["rel_id"]]["id"]

        paths = []
        ids = [l.entity_id for l in links]
        name = {l.entity_id: l.name for l in links}
        if len(ids) >= 2:
            for a, b in list(itertools.combinations(ids, 2))[:self.max_pairs]:
                for rels in self._paths(a, b, max_hops):
                    paths.append({"between": [name[a], name[b]], "hops": len(rels), "fact_ids": [add(r) for r in rels]})
        if not paths:
            # 单实体，或多个实体之间无路径：返回各自的邻域
            per_entity = max(3, self.max_facts // max(len(ids), 1))
            for eid in ids:
                for rel in self._neighborhood(eid, per_entity):
                    add(rel)
            # 预取只有一个种子实体时：补充与问题相关的第二跳（问题里只出现 iOS，也能查到 Apple 的创始人）
            if question and len(ids) == 1:
                for rel in self._two_hop(ids[0], question, max(self.max_facts - len(facts), 0)):
                    add(rel)
        return {"facts": sorted(facts.values(), key=lambda f: f["id"]), "paths": paths}

    def verify(self, entities: List[str], max_hops: Optional[int] = None) -> Dict[str, Any]:
        """工具 / MCP 入口：按实体名查询（名称对齐失败的计入 not_in_graph）"""
        start = time.time()
        hops = min(max_hops or self.max_hops, self.max_hops)
        logger.info(f"🔍 实体关系验证输入: entities={entities}, max_hops={hops}")
        try:
            links, missing = self.linker.link_names(entities)
            result = self._query(links, hops) if links else {"facts": [], "paths": []}
        except Exception as e:
            logger.error(f"❌ 知识图谱查询失败: {e}")
            return {"linked": [], "not_in_graph": [], "facts": [], "paths": [],
                    "note": "知识图谱暂不可用", "stats": {"error": str(e)[:200]}}
        out = {
            "linked": [l.to_dict() for l in links], "not_in_graph": missing, **result, "note": GRAPH_NOTE,
            "stats": {"entities": len(entities), "linked": len(links), "facts": len(result["facts"]),
                      "paths": len(result["paths"]), "graph_version": self.linker.version,
                      "query_time_ms": int((time.time() - start) * 1000)},
        }
        logger.info(f"✅ 实体关系验证: 对齐 {len(links)}/{len(entities)}，事实 {len(result['facts'])}，"
                    f"路径 {len(result['paths'])}，未收录 {missing}，耗时 {out['stats']['query_time_ms']}ms")
        return out

    def prefetch(self, texts: List[str]) -> Optional[Dict[str, Any]]:
        """
        多跳题预取：从问题文本中找实体并查询；没有实体或没有事实时返回 None（静默跳过）。
        texts[0] 为英文改写问题，用于单种子实体时挑选相关的第二跳
        """
        start = time.time()
        links = self.linker.link_text(texts)
        if not links:
            return None
        result = self._query(links, self.max_hops, question=texts[0] if texts else None)
        if not result["facts"]:
            return None
        return {"linked": [l.to_dict() for l in links], "not_in_graph": [], **result, "note": GRAPH_NOTE,
                "stats": {"linked": len(links), "facts": len(result["facts"]), "paths": len(result["paths"]),
                          "graph_version": self.linker.version,
                          "query_time_ms": int((time.time() - start) * 1000)}}

    @staticmethod
    def render(result: Dict[str, Any]) -> str:
        """把查询结果渲染成给模型看的文本（写入系统提示词）"""
        lines = ["实体：" + "、".join(f"{l['name']}（{l['class']}）" for l in result["linked"])]
        for i, p in enumerate(result["paths"], 1):
            chain = " → ".join(f"[{f}]" for f in p["fact_ids"])
            lines.append(f"路径 {i}（{p['between'][0]} ↔ {p['between'][1]}，{p['hops']} 跳）：{chain}")
        lines.append("事实：")
        for f in result["facts"]:
            lines.append(f"[{f['id']}] {f['subject']} -[{f['relation']}｜{f['relation_zh']}]-> {f['object']}")
            lines.append(f"    证据：{f['evidence']}")
        return "\n".join(lines)

    @staticmethod
    def evidence_of(result: Optional[Dict[str, Any]]) -> List[str]:
        """结果中的证据原文（去重），计入答案依据与 RAGAS 上下文"""
        if not result:
            return []
        return list(dict.fromkeys(f["evidence"] for f in result.get("facts", []) if f.get("evidence")))

    def as_tool(self) -> StructuredTool:
        """转换为 LangChain 工具"""
        return StructuredTool.from_function(
            func=self.verify,
            name="relation_verifier",
            description=TOOL_DESCRIPTION,
            args_schema=VerifyRelationsInput,
            return_direct=False,
        )


# ==================== 工厂函数 ====================
def create_relation_verifier(
    neo4j_url: str,
    neo4j_user: str,
    neo4j_password: str,
    database: str = "neo4j",
    config: Optional[dict] = None,
) -> RelationVerifier:
    """创建关系验证器实例（Agent 工具、多跳预取与 /tools/verify 接口共用同一实例与 Neo4j 连接）"""
    return RelationVerifier(neo4j_url, neo4j_user, neo4j_password, database=database, config=config)


def create_relation_verifier_tool(
    neo4j_url: str,
    neo4j_user: str,
    neo4j_password: str,
    database: str = "neo4j",
    config: Optional[dict] = None,
) -> StructuredTool:
    """快速创建关系验证工具实例"""
    return create_relation_verifier(neo4j_url, neo4j_user, neo4j_password, database, config).as_tool()
