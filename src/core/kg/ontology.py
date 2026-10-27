"""
知识图谱本体加载与生成器 (ontology)

ontology/wow_ontology.ttl 是唯一标准，本模块从它生成：
    - 类层次与关系定义（domain / range / 子关系 / 传递性 / 中文名），供抽取校验、入库、查询共用
    - SHACL 校验规则（ontology/wow_shapes.ttl）：关系起点 / 终点类型、互斥类、必填名称
    - Neo4j 约束与索引（constraints.cypher）
    - 抽取说明（类与关系清单文本），供抽取提示词使用

Neo4j 命名约定（读写两端都只通过本模块取名，避免字段不一致）：
    节点标签   :Entity + 类及其所有父类（如 :Entity:CreativeWork:Film）
    唯一键     entity_id（如 "film/goodfellas"），对应 IRI http://wow-agent.local/kg/film/goodfellas
    关系类型   驼峰关系名转大写下划线（directedBy → DIRECTED_BY）

Author: Ke Meng
Created: 2026-10-18
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS, SH, XSD

ONTOLOGY_PATH = Path(__file__).resolve().parents[3] / "ontology" / "wow_ontology.ttl"
SHAPES_PATH = ONTOLOGY_PATH.with_name("wow_shapes.ttl")

WOW = Namespace("http://wow-agent.local/ontology#")
KG = Namespace("http://wow-agent.local/kg/")


@dataclass
class ClassDef:
    name: str
    label_zh: str
    parents: list = field(default_factory=list)      # 直接父类
    ancestors: list = field(default_factory=list)    # 全部祖先（不含自身、不含 Entity）
    comment: str = ""


@dataclass
class PropertyDef:
    name: str
    label_zh: str
    inverse_label_zh: str
    domain: set
    range: set
    parent: Optional[str] = None     # rdfs:subPropertyOf（推理时补出父关系）
    transitive: bool = False
    comment: str = ""

    @property
    def neo4j_type(self) -> str:
        return to_neo4j_type(self.name)


# 实体编号前缀：默认为类名小写（person/、film/），个别过长的类名用简写
ID_PREFIX = {"FoodAndDrink": "food"}


def id_prefix(cls: str) -> str:
    return ID_PREFIX.get(cls, cls.lower())


def to_neo4j_type(prop: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", prop).upper()


def _local(uri) -> str:
    return str(uri).split("#")[-1]


def _label(g: Graph, node, prop=RDFS.label, lang="zh") -> str:
    for o in g.objects(node, prop):
        if isinstance(o, Literal) and o.language == lang:
            return str(o)
    return ""


def _class_set(g: Graph, node) -> set:
    """rdfs:domain / range 可能是单个类或 owl:unionOf 列表"""
    if node is None:
        return set()
    if isinstance(node, BNode):
        union = g.value(node, OWL.unionOf)
        if union is not None:
            return {_local(c) for c in Collection(g, union)}
    return {_local(node)}


class Ontology:
    def __init__(self, path: Path = ONTOLOGY_PATH):
        self.path = path
        self.graph = Graph().parse(path, format="turtle")
        g = self.graph
        self.version = str(g.value(URIRef("http://wow-agent.local/ontology"), OWL.versionInfo))

        self.classes: dict[str, ClassDef] = {}
        for c in g.subjects(RDF.type, OWL.Class):
            if isinstance(c, BNode):
                continue
            name = _local(c)
            parents = [_local(p) for p in g.objects(c, RDFS.subClassOf) if not isinstance(p, BNode)]
            self.classes[name] = ClassDef(name, _label(g, c), parents,
                                          comment=_label(g, c, RDFS.comment))
        for cd in self.classes.values():
            cd.ancestors = [a for a in self._ancestors(cd.name) if a != "Entity"]

        self.properties: dict[str, PropertyDef] = {}
        for p in g.subjects(RDF.type, OWL.ObjectProperty):
            name = _local(p)
            domain = _class_set(g, g.value(p, RDFS.domain))
            if not domain:                       # 只声明了 inverseOf 的反向名（如 created），不作为可抽取关系
                continue
            parent = g.value(p, RDFS.subPropertyOf)
            self.properties[name] = PropertyDef(
                name=name,
                label_zh=_label(g, p),
                inverse_label_zh=_label(g, p, WOW.inverseLabel),
                domain=domain,
                range=_class_set(g, g.value(p, RDFS.range)),
                parent=_local(parent) if parent is not None else None,
                transitive=(p, RDF.type, OWL.TransitiveProperty) in g,
                comment=_label(g, p, RDFS.comment),
            )

        self.disjoint_groups: list[list[str]] = []
        for d in g.subjects(RDF.type, OWL.AllDisjointClasses):
            self.disjoint_groups.append([_local(c) for c in Collection(g, g.value(d, OWL.members))])

        self.datatype_properties = {_local(p) for p in g.subjects(RDF.type, OWL.DatatypeProperty)}

    # ── 类层次 ───────────────────────────────────────────────────
    def _ancestors(self, name: str) -> list:
        out, stack = [], list(self.classes.get(name, ClassDef(name, "")).parents)
        while stack:
            p = stack.pop()
            if p not in out:
                out.append(p)
                stack.extend(self.classes.get(p, ClassDef(p, "")).parents)
        return out

    def is_a(self, cls: str, allowed: set) -> bool:
        """cls 是否属于 allowed 中任一类（含子类）"""
        return cls in allowed or any(a in allowed for a in self.classes[cls].ancestors)

    def labels_for(self, cls: str) -> list:
        """Neo4j 节点标签：Entity + 祖先链 + 自身，例如 ["Entity", "CreativeWork", "Film"]"""
        return ["Entity"] + list(reversed(self.classes[cls].ancestors)) + [cls]

    def disjoint_violation(self, classes: set) -> Optional[tuple]:
        """同一实体被标成互斥的两个类时返回冲突对"""
        for group in self.disjoint_groups:
            hit = [c for c in group if any(self.is_a(x, {c}) for x in classes)]
            if len(hit) > 1:
                return tuple(hit[:2])
        return None

    # ── 关系 ─────────────────────────────────────────────────────
    def check_relation(self, prop: str, subj_cls: str, obj_cls: str) -> Optional[str]:
        """关系是否符合本体：返回 None 表示合规，否则返回原因"""
        p = self.properties.get(prop)
        if p is None:
            return f"未定义的关系 {prop}"
        if not self.is_a(subj_cls, p.domain):
            return f"{prop} 的起点应为 {sorted(p.domain)}，实际为 {subj_cls}"
        if not self.is_a(obj_cls, p.range):
            return f"{prop} 的终点应为 {sorted(p.range)}，实际为 {obj_cls}"
        return None

    def super_properties(self, prop: str) -> list:
        out, cur = [], self.properties[prop].parent
        while cur:
            out.append(cur)
            cur = self.properties[cur].parent if cur in self.properties else None
        return out

    # ── 生成：抽取说明 ───────────────────────────────────────────
    def extraction_spec(self) -> str:
        lines = ["## 实体类（只能使用以下类，选最具体的子类）"]
        for c in self.classes.values():
            if c.name == "Entity":
                continue
            parent = f"（{c.parents[0]} 的子类）" if c.parents and c.parents[0] != "Entity" else ""
            lines.append(f"- {c.name}{parent}：{c.label_zh}{'；' + c.comment if c.comment else ''}")
        lines.append("\n## 关系（只能使用以下关系，方向为 起点 → 终点）")
        for p in self.properties.values():
            lines.append(f"- {p.name}：{p.label_zh}（{'/'.join(sorted(p.domain))} → {'/'.join(sorted(p.range))}）"
                         f"{'；' + p.comment if p.comment else ''}")
        return "\n".join(lines)

    # ── 生成：SHACL ──────────────────────────────────────────────
    def build_shapes(self) -> Graph:
        sg = Graph()
        sg.bind("sh", SH)
        sg.bind("wow", WOW)

        def class_or(classes: set) -> BNode:
            items = []
            for c in sorted(classes):
                b = BNode()
                sg.add((b, SH["class"], WOW[c]))
                items.append(b)
            head = BNode()
            Collection(sg, head, items)
            return head

        # 实体：必须有且只有一个规范名
        es = WOW.EntityShape
        sg.add((es, RDF.type, SH.NodeShape))
        sg.add((es, SH.targetClass, WOW.Entity))
        ps = BNode()
        sg.add((es, SH.property, ps))
        sg.add((ps, SH.path, WOW.name))
        sg.add((ps, SH.minCount, Literal(1)))
        sg.add((ps, SH.maxCount, Literal(1)))
        sg.add((ps, SH.datatype, XSD.string))
        ds = BNode()
        sg.add((es, SH.property, ds))
        sg.add((ds, SH.path, WOW.descriptionSource))
        sg.add((ds, SH.maxCount, Literal(1)))
        allowed = BNode()
        Collection(sg, allowed, [Literal("kb"), Literal("gloss")])
        sg.add((ds, SH["in"], allowed))

        # 关系：起点类型（targetSubjectsOf）+ 终点类型（property shape）
        for p in self.properties.values():
            shape = WOW[f"{p.name}Shape"]
            sg.add((shape, RDF.type, SH.NodeShape))
            sg.add((shape, SH.targetSubjectsOf, WOW[p.name]))
            sg.add((shape, SH["or"], class_or(p.domain)))
            sg.add((shape, SH.message, Literal(f"{p.name} 的起点类型不符合本体", lang="zh")))
            rp = BNode()
            sg.add((shape, SH.property, rp))
            sg.add((rp, SH.path, WOW[p.name]))
            sg.add((rp, SH["or"], class_or(p.range)))
            sg.add((rp, SH.message, Literal(f"{p.name} 的终点类型不符合本体", lang="zh")))

        # 互斥类
        for group in self.disjoint_groups:
            for c in group:
                shape = WOW[f"{c}DisjointShape"]
                sg.add((shape, RDF.type, SH.NodeShape))
                sg.add((shape, SH.targetClass, WOW[c]))
                notb = BNode()
                sg.add((shape, SH["not"], notb))
                sg.add((notb, SH["or"], class_or(set(group) - {c})))
                sg.add((shape, SH.message, Literal(f"{c} 与其他互斥类同时出现", lang="zh")))
        return sg

    # ── 生成：Neo4j 约束与索引 ───────────────────────────────────
    def neo4j_constraints(self) -> list:
        return [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.entity_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            # 实体链接：名称 / 中文名 / 别名全文检索（Neo4j 内置 Lucene，无需插件）
            "CREATE FULLTEXT INDEX entity_names IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.name_zh, n.aliases_text]",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
        ]


@lru_cache(maxsize=1)
def load_ontology() -> Ontology:
    return Ontology()


def entity_iri(entity_id: str) -> URIRef:
    return KG[entity_id]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    o = load_ontology()
    shapes = o.build_shapes()
    SHAPES_PATH.write_text(
        "# 由 src/core/kg/ontology.py 从 wow_ontology.ttl 自动生成，请勿手改\n"
        + shapes.serialize(format="turtle"), encoding="utf-8")
    print(f"本体 v{o.version}：类 {len(o.classes) - 1} 个，关系 {len(o.properties)} 种，"
          f"互斥组 {len(o.disjoint_groups)} 个；SHACL 规则 {len(shapes)} 条三元组 → {SHAPES_PATH.name}")
    print(o.extraction_spec())
