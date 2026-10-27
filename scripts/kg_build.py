"""
知识图谱构建：抽取结果 → 校验 → 合并对齐 → 推理 → Neo4j 待入库数据 + RDF + 质量报告

输入
    ontology/wow_ontology.ttl                本体（唯一标准）
    data/kg/selected_records.json            选取的知识库记录（句子编号 = Milvus chunk_id）
    data/kg/extraction/batch_*.json          抽取结果（本批由 claude-opus-5.5 标注，作为标准答案）
    data/kg/extraction/descriptions.json     描述来源复核（kb / gloss），超出原文的描述在此改写

校验（不合格的进 rejected.jsonl，不入库）
    1. 结构：实体类在本体中、实体编号前缀与类一致、属性在本体中
    2. 本体：关系已定义，起点 / 终点类型符合 domain / range（含子类）
    3. 证据：引用的句子存在；主语、宾语须在证据句中出现（名称 / 别名），
       或是证据句所属 Wikipedia 条目的主体（处理 "It was ..." 一类代词）；
       确需语义判断的（如 "a traditional Italian dish" → 意大利菜）由抽取者写 note 说明，报告中单独统计
    4. SHACL：asserted 三元组转 RDF 后用本体生成的规则再校验一遍

别名分强弱：常见英文词或 3 个字母以内的名称 / 别名（Apple、Mac、metal、UK）是弱别名，
只按原大小写匹配（NeXT 不匹配 "next year"），实体链接时还需结合类型或上下文确认

输出 data/kg/import/
    entities.jsonl / relations.jsonl / chunks.jsonl / mentions.jsonl   Neo4j 待入库数据
    kg.ttl            RDF（Turtle）
    constraints.cypher 本体生成的约束与索引
    rejected.jsonl    被拒绝的关系与原因
    manifest.json     版本与统计
    report.md         质量报告（校验、健康度、多跳路径示例）

运行：python scripts/kg_build.py

Author: Ke Meng
Created: 2026-10-18
"""

import glob
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rdflib import Graph, Literal
from rdflib.namespace import RDF, XSD

from src.core.kg.ontology import KG, WOW, entity_iri, id_prefix, load_ontology

KG_DIR = ROOT / "data" / "kg"
OUT = KG_DIR / "import"
EXTRACTOR = "claude-opus-5.5"


# 常见英文词：作为名称 / 别名时容易误匹配（apple 水果 ≠ Apple 公司）
COMMON_WORDS = {"apple", "mac", "metal", "walk", "walks", "walking", "hike", "hikes", "cocoa", "soy", "fish", "meat",
                "cheese", "sugar", "beer", "wood", "trap", "boat", "boats", "blue", "running", "shopping", "fishing",
                "pet", "pets", "sea", "beach", "village", "cat", "cats", "dog", "dogs", "horse", "goat", "chicken",
                "core", "ten", "dirt", "sweet", "next", "rock", "draco", "komodo", "kindle", "disney"}
# 专名类：这些类的实体名称是专有名词（Sweet 乐队 ≠ sweet）
PROPER_CLASSES = ("Person", "Organization", "Place", "CreativeWork", "Event")
# 专有名词但所属类不是专名类（产品 Mac、属名 Draco）
PROPER_NAMES = {"mac", "draco"}


def clean(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def norm(text: str) -> str:
    return clean(text).lower()


def is_weak(form: str, ent: dict) -> bool:
    """弱别名：缩写（≤3 个字母），或常见词作专有名词 / 指代别的东西（Apple Inc. 的 Apple、Heavy metal 的 metal）。
    常见名词实体的本名及其屈折形式（Dog / dogs、Walking / walk）不算弱"""
    f, n = norm(form), norm(ent["name"])
    if f not in COMMON_WORDS and len(f) > 3:
        return False
    if f == n or (" " not in n and n.startswith(f.rstrip("s").rstrip("e"))):
        return f in PROPER_NAMES or load_ontology().is_a(ent["class"], set(PROPER_CLASSES))
    return True


def surface_forms(ent: dict) -> list:
    """全部名称 / 别名（小写），用于对齐检查与条目主体判断"""
    return [f for f in {norm(ent["name"]), *(norm(a) for a in ent.get("aliases", []))} if f]


def mentioned(ent: dict, text: str) -> bool:
    """实体是否在文本中出现：强别名忽略大小写，弱别名只按原大小写"""
    low = norm(text)
    for f in {ent["name"], *ent.get("aliases", [])}:
        if is_weak(f, ent):
            if appears(clean(f), clean(text)):
                return True
        elif appears(norm(f), low):
            return True
    return False


_ALL_FORMS: set = set()   # 全部实体的名称 / 别名，main() 中填充


def appears(form: str, text: str) -> bool:
    """名称 / 别名在文本中出现（词边界，容忍英文复数）；加复数后缀恰好是另一个实体名时不按复数匹配（blue / blues）"""
    plural = "" if {form.lower() + "s", form.lower() + "es"} & _ALL_FORMS else "(?:s|es)?"
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(form) + plural + r"(?![A-Za-z0-9])", text) is not None


def rel_id(s: str, p: str, o: str) -> str:
    return "rel_" + hashlib.md5(f"{s}|{p}|{o}".encode()).hexdigest()[:16]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    onto = load_ontology()
    records = {r["record_id"]: r for r in json.load(open(KG_DIR / "selected_records.json", encoding="utf-8"))}
    batches = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(str(KG_DIR / "extraction" / "batch_*.json")))]

    rejected, issues = [], []

    # ── 1. 实体合并与校验 ──────────────────────────────────────
    entities: dict = {}
    for b in batches:
        for e in b["entities"]:
            eid, cls = e["id"], e["class"]
            if cls not in onto.classes:
                issues.append(f"实体 {eid}: 类 {cls} 不在本体中")
                continue
            if not eid.startswith(id_prefix(cls) + "/"):
                issues.append(f"实体 {eid}: 编号前缀应为 {id_prefix(cls)}/")
            bad_attrs = [k for k in e.get("attrs", {}) if k not in onto.datatype_properties]
            if bad_attrs:
                issues.append(f"实体 {eid}: 未定义的属性 {bad_attrs}")
            if eid in entities:
                old = entities[eid]
                if (old["class"], old["name"]) != (cls, e["name"]):
                    issues.append(f"实体 {eid} 在不同批次中定义不一致：{old['class']}/{old['name']} vs {cls}/{e['name']}")
                old["aliases"] = sorted(set(old["aliases"]) | set(e.get("aliases", [])))
                old["attrs"] = {**e.get("attrs", {}), **old.get("attrs", {})}   # 后续批次补充的属性合并进来，已有值不覆盖
                continue
            entities[eid] = {**e, "aliases": sorted(set(e.get("aliases", [])))}

    # 描述来源复核：未列出的默认 kb；列出的按复核结果标注，超出原文的改写
    review = json.load(open(KG_DIR / "extraction" / "descriptions.json", encoding="utf-8"))["entries"]
    for eid, rv in review.items():
        if eid not in entities:
            issues.append(f"descriptions.json: 实体 {eid} 不存在")
            continue
        if "description" in rv:
            entities[eid]["description"] = rv["description"]
    for eid, e in entities.items():
        e["description_source"] = review.get(eid, {}).get("source", "kb")

    # 对齐检查：不同编号的实体名称 / 别名撞车，可能漏了合并
    form_owner = defaultdict(set)
    for eid, e in entities.items():
        for f in surface_forms(e):
            form_owner[f].add(eid)
    for f, owners in form_owner.items():
        if len(owners) > 1:
            issues.append(f"名称 '{f}' 同时属于 {sorted(owners)}（确认是否应合并或去掉歧义别名）")

    _ALL_FORMS.update(form_owner)

    def article_entities(article: str) -> set:
        a = norm(article)
        return {eid for eid, e in entities.items() if a and a in surface_forms(e)}

    # ── 2. 关系校验 ────────────────────────────────────────────
    rel_acc: dict = {}
    grounding = Counter()
    for b in batches:
        for rec in b["records"]:
            r = records.get(rec["record_id"])
            if r is None:
                issues.append(f"记录 {rec['record_id']} 不在 selected_records 中")
                continue
            sents = r["sentences"]
            for rel in rec["relations"]:
                s, p, o, ev = rel["s"], rel["p"], rel["o"], rel["ev"]
                reason = None
                if s not in entities or o not in entities:
                    reason = f"实体未定义 {[x for x in (s, o) if x not in entities]}"
                elif not ev or any(i >= len(sents) for i in ev):
                    reason = f"证据句编号无效 {ev}"
                else:
                    reason = onto.check_relation(p, entities[s]["class"], entities[o]["class"])
                mode = None
                if reason is None:
                    modes = []
                    for x in (s, o):
                        if any(mentioned(entities[x], sents[i]["text"]) for i in ev):
                            modes.append("lexical")
                        elif any(x in article_entities(sents[i]["article"] or r["topic"]) for i in ev):
                            modes.append("article")
                        elif rel.get("note"):
                            modes.append("note")
                        else:
                            modes.append(None)
                    if None in modes:
                        miss = [x for x, m in zip((s, o), modes) if m is None]
                        reason = f"证据句中找不到 {miss}"
                    else:
                        mode = "note" if "note" in modes else ("article" if "article" in modes else "lexical")
                if reason:
                    rejected.append({"record_id": rec["record_id"], **rel, "reason": reason})
                    continue
                grounding[mode] += 1
                key = (s, p, o)
                acc = rel_acc.setdefault(key, {"evidence": [], "source_chunks": [], "source_records": [],
                                               "grounding": set(), "notes": [], "ev_hashes": defaultdict(set)})
                for i in ev:
                    if sents[i]["text"] not in acc["evidence"]:
                        acc["evidence"].append(sents[i]["text"])
                    if sents[i]["chunk_id"] not in acc["source_chunks"]:
                        acc["source_chunks"].append(sents[i]["chunk_id"])
                # 原始数据中同一句话会出现在多条记录里：按证据句哈希去重，统计独立来源数
                acc["ev_hashes"][rec["record_id"]].update(sents[i]["text_hash"] for i in ev)
                if rec["record_id"] not in acc["source_records"]:
                    acc["source_records"].append(rec["record_id"])
                acc["grounding"].add(mode)
                if rel.get("note") and rel["note"] not in acc["notes"]:
                    acc["notes"].append(rel["note"])

    # ── 3. 推理：子关系补出父关系（directedBy → createdBy）─────────
    relations = []
    for (s, p, o), acc in rel_acc.items():
        pdef = onto.properties[p]
        relations.append({
            "rel_id": rel_id(s, p, o), "type": pdef.neo4j_type, "predicate": p,
            "start": s, "end": o, "label_zh": pdef.label_zh, "inverse_label_zh": pdef.inverse_label_zh,
            "evidence": acc["evidence"][:3], "source_chunks": acc["source_chunks"],
            "source_records": acc["source_records"], "support": len(acc["source_records"]),
            "support_unique": len({frozenset(h) for h in acc["ev_hashes"].values()}),
            "grounding": sorted(acc["grounding"]), "notes": acc["notes"],
            "inferred": False, "inferred_from": None,
            "extractor": EXTRACTOR, "ontology_version": onto.version,
        })
    asserted_keys = set(rel_acc)
    for r in list(relations):
        for parent in onto.super_properties(r["predicate"]):
            if (r["start"], parent, r["end"]) in asserted_keys:
                continue
            pdef = onto.properties[parent]
            relations.append({**r, "rel_id": rel_id(r["start"], parent, r["end"]), "type": pdef.neo4j_type,
                              "predicate": parent, "label_zh": pdef.label_zh, "inverse_label_zh": pdef.inverse_label_zh,
                              "inferred": True, "inferred_from": r["predicate"]})
            asserted_keys.add((r["start"], parent, r["end"]))

    # ── 4. chunk 与 mentions（chunk → 实体，供检索相关性检查与 GraphRAG 回查原文）──
    chunks, mentions = [], []
    covered = {rec["record_id"] for b in batches for rec in b["records"]}
    for rid in covered:
        r = records[rid]
        for s in r["sentences"]:
            chunks.append({"chunk_id": s["chunk_id"], "record_id": rid, "topic": r["topic"],
                           "article": s["article"], "text": s["text"], "text_hash": s["text_hash"]})
            hit = {eid for eid, e in entities.items() if mentioned(e, s["text"])}
            hit |= article_entities(s["article"])
            for eid in sorted(hit):
                mentions.append({"chunk_id": s["chunk_id"], "entity_id": eid})

    # ── 5. 实体输出（带 Neo4j 标签）──────────────────────────────
    mentioned_by = defaultdict(set)
    for m in mentions:
        mentioned_by[m["entity_id"]].add(m["chunk_id"])
    ent_rows = []
    for eid, e in entities.items():
        ent_rows.append({
            "entity_id": eid, "labels": onto.labels_for(e["class"]), "class": e["class"],
            "name": e["name"], "name_zh": e.get("name_zh", ""),
            "aliases": [a for a in e["aliases"] if not is_weak(a, e)],
            "weak_aliases": [a for a in e["aliases"] if is_weak(a, e)], "name_weak": is_weak(e["name"], e),
            "aliases_text": " | ".join(e["aliases"]),   # 全文索引召回用全部别名，弱别名在链接时再确认
            "description": e.get("description", ""), "description_source": e["description_source"],
            **e.get("attrs", {}), "mention_count": len(mentioned_by[eid]),
            "iri": str(entity_iri(eid)), "extractor": EXTRACTOR, "ontology_version": onto.version,
        })

    # ── 6. RDF + SHACL ─────────────────────────────────────────
    g = Graph()
    g.bind("wow", WOW)
    g.bind("kg", KG)
    for e in ent_rows:
        node = entity_iri(e["entity_id"])
        g.add((node, RDF.type, WOW[e["class"]]))
        g.add((node, WOW.name, Literal(e["name"], datatype=XSD.string)))
        if e["name_zh"]:
            g.add((node, WOW.nameZh, Literal(e["name_zh"], lang="zh")))
        for a in e["aliases"]:
            g.add((node, WOW.alias, Literal(a)))
        for a in e["weak_aliases"]:
            g.add((node, WOW.weakAlias, Literal(a)))
        if e["description"]:
            g.add((node, WOW.description, Literal(e["description"], lang="en")))
            g.add((node, WOW.descriptionSource, Literal(e["description_source"])))
        for k in ("birthYear", "foundedYear", "releaseYear", "year"):
            if k in e:
                g.add((node, WOW[k], Literal(int(e[k]), datatype=XSD.integer)))
    for r in relations:
        if not r["inferred"]:
            g.add((entity_iri(r["start"]), WOW[r["predicate"]], entity_iri(r["end"])))

    from pyshacl import validate
    # 不开 RDFS 推理：开放世界语义下 domain / range 会反推类型（把“电影 directedBy 电影”的终点推成人物），
    # 违例被报成互斥冲突而不是类型错误；sh:class 本身沿 subClassOf 检查（本体图并入数据图），足够判断子类
    conforms, _, shacl_text = validate(g, shacl_graph=onto.build_shapes(), ont_graph=onto.graph,
                                       inference="none", abort_on_first=False)

    # OWL 2 RL 推理（统计用）：传递 / 互逆 / 子关系可推出的三元组数
    import owlrl
    closure = Graph()
    for t in g:
        closure.add(t)
    for t in onto.graph:
        closure.add(t)
    before = len(closure)
    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(closure)
    owl_inferred = len(closure) - before

    # ── 7. 图谱健康度 ──────────────────────────────────────────
    adj = defaultdict(set)
    for r in relations:
        adj[r["start"]].add(r["end"])
        adj[r["end"]].add(r["start"])
    isolated = [eid for eid in entities if eid not in adj]
    seen, comps = set(), []
    for eid in entities:
        if eid in seen or eid not in adj:
            continue
        comp, dq = [], deque([eid])
        seen.add(eid)
        while dq:
            x = dq.popleft()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    dq.append(y)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    ent_records = defaultdict(set)
    for r in relations:
        for x in (r["start"], r["end"]):
            ent_records[x].update(r["source_records"])
    shared_entities = [eid for eid, rs in ent_records.items() if len(rs) > 1]

    # 多跳路径示例：在最大连通分量里找跨记录的 2–3 跳最短路径
    directed = defaultdict(list)
    for r in relations:
        if not r["inferred"]:
            directed[r["start"]].append((r["end"], r["predicate"], "→"))
            directed[r["end"]].append((r["start"], r["predicate"], "←"))

    def shortest(a, b, max_hops=3):
        dq, prev = deque([a]), {a: None}
        while dq:
            x = dq.popleft()
            depth, y = 0, x
            while prev[y] is not None:
                depth, y = depth + 1, prev[y][0]
            if x == b:
                break
            if depth >= max_hops:
                continue
            for nxt, p, d in directed[x]:
                if nxt not in prev:
                    prev[nxt] = (x, p, d)
                    dq.append(nxt)
        if b not in prev:
            return None
        path, x = [], b
        while prev[x] is not None:
            px, p, d = prev[x]
            path.append((px, p, d, x))
            x = px
        return list(reversed(path))

    examples = []
    for comp in comps[:6]:
        cand = sorted(comp)
        for i, a in enumerate(cand):
            for b in cand[i + 1:]:
                if not (ent_records[a] & ent_records[b]):
                    path = shortest(a, b)
                    if path and len(path) >= 2:
                        examples.append(path)
                        break
            if len(examples) >= 2 * (comps.index(comp) + 1):
                break
    examples = examples[:10]

    # ── 8. 输出 ────────────────────────────────────────────────
    OUT.mkdir(parents=True, exist_ok=True)

    def dump(name, rows):
        with open(OUT / name, "w", encoding="utf-8") as f:
            for row in rows:
                # Neo4j 中属性为 null 等于不写入，直接去掉（如 asserted 关系的 inferred_from）
                row = {k: v for k, v in row.items() if v is not None}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    dump("entities.jsonl", ent_rows)
    dump("relations.jsonl", relations)
    dump("chunks.jsonl", chunks)
    dump("mentions.jsonl", mentions)
    dump("rejected.jsonl", rejected)
    (OUT / "kg.ttl").write_text(g.serialize(format="turtle"), encoding="utf-8")
    (OUT / "constraints.cypher").write_text(";\n".join(onto.neo4j_constraints()) + ";\n", encoding="utf-8")

    n_asserted = sum(1 for r in relations if not r["inferred"])
    pred_count = Counter(r["predicate"] for r in relations if not r["inferred"])
    stats = {
        "built_at": datetime.now().isoformat(timespec="seconds"), "extractor": EXTRACTOR,
        "ontology_version": onto.version, "records": len(covered), "chunks": len(chunks),
        "entities": len(ent_rows), "relations_asserted": n_asserted,
        "relations_inferred": len(relations) - n_asserted, "mentions": len(mentions),
        "rejected": len(rejected), "grounding": dict(grounding), "shacl_conforms": conforms,
        "owlrl_inferred_triples": owl_inferred, "isolated_entities": len(isolated),
        "isolated_ratio": round(len(isolated) / max(len(entities), 1), 3),
        "components": len(comps), "largest_component": len(comps[0]) if comps else 0,
        "entities_shared_across_records": len(shared_entities),
        "description_source": dict(Counter(e["description_source"] for e in ent_rows)),
        "descriptions_rewritten": sum(1 for rv in review.values() if "description" in rv),
        "weak_forms": sum(len(e["weak_aliases"]) + e["name_weak"] for e in ent_rows),
        "relations_support_inflated": sum(1 for r in relations if not r["inferred"] and r["support"] > r["support_unique"]),
        "relations_by_predicate": dict(pred_count.most_common()),
    }
    (OUT / "manifest.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    name = {eid: e["name"] for eid, e in entities.items()}

    def fmt(path):
        parts = [name[path[0][0]]]
        for _, p, d, y in path:
            parts.append(f" -[{p}]-> " if d == "→" else f" <-[{p}]- ")
            parts.append(name[y])
        return "".join(parts)

    lines = [f"# 知识图谱质量报告（{stats['built_at']}）", "",
             f"抽取者 {EXTRACTOR} · 本体 v{onto.version} · 记录 {len(covered)} 条 · chunk {len(chunks)} 个", "",
             "## 规模", "",
             f"- 实体 {len(ent_rows)}，关系 {n_asserted}（另有子关系推理补出 {len(relations) - n_asserted} 条），"
             f"mentions {len(mentions)}",
             f"- OWL 2 RL 闭包可推出 {owl_inferred} 条三元组（传递 / 互逆 / 子类，查询时按需使用，不预先入库）", "",
             "## 校验", "",
             f"- 拒绝 {len(rejected)} 条；SHACL：{'通过' if conforms else '不通过'}",
             f"- 证据依据：{dict(grounding)}（lexical=证据句出现名称 / article=代词指代条目主体 / note=语义判断，见备注）", ""]
    if issues:
        lines += ["### 待确认问题", ""] + [f"- {x}" for x in issues] + [""]
    if rejected:
        lines += ["### 被拒绝的关系", ""] + [f"- {x['s']} -[{x['p']}]-> {x['o']}：{x['reason']}" for x in rejected] + [""]
    if not conforms:
        lines += ["### SHACL 报告", "", "```", shacl_text[:4000], "```", ""]
    lines += ["## 图谱健康度（对比旧版：孤立节点 62%，跨切片关系 0，最大连通分量 12）", "",
              f"- 孤立实体 {len(isolated)}（{stats['isolated_ratio']:.0%}）",
              f"- 连通分量 {len(comps)}，最大分量 {stats['largest_component']} 个实体，前五大：{[len(c) for c in comps[:5]]}",
              f"- 跨记录共享的实体 {len(shared_entities)} 个（同一实体被多条记录的关系引用，多跳的基础）", "",
              "## 关系分布", ""] + [f"- {p}：{c}" for p, c in pred_count.most_common()] + [
              "", "## 多跳路径示例（跨记录，最短路径）", ""] + [f"- {fmt(p)}" for p in examples]
    if isolated:
        lines += ["", "## 孤立实体（有提及、无关系）", "", ", ".join(sorted(name[x] for x in isolated))]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=1))
    for x in issues:
        print("⚠", x)
    for x in rejected:
        print("✗", x["s"], x["p"], x["o"], "|", x["reason"])
    if not conforms:
        print(shacl_text[:3000])
    print("\n多跳示例:")
    for p in examples:
        print("  ", fmt(p))


if __name__ == "__main__":
    main()
