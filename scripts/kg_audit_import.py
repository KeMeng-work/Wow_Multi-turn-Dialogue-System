"""
知识图谱待入库数据质量审计（独立于 kg_build.py 的校验逻辑，入库前运行）

检查层次
    A. 格式与引用完整性：必填字段、类型、编号唯一、关系 / mentions / 来源 chunk 引用都存在、标签与本体一致
    B. 与 Milvus 的编号一致性：直接从 producer.py 源码取出 get_checked_sentences（不用 kg_select_records 的副本），
       重新推导 chunk_id 与句子，逐句比对
    C. 图结构：自环、传递关系成环、互为反向的重复边、枢纽节点、可达性（3 跳内实体对数）
    D. 实体质量：疑似漏合并（名称近似）、易误匹配的别名（过短 / 常见英文词）、描述是否超出知识库原文
    E. Neo4j 兼容：属性值类型（Neo4j 不支持 null / 嵌套 dict / 混合类型列表）、标签与关系类型命名
    F. RDF：kg.ttl 可解析、三元组数与 asserted 关系一致

输出：data/kg/import/audit.md，并打印摘要
运行：python scripts/kg_audit_import.py

Author: Ke Meng
Created: 2026-10-18
"""

import ast
import json
import random
import re
import sys
from collections import Counter, defaultdict, deque
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.core.kg.ontology import WOW, load_ontology

IMP = ROOT / "data" / "kg" / "import"
CHUNKS_DIR = ROOT / "data" / "processed" / "chunks"

# 常见英文词：作为别名时容易在其他句子里误匹配
COMMON_WORDS = {"apple", "mac", "metal", "walk", "walks", "hike", "hikes", "cocoa", "soy", "fish", "meat", "cheese",
                "sugar", "beer", "wood", "trap", "boat", "boats", "blue", "running", "walking", "shopping", "fishing",
                "pet", "pets", "sea", "beach", "village", "cat", "cats", "dog", "dogs", "horse", "goat", "chicken",
                "running", "core", "ten", "dirt", "sweet", "next", "rock", "draco", "komodo", "kindle", "disney"}
# 已人工确认是不同实体（颜色 Blue ≠ 音乐流派 Blues）
KNOWN_DISTINCT = {("concept/blue", "genre/blues")}


def load(name):
    return [json.loads(l) for l in open(IMP / name, encoding="utf-8")]


def producer_get_checked_sentences():
    """从 producer.py 源码中取出 get_checked_sentences 函数单独执行（避免导入时加载向量模型）"""
    src = (ROOT / "src" / "core" / "redis-stream" / "producer.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_checked_sentences")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "producer.py", "exec"), ns)
    return ns["get_checked_sentences"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    onto = load_ontology()
    ents, rels, chunks, mentions = load("entities.jsonl"), load("relations.jsonl"), load("chunks.jsonl"), load("mentions.jsonl")
    E = {e["entity_id"]: e for e in ents}
    C = {c["chunk_id"]: c for c in chunks}
    findings = defaultdict(list)     # 级别 → 问题
    out = ["# 待入库数据质量审计", ""]

    def section(title):
        out.extend(["", f"## {title}", ""])

    # ── A. 格式与引用完整性 ─────────────────────────────────────
    section("A. 格式与引用完整性")
    req_e = ["entity_id", "labels", "class", "name", "name_zh", "aliases", "weak_aliases", "description",
             "description_source", "iri"]
    req_r = ["rel_id", "type", "predicate", "start", "end", "evidence", "source_chunks", "support", "support_unique",
             "inferred"]
    for r in rels:
        miss = [k for k in req_r if k not in r]
        if miss:
            findings["错误"].append(f"关系 {r['rel_id']} 缺字段 {miss}")
        elif not 1 <= r["support_unique"] <= r["support"]:
            findings["错误"].append(f"关系 {r['rel_id']} support_unique={r['support_unique']} 不在 [1, support] 内")
    for e in ents:
        miss = [k for k in req_e if k not in e or e[k] is None or (e[k] == "" and k != "name_zh")]
        if miss:
            findings["错误"].append(f"实体 {e['entity_id']} 缺字段 {miss}")
        if e["labels"] != onto.labels_for(e["class"]):
            findings["错误"].append(f"实体 {e['entity_id']} 标签 {e['labels']} 与本体不一致")
    if len(E) != len(ents):
        findings["错误"].append("实体编号重复")
    if len({r["rel_id"] for r in rels}) != len(rels):
        findings["错误"].append("关系编号重复")
    dangling = [r["rel_id"] for r in rels if r["start"] not in E or r["end"] not in E]
    bad_src = [r["rel_id"] for r in rels if any(c not in C for c in r["source_chunks"])]
    bad_m = [m for m in mentions if m["chunk_id"] not in C or m["entity_id"] not in E]
    no_ev = [r["rel_id"] for r in rels if not r["evidence"] or not r["source_chunks"]]
    for name, lst in (("关系端点不存在", dangling), ("来源 chunk 不存在", bad_src), ("mentions 引用不存在", bad_m),
                      ("关系缺证据", no_ev)):
        if lst:
            findings["错误"].append(f"{name}：{len(lst)} 条")
    ev_not_in_chunk = [r["rel_id"] for r in rels
                       if not all(any(ev == C[c]["text"] for c in r["source_chunks"]) for ev in r["evidence"])]
    if ev_not_in_chunk:
        findings["错误"].append(f"证据原文与来源 chunk 文本对不上：{len(ev_not_in_chunk)} 条")
    for r in rels:
        err = onto.check_relation(r["predicate"], E[r["start"]]["class"], E[r["end"]]["class"])
        if err:
            findings["错误"].append(f"{r['rel_id']} {err}")
    out.append(f"实体 {len(ents)}、关系 {len(rels)}、chunk {len(chunks)}、mentions {len(mentions)}；"
               f"端点 / 来源 / 证据 / 本体类型逐条检查")

    # ── B. 与 Milvus 编号一致性 ────────────────────────────────
    section("B. 与 Milvus chunk_id 的一致性（直接执行 producer.py 的取句函数）")
    get_cs = producer_get_checked_sentences()
    items = json.load(open(CHUNKS_DIR / "wow_ner_keywords_merged.jsonl", encoding="utf-8")) + \
        json.load(open(CHUNKS_DIR / "wow_increment_330.json", encoding="utf-8"))
    by_rid = {(it.get("metadata", {}).get("chunk_id") or it.get("chunk_id")): it for it in items}
    mismatch, missing_rec = 0, set()
    for c in chunks:
        it = by_rid.get(c["record_id"])
        if it is None:
            missing_rec.add(c["record_id"])
            continue
        sents, _ = get_cs(it)
        idx = int(c["chunk_id"].rsplit("_s", 1)[1])
        if idx >= len(sents) or sents[idx] != c["text"]:
            mismatch += 1
    out.append(f"逐句比对 {len(chunks)} 个 chunk：不一致 {mismatch}，记录缺失 {len(missing_rec)}")
    if mismatch or missing_rec:
        findings["错误"].append(f"chunk 与 producer 取句不一致 {mismatch} 个 / 记录缺失 {len(missing_rec)}")
    dup_text = Counter(c["text"] for c in chunks)
    out.append(f"重复句子（不同 chunk 文本相同，WoW 原始数据如此）：{sum(v for v in dup_text.values() if v > 1)} 个 chunk")

    # ── C. 图结构 ──────────────────────────────────────────────
    section("C. 图结构")
    asserted = [r for r in rels if not r["inferred"]]
    inflated = [r for r in asserted if r["support"] > r["support_unique"]]
    out.append(f"support 因原始数据重复句虚高的关系 {len(inflated)} 条（按证据句去重后的独立来源数见 support_unique）")
    selfloop =[r for r in rels if r["start"] == r["end"]]
    if selfloop:
        findings["错误"].append(f"自环 {len(selfloop)} 条")
    pairs = {(r["start"], r["end"], r["predicate"]) for r in asserted}
    reverse_dup = [(s, o, p) for s, o, p in pairs if (o, s, p) in pairs and s < o]
    for s, o, p in reverse_dup:
        findings["警告"].append(f"互为反向的同类关系：{s} -[{p}]- {o}")
    # 传递关系成环
    for p in [x for x, d in onto.properties.items() if d.transitive]:
        g = defaultdict(set)
        for r in asserted:
            if r["predicate"] == p:
                g[r["start"]].add(r["end"])
        color = {}

        def dfs(u, stack):
            color[u] = 1
            for v in g[u]:
                if color.get(v) == 1:
                    findings["错误"].append(f"传递关系 {p} 成环：{' → '.join(stack + [v])}")
                elif v not in color:
                    dfs(v, stack + [v])
            color[u] = 2
        for n in list(g):
            if n not in color:
                dfs(n, [n])
    # 传递冗余：A→C 已可由 A→B→C 推出
    redundant = []
    for p in [x for x, d in onto.properties.items() if d.transitive]:
        g = defaultdict(set)
        for r in asserted:
            if r["predicate"] == p:
                g[r["start"]].add(r["end"])
        for a in g:
            for c in g[a]:
                if any(c in g.get(b, ()) for b in g[a] if b != c):
                    redundant.append(f"{a} -[{p}]-> {c}")
    out.append(f"自环 {len(selfloop)}；互为反向重复 {len(reverse_dup)}；传递冗余边 {len(redundant)}（可由其他边推出，保留无害）")
    degree = Counter()
    for r in asserted:
        degree[r["start"]] += 1
        degree[r["end"]] += 1
    out.append("度数最高的实体（枢纽，多跳查询需降权）：" + "，".join(f"{E[k]['name']} {v}" for k, v in degree.most_common(8)))
    adj = defaultdict(set)
    for r in asserted:
        adj[r["start"]].add(r["end"])
        adj[r["end"]].add(r["start"])
    reach = Counter()
    for s in adj:
        dist, dq = {s: 0}, deque([s])
        while dq:
            x = dq.popleft()
            if dist[x] == 3:
                continue
            for y in adj[x]:
                if y not in dist:
                    dist[y] = dist[x] + 1
                    dq.append(y)
        for d in dist.values():
            if d:
                reach[d] += 1
    out.append(f"有序实体对可达数：1 跳 {reach[1]}，2 跳 {reach[2]}，3 跳 {reach[3]}（2–3 跳是多跳问答的有效范围）")
    rec_rel = Counter()
    for r in asserted:
        for rid in r["source_records"]:
            rec_rel[rid] += 1
    zero_rec = sorted({c["record_id"] for c in chunks} - set(rec_rel))
    out.append(f"没有任何关系的记录 {len(zero_rec)} 条（原文只有描述性内容，如定义、统计数字）")

    # ── D. 实体质量 ────────────────────────────────────────────
    section("D. 实体质量")
    names = [(e["entity_id"], e["name"].lower()) for e in ents]
    near = []
    for i, (a, na) in enumerate(names):
        for b, nb in names[i + 1:]:
            if (a, b) in KNOWN_DISTINCT or (b, a) in KNOWN_DISTINCT:
                continue
            if na != nb and (SequenceMatcher(None, na, nb).ratio() > 0.88 or na.rstrip("s") == nb.rstrip("s")):
                near.append(f"{a}（{E[a]['name']}） ≈ {b}（{E[b]['name']}）")
    for x in near:
        findings["待人工确认"].append(f"名称相近，确认是否同一实体：{x}")
    # 易误匹配的名称 / 别名：应标为弱别名（只按原大小写匹配、链接时需上下文）；常见名词实体的本名（Dog、Meat）按词义匹配是对的
    weak, plain = [], []
    for e in ents:
        weak += [f"{e['entity_id']}：'{a}'" for a in e["weak_aliases"]]
        if e["name_weak"]:
            weak.append(f"{e['entity_id']}：'{e['name']}'（名称）")
        for f in [e["name"], *e["aliases"]]:
            fl = f.lower()
            if (fl in COMMON_WORDS or len(fl) <= 3) and not e["name_weak"] and f not in e["weak_aliases"]:
                plain.append(f"{e['entity_id']}：'{f}'")
    out.append(f"疑似漏合并 {len(near)} 对（已确认不同的 {len(KNOWN_DISTINCT)} 对不计）")
    out.append(f"\n弱别名 {len(weak)} 个（只按原大小写匹配，实体链接时需结合类型或上下文确认）：" + "、".join(weak))
    out.append(f"\n常见词但按普通词义匹配 {len(plain)} 个（实体本身就是这个常见名词，请确认无专有名词混入）：" + "、".join(plain))
    # 描述是否来自知识库原文：描述中的实词在该实体出现过的句子里能否找到
    ent_text = defaultdict(str)
    for m in mentions:
        ent_text[m["entity_id"]] += " " + C[m["chunk_id"]]["text"].lower()
    stop = {"a", "an", "the", "of", "in", "and", "or", "to", "by", "for", "with", "as", "on", "from", "is", "its", "that",
            "type", "member", "group", "u.s.", "state", "country", "city", "region", "company", "used"}
    review = json.load(open(ROOT / "data" / "kg" / "extraction" / "descriptions.json", encoding="utf-8"))["entries"]
    ext_desc, gloss_fact = [], []
    for e in ents:
        if e["description_source"] == "gloss" and re.search(r"\d", e["description"]):
            gloss_fact.append(e["entity_id"])
        if e["description_source"] != "kb" or e["entity_id"] in review:
            continue     # 通用释义不要求出自原文；已人工复核的 kb 描述不再报
        words = [w for w in re.findall(r"[a-z][a-z\-]+", e["description"].lower()) if w not in stop and len(w) > 3]
        if words:
            cov = sum(1 for w in words if w in ent_text[e["entity_id"]]) / len(words)
            if cov < 0.5:
                ext_desc.append((cov, e["entity_id"], e["description"]))
    ext_desc.sort()
    src = Counter(e["description_source"] for e in ents)
    rewritten = [(eid, rv) for eid, rv in review.items() if "rewritten_from" in rv]
    out.append(f"\n描述来源：{dict(src)}；人工复核 {len(review)} 个，其中改写 {len(rewritten)} 个（原描述含知识库没有的事实）：")
    out += [f"- {eid}：{rv['rewritten_from']} → {rv['description']}（{rv['source']}）" for eid, rv in rewritten]
    out.append(f"\n未复核且与原文重合度 < 50% 的 kb 描述 {len(ext_desc)} 个；含数字的 gloss 描述 {len(gloss_fact)} 个")
    out += [f"- {eid}（{cov:.0%}）：{d}" for cov, eid, d in ext_desc[:40]]
    if ext_desc:
        findings["警告"].append(f"{len(ext_desc)} 个标为 kb 的描述与原文重合度低且未复核，需确认是否来自原文")
    if gloss_fact:
        findings["警告"].append(f"gloss 描述含具体数字（通用释义不应含事实）：{gloss_fact}")
    no_zh = [e["entity_id"] for e in ents if not e["name_zh"]]
    out.append(f"\n缺中文名 {len(no_zh)} 个")

    # ── E. Neo4j 兼容性 ────────────────────────────────────────
    section("E. Neo4j 兼容性")
    bad_props = []
    for row in ents + rels:
        for k, v in row.items():
            if v is None:
                bad_props.append(f"{row.get('entity_id') or row['rel_id']}.{k} = null")
            elif isinstance(v, dict):
                bad_props.append(f"{row.get('entity_id') or row['rel_id']}.{k} 是嵌套对象")
            elif isinstance(v, list) and len({type(x) for x in v}) > 1:
                bad_props.append(f"{row.get('entity_id') or row['rel_id']}.{k} 列表类型混合")
    nulls = [x for x in bad_props if x.endswith("null")]
    others = [x for x in bad_props if not x.endswith("null")]
    out.append(f"null 属性 {len(nulls)} 个；其他类型问题 {len(others)} 个")
    if bad_props:
        findings["错误"].extend(bad_props[:10])
    bad_names = [r["type"] for r in rels if not re.fullmatch(r"[A-Z][A-Z_]*", r["type"])] + \
                [l for e in ents for l in e["labels"] if not re.fullmatch(r"[A-Z][A-Za-z]*", l)]
    out.append(f"标签 / 关系类型命名不合规 {len(bad_names)} 个")

    # ── F. RDF ─────────────────────────────────────────────────
    section("F. RDF")
    from rdflib import Graph
    g = Graph().parse(IMP / "kg.ttl", format="turtle")
    rdf_rel = sum(1 for _, p, _ in g if str(p).startswith(str(WOW)) and str(p).split("#")[-1] in onto.properties)
    out.append(f"kg.ttl 解析成功，三元组 {len(g)}；关系三元组 {rdf_rel}，asserted 关系 {len(asserted)}"
               f"（{'一致' if rdf_rel == len(asserted) else '不一致'}）")
    if rdf_rel != len(asserted):
        findings["错误"].append("RDF 关系数与 asserted 关系数不一致")

    # ── 语义抽查清单（供人工复核）──────────────────────────────
    section("G. 语义抽查（随机 30 条 asserted 关系，人工复核）")
    random.seed(20261018)
    for r in random.sample(asserted, 30):
        out.append(f"- [ ] {E[r['start']]['name']} -[{r['predicate']}]-> {E[r['end']]['name']}　｜　{r['evidence'][0][:150]}")
    section("H. 依赖语义判断（note）的关系，全部列出供复核")
    for r in asserted:
        if "note" in r["grounding"]:
            out.append(f"- {E[r['start']]['name']} -[{r['predicate']}]-> {E[r['end']]['name']}：{'；'.join(r['notes'])}")

    head = ["## 结论", ""]
    for level in ("错误", "警告", "待人工确认"):
        head.append(f"- {level}：{len(findings[level])}")
        head += [f"  - {x}" for x in findings[level][:30]]
    out[2:2] = head
    (IMP / "audit.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:2] + head))
    print("\n".join(l for l in out if l and not l.startswith("- [ ]") and not l.startswith("- ") or l.startswith("- ") and "（" in l and "%" in l)[:6000])


if __name__ == "__main__":
    main()
