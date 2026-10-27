"""
知识图谱演示数据选取：按主题簇从知识库中选约 100 条记录

为什么不取前 100 条：知识库 432 条记录话题分散（313 个话题），一半记录与其他记录没有任何共享的 Wikipedia 条目，
随机取样抽出的图谱只是一堆孤立小图，无法展示多跳。这里把"引用了同一 Wikipedia 条目"的记录连成簇，
优先选大簇，使簇内实体能连成网。

句子编号与 Milvus 完全一致：取句规则复制自 producer.get_checked_sentences（顺序相同），
第 i 句对应 Milvus chunk_id = f"{record_id}_s{i}"。

输出：data/kg/selected_records.json
运行：python scripts/kg_select_records.py [--target 100]

Author: Ke Meng
Created: 2026-10-18
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "processed" / "chunks"
SOURCES = [CHUNKS / "wow_ner_keywords_merged.jsonl", CHUNKS / "wow_increment_330.json"]
OUTPUT = ROOT / "data" / "kg" / "selected_records.json"


def get_checked_sentences(item) -> list:
    """与 src/core/redis-stream/producer.py 的 get_checked_sentences 保持一致（顺序决定 chunk_id）"""
    dialog = item.get("metadata", {}).get("original_data", {}).get("dialog", [])
    sentences = []
    for turn in dialog:
        cs = turn.get("checked_sentence", {})
        if not isinstance(cs, dict):
            continue
        for v in cs.values():
            if isinstance(v, str):
                v = v.strip()
                if v and v != "no_passages_used":
                    sentences.append(v)
            elif isinstance(v, list):
                sentences.extend(s.strip() for s in v
                                 if isinstance(s, str) and s.strip() and s.strip() != "no_passages_used")
    return sentences


def sentence_articles(item) -> list:
    """每句所属的 Wikipedia 条目（与 get_checked_sentences 同序）；句中代词（It / They）常指代该条目主体，证据核对时使用"""
    out = []
    for turn in item.get("metadata", {}).get("original_data", {}).get("dialog", []) or []:
        cs = turn.get("checked_sentence", {})
        if not isinstance(cs, dict):
            continue
        article = next((v for v in (turn.get("checked_passage") or {}).values()
                        if v and v != "no_passages_used"), "")
        for v in cs.values():
            vals = [v] if isinstance(v, str) else (v if isinstance(v, list) else [])
            out.extend(article for x in vals if isinstance(x, str) and x.strip() and x.strip() != "no_passages_used")
    return out


def cited_titles(item) -> set:
    titles = set()
    for turn in item.get("metadata", {}).get("original_data", {}).get("dialog", []) or []:
        for v in (turn.get("checked_passage") or {}).values():
            if v and v != "no_passages_used":
                titles.add(v.strip().lower())
    return titles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=100)
    args = ap.parse_args()

    items = []
    for src in SOURCES:
        items.extend(json.load(open(src, encoding="utf-8")))
    recs = []
    for it in items:
        md = it.get("metadata", {})
        sents = get_checked_sentences(it)
        if not sents:
            continue
        arts = sentence_articles(it)
        assert len(arts) == len(sents), md.get("chunk_id")
        recs.append({"record_id": md.get("chunk_id") or it.get("chunk_id"), "articles": arts,
                     "topic": (md.get("original_data", {}) or {}).get("chosen_topic") or md.get("topic", ""),
                     "titles": cited_titles(it), "sentences": sents})

    # 按共享引用条目连成簇
    by_title = defaultdict(list)
    for i, r in enumerate(recs):
        for t in r["titles"]:
            by_title[t].append(i)
    adj = defaultdict(set)
    for ids in by_title.values():
        for a in ids:
            adj[a].update(b for b in ids if b != a)
    seen, clusters = set(), []
    for i in range(len(recs)):
        if i in seen:
            continue
        stack, comp = [i], []
        seen.add(i)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        clusters.append(comp)
    # 大簇优先；同样大小时引用句多的优先（可抽关系更多）
    clusters.sort(key=lambda c: (-len(c), -sum(len(recs[i]["sentences"]) for i in c)))

    chosen = []
    for cid, comp in enumerate(clusters):
        if len(comp) < 2 or len(chosen) + len(comp) > args.target + 5:
            continue
        for i in sorted(comp, key=lambda i: recs[i]["topic"]):
            chosen.append((cid, recs[i]))
        if len(chosen) >= args.target:
            break

    out = []
    for cid, r in chosen:
        out.append({
            "record_id": r["record_id"], "topic": r["topic"], "cluster": cid,
            "sentences": [{"idx": i, "chunk_id": f"{r['record_id']}_s{i}", "article": r["articles"][i], "text": s,
                           "text_hash": hashlib.md5(s.encode("utf-8")).hexdigest()}
                          for i, s in enumerate(r["sentences"])],
        })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUTPUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    sys.stdout.reconfigure(encoding="utf-8")
    n_clusters = len({c for c, _ in chosen})
    print(f"选取 {len(out)} 条记录，来自 {n_clusters} 个主题簇，共 {sum(len(r['sentences']) for r in out)} 句 → {OUTPUT}")
    cl = defaultdict(list)
    for cid, r in chosen:
        cl[cid].append(r["topic"])
    for cid, topics in cl.items():
        print(f"  簇 {cid:>3}（{len(topics)} 条）: {', '.join(sorted(set(topics)))}")


if __name__ == "__main__":
    main()
