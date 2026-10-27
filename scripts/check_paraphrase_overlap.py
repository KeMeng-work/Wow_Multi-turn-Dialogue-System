"""
对比改写前后问题经 QueryRewriter 翻译出的英文查询与 gold chunk 的词汇重叠率。
需在服务器上运行（依赖 vLLM qwen3-4b @ localhost:8001）：
    python scripts/check_paraphrase_overlap.py
输出：data/evaluation/paraphrase_overlap_report.json
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.config_path import STM_CONFIG  # noqa: E402
from src.core.query_rewrite.query_rewriter import QueryRewriter  # noqa: E402

SRC = ROOT / "data/evaluation/paraphrase_source_30.json"
OUT = ROOT / "data/evaluation/paraphrase_overlap_report.json"

STOP = set(
    "a an the of in on at to for and or is are was were be been by with as from that this it its "
    "what which who whom when where why how do does did can could would should has have had not no "
    "their there they them his her he she also into than then such more most very".split()
)


def content_words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 2}


def overlap(query, chunk):
    q = content_words(query)
    if not q:
        return 0.0
    return len(q & content_words(chunk)) / len(q)


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    rw = QueryRewriter(memory=None, stm_config=STM_CONFIG, enable_warmup=False)   # memory=None：无对话历史
    rows = []
    for item in data:
        gold = item["contexts"][0]
        r_orig = rw.rewrite(item["question_original"], f"ov_o_{item['id']}")
        r_para = rw.rewrite(item["question"], f"ov_p_{item['id']}")
        rows.append({
            "id": item["id"],
            "topic": item["topic"],
            "question_original": item["question_original"],
            "question": item["question"],
            "en_original": r_orig["en_query"],
            "en_paraphrase": r_para["en_query"],
            "overlap_original": round(overlap(r_orig["en_query"], gold), 3),
            "overlap_paraphrase": round(overlap(r_para["en_query"], gold), 3),
        })
    n = len(rows)
    mo = sum(r["overlap_original"] for r in rows) / n
    mp = sum(r["overlap_paraphrase"] for r in rows) / n
    report = {"mean_overlap_original": round(mo, 3), "mean_overlap_paraphrase": round(mp, 3), "rows": rows}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"mean overlap: original={mo:.3f} paraphrase={mp:.3f}")
    for r in sorted(rows, key=lambda x: x["overlap_original"] - x["overlap_paraphrase"]):
        print(f"id={r['id']:>2} {r['overlap_original']:.2f}->{r['overlap_paraphrase']:.2f} | {r['en_paraphrase']}")


if __name__ == "__main__":
    main()
