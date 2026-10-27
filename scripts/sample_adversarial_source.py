"""
从 Round 2 评估集中分层随机抽取 30 个 gold chunk，作为对抗性问题的生成源（chunk → GT → question）。
排除 paraphrase_source_30 已用的 source_id 与 topic；每 topic 一条；按题型分层，层内固定种子随机。
"""
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data/evaluation"
QUOTA = {"simple": 15, "reasoning": 12, "multi_hop": 3}
SEED = 42
MIN_LEN = 300
MIN_UNIQUE_RATIO = 0.7


def unique_sentence_ratio(text):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return len(set(sents)) / len(sents) if sents else 0.0


def main():
    results = json.loads((EVAL / "ragas_results_batch_2.json").read_text(encoding="utf-8"))["details"]
    inputs = {x["id"]: x for x in json.loads((EVAL / "ragas_input_batch_2_deduped.json").read_text(encoding="utf-8"))}
    kb = {k["chunk_id"]: k for k in json.loads((EVAL / "kb_snapshot.json").read_text(encoding="utf-8"))}
    para = json.loads((EVAL / "paraphrase_source_30.json").read_text(encoding="utf-8"))
    used_ids = {p["source_id"] for p in para}
    used_topics = {p["topic"] for p in para}

    pool = {t: [] for t in QUOTA}
    for r in results:
        inp = inputs.get(r["id"])
        gold = kb.get(r["chunk_id"], {}).get("content", "")
        if (r["id"] in used_ids or r["topic"] in used_topics or r["faithfulness"] is None
                or not inp or not inp.get("contexts") or r["question_type"] not in QUOTA
                or len(gold) < MIN_LEN or unique_sentence_ratio(gold) < MIN_UNIQUE_RATIO):
            continue
        pool[r["question_type"]].append((r, inp, gold))

    rng = random.Random(SEED)
    picked, topics = [], set()
    for qtype, quota in QUOTA.items():
        cands = pool[qtype][:]
        rng.shuffle(cands)
        n = 0
        for r, inp, gold in cands:
            if n == quota:
                break
            if r["topic"] in topics:
                continue
            topics.add(r["topic"])
            picked.append((r, inp, gold))
            n += 1
        print(f"{qtype}: pool={len(pool[qtype])} picked={n}/{quota}")

    out = []
    for i, (r, inp, gold) in enumerate(picked):
        out.append({
            "id": i,
            "source_id": r["id"],
            "source_r2_faithfulness": r["faithfulness"],
            "topic": r["topic"],
            "question_type": r["question_type"],
            "chunk_id": r["chunk_id"],
            "gold_chunk": gold,
            "question_original": inp["question"],
            "ground_truth": "",
            "question": "",
            "contexts": inp["contexts"],
        })
    (EVAL / "adversarial_source_30.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    fs = [o["source_r2_faithfulness"] for o in out]
    print(f"saved {len(out)} | F min={min(fs):.3f} mean={sum(fs)/len(fs):.3f} max={max(fs):.3f}")


if __name__ == "__main__":
    main()
