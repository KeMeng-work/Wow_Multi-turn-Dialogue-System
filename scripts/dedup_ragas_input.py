"""
ragas_input_batch_N 上下文去重预处理

对每条记录的 contexts 列表做精确字符串去重，
保留首次出现的顺序，不修改其他字段。
原文件不动，输出到 ragas_input_batch_N_deduped.json。

用法：
  python scripts/dedup_ragas_input.py --batch 1
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    in_path  = ROOT / f"data/evaluation/ragas_input_batch_{args.batch}.json"
    out_path = ROOT / f"data/evaluation/ragas_input_batch_{args.batch}_deduped.json"

    if not in_path.exists():
        print(f"❌ 找不到输入文件：{in_path}")
        return

    with open(in_path, encoding="utf-8") as f:
        data = json.load(f)

    total_before      = 0
    total_after       = 0
    records_with_dups = 0
    dist_before       = Counter()
    dist_after        = Counter()

    for rec in data:
        ctxs = rec.get("contexts", [])
        dist_before[len(ctxs)] += 1

        seen, unique = set(), []
        for c in ctxs:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        if len(unique) < len(ctxs):
            records_with_dups += 1

        total_before += len(ctxs)
        total_after  += len(unique)
        dist_after[len(unique)] += 1
        rec["contexts"] = unique

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    removed = total_before - total_after
    print(f"输入文件            : {in_path.name}")
    print(f"输出文件            : {out_path.name}")
    print(f"总记录数            : {len(data)}")
    print(f"含重复 contexts 的  : {records_with_dups} 条")
    print(f"contexts 总量       : {total_before} → {total_after}（移除 {removed} 条重复，压缩 {removed/total_before*100:.1f}%）")
    print(f"去重前 ctx 数量分布 : {dict(sorted(dist_before.items()))}")
    print(f"去重后 ctx 数量分布 : {dict(sorted(dist_after.items()))}")

    zero_ctx = [r.get("id") for r in data if not r.get("contexts")]
    if zero_ctx:
        print(f"⚠️  去重后 contexts 为空的记录 id: {zero_ctx}（这些会被 RAGAS 脚本跳过）")
    else:
        print("✅ 所有记录去重后仍有至少 1 条 context")


if __name__ == "__main__":
    main()
