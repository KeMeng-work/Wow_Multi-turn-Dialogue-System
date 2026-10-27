"""
RAGAS 指标改进统计显著性检验

比较两轮评估结果，对每个指标执行：
  - Mann-Whitney U 检验（非参数，适用于 [0,1] 有界非正态分布）
  - Bootstrap 95% 置信区间（1000次重采样）
  - Cohen's d 效应量

使用方式：
  # 比较 Round2 batch1 vs batch2
  python scripts/significance_test.py \
    --a data/evaluation/ragas_results_batch_1.json \
    --b data/evaluation/ragas_results_batch_2.json \
    --label-a "Round2-Batch1" --label-b "Round2-Batch2"

  # 比较 Round2 vs Round3（Round3 跑完后用）
  python scripts/significance_test.py \
    --a data/evaluation/ragas_results_batch_2.json \
    --b data/evaluation/ragas_results_batch_3.json \
    --label-a "Round2" --label-b "Round3"
"""

import json
import argparse
import random
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]

EFFECT_LABELS = {
    "negligible": (0.0,  0.2),
    "small":      (0.2,  0.5),
    "medium":     (0.5,  0.8),
    "large":      (0.8,  float("inf")),
}


def load_scores(path: str, metric: str) -> list[float]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    details = data.get("details", data) if isinstance(data, dict) else data
    return [r[metric] for r in details if r.get(metric) is not None]


def bootstrap_ci(values: list[float], n_boot: int = 1000, ci: float = 0.95) -> tuple[float, float, float]:
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[random.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1 - ci) / 2
    lo = means[int(alpha * n_boot)]
    hi = means[int((1 - alpha) * n_boot)]
    return sum(values) / n, lo, hi


def mann_whitney_u(a: list[float], b: list[float]) -> tuple[float, float]:
    """Mann-Whitney U 统计量 + 近似 z-score p 值（正态近似，n>20 时可靠）"""
    na, nb = len(a), len(b)
    combined = sorted([(v, 0) for v in a] + [(v, 1) for v in b])

    # 计算 rank（处理 tie）
    ranks = [0.0] * (na + nb)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) - 1 and combined[j][0] == combined[j + 1][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    rank_sum_a = sum(ranks[k] for k, (_, g) in enumerate(combined) if g == 0)
    u_a = rank_sum_a - na * (na + 1) / 2
    u_b = na * nb - u_a
    u = min(u_a, u_b)

    # 正态近似
    mu_u = na * nb / 2
    sigma_u = math.sqrt(na * nb * (na + nb + 1) / 12)
    z = (u - mu_u) / sigma_u if sigma_u > 0 else 0

    # 双尾 p 值（标准正态近似）
    def norm_cdf(z):
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    p = 2 * min(norm_cdf(z), 1 - norm_cdf(z))
    return u, p


def cohens_d(a: list[float], b: list[float]) -> float:
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    var_a = sum((x - mean_a) ** 2 for x in a) / (len(a) - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (len(b) - 1)
    pooled_std = math.sqrt((var_a + var_b) / 2)
    return (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0.0


def effect_label(d: float) -> str:
    d = abs(d)
    for label, (lo, hi) in EFFECT_LABELS.items():
        if lo <= d < hi:
            return label
    return "large"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a",       required=True, help="对照组 JSON 路径")
    parser.add_argument("--b",       required=True, help="实验组 JSON 路径")
    parser.add_argument("--label-a", default="Group-A")
    parser.add_argument("--label-b", default="Group-B")
    parser.add_argument("--alpha",   type=float, default=0.05, help="显著性水平")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(42)

    print(f"\n{'='*64}")
    print(f"  统计显著性检验")
    print(f"  对照组: {args.label_a}  ({args.a})")
    print(f"  实验组: {args.label_b}  ({args.b})")
    print(f"  显著性水平 α = {args.alpha}")
    print(f"{'='*64}\n")

    for metric in METRICS:
        scores_a = load_scores(args.a, metric)
        scores_b = load_scores(args.b, metric)

        if not scores_a or not scores_b:
            print(f"[{metric}] 数据不足，跳过\n")
            continue

        mean_a, lo_a, hi_a = bootstrap_ci(scores_a)
        mean_b, lo_b, hi_b = bootstrap_ci(scores_b)
        u_stat, p_val     = mann_whitney_u(scores_a, scores_b)
        d                 = cohens_d(scores_a, scores_b)
        delta             = mean_b - mean_a
        sig               = p_val < args.alpha

        print(f"[{metric}]")
        print(f"  {args.label_a:20s}  n={len(scores_a):3d}  mean={mean_a:.4f}  95%CI=[{lo_a:.4f}, {hi_a:.4f}]")
        print(f"  {args.label_b:20s}  n={len(scores_b):3d}  mean={mean_b:.4f}  95%CI=[{lo_b:.4f}, {hi_b:.4f}]")
        print(f"  Δ={delta:+.4f}  U={u_stat:.1f}  p={p_val:.4f}  Cohen's d={d:.3f} ({effect_label(d)})")
        verdict = "✅ 显著提升" if (sig and delta > 0) else \
                  "⚠️  显著下降" if (sig and delta < 0) else \
                  "— 无统计显著差异"
        print(f"  → {verdict}  (p {'<' if sig else '>='} {args.alpha})\n")

    print(f"{'='*64}")
    print("说明：Mann-Whitney U 为非参数检验，适用于 [0,1] 有界非正态分布。")
    print("      Bootstrap CI 基于 1000 次重采样。Cohen's d ≥ 0.5 为中等效应。")


if __name__ == "__main__":
    main()
