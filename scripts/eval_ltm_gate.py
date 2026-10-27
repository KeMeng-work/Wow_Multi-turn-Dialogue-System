"""
长期记忆召回门 1 评测：判断"回答当前问题是否需要用户个人背景"，比较 1.7B 与 4B（只读，不改线上）

背景（docs/long_term_memory_write_gate.md §5）：余弦门槛挡得住毫不相关的记忆，但"话题相近、回答用不上个人信息"的问题
（喜欢星际穿越的用户问"星际穿越的导演是谁"）余弦分数比真正相关的还高；重排模型也区分不了。只能按问题本身判断要不要召回。

部署形态的差别：
  1.7B  与 4B 改写并行，只能看到用户原话 + 上一轮问题，不增加用户等待（改写约 1 秒以上）
  4B    作为改写 JSON 的一个字段，可看到完整上下文；此处单独调用只为比较判断质量

样本：yes = 推荐 / 建议 / 计划 / 与"我"有关（含隐式：周末去哪玩）；
      no_knowledge = 客观知识问题（含今天演示的问题与多轮追问）；
      no_topical = 话题与用户兴趣相近但答案对任何人都一样；no_chitchat = 寒暄
指标：yes 召回率（目标 ≥ 70%）、no 准确率（目标 ≥ 95%，分知识 / 话题相近 / 寒暄）、耗时均值与 P95、解析失败

运行（app 容器内）：
    docker compose exec app python scripts/eval_ltm_gate.py              # 1.7B 与 4B 各跑一遍
    docker compose exec app python scripts/eval_ltm_gate.py --model 1.7b
"""

import argparse
import asyncio
import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "evaluation"

SYSTEM_PROMPT = """\
你判断：回答用户当前的问题时，知道用户本人的背景（职业、所在地、偏好、饮食禁忌、宠物、计划、正在做的事等）会不会让回答更合适。
- 需要（true）：推荐、建议、计划、选择，或问题与"我"的情况、能力、安排有关，例如"给我推荐几本书""这个我能吃吗""周末去哪玩比较好""我该学什么"
- 不需要（false）：客观知识或事实问题，答案对任何人都一样——即使话题与用户可能的兴趣相关，例如"长城有多长""贝多芬是哪国人"
- 寒暄、致谢、确认：false
只输出一个词：true 或 false

示例：
当前问题：推荐几本适合入门的书 → true
当前问题：我下班后适合做什么运动 → true
当前问题：长城有多长？ → false
当前问题：贝多芬是哪国人？ → false"""
# v3 曾加"代词追问按指代对象判断"规则 + 示例：代词追问仍判错，1.7B 话题相近从 100% 降到 75% → 已撤回

# 不用 JSON 约束解码：1.7B 在约束解码下会先输出大量空白，耗尽 max_tokens 后截断（10-26 首次评测 58/63 解析失败）

# (分组, 当前问题, 上一轮问题或 None)
CASES = [
    # ── 需要个人背景 ──
    ("yes", "给我推荐几部电影", None),
    ("yes", "最近有什么好看的科幻片值得看", None),
    ("yes", "今晚看点什么好", None),
    ("yes", "推荐一些工作时听的音乐", None),
    ("yes", "我想学点新技术，有什么建议", None),
    ("yes", "下一步我该学哪个 Web 框架", None),
    ("yes", "我现在的项目适合用什么工具", None),
    ("yes", "这道菜里有花生酱，我能吃吗", None),
    ("yes", "这道泰式沙爹对我来说安全吗", None),
    ("yes", "去川菜馆我该点什么", None),
    ("yes", "推荐几道家常菜", None),
    ("yes", "这周末附近有什么地方可以短途旅行", None),
    ("yes", "周末去哪里玩比较好", None),
    ("yes", "这个月的学习计划该怎么安排", None),
    ("yes", "怎么提高我的英语听力", None),
    ("yes", "我家狗该吃什么狗粮", None),
    ("yes", "夏天怎么给我的狗防暑", None),
    ("yes", "这周末有哪些球赛值得看", None),
    ("yes", "给我推荐一些体育纪录片", None),
    ("yes", "我这次旅行该带什么货币", None),
    ("yes", "去南美旅行我要注意什么", None),
    ("yes", "我运动后膝盖疼怎么办", None),
    ("yes", "什么鞋适合我平时的运动", None),
    ("yes", "根据我的情况，你觉得我适合转行做 AI 吗", None),
    ("yes", "那它适合我练吗？", "克拉夫·马加是什么格斗体系？"),
    ("yes", "帮我制定一个健身计划", None),
    # ── 客观知识问题（含演示问题与多轮追问）──
    ("no_knowledge", "丹麦王国的地理位置在哪里？", None),
    ("no_knowledge", "它的首都在哪里？", "丹麦王国的地理位置在哪里？"),
    ("no_knowledge", "丹麦和瑞典是什么关系？", None),
    ("no_knowledge", "钢琴是谁发明的？", None),
    ("no_knowledge", "他是哪个国家的人？", "钢琴是谁发明的？"),
    ("no_knowledge", "好时公司总部在哪里？", None),
    ("no_knowledge", "它有哪些主要产品？", "好时公司总部在哪里？"),
    ("no_knowledge", "滑板运动起源于哪里？", None),
    ("no_knowledge", "飞蝇钓需要什么装备？", None),
    ("no_knowledge", "红发在人类中的自然出现概率是多少？", None),
    ("no_knowledge", "药剂师的职责是什么？", None),
    ("no_knowledge", "历史上第一次跳伞是由谁完成的？", None),
    ("no_knowledge", "克拉夫·马加是什么格斗体系？", None),
    ("no_knowledge", "双相情感障碍 II 型的定义是什么？", None),
    ("no_knowledge", "红葡萄酒的制作原料是什么？", None),
    ("no_knowledge", "沙漠是什么样的地形？", None),
    ("no_knowledge", "Metallica 是哪个国家的乐队？", None),
    ("no_knowledge", "东京是什么时候成为日本首都的？", None),
    ("no_knowledge", "美国最古老的政党是哪个？", None),
    ("no_knowledge", "披萨这个词最早什么时候被记录？", None),
    ("no_knowledge", "铁补充剂的作用是什么？", None),
    ("no_knowledge", "雪佛兰Corvette经历了几代发展？", None),
    ("no_knowledge", "讴歌是哪家公司推出的汽车品牌？", None),
    ("no_knowledge", "为什么说古典音乐的定义不仅限于特定的历史时期？", None),
    ("no_knowledge", "回到丹麦，它的国家性质是什么？", "钢琴是谁发明的？"),
    ("no_knowledge", "法国斗牛犬是由什么犬种杂交而来的？", None),
    # ── 话题相近，但答案对任何人都一样 ──
    ("no_topical", "《星际穿越》的导演是谁？", None),
    ("no_topical", "英超联赛是哪一年成立的？", None),
    ("no_topical", "阿根廷在世界葡萄酒生产中排第几？", None),
    ("no_topical", "Python 语言是谁创造的？", None),
    ("no_topical", "上海的人口是多少？", None),
    ("no_topical", "柯基犬起源于哪里？", None),
    ("no_topical", "花生原产于哪里？", None),
    ("no_topical", "巴洛克时期有哪些作曲家？", None),
    # ── 寒暄 ──
    ("no_chitchat", "你好", None),
    ("no_chitchat", "谢谢，我明白了", None),
    ("no_chitchat", "好的", None),
]

# --resolved：门放在改写之后，用改写后的完整问题判断（代词已还原，取自 10-26 端到端评测中 4B 的实际改写）
RESOLVED = {
    "那它适合我练吗？": "克拉夫·马加适合我练吗？",
    "它的首都在哪里？": "丹麦王国的首都在哪里？",
    "他是哪个国家的人？": "巴托洛梅奥·克里斯托福里是哪个国家的人？",
    "它有哪些主要产品？": "好时公司的主要产品是什么？",
    "回到丹麦，它的国家性质是什么？": "丹麦王国的国家性质是什么？",
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def _parse(raw):
    if raw is None:
        return None
    text = _THINK_RE.sub("", raw).strip().lower()
    m = re.match(r'^\W*(true|false)\b', text)
    return (m.group(1) == "true") if m else None


async def run_model(name, url, model, api_key, http, repeat, resolved=False):
    from src.core.vllm_client import vllm_generate_async
    try:      # 线上已部署召回门时，用线上同一份提示词（与本文件 v2 / v4 所用文本一致）
        from src.core.memory_long.recall_gate import GATE_SYSTEM_PROMPT as system_prompt
    except ImportError:
        system_prompt = SYSTEM_PROMPT
    rows = []
    for group, q, prev in CASES:
        if resolved:
            q, prev = RESOLVED.get(q, q), None
        prompt = (f"上一轮问题：{prev}\n" if prev else "") + f"当前问题：{q}"
        for _ in range(repeat):
            t0 = time.perf_counter()
            raw = await vllm_generate_async(http_client=http, base_url=url, model=model, prompt=prompt,
                                            system_prompt=system_prompt, api_key=api_key, timeout=30,
                                            temperature=0.0, max_tokens=16)
            ms = round((time.perf_counter() - t0) * 1000)
            pred = _parse(raw)
            ok = (pred is True) if group == "yes" else (pred is False)
            rows.append({"group": group, "q": q, "prev": prev, "pred": pred, "ok": ok, "ms": ms, "raw": raw})
            if not ok:
                print(f"  [{name}] 错 {group:<12} 判为 {pred}  {q}")
    return rows


def summarize(rows):
    def acc(g):
        sel = [r for r in rows if r["group"] == g]
        return round(sum(r["ok"] for r in sel) / max(1, len(sel)), 3), len(sel)
    ms = sorted(r["ms"] for r in rows)
    no = [r for r in rows if r["group"] != "yes"]
    return {
        "yes_recall": acc("yes"),
        "no_accuracy": (round(sum(r["ok"] for r in no) / max(1, len(no)), 3), len(no)),
        "no_knowledge": acc("no_knowledge"), "no_topical": acc("no_topical"), "no_chitchat": acc("no_chitchat"),
        "parse_fail": sum(r["pred"] is None for r in rows),
        "ms_mean": round(sum(ms) / len(ms)), "ms_p95": ms[max(0, math.ceil(len(ms) * 0.95) - 1)],
    }


async def main(args) -> int:
    sys.path.insert(0, str(ROOT))
    import httpx
    from config.config_path import VLLM_CONFIG

    targets = {"1.7b": (VLLM_CONFIG["extract_base_url"], VLLM_CONFIG["extract_model"]),
               "4b": (VLLM_CONFIG["rewrite_base_url"], VLLM_CONFIG["rewrite_model"])}
    names = ["1.7b", "4b"] if args.model == "both" else [args.model]
    http = httpx.AsyncClient(timeout=60, trust_env=False)
    results = {}
    try:
        for n in names:
            url, model = targets[n]
            print(f"\n== {n}（{model}）==")
            rows = await run_model(n, url, model, VLLM_CONFIG["api_key"], http, args.repeat, args.resolved)
            results[n] = {"model": model, "summary": summarize(rows), "rows": rows}
    finally:
        await http.aclose()

    print(f"\n{'模型':<6} {'yes召回':>10} {'no准确':>10} {'知识':>10} {'话题相近':>10} {'寒暄':>8} {'解析失败':>6} {'均值ms':>7} {'P95ms':>7}")
    for n, r in results.items():
        s = r["summary"]
        f = lambda t: f"{t[0]:.0%}({t[1]})"
        print(f"{n:<6} {f(s['yes_recall']):>10} {f(s['no_accuracy']):>10} {f(s['no_knowledge']):>10} "
              f"{f(s['no_topical']):>10} {f(s['no_chitchat']):>8} {s['parse_fail']:>6} {s['ms_mean']:>7} {s['ms_p95']:>7}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"ltm_gate_{args.tag}.json"
    path.write_text(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告：{path}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="长期记忆召回门 1 评测（1.7B vs 4B）")
    ap.add_argument("--model", choices=("1.7b", "4b", "both"), default="both")
    ap.add_argument("--repeat", type=int, default=1, help="温度 0，默认每条 1 次")
    ap.add_argument("--resolved", action="store_true", help="用改写后的完整问题判断（门放在改写之后）")
    ap.add_argument("--tag", default="run")
    sys.exit(asyncio.run(main(ap.parse_args())))
