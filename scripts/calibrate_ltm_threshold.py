"""
长期记忆召回阈值标定：用线上同款嵌入模型（nomic-embed-text ONNX int8）计算"记忆—问题"相关 / 不相关样例对的余弦相似度，
按两组分布给出阈值建议（不相关组 P95 与相关组 P10 之间）。

对比三种口径：中文原文 / 英文 / 英文 + nomic 任务前缀（线上口径：记忆 search_document:，查询 search_query:）。
2026-10-10 实测：中文两组重叠无法区分；英文 + 前缀在 0.38 处相关保留 88%、不相关放过 12%。
2026-10-26 新增：重排模型（TEI bge-reranker-v2-m3）打分，为召回后的重排验证（门 2）标定阈值；
            补充线上真实记忆与演示问题；新增"话题相近但用不上个人信息"一组，只统计不选阈值。

用法（app 容器内：需要访问 TEI 重排服务）：
    docker compose exec app python scripts/calibrate_ltm_threshold.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config_path import EMBED_CONFIG
from src.core.embedder.onnx_embedder import ONNXEmbedder

# (记忆, 问题)：相关 = 回答该问题时应该用上这条记忆
RELATED = [
    ("用户喜欢硬科幻电影，尤其是星际穿越", "推荐几部电影给我"),
    ("用户喜欢硬科幻电影，尤其是星际穿越", "最近有什么好看的科幻片"),
    ("用户计划下个月去阿根廷旅行", "阿根廷有什么值得去的地方"),
    ("用户计划下个月去阿根廷旅行", "南美洲旅行要注意什么"),
    ("用户是一名后端开发工程师", "我想学点新技术，有什么建议"),
    ("用户是一名后端开发工程师", "程序员怎么保护颈椎"),
    ("用户对花生过敏", "这道菜我能吃吗，里面有花生酱"),
    ("用户对花生过敏", "食物过敏有哪些症状"),
    ("用户喜欢古典音乐，常听巴赫", "推荐一些适合工作时听的音乐"),
    ("用户喜欢古典音乐，常听巴赫", "巴洛克时期有哪些作曲家"),
    ("用户养了一只柯基犬", "狗狗夏天怎么防暑"),
    ("用户养了一只柯基犬", "柯基容易得什么病"),
    ("用户正在备考雅思", "怎么提高英语听力"),
    ("用户住在上海", "周末去哪里玩比较好"),
    ("用户不吃辣", "推荐几道家常菜"),
    ("用户喜欢打篮球", "运动后膝盖疼怎么办"),
]
# 不相关 = 记忆与问题毫无关系，召回只会干扰回答
UNRELATED = [
    ("用户喜欢硬科幻电影，尤其是星际穿越", "双相情感障碍 II 型的定义是什么"),
    ("用户喜欢硬科幻电影，尤其是星际穿越", "红葡萄酒的制作原料是什么"),
    ("用户计划下个月去阿根廷旅行", "为什么古典音乐的定义不限于特定历史时期"),
    ("用户计划下个月去阿根廷旅行", "柯基犬起源于哪里"),
    ("用户是一名后端开发工程师", "沙漠是什么样的地形"),
    ("用户是一名后端开发工程师", "芭蕾舞在哪些地方被教授"),
    ("用户对花生过敏", "Metallica 是哪个国家的乐队"),
    ("用户对花生过敏", "东京是什么时候成为日本首都的"),
    ("用户喜欢古典音乐，常听巴赫", "法国斗牛犬是由什么犬种杂交而来的"),
    ("用户喜欢古典音乐，常听巴赫", "丰田普锐斯是什么类型的汽车"),
    ("用户养了一只柯基犬", "美国哪个政党是最古老的政党"),
    ("用户养了一只柯基犬", "双相情感障碍 II 型的分类依据是什么"),
    ("用户正在备考雅思", "披萨这个词最早什么时候被记录"),
    ("用户住在上海", "铁补充剂的作用是什么"),
    ("用户不吃辣", "雪佛兰科尔维特首代车型什么时候推出"),
    ("用户喜欢打篮球", "Hershey 公司的总部在哪里"),
    # 截图中的实际误召回：知识性"记忆"与无关问题
    ("古典音乐的定义不局限于特定历史时期", "双相情感障碍 II 型好治吗"),
    ("阿根廷是全球第五大葡萄酒生产国", "双相情感障碍 II 型好治吗"),
]

RELATED_EN = [
    ("The user likes hard science fiction movies, especially Interstellar", "Recommend some movies for me"),
    ("The user likes hard science fiction movies, especially Interstellar", "Any good sci-fi films recently"),
    ("The user plans to travel to Argentina next month", "What places are worth visiting in Argentina"),
    ("The user plans to travel to Argentina next month", "What should I pay attention to when traveling in South America"),
    ("The user is a backend software engineer", "I want to learn some new technology, any suggestions"),
    ("The user is a backend software engineer", "How can programmers protect their neck"),
    ("The user is allergic to peanuts", "Can I eat this dish, it contains peanut butter"),
    ("The user is allergic to peanuts", "What are the symptoms of food allergies"),
    ("The user likes classical music and often listens to Bach", "Recommend some music to listen to while working"),
    ("The user likes classical music and often listens to Bach", "Who are the composers of the Baroque period"),
    ("The user has a Corgi dog", "How to keep dogs cool in summer"),
    ("The user has a Corgi dog", "What diseases are Corgis prone to"),
    ("The user is preparing for the IELTS exam", "How to improve English listening"),
    ("The user lives in Shanghai", "Where is a good place to go on the weekend"),
    ("The user does not eat spicy food", "Recommend some home-style dishes"),
    ("The user likes playing basketball", "What to do about knee pain after exercise"),
    # 2026-10-26 补充：线上真实记忆 + 隐式需要个人信息的问题
    ("The user has repeatedly mentioned interest in the Premier League", "Which football matches are worth watching this weekend"),
    ("The user has repeatedly mentioned interest in the Premier League", "Recommend some sports documentaries for me"),
    ("The user is developing AI applications", "What tools would you recommend for my current project"),
    ("The user is developing AI applications", "How should I evaluate the system I am building"),
    ("The user is a Python backend developer", "Which web framework should I learn next"),
    ("The user lives in Shanghai", "Where can I go for a short trip nearby this weekend"),
    ("The user has a Corgi dog", "What dog food is good for my dog"),
    ("The user is preparing for the IELTS exam", "How should I plan my study schedule this month"),
    ("The user does not eat spicy food", "What should I order at a Sichuan restaurant"),
    ("The user is allergic to peanuts", "Is this Thai satay dish safe for me to eat"),
    ("The user likes hard science fiction movies, especially Interstellar", "What should I watch tonight"),
    ("The user plans to travel to Argentina next month", "What currency should I bring for my trip"),
    ("The user likes playing basketball", "What shoes are good for my sport"),
]
UNRELATED_EN = [
    ("The user likes hard science fiction movies, especially Interstellar", "What is the definition of bipolar II disorder"),
    ("The user likes hard science fiction movies, especially Interstellar", "What are the raw materials of red wine"),
    ("The user plans to travel to Argentina next month", "Why is the definition of classical music not limited to a specific historical period"),
    ("The user plans to travel to Argentina next month", "Where did the Corgi originate"),
    ("The user is a backend software engineer", "What kind of terrain is a desert"),
    ("The user is a backend software engineer", "Where is ballet taught"),
    ("The user is allergic to peanuts", "Which country is Metallica from"),
    ("The user is allergic to peanuts", "When did Tokyo become the capital of Japan"),
    ("The user likes classical music and often listens to Bach", "Which breeds were crossed to create the French Bulldog"),
    ("The user likes classical music and often listens to Bach", "What type of car is the Toyota Prius"),
    ("The user has a Corgi dog", "Which is the oldest political party in the United States"),
    ("The user has a Corgi dog", "What is the classification basis of bipolar II disorder"),
    ("The user is preparing for the IELTS exam", "When was the word pizza first recorded"),
    ("The user lives in Shanghai", "What is the function of iron supplements"),
    ("The user does not eat spicy food", "When was the first generation Chevrolet Corvette launched"),
    ("The user likes playing basketball", "Where is the headquarters of the Hershey Company"),
    # 2026-10-26 线上实际误召回（相似度 0.39 / 0.388，越过旧阈值 0.38）
    ("The user has repeatedly mentioned interest in the Premier League", "Where is the Kingdom of Denmark located?"),
    ("The user has repeatedly mentioned interest in the Premier League", "What is the capital of the Kingdom of Denmark?"),
    # 2026-10-26 补充：线上真实记忆 × 演示中的知识问题（含地理、欧洲、城市等表面相近的难负例）
    ("The user has repeatedly mentioned interest in the Premier League", "Who invented the piano?"),
    ("The user has repeatedly mentioned interest in the Premier League", "When did skateboarding originate?"),
    ("The user is developing AI applications", "Who invented the piano?"),
    ("The user is developing AI applications", "What equipment is needed for fly fishing?"),
    ("The user is a Python backend developer", "Where is the headquarters of the Hershey Company?"),
    ("The user is a Python backend developer", "What is the relationship between Denmark and Sweden?"),
    ("The user lives in Shanghai", "What is the capital of the Kingdom of Denmark?"),
    ("The user lives in Shanghai", "Where is the Kingdom of Denmark located?"),
    ("The user plans to travel to Argentina next month", "Where is the Kingdom of Denmark located?"),
    ("The user plans to travel to Argentina next month", "What is the natural occurrence rate of red hair in humans?"),
    ("The user is allergic to peanuts", "What are the duties of a pharmacist?"),
    ("The user likes playing basketball", "When did skateboarding originate?"),
]
# 话题相近、但回答这个知识问题用不上个人信息：向量和重排都可能打高分（话题相关 ≠ 需要个性化）。
# 单独统计，不参与选阈值；用来判断"要不要召回"的门 1 是否必要
TOPICAL_EN = [
    ("The user likes classical music and often listens to Bach", "Who invented the piano?"),
    ("The user likes hard science fiction movies, especially Interstellar", "Who directed Interstellar?"),
    ("The user has repeatedly mentioned interest in the Premier League", "When was the Premier League founded?"),
    ("The user plans to travel to Argentina next month", "What is Argentina's rank in world wine production?"),
    ("The user is a Python backend developer", "Who created the Python programming language?"),
    ("The user lives in Shanghai", "What is the population of Shanghai?"),
    ("The user has a Corgi dog", "Where did the Corgi breed originate?"),
    ("The user is allergic to peanuts", "Where did peanuts originate?"),
]


def sims(emb, pairs, doc_prefix="", query_prefix=""):
    out = []
    for mem, q in pairs:
        a, b = np.array(emb.embed_query(doc_prefix + mem)), np.array(emb.embed_query(query_prefix + q))
        out.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    return np.array(out)


def rerank_scores(pairs):
    """线上同款重排（TEI bge-reranker-v2-m3，sigmoid 0–1）：问题 vs 记忆英文正文，逐对打分"""
    import requests
    from config.config_path import RERANKER_CONFIG
    out = []
    for mem, q in pairs:
        r = requests.post(RERANKER_CONFIG["tei_url"], json={"query": q, "texts": [mem], "truncate": True}, timeout=30)
        r.raise_for_status()
        out.append(float(r.json()[0]["score"]))
    return np.array(out)


def _fmt(x):
    return (f"min {x.min():.3f}  P10 {np.percentile(x, 10):.3f}  median {np.median(x):.3f}  "
            f"P90 {np.percentile(x, 90):.3f}  max {x.max():.3f}")


def _report(label, rel, unr, top, grid):
    print(f"\n== {label} ==")
    print(f"相关     n={len(rel):>2}  {_fmt(rel)}")
    print(f"不相关   n={len(unr):>2}  {_fmt(unr)}")
    if top is not None:
        print(f"话题相近 n={len(top):>2}  {_fmt(top)}   （不参与选阈值，用于判断门 1 是否必要）")
    for t in grid:
        line = f"阈值 {t:.3f}：相关保留 {np.mean(rel >= t):5.0%}  不相关放过 {np.mean(unr >= t):5.0%}"
        if top is not None:
            line += f"  话题相近放过 {np.mean(top >= t):5.0%}"
        print(line)


def main():
    emb = ONNXEmbedder(model_path=EMBED_CONFIG["model_path"], onnx_file=EMBED_CONFIG["onnx_file"],
                       providers=EMBED_CONFIG["providers"], intra_op_num_threads=2)
    cos_grid = np.arange(0.30, 0.62, 0.02)
    for label, rp, up, tp, dp, qp in [
        ("中文", RELATED, UNRELATED, None, "", ""),
        ("英文", RELATED_EN, UNRELATED_EN, None, "", ""),
        ("英文+前缀（线上口径）", RELATED_EN, UNRELATED_EN, TOPICAL_EN, "search_document: ", "search_query: "),
    ]:
        _report(label, sims(emb, rp, dp, qp), sims(emb, up, dp, qp),
                sims(emb, tp, dp, qp) if tp else None, cos_grid)

    # 门 2：重排验证的阈值标定（sigmoid 分数跨度大，用对数刻度网格）
    rel, unr, top = rerank_scores(RELATED_EN), rerank_scores(UNRELATED_EN), rerank_scores(TOPICAL_EN)
    _report("重排 bge-reranker-v2-m3（门 2）", rel, unr, top,
            [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
    print("\n重排分数明细（便于核对边界样本）：")
    for name, pairs, scores in (("相关", RELATED_EN, rel), ("不相关", UNRELATED_EN, unr), ("话题相近", TOPICAL_EN, top)):
        for (mem, q), s in sorted(zip(pairs, scores), key=lambda x: x[1]):
            print(f"  [{name}] {s:.4f}  {q[:50]}  ←  {mem[:50]}")

if __name__ == "__main__":
    main()
