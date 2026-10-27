"""
长期记忆抽取冒烟测试：直接调用 1.7B 抽取（与线上相同的提示词与过滤规则），检查"该记的记、不该记的不记"。

在 app 容器内运行（项目根目录）：
    docker compose exec app python scripts/test_ltm_extract.py
"""

import asyncio
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config_path import VLLM_CONFIG
from src.core.memory_long.long_term_memory import LongTermMemory

CASES = [
    # (用户输入, 助手回答节选, 期望是否产生记忆)
    ("我是一名开发工程师", "你好！很高兴认识你。有什么可以帮你的吗？", True),
    ("我对花生过敏，下个月要去阿根廷旅行", "了解，出行时注意饮食成分。", True),
    ("我住在上海，平时喜欢听古典音乐，尤其是巴赫", "巴赫是巴洛克时期的代表作曲家。", True),
    ("我在备考雅思，听力总是很差，有什么建议吗？", "可以每天精听 30 分钟真题音频。", True),
    ("双相情感障碍 II 型（Bipolar II disorder）的定义是什么？", "双相情感障碍 II 型是一种以轻躁狂发作和重度抑郁发作为特征的心境障碍。", False),
    ("它是什么疾病？", "它属于心境障碍类精神疾病。", False),
    ("红葡萄酒的制作原料是什么？阿根廷在世界葡萄酒生产中的排名如何？", "红葡萄酒由深色葡萄酿制，阿根廷是全球第五大葡萄酒生产国。", False),
    ("今天天气不错", "是的，适合出门走走。", False),
]


async def main():
    async with httpx.AsyncClient() as http:
        ltm = LongTermMemory.__new__(LongTermMemory)
        ltm.http = http
        ltm.extract_base_url = VLLM_CONFIG["extract_base_url"]
        ltm.extract_model = VLLM_CONFIG["extract_model"]
        ltm.extract_timeout = VLLM_CONFIG["extract_timeout"]
        ltm.api_key = VLLM_CONFIG["api_key"]
        passed = 0
        for user_input, answer, expect in CASES:
            mems = await ltm._extract_memories(user_input, answer)
            got = bool(mems)
            ok = got == expect
            passed += ok
            print(f"{'✅' if ok else '❌'} 期望{'记' if expect else '不记'} | {user_input[:30]}")
            for m in mems or []:
                print(f"     - {m['content']} | {m['content_en']} | {m['memory_type']} {m['importance_score']}")
        print(f"\n通过 {passed}/{len(CASES)}")


if __name__ == "__main__":
    asyncio.run(main())
