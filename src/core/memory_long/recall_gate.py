"""
长期记忆召回门（门 1）：回答当前问题是否需要用户个人背景，不需要则整轮跳过长期记忆召回

为什么需要：余弦门槛挡得住毫不相关的记忆，但"话题相近、答案对任何人都一样"的问题（喜欢星际穿越的用户问"星际穿越的导演是谁"）
余弦分数比真正相关的还高，重排模型也区分不了（标定见 scripts/calibrate_ltm_threshold.py）。只能按问题本身判断要不要召回。

位置：改写之后、召回之前，用改写后的完整问题判断（代词已还原）。模型：Qwen3-1.7B（不占 4B），约 64 ms。
评测（scripts/eval_ltm_gate.py --resolved，2026-10-26 v4，63 题）：
    话题相近 → 不召回 100%（8/8）；知识问题 → 不召回 96%（25/26）；需要个人背景 → 召回 88%（23/26）
    代词追问直接用原话判断会误判为"需要"，改用改写结果后全部判对；1.7B 用 JSON 约束解码会输出大量空白，故只输出 true / false
降级：1.7B 调用失败 / 超时 / 无法解析 → 规则判断（自称或推荐、建议类用语）。误判代价不对称：
    误判为"需要" → 多一次召回，余弦门槛仍会挡住无关记忆；误判为"不需要" → 个性化不足，但不会引入错误信息
"""

import logging
import re
from typing import Callable, Tuple

from src.core.vllm_client import vllm_generate_async

logger = logging.getLogger(__name__)

GATE_SYSTEM_PROMPT = """\
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

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_ANSWER_RE = re.compile(r"^\W*(true|false)\b")
# 降级规则：自称，或推荐 / 建议 / 计划 / 适合类用语
_RULE_RE = re.compile(r"我|本人|咱|\b(?:I|I'm|my|me)\b|推荐|建议|适合|帮我|给我|应该|该怎么|计划|安排|能不能|可以吃|"
                      r"recommend|suggest|should I|plan", re.I)


def parse_answer(raw) -> "bool | None":
    if raw is None:
        return None
    m = _ANSWER_RE.match(_THINK_RE.sub("", raw).strip().lower())
    return (m.group(1) == "true") if m else None


def rule_decide(question: str) -> bool:
    return bool(_RULE_RE.search(question or ""))


class RecallGate:
    def __init__(self, get_http_client: Callable, base_url: str, model: str, api_key: str, timeout: float = 2.0):
        self._http = get_http_client
        self.base_url, self.model, self.api_key, self.timeout = base_url, model, api_key, timeout

    async def decide(self, question: str) -> Tuple[bool, str]:
        """返回 (是否召回长期记忆, 判定来源 model / rule_fallback)"""
        raw = await vllm_generate_async(
            http_client=self._http(), base_url=self.base_url, model=self.model, prompt=f"当前问题：{question}",
            system_prompt=GATE_SYSTEM_PROMPT, api_key=self.api_key, timeout=self.timeout,
            temperature=0.0, max_tokens=16)
        need = parse_answer(raw)
        if need is None:
            logger.warning(f"长期记忆召回门：1.7B 无有效判定（{'调用失败' if raw is None else '无法解析'}），按规则降级")
            return rule_decide(question), "rule_fallback"
        return need, "model"
