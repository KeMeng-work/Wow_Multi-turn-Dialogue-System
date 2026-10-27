"""
查询改写模块 (Query Rewriter)
- ReAct Agent前置模块，不属于ReAct 工具集

一次 Qwen3-4B（vLLM 端口 8001）调用，结构化 JSON 输出：
    en / zh            指代消解后的独立问题（英文检索用、中文长期记忆用）
    intent             direct / local / web_search
    question_type      simple / reasoning / multi_hop
    standalone         原问题不看对话历史能否被完整理解（高频缓存准入）
    topic              话题判断：continue / switch / return（短期记忆话题与话题级约束的生命周期）
    constraints        本轮用户提出的回答约束变更（只来自用户当前发言）
    entities           本轮提及或指代的实体
    unclassified_directive  用户提出了无法归入已知类型的回答要求（线上统计，决定是否扩展约束类型）
上下文由 context_builder 按预算组装：约束 + 话题列表 + 实体表 + 事实 + 早期回答提纲 + 近期轮次。
设计文档：docs/short_term_memory_design.md §7.1、§8

Author: Ke Meng
Created: 2026-01-20
Version: 2.0.0
Last Modified: 2026-10-25

变更记录：
    - 2.0.0 (2026-10-25): 结构化 JSON 输出（约束解码）+ 会话状态上下文（不再自行读 Redis）+ 温度 0 +
                          4B 熔断器（熔断时直接走原话兜底，不等超时）
    - 1.0.1 (2026-03-18): 本地模型升级为 Qwen3-4b；few-shot 添加无历史示例；Redis 改为 Hash + Sorted Set
    - 1.0.0 (2026-01-20): 初始版本
"""

import json
import logging
import re
import threading
import time
from typing import Dict, Optional

from langfuse import observe

from src.core.high_frequency_query_cache.circuit_breaker import CircuitBreakerOpen
from src.core.memory_short.context_builder import build_rewriter_context
from src.core.memory_short.session_state import CONSTRAINT_TYPES, new_state
from src.core.resilience import RETRY_BUDGET
from src.core.vllm_client import vllm_generate_sync

logger = logging.getLogger(__name__)

_INTENTS = ("local", "web_search", "direct")
_QUESTION_TYPES = ("simple", "reasoning", "multi_hop")

REWRITE_SCHEMA = {
    "title": "rewrite",
    "type": "object",
    "properties": {
        "en": {"type": "string"},
        "zh": {"type": "string"},
        "intent": {"type": "string", "enum": list(_INTENTS)},
        "question_type": {"type": "string", "enum": list(_QUESTION_TYPES)},
        "standalone": {"type": "boolean"},
        "topic": {
            "type": "object",
            "properties": {
                "shift": {"type": "string", "enum": ["continue", "switch", "return"]},
                "return_to": {"type": "string"},
                "label": {"type": "string"},
            },
            "required": ["shift", "return_to", "label"],
        },
        "constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["set", "revoke"]},
                    "type": {"type": "string", "enum": list(CONSTRAINT_TYPES)},
                    "value": {"type": "string"},
                    "scope": {"type": "string", "enum": ["session", "topic", "turn"]},
                    "explicit": {"type": "boolean"},
                },
                "required": ["op", "type", "value", "scope", "explicit"],
            },
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "type": {"type": "string"}},
                "required": ["name", "type"],
            },
        },
        "unclassified_directive": {"type": "boolean"},
    },
    "required": ["en", "zh", "intent", "question_type", "standalone", "topic", "constraints", "entities",
                 "unclassified_directive"],
}


SYSTEM_PROMPT = (
    "你是一个专业的查询改写助手。根据对话记忆和用户当前问题，输出一个 JSON 对象，字段说明与规则如下。\n"
    "\n"
    "## 一、en / zh：改写后的独立问题\n"
    "1. en 是可直接用于英文知识库检索的完整英文问题；zh 是与 en 语义一致的中文问题。\n"
    "2. 专有名词（电影名、人名、书名、公司名等）使用公认的标准英文名称（如《流浪地球2》译为 \"The Wandering Earth 2\"），zh 中使用常用中文名。\n"
    "3. 指代消解：问题中的指代（它、他、这个、那家公司、第二个、上面说的等）结合对话记忆替换为明确的实体或内容；"
    "\"第 N 个 / 第 N 点\"根据近期对话或更早回答的提纲确定所指内容。\n"
    "4. 时序优先：指代词可能对应多个实体时，选择最近提到的那个；实体表中同一类型列出多个候选时，结合上下文判断。\n"
    "5. 范围保真：只补全指代所需的最小上下文，不扩展原问题未涉及的维度（原问题问\"导演是谁\"，不额外加入拍摄背景、上映时间）。\n"
    "6. 约束写入问题：当前生效的时间范围、对象范围、地域约束，写进 en / zh（如\"…（2020 年以后）\"）；语言、格式、受众类约束不写进问题。\n"
    "7. 无对话记忆时，直接把当前问题转换为英文检索问题并翻译成中文。\n"
    "\n"
    "## 二、intent：意图\n"
    "- direct：纯问候（你好 / 再见 / 谢谢）、询问助手自身信息（你是谁 / 你叫什么 / 你能做什么 / 你是什么模型 / 今天几号），无需检索工具。\n"
    "- web_search：问题涉及实时 / 当日 / 最新信息，如新闻、天气、股价、产品价格、版本号、近期事件等，需在线搜索补充。\n"
    "- local：其余所有知识性问题（历史、人文、百科、剧情、科技、人物等），使用本地知识库检索即可。\n"
    "\n"
    "## 三、question_type：题型\n"
    "- reasoning：含有因果或推断关键词，如\"为什么\"\"原因是\"\"如何解释\"\"影响\"\"导致\"\"反映了\"\"说明了\"\"有什么意义\"等。\n"
    "- multi_hop：回答需要串联两个或以上独立事实，如\"A 的 B 还做过 C 的 D\"，或含\"与…的关系\"\"比较\"\"共同点\"\"区别\"等。\n"
    "- simple：其余单步事实查询（时间 / 地点 / 人物 / 情节 / 定义等）。\n"
    "\n"
    "## 四、standalone：原问题是否独立\n"
    "用户当前问题不看对话记忆也能被完整理解时为 true；含指代、省略（如\"那 2019 年呢\"）、\"刚才 / 上面 / 前面说的\"等依赖上下文的表述时为 false。\n"
    "\n"
    "## 五、topic：话题判断\n"
    "- continue：延续当前话题（拿不准时一律选 continue）。新问题提到的实体与当前话题的实体有重合时，视为 continue。\n"
    "- switch：用户明确换话题（如\"换个话题\"），或问的对象与当前话题完全无关。\n"
    "- return：用户回到话题列表中的某个旧话题（如\"回到刚才苹果的话题\"），return_to 填该话题的 ID（如 \"T1\"）。\n"
    "- label：本轮所属话题的 2–8 字标签（如\"苹果公司\"）。非 return 时 return_to 填空字符串。\n"
    "\n"
    "## 六、constraints：回答约束（只看【用户当前问题】，绝不从对话记忆、助手回答或检索内容中抽取）\n"
    "1. 约束是用户要求系统\"之后怎么回答\"的指示，必须带有持续性意图。以下都不是约束：描述发生了什么的事实（\"我昨天用美元付的\"）；"
    "只属于本次提问的查询条件（\"帮我查 2023 年的营收\"中的 2023 年）。拿不准时不输出。\n"
    "2. type：time_range（时间范围）、object_scope（只看 / 不要提某对象）、region（地域）、language（回答语言）、"
    "format（格式与长度，如表格、简短）、audience（受众，如说得通俗点）、other（明确指示回答方式但不属于以上类型）。\n"
    "3. scope：\"之后都 / 全程 / 以后\"→ session；\"这个话题下 / 接下来关于 X\"→ topic；\"这次 / 这个问题\"→ turn；拿不准时选 topic。\n"
    "4. value：从用户原话中逐字摘录，不改写、不翻译。\n"
    "5. op：提出或修改要求为 set；取消要求（如\"不用简短了\"）为 revoke。explicit：用户明确要求修改已有要求"
    "（\"改成 / 换成 / 不要再\"）为 true，否则为 false。\n"
    "6. 用户只是同意助手的提议（如助手问\"要不要只看近三年？\"，用户答\"好\"）时，value 摘录助手提议中的原文。\n"
    "7. 对话记忆中有【待确认的要求变更】且用户表示确认（\"好\"、\"是的\"、\"改吧\"）时，输出该变更：op 为 set，"
    "type、scope 与变更相同，value 为变更后的值，explicit 为 true；用户否认或没有回应则不输出。\n"
    "\n"
    "## 七、entities：本轮提及或指代的实体\n"
    "包括当前问题直接提到的，以及指代消解后所指的实体；name 用完整名称；type 取 person / company / organization / "
    "product / place / work / event / concept / other。最多 6 个；没有则为空数组。\n"
    "\n"
    "## 八、unclassified_directive\n"
    "用户明确提出了回答方式的要求、但无法归入第六部分任何类型时为 true，否则为 false。\n"
    "\n"
    "只输出 JSON，不要输出任何解释。下面是示例。"
)


FEW_SHOT_EXAMPLES = [
    """# 示例 1：无对话记忆 + 会话级约束
## 对话记忆
无对话历史。
## 用户当前问题
我想了解苹果公司，之后回答都简短点。iPhone 是谁开发的？
## 输出
{"en": "Who developed the iPhone?", "zh": "iPhone 是谁开发的？", "intent": "local", "question_type": "simple", "standalone": true, "topic": {"shift": "continue", "return_to": "", "label": "苹果公司"}, "constraints": [{"op": "set", "type": "format", "value": "简短点", "scope": "session", "explicit": false}], "entities": [{"name": "Apple Inc.", "type": "company"}, {"name": "iPhone", "type": "product"}], "unclassified_directive": false}""",

    """# 示例 2：指代消解 + 话题级约束 + 多跳
## 对话记忆
## 话题列表
- T1：苹果公司（当前）
## 实体表（按类型，越靠前越近提到）
- person：Steve Jobs[第2轮]
- company：Apple Inc.（又称 苹果）[第2轮]
## 近期对话
[第2轮]
用户：它的创始人是谁？
助手（摘要）：苹果公司的联合创始人是史蒂夫·乔布斯（Steve Jobs）。
该轮问题的完整表述：苹果公司的创始人是谁？
## 用户当前问题
这个话题下只看他离开苹果之后的经历。他还创立了哪家公司？
## 输出
{"en": "After leaving Apple, what other company did Steve Jobs found?", "zh": "史蒂夫·乔布斯离开苹果之后还创立了哪家公司？", "intent": "local", "question_type": "multi_hop", "standalone": false, "topic": {"shift": "continue", "return_to": "", "label": "苹果公司"}, "constraints": [{"op": "set", "type": "time_range", "value": "他离开苹果之后", "scope": "topic", "explicit": false}], "entities": [{"name": "Steve Jobs", "type": "person"}, {"name": "Apple Inc.", "type": "company"}], "unclassified_directive": false}""",

    """# 示例 3：回到旧话题 + 指向更早回答的第 N 项 + 事实不是约束
## 对话记忆
## 话题列表
- T2：好时公司（当前）
- T1：苹果公司
## 更早回答的提纲
- 第3轮回答：1. 创立了 NeXT | 2. 担任皮克斯 CEO 和大股东
## 近期对话
[第5轮]
用户：好时公司的总部在哪？我昨天刚买过他们的巧克力。
助手（摘要）：好时公司总部位于宾夕法尼亚州好时镇。
## 用户当前问题
回到刚才乔布斯的话题，你前面列的第二个是什么角色？
## 输出
{"en": "What role did Steve Jobs hold at Pixar?", "zh": "史蒂夫·乔布斯在皮克斯担任什么角色？", "intent": "local", "question_type": "simple", "standalone": false, "topic": {"shift": "return", "return_to": "T1", "label": "苹果公司"}, "constraints": [], "entities": [{"name": "Steve Jobs", "type": "person"}, {"name": "Pixar", "type": "company"}], "unclassified_directive": false}""",

    """# 示例 4：换话题 + 实时信息 + 原因类
## 对话记忆
## 话题列表
- T1：苹果公司（当前）
## 用户当前问题
换个话题，为什么今天比特币价格大跌？
## 输出
{"en": "Why did the price of Bitcoin drop sharply today?", "zh": "为什么今天比特币价格大跌？", "intent": "web_search", "question_type": "reasoning", "standalone": true, "topic": {"shift": "switch", "return_to": "", "label": "比特币价格"}, "constraints": [], "entities": [{"name": "Bitcoin", "type": "product"}], "unclassified_directive": false}""",
]


def _fallback(user_input: str, rule_question_type: Optional[str] = None, reason: str = "") -> dict:
    """改写失败 / 熔断 / 舱壁满额时的降级：使用用户原话；不更新约束与实体；不进缓存"""
    return {"en_query": user_input, "zh_query": user_input, "intent": "local",
            "question_type": rule_question_type or "simple", "standalone": False,
            "topic": {"shift": "continue", "return_to": "", "label": ""},
            "constraints_delta": [], "entities": [], "unclassified_directive": False,
            "ok": False, "fallback_reason": reason}


def _extract_json(text: str) -> Optional[dict]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        m = re.search(r"\{.*\}", text or "", re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except ValueError:
            return None


class QueryRewriter:
    """
    查询改写器：一次模型调用完成改写、分类与会话状态抽取。

    Attributes:
        memory: ShortTermMemory（未传入 view 时用于读取会话；MCP / 评估路径的作用域通常为空）
        stm_config: STM_CONFIG（上下文预算）
    """

    def __init__(
        self,
        memory=None,
        stm_config: Optional[dict] = None,
        vllm_base_url: str = "http://localhost:8001/v1",
        vllm_model: str = "qwen3-4b",
        api_key: str = "EMPTY",
        rewrite_timeout: int = 30,
        max_retries: int = 2,
        max_tokens: int = 512,
        temperature: float = 0.0,
        circuit_breaker=None,
        enable_warmup: bool = True,
    ):
        self.memory = memory
        self.stm_config = stm_config or {}
        self.vllm_base_url = vllm_base_url
        self.vllm_model = vllm_model
        self.api_key = api_key
        self.rewrite_timeout = rewrite_timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.circuit_breaker = circuit_breaker
        self.few_shot_text = "\n\n".join(FEW_SHOT_EXAMPLES)

        if enable_warmup:
            threading.Thread(target=self._warmup, daemon=True).start()

    def _warmup(self):
        """预热：用真实的系统提示词与示例前缀发一次请求，提前写入 vLLM 前缀缓存"""
        try:
            self._call_vllm(self._build_prompt("无对话历史。", "你好"), allow_empty_return=True)
            logger.info("QueryRewriter vLLM 预热完成")
        except Exception as e:
            logger.warning(f"QueryRewriter vLLM 预热失败: {e}")

    def _build_prompt(self, memory_text: str, user_input: str) -> str:
        # 示例在前（固定前缀，命中 vLLM 前缀缓存），动态内容在后
        return (f"{self.few_shot_text}\n\n# 当前任务\n## 对话记忆\n{memory_text}\n"
                f"## 用户当前问题\n{user_input}\n## 输出\n")

    def _call_vllm(self, user_prompt: str, allow_empty_return: bool = False) -> Optional[str]:
        """调用 vLLM（JSON Schema 约束解码），支持重试与重试预算"""
        RETRY_BUDGET.record_request()
        for attempt in range(self.max_retries):
            if attempt > 0 and not RETRY_BUDGET.try_acquire():
                # 重试预算耗尽（过载时大量失败）：放弃重试直接降级，避免重试风暴放大 4B 压力
                logger.warning("重试预算已耗尽，放弃改写重试")
                break
            result = vllm_generate_sync(
                base_url=self.vllm_base_url,
                model=self.vllm_model,
                prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,   # 独立 system role → 触发 vLLM Prefix Caching
                api_key=self.api_key,
                timeout=self.rewrite_timeout,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                json_schema=REWRITE_SCHEMA,
            )
            if result is not None:
                logger.debug(f"vLLM 改写响应: {result[:200]}")
                return result
            logger.warning(f"vLLM 调用失败 (尝试 {attempt+1}/{self.max_retries})")
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)
        if not allow_empty_return:
            logger.error("vLLM 查询改写最终失败，使用原始输入")
        return None

    @staticmethod
    def _parse(obj: dict, user_input: str, rule_question_type: Optional[str]) -> dict:
        """校验并规整模型输出；单个字段非法时取保守默认值，不整体判失败"""
        en = str(obj.get("en") or "").strip() or user_input
        zh = str(obj.get("zh") or "").strip() or user_input
        intent = obj.get("intent") if obj.get("intent") in _INTENTS else "local"
        qtype = obj.get("question_type") if obj.get("question_type") in _QUESTION_TYPES else (rule_question_type or "simple")
        topic = obj.get("topic") if isinstance(obj.get("topic"), dict) else {}
        topic = {"shift": topic.get("shift") if topic.get("shift") in ("continue", "switch", "return") else "continue",
                 "return_to": str(topic.get("return_to") or ""), "label": str(topic.get("label") or "")[:30]}
        constraints = [c for c in (obj.get("constraints") or []) if isinstance(c, dict)][:5]
        entities = [e for e in (obj.get("entities") or []) if isinstance(e, dict)][:6]
        return {"en_query": en, "zh_query": zh, "intent": intent, "question_type": qtype,
                "standalone": obj.get("standalone") is True, "topic": topic,
                "constraints_delta": constraints, "entities": entities,
                "unclassified_directive": obj.get("unclassified_directive") is True, "ok": True}

    # 规则预过滤：匹配则直接返回 direct，省去一次模型调用
    # 助手自身信息类：以“你/今天/现在”为主语的完整问句，子串匹配即可
    _DIRECT_SELF_PATTERNS = (
        '你是谁', '你叫什么', '你是什么模型', '你是哪个模型', '你的名字',
        '你能做什么', '你有什么功能', '你会什么',
        '今天是几号', '今天几号', '现在是几号', '今天是多少号', '今天是哪天',
    )
    # 问候/感谢/道别类：必须整句（去标点后）就是问候语，避免子串误判
    # （如 'hi' 命中 "Shaun White"/"Hindu"，'感谢' 命中“感谢信”，'你好' 命中“你好像”）
    _DIRECT_GREETING_RE = re.compile(
        r'(你好|您好|嗨|哈喽|再见|拜拜|谢谢|多谢|感谢|hello|hi|hey|bye|goodbye|thanks|thankyou)'
        r'(你|您|啦|呀|啊|哈|了|哦)*'
    )
    _PUNCT_RE = re.compile(r'[\s\W_]+', re.UNICODE)

    # 题型规则预判：高精度关键词，模型未给出合法题型时兜底
    _REASONING_PATTERNS = (
        '为什么', '原因是', '原因有', '什么原因', '如何解释', '怎么解释',
        '影响了', '影响是', '有什么影响', '导致了', '导致', '反映了',
        '说明了', '有什么意义', '意义是', '为何',
    )
    _MULTI_HOP_PATTERNS = (
        '与.*关系', '有什么关系', '有何关系', '相比', '比较', '区别是',
        '共同点', '不同点', '哪个更', '既.*又', '还.*过哪些', '还.*了哪些',
    )

    def _rule_based_question_type(self, user_input: str) -> Optional[str]:
        for pattern in self._REASONING_PATTERNS:
            if pattern in user_input:
                return 'reasoning'
        for pattern in self._MULTI_HOP_PATTERNS:
            if re.search(pattern, user_input):
                return 'multi_hop'
        return None

    def _rule_based_intent(self, user_input: str) -> Optional[str]:
        """规则预过滤：高精度匹配明显的 direct 场景（仅输入 ≤20 字时激活）"""
        q = user_input.strip()
        if len(q) > 20:
            return None
        if any(p in q for p in self._DIRECT_SELF_PATTERNS):
            return 'direct'
        normalized = self._PUNCT_RE.sub('', q.lower())
        if self._DIRECT_GREETING_RE.fullmatch(normalized):
            return 'direct'
        return None

    def _load_view(self, session_id: str) -> dict:
        if self.memory is None:
            return {"state": new_state(), "turns": []}
        try:
            state, turns, _ = self.memory.load(session_id)
            return {"state": state or new_state(), "turns": turns}
        except Exception as e:
            logger.warning(f"改写器读取短期记忆失败，按无历史处理: {e}")
            return {"state": new_state(), "turns": []}

    @observe     # 入口函数添加Langfuse装饰器
    def rewrite(self, user_input: str, session_id: str, view: Optional[dict] = None) -> Dict:
        """
        执行查询改写。

        Args:
            user_input: 用户当前输入。
            session_id: 短期记忆作用域 {user_id}:{session_id}（stm_scope）。
            view:       主流程已读取的会话 {"state", "turns"}；未传入时自行读取（MCP / 评估路径）。

        Returns:
            dict：en_query / zh_query / intent / question_type / standalone / topic / constraints_delta /
                  entities / unclassified_directive / ok / context_stats
        """
        logger.info(f"📥 用户原始输入: {user_input}")
        rule_question_type = self._rule_based_question_type(user_input)

        if self._rule_based_intent(user_input) == 'direct':
            logger.info(f"会话 {session_id} 规则命中 direct，跳过改写")
            return {"en_query": user_input, "zh_query": user_input, "intent": "direct", "question_type": "simple",
                    "standalone": True, "topic": {"shift": "continue", "return_to": "", "label": ""},
                    "constraints_delta": [], "entities": [], "unclassified_directive": False, "ok": True,
                    "rule_direct": True, "context_stats": {}}

        view = view or self._load_view(session_id)
        memory_text, stats = build_rewriter_context(view["state"], view["turns"], self.stm_config)
        prompt = self._build_prompt(memory_text, user_input)

        if self.circuit_breaker is not None:
            try:
                self.circuit_breaker.before_call()
            except CircuitBreakerOpen:
                logger.warning("4B 改写熔断中，直接使用原始输入")
                return {**_fallback(user_input, rule_question_type, "circuit_open"), "context_stats": stats}

        response = self._call_vllm(prompt)
        obj = _extract_json(response) if response else None
        if obj is None:
            if self.circuit_breaker is not None:
                self.circuit_breaker.on_failure(RuntimeError("rewrite failed"))
            reason = "invalid_json" if response else "call_failed"
            logger.warning(f"会话 {session_id} 改写失败（{reason}），使用原始输入")
            return {**_fallback(user_input, rule_question_type, reason), "context_stats": stats}
        if self.circuit_breaker is not None:
            self.circuit_breaker.on_success()

        out = self._parse(obj, user_input, rule_question_type)
        out["context_stats"] = stats
        logger.info(f"会话 {session_id} 改写成功: EN='{out['en_query']}', ZH='{out['zh_query']}', "
                    f"INTENT='{out['intent']}', QUESTION_TYPE='{out['question_type']}', "
                    f"standalone={out['standalone']}, topic={out['topic']['shift']}, "
                    f"constraints={len(out['constraints_delta'])}, entities={len(out['entities'])}")
        return out
