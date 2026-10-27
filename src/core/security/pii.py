"""
敏感信息可逆脱敏 (PII pseudonymization)

原则：信任边界。服务器内部（本地 4B / 1.7B、检索、数据库）使用原文；数据离开服务器前换成占位符，回到用户前按角色还原。
    - 出口：主 LLM（百炼）调用封装层统一脱敏（reasoning_chat.py），覆盖 SP / 长期记忆 / 短期记忆 / 用户输入 / 工具结果
    - 工具：本地工具执行前还原参数（agent.py tool_node），web_search 不还原（不把个人信息发给第三方搜索）
    - 存储：PG 对话历史、短期记忆、长期记忆只存占位符版本
    - 展示：api.py 按角色还原 —— admin 完整值（记还原审计）、user 部分遮盖；URL 内的占位符不还原（防外泄）
    - 观测：Langfuse mask 与日志过滤器只保留占位符

识别类型（2026-10-17 确认）：
    人名   —— 配置名单匹配（PII_CONFIG['protected_names']），不用 NER：知识库里大量演员、角色名，NER 会误伤
    手机号 —— 1[3-9] 开头 11 位，允许空格 / 连字符分隔，前后不能紧挨数字
    银行卡 —— 16–19 位数字（允许空格 / 连字符），必须通过 Luhn 校验，避免误伤订单号、时间戳
    邮箱   —— 标准邮箱格式
    密钥   —— sk- / Bearer / JWT / password= 等，替换为 [密钥]，对任何人都不还原

占位符：[手机号_1]，同一会话内同一值始终同一编号（多轮对话中模型能对应上）。
映射表：Redis Hash pii:{user_id}:{session_id}，TTL 与短期记忆一致（24h）；过期后历史只显示占位符。
    字段 "手机号_1" → 原值，"#原值" → "手机号_1"（反查，保证同值同号）。

Author: Ke Meng
Created: 2026-10-17
"""

import logging
import re
from typing import Any, Callable, Optional

from config.config_path import PII_CONFIG
from src.core.security.context import get_context

logger = logging.getLogger(__name__)

NAME, PHONE, CARD, EMAIL, SECRET = "人名", "手机号", "银行卡", "邮箱", "密钥"
REVERSIBLE_TYPES = (NAME, PHONE, CARD, EMAIL)


class PIIMaskError(RuntimeError):
    """脱敏失败：调用方必须阻断外发（出错即阻断，不发送原文）"""


# ── 识别规则 ────────────────────────────────────────────────────
_SECRET_RE = re.compile(
    r"\bsk-[A-Za-z0-9_-]{16,}"
    r"|\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"
    r"|\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
    r"|(?i:\b(?:password|passwd|pwd|api[_-]?key|secret|token)\s*[:=]\s*[^\s,;，；]{6,})"
)
_EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w-])")
_CARD_RE = re.compile(r"(?<![\d])\d(?:[ -]?\d){15,18}(?![\d])")
_PHONE_RE = re.compile(r"(?<![\d])1[3-9]\d(?:[ -]?\d{4}){2}(?![\d])")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _card_ok(raw: str) -> bool:
    digits = re.sub(r"\D", "", raw)
    return 16 <= len(digits) <= 19 and _luhn_ok(digits)


def _name_regex(names) -> Optional[re.Pattern]:
    """中文名直接匹配；英文名大小写不敏感、前后不能紧挨字母，空格可多个"""
    parts = []
    for n in sorted({n.strip() for n in names if n and n.strip()}, key=len, reverse=True):
        if re.search(r"[A-Za-z]", n):
            words = [re.escape(w) for w in n.split()]
            parts.append(r"(?<![A-Za-z])(?i:" + r"\s+".join(words) + r")(?![A-Za-z])")
        else:
            parts.append(re.escape(n))
    return re.compile("|".join(parts)) if parts else None


_NAME_RE = _name_regex(PII_CONFIG['protected_names'])

# 按优先级识别：先认出的片段占位，后面的规则不再匹配其中的子串（如邮箱里的名字、卡号里的 11 位数字）
_DETECTORS: list[tuple[str, re.Pattern, Optional[Callable[[str], bool]]]] = [
    (SECRET, _SECRET_RE, None),
    (EMAIL, _EMAIL_RE, None),
    (CARD, _CARD_RE, _card_ok),
    (PHONE, _PHONE_RE, None),
]
if _NAME_RE is not None:
    _DETECTORS.append((NAME, _NAME_RE, None))

# 占位符：兼容模型偶尔改写成英文或换分隔符（[Phone_1] / [手机号 1]）
_TYPE_ALIASES = {
    "人名": NAME, "姓名": NAME, "name": NAME, "person": NAME,
    "手机号": PHONE, "手机": PHONE, "电话": PHONE, "phone": PHONE, "mobile": PHONE,
    "银行卡": CARD, "银行卡号": CARD, "卡号": CARD, "card": CARD, "bankcard": CARD, "bank card": CARD,
    "邮箱": EMAIL, "email": EMAIL, "e-mail": EMAIL, "mail": EMAIL,
}
PLACEHOLDER_RE = re.compile(
    r"\[(" + "|".join(sorted((re.escape(k) for k in _TYPE_ALIASES), key=len, reverse=True)) + r")[_ ]?(\d{1,3})\]",
    re.IGNORECASE,
)
# URL / Markdown 链接目标：其中的占位符不还原，防止"把手机号拼进图片或链接地址"一类的外泄
_URL_CONTEXT_RE = re.compile(r"(?:https?:)?//[^\s<>\"'）)]*|\]\([^)\s]*")


def find_pii(text: str) -> list[tuple[int, int, str, str]]:
    """返回不重叠的 (start, end, 类型, 原值)，按位置排序"""
    spans: list[tuple[int, int, str, str]] = []
    taken: list[tuple[int, int]] = []
    for typ, pattern, check in _DETECTORS:
        for m in pattern.finditer(text):
            s, e = m.span()
            if any(s < te and e > ts for ts, te in taken):
                continue
            if PLACEHOLDER_RE.fullmatch(m.group()):
                continue
            if check and not check(m.group()):
                continue
            spans.append((s, e, typ, m.group()))
            taken.append((s, e))
    return sorted(spans)


def contains_pii(text: str) -> bool:
    """文本含原值或占位符（用于判断回答能否写入全局缓存）"""
    return bool(text) and (bool(PLACEHOLDER_RE.search(text)) or bool(find_pii(text)))


# 遮盖字符用全角星号：半角 * 是 Markdown 强调符号，"孟*的手机号是131****7618" 会被前端渲染成斜体 / 加粗并吞掉星号
MASK = "＊"


def partial(typ: str, value: str) -> str:
    """user 角色看到的部分遮盖形式"""
    if typ == PHONE:
        d = re.sub(r"\D", "", value)
        return f"{d[:3]}{MASK * 4}{d[-4:]}"
    if typ == CARD:
        return f"{MASK * 4} {MASK * 4} {MASK * 4} " + re.sub(r"\D", "", value)[-4:]
    if typ == EMAIL:
        local, _, domain = value.partition("@")
        return f"{local[:1]}{MASK * 3}@{domain}"
    if typ == NAME:
        if re.search(r"[A-Za-z]", value):
            return " ".join(w[:1] + MASK * 3 for w in value.split())
        return value[:1] + MASK * (len(value) - 1)
    return f"[{typ}]"


def _canonical(alias: str) -> str:
    return _TYPE_ALIASES.get(alias.lower(), _TYPE_ALIASES.get(alias, alias))


# ── 映射表 ──────────────────────────────────────────────────────
class PIIVault:
    """
    会话级映射表。redis_client 为 None 时只在内存中（离线评估、MCP 工具接口：只脱敏，不需要跨请求还原）。
    同一请求内 main.py 与 api.py 共用一个实例（经 security context 传递），保证编号一致。
    """

    def __init__(self, redis_client=None, user_id: str = "", session_id: str = ""):
        self.redis = redis_client
        self.key = f"pii:{user_id}:{session_id}"
        self.ttl = PII_CONFIG['vault_ttl_s']
        self._tokens: dict[str, str] = {}    # "手机号_1" → 原值
        self._values: dict[str, str] = {}    # 原值 → "手机号_1"
        if redis_client is not None:
            try:
                for field, val in (redis_client.hgetall(self.key) or {}).items():
                    if field.startswith("#"):
                        self._values[field[1:]] = val
                    else:
                        self._tokens[field] = val
            except Exception as e:
                # 读不到映射表时退化为内存映射：照常脱敏（不会外发原文），只是看不到之前轮次的还原
                logger.warning(f"PII 映射表读取失败，退化为内存映射 {self.key}: {e}")
                self.redis = None

    def _token_for(self, typ: str, value: str) -> str:
        key = re.sub(r"\D", "", value) if typ in (PHONE, CARD) else value   # 同一号码不同分隔写法视为同值
        if key in self._values:
            return self._values[key]
        n = 1 + sum(1 for t in self._tokens if t.rsplit("_", 1)[0] == typ)
        token = f"{typ}_{n}"
        self._tokens[token], self._values[key] = value, token
        if self.redis is not None:
            try:
                pipe = self.redis.pipeline()
                pipe.hset(self.key, mapping={token: value, f"#{key}": token})
                pipe.expire(self.key, self.ttl)
                pipe.execute()
            except Exception as e:
                logger.warning(f"PII 映射表写入失败（本请求内仍可还原）{self.key}: {e}")
        return token

    def mask(self, text: str) -> str:
        """原值 → 占位符；密钥 → [密钥]（不入映射表）"""
        if not text or not PII_CONFIG['enabled']:
            return text
        out, last = [], 0
        for s, e, typ, value in find_pii(text):
            out.append(text[last:s])
            out.append(f"[{SECRET}]" if typ == SECRET else f"[{self._token_for(typ, value)}]")
            last = e
        out.append(text[last:])
        return "".join(out)

    def lookup_mask(self, text: str) -> str:
        """只读脱敏（日志、Langfuse）：已登记的值用会话编号，未登记的用不带编号的类型标签，不写映射表"""
        if not text or not PII_CONFIG['enabled']:
            return text
        out, last = [], 0
        for s, e, typ, value in find_pii(text):
            key = re.sub(r"\D", "", value) if typ in (PHONE, CARD) else value
            token = self._values.get(key) if typ != SECRET else None
            out.append(text[last:s])
            out.append(f"[{token}]" if token else f"[{typ}]")
            last = e
        out.append(text[last:])
        return "".join(out)

    def unmask(self, text: str) -> str:
        """占位符 → 原值（仅用于本地工具参数，不经过角色判断，结果不得外发）"""
        if not text:
            return text
        return PLACEHOLDER_RE.sub(lambda m: self._tokens.get(f"{_canonical(m.group(1))}_{m.group(2)}", m.group()), text)

    def restore(self, text: str, role: str, prefix: str = "") -> tuple[str, set]:
        """
        按角色还原占位符，返回 (文本, 还原了的类型集合)。
        admin → 完整值；其他角色 → 部分遮盖；URL / 链接目标内的占位符保持不变。
        :param prefix: 流式输出中此前已发出的末尾文本，只用于判断占位符是否位于 URL 内
        """
        if not text:
            return text, set()
        offset = len(prefix)
        url_spans = [m.span() for m in _URL_CONTEXT_RE.finditer(prefix + text)]
        restored: set = set()

        def _sub(m: re.Match) -> str:
            pos = m.start() + offset
            if any(s <= pos < e for s, e in url_spans):
                return m.group()
            typ = _canonical(m.group(1))
            value = self._tokens.get(f"{typ}_{m.group(2)}")
            if value is None:
                return m.group()
            restored.add(typ)
            return value if role == "admin" else partial(typ, value)

        return PLACEHOLDER_RE.sub(_sub, text), restored

    def display(self, text: str, role: str) -> tuple[str, set]:
        """展示给用户的完整文本（含原文来源，如知识库段落）：先脱敏再按角色还原"""
        return self.restore(self.mask(text), role)


class StreamRestorer:
    """
    流式输出的占位符还原：占位符可能被拆在两个数据块里（"[手机" + "号_1]"），
    遇到未闭合的 "[" 先缓存，等到 "]" 或缓存超长再输出。
    """

    _MAX_HOLD = 16

    def __init__(self, vault: PIIVault, role: str):
        self.vault, self.role = vault, role
        self.buf = ""
        self.tail = ""          # 已输出文本的末尾（判断 URL 上下文）
        self.restored: set = set()

    def _emit(self, text: str) -> str:
        out, types = self.vault.restore(text, self.role, prefix=self.tail)
        self.restored |= types
        self.tail = (self.tail + text)[-300:]
        return out

    def feed(self, chunk: str) -> str:
        text = self.buf + (chunk or "")
        i = text.rfind("[")
        if i != -1 and "]" not in text[i:] and len(text) - i <= self._MAX_HOLD:
            self.buf, text = text[i:], text[:i]
        else:
            self.buf = ""
        return self._emit(text) if text else ""

    def flush(self) -> str:
        text, self.buf = self.buf, ""
        return self._emit(text) if text else ""


# ── 便捷函数（读取当前请求上下文中的映射表）─────────────────────────
def get_vault() -> Optional[PIIVault]:
    return get_context().get("vault")


def deep_apply(obj: Any, fn: Callable[[str], str], skip_keys: tuple = ()) -> Any:
    """对 dict / list 中所有字符串应用 fn（skip_keys 中的字段原样保留）"""
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: (v if k in skip_keys else deep_apply(v, fn, skip_keys)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(deep_apply(v, fn, skip_keys) for v in obj)
    return obj


def mask_for_storage(obj: Any) -> Any:
    """持久化前脱敏（PG / 短期记忆 / 长期记忆 / 第三方评估）：使用当前会话映射表，无上下文时用内存映射表"""
    if not PII_CONFIG['enabled']:
        return obj
    vault = get_vault() or PIIVault()
    return deep_apply(obj, vault.mask)


def mask_for_observability(obj: Any) -> Any:
    """日志与 Langfuse：只读脱敏，不写映射表；任何异常都退化为通用标签，不能让观测链路抛错"""
    if not PII_CONFIG['enabled']:
        return obj
    vault = get_vault() or _STATELESS
    try:
        return deep_apply(obj, vault.lookup_mask)
    except Exception:
        return obj if not isinstance(obj, str) else _STATELESS.lookup_mask(obj)


_STATELESS = PIIVault()


class PIILogFilter(logging.Filter):
    """日志脱敏：把格式化后的消息整体脱敏后写回（f-string 拼接的用户输入、检索预览等）"""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            masked = mask_for_observability(msg)
            if masked != msg:
                record.msg, record.args = masked, None
        except Exception:
            pass
        return True


def langfuse_mask(*, data: Any, **kwargs) -> Any:
    """Langfuse 客户端 mask 钩子：所有观测的 input / output / metadata 上报前脱敏"""
    return mask_for_observability(data)
