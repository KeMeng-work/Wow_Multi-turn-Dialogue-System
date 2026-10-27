"""
在线搜索工具模块 (Web Search Tool)

基于百度千帆 ai_search/web_summary 接口，封装为 LangChain StructuredTool，
供 ReAct Agent 在本地检索不足时调用。

设计约束：
  - 强制返回摘要模式（resource_type_filter type=web），绝不返回全文
  - 硬上限 top_k=3，避免上下文膨胀
  - 每条摘要 300 字硬截断（句子边界优先），进一步限制 token 消耗
  - API Key 从环境变量 QIANFAN_API_KEY 读取，绝不硬编码

2026-10-17 提示词注入防护（网页内容是间接注入的主要来源，见 src/core/security/injection.py）：
  - 第 1 层 清洗：删除隐藏字符、图片、伪造角色标签
  - 第 2 层 检测：命中注入规则的整条结果不提供给模型，记安全事件 injection_blocked
  - 第 3 层 隔离：结果包在随机标识的 <web_result> 标签内，前置不可信数据说明
  - 网址只保留 http / https（防 javascript: 等伪协议）
  - 返回 (给模型的文本, 结构化来源)：结构化来源作为 ToolMessage.artifact，前端逐条展示 ✅ / 🧹 / ⛔ 状态与原因

Author: Ke Meng
Created: 2026-09-12
"""

import time
import logging
from urllib.parse import urlparse

import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.core.security import injection
from src.core.security.events import record as record_security_event

logger = logging.getLogger(__name__)

QIANFAN_API_URL = "https://qianfan.baidubce.com/v2/ai_search/web_summary"


class WebSearchInput(BaseModel):
    query: str = Field(description="用户原始中文问题，直接传入，无需翻译或改写")


def _truncate_at_boundary(text: str, max_chars: int) -> str:
    """在句子边界截断文本，不超过 max_chars 字符。"""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    for sep in ('。', '！', '？', '.', '!', '?', '；', ';'):
        idx = truncated.rfind(sep)
        if idx > max_chars // 2:
            return truncated[:idx + 1]
    return truncated + "…"


def create_web_search_tool(
    api_key: str,
    top_k: int = 3,
    max_snippet_chars: int = 300,
    timeout: int = 15,
) -> StructuredTool:
    """
    创建百度千帆在线搜索工具实例。

    :param api_key:           千帆 API Key（从 QIANFAN_API_KEY 环境变量读取）
    :param top_k:             最多返回结果数（硬上限，默认 3）
    :param max_snippet_chars: 每条摘要字符数上限（默认 300）
    :param timeout:           HTTP 请求超时秒数（默认 15）
    :return: LangChain StructuredTool
    """

    def _failed(text: str) -> tuple[str, dict]:
        return text, {"status": "failed", "sources": []}

    def _run(query: str) -> tuple[str, dict]:
        if not api_key:
            logger.error("QIANFAN_API_KEY 未配置，在线搜索不可用")
            return _failed("在线搜索不可用（未配置 API Key）。")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "messages": [{"role": "user", "content": query}],
            "stream": False,
            "resource_type_filter": [{"type": "web", "top_k": top_k}],
        }

        _t = time.perf_counter()
        try:
            resp = requests.post(QIANFAN_API_URL, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            elapsed_ms = round((time.perf_counter() - _t) * 1000, 1)
        except requests.Timeout:
            logger.warning(f"千帆在线搜索超时（>{timeout}s），query='{query[:50]}'")
            return _failed("在线搜索超时，请稍后重试。")
        except requests.HTTPError as e:
            logger.error(f"千帆在线搜索 HTTP 错误: {e}")
            return _failed(f"在线搜索请求失败（HTTP {e.response.status_code}）。")
        except Exception as e:
            logger.error(f"千帆在线搜索异常: {e}")
            return _failed(f"在线搜索请求失败: {e}")

        references = data.get("references", data.get("search_results", []))
        if not references:
            logger.warning(f"千帆在线搜索无结果，query='{query[:50]}'")
            return "在线搜索未找到相关内容。", {"status": "empty", "sources": []}

        logger.info(f"千帆在线搜索完成: {len(references)} 条结果，耗时 {elapsed_ms}ms")

        boundary = injection.new_boundary()
        parts, sources = [], []
        for ref in references[:top_k]:
            url = (ref.get("url") or ref.get("link") or "").strip()
            if urlparse(url).scheme not in ("http", "https"):
                url = ""                                   # 只允许 http / https
            domain = urlparse(url).hostname or ""
            title, t_counts = injection.sanitize(ref.get("title") or ref.get("web_anchor") or "无标题")
            content, c_counts = injection.sanitize(ref.get("content") or ref.get("snippet") or "")
            source = {"url": url, "domain": domain, "title": title.strip()[:100] or "无标题"}

            hit = injection.detect(f"{title}\n{content}")
            if hit:
                rule, fragment = hit
                record_security_event("injection_blocked", {"source": "web_search", "rule": rule, "domain": domain})
                # rule / excerpt 只给 admin 看（api.py 按角色裁剪）；excerpt 以纯文本展示，不渲染
                sources.append({**source, "status": "blocked", "reason": f"疑似指令注入（{rule}），内容未提供给模型",
                                "rule": rule, "excerpt": fragment})
                continue

            snippet = _truncate_at_boundary(content, max_snippet_chars)
            counts = {k: t_counts.get(k, 0) + c_counts.get(k, 0) for k in {**t_counts, **c_counts}}
            sources.append({**source, "status": "cleaned" if counts else "ok", "snippet": snippet,
                            "reason": injection.describe_cleaning(counts) if counts else ""})
            parts.append(injection.wrap(boundary, len(parts) + 1, f"标题: {title}\n来源: {url}\n摘要: {snippet}"))

        if not parts:
            return "在线搜索结果均未通过安全检查，没有可用的网页资料。", {"status": "empty", "sources": sources}
        return injection.untrusted_header(boundary) + "\n\n" + "\n\n".join(parts), {"status": "ok", "sources": sources}

    return StructuredTool.from_function(
        func=_run,
        name="web_search",
        description=(
            "使用百度千帆在线搜索，获取最新互联网信息。"
            "在当前意图指令要求时调用：web_search 意图下，于 knowledge_retriever 之后调用。"
            "参数query使用用户原始中文问题，直接传入，无需翻译。"
            "返回至多3条摘要结果，每条不超过300字。"
        ),
        args_schema=WebSearchInput,
        response_format="content_and_artifact",   # artifact = {"status", "sources"}，不进入模型上下文
    )
