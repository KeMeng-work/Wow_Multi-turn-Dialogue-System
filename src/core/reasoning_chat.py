"""
支持原生思考过程的 ChatOpenAI 扩展（主 LLM：百炼云端）

百炼（OpenAI 兼容模式）开启 enable_thinking 后，流式 delta 中思考内容在 `reasoning_content` 字段，
langchain-openai（1.1.x）的 _convert_delta_to_message_chunk 只解析 content / tool_calls 等标准字段，
reasoning_content 被丢弃，导致 /chat/stream 拿不到 thinking 事件、前端"思考过程"不显示。
这里在每个流式 chunk 转换后，把 delta.reasoning_content 补进 AIMessageChunk.additional_kwargs，
main.py 现有逻辑（chunk.additional_kwargs.get("reasoning_content")）即可读到。

2026-10-17 敏感信息出口：主 LLM 是数据离开服务器的唯一出口，所有消息（SP、长期记忆、短期记忆、用户输入、工具结果）
在这里统一换成占位符再发出；脱敏出错时抛 PIIMaskError，阻断本次调用，不发送原文。

Author: Ke Meng
Created: 2026-10-09
"""

from langchain_core.messages import AIMessageChunk
from langchain_openai import ChatOpenAI

from src.core.security.events import record as record_security_event
from src.core.security.pii import PIIMaskError, PIIVault, get_vault


def _mask_content(content, vault: PIIVault):
    if isinstance(content, str):
        return vault.mask(content)
    if isinstance(content, list):   # 多模态分段内容：只处理文本段
        return [{**p, "text": vault.mask(p["text"])} if isinstance(p, dict) and isinstance(p.get("text"), str)
                else (vault.mask(p) if isinstance(p, str) else p) for p in content]
    return content


def mask_messages(messages):
    """发往云端前脱敏；无请求上下文（CLI / 离线脚本）时用内存映射表，同样不外发原文"""
    vault = get_vault() or PIIVault()
    try:
        return [m.model_copy(update={"content": _mask_content(m.content, vault)}) for m in messages]
    except Exception as e:
        record_security_event("pii_mask_failed", {"error": type(e).__name__})
        raise PIIMaskError("敏感信息脱敏失败，已阻断外部模型调用") from e


class ReasoningChatOpenAI(ChatOpenAI):
    """把流式 delta 中的 reasoning_content 透传到 additional_kwargs；发出前统一脱敏"""

    def _generate(self, messages, *args, **kwargs):
        return super()._generate(mask_messages(messages), *args, **kwargs)

    async def _agenerate(self, messages, *args, **kwargs):
        return await super()._agenerate(mask_messages(messages), *args, **kwargs)

    def _stream(self, messages, *args, **kwargs):
        yield from super()._stream(mask_messages(messages), *args, **kwargs)

    async def _astream(self, messages, *args, **kwargs):
        async for chunk in super()._astream(mask_messages(messages), *args, **kwargs):
            yield chunk

    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        generation_chunk = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)
        if generation_chunk is None or not isinstance(generation_chunk.message, AIMessageChunk):
            return generation_chunk
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []
        if choices:
            reasoning = (choices[0].get("delta") or {}).get("reasoning_content")
            if reasoning:
                generation_chunk.message.additional_kwargs["reasoning_content"] = reasoning
        return generation_chunk
