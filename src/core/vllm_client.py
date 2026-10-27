"""
vLLM OpenAI 兼容接口客户端 (vLLM Client)

封装对 vLLM /v1/chat/completions 端点的 HTTP 调用，
统一替换原 Ollama /api/generate 和 /api/chat 调用。

提供两个函数：
  - vllm_generate_async: 供异步模块使用（long_term_memory, reflection_task）
  - vllm_generate_sync:  供同步模块使用（query_rewriter）

Ollama → vLLM 格式差异：
  Ollama:  POST /api/generate  {"prompt": "...", "options": {"temperature": 0.1}}
           resp["response"]
  vLLM:    POST /v1/chat/completions  {"messages": [...], "temperature": 0.1}
           resp["choices"][0]["message"]["content"]

Prefix Caching 设计：
  vLLM 服务端启用 --enable-prefix-caching 后，相同前缀的 KV Cache 只计算一次。
  客户端侧需将固定 System Prompt 作为独立的 system role 消息传入（system_prompt 参数），
  与 user 消息分离，才能被 vLLM 识别为可缓存前缀。
  若将 System Prompt 拼进 user 消息，前缀会因 user 内容变化而永远无法命中缓存。
"""

import logging
from typing import List, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

_CHAT_PATH = "/chat/completions"

# 模块级同步 httpx 客户端（连接池复用，避免每次调用重建 TCP 连接）
# 查询改写经 asyncio.to_thread 在线程池中并发调用本客户端，连接上限需与线程池（64）匹配
_sync_client = httpx.Client(
    timeout=None,        # 各函数内按需传入 timeout
    limits=httpx.Limits(max_keepalive_connections=32, max_connections=64),
)


def _build_messages(
    prompt: str,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    构建 messages 列表。

    system_prompt 单独作为 system role 消息放在首位，是 Prefix Caching 生效的前提：
      vLLM 以 messages 前缀的 hash 作为 cache key，system 消息固定不变时
      整个前缀的 KV Cache 在多次请求间复用，显著降低首 token 延迟。
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


async def vllm_generate_async(
    http_client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    api_key: str = 'EMPTY',
    timeout: float = 30.0,
    temperature: float = 0.1,
    max_tokens: int = 512,
    json_mode: bool = False,
    thinking: bool = False,
    json_schema: Optional[dict] = None,
) -> Optional[str]:
    """
    异步调用 vLLM，返回模型文本响应。

    :param http_client:   外部传入的 httpx.AsyncClient（复用连接池）
    :param base_url:      vLLM 服务地址，如 http://localhost:8001/v1
    :param model:         模型名称，需与 vLLM 启动时 --model 参数一致
    :param prompt:        用户侧 prompt 文本（user role）
    :param system_prompt: 固定 System Prompt（system role），传入后触发 Prefix Caching
    :param api_key:       vLLM 默认不鉴权，传 'EMPTY' 即可
    :param timeout:       请求超时（秒）
    :param temperature:   生成温度（JSON 提取任务建议 0.1，保证格式稳定）
    :param max_tokens:    最大生成 token 数
    :param json_mode:     True 时启用 response_format=json_object，适合 JSON 提取任务
    :param thinking:      False 时在 prompt 前注入 /no_think，跳过 Qwen3 思考阶段，
                          节省 50-200 token 无效输出；结构化 JSON 任务建议关闭，
                          反思生成等复杂推理任务建议开启（默认 False）
    :param json_schema:   JSON Schema 约束解码；服务端不支持时自动降级为 json_object
    :return:              模型返回的文本，失败返回 None
    """
    global _schema_unsupported
    if not thinking:
        prompt = "/no_think\n" + prompt
    url = base_url.rstrip('/') + _CHAT_PATH
    payload = {
        "model": model,
        "messages": _build_messages(prompt, system_prompt),
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    schema = None if _schema_unsupported else json_schema
    fmt = _response_format(json_mode or json_schema is not None, schema)
    if fmt:
        payload["response_format"] = fmt

    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        resp = await http_client.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code == 400 and schema is not None:
            # 400 也可能与约束解码无关（如超出 max_model_len）：降级重试成功才认定不支持 json_schema
            first_err = resp.text[:200]
            payload["response_format"] = {"type": "json_object"}
            resp = await http_client.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code != 400:
                logger.warning(f"vLLM 不支持 json_schema 约束解码，本进程降级为 json_object: {first_err}")
                _schema_unsupported = True
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except httpx.TimeoutException:
        logger.error(f"vLLM async 超时 (timeout={timeout}s) model={model}")
        return None
    except httpx.HTTPStatusError as e:
        # 记录响应体：vLLM 4xx 的原因（如超出 max_model_len）只在响应体里
        logger.error(f"vLLM async 请求失败 model={model}: {e} | body={e.response.text[:300]}")
        return None
    except Exception as e:
        logger.error(f"vLLM async 请求失败 model={model}: {e}")
        return None


def _response_format(json_mode: bool, json_schema: Optional[dict]) -> Optional[dict]:
    if json_schema is not None:
        return {"type": "json_schema", "json_schema": {"name": json_schema.get("title", "output"), "schema": json_schema}}
    if json_mode:
        return {"type": "json_object"}
    return None


# 服务端不支持 json_schema 约束解码时（400 且 json_object 重试成功），本进程内降级为 json_object，不再重复尝试
_schema_unsupported = False


def vllm_generate_sync(
    base_url: str,
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    api_key: str = 'EMPTY',
    timeout: float = 30.0,
    temperature: float = 0.7,
    max_tokens: int = 512,
    json_mode: bool = False,
    thinking: bool = False,
    json_schema: Optional[dict] = None,
) -> Optional[str]:
    """
    同步调用 vLLM，返回模型文本响应。

    使用模块级 _sync_client 复用 TCP 连接，避免每次调用重建连接的开销。

    :param base_url:      vLLM 服务地址，如 http://localhost:8001/v1
    :param model:         模型名称
    :param prompt:        用户侧 prompt 文本（user role）
    :param system_prompt: 固定 System Prompt（system role），传入后触发 Prefix Caching
    :param api_key:       vLLM 默认不鉴权，传 'EMPTY' 即可
    :param timeout:       请求超时（秒）
    :param temperature:   生成温度
    :param max_tokens:    最大生成 token 数
    :param json_mode:     True 时启用 response_format=json_object
    :param thinking:      False 时注入 /no_think 跳过 Qwen3 思考阶段（默认 False）
    :param json_schema:   JSON Schema（约束解码，保证输出合法且字段齐全）；服务端不支持时自动降级为 json_object
    :return:              模型返回的文本，失败返回 None
    """
    global _schema_unsupported
    if not thinking:
        prompt = "/no_think\n" + prompt
    url = base_url.rstrip('/') + _CHAT_PATH
    payload = {
        "model": model,
        "messages": _build_messages(prompt, system_prompt),
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    schema = None if _schema_unsupported else json_schema
    fmt = _response_format(json_mode or json_schema is not None, schema)
    if fmt:
        payload["response_format"] = fmt

    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        resp = _sync_client.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code == 400 and schema is not None:
            # 400 也可能与约束解码无关（如超出 max_model_len）：降级重试成功才认定不支持 json_schema
            first_err = resp.text[:200]
            payload["response_format"] = {"type": "json_object"}
            resp = _sync_client.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code != 400:
                logger.warning(f"vLLM 不支持 json_schema 约束解码，本进程降级为 json_object: {first_err}")
                _schema_unsupported = True
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except httpx.TimeoutException:
        logger.error(f"vLLM sync 超时 (timeout={timeout}s) model={model}")
        return None
    except httpx.HTTPStatusError as e:
        logger.error(f"vLLM sync 请求失败 model={model}: {e} | body={e.response.text[:300]}")
        return None
    except Exception as e:
        logger.error(f"vLLM sync 请求失败 model={model}: {e}")
        return None
