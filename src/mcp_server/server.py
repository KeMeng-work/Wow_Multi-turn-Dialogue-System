"""
WOW 知识服务 MCP Server (mcp_server)

把 WOW 系统的只读能力按 MCP（Model Context Protocol）标准对外提供，Claude Code / Claude Desktop / Cursor
等 MCP 客户端连接后可在运行时发现并调用工具，无需针对每个客户端单独适配。

工具：
    - search_knowledge_base：知识库检索（改写 → 双路召回 → MaxP 重排 → 证据分级），返回段落与证据等级
    - verify_relations：知识图谱实体关系查询（Neo4j）：单实体邻域 / 多实体多跳路径，事实带知识库原文证据
    - ask_agent：完整 Agent 问答，返回回答、证据等级与参考来源（其他 Agent 可把 WOW 当作一个工具调用）
    不对外提供：长期记忆读写（用户隐私）、联网搜索与翻译（客户端自身具备）。

架构：本服务是薄适配层，只做协议转换、参数校验与错误转换；通过 HTTP 调用主应用接口
（/tools/retrieve、/tools/verify、/chat/stream），不导入主应用代码。限流、准入、链路追踪均复用主应用。

鉴权分两段：
    - 客户端 → 本服务：HTTP 模式必须携带 Authorization: Bearer <MCP_ACCESS_TOKEN>
    - 本服务 → 主应用：服务账号（WOW_API_USERNAME / WOW_API_PASSWORD）自动登录取 JWT，过期（401）自动重新登录

运行：
    HTTP（部署在服务器，compose 中的 mcp 服务）：python -m src.mcp_server.server --transport http
    stdio（本地进程，客户端直接拉起）：         python -m src.mcp_server.server --transport stdio

环境变量：
    WOW_API_URL          主应用地址，默认 http://app:8000（compose 内网）
    WOW_API_USERNAME     服务账号用户名（scripts/create_user.py 创建）
    WOW_API_PASSWORD     服务账号密码
    MCP_ACCESS_TOKEN     客户端访问令牌（HTTP 模式必填）
    MCP_HOST / MCP_PORT  HTTP 监听地址，默认 0.0.0.0:8100
    MCP_API_TIMEOUT      调用主应用的超时秒数，默认 90（主应用请求截止 60s）

Author: Ke Meng
Created: 2026-10-15
"""

import argparse
import asyncio
import hmac
import json
import logging
import os
import sys
import uuid
from typing import Annotated, Any, Dict, List, Literal, Optional

import httpx
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

# stdio 模式下 stdout 是协议通道，日志必须写 stderr
logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("wow_mcp")

API_URL = os.getenv("WOW_API_URL", "http://app:8000").rstrip("/")
API_USERNAME = os.getenv("WOW_API_USERNAME", "")
API_PASSWORD = os.getenv("WOW_API_PASSWORD", "")
ACCESS_TOKEN = os.getenv("MCP_ACCESS_TOKEN", "")
API_TIMEOUT = float(os.getenv("MCP_API_TIMEOUT", 90))
MAX_CONTEXTS = 5            # ask_agent 返回的参考段落条数上限
MAX_CONTEXT_CHARS = 800     # 每条参考段落截断长度，避免撑爆调用方上下文

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

EVIDENCE_GUIDANCE = {
    "high": "检索结果与问题高度相关，可作为回答依据。",
    "low": "检索结果相关性较低，引用时请提示用户核实。",
    "none": "知识库中没有相关资料，请如实告知，不要编造。",
}


# ==================== 主应用 HTTP 客户端 ====================
class WowApiClient:
    """调用主应用接口：服务账号自动登录，401 时重新登录并重试一次；不使用系统代理，直连服务器"""

    def __init__(self, base_url: str, username: str, password: str, timeout: float):
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout, trust_env=False)
        self._username = username
        self._password = password
        self._token: Optional[str] = None
        self._login_lock = asyncio.Lock()

    async def _login(self, stale: Optional[str] = None) -> str:
        async with self._login_lock:
            if self._token and self._token != stale:   # 其他请求已刷新过
                return self._token
            if not self._username or not self._password:
                raise ToolError("MCP 服务未配置服务账号（WOW_API_USERNAME / WOW_API_PASSWORD）")
            resp = await self._client.post("/auth/login", json={"username": self._username, "password": self._password})
            if resp.status_code != 200:
                raise ToolError(f"服务账号登录失败（HTTP {resp.status_code}）")
            self._token = resp.json()["token"]
            logger.info("服务账号登录成功")
            return self._token

    async def _headers(self, stale: Optional[str] = None) -> Dict[str, str]:
        token = self._token if self._token and not stale else await self._login(stale)
        return {"Authorization": f"Bearer {token}"}

    async def post_json(self, path: str, payload: dict) -> dict:
        try:
            headers = await self._headers()
            resp = await self._client.post(path, json=payload, headers=headers)
            if resp.status_code == 401:
                resp = await self._client.post(path, json=payload, headers=await self._headers(stale=self._token))
        except httpx.TimeoutException:
            raise ToolError("WOW 服务响应超时，请稍后再试")
        except httpx.HTTPError as e:
            raise ToolError(f"无法连接 WOW 服务：{type(e).__name__}")
        _raise_for_status(resp.status_code, resp.text)
        return resp.json()

    async def stream_events(self, path: str, payload: dict) -> List[dict]:
        """调用 SSE 接口并收集全部事件（data: {JSON} 行）"""
        for attempt in range(2):
            stale = self._token if attempt else None
            try:
                headers = await self._headers(stale=stale)
                async with self._client.stream("POST", path, json=payload, headers=headers) as resp:
                    if resp.status_code == 401 and attempt == 0:
                        continue
                    if resp.status_code != 200:
                        _raise_for_status(resp.status_code, (await resp.aread()).decode("utf-8", "replace"))
                    events = []
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                events.append(json.loads(line[6:]))
                            except json.JSONDecodeError:
                                logger.warning(f"无法解析的 SSE 行: {line[:200]}")
                    return events
            except httpx.TimeoutException:
                raise ToolError("WOW 服务响应超时，请稍后再试")
            except httpx.HTTPError as e:
                raise ToolError(f"无法连接 WOW 服务：{type(e).__name__}")
        raise ToolError("服务账号鉴权失败")


def _raise_for_status(status: int, body: str) -> None:
    """主应用错误码转换为调用方（模型）能理解的提示"""
    if status == 200:
        return
    messages = {
        401: "服务账号鉴权失败",
        422: "参数不合法",
        429: "请求过于频繁，请稍后再试",
        503: "WOW 服务繁忙，请稍后再试",
        504: "WOW 服务处理超时，请稍后再试",
    }
    detail = messages.get(status, f"WOW 服务返回错误（HTTP {status}）")
    logger.warning(f"主应用返回 {status}: {body[:300]}")
    raise ToolError(detail)


api = WowApiClient(API_URL, API_USERNAME, API_PASSWORD, API_TIMEOUT)


# ==================== 输出 Schema ====================
class Passage(BaseModel):
    content: str = Field(description="段落原文（英文知识库）")
    low_relevance: bool = Field(description="该段落是否为低置信结果")


class SearchResult(BaseModel):
    query: str = Field(description="原始问题")
    en_query: str = Field(description="实际用于检索的英文查询（服务端改写）")
    question_type: str = Field(description="题型：simple / reasoning / multi_hop")
    evidence_level: Literal["high", "low", "none"] = Field(description="证据等级")
    guidance: str = Field(description="如何使用本次检索结果")
    passages: List[Passage] = Field(description="检索到的段落，证据不足时为空")


class LinkedEntity(BaseModel):
    mention: str = Field(description="调用方给出的名称")
    entity_id: str = Field(description="图谱实体编号")
    name: str = Field(description="规范英文名")
    name_zh: str = Field(description="中文名")
    entity_class: str = Field(alias="class", description="本体类，如 Company / Person / Film")
    match: str = Field(description="对齐方式：name / alias / weak / zh / fulltext")


class Fact(BaseModel):
    id: int
    subject: str
    relation: str = Field(description="本体关系，如 foundedBy / developedBy")
    relation_zh: str = Field(description="关系中文标签")
    object: str
    evidence: str = Field(description="知识库原文证据")
    source_chunks: List[str] = Field(description="证据所在 chunk_id（与向量库一致）")
    support: int = Field(description="独立来源数")


class GraphPath(BaseModel):
    between: List[str] = Field(description="路径两端的实体")
    hops: int
    fact_ids: List[int] = Field(description="按顺序组成路径的事实编号")


class VerifyResult(BaseModel):
    linked: List[LinkedEntity] = Field(description="已对齐到图谱的实体")
    not_in_graph: List[str] = Field(description="图谱未收录的实体（不代表没有关系；可检查是否使用了英文原名）")
    facts: List[Fact] = Field(description="相关事实，每条附知识库原文")
    paths: List[GraphPath] = Field(description="实体之间的多跳路径（多个实体时）")
    note: str = Field(description="使用说明")
    stats: Dict[str, Any] = Field(description="统计信息")


class AgentAnswer(BaseModel):
    answer: str = Field(description="WOW Agent 的最终回答")
    session_id: str = Field(description="会话 ID，传回可继续多轮对话")
    intent: Optional[str] = Field(description="意图：direct / local / web_search")
    evidence_level: Optional[str] = Field(description="证据等级：high / low / mixed / none；闲聊类问题为空")
    refused: bool = Field(description="是否因证据不足而拒答")
    web_sources: List[str] = Field(description="联网搜索的来源链接")
    contexts: List[str] = Field(description="回答引用的知识库段落（截断）")


# ==================== MCP Server ====================
mcp = FastMCP(
    name="wow-knowledge",
    instructions=(
        "WOW 知识服务：基于 Wizard of Wikipedia 英文百科知识库（历史、人文、百科主题）。"
        "回答事实类问题时先调用 search_knowledge_base，并按 evidence_level 决定是否引用；"
        "涉及实体之间的关系或需要多跳串联时，用检索段落中的英文原名调用 verify_relations；"
        "需要完整的多轮问答时调用 ask_agent。"
    ),
    host=os.getenv("MCP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_PORT", 8100)),
    stateless_http=True,   # 无会话状态，便于水平扩展与重启
)


@mcp.tool(title="知识库检索（带证据分级）", annotations=READ_ONLY)
async def search_knowledge_base(
    query: Annotated[str, Field(description="要检索的问题，中文或英文均可（服务端会改写为英文检索词）",
                                min_length=1, max_length=1000)],
    question_type: Annotated[Optional[Literal["simple", "reasoning", "multi_hop"]],
                             Field(description="题型，可选；不填由服务端判定")] = None,
) -> SearchResult:
    """
    检索 WOW 英文百科知识库：向量 + BM25 双路召回、段落级 RRF 融合、MaxP 重排，并给出证据等级。
    evidence_level 为 high 可作为依据；low 时需提示用户核实；none 表示知识库无相关资料，不要编造。
    """
    data = await api.post_json("/tools/retrieve", {"query": query, "question_type": question_type})
    level = data.get("evidence_level") or "none"
    return SearchResult(
        query=data.get("query", query),
        en_query=data.get("en_query", query),
        question_type=data.get("question_type", "simple"),
        evidence_level=level,
        guidance=EVIDENCE_GUIDANCE.get(level, EVIDENCE_GUIDANCE["none"]),
        passages=[Passage(**p) for p in data.get("passages", [])],
    )


@mcp.tool(title="知识图谱关系查询", annotations=READ_ONLY)
async def verify_relations(
    entities: Annotated[List[str], Field(description="1–5 个实体，使用检索段落中出现的英文原名（如 'Steve Jobs'），也可用中文名",
                                         min_length=1, max_length=5)],
    max_hops: Annotated[int, Field(description="多个实体之间路径的最大跳数", ge=1, le=3)] = 3,
) -> VerifyResult:
    """
    在知识图谱（Neo4j，OWL 本体约束）中查询实体关系：1 个实体返回直接相连的事实，
    多个实体返回它们之间的多跳路径。每条事实附知识库原文证据与来源 chunk_id，可作为回答依据。
    not_in_graph 只表示图谱未收录该实体（图谱覆盖部分知识库），不代表没有关系。
    """
    data = await api.post_json("/tools/verify", {"entities": entities, "max_hops": max_hops})
    return VerifyResult(
        linked=[LinkedEntity(**l) for l in data.get("linked", [])],
        not_in_graph=data.get("not_in_graph", []),
        facts=[Fact(**f) for f in data.get("facts", [])],
        paths=[GraphPath(**p) for p in data.get("paths", [])],
        note=data.get("note", ""),
        stats=data.get("stats", {}),
    )


@mcp.tool(title="WOW Agent 问答", annotations=READ_ONLY)
async def ask_agent(
    question: Annotated[str, Field(description="要提问的问题，中文或英文均可", min_length=1, max_length=2000)],
    session_id: Annotated[Optional[str], Field(description="会话 ID，传入上次返回的值可继续多轮对话；不填则新建会话")] = None,
) -> AgentAnswer:
    """
    调用完整的 WOW ReAct Agent（查询改写 → 检索 / 知识图谱 / 联网搜索 → 推理作答），
    返回回答、证据等级与参考来源。耗时较长（数秒到数十秒），简单检索请优先使用 search_knowledge_base。
    """
    sid = session_id if (session_id and session_id.startswith("mcp-")) else f"mcp-{session_id or uuid.uuid4().hex[:12]}"
    events = await api.stream_events("/chat/stream", {"user_input": question, "session_id": sid})

    answer_parts: List[str] = []
    out: Dict[str, Any] = {"intent": None, "evidence_level": None, "refused": False, "web_sources": [], "contexts": []}
    for ev in events:
        etype = ev.get("type")
        if etype == "answer":
            answer_parts.append(ev.get("content", ""))
        elif etype == "rewrite":
            out["intent"] = ev.get("intent")
        elif etype == "retrieval_meta":
            out["evidence_level"] = ev.get("evidence_level")
            out["refused"] = bool(ev.get("refused"))
        elif etype == "web_sources":
            out["web_sources"] = list(ev.get("urls") or [])
        elif etype == "contexts":
            out["contexts"] = [str(c)[:MAX_CONTEXT_CHARS] for c in (ev.get("data") or [])[:MAX_CONTEXTS]]
        elif etype == "error":
            _raise_for_status(int(ev.get("code") or 500), str(ev.get("detail", "")))

    answer = "".join(answer_parts).strip()
    if not answer:
        raise ToolError("WOW Agent 未返回回答")
    return AgentAnswer(answer=answer, session_id=sid, **out)


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


# ==================== HTTP 鉴权 ====================
class BearerAuthMiddleware:
    """纯 ASGI 中间件：HTTP 请求必须携带正确的 Bearer 令牌（/health 除外）；不影响 SSE 流式响应"""

    def __init__(self, app, token: str):
        self.app = app
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path") != "/health":
            auth = dict(scope.get("headers") or []).get(b"authorization", b"")
            if not hmac.compare_digest(auth, self.expected):
                resp = JSONResponse({"detail": "unauthorized"}, status_code=401, headers={"WWW-Authenticate": "Bearer"})
                await resp(scope, receive, send)
                return
        await self.app(scope, receive, send)


def main() -> None:
    parser = argparse.ArgumentParser(description="WOW 知识服务 MCP Server")
    parser.add_argument("--transport", choices=["http", "stdio"], default=os.getenv("MCP_TRANSPORT", "http"))
    args = parser.parse_args()

    if args.transport == "stdio":
        logger.info(f"MCP（stdio）启动，主应用 {API_URL}")
        mcp.run(transport="stdio")
        return

    if len(ACCESS_TOKEN) < 16:
        raise SystemExit("HTTP 模式必须设置 MCP_ACCESS_TOKEN（至少 16 位），拒绝无鉴权启动")
    import uvicorn
    app = BearerAuthMiddleware(mcp.streamable_http_app(), ACCESS_TOKEN)
    logger.info(f"MCP（Streamable HTTP）启动 {mcp.settings.host}:{mcp.settings.port}{mcp.settings.streamable_http_path}，"
                f"主应用 {API_URL}")
    uvicorn.run(app, host=mcp.settings.host, port=mcp.settings.port, log_level="info")


if __name__ == "__main__":
    main()
