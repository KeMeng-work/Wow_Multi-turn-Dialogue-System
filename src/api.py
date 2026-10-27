"""
API 服务模块 (API)

该模块基于 FastAPI 实现，作为整个系统的 HTTP 入口，对外提供 RESTful 接口。
它将用户请求转发给核心处理函数 handle_request_async，并返回最终答案。

主要功能：
    - `/chat` (POST): 接收用户输入、会话 ID 和用户 ID，返回助手回答。
    - `/health` (GET): 健康检查接口，用于服务存活探针。

变更记录：
    - 1.0.6 (2026-10-17，安全管控·身份与权限):
        1. 角色 admin / user / ops：对话、会话、记忆、反馈、/tools/* 限 user+admin；/chat_eval、/metrics、/openapi.json 限 ops+admin
        2. 关闭 /docs、/redoc（生产不提供交互式文档）；/openapi.json 需 ops/admin 令牌
        3. 启动时建 security_events 表（权限拒绝等安全事件审计）
        4. 敏感信息展示：回答、流式事件、历史会话按角色还原占位符（admin 完整值并记还原审计，user 部分遮盖）；
           /tools/* 返回给 MCP 客户端（外部）只给占位符；反馈入库前脱敏

    - 1.0.5 (2026-10-09，前端):
        1. 登录认证：/auth/login 签发 JWT；对话接口通过 current_user 依赖取 user_id（AUTH_ENABLED=false 时取请求体）
        2. 会话接口：/sessions（会话列表）、/sessions/{session_id}/messages（会话消息），读 PG conversation_history
        3. /memories：当前用户的长期记忆（只读）；/feedback：回答 👍 / 👎 写入 message_feedback，供数据飞轮挖掘 badcase
        4. 前端：/ui 提供单文件页面，/ 重定向到 /ui

    - 1.0.4 (2026-10-07，压测前优化 P2):
        1. 准入控制（ADMISSION_MAX_INFLIGHT）：/chat 返回 HTTP 503，/chat/stream 返回 error 事件 code=503
        2. 请求截止时间（REQUEST_DEADLINE_S）：/chat 返回 504，/chat/stream 返回 error 事件 code=504
        3. 新增 /metrics；lifespan 启动时初始化资源、启动事件循环延迟监测
        4. 修复：关闭阶段按值导入的 pg_memory / http_client 恒为 None，资源未释放

    - 1.0.3 (2026-10-07，压测前优化):
        1. lifespan 启动时设置默认线程池 64（APP_THREAD_POOL_SIZE），替代默认 12
        2. 关闭阶段统一 Langfuse flush（请求路径不再逐次 flush）
        3. /chat/stream 未捕获异常记录类型与堆栈

    - 1.0.2 (2026-04-11):
        1. 新增 RateLimitExceeded 捕获 → HTTP 429 + Retry-After 响应头
        2. 新增 asyncio.wait_for 整体请求 30 秒兜底超时 → HTTP 504

    - 1.0.1 (2026-04-10):
        1. shutdown 新增 pg_memory 连接池关闭，防止 PG 连接泄漏
        2. @app.on_event("shutdown") 替换为 lifespan（FastAPI 推荐方式）

Author: Ke Meng
Created: 2026-03-18
Last Modified: 2026-04-10
"""

import asyncio
import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Literal
from src.core.auth import authenticate, current_user, ensure_users_table, issue_token, require_role
from src.core.security.events import ensure_security_events_table, record as record_security_event
from src.core.security.pii import PIIVault, StreamRestorer, deep_apply
from src.core.feedback import delete_feedback, ensure_feedback_table, save_feedback
from src.core.memory_long.long_term_memory import ensure_memory_schema
from src.core.session_store import ensure_deleted_sessions_table, is_session_deleted, soft_delete_session
from src.core.high_frequency_query_cache.rate_limiter import RateLimitExceeded
import src.main as main_mod
from src.core.memory_short.redis_short_memory import STM_METRICS
from src.core.memory_short.session_store_pg import ensure_session_state_table
from src.main import handle_request_async, handle_request_stream, background_tasks, init_resources
from src.core.resilience import (
    ADMISSION, AdmissionRejected, LOOP_LAG, register_executor, snapshot as resilience_snapshot,
)
from config.config_path import OVERLOAD_CONFIG

logger = logging.getLogger(__name__)

# 默认线程池大小：asyncio.to_thread（查询改写、长期记忆嵌入）与 LangGraph 同步节点（检索工具）共用。
# Python 默认 min(32, CPU+4) = 12（8 核），并发上来后会在线程池排队；须 ≥ 准入控制的在途上限
APP_THREAD_POOL_SIZE = int(os.getenv("APP_THREAD_POOL_SIZE", 64))


# ── lifespan：统一管理启动和关闭逻辑 ────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # === 启动阶段 ===
    executor = ThreadPoolExecutor(max_workers=APP_THREAD_POOL_SIZE, thread_name_prefix="app-worker")
    asyncio.get_running_loop().set_default_executor(executor)
    register_executor(executor)
    logger.info(f"默认线程池: {APP_THREAD_POOL_SIZE} 线程")
    # 启动时初始化 PG / 长期记忆（原为首请求懒加载，并发首批请求存在竞态）
    await init_resources()
    if main_mod.pg_memory is not None and main_mod.pg_memory.pool is not None:
        await ensure_users_table(main_mod.pg_memory.pool)
        await ensure_feedback_table(main_mod.pg_memory.pool)
        await ensure_deleted_sessions_table(main_mod.pg_memory.pool)
        await ensure_memory_schema(main_mod.pg_memory.pool)
        await ensure_security_events_table(main_mod.pg_memory.pool)
        await ensure_session_state_table(main_mod.pg_memory.pool)
    LOOP_LAG.start()
    logger.info(f"过载保护: 准入上限 {OVERLOAD_CONFIG['admission_max_inflight']}（≤0 不限），"
                f"排队 {OVERLOAD_CONFIG['admission_queue_timeout_s']}s，请求截止 {OVERLOAD_CONFIG['request_deadline_s']}s")
    yield
    # === 关闭阶段 ===
    await LOOP_LAG.stop()
    # 1. 等待后台任务（已有）
    if background_tasks:
        print(f"等待 {len(background_tasks)} 个后台任务完成...")
        done, pending = await asyncio.wait(background_tasks, timeout=30)
        if pending:
            print(f"警告：{len(pending)} 个后台任务未在 30 秒内完成")
        else:
            print("所有后台任务已完成")

    # 2. 关闭 HTTP 客户端
    # 注意从模块读取最新值：from src.main import http_client 拿到的是导入时的 None，关闭逻辑会被跳过
    if main_mod.http_client is not None:
        await main_mod.http_client.aclose()
        print("HTTP 客户端已关闭")

    # 3. 关闭 PG 连接池
    if main_mod.pg_memory is not None:
        await main_mod.pg_memory.close()
        print("PostgreSQL 连接池已关闭")

    # 4. Langfuse 统一 flush（请求路径上不再逐次 flush，避免阻塞事件循环）
    try:
        from langfuse import get_client
        await asyncio.to_thread(get_client().flush)
        print("Langfuse 已 flush")
    except Exception as e:
        print(f"Langfuse flush 失败: {e}")


app = FastAPI(
    title="ReAct Agent API",
    description="多轮对话系统",
    lifespan=lifespan,
    # 生产不公开接口文档：/docs、/redoc 关闭，/openapi.json 改为下方需 ops/admin 令牌的路由
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


# ── 请求/响应模型 ────────────────────────────────────────────────
class ChatRequest(BaseModel):
    user_input: str
    session_id: str
    user_id: Optional[str] = None   # 登录后以 token 中的用户为准；仅 AUTH_ENABLED=false 时使用


class LoginRequest(BaseModel):
    username: str
    password: str


class FeedbackRequest(BaseModel):
    session_id: str
    question: str
    answer: str
    rating: Literal[-1, 0, 1]   # 1 赞 / -1 踩 / 0 取消


def _resolve_user_id(request: ChatRequest, user: Optional[dict]) -> str:
    """已登录取 token 中的 user_id；认证关闭时取请求体，缺失则 400"""
    if user is not None:
        return user["user_id"]
    if not request.user_id:
        raise HTTPException(status_code=400, detail="缺少 user_id")
    return request.user_id


def _role(user: Optional[dict], default: str = "user") -> str:
    """当前角色；认证关闭（压测 / 离线评估）时按接口默认角色处理"""
    return user["role"] if user else default


# ── 敏感信息展示（按角色还原占位符）──────────────────────────────
# 不参与还原的字段：事件类型、枚举值、网址（网址里的占位符不还原，防外泄）
_DISPLAY_SKIP_KEYS = ("type", "status", "name", "intent", "question_type", "reason", "code",
                      "urls", "url", "domain", "web_urls", "evidence_level", "web_status")
# 被拦截网页的命中规则与原文片段只给 admin（排查误拦），其他角色只看状态与原因
_ADMIN_ONLY_SOURCE_KEYS = ("rule", "excerpt")


def _trim_sources(sources, role: str):
    if role == "admin" or not isinstance(sources, list):
        return sources
    return [{k: v for k, v in x.items() if k not in _ADMIN_ONLY_SOURCE_KEYS} if isinstance(x, dict) else x
            for x in sources]


def _audit_restore(types: set, where: str, role: str, user_id: str, session_id: Optional[str]) -> None:
    """admin 查看了敏感信息完整值 → 记还原审计（只记类型，不记原值）"""
    if role == "admin" and types:
        record_security_event("pii_restore", {"types": sorted(types), "where": where},
                              user_id=user_id, session_id=session_id, role=role)


def _display(obj, vault: PIIVault, role: str, restored: set):
    """展示前处理：原文中的敏感信息先换占位符，再按角色还原；restored 收集还原了的类型"""
    def _one(text: str) -> str:
        out, types = vault.display(text, role)
        restored.update(types)
        return out
    return deep_apply(obj, _one, skip_keys=_DISPLAY_SKIP_KEYS)


def _pg_pool():
    if main_mod.pg_memory is None or main_mod.pg_memory.pool is None:
        raise HTTPException(status_code=503, detail="数据库未就绪")
    return main_mod.pg_memory.pool


class ChatResponse(BaseModel):
    answer: str


# ── 路由 ─────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user: Optional[dict] = Depends(require_role("user", "admin"))):
    """
    处理用户对话请求

    异常处理：
        - AdmissionRejected → HTTP 503 + Retry-After 头（准入控制：在途请求已满）
        - RateLimitExceeded → HTTP 429 + Retry-After 头
        - asyncio.TimeoutError → HTTP 504（超过请求截止时间 REQUEST_DEADLINE_S）
        - 其他异常 → HTTP 500（由 FastAPI 默认处理）
    """
    uid, role = _resolve_user_id(request, user), _role(user)
    vault = main_mod.load_vault(uid, request.session_id)   # 与 handle_request_async 共用，保证占位符编号一致
    try:
        async with ADMISSION:
            answer = await asyncio.wait_for(
                handle_request_async(
                    request.user_input,
                    request.session_id,
                    user_id=uid,
                    role=role,
                    vault=vault,
                ),
                timeout=OVERLOAD_CONFIG['request_deadline_s'],
            )
        restored: set = set()
        answer = _display(answer, vault, role, restored)
        _audit_restore(restored, "chat", role, uid, request.session_id)
        return ChatResponse(answer=answer)

    except AdmissionRejected:
        return JSONResponse(
            status_code=503,
            content={"detail": "系统繁忙，请稍后再试"},
            headers={"Retry-After": "2"},
        )

    except RateLimitExceeded as e:
        # 限流触发：返回 429 + Retry-After 响应头
        return JSONResponse(
            status_code=429,
            content={"detail": e.reason},
            headers={"Retry-After": str(e.retry_after)},
        )

    except asyncio.TimeoutError:
        # 整体请求超时：返回 504
        return JSONResponse(
            status_code=504,
            content={"detail": "请求处理超时，请稍后再试"},
        )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, user: Optional[dict] = Depends(require_role("user", "admin"))):
    """
    流式对话接口（SSE）。
    返回 text/event-stream，每行格式：data: {JSON}\n\n
    事件类型：thinking / tool_call / answer / done / error

    过载保护：
        - 准入控制：在途请求已满且排队超时 → error 事件 code=503, reason=admission
        - 请求截止：处理超过 REQUEST_DEADLINE_S → error 事件 code=504, reason=deadline
      名额由 async with 在生成器结束时归还（含客户端断开、取消、异常）
    """
    uid = _resolve_user_id(request, user)   # 在生成器外解析，未登录时直接返回 401 而不是流内错误
    role = _role(user)

    async def event_generator():
        # 敏感信息展示：answer / thinking 为模型输出的占位符，逐块还原（缓存被拆开的占位符）；
        # 其他事件（改写、记忆、检索原文等）含原文，整体先脱敏再按角色还原
        vault = main_mod.load_vault(uid, request.session_id)
        streams = {"answer": StreamRestorer(vault, role), "thinking": StreamRestorer(vault, role)}
        restored: set = set()

        def _flush(skip: Optional[str] = None):
            for typ, r in streams.items():
                if typ != skip:
                    rest = r.flush()
                    if rest:
                        yield f"data: {json.dumps({'type': typ, 'content': rest}, ensure_ascii=False)}\n\n"

        def _render(chunk: dict):
            typ = chunk.get("type")
            if typ in streams:
                yield from _flush(skip=typ)        # 切换输出流前先清空另一路缓存，保证顺序
                text = streams[typ].feed(chunk.get("content", ""))
                if text:
                    yield f"data: {json.dumps({**chunk, 'content': text}, ensure_ascii=False)}\n\n"
                return
            yield from _flush()
            if typ == "web_sources":
                chunk = {**chunk, "sources": _trim_sources(chunk.get("sources"), role)}
            yield f"data: {json.dumps(_display(chunk, vault, role, restored), ensure_ascii=False)}\n\n"

        try:
            async with ADMISSION:
                # 截止时间只包住"取下一个事件"的 await，不包住 yield：
                # 生成器挂起在 yield 时若触发超时取消，取消会落在 Starlette 的发送协程上，响应被直接中断
                loop = asyncio.get_running_loop()
                deadline_at = loop.time() + OVERLOAD_CONFIG['request_deadline_s']
                agen = handle_request_stream(
                    request.user_input,
                    request.session_id,
                    user_id=uid,
                    role=role,
                    vault=vault,
                )
                try:
                    while True:
                        cm = asyncio.timeout_at(deadline_at)
                        try:
                            async with cm:
                                chunk = await agen.__anext__()
                        except StopAsyncIteration:
                            break
                        except TimeoutError:
                            if not cm.expired():
                                raise              # 内部组件抛出的超时，按未捕获异常处理
                            logger.warning(f"[stream] 超过请求截止时间 {OVERLOAD_CONFIG['request_deadline_s']}s "
                                           f"session_id={request.session_id}")
                            err = {"type": "error", "code": 504, "reason": "deadline",
                                   "detail": "请求处理超时，请稍后再试"}
                            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                            break
                        for line in _render(chunk):
                            yield line
                    for line in _flush():
                        yield line
                finally:
                    await agen.aclose()
                    for r in streams.values():
                        restored |= r.restored
                    _audit_restore(restored, "chat_stream", role, uid, request.session_id)
        except AdmissionRejected:
            err = {"type": "error", "code": 503, "reason": "admission", "detail": "系统繁忙，请稍后再试"}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        except Exception as e:
            # 记录异常类型与堆栈（压测中出现过 str(e) 为空、无法定位的 500）
            logger.exception(f"[stream] 未捕获异常 session_id={request.session_id}: {type(e).__name__}: {e}")
            err = {"type": "error", "code": 500, "detail": str(e) or type(e).__name__}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── 工具直调（供 MCP 服务使用，只读）─────────────────────────────
class ToolRetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    question_type: Optional[Literal["simple", "reasoning", "multi_hop"]] = None


class ToolVerifyRequest(BaseModel):
    entities: List[str] = Field(min_length=1, max_length=5)
    max_hops: int = Field(default=3, ge=1, le=3)


async def _run_tool(coro):
    """工具接口与 /chat 共用过载保护：准入控制 + 请求截止；限流 429、准入 503、超时 504"""
    try:
        async with ADMISSION:
            return await asyncio.wait_for(coro, timeout=OVERLOAD_CONFIG['request_deadline_s'])
    except AdmissionRejected:
        return JSONResponse(status_code=503, content={"detail": "系统繁忙，请稍后再试"}, headers={"Retry-After": "2"})
    except RateLimitExceeded as e:
        return JSONResponse(status_code=429, content={"detail": e.reason}, headers={"Retry-After": str(e.retry_after)})
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"detail": "请求处理超时，请稍后再试"})
    finally:
        coro.close()   # 准入被拒时协程未被 await，关闭以免 "never awaited" 警告；已执行完的协程 close 无副作用


def _mask_external(result):
    """工具接口的调用方（MCP 客户端，如 Claude Code）在服务器外：知识库原文中的敏感信息只给占位符，不还原"""
    if isinstance(result, (dict, list)):
        return deep_apply(result, PIIVault().mask, skip_keys=_DISPLAY_SKIP_KEYS)
    return result


@app.post("/tools/retrieve")
async def tools_retrieve(req: ToolRetrieveRequest, user: Optional[dict] = Depends(require_role("user", "admin"))):
    """知识库检索（改写 → 检索 → 证据分级），返回段落与证据等级；不经主模型、不写记忆"""
    uid = user["user_id"] if user else "tools_anonymous"
    return _mask_external(await _run_tool(main_mod.tool_retrieve(req.query, uid, req.question_type)))


@app.post("/tools/verify")
async def tools_verify(req: ToolVerifyRequest, user: Optional[dict] = Depends(require_role("user", "admin"))):
    """知识图谱实体关系查询：1 个实体返回直接相连的事实，多个实体返回之间的多跳路径；实体用英文原名或中文名"""
    uid = user["user_id"] if user else "tools_anonymous"
    return _mask_external(await _run_tool(main_mod.tool_verify(req.entities, uid, req.max_hops)))


# ── 登录与会话 ───────────────────────────────────────────────────
@app.post("/auth/login")
async def login(req: LoginRequest):
    """用户名密码登录，成功返回 JWT；账号由 scripts/create_user.py 创建"""
    user = await authenticate(_pg_pool(), req.username.strip(), req.password)
    if user is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return {"token": issue_token(user["user_id"], user["username"], user["role"]), **user}


@app.get("/auth/me")
async def me(user: Optional[dict] = Depends(current_user)):
    if user is None:
        raise HTTPException(status_code=401, detail="未登录")
    return user


@app.get("/sessions")
async def list_sessions(user: Optional[dict] = Depends(require_role("user", "admin")), limit: int = 50):
    """当前用户的会话列表（按最近活动倒序），标题取该会话第一条用户消息"""
    if user is None:
        raise HTTPException(status_code=401, detail="未登录")
    sql = """
        SELECT session_id,
               (array_agg(content ORDER BY created_at, id) FILTER (WHERE role = 'user'))[1] AS title,
               MAX(created_at) AS updated_at,
               COUNT(*) FILTER (WHERE role = 'user') AS turns
        FROM   conversation_history c
        WHERE  user_id = $1
          AND  NOT EXISTS (SELECT 1 FROM deleted_sessions d WHERE d.user_id = c.user_id AND d.session_id = c.session_id)
        GROUP  BY session_id
        ORDER  BY MAX(created_at) DESC
        LIMIT  $2
    """
    async with _pg_pool().acquire() as conn:
        rows = await conn.fetch(sql, user["user_id"], min(max(limit, 1), 200))

    def _titles() -> list:
        # 标题按各会话的映射表还原（Redis 读，放线程池）
        restored: set = set()
        out = [_display(r["title"] or "新对话", main_mod.load_vault(user["user_id"], r["session_id"]),
                        user["role"], restored)[:40] for r in rows]
        _audit_restore(restored, "session_titles", user["role"], user["user_id"], None)
        return out
    titles = await asyncio.to_thread(_titles)
    return [{"session_id": r["session_id"], "title": t,
             "updated_at": r["updated_at"].isoformat(), "turns": r["turns"]} for r, t in zip(rows, titles)]


@app.get("/sessions/{session_id}/messages")
async def session_messages(session_id: str, user: Optional[dict] = Depends(require_role("user", "admin"))):
    """某个会话的全部消息（只能读取自己的会话；已删除的会话返回 404）"""
    if user is None:
        raise HTTPException(status_code=401, detail="未登录")
    if await is_session_deleted(_pg_pool(), user["user_id"], session_id):
        raise HTTPException(status_code=404, detail="会话不存在或已删除")
    sql = """
        SELECT role, content, created_at, metadata
        FROM   conversation_history
        WHERE  user_id = $1 AND session_id = $2 AND role IN ('user', 'assistant')
        ORDER  BY created_at, id
        LIMIT  400
    """
    async with _pg_pool().acquire() as conn:
        rows = await conn.fetch(sql, user["user_id"], session_id)
    def _trace(r):
        # metadata 为 JSONB（asyncpg 默认返回字符串）；助手消息的 trace 记录处理过程，供前端还原步骤与原文
        if r["role"] != "assistant" or not r["metadata"]:
            return None
        md = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]
        trace = md.get("trace")
        if isinstance(trace, dict) and "web_sources" in trace:
            trace = {**trace, "web_sources": _trim_sources(trace["web_sources"], user["role"])}
        return trace
    # 历史按角色还原（映射表 24h 过期后只显示占位符）；脱敏上线前的旧记录含原文，同样先脱敏再还原
    vault = await asyncio.to_thread(main_mod.load_vault, user["user_id"], session_id)
    restored: set = set()
    out = [{"role": r["role"], "content": _display(r["content"], vault, user["role"], restored),
            "created_at": r["created_at"].isoformat(), "trace": _display(_trace(r), vault, user["role"], restored)}
           for r in rows]
    _audit_restore(restored, "session_messages", user["role"], user["user_id"], session_id)
    return out


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user: Optional[dict] = Depends(require_role("user", "admin"))):
    """软删除会话：记入 deleted_sessions（原始对话保留供数据飞轮使用），并清空该会话的短期记忆（Redis + PG 会话状态）"""
    if user is None:
        raise HTTPException(status_code=401, detail="未登录")
    await soft_delete_session(_pg_pool(), user["user_id"], session_id)
    await main_mod.stm.clear(user["user_id"], session_id)   # 内部吞异常，不影响删除
    return {"ok": True}


@app.get("/memories")
async def list_memories(user: Optional[dict] = Depends(require_role("user", "admin"))):
    """当前用户的长期记忆（有效记忆，按重要性与更新时间排序）"""
    if user is None:
        raise HTTPException(status_code=401, detail="未登录")
    sql = """
        SELECT id::text, content, memory_type, importance_score, access_count, updated_at
        FROM   memories
        WHERE  user_id = $1 AND is_active AND content_en IS NOT NULL
        ORDER  BY importance_score DESC, updated_at DESC
        LIMIT  100
    """
    async with _pg_pool().acquire() as conn:
        rows = await conn.fetch(sql, user["user_id"])
    # 长期记忆是用户级的，跨会话占位符无法还原；旧记忆中的原文按角色展示
    vault, restored = PIIVault(), set()
    out = [{"id": r["id"], "content": _display(r["content"], vault, user["role"], restored), "memory_type": r["memory_type"],
            "importance": round(r["importance_score"], 2), "hits": r["access_count"],
            "updated_at": r["updated_at"].isoformat()} for r in rows]
    _audit_restore(restored, "memories", user["role"], user["user_id"], None)
    return out


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str, user: Optional[dict] = Depends(require_role("user", "admin"))):
    """软删除当前用户的一条长期记忆（is_active=FALSE，不再召回、不再展示；只能删自己的）"""
    if user is None:
        raise HTTPException(status_code=401, detail="未登录")
    try:
        uuid.UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="记忆 ID 无效")
    async with _pg_pool().acquire() as conn:
        res = await conn.execute(
            "UPDATE memories SET is_active = FALSE, updated_at = NOW() WHERE id = $1::uuid AND user_id = $2 AND is_active",
            memory_id, user["user_id"])
    if res.endswith(" 0"):
        raise HTTPException(status_code=404, detail="记忆不存在")
    return {"ok": True}


@app.post("/feedback")
async def feedback(req: FeedbackRequest, user: Optional[dict] = Depends(require_role("user", "admin"))):
    """回答评价：写入 message_feedback；rating=0 表示取消评价"""
    if user is None:
        raise HTTPException(status_code=401, detail="未登录")
    # 前端回传的是展示文本（admin 为完整值）：入库前脱敏；同一文本脱敏结果确定，取消评价时哈希仍能对上
    question, answer = PIIVault().mask(req.question), PIIVault().mask(req.answer)
    if req.rating == 0:
        await delete_feedback(_pg_pool(), user["user_id"], req.session_id, answer)
    else:
        await save_feedback(_pg_pool(), user["user_id"], req.session_id, question[:2000], answer[:20000], req.rating)
    return {"ok": True}


# ── 前端页面 ─────────────────────────────────────────────────────
WEB_DIR = Path(__file__).resolve().parent / "web"
app.mount("/ui/static", StaticFiles(directory=WEB_DIR / "static"), name="ui-static")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/ui")


@app.get("/ui", include_in_schema=False)
async def ui():
    # 禁止浏览器缓存页面：每次部署后刷新即可拿到新版本（页面很小，不影响性能）
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics(user: Optional[dict] = Depends(require_role("ops", "admin"))):
    """
    （ops / admin；压测关闭认证时放行）
    运行时指标（JSON）：准入、各舱壁、重试预算、事件循环延迟、线程池排队、后台任务数。
    峰值类指标（peak_in_use、loop_lag.max_ms）读取后重置，每次采样反映"上次采样以来"的峰值。
    """
    data = resilience_snapshot()
    data["background_tasks"] = len(background_tasks)
    data["stm"] = STM_METRICS.snapshot()
    return data


@app.get("/openapi.json", include_in_schema=False)
async def openapi_schema(user: Optional[dict] = Depends(require_role("ops", "admin"))):
    """接口定义（ops / admin），用于联调与排查；不提供交互式文档页面"""
    return app.openapi()


# ── 飞轮评估专用接口（不影响生产路径）────────────────────────────
from src.eval_handler import handle_request_eval
from src.core.high_frequency_query_cache.circuit_breaker import CircuitBreakerOpen


class ChatEvalResponse(BaseModel):
    answer: str
    contexts: list[str]
    intent: str


@app.post("/chat_eval", response_model=ChatEvalResponse)
async def chat_eval(request: ChatRequest, user: Optional[dict] = Depends(require_role("ops", "admin"))):
    """
    飞轮离线评估专用接口。
    走完整 RAG 推理，返回 answer + retrieved_contexts，供 RAGAS 计算。
    不写缓存/短期记忆/长期记忆/PG，与生产 /chat 完全隔离。
    """
    try:
        result = await asyncio.wait_for(
            handle_request_eval(
                request.user_input,
                request.session_id,
                user_id=_resolve_user_id(request, user),
                role=_role(user, "ops"),
            ),
            timeout=240,
        )
        return ChatEvalResponse(answer=result["answer"], contexts=result["contexts"], intent=result["intent"])

    except CircuitBreakerOpen:
        return JSONResponse(
            status_code=503,
            content={"detail": "系统暂时繁忙，请稍后再试"},
        )

    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"detail": "评估请求超时"},
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
