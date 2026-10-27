"""
请求安全上下文 (security context)

每个请求在入口（main.py 的 handle_request_*）设置当前用户、会话、角色与脱敏映射表，下游模块无需逐层传参即可读取：
    - 工具节点按角色检查工具权限（ROLE_TOOL_POLICY），并用映射表还原本地工具参数
    - 主 LLM 封装层、持久化、日志、Langfuse 用映射表脱敏
    - 安全事件记录当前用户与会话
ContextVar 随 asyncio 任务复制；asyncio.to_thread 与 LangChain 的线程池调用都会复制上下文，同步工具内也能读到。

Author: Ke Meng
Created: 2026-10-17
"""

from contextvars import ContextVar
from typing import Optional

from config.config_path import ROLE_TOOL_POLICY

_ctx: ContextVar[Optional[dict]] = ContextVar("security_ctx", default=None)


def set_context(user_id: str, session_id: str, role: str = "user", vault=None) -> None:
    _ctx.set({"user_id": user_id, "session_id": session_id, "role": role, "vault": vault})


def get_context() -> dict:
    return _ctx.get() or {"user_id": None, "session_id": None, "role": "user"}


def tool_allowed(tool_name: str, role: Optional[str] = None) -> bool:
    """角色是否允许使用该工具；策略为 "*" 表示全部工具，未配置的角色一律拒绝"""
    policy = ROLE_TOOL_POLICY.get(role or get_context()["role"])
    return policy == "*" or (policy is not None and tool_name in policy)
