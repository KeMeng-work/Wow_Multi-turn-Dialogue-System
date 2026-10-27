"""
登录认证模块 (auth)

账号不开放注册，由 scripts/create_user.py 创建。密码 bcrypt 哈希存 PG，登录成功后签发 JWT（HS256）。
对话 / 会话接口通过 current_user 依赖解析当前用户：
    - AUTH_ENABLED=true（默认）：必须携带 Authorization: Bearer <token>，user_id 取自 token，忽略请求体
    - AUTH_ENABLED=false：不校验，user_id 取请求体（压测、离线评估脚本使用）

角色（2026-10-17）：admin / user / ops，登录时写入 JWT，接口用 require_role 限制：
    - user：对话、会话历史、长期记忆、反馈、/tools/*（MCP 服务账号也是 user）
    - ops（测试与运维）：/chat_eval、/metrics、/openapi.json；不能对话，也读不到用户数据
    - admin：全部
  角色变更后需重新登录才生效；旧 token 无 role 字段，按 user 处理。

Author: Ke Meng
Created: 2026-10-09
"""

import asyncio
import logging
import secrets
import time
import uuid
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, Header, HTTPException, Request

from config.config_path import AUTH_CONFIG

logger = logging.getLogger(__name__)

USERS_DDL = """
CREATE TABLE IF NOT EXISTS app_users (
    user_id        VARCHAR(255) PRIMARY KEY,
    username       VARCHAR(64)  NOT NULL UNIQUE,
    password_hash  TEXT         NOT NULL,
    disabled       BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    last_login_at  TIMESTAMPTZ
);
ALTER TABLE app_users ADD COLUMN IF NOT EXISTS role VARCHAR(16) NOT NULL DEFAULT 'user';
"""

ROLES = ("admin", "user", "ops")

_secret = AUTH_CONFIG['jwt_secret'] or secrets.token_urlsafe(32)
if not AUTH_CONFIG['jwt_secret']:
    logger.warning("JWT_SECRET 未设置，已随机生成：服务重启后需重新登录")


# ── 密码 ────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


# ── JWT ─────────────────────────────────────────────────────────
def issue_token(user_id: str, username: str, role: str = "user") -> str:
    now = int(time.time())
    payload = {"sub": user_id, "name": username, "role": role, "iat": now, "exp": now + AUTH_CONFIG['jwt_expire_h'] * 3600}
    return jwt.encode(payload, _secret, algorithm="HS256")


def decode_token(token: str) -> dict:
    return jwt.decode(token, _secret, algorithms=["HS256"])


# ── 用户表操作（asyncpg 连接池）─────────────────────────────────
async def ensure_users_table(pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(USERS_DDL)


async def authenticate(pool, username: str, password: str) -> Optional[dict]:
    """校验用户名密码，成功返回 {user_id, username, role}；bcrypt 校验放线程池，避免阻塞事件循环"""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT user_id, username, password_hash, disabled, role FROM app_users WHERE username = $1", username)
    if row is None or row["disabled"]:
        return None
    if not await asyncio.to_thread(verify_password, password, row["password_hash"]):
        return None
    async with pool.acquire() as conn:
        await conn.execute("UPDATE app_users SET last_login_at = NOW() WHERE user_id = $1", row["user_id"])
    return {"user_id": row["user_id"], "username": row["username"], "role": row["role"]}


async def create_user(pool, username: str, password: str, role: str = "user") -> str:
    user_id = f"u_{uuid.uuid4().hex[:12]}"
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO app_users (user_id, username, password_hash, role) VALUES ($1, $2, $3, $4)",
            user_id, username, hash_password(password), role)
    return user_id


# ── FastAPI 依赖 ────────────────────────────────────────────────
def current_user(authorization: Optional[str] = Header(default=None)) -> Optional[dict]:
    """
    解析当前登录用户。
    返回 {user_id, username, role}；认证关闭且未带 token 时返回 None（由调用方使用请求体中的 user_id）。
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        if AUTH_CONFIG['enabled']:
            raise HTTPException(status_code=401, detail="未登录或登录已过期")
        return None
    try:
        payload = decode_token(authorization.split(" ", 1)[1])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="登录凭证无效")
    return {"user_id": payload["sub"], "username": payload.get("name", ""), "role": payload.get("role", "user")}


def require_role(*roles: str):
    """
    接口角色限制依赖：用法 Depends(require_role("ops", "admin"))，返回值与 current_user 相同。
    角色不符 → 403 并记录安全事件；认证关闭且未带 token（压测）时放行，返回 None。
    """
    async def _dep(request: Request, user: Optional[dict] = Depends(current_user)) -> Optional[dict]:
        if user is None or user["role"] in roles:
            return user
        from src.core.security.events import record   # 延迟导入，避免 scripts/create_user.py 引入事件模块
        record("permission_denied", {"path": request.url.path, "required": list(roles)},
               user_id=user["user_id"], role=user["role"], session_id=None)
        raise HTTPException(status_code=403, detail="当前账号无权访问该功能")
    return _dep
