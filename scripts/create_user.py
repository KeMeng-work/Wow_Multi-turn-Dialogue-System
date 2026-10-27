"""
账号管理脚本（前端 /ui 登录用，不开放注册）

在 app 容器内运行（项目根目录）：
    docker compose exec app python scripts/create_user.py create alice            # 随机生成 12 位密码并打印
    docker compose exec app python scripts/create_user.py create alice --password xxx
    docker compose exec app python scripts/create_user.py create ops --role ops       # 角色 admin / user（默认）/ ops
    docker compose exec app python scripts/create_user.py set-role kemeng admin        # 修改角色，重新登录后生效
    docker compose exec app python scripts/create_user.py passwd alice            # 重置为随机密码
    docker compose exec app python scripts/create_user.py disable alice           # 停用（保留会话与记忆）
    docker compose exec app python scripts/create_user.py enable alice
    docker compose exec app python scripts/create_user.py list
"""

import argparse
import asyncio
import secrets
import string
import sys
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config_path import PG_CONFIG
from src.core.auth import ROLES, USERS_DDL, create_user, hash_password


def random_password(n: int = 12) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(n))


async def main(args):
    conn = await asyncpg.connect(PG_CONFIG["dsn"])
    try:
        await conn.execute(USERS_DDL)
        if args.cmd == "list":
            rows = await conn.fetch(
                "SELECT username, user_id, role, disabled, created_at, last_login_at FROM app_users ORDER BY created_at")
            if not rows:
                print("（暂无账号）")
            for r in rows:
                state = "停用" if r["disabled"] else "正常"
                last = r["last_login_at"].strftime("%Y-%m-%d %H:%M") if r["last_login_at"] else "从未登录"
                print(f"{r['username']:<16} {r['user_id']:<16} {r['role']:<6} {state}  创建 {r['created_at']:%Y-%m-%d}  最近登录 {last}")
            return

        if args.cmd == "create":
            password = args.password or random_password()

            class _Pool:  # create_user 需要连接池接口，这里用单连接包装
                def acquire(self_inner):
                    class _Ctx:
                        async def __aenter__(self_c): return conn
                        async def __aexit__(self_c, *a): return False
                    return _Ctx()

            try:
                user_id = await create_user(_Pool(), args.username, password, args.role)
            except asyncpg.UniqueViolationError:
                print(f"用户名已存在：{args.username}")
                return
            print(f"已创建  用户名 {args.username}  角色 {args.role}  密码 {password}  user_id {user_id}")
            return

        if args.cmd == "passwd":
            password = args.password or random_password()
            n = await conn.execute("UPDATE app_users SET password_hash = $1 WHERE username = $2",
                                   hash_password(password), args.username)
            print(f"已重置  用户名 {args.username}  新密码 {password}" if n.endswith("1") else f"用户不存在：{args.username}")
            return

        if args.cmd == "set-role":
            n = await conn.execute("UPDATE app_users SET role = $1 WHERE username = $2", args.role, args.username)
            print(f"已设置  {args.username} → {args.role}（重新登录后生效）" if n.endswith("1")
                  else f"用户不存在：{args.username}")
            return

        if args.cmd in ("disable", "enable"):
            n = await conn.execute("UPDATE app_users SET disabled = $1 WHERE username = $2",
                                   args.cmd == "disable", args.username)
            print(f"已{'停用' if args.cmd == 'disable' else '启用'}：{args.username}" if n.endswith("1")
                  else f"用户不存在：{args.username}")
    finally:
        await conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="前端登录账号管理")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("create", "passwd"):
        sp = sub.add_parser(name)
        sp.add_argument("username")
        sp.add_argument("--password", help="不填则随机生成 12 位")
        if name == "create":
            sp.add_argument("--role", choices=ROLES, default="user")
    sp = sub.add_parser("set-role")
    sp.add_argument("username")
    sp.add_argument("role", choices=ROLES)
    for name in ("disable", "enable"):
        sub.add_parser(name).add_argument("username")
    sub.add_parser("list")
    asyncio.run(main(ap.parse_args()))
