"""
长期记忆存量清理：按写入前校验中"只看正文"的规则（提问记录 / 占位符 / 格式不符）找出旧的错误记忆并软删除

默认只读预览，不改数据；确认后加 --apply 执行软删除（is_active=FALSE，可恢复）。
归纳类（reflection）记忆由反思任务生成，不在本脚本自动处理范围内，只列出供人工判断。

运行（app 容器内）：
    docker compose exec app python scripts/ltm_cleanup.py                 # 预览全部用户
    docker compose exec app python scripts/ltm_cleanup.py --user <ID>     # 预览单个用户
    docker compose exec app python scripts/ltm_cleanup.py --apply         # 执行软删除
恢复（如误删）：UPDATE memories SET is_active = TRUE WHERE id = ANY(<预览中列出的 id>)
"""

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


async def main(args) -> int:
    import asyncpg
    from config.config_path import PG_CONFIG
    from src.core.memory_long.long_term_memory import memory_reject_reason

    conn = await asyncpg.connect(PG_CONFIG["dsn"])
    try:
        sql = """SELECT id::text, user_id, content, content_en, memory_type, access_count
                 FROM memories WHERE is_active AND content_en IS NOT NULL"""
        rows = await conn.fetch(sql + (" AND user_id = $1" if args.user else "") + " ORDER BY user_id, created_at",
                                *([args.user] if args.user else []))
        bad, reflections = [], []
        for r in rows:
            if r["memory_type"] == "reflection":
                reflections.append(r)
                continue
            reason = memory_reject_reason({"content": r["content"], "content_en": r["content_en"]}, None)
            if reason:
                bad.append((r, reason))

        print(f"有效记忆 {len(rows)} 条，判为错误记忆 {len(bad)} 条，归纳类 {len(reflections)} 条（不自动处理）\n")
        print("按原因：", dict(Counter(reason for _, reason in bad)))
        print("按用户：", dict(Counter(r["user_id"] for r, _ in bad)), "\n")
        for r, reason in bad:
            print(f"  [{reason:<12}] {r['memory_type']:<10} 召回{r['access_count']:>3}次  {r['content'][:50]}  ({r['id']})")
        if reflections:
            print("\n归纳类（请人工判断，可在前端'我的记忆'中逐条删除）：")
            for r in reflections:
                print(f"  {r['content'][:60]}  ({r['id']})")

        if not bad:
            return 0
        if not args.apply:
            print(f"\n以上为预览，未修改数据。确认后加 --apply 软删除这 {len(bad)} 条。")
            return 0
        ids = [r["id"] for r, _ in bad]
        res = await conn.execute(
            "UPDATE memories SET is_active = FALSE, updated_at = NOW() WHERE id = ANY($1::uuid[]) AND is_active", ids)
        print(f"\n已软删除：{res}")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="长期记忆存量清理（默认只读预览）")
    ap.add_argument("--user", default="", help="只处理指定用户")
    ap.add_argument("--apply", action="store_true", help="执行软删除")
    sys.exit(asyncio.run(main(ap.parse_args())))
