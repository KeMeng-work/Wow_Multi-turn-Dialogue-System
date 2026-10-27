"""
MCP 服务冒烟测试：连接 → 列出工具 → 逐个调用 → 校验返回结构

用法（本地运行，需安装 mcp：pip install mcp==1.30.0）：
    python scripts/test_mcp.py --url http://<服务器IP>:8100/mcp --token <MCP_ACCESS_TOKEN>
    也可用环境变量 MCP_URL / MCP_ACCESS_TOKEN 代替参数。
    --skip-agent 跳过 ask_agent（它会调用主模型，耗时数秒到数十秒）

脚本不使用系统代理（trust_env=False），开着 VPN 也直连服务器。
"""
import argparse
import asyncio
import os
import sys
import time

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

EXPECTED_TOOLS = {"search_knowledge_base", "verify_relations", "ask_agent"}


def _no_proxy_client(headers=None, timeout=None, auth=None) -> httpx.AsyncClient:
    return httpx.AsyncClient(headers=headers, timeout=timeout or httpx.Timeout(30, read=300),
                             auth=auth, trust_env=False)


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'✅' if ok else '❌'} {name}{('：' + detail) if detail else ''}")
    return ok


async def call(session: ClientSession, tool: str, args: dict):
    t0 = time.perf_counter()
    res = await session.call_tool(tool, args)
    ms = int((time.perf_counter() - t0) * 1000)
    if res.isError:
        text = res.content[0].text if res.content else ""
        return None, ms, text
    return res.structuredContent, ms, ""


async def main(url: str, token: str, skip_agent: bool) -> int:
    results = []
    async with streamablehttp_client(url, headers={"Authorization": f"Bearer {token}"},
                                     httpx_client_factory=_no_proxy_client) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("1. 工具发现")
            tools = {t.name: t for t in (await session.list_tools()).tools}
            results.append(check("工具齐全", EXPECTED_TOOLS <= set(tools), str(sorted(tools))))
            results.append(check("全部标注只读", all(t.annotations and t.annotations.readOnlyHint for t in tools.values())))

            print("2. search_knowledge_base（中文问题，服务端改写为英文）")
            out, ms, err = await call(session, "search_knowledge_base", {"query": "谁导演了电影《星际穿越》？"})
            results.append(check("调用成功", out is not None, err or f"{ms} ms"))
            if out:
                results.append(check("证据等级合法", out.get("evidence_level") in ("high", "low", "none"),
                                     f"{out.get('evidence_level')}，段落 {len(out.get('passages', []))} 条，"
                                     f"en_query='{out.get('en_query')}'"))

            print("3. search_knowledge_base（知识库外问题，应为 none）")
            out, ms, err = await call(session, "search_knowledge_base", {"query": "我们公司下季度的销售目标是多少？"})
            if out:
                results.append(check("无依据时不返回段落",
                                     out.get("evidence_level") == "none" and not out.get("passages"),
                                     f"{out.get('evidence_level')}，{ms} ms"))
            else:
                results.append(check("调用成功", False, err))

            print("4. verify_relations（多实体路径：NeXT → Steve Jobs → Apple → iOS）")
            out, ms, err = await call(session, "verify_relations", {"entities": ["NeXT", "iOS"]})
            results.append(check("找到多跳路径", bool(out and out.get("paths")), err or
                                 f"{ms} ms，路径 {len((out or {}).get('paths', []))} 条，"
                                 f"事实 {len((out or {}).get('facts', []))} 条"))

            print("5. verify_relations（单实体邻域 + 图谱未收录的实体）")
            out, ms, err = await call(session, "verify_relations", {"entities": ["Steve Jobs", "Interstellar"]})
            results.append(check("未收录实体单独列出", bool(out) and "Interstellar" in out.get("not_in_graph", [])
                                 and bool(out.get("facts")), err or f"{ms} ms，未收录 {(out or {}).get('not_in_graph')}"))

            print("6. 参数校验（超过 5 个实体应被拒绝）")
            out, ms, err = await call(session, "verify_relations", {"entities": ["a", "b", "c", "d", "e", "f"]})
            results.append(check("非法参数返回错误", out is None, err[:80]))

            if not skip_agent:
                print("7. ask_agent（完整问答，较慢）")
                out, ms, err = await call(session, "ask_agent", {"question": "介绍一下《星际穿越》的导演"})
                results.append(check("调用成功", out is not None and bool(out.get("answer")), err or f"{ms} ms"))
                if out:
                    print(f"     回答前 120 字：{out['answer'][:120]}")
                    print(f"     证据等级：{out.get('evidence_level')}，会话：{out.get('session_id')}")

    print(f"\n通过 {sum(1 for r in results if r)}/{len(results)}")
    return 0 if all(results) else 1


async def check_auth(url: str) -> bool:
    """无令牌访问应被拒绝（401）"""
    async with httpx.AsyncClient(trust_env=False, timeout=10) as c:
        resp = await c.post(url, json={})
    return check("无令牌访问被拒绝", resp.status_code == 401, f"HTTP {resp.status_code}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=os.getenv("MCP_URL"))
    p.add_argument("--token", default=os.getenv("MCP_ACCESS_TOKEN"))
    p.add_argument("--skip-agent", action="store_true")
    a = p.parse_args()
    if not a.url or not a.token:
        sys.exit("需要 --url 与 --token（或环境变量 MCP_URL / MCP_ACCESS_TOKEN）")
    print("0. 鉴权")
    auth_ok = asyncio.run(check_auth(a.url))
    code = asyncio.run(main(a.url, a.token, a.skip_agent))
    sys.exit(code if auth_ok else 1)
