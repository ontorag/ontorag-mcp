"""Drive the Vercel ASGI entrypoint (api/index.py:app) with the real MCP client
over streamable-http, to confirm the stateless serverless shape works."""
import asyncio
import json


async def main():
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client("http://127.0.0.1:8799/mcp") as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("tools:", [t.name for t in tools.tools])
            res = await s.call_tool("search", {"query": "How does House Tremere use certamen?", "k": 2})
            print("isError:", res.isError)
            hits = [json.loads(c.text) for c in res.content if hasattr(c, "text")]
            print("search hits:", [(round(h.get("score", 0), 3), h.get("doc")) for h in hits][:2])
    print("OK")


asyncio.run(main())
