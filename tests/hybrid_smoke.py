"""Hybrid (sparse candidates -> dense cosine re-rank) over Redis. Uses fakeredis +
the host ollama for query embedding."""
import asyncio
import os
import time

import fakeredis.aioredis

from ontorag_mcp.redis_store import RedisDataset
from ontorag_mcp.store import resolve_source


async def main():
    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    ds = RedisDataset(r, "openfantasymap/amol-ontorag", resolve_source("/data"),
                      ttl=86400, mode="hybrid", ollama_url=os.environ["OLLAMA_URL"])

    t = time.time()
    await ds.ensure()                       # populate: ontology keys + per-chunk vectors
    print("populate(hybrid): %.1fs" % (time.time() - t))
    print("emb config:", await r.get(ds.P + "emb"))
    veckeys = [k async for k in r.scan_iter(match=ds.P + "vec:*", count=5000)]
    print("vec keys stored:", len(veckeys))

    for q in ["How does the Parma Magica grant magic resistance?",
              "ways to protect a stronghold from hostile sorcery"]:
        print("\nQ:", q)
        for h in await ds.search(q, k=3):
            print("  %.3f %-26s %s" % (h["score"], h["doc"], h["entities"][:2]))

    a = await ds.answer("How does House Tremere use certamen?", k=4, expand=2)
    print("\nanswer matched:", a["matched_entities"][:5], "| passages:", len(a["passages"]))

    t = time.time()
    await ds.search("certamen", k=2)
    print("warm query: %.3fs" % (time.time() - t))
    print("OK")


asyncio.run(main())
