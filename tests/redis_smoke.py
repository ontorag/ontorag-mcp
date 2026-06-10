"""Redis-backed ontology mode: populate (with TTL) + query. Uses fakeredis (the
real redis.asyncio API) so it runs without a server."""
import asyncio
import time

import fakeredis.aioredis

from ontorag_mcp.redis_store import RedisDataset
from ontorag_mcp.store import resolve_source


async def main():
    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    ds = RedisDataset(r, "openfantasymap/amol-ontorag", resolve_source("/data"), ttl=86400)

    t = time.time()
    await ds.ensure()                      # cache miss -> populate Redis with TTL
    print("populate: %.1fs" % (time.time() - t))
    P = ds.P
    print("TTL  ready=%s  meta=%s  (data keys ~86400, ready expires earlier)"
          % (await r.ttl(P + "ready"), await r.ttl(P + "meta")))
    print("counts:", (await ds.info())["counts"])

    for q in ["How does the Parma Magica grant magic resistance?",
              "How does House Tremere use certamen to settle disputes?",
              "ways to protect a stronghold from hostile sorcery"]:
        print("\nQ:", q)
        for h in await ds.search(q, k=3):
            print("  %.3f %-26s %s" % (h["score"], h["doc"], h["entities"][:2]))

    a = await ds.answer("How does House Tremere use certamen?", k=4, expand=2)
    print("\nanswer matched:", a["matched_entities"][:5], "| passages:", len(a["passages"]))
    e = await ds.get_entity("Parma Magica")
    print("get_entity:", {k: e[k] for k in ("label", "types", "linked_chunks")} if e else None)

    t = time.time()
    await ds.search("certamen", k=2)        # warm: no repopulate, only key probes
    print("warm query: %.3fs" % (time.time() - t))
    print("OK")


asyncio.run(main())
