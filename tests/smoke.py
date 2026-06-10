"""Smoke test the Mirage-backed loader: read the dataset from GitHub via Mirage's
GitHub resource (no clone), then run retrieval. Needs GITHUB_TOKEN + OLLAMA_URL.

  REPO=openfantasymap/amol-ontorag python tests/smoke.py
"""
import asyncio
import os
import time

from ontorag_mcp.store import Dataset, resolve_source

REPO = os.environ.get("REPO", "openfantasymap/amol-ontorag")
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")


async def main():
    print("resolving source for %r ..." % REPO)
    src = resolve_source(REPO, token=os.environ.get("GITHUB_TOKEN"))
    print("  source:", src.describe())

    t0 = time.time()
    ds = await Dataset.from_source(src, ollama_url=OLLAMA)
    print("loaded via Mirage in %.1fs:" % (time.time() - t0), ds.info()["counts"])

    print("== search ==")
    for h in ds.search("How does the Parma Magica grant magic resistance?", k=3):
        print("  %.3f %-28s %s" % (h["score"], h["doc"], h["entities"][:3]))

    print("== answer ==")
    a = ds.answer("What is a heartbeast and which House has one?", k=4, expand=2)
    print("  facts:", [f["label"] for f in a["ontology_facts"][:6]], "| passages:", len(a["passages"]))

    print("== get_entity('Parma Magica') ==")
    e = ds.get_entity("Parma Magica")
    print("  ", {k: e[k] for k in ("label", "types", "linked_chunks")} if e else "NOT FOUND")

    print("== MCP tools ==")
    from ontorag_mcp.server import mcp
    print("  ", [t.name for t in await mcp.list_tools()])
    print("OK")


asyncio.run(main())
