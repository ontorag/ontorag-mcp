"""Embedding-free retrieval smoke test (no vectors, no ollama)."""
import asyncio

from ontorag_mcp.store import Dataset, resolve_source


async def main():
    ds = await Dataset.from_source(resolve_source("/data"), retrieval="ontology")
    print("retrieval=%s | counts=%s" % (ds.retrieval, ds.info()["counts"]))

    for q in ["How does the Parma Magica grant magic resistance?",
              "How does House Tremere use certamen to settle disputes?",
              "ways to protect a stronghold from hostile sorcery"]:  # names no entity
        print("\nQ:", q)
        for h in ds.search(q, k=3):
            print("  %.3f %-26s %s" % (h["score"], h["doc"], h["entities"][:3]))

    a = ds.answer("How does House Tremere use certamen?", k=4, expand=2)
    print("\nanswer matched_entities:", a["matched_entities"][:6],
          "| facts:", [f["label"] for f in a["ontology_facts"][:5]],
          "| passages:", len(a["passages"]))
    print("OK")


asyncio.run(main())
