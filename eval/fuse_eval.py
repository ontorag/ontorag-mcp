#!/usr/bin/env python3
"""Reciprocal-rank fusion of two retrieval modes, scored like run_eval.py.

    python eval/fuse_eval.py --queries eval/queries.indep.jsonl --modes ontology entity

Each mode returns its top-N chunks; a chunk's fused score is sum 1/(K + rank).
"""
import argparse, asyncio, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ontorag_mcp.store import Dataset, resolve_source  # noqa: E402
from run_eval import batch_embed, rank_of_gold  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/srv/ofm/amol-ontorag")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--modes", nargs=2, default=["ontology", "entity"])
    ap.add_argument("--depth", type=int, default=50)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    args = ap.parse_args()

    queries = [json.loads(l) for l in open(args.queries) if l.strip()]
    for q in queries:
        q["_gold"] = set(q.get("gold_ids", []))
    src = resolve_source(args.dataset)
    dss = {m: await Dataset.from_source(src, ollama_url=args.ollama_url, retrieval=m) for m in args.modes}
    emb = next(ds for ds in dss.values() if ds.model)
    qvecs = batch_embed({q["qid"]: q["text"] for q in queries}, emb.model, args.ollama_url)

    agg = {}
    for q in queries:
        fused = {}
        hits_by_id = {}
        for m, ds in dss.items():
            for r, h in enumerate(ds.search(q["text"], k=args.depth, qvec=qvecs[q["qid"]]), 1):
                fused[h["id"]] = fused.get(h["id"], 0.0) + 1.0 / (args.rrf_k + r)
                hits_by_id[h["id"]] = h
        ranked = [hits_by_id[c] for c, _ in sorted(fused.items(), key=lambda kv: -kv[1])[:args.k]]
        rank = rank_of_gold(ranked, q)
        a = agg.setdefault(q["kind"], {"n": 0, "r1": 0, "r5": 0, "r10": 0, "mrr": 0.0})
        a["n"] += 1
        a["r1"] += rank == 1
        a["r5"] += 0 < rank <= 5
        a["r10"] += 0 < rank <= 10
        a["mrr"] += 1.0 / rank if rank else 0.0
    out = {kind: {"n": a["n"], "recall@1": round(a["r1"] / a["n"], 3), "recall@5": round(a["r5"] / a["n"], 3),
                  "recall@10": round(a["r10"] / a["n"], 3), "MRR": round(a["mrr"] / a["n"], 3)}
           for kind, a in agg.items()}
    print(json.dumps({"+".join(args.modes): out}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
