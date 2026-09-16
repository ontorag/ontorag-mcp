#!/usr/bin/env python3
"""
Score each retrieval mode (ontology / vector / hybrid) on the eval query set and
report recall@k and MRR per query kind. Query embeddings are batched once (the
dataset's declared model) and reused across modes via the store's `qvec` param.
"""
import argparse
import asyncio
import json
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ontorag_mcp.store import Dataset, resolve_source  # noqa: E402


def batch_embed(qid_text, model, url, batch=32):
    out, items = {}, list(qid_text.items())
    for i in range(0, len(items), batch):
        part = items[i:i + batch]
        payload = json.dumps({"model": model, "input": [t for _, t in part]}).encode()
        req = urllib.request.Request(url.rstrip("/") + "/api/embed", data=payload,
                                     headers={"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(req, timeout=120).read())
        for (qid, _), v in zip(part, d["embeddings"]):
            a = np.asarray(v, dtype=np.float32)
            n = float(np.linalg.norm(a))
            out[qid] = a / n if n else a
    return out


def rank_of_gold(hits, q):
    for i, h in enumerate(hits, 1):
        if q["kind"] == "niah":
            if h["doc"] == q["gold_book"]:
                return i
        elif h["id"] in q["_gold"]:
            return i
    return 0


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/srv/ofm/amol-ontorag")
    ap.add_argument("--queries", default=os.path.join(os.path.dirname(__file__), "queries.jsonl"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--modes", nargs="*", default=["ontology", "vector", "hybrid"])
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results.json"))
    args = ap.parse_args()

    queries = [json.loads(l) for l in open(args.queries, encoding="utf-8")]
    for q in queries:
        q["_gold"] = set(q.get("gold_ids", []))
    kinds = sorted({q["kind"] for q in queries})

    emb = json.load(open(os.path.join(args.dataset, "embeddings/config.json")))
    qvec = {}
    if any(m in args.modes for m in ("vector", "hybrid")) and emb["provider"] == "ollama":
        print("batch-embedding %d queries via %s ..." % (len(queries), emb["model"]), file=sys.stderr)
        qvec = batch_embed({q["qid"]: q["text"] for q in queries}, emb["model"], args.ollama_url)

    results = {}
    for mode in args.modes:
        print("loading dataset (mode=%s) ..." % mode, file=sys.stderr)
        ds = await Dataset.from_source(resolve_source(args.dataset),
                                       ollama_url=args.ollama_url, retrieval=mode)
        agg = {kind: {"n": 0, "r1": 0, "r5": 0, "r10": 0, "mrr": 0.0} for kind in kinds}
        for q in queries:
            hits = ds.search(q["text"], k=args.k,
                             qvec=(qvec.get(q["qid"]) if mode != "ontology" else None))
            r = rank_of_gold(hits, q)
            a = agg[q["kind"]]
            a["n"] += 1
            if r:
                a["mrr"] += 1.0 / r
                a["r1"] += r <= 1
                a["r5"] += r <= 5
                a["r10"] += r <= 10
        results[mode] = {kind: {
            "n": a["n"],
            "recall@1": round(a["r1"] / max(1, a["n"]), 3),
            "recall@5": round(a["r5"] / max(1, a["n"]), 3),
            "recall@10": round(a["r10"] / max(1, a["n"]), 3),
            "MRR": round(a["mrr"] / max(1, a["n"]), 3),
        } for kind, a in agg.items()}

    metrics = ["recall@1", "recall@5", "recall@10", "MRR"]
    print("\n=== Retrieval eval — recall@k / MRR (k=%d) ===" % args.k)
    for kind in kinds:
        n = results[args.modes[0]][kind]["n"]
        print("\n[%s]  n=%d" % (kind, n))
        print("  %-10s " % "mode" + " ".join("%10s" % m for m in metrics))
        for mode in args.modes:
            a = results[mode][kind]
            print("  %-10s " % mode + " ".join("%10s" % a[m] for m in metrics))
    json.dump(results, open(args.out, "w"), indent=2)
    print("\nwrote", args.out)


asyncio.run(main())
