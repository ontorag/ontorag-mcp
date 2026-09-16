#!/usr/bin/env python3
"""
Generate an evaluation query set from an OntoRAG dataset — deterministic, no LLM.

Three query kinds probe the retrieval-mode tradeoff:
  entity_named — "Tell me about <Entity>." (names the entity → favors `ontology`)
  paraphrase   — the entity's summary with its name/aliases MASKED out (no entity
                 mention → tests semantic recall; `ontology` can only use its BM25
                 fallback, `vector`/`hybrid` should shine)
  niah         — a book-unique entity (attestedIn == exactly one book); gold is any
                 chunk from that single attesting book (per-book findability, the
                 use the provenance layer was built for)

Gold for named/paraphrase = the chunks linked to the entity (chunk.entities). For
niah = the attesting book slug (== chunk.doc).
"""
import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict


def mask(text, terms):
    for t in sorted({t for t in terms if len(t) >= 3}, key=len, reverse=True):
        text = re.sub(r"\b" + re.escape(t) + r"\b", "it", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def stride(items, cap):
    if len(items) <= cap:
        return items
    step = len(items) / cap
    return [items[int(i * step)] for i in range(cap)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/srv/ofm/amol-ontorag")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "queries.jsonl"))
    ap.add_argument("--cap-named", type=int, default=120)
    ap.add_argument("--cap-para", type=int, default=120)
    ap.add_argument("--cap-niah", type=int, default=80)
    args = ap.parse_args()
    D = args.dataset

    ents = {}
    for line in open(os.path.join(D, "ontology/entities.jsonl"), encoding="utf-8"):
        e = json.loads(line)
        ents[e["iri"]] = e
    ent_chunks = defaultdict(list)
    for f in glob.glob(os.path.join(D, "content/chunks/*.jsonl")):
        for line in open(f, encoding="utf-8"):
            c = json.loads(line)
            for iri in c.get("entities", []):
                ent_chunks[iri].append(c["id"])

    rows, qid = [], 0

    named = [e for e in ents.values()
             if 2 <= len(ent_chunks.get(e["iri"], [])) <= 60
             and len(e["label"]) >= 4 and re.search(r"[A-Za-z]", e["label"])]
    named.sort(key=lambda e: e["iri"])
    for e in stride(named, args.cap_named):
        rows.append({"qid": f"named-{qid}", "kind": "entity_named",
                     "text": f"Tell me about {e['label']}.",
                     "gold_ids": ent_chunks[e["iri"]], "entity": e["label"]})
        qid += 1

    para = [e for e in ents.values()
            if len(e.get("summary", "")) >= 40 and len(ent_chunks.get(e["iri"], [])) >= 2]
    para.sort(key=lambda e: e["iri"])
    for e in stride(para, args.cap_para):
        q = mask(e["summary"], [e["label"]] + e.get("aliases", []))
        if len(q.split()) < 6:
            continue
        rows.append({"qid": f"para-{qid}", "kind": "paraphrase", "text": q,
                     "gold_ids": ent_chunks[e["iri"]], "entity": e["label"]})
        qid += 1

    by_book = defaultdict(list)
    for e in ents.values():
        if len(e.get("attestedIn", [])) == 1 and len(ent_chunks.get(e["iri"], [])) >= 1 \
                and len(e["label"]) >= 4 and re.search(r"[A-Za-z]", e["label"]):
            by_book[e["attestedIn"][0]].append(e)
    for lst in by_book.values():
        lst.sort(key=lambda e: e["iri"])
    # round-robin one book-unique entity per book per round -> balanced across books
    books, picked, i = sorted(by_book), [], 0
    while len(picked) < args.cap_niah and any(len(by_book[b]) > i for b in books):
        for b in books:
            if len(picked) >= args.cap_niah:
                break
            if i < len(by_book[b]):
                picked.append(by_book[b][i])
        i += 1
    for e in picked:
        rows.append({"qid": f"niah-{qid}", "kind": "niah",
                     "text": f"Tell me about {e['label']}.",
                     "gold_book": e["attestedIn"][0], "entity": e["label"]})
        qid += 1

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote %d queries %s -> %s" % (len(rows), dict(Counter(r["kind"] for r in rows)), args.out))


if __name__ == "__main__":
    main()
