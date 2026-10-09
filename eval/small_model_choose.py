#!/usr/bin/env python3
"""Arm E of the small-model benchmark: the model chooses, code proposes.

small_model_eval.py's arm C hands a small model the top-k passages and asks it to
name the thing; when the right passage *was* retrieved, small models still answer
wrong about 60% of the time — rambling, a wrong name, or "unknown". Here the
candidates come from the graph instead: every entity linked to the retrieved
passages, each with its one-line summary. The model only picks one, and
grammar-constrained decoding (an enum in ollama's structured output) makes it
impossible to answer anything else. Whenever retrieval found a gold passage, the
gold entity is among the candidates, since gold passages are its linked chunks.

    python eval/small_model_choose.py --models hf.co/prism-ml/Bonsai-1.7B-gguf:Q1_0 \
        --out eval/small_models_choose.jsonl
"""
import argparse, asyncio, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ontorag_mcp.store import Dataset, resolve_source  # noqa: E402
from run_eval import batch_embed  # noqa: E402
from small_model_eval import chat, names_entity  # noqa: E402

NONE = "none of these"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/srv/ofm/amol-ontorag")
    ap.add_argument("--queries", default=os.path.join(os.path.dirname(__file__), "queries.indep.jsonl"))
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-candidates", type=int, default=40)
    ap.add_argument("--rank", choices=["passage", "similarity"], default="passage",
                    help="order candidates by passage order, or by how close their description "
                         "is to the question (entity vectors) before cutting to --max-candidates")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "small_models_choose.jsonl"))
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.queries) if l.strip()]
    strict = {q["qid"][:-2] for q in rows if q["kind"] == "indep_strict"}
    queries = [q for q in rows if q["kind"] == "indep"]
    ds = await Dataset.from_source(resolve_source(args.dataset), ollama_url=args.ollama_url,
                                   retrieval="fused")
    qvec = batch_embed({q["qid"]: q["text"] for q in queries}, ds.model, args.ollama_url)

    done = set()
    if os.path.exists(args.out):
        done = {(r["model"], r["qid"]) for r in map(json.loads, open(args.out))}

    for model in args.models:
        todo = [q for q in queries if (model, q["qid"]) not in done]
        print(f"{model}: {len(todo)} questions to run", flush=True)
        for q in todo:
            t0 = time.time()
            gold = ds.entities[q["entity"]]
            hits = ds.search(q["text"], k=args.k, qvec=qvec[q["qid"]])
            # candidates: entities linked to the retrieved passages, in passage order,
            # most specific (fewest linked chunks) first within a passage
            cand, seen = [], set()
            for h in hits:
                ents = sorted(ds.chunks[h["id"]].get("entities", []),
                              key=lambda i: len(ds._ent_chunks.get(i, [])))
                for iri in ents:
                    e = ds.entities.get(iri)
                    if e and iri not in seen and e.get("label"):
                        seen.add(iri)
                        cand.append(e)
            if args.rank == "similarity" and ds.has_entity_vectors:
                row = {iri: r for r, iri in enumerate(ds.ent_ids)}
                qv = qvec[q["qid"]]
                cand.sort(key=lambda e: -(float(ds.ent_mat[row[e["iri"]]] @ qv)
                                          if e["iri"] in row else -1.0))
            cand = cand[:args.max_candidates]
            labels = []
            for e in cand:                      # enum values must be unique strings
                if e["label"] not in labels:
                    labels.append(e["label"])
            listing = "\n".join(f"- {e['label']}: {(e.get('summary') or '')[:110]}" for e in cand)
            schema = {"type": "object", "required": ["answer"],
                      "properties": {"answer": {"type": "string", "enum": labels + [NONE]}}}
            try:
                out = json.loads(chat(args.ollama_url, model, [
                    {"role": "system", "content": "Pick the one candidate that the question is asking "
                     "about. If none fits, pick \"" + NONE + "\"."},
                    {"role": "user", "content": f"Candidates:\n{listing}\n\nQuestion: {q['text']}"}],
                    fmt=schema, num_predict=60))
                answer = out.get("answer", "")
            except Exception as ex:
                answer = "error: " + str(ex)[:80]
            rec = {"model": model, "qid": q["qid"], "strict": q["qid"] in strict,
                   "entity": gold["label"], "n_candidates": len(labels),
                   "gold_in_candidates": any(e["iri"] == q["entity"] for e in cand),
                   "E_answer": answer,
                   "E_correct": names_entity(answer, gold["label"], gold.get("aliases", [])),
                   "seconds": round(time.time() - t0, 1)}
            with open(args.out, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    recs = [json.loads(l) for l in open(args.out)]
    print(f"\n{'model':45s} {'set':6s} {'n':>4s} {'gold in cand':>12s} {'E choose':>8s} {'s/q':>5s}")
    by = {}
    for r in recs:
        by.setdefault((r["model"], "all"), []).append(r)
        if r["strict"]:
            by.setdefault((r["model"], "strict"), []).append(r)
    for (m, s), rs in sorted(by.items()):
        f = lambda k: sum(bool(r[k]) for r in rs) / len(rs)
        print(f"{m.split('/')[-1]:45s} {s:6s} {len(rs):4d} {f('gold_in_candidates'):12.2f} "
              f"{f('E_correct'):8.2f} {sum(r['seconds'] for r in rs) / len(rs):5.0f}")


if __name__ == "__main__":
    asyncio.run(main())
