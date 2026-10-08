#!/usr/bin/env python3
"""End-to-end RAG with small models: does retrieval do the work, or the model?

On the independent paraphrase set (questions that describe an entity without
naming it), per model:

  A  no model      question -> fused retrieval top-k            (retrieval only)
  B  slot filling  model fills {kind, keywords, name_guess} under a JSON schema
                   (ollama structured output = grammar-constrained decoding);
                   question + slots -> fused retrieval top-k
  C  RAG answer    model names the entity from the question + A's top-k passages
  D  closed book   model names the entity from the question alone (control)

Recall@k is over the gold chunks (as run_eval.py); answers are correct when they
name the entity (label or an alias, normalised, or a close fuzzy match).
Every record is appended to --out as JSONL, so a long CPU run can be resumed.

    python eval/small_model_eval.py --models hf.co/prism-ml/Bonsai-1.7B-gguf:Q1_0 \
        --queries eval/queries.indep.jsonl --out eval/small_models.jsonl
"""
import argparse, asyncio, difflib, json, os, re, sys, time, urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ontorag_mcp.store import Dataset, resolve_source  # noqa: E402
from run_eval import batch_embed  # noqa: E402  (retries a busy ollama)


def chat(url, model, messages, fmt=None, num_predict=128):
    body = {"model": model, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": 0, "num_predict": num_predict}}
    if fmt:
        body["format"] = fmt
    req = urllib.request.Request(url + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(5):        # a busy or restarting ollama should not end a long run
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                text = json.loads(r.read())["message"]["content"]
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(60 * (attempt + 1))
    # some reasoning models leak their think tags even with thinking off
    text = re.sub(r"(?s)<think>.*?</think>", "", text)
    return text.split("</think>")[-1].strip()


def norm(s):
    s = re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())
    s = re.sub(r"^\s*the\s+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def names_entity(answer, label, aliases):
    a = norm(answer)
    if not a:
        return False
    for n in [label] + list(aliases):
        n = norm(n)
        # the name in the answer, or an answer that is most of the name ("magic" is
        # not an answer for "Magic Resistance")
        if len(n) >= 4 and (n in a or (len(a) >= 0.6 * len(n) and a in n)):
            return True
        if difflib.SequenceMatcher(None, a, n).ratio() >= 0.85:
            return True
    return False


def recall_at(hits, gold, k):
    return any(h["id"] in gold for h in hits[:k])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/srv/ofm/amol-ontorag")
    ap.add_argument("--queries", default=os.path.join(os.path.dirname(__file__), "queries.indep.jsonl"))
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--passage-chars", type=int, default=1200)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "small_models.jsonl"))
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.queries) if l.strip()]
    strict = {q["qid"][:-2] for q in rows if q["kind"] == "indep_strict"}
    queries = [q for q in rows if q["kind"] == "indep"]
    ds = await Dataset.from_source(resolve_source(args.dataset), ollama_url=args.ollama_url,
                                   retrieval="fused")
    ents = ds.entities
    kinds = [k for k, _ in Counter((e.get("tags") or [e["types"][0].rsplit("/", 1)[-1]])[0]
                                   for e in ents.values() if e.get("types")).most_common(24)] + ["Other"]
    slot_schema = {"type": "object", "required": ["kind", "keywords", "name_guess"],
                   "properties": {"kind": {"type": "string", "enum": kinds},
                                  "keywords": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
                                  "name_guess": {"type": "string"}}}

    qvec = batch_embed({q["qid"]: q["text"] for q in queries}, ds.model, args.ollama_url)
    base = {q["qid"]: ds.search(q["text"], k=args.k, qvec=qvec[q["qid"]]) for q in queries}

    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            r = json.loads(l)
            done.add((r["model"], r["qid"]))

    for model in args.models:
        todo = [q for q in queries if (model, q["qid"]) not in done]
        print(f"{model}: {len(todo)} questions to run", flush=True)
        for n, q in enumerate(todo, 1):
            e = ents[q["entity"]]
            gold = set(q["gold_ids"])
            t0 = time.time()
            rec = {"model": model, "qid": q["qid"], "strict": q["qid"] in strict,
                   "entity": e["label"], "question": q["text"],
                   "A_recall": recall_at(base[q["qid"]], gold, args.k)}

            # B: slot filling under a JSON schema, then retrieval
            try:
                slots = json.loads(chat(args.ollama_url, model, [
                    {"role": "system", "content": "You help search an Ars Magica (tabletop RPG) rules "
                     "database. Read the question and fill the fields. kind: what sort of thing is asked "
                     "for. keywords: up to 6 distinctive search words or short phrases. name_guess: the "
                     "name of the thing if you know it, else an empty string."},
                    {"role": "user", "content": q["text"]}], fmt=slot_schema, num_predict=160))
            except Exception as ex:                      # a model that cannot keep to the schema
                slots = {"kind": "Other", "keywords": [], "name_guess": "", "error": str(ex)[:120]}
            expanded = " ".join([q["text"], " ".join(slots.get("keywords") or []),
                                 slots.get("name_guess") or ""]).strip()
            hits_b = ds.search(expanded, k=args.k, qvec=batch_embed({"x": expanded}, ds.model,
                                                                    args.ollama_url)["x"])
            rec.update({"slots": slots, "B_recall": recall_at(hits_b, gold, args.k),
                        "B_guess_correct": names_entity(slots.get("name_guess", ""), e["label"],
                                                        e.get("aliases", []))})

            # C: answer from A's passages
            ctx = "\n\n".join(f"[{i}] {h['text'][:args.passage_chars]}" for i, h in
                              enumerate(base[q["qid"]], 1))
            ans_c = chat(args.ollama_url, model, [
                {"role": "system", "content": "Answer using only the passages. Reply with the name of "
                 "the thing asked about and nothing else. If the passages do not say, reply: unknown."},
                {"role": "user", "content": f"Passages:\n{ctx}\n\nQuestion: {q['text']}"}], num_predict=40)
            # D: closed book
            ans_d = chat(args.ollama_url, model, [
                {"role": "system", "content": "Reply with the name of the thing asked about and nothing "
                 "else. If you do not know, reply: unknown."},
                {"role": "user", "content": q["text"]}], num_predict=40)
            rec.update({"C_answer": ans_c, "C_correct": names_entity(ans_c, e["label"], e.get("aliases", [])),
                        "D_answer": ans_d, "D_correct": names_entity(ans_d, e["label"], e.get("aliases", [])),
                        "seconds": round(time.time() - t0, 1)})
            with open(args.out, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n % 10 == 0:
                print(f"  {n}/{len(todo)} ({rec['seconds']}s/question)", flush=True)

    summarize(args.out)


def summarize(path):
    recs = [json.loads(l) for l in open(path)]
    by = {}
    for r in recs:
        for subset in ("all", "strict"):
            if subset == "strict" and not r["strict"]:
                continue
            by.setdefault((r["model"], subset), []).append(r)
    print(f"\n{'model':50s} {'set':6s} {'n':>4s} {'A R@5':>6s} {'B R@5':>6s} {'B guess':>7s} "
          f"{'C RAG':>6s} {'D closed':>8s} {'s/q':>5s}")
    for (m, subset), rs in sorted(by.items()):
        f = lambda k: sum(bool(r[k]) for r in rs) / len(rs)
        print(f"{m.split('/')[-1]:50s} {subset:6s} {len(rs):4d} {f('A_recall'):6.2f} {f('B_recall'):6.2f} "
              f"{f('B_guess_correct'):7.2f} {f('C_correct'):6.2f} {f('D_correct'):8.2f} "
              f"{sum(r['seconds'] for r in rs) / len(rs):5.0f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        summarize(sys.argv[2])
    else:
        asyncio.run(main())
