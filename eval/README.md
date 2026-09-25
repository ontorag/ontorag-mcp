# eval — retrieval quality harness

Measures **ontology vs vector vs hybrid** retrieval on an OntoRAG dataset, with no
LLM required. Two scripts:

- `build_queryset.py` — deterministically derives a query set from the dataset
  (entities + chunks + provenance). Three query kinds probe the mode tradeoff:
  - **entity_named** — `"Tell me about <Entity>."` (names the entity → favors `ontology`)
  - **paraphrase** — the entity's summary with its name/aliases **masked** (no entity
    mention → meant to test semantic recall)
  - **niah** — a *book-unique* entity (`attestedIn` == one book); gold = any chunk
    from that single attesting book (per-book findability; balanced round-robin
    across books). Uses the v0.4.2 provenance layer.
  - Gold for named/paraphrase = the chunks linked to the entity.
- `run_eval.py` — runs each mode over the query set, reports **recall@1/5/10** and
  **MRR** per kind. Query embeddings are batched once and reused across modes via
  the store's `qvec` param.

```bash
python3 eval/build_queryset.py --dataset /srv/ofm/amol-ontorag
# in Docker (needs numpy + a query embedder for vector/hybrid):
docker run --rm -v $PWD:/app -w /app -v /srv/ofm/amol-ontorag:/srv/ofm/amol-ontorag:ro \
  --add-host host.docker.internal:host-gateway -e OLLAMA_URL=http://host.docker.internal:11434 \
  -e PYTHONPATH=/app --entrypoint python ontorag-mcp:latest eval/run_eval.py
```

## Independent paraphrase set (the honest paraphrase test)

The `paraphrase` queries below are built from each entity's own stored summary with
the name masked, so they share wording with the data — and an index of those
summaries would match them almost verbatim. `queries.indep.jsonl` fixes that:
**200 questions**, 8 per book across all 25 books, about entities specific to one
book, written by separate Claude agents who saw only the entity's name, type and
source passages — never the stored description (`indep/`: inputs, outputs and the
answer key). A code check rejects any question that uses the entity's name or an
alias; **`indep_strict`** (148) further drops questions sharing any 4-word run with
the source passages or the description.

| set | mode | recall@1 | recall@5 | recall@10 | MRR |
|-----|------|---------:|---------:|----------:|----:|
| indep (200) | ontology | **0.77** | 0.85 | 0.86 | **0.80** |
| | entity | 0.69 | 0.78 | 0.84 | 0.73 |
| | hybrid | 0.54 | 0.73 | 0.76 | 0.61 |
| | vector | 0.51 | 0.69 | 0.75 | 0.58 |
| | ontology + entity (RRF) | 0.70 | **0.89** | **0.93** | 0.78 |
| indep_strict (148) | ontology | **0.72** | 0.82 | 0.84 | **0.77** |
| | entity | 0.66 | 0.76 | 0.82 | 0.70 |
| | hybrid | 0.51 | 0.69 | 0.72 | 0.58 |
| | vector | 0.50 | 0.67 | 0.73 | 0.57 |
| | ontology + entity (RRF) | 0.66 | **0.87** | **0.91** | 0.75 |

(`results.indep.json`, `results.fusion.json`; fusion by `fuse_eval.py`,
reciprocal-rank fusion of the two modes' top 50.)

- **`entity` mode** (question → nearest entity descriptions → their chunks) beats
  `vector` by 0.16–0.18 recall@1: comparing a question with short entity
  descriptions works better than comparing it with rulebook prose.
- **`ontology` still wins at rank 1.** Real questions carry other names ("Bologna",
  "Rego Terram", "Pyrenean") that the entity matcher and BM25 exploit.
- **Fusion is best for retrieval-augmented answers:** recall@5 0.89 / @10 0.93
  (strict 0.87 / 0.91) — the correct passage is in the context handed to the model
  far more often than with any single mode, at a small cost at rank 1.
- On the **old, leaky `paraphrase` set**, `entity` scores 0.975 recall@1 — exactly
  the inflation expected when the index embeds the texts the questions were made
  from. Don't use that set to judge description-based retrieval.

## Results — amol-ontorag v0.6.0 (n: 120 named / 120 paraphrase / 80 niah)

Since v0.6.0, entities are extracted from **all 25 books** (before: 8), and the query
set covers every book: the needle (niah) queries target all 25 books (3–4 each),
where the v0.4.2 set could only reach 17.

| kind | mode | recall@1 | recall@10 | MRR |
|------|------|---------:|----------:|----:|
| entity_named | ontology | **0.99** | 1.00 | 1.00 |
| | hybrid | 0.82 | 0.92 | 0.87 |
| | vector | 0.42 | 0.62 | 0.49 |
| niah | ontology | **0.97** | 1.00 | 0.98 |
| | hybrid | 0.75 | 0.94 | 0.80 |
| | vector | 0.28 | 0.62 | 0.37 |
| paraphrase | ontology | **0.65** | 0.85 | 0.72 |
| | hybrid | 0.49 | 0.81 | 0.59 |
| | vector | 0.48 | 0.78 | 0.58 |

(`results.json`, from `queries.jsonl`.) With `entity` mode added on the same set:
entity_named 0.94, niah 0.94 recall@1 (below `ontology` but far above `vector`),
and paraphrase 0.975 — inflated, see above.

### Same questions, old vs new dataset

The two query sets differ, so the tables are not comparable row by row. Running the
**v0.4.2 question set** against both datasets isolates the effect of the new
entities (`results.amol-v0.4.2.json` → `results.oldqueries-on-v0.6.0.json`):

| kind | mode | recall@1 | recall@10 | MRR |
|------|------|---------:|----------:|----:|
| entity_named | ontology | 1.00 → 0.99 | 1.00 → 1.00 | 1.00 → 1.00 |
| | hybrid | 0.93 → **0.86** | 0.95 → 0.93 | 0.94 → 0.89 |
| | vector | 0.39 → 0.39 | 0.60 → 0.60 | 0.46 → 0.46 |
| niah | ontology | 1.00 → 0.99 | 1.00 → 1.00 | 1.00 → 0.99 |
| | hybrid | 0.86 → **0.76** | 0.94 → 0.90 | 0.88 → 0.80 |
| | vector | 0.34 → 0.34 | 0.62 → 0.62 | 0.42 → 0.42 |
| paraphrase | ontology | 0.55 → 0.55 | 0.73 → 0.77 | 0.61 → 0.62 |
| | hybrid | 0.39 → 0.38 | 0.65 → 0.65 | 0.49 → 0.49 |
| | vector | 0.42 → 0.42 | 0.67 → 0.67 | 0.51 → 0.51 |

Vector is unchanged (same embeddings) and ontology holds. **Hybrid regresses:** with
three times as many entities, a query matches more of them, so the candidate set
hybrid re-ranks is larger and noisier. Its candidate selection needs tightening
(e.g. weight matches by entity specificity) before hybrid is the default.

### Previous results — amol-ontorag v0.4.2 (8 books extracted) (n: 120 named / 120 paraphrase / 80 niah)

| kind | mode | recall@1 | recall@10 | MRR |
|------|------|---------:|----------:|----:|
| entity_named | ontology | **1.00** | 1.00 | 1.00 |
| | hybrid | 0.93 | 0.95 | 0.94 |
| | vector | 0.39 | 0.60 | 0.46 |
| niah | ontology | **1.00** | 1.00 | 1.00 |
| | hybrid | 0.86 | 0.94 | 0.88 |
| | vector | 0.34 | 0.63 | 0.42 |
| paraphrase | ontology | **0.55** | 0.73 | 0.61 |
| | vector | 0.42 | 0.67 | 0.51 |
| | hybrid | 0.39 | 0.65 | 0.49 |

## Reading it (honestly)

- **On entity- and book-anchored queries, ontology ≈ hybrid ≫ vector.** For a
  corpus this entity-dense, naming the entity and walking the graph beats dense
  cosine by a wide margin (vector recall@1 ≈ 0.34–0.39). `hybrid` recovers almost
  all of ontology's win while adding dense re-ranking.
- **`ontology` scores ~1.00 on named/niah *nearly by construction*** — the query names
  the entity, and gold is that entity's chunks, so the entity-matcher rarely misses
  (0.99/0.97 on v0.6.0, where more entities share names and aliases). Treat
  those as an upper reference, not a fair fight; the informative number there is how
  close `vector`/`hybrid` get.
- **The paraphrase result is confounded and inconclusive.** Summaries are
  *extractive* (drawn from the gold chunks), so they share surface vocabulary with
  the gold → BM25 (inside `ontology`) stays competitive and vectors show no clear
  semantic edge. A clean semantic test needs **LLM-generated paraphrases / novel
  questions** (a natural next step — plug a `questions.jsonl` into `run_eval.py`).
- **`hybrid` ≤ `vector` on paraphrase** confirms the known cap: hybrid only
  re-ranks a *candidate set* from entity/lexical matching, so a truly entity-less
  query whose gold isn't in the candidates can't be recovered — full dense can.

## Caveats / next
- No human labels; golds are derived from the ontology links, so the harness
  rewards the graph it was built from. Add an LLM QA set for an independent signal.
- Provenance-scoped eval (retrieve within `attestedIn ⊆ book-subset S`) is the
  obvious next axis once composition/scoping lands.
