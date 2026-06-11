# ontorag-mcp — a generic OntoRAG MCP server

A Model Context Protocol server that turns **any OntoRAG GitHub-as-storage
dataset** into a live RAG service, addressed by `<org>/<repo>`.

Point it at a dataset repo built in the [OntoRAG layout](https://github.com/openfantasymap/amol-ontorag)
(root `manifest.json` + `ontology/` + `content/chunks` + `embeddings/vectors`) and
it exposes graph-aware retrieval tools to any MCP client (Claude Code / Desktop, …).
It is **generic**: the same server serves any number of datasets, each identified
by its GitHub `<org>/<repo>` (or a local path).

## How it works

1. A tool call names a dataset as `<org>/<repo>`.
2. The server reads it **through [Mirage](https://github.com/strukto-ai/mirage)'s
   GitHub resource** — the repo is surfaced as a read-only virtual disk (no clone);
   `GITHUB_TOKEN` authenticates private repos.
3. It reads `manifest.json` and loads the ontology entities, chunks, and vectors
   from that virtual disk into memory.
4. Retrieval runs in one of two modes (`ONTORAG_RETRIEVAL`), both **graph-aware**
   (results carry the ontology entities each chunk mentions, and `answer` expands
   via shared entities + injects ontology facts):

   - **`vector`** (default) — embeds the query with the **same provider/model the
     dataset declares** (`embeddings/config.json`, e.g. local ollama
     `nomic-embed-text`) and does dense cosine top-k. Best semantic recall; needs
     an embedder reachable and loads the vectors (~140 MB / ~2 min for amol).
   - **`ontology`** — **embedding-free**: detect entities named in the query →
     pull their linked chunks → rank by entity specificity (idf) + a lexical
     BM25 fallback for queries that name no entity. **No embedder, no vectors
     loaded** (~24 s load for amol via Mirage), fully deterministic & explainable.
     Trade-off: weaker on pure paraphrase that names neither an entity nor a
     keyword.
   - **`hybrid`** (Redis only) — **sparse candidates, dense re-rank**: ontology +
     lexical pick a bounded candidate set, then **fetch only those chunks' vectors
     from Redis and cosine re-rank** with the query embedding. Recovers dense
     quality (entity-bearing queries match in-memory `vector` scores) without
     loading the whole index, so it stays serverless. Needs a query embedder; pays
     one embed call per request.

> **Cold load:** the first access to a dataset streams its files through Mirage
> (~2 min for the ~140 MB amol-ontorag dataset), then everything is in memory and
> queries are fast. A local path (e.g. a Mirage FUSE mount) loads instantly.

## Tools

| tool | purpose |
|------|---------|
| `load_dataset(repo, ref?, refresh?)` | clone/cache a dataset by `<org>/<repo>` (or path); returns its manifest summary |
| `list_datasets()` | datasets currently loaded + the configured default |
| `search(query, repo?, k?)` | semantic top-k chunks (text, heading path, score, linked entities) |
| `answer(query, repo?, k?, expand?)` | **graph-aware RAG bundle**: ontology facts + cited passages to compose an answer from |
| `search_entities(query, repo?, limit?)` | find ontology entities by name/alias/summary, ranked by chunk references |
| `get_entity(name_or_iri, repo?)` | one entity: type, tags, aliases, description, #linked chunks |
| `entity_chunks(name_or_iri, repo?, k?)` | chunks explicitly linked to an entity (graph-grounded, no vector search) |

`repo` defaults to `ONTORAG_DEFAULT_REPO`, so clients can omit it.

## Quick start — HTTP transport (default)

The server runs as a persistent **streamable-http** MCP server (the default), so
any number of clients can connect over the network and the dataset is loaded once.

```bash
docker compose build
docker compose up -d server      # MCP at http://localhost:8765/mcp
```

Connect a client:
```bash
# Claude Code
claude mcp add --transport http ontorag http://localhost:8765/mcp
```
```json
// Claude Desktop / any MCP client config
{ "mcpServers": { "ontorag": { "url": "http://localhost:8765/mcp" } } }
```

The endpoint is `/mcp`; `sse` is also available via `ONTORAG_TRANSPORT=sse`.

### Alternative: stdio (client spawns the process)

```bash
claude mcp add ontorag -- docker run --rm -i \
  -e ONTORAG_TRANSPORT=stdio \
  --add-host host.docker.internal:host-gateway \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  -e ONTORAG_DEFAULT_REPO=openfantasymap/amol-ontorag \
  -e GITHUB_TOKEN=$GITHUB_TOKEN ontorag-mcp:latest
```

## Environment

| var | default | meaning |
|-----|---------|---------|
| `ONTORAG_DEFAULT_REPO` | – | `<org>/<repo>` (or local path) so tools can omit `repo` |
| `ONTORAG_REF` | `main` | git ref for GitHub-backed datasets |
| `ONTORAG_RETRIEVAL` | `vector` | `vector` (dense, in-memory) \| `ontology` (embedding-free entity-graph + BM25) \| `hybrid` (Redis: sparse candidates + dense re-rank) \| `auto` |
| `REDIS_URL` | – | if set (with `ontology` or `hybrid` mode), materialize the index into Redis (24 h TTL) and query from there — no per-instance load (serverless) |
| `ONTORAG_REDIS_TTL` | `86400` | Redis index TTL in seconds |
| `OLLAMA_URL` | `http://localhost:11434` | embedding server (only used in `vector` mode) |
| `GITHUB_TOKEN` | – | token for Mirage's GitHub resource (private dataset repos) |
| `ONTORAG_TRANSPORT` | `streamable-http` | `streamable-http` \| `sse` \| `stdio` |
| `ONTORAG_HOST` / `ONTORAG_PORT` | `0.0.0.0` / `8765` | bind for HTTP transports |

## Requirements

- **Docker** (modern Python + `mirage-ai` baked into the image).
- Datasets are read via **Mirage's GitHub resource**, so a `GITHUB_TOKEN` with
  read access is needed for private dataset repos (public repos need none).
- In **`ontology` mode there is no embedder requirement** — no ollama, no API, no
  vectors loaded. Ideal when you want a light, portable, dependency-free deployment.
- In **`vector` mode**, datasets built with the `ollama` provider need an **ollama
  server** at `OLLAMA_URL` with the model pulled (e.g. `nomic-embed-text`);
  `hashed`-provider datasets need nothing extra.

## Example session

```
load_dataset("openfantasymap/amol-ontorag")
answer("What is the Parma Magica and how does it grant magic resistance?")
  → ontology_facts: [Parma Magica, Magic Resistance, House Bonisagus …]
    passages: [definitive-edition-core-rules::… , houses-of-hermes-true-lineages::… ]
get_entity("Parma Magica")  → Proficiency, 321 linked chunks
entity_chunks("House Tremere")  → everything the corpus links to House Tremere
```

## Serverless / stateless (Redis-backed ontology mode)

Set `REDIS_URL` together with `ONTORAG_RETRIEVAL=ontology` and the server stops
loading datasets into process memory. Instead it **materializes the ontology +
lexical index into Redis on a cache miss, with a 24 h TTL**, and every request
probes only the keys it needs:

```
query n-gram aliases → matched entities → their chunk-id lists → just those chunks
```

- **No per-instance load** — instances share the index via Redis; a request that
  finds the index present runs in **~25 ms**.
- **Dynamic, self-refreshing** — the `ready` marker expires ~10 min before the data
  keys, so the first query after 24 h repopulates from source (also picking up any
  dataset update). Populating costs one ~20 s request.
- **`ontology`**: no embedder, no vectors — pure entity-graph + BM25.
- **`hybrid`**: also stores per-chunk vectors in Redis (24 h TTL) and, per request,
  fetches only the candidate chunks' vectors and cosine re-ranks against the query
  embedding — dense quality, still bounded per request. Requires a query embedder
  (`OLLAMA_URL` or a hosted one); the embedding model is recorded in the dataset's
  `embeddings/config.json` and stored alongside the index.

This is what makes it deployable on serverless platforms (Vercel, Cloud Run,
Lambda, …): pair it with a hosted Redis such as **Upstash**. Example:

```bash
REDIS_URL=rediss://…upstash.io:6379 \
ONTORAG_RETRIEVAL=ontology \
ONTORAG_DEFAULT_REPO=openfantasymap/amol-ontorag \
GITHUB_TOKEN=$(gh auth token) docker compose up server
```

Locally, `docker compose --profile redis up` starts a Redis alongside the server.

## Deploy to Vercel (serverless)

Vercel loads the stateless ASGI app declared in `pyproject.toml`
(`[tool.vercel] entrypoint = "ontorag_mcp.asgi:app"`); the app serves the MCP
endpoint at `/mcp`. Use the **Redis-backed** retrieval (`ontology` recommended —
no embedder; or `hybrid` with a hosted embedder), with a hosted Redis like Upstash.

Files: `pyproject.toml` (entrypoint, `requires-python >=3.12`, deps),
`ontorag_mcp/asgi.py` (`app = build_asgi()`, sets `ONTORAG_STATELESS=1`),
`.vercelignore`. No `/api` directory and no `functions`/`rewrites` config are
needed — Vercel builds the entrypoint and serves all routes through it.

```bash
vercel link        # or import the repo in the Vercel dashboard
vercel deploy --prod
```

Set these Environment Variables in the Vercel project:

| var | value |
|-----|-------|
| `REDIS_URL` | `rediss://…upstash.io:6379` (hosted Redis) |
| `ONTORAG_RETRIEVAL` | `ontology` (no embedder) — or `hybrid` |
| `ONTORAG_DEFAULT_REPO` | e.g. `openfantasymap/amol-ontorag` |
| `ONTORAG_REF` | `main` |
| `GITHUB_TOKEN` | token for Mirage to read the dataset repo |
| `OLLAMA_URL` | hosted embedder URL — **only** for `hybrid` |

The MCP endpoint is `https://<deployment>/mcp`; connect with
`claude mcp add --transport http ontorag https://<deployment>/mcp`.
Default function duration on Fluid is 300 s (enough for the cold populate); set
`maxDuration` via a `vercel.json` `functions` glob (`api/**/*.py` style targeting
the entrypoint) only if you need to raise it.

**Caveats (serverless):**
- *Cold populate.* The first request after the 24 h TTL repopulates Redis from the
  dataset via Mirage — ~25 s for `ontology`, longer for `hybrid` (reads the
  vectors). It fits within `maxDuration` 300 s, but add a **Vercel Cron** that pings
  the endpoint to pre-warm Redis so user requests never pay it (and to avoid GitHub
  API rate-limits during the paged read).
- *`hybrid` needs a hosted embedder* — `ollama` isn't on Vercel; either point
  `OLLAMA_URL` at a hosted ollama, build the dataset with the `hashed` provider, or
  use `ontology` mode (embedder-free).
- *Bundle size.* `mirage-ai` + `numpy` are sizable but within Vercel's limit;
  `.vercelignore` drops `tests/`, `Dockerfile`, and compose.

## Notes

- A loaded dataset is held in memory (~55 MB of vectors for 18k × 768-dim) and
  reused across requests. Prefer the **persistent HTTP server** so the ~2 min cold
  load happens once; `stdio` respawns (and reloads) per client session.
- Files are read through Mirage's shell, whose stdout caps at ~2000 lines, so the
  loader pages large files with `tail`/`head` windows and reassembles them — this
  is transparent, but it means big files are fetched in a few passes on cold load.
- Point `ONTORAG_DEFAULT_REPO` at a **local path** (e.g. a Mirage FUSE mount of the
  repo) to skip the network entirely and load instantly.
