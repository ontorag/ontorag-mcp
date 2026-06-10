"""
OntoRAG MCP server — a generic RAG service over any OntoRAG GitHub-as-storage
dataset, addressed by `<org>/<repo>`.

Datasets are read through **Mirage's GitHub resource** (mirage-ai): the repo is
surfaced as a read-only virtual disk and the dataset files are read from it — no
clone. Query embeddings use the same provider/model the dataset declares, so
retrieval is consistent with how the dataset was built.

Transport: HTTP (streamable-http) by default — MCP at http://<host>:<port>/mcp.

Env:
  ONTORAG_DEFAULT_REPO   default "<org>/<repo>" (or local path) so tools can omit `repo`
  ONTORAG_REF            git ref for GitHub-backed datasets (default "main")
  OLLAMA_URL             embedding server (default http://localhost:11434)
  GITHUB_TOKEN           token for Mirage's GitHub resource (private repos)
  ONTORAG_TRANSPORT      "streamable-http" (default) | "sse" | "stdio"
  ONTORAG_HOST / PORT    bind for http transports (default 0.0.0.0:8765)
"""
import asyncio
import os

from mcp.server.fastmcp import FastMCP

from .store import Dataset, resolve_source

DEFAULT_REPO = os.environ.get("ONTORAG_DEFAULT_REPO", "").strip()
DEFAULT_REF = os.environ.get("ONTORAG_REF") or None
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or None
# "vector" (dense, needs the dataset's embeddings + embedder), "ontology"
# (embedding-free: entity-graph + lexical/BM25; no vectors loaded), or "auto".
RETRIEVAL = os.environ.get("ONTORAG_RETRIEVAL", "vector").strip()
# When set, ontology-mode datasets are materialized into Redis (24 h TTL) and
# queried from there — no per-instance load (serverless-friendly).
REDIS_URL = os.environ.get("REDIS_URL") or None
REDIS_TTL = int(os.environ.get("ONTORAG_REDIS_TTL", str(24 * 3600)))

mcp = FastMCP("ontorag")
_loaded = {}              # spec -> Dataset | RedisDataset
_lock = asyncio.Lock()
_redis = None


def _use_redis():
    return bool(REDIS_URL) and RETRIEVAL in ("ontology", "hybrid")


async def _get(repo):
    spec = (repo or DEFAULT_REPO).strip()
    if not spec:
        raise ValueError("No dataset specified. Pass repo='<org>/<repo>' "
                         "or set ONTORAG_DEFAULT_REPO.")
    if spec not in _loaded:
        async with _lock:
            if spec not in _loaded:
                src = resolve_source(spec, ref=DEFAULT_REF, token=GITHUB_TOKEN)
                if _use_redis():
                    global _redis
                    from . import redis_store
                    if _redis is None:
                        _redis = redis_store.client(REDIS_URL)
                    # cheap: no data load — populated lazily in Redis with 24 h TTL
                    _loaded[spec] = redis_store.RedisDataset(
                        _redis, spec, src, ttl=REDIS_TTL, mode=RETRIEVAL, ollama_url=OLLAMA_URL)
                else:
                    _loaded[spec] = await Dataset.from_source(
                        src, ollama_url=OLLAMA_URL, retrieval=RETRIEVAL)
    return _loaded[spec]


async def _result(x):
    return await x if asyncio.iscoroutine(x) else x


@mcp.tool()
async def list_datasets() -> dict:
    """List datasets currently loaded into this server, plus the configured default."""
    return {"default": DEFAULT_REPO or None, "backend": "redis" if _use_redis() else "memory",
            "loaded": {spec: await _result(ds.info()) for spec, ds in _loaded.items()}}


@mcp.tool()
async def load_dataset(repo: str, ref: str = "", retrieval: str = "") -> dict:
    """Load an OntoRAG dataset by '<org>/<repo>' (read via Mirage's GitHub resource)
    or a local path, and return its manifest summary. `ref` selects a branch/tag.
    `retrieval` overrides the mode: 'vector' (dense embeddings), 'ontology'
    (embedding-free entity-graph + lexical), or 'auto'."""
    spec = repo.strip()
    src = resolve_source(spec, ref=ref or DEFAULT_REF, token=GITHUB_TOKEN)
    mode = retrieval or RETRIEVAL
    if REDIS_URL and mode in ("ontology", "hybrid"):
        global _redis
        from . import redis_store
        if _redis is None:
            _redis = redis_store.client(REDIS_URL)
        _loaded[spec] = redis_store.RedisDataset(
            _redis, spec, src, ttl=REDIS_TTL, mode=mode, ollama_url=OLLAMA_URL)
    else:
        _loaded[spec] = await Dataset.from_source(src, ollama_url=OLLAMA_URL, retrieval=mode)
    return await _result(_loaded[spec].info())


@mcp.tool()
async def search(query: str, repo: str = "", k: int = 6) -> list:
    """Semantic search: top-k most relevant chunks (text, heading path, similarity
    score, and the ontology entities each chunk mentions)."""
    return await _result((await _get(repo)).search(query, k=k))


@mcp.tool()
async def answer(query: str, repo: str = "", k: int = 6, expand: int = 3) -> dict:
    """Graph-aware RAG retrieval. Returns a grounded, cited context bundle
    (ontology_facts + passages) for you to compose the final answer from — cite
    passages by their `cite` id. `expand` pulls in sibling chunks sharing the same
    ontology entities as the top hits."""
    return await _result((await _get(repo)).answer(query, k=k, expand=expand))


@mcp.tool()
async def search_entities(query: str, repo: str = "", limit: int = 20) -> list:
    """Find ontology entities (Characters, Spells, Houses, Covenants, Creatures,
    concepts, …) whose name/alias/summary matches `query`, ranked by chunk references."""
    return await _result((await _get(repo)).search_entities(query, limit=limit))


@mcp.tool()
async def get_entity(name_or_iri: str, repo: str = "") -> dict:
    """Look up one ontology entity by IRI or name/alias: its type, tags, aliases,
    description, and how many chunks reference it."""
    e = await _result((await _get(repo)).get_entity(name_or_iri))
    return e or {"error": "entity not found: %s" % name_or_iri}


@mcp.tool()
async def entity_chunks(name_or_iri: str, repo: str = "", k: int = 8) -> list:
    """Graph-grounded retrieval: chunks explicitly linked to a given ontology
    entity (no vector search) — read everything the corpus says about a specific
    Character, Spell, House, etc."""
    return await _result((await _get(repo)).entity_chunks(name_or_iri, k=k))


def main():
    transport = os.environ.get("ONTORAG_TRANSPORT", "streamable-http")
    if transport in ("streamable-http", "sse"):
        mcp.settings.host = os.environ.get("ONTORAG_HOST", "0.0.0.0")
        mcp.settings.port = int(os.environ.get("ONTORAG_PORT", "8765"))
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
