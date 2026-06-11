"""Vercel serverless entrypoint — exposes the OntoRAG MCP server as a stateless
ASGI app. Vercel routes `/mcp` here (see vercel.json); we normalize the path so
the MCP handler always matches, and pass lifespan events through so the
streamable-http session manager starts.

Configure in the Vercel project (Environment Variables):
  REDIS_URL              hosted Redis (e.g. Upstash rediss://…)   [required for serverless]
  ONTORAG_RETRIEVAL      ontology  (recommended: no embedder)  |  hybrid (needs an embedder)
  ONTORAG_DEFAULT_REPO   <org>/<repo> of the dataset (e.g. openfantasymap/amol-ontorag)
  ONTORAG_REF            dataset git ref (default main)
  GITHUB_TOKEN           token for Mirage to read the dataset repo (private repos)
  OLLAMA_URL             hosted embedder (only for `hybrid`)
"""
import os
import sys

# stateless MCP + import the package from the repo root
os.environ.setdefault("ONTORAG_STATELESS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ontorag_mcp.server import build_asgi  # noqa: E402

_mcp_app = build_asgi()


async def app(scope, receive, send):
    # Force the MCP endpoint path regardless of how Vercel rewrites the URL;
    # forward lifespan/other events untouched so the session manager starts.
    if scope.get("type") == "http":
        scope = dict(scope)
        scope["path"] = "/mcp"
        scope["raw_path"] = b"/mcp"
    await _mcp_app(scope, receive, send)
