"""ASGI entrypoint for serverless hosts (Vercel).

Vercel loads `ontorag_mcp.asgi:app` (see pyproject.toml [tool.vercel].entrypoint).
The app is the stateless streamable-http MCP server; it serves the MCP endpoint
at `/mcp`, so the deployed URL is https://<deployment>/mcp.

Configure via Environment Variables in the Vercel project: REDIS_URL,
ONTORAG_RETRIEVAL (ontology|hybrid), ONTORAG_DEFAULT_REPO, ONTORAG_REF,
GITHUB_TOKEN, and OLLAMA_URL (only for hybrid).
"""
import os

os.environ.setdefault("ONTORAG_STATELESS", "1")

from .server import build_asgi  # noqa: E402

app = build_asgi()
