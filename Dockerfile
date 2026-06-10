FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY ontorag_mcp /app/ontorag_mcp

# HTTP (streamable-http) transport by default; MCP at http://<host>:8765/mcp
ENV ONTORAG_TRANSPORT=streamable-http ONTORAG_HOST=0.0.0.0 ONTORAG_PORT=8765
EXPOSE 8765
ENTRYPOINT ["python", "-m", "ontorag_mcp.server"]
