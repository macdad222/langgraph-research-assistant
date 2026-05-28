# Operations Guide

## Recommended Linux Layout

The deployment is split into separate Compose stacks:

```text
admin-portal/       Static landing page on port 80
litellm-linux/      LiteLLM + Postgres
langfuse-platform/  Langfuse + Postgres + ClickHouse + Redis + MinIO
agent-platform/     LangGraph API + Redis + Neo4j + Research/Network Design/FortiGate UIs
```

LiteLLM and the LangGraph API use host networking in the Linux template. This keeps the LangGraph-to-LiteLLM request path on the host/LAN network stack, so LiteLLM access logs show the real host/LAN source address instead of Docker bridge addresses like `172.x.x.x`.

## Admin Portal

```bash
cd admin-portal
docker compose up -d
docker compose ps
```

Open:

```text
http://agent-host.example
```

## Agent Platform

```bash
cd agent-platform
docker compose up -d --build
docker compose ps
docker compose logs -f langgraph-app
```

Health checks:

```bash
curl http://localhost:8001/health
curl http://localhost:8001/memory/health
```

Research UI:

```text
http://localhost:8080
```

Network Design Helper:

```text
http://localhost:8080/network-design
```

FortiGate Agent:

```text
http://localhost:8080/fortigate
```

Graph viewers:

```text
http://localhost:8001/graph
http://localhost:8001/research/graph
http://localhost:8001/network-design/graph
http://localhost:8001/fortigate/graph
```

Neo4j Browser:

```text
http://localhost:7474
```

## Langfuse Template

```bash
cd langfuse-platform
cp .env.example .env
# edit .env
docker compose up -d
```

## LiteLLM Linux Template

```bash
cd litellm-linux
cp .env.example .env
cp config.example.yaml config.yaml
# edit .env and config.yaml
docker compose up -d
```

The template publishes LiteLLM Postgres only on host loopback:

```text
127.0.0.1:5433 -> litellm-db:5432
```

LiteLLM itself listens directly on host port `4010` using `network_mode: host`.

## Common Local Ports

- Agent API: `8001`
- Research UI: `8080`
- Network Design Helper: `8080/network-design`
- FortiGate Agent: `8080/fortigate`
- Redis: `6379`
- Neo4j Browser: `7474`
- Neo4j Bolt: `7687`
- Langfuse: `3001`
- LiteLLM: `4010`
- vLLM: `8000`

## Common Checks

```bash
curl http://localhost:8001/health
curl http://localhost:8001/memory/health
curl http://localhost:3001/api/public/health
curl http://localhost:4010/health/liveliness
```

Standards index:

```bash
curl -sS http://localhost:8001/fortigate/standards/ingest \
  -H 'Content-Type: application/json' \
  -d '{"source_dir":"/data/fortigate-standards/raw"}'
curl -sS 'http://localhost:8001/network-design/standards/search?q=sd-wan&limit=5'
```

The standards ingester writes `/data/fortigate-standards/index.json` as a fallback and builds a Redis Search index named `idx:standards` when Redis hybrid retrieval is enabled. Search uses Redis full-text results plus Redis vector results, fused with Reciprocal Rank Fusion. Optional cross-encoder reranking is controlled by `STANDARDS_RERANK_ENABLED`. If Redis or embeddings are unavailable, the app falls back to JSON keyword search.

Saved run archives:

```text
/data/research-runs
/data/network-design-runs
/data/fortigate-runs
/data/fortigate-standards/index.json
```

Check LiteLLM source IP logging:

```bash
cd litellm-linux
docker compose logs --tail=100 litellm-linux-test | grep "POST /v1/chat/completions"
```
