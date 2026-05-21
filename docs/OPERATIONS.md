# Operations Guide

## Agent Platform

```bash
cd agent-platform
docker-compose up -d --build
docker-compose ps
docker-compose logs -f langgraph-app
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

Neo4j Browser:

```text
http://localhost:7474
```

## Langfuse Template

```bash
cd langfuse-platform
cp .env.example .env
# edit .env
docker-compose up -d
```

## LiteLLM Linux Template

```bash
cd litellm-linux
cp .env.example .env
cp config.example.yaml config.yaml
# edit .env and config.yaml
docker-compose up -d
```

## Common Local Ports

- Agent API: `8001`
- Research UI: `8080`
- Redis: `6379`
- Neo4j Browser: `7474`
- Neo4j Bolt: `7687`
- Langfuse: `3001`
- LiteLLM: `4010`
- vLLM: `8000`
