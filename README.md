# LangGraph Research Assistant

A local AI research assistant platform built with LangGraph, FastAPI, LiteLLM, Redis, Neo4j, Langfuse, and a simple web frontend.

This repository is a sanitized public version of a local lab stack. It includes source code, Docker Compose templates, configuration examples, and documentation. It intentionally excludes live secrets, run history, database volumes, and generated credentials.

## What It Does

The platform runs a structured research workflow rather than a single chatbot response:

```text
retrieve_memory
  -> plan_research
  -> search_sources
  -> read_sources
  -> score_sources
  -> synthesize_research
  -> analyze_gaps
  -> optional deepening
  -> critique_research
  -> revise_final_answer
  -> check_policy
  -> optional policy repair
  -> verify_citations
```

Key capabilities:

- Web-based Research Assistant UI.
- LangGraph state machine for research workflows.
- LiteLLM model gateway to an OpenAI-compatible model server.
- Redis checkpointing for graph state and interrupt/resume.
- Langfuse tracing.
- Neo4j research memory for prior runs, sources, claims, reviews, and follow-up chains.
- Human-in-the-loop review controls.
- True LangGraph interrupt/resume for interactive review.
- Backlog, audit, dedup, source lookup, and memory search endpoints.

## Repository Layout

```text
agent-platform/       FastAPI + LangGraph app and frontend
langfuse-platform/    Sanitized Langfuse self-hosted Compose template
litellm-linux/        Sanitized LiteLLM + Postgres template for a vLLM server
docs/                 Security, operations, and sanitization notes
README.txt            Long-form beginner-friendly explanation
PLATFORM_REFERENCE.md Detailed architecture and operations reference
```

## Quick Start

1. Copy example env files:

```bash
cp agent-platform/.env.example agent-platform/.env
cp langfuse-platform/.env.example langfuse-platform/.env
cp litellm-linux/.env.example litellm-linux/.env
cp litellm-linux/config.example.yaml litellm-linux/config.yaml
```

2. Replace every placeholder secret and host value.

3. Start the agent platform:

```bash
cd agent-platform
docker-compose up -d --build
```

4. Open the UI:

```text
http://localhost:8080
```

The original lab deployment used LAN hostnames like `localhost` and `your-vllm-host.example`. For your own environment, update `.env`, Compose files, and frontend API base URLs as needed.

## Documentation

Start here if you are new to agents or context engineering:

- [`README.txt`](README.txt)

Detailed technical reference:

- [`PLATFORM_REFERENCE.md`](PLATFORM_REFERENCE.md)

Operational and safety docs:

- [`docs/OPERATIONS.md`](docs/OPERATIONS.md)
- [`docs/SECURITY.md`](docs/SECURITY.md)
- [`docs/SANITIZATION.md`](docs/SANITIZATION.md)

## Security Note

This repository is public-safe by design, but you must create your own `.env` files with your own secrets. Do not commit `.env`, generated credentials, run data, database volumes, or API keys.

## License

MIT License. See [`LICENSE`](LICENSE).
