# LangGraph Agent Platform

A local AI agent platform built with LangGraph, FastAPI, LiteLLM, Redis, Neo4j, Langfuse, and simple web frontends for research, network design, and FortiGate configuration-package generation.

This repository is a sanitized public version of a local lab stack. It includes source code, Docker Compose templates, configuration examples, and documentation. It intentionally excludes live secrets, run history, database volumes, and generated credentials.

## What It Does

The platform now has three primary workflows:

- **Research Assistant**: structured research with source retrieval, citations, quality checks, memory, and human review.
- **Network Design Helper**: chat-first network design assistant that retrieves uploaded standards, extracts requirements, builds a design package, creates a FortiGate handoff, and produces a compliance matrix.
- **FortiGate Provisioning Agent**: artifact-only FortiGate design/configuration workflow that generates draft CLI configuration, validation reports, standards checks, model judge feedback, and review-ready packages.

The research workflow runs a structured pipeline rather than a single chatbot response:

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
- Chat-first Network Design Helper UI with downloadable design packages, FortiGate handoff JSON, audit files, raw run JSON, and FortiGate `.conf` exports.
- Artifact-only FortiGate provisioning agent for draft designs, config packages, standards checks, frontier judge review, and human review.
- Redis-first hybrid standards retrieval: full-text keyword search, vector search, Reciprocal Rank Fusion, and JSON fallback.
- Standards ingestion for Markdown/text/config files, HTML, PDFs, Word docs, PowerPoint decks, and spreadsheets.
- Standards requirement extraction and compliance matrix generation.
- Polished Mermaid graph viewers with pan/zoom and SVG/Mermaid downloads.
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
admin-portal/         Static landing page with links to service UIs
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

3. Start the separate stacks:

```bash
cd litellm-linux
docker compose up -d

cd ../langfuse-platform
docker compose up -d

cd ../agent-platform
docker compose up -d --build

cd ../admin-portal
docker compose up -d
```

4. Open the admin portal:

```text
http://agent-host.example
```

For a single local development stack, you can still start just the agent platform:

```bash
cd agent-platform
docker compose up -d --build
```

Then open:

```text
http://localhost:8080
```

Additional local web tools:

```text
http://localhost:8080/network-design
http://localhost:8080/fortigate
```

Graph viewers:

```text
http://localhost:8001/graph
http://localhost:8001/research/graph
http://localhost:8001/network-design/graph
http://localhost:8001/fortigate/graph
```

The Linux deployment templates run LiteLLM and the LangGraph API with host networking so LiteLLM logs see the real LAN host address instead of Docker bridge addresses such as `172.x.x.x`. For your own environment, update `.env`, Compose files, and frontend API base URLs as needed.

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
