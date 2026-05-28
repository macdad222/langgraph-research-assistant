# Agent Platform

Local AI agent platform running on a Linux agent host with LangGraph, LiteLLM, Redis, Langfuse, Neo4j, a web Research Assistant, a Network Design Helper, and an artifact-only FortiGate provisioning agent.

For the full architecture and operations guide, see [`PLATFORM_REFERENCE.md`](PLATFORM_REFERENCE.md).

## Services

| Service | URL | Purpose |
| --- | --- | --- |
| Admin Portal | `http://agent-host.example` | Landing page for service UIs |
| Research Assistant | `http://agent-host.example:8080` | Main web UI |
| Network Design Helper | `http://agent-host.example:8080/network-design` | Chat-first design assistant with standards, handoff, compliance matrix, and exports |
| FortiGate Agent | `http://agent-host.example:8080/fortigate` | Draft FortiGate design/config package generation |
| Agent API | `http://agent-host.example:8001` | FastAPI + LangGraph |
| Neo4j Browser | `http://agent-host.example:7474` | Research memory graph |
| Langfuse | `http://agent-host.example:3001` | Traces and observability |
| LiteLLM | `http://agent-host.example:4010/v1` | Model gateway |
| vLLM | `http://vllm-host.example:8000/v1` | Dedicated inference server |

## Run

```bash
cd ~/dev/agent-platform
docker compose up -d --build
```

Check status:

```bash
docker compose ps
curl http://localhost:8001/health
curl http://localhost:8001/memory/health
```

## Linux Host Networking

The Linux template runs `langgraph-app` with `network_mode: host`.

That is intentional: it lets LiteLLM see LangGraph requests from the real agent host/LAN source address instead of a Docker bridge IP. Because of this, the app uses host-local service URLs such as `redis://127.0.0.1:6379/0` and `bolt://127.0.0.1:7687`.

## What The Research Assistant Does

The current research graph runs a full structured pipeline:

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

Key features:

- Tavily search with DuckDuckGo fallback.
- Source fetching and source quality scoring.
- Structured citations and citation verification.
- Critic/quality report.
- Deterministic policy report.
- Optional deepening and bounded policy repair.
- Saved research run history.
- Markdown export.
- Human review and follow-up pass controls.
- True LangGraph interrupt/resume workflow for interactive review.
- Neo4j-backed research memory retrieval before web search.

## What The Network Design Helper Does

The Network Design Helper is a chat-first workflow for network design engineers:

```text
intake_conversation
  -> retrieve_standards
  -> curate_standards
  -> summarize_requirements
  -> identify_gaps
  -> build_design_package
  -> build_fortigate_handoff
  -> check_compliance
  -> validate_design
  -> finalize_package
```

Key features:

- Chat-style design conversation.
- Sidebar for customer/site/design context.
- Standards search against uploaded company/Fortinet documents.
- Redis-first hybrid retrieval using keyword search, vector search, and Reciprocal Rank Fusion.
- Structured standard requirement extraction.
- Compliance matrix mapping standards to design and FortiGate handoff evidence.
- Downloadable design Markdown, raw run JSON, detailed audit Markdown, and FortiGate handoff JSON.
- Optional call into the FortiGate agent to generate a downloadable draft `.conf`.

## What The FortiGate Agent Does

The FortiGate agent is artifact-only. It can:

- Parse existing FortiGate configs.
- Retrieve standards evidence.
- Build logical and FortiGate-specific designs.
- Generate draft CLI artifacts.
- Run deterministic validation and standards checks.
- Run a Qwen-configured model judge before human review.
- Regenerate draft config artifacts when the judge finds more than three review items.
- Save packages under `/data/fortigate-runs`.

No live device changes are made.

## Human In The Loop

There are two human review paths:

1. Saved-run review in the UI:
   - Approve final answer.
   - Mark needs work.
   - Select issues.
   - Run follow-up from review.

2. True LangGraph pause/resume:
   - Use **Run Interactive Review** in the UI.
   - The graph pauses at `human_review_checkpoint`.
   - The user approves or requests one more repair pass.
   - The graph resumes using `Command(resume=...)`.

## Neo4j Research Memory

Neo4j stores completed research as a graph:

```text
(:ResearchRun)-[:ANSWERED]->(:Question)
(:ResearchRun)-[:USED_SOURCE]->(:Source)
(:ResearchRun)-[:MADE_CLAIM]->(:Claim)
(:Claim)-[:CITED]->(:Source)
(:ResearchRun)-[:HAS_REVIEW]->(:HumanReview)
(:ResearchRun)-[:HAS_POLICY]->(:PolicyReport)
(:ResearchRun)-[:HAS_QUALITY]->(:QualityReport)
(:ResearchRun)-[:FOLLOWED_BY]->(:ResearchRun)
```

Neo4j is used for:

- Retrieval before web search.
- Prior run and source lookup.
- Research backlog.
- Human review audit trails.
- Cross-run source and claim deduplication.
- Follow-up chain tracking.

Neo4j local dev login:

```text
URL: http://localhost:7474
Username: neo4j
Password: change-me-neo4j-password
```

## API Quick Reference

Platform:

```text
GET  /health
GET  /graph
GET  /graph/mermaid
GET  /research/graph
GET  /research/graph/mermaid
GET  /network-design/graph
GET  /network-design/graph/mermaid
GET  /fortigate/graph
GET  /fortigate/graph/mermaid
```

Chat:

```text
POST /chat
```

Research:

```text
POST /research
GET  /research/runs
GET  /research/runs/{run_id}
POST /research/runs/{run_id}/review
POST /research/runs/{run_id}/follow-up
POST /research/interactive
POST /research/interactive/{thread_id}/review
```

Network Design:

```text
POST /network-design/message
POST /network-design/chat
GET  /network-design/runs
GET  /network-design/runs/{run_id}
GET  /network-design/standards/search?q=...
```

FortiGate:

```text
POST /fortigate/design
POST /fortigate/interactive
POST /fortigate/interactive/{thread_id}/resume
POST /fortigate/configs/parse
POST /fortigate/changes/analyze
POST /fortigate/runs/{run_id}/judge
POST /fortigate/runs/{run_id}/review
POST /fortigate/standards/ingest
GET  /fortigate/standards/search?q=...
GET  /fortigate/runs
GET  /fortigate/runs/{run_id}
```

Memory:

```text
GET /memory/health
GET /memory/research/search?q=...
GET /memory/research/runs/{run_id}
GET /memory/research/runs/{run_id}/audit
GET /memory/research/sources/{source_hash}/runs
GET /memory/research/backlog
GET /memory/research/dedup?q=...
```

## Example Calls

Health:

```bash
curl http://localhost:8001/health
```

Research:

```bash
curl -sS http://localhost:8001/research \
  -H 'Content-Type: application/json' \
  -d '{"thread_id":"demo","mode":"quick","question":"What is LangGraph interrupt/resume useful for?","constraints":"Prefer official docs."}'
```

Memory search:

```bash
curl -sS 'http://localhost:8001/memory/research/search?q=human&limit=5'
```

Backlog:

```bash
curl -sS 'http://localhost:8001/memory/research/backlog?limit=10'
```

## Persistence

- Redis stores LangGraph checkpoints.
- JSON files in `/data/research-runs` remain the fallback archive.
- JSON files in `/data/network-design-runs` store Network Design Helper runs.
- JSON files in `/data/fortigate-runs` store FortiGate packages.
- `/data/fortigate-standards/index.json` stores the standards search index.
- Neo4j stores queryable research memory.
- Langfuse stores traces and observability data.

## Standards Library

Place source documents under `/data/fortigate-standards/raw`, then rebuild the index with:

```bash
curl -sS http://localhost:8001/fortigate/standards/ingest \
  -H 'Content-Type: application/json' \
  -d '{"source_dir":"/data/fortigate-standards/raw"}'
```

Supported input types include Markdown/text/config files, HTML, PDF, Word, PowerPoint, and Excel. The ingester skips Mac metadata files such as `.DS_Store`, `._*`, and `__MACOSX`.

Ingestion writes the JSON fallback index and also attempts to build a Redis Search index with full-text and vector fields. Existing API routes still call `search_standards(...)`, which now uses Redis hybrid retrieval when available and falls back to JSON keyword search when Redis or embeddings are unavailable.

## Operational Notes

- The Linux template uses host networking for LangGraph so LiteLLM can log the real agent host/LAN source address.
- Neo4j writes are best-effort; research runs still save to JSON if Neo4j is temporarily unavailable.
- OPA is not wired yet. Current policy checks are Python checks inside the research graph.
- Do not commit or paste active API keys from `.env`.
