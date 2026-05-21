# Agent Platform

Local AI agent platform running on `localhost` with LangGraph, LiteLLM, Redis, Langfuse, Neo4j, and a web Research Assistant.

For the full architecture and operations guide, see [`PLATFORM_REFERENCE.md`](PLATFORM_REFERENCE.md).

## Services

| Service | URL | Purpose |
| --- | --- | --- |
| Research Assistant | `http://localhost:8080` | Main web UI |
| Agent API | `http://localhost:8001` | FastAPI + LangGraph |
| Neo4j Browser | `http://localhost:7474` | Research memory graph |
| Langfuse | `http://localhost:3001` | Traces and observability |
| LiteLLM | `http://your-vllm-host.example:4010/v1` | Model gateway |
| vLLM | `http://your-vllm-host.example:8000/v1` | Dedicated inference server |

## Run

```bash
cd ~/dev/agent-platform
docker-compose up -d --build
```

Check status:

```bash
docker-compose ps
curl http://localhost:8001/health
curl http://localhost:8001/memory/health
```

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
GET  /research/graph/mermaid
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
- Neo4j stores queryable research memory.
- Langfuse stores traces and observability data.

## Operational Notes

- Docker Desktop on macOS can block image pulls over SSH due to keychain access. Pull from an interactive `omlx` session if needed.
- Neo4j writes are best-effort; research runs still save to JSON if Neo4j is temporarily unavailable.
- OPA is not wired yet. Current policy checks are Python checks inside the research graph.
- Do not commit or paste active API keys from `.env`.
