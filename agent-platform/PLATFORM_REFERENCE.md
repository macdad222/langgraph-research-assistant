# AI Agent Platform Reference

Last updated: 2026-05-30

This document captures the current Linux-oriented agent platform deployment using sanitized example hostnames.

## What Exists Now

The platform is a local agent stack for structured research and experimentation with LangGraph-based workflows. It includes:

- A FastAPI + LangGraph application container.
- LiteLLM model access to a dedicated vLLM server.
- Redis-backed LangGraph checkpoints.
- Langfuse tracing.
- A password-protected web frontend with a same-origin `/api` proxy to the internal LangGraph API.
- A web-based Research Assistant frontend, available at `/research`.
- A web-based Network Design Helper for chat-first design conversations, standards retrieval, compliance matrix generation, and FortiGate handoff payloads.
- An artifact-only FortiGate provisioning agent for draft designs, validation, model judge review, and CLI configuration artifacts.
- Redis-first hybrid standards retrieval for Markdown/text/config files, HTML, PDFs, Word docs, PowerPoint decks, and spreadsheets.
- Hybrid retrieval combines Redis full-text keyword search, Redis vector search, Reciprocal Rank Fusion, and JSON fallback.
- Downloadable design packages, FortiGate `.conf` files, detailed audit Markdown files, raw run JSON, SVG graphs, and Mermaid source.
- A static admin portal with links to all service UIs.
- File-backed research run persistence in `/data/research-runs`.
- File-backed Network Design Helper run persistence in `/data/network-design-runs`.
- File-backed FortiGate run persistence in `/data/fortigate-runs`.
- Neo4j research memory for prior runs, sources, claims, reviews, policy reports, and follow-up chains.
- Human-in-the-loop review controls, including true LangGraph `interrupt()` / `Command(resume=...)` flow.

## High-Level Architecture

```text
Browser / Cloudflare Tunnel / API Client
  |
  | public HTTPS or http://agent-host.example:8080
  v
Password-Protected Frontend
  |
  | static pages: /network-design, /fortigate, /research
  v
Research / Network Design / FortiGate Frontends
  |
  | same-origin /api proxy, internal target API_PROXY_TARGET
  v
FastAPI + LangGraph App
  |-- Redis checkpoints
  |-- JSON run archive: /data/research-runs
  |-- JSON run archive: /data/network-design-runs
  |-- JSON run archive: /data/fortigate-runs
  |-- Standards index: /data/fortigate-standards/index.json
  |-- Neo4j research memory
  |-- Langfuse tracing callback
  |
  | OpenAI-compatible API over host/LAN network stack
  v
LiteLLM Proxy on agent-host.example:4010
  |
  | hosted_vllm / OpenAI-compatible API
  v
vLLM Inference Server on vllm-host.example:8000
```

## Hosts And URLs

| Service | URL | Notes |
| --- | --- | --- |
| Admin portal | `http://agent-host.example` | Landing page for service links |
| Network Design Helper UI | `http://agent-host.example:8080/network-design` | Default password-protected frontend |
| FortiGate Agent UI | `http://agent-host.example:8080/fortigate` | Artifact-only FortiGate provisioning/change planning |
| Research Assistant UI | `http://agent-host.example:8080/research` | Research UI, not linked from the FortiGate design frontend |
| Frontend API proxy | `http://agent-host.example:8080/api/*` | Same-origin proxy to internal FastAPI |
| Agent API health | `http://agent-host.example:8001/health` | Internal FastAPI health check |
| Agent API docs | `http://agent-host.example:8001/docs` | FastAPI Swagger docs |
| Chat graph visualizer | `http://agent-host.example:8001/graph` | Browser-rendered Mermaid graph |
| Research graph visualizer | `http://agent-host.example:8001/research/graph` | Rendered graph viewer with zoom/export |
| Network Design graph visualizer | `http://agent-host.example:8001/network-design/graph` | Rendered graph viewer with zoom/export |
| FortiGate graph visualizer | `http://agent-host.example:8001/fortigate/graph` | Rendered graph viewer with zoom/export |
| Chat graph Mermaid | `http://agent-host.example:8001/graph/mermaid` | Raw Mermaid graph text |
| Research graph Mermaid | `http://agent-host.example:8001/research/graph/mermaid` | Raw research graph text |
| Network Design graph Mermaid | `http://agent-host.example:8001/network-design/graph/mermaid` | Raw Network Design Helper graph text |
| FortiGate graph Mermaid | `http://agent-host.example:8001/fortigate/graph/mermaid` | Raw FortiGate graph text |
| Neo4j Browser | `http://agent-host.example:7474` | Research memory graph UI |
| Neo4j Bolt | `bolt://agent-host.example:7687` | Driver connection from host/LAN |
| Langfuse UI | `http://agent-host.example:3001` | Observability and traces |
| LiteLLM API | `http://agent-host.example:4010/v1` | Current LangGraph model gateway |
| LiteLLM UI | `http://agent-host.example:4010/ui` | LiteLLM management UI |
| vLLM API | `http://vllm-host.example:8000/v1` | Dedicated inference server |

## Project Locations

| Path | Purpose |
| --- | --- |
| `~/dev/agent-platform` | LangGraph app, Dockerfile, Compose file, docs, frontend |
| `~/dev/agent-platform/frontend` | Password-protected frontend and `/api` proxy |
| `~/dev/agent-platform/data/research-runs` | JSON archive of saved research runs |
| `~/dev/agent-platform/data/network-design-runs` | JSON archive of saved Network Design Helper runs |
| `~/dev/agent-platform/data/fortigate-runs` | JSON archive of saved FortiGate package runs |
| `~/dev/agent-platform/data/fortigate-standards` | Uploaded standards source files and generated standards index |
| `~/dev/admin-portal` | Static landing page with links to admin UIs |
| `~/dev/langfuse-platform` | Langfuse self-hosted Compose stack |
| `~/dev/litellm-platform` | LiteLLM + Postgres stack on the agent host |
| `~/agent-platform` | Helper script symlink, if present |
| `~/langfuse-platform` | Langfuse helper script symlink, if present |
| `~/langfuse-credentials.txt` | Local Langfuse credentials, mode `600` |

## Containers

Core agent stack in `~/dev/agent-platform`:

| Container | Purpose | Ports |
| --- | --- | --- |
| `langgraph-app` | FastAPI + LangGraph agent service | `8001` |
| `langgraph-redis` | Redis Stack for LangGraph checkpoints | `6379` |
| `research-neo4j` | Neo4j research memory graph | `7474`, `7687` |
| `research-frontend` | Password-protected frontend and same-origin `/api` proxy | `8080` |

Admin portal stack in `~/dev/admin-portal`:

| Container | Purpose | Port |
| --- | --- | --- |
| `admin-portal` | Static link page for service UIs | `80` |

Langfuse stack in `~/dev/langfuse-platform`:

| Container | Purpose | Port |
| --- | --- | --- |
| `langfuse-platform-langfuse-web-1` | Langfuse web UI/API | `3001` -> container `3000` |
| `langfuse-platform-langfuse-worker-1` | Langfuse background worker | internal |
| `langfuse-platform-postgres-1` | Langfuse transactional DB | internal |
| `langfuse-platform-clickhouse-1` | Langfuse analytics/traces DB | internal |
| `langfuse-platform-minio-1` | Object storage for Langfuse | `9090` |
| `langfuse-platform-redis-1` | Queue/cache for Langfuse | internal |

Linux LiteLLM stack:

| Container | Purpose | Port |
| --- | --- | --- |
| `litellm-linux-test` | LiteLLM proxy + UI, using host networking | `4010` |
| `litellm-db` | Postgres database for LiteLLM | `127.0.0.1:5433` |

## Environment Variables

The agent app reads `~/dev/agent-platform/.env`.

Important variables:

```text
LITELLM_BASE_URL=http://agent-host.example:4010/v1
LITELLM_API_KEY=<LiteLLM key>
FRONTEND_PASSWORD=<frontend password>
API_PROXY_TARGET=http://agent-host.example:8001
MODEL_NAME=gemma-local
NETWORK_CHAT_MODEL_NAME=gemma-local
NETWORK_INTAKE_MODEL_NAME=gemma-local
NETWORK_PACKAGE_MODEL_NAME=gemma-local
NETWORK_HANDOFF_MODEL_NAME=gemma-local
NETWORK_CHAT_TEMPERATURE=0.2
NETWORK_CHAT_ENABLE_THINKING=false
NETWORK_INTAKE_TEMPERATURE=0.0
NETWORK_PACKAGE_TEMPERATURE=0.2
NETWORK_HANDOFF_TEMPERATURE=0.0
FORTIGATE_JUDGE_MODEL_NAME=Qwen3.6-27B
FORTIGATE_BUILDER_REVIEW_MODE=pre_refine
FORTIGATE_BUILDER_REVIEW_MODEL_NAME=gemma-local
FORTIGATE_BUILDER_REVIEW_TEMPERATURE=1.0
FORTIGATE_CONFIG_REFINEMENT_MODE=pre_judge
FORTIGATE_CONFIG_REFINER_MODEL_NAME=Qwen3.6-27B
FORTIGATE_CONTEXT_FALLBACK_MODEL_NAME=extl-gemma-4-31b
FORTIGATE_CONTEXT_FALLBACK_MAX_OUTPUT_TOKENS=32768
FORTIGATE_CONTEXT_FALLBACK_TEMPERATURE=0.2
FORTIGATE_AUTONOMOUS_REPAIR_LIMIT=1
FORTIGATE_SECTIONAL_GENERATION_ENABLED=false
LANGFUSE_PUBLIC_KEY=<Langfuse public key>
LANGFUSE_SECRET_KEY=<Langfuse secret key>
LANGFUSE_BASE_URL=http://agent-host.example:3001
TAVILY_API_KEY=<optional Tavily key>
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=change-me-neo4j-password
FORTIGATE_STANDARDS_DIR=/data/fortigate-standards/raw
FORTIGATE_STANDARDS_INDEX=/data/fortigate-standards/index.json
FORTIGATE_RUNS_DIR=/data/fortigate-runs
NETWORK_DESIGN_RUNS_DIR=/data/network-design-runs
STANDARDS_RETRIEVAL_BACKEND=redis_hybrid
STANDARDS_REDIS_URL=redis://127.0.0.1:6379/0
STANDARDS_REDIS_INDEX=idx:standards
STANDARDS_REDIS_PREFIX=std:chunk:
STANDARDS_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
STANDARDS_VECTOR_DIM=384
STANDARDS_RERANK_ENABLED=false
STANDARDS_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

Do not paste active API keys into docs or chat. `.env.example` contains placeholders/default local development values only.

## Credentials And Login

Credentials should be generated locally and stored outside git. A practical layout is:

```text
~/dev/litellm-platform/ui-credentials.txt
~/dev/langfuse-platform/credentials.txt
~/dev/agent-platform/credentials.txt
```

Neo4j local development login:

```text
URL: http://agent-host.example:7474
Username: neo4j
Password: change-me-neo4j-password
```

Do not paste the LiteLLM master key, generated UI passwords, Langfuse keys, Tavily key, or Neo4j password into docs or logs.

### Frontend Password Gate

The web frontend uses a simple shared password gate implemented in `frontend/server.js`.

```text
Password source: FRONTEND_PASSWORD in .env
Cookie: fortigate_frontend_auth=1
```

Set `FRONTEND_PASSWORD` in `.env` before starting the frontend. This is a lightweight access gate for the design UI, not a replacement for Cloudflare Access, SSO, or per-user authorization.

### Public / Cloudflare Access Pattern

When publishing the UI through Cloudflare Tunnel or another reverse proxy, expose only the frontend service on `:8080`.

```text
Public browser
  -> Cloudflare hostname
  -> research-frontend:8080
  -> /api proxy inside frontend server
  -> API_PROXY_TARGET, usually http://agent.lab.internal:8001
```

Do not require browsers to access `:8001` directly. The UI files set `apiBase` to `/api`, which keeps backend API calls same-origin and avoids public backend exposure and CORS failures.

## Model Routing

The LangGraph app calls LiteLLM using:

```text
base_url: http://agent-host.example:4010/v1
model: gemma-local
```

LiteLLM routes `gemma-local` to the dedicated vLLM server:

```text
api_base: http://vllm-host.example:8000/v1
```

The vLLM server exposes a Gemma 31B model through an OpenAI-compatible API.

## Network Source IP Policy

On Linux, the deployment runs LiteLLM and the LangGraph API with `network_mode: host`. This is intentional.

Normal Docker bridge networking makes LiteLLM see requests from Docker bridge/gateway addresses such as `172.x.x.x`. Host networking keeps this path on the host network stack, so LiteLLM logs show the real host/LAN source address for LangGraph calls.

Support services remain containerized:

- LiteLLM Postgres is published only on `127.0.0.1:5433`.
- Redis remains published on `6379`.
- Neo4j remains published on `7474` and `7687`.
- Langfuse uses its own Compose network and published web port `3001`.
- The frontend does not use host networking; it reaches FastAPI through `API_PROXY_TARGET`.

## LangGraph Workflows

### Chat Graph

Simple checkpointed chat endpoint:

```text
POST /chat
```

Shape:

```text
messages -> model node -> response
```

Redis checkpointing lets callers reuse `thread_id` to continue conversation memory.

### Structured Research Graph

Main non-interactive research path:

```text
retrieve_memory
  -> plan_research
  -> search_sources
  -> read_sources
  -> score_sources
  -> synthesize_research
  -> analyze_gaps
  -> optional deepening pass
  -> critique_research
  -> revise_final_answer
  -> check_policy
  -> optional policy repair pass
  -> verify_citations
  -> END
```

Key behaviors:

- `retrieve_memory` queries Neo4j before planning.
- `plan_research` receives prior related runs, claims, and trusted/repeated sources as context.
- Search uses Tavily if `TAVILY_API_KEY` is set, with DuckDuckGo HTML fallback.
- Source pages are fetched and cleaned with `httpx` + BeautifulSoup.
- Sources are scored for authority, relevance, risk, and score.
- The model returns structured findings, citations, confidence, limitations, and follow-up questions.
- A critic node produces a quality report.
- A policy node flags citation count, weak citations, unsupported claims, low diversity, and incomplete deepening.
- Deep/comparison runs can perform a bounded policy repair loop.

### Interactive Human Review Graph

Interactive path:

```text
POST /research/interactive
```

The graph executes until the `human_review_checkpoint` node, then uses LangGraph `interrupt()` to pause.

Resume path:

```text
POST /research/interactive/{thread_id}/review
```

Resume uses `Command(resume=...)` with a human decision:

```json
{
  "decision": "approved",
  "reviewer_notes": "Looks good.",
  "selected_issues": []
}
```

Decisions:

- `approved`: resumes to citation verification/finalization.
- `needs_work`: routes into one more repair pass before finalization.

## Research Assistant Frontend

Open:

```text
http://localhost:8080
```

The frontend supports:

- Standard structured research runs.
- Interactive review runs using true LangGraph pause/resume.
- Human review on saved runs.
- Follow-up pass from human-selected issues.
- Copy Markdown.
- Recent saved runs.
- Policy report and quality report cards.
- Citation verification cards.
- Source quality views.
- Execution trace view.
- Neo4j memory panel with health, search, current-run memory, backlog, dedup, and audit.

## Network Design Helper

Open:

```text
http://localhost:8080/network-design
```

The Network Design Helper is a chat-first workflow for network design engineers. It retrieves uploaded standards, extracts structured requirements, summarizes the design discussion, builds a design package, prepares a FortiGate handoff payload, and creates a compliance matrix.

Workflow shape:

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

Browser/API shape:

```text
Browser /network-design
  -> POST /api/network-design/message
  -> POST /api/network-design/chat
  -> optional POST /api/fortigate/design from generated handoff
```

Exports:

- Design package Markdown.
- FortiGate handoff JSON.
- Raw run JSON.
- Detailed audit Markdown with standards evidence, extracted requirements, compliance matrix, execution traces, and optional FortiGate judge/config evidence.
- Draft FortiGate `.conf` after the UI calls the FortiGate agent from the handoff.

## FortiGate Provisioning Agent

Open:

```text
http://localhost:8080/fortigate
```

The FortiGate agent is artifact-only. It can parse existing configs, build logical and FortiGate-specific designs, generate draft CLI configuration, validate the result, check standards, run a model judge, and save review-ready packages. It does not make live device changes.

Full graph shape:

```text
intake_request
  -> parse_existing_config
  -> retrieve_standards
  -> identify_missing_inputs
  -> optional human_clarification_checkpoint
  -> build_logical_design
  -> build_fortigate_design
  -> build_implementation_intent
  -> check_intent_contract
  -> optional repair_implementation_intent
  -> analyze_change_impact
  -> monolith generate_config_artifacts
     or sectional generation:
        build_interfaces_dhcp_section
        -> build_fortiswitch_section
        -> build_wifi_section
        -> build_sdwan_routing_section
        -> build_objects_services_section
        -> build_firewall_policies_section
        -> validate_config_sections
        -> assemble_sectional_config_artifacts
  -> validate_config
  -> check_standards
  -> risk_review
  -> check_cli_contract
  -> optional autonomous_repair_config_artifacts
  -> optional builder_review_config_artifacts
  -> validation / standards / risk / CLI completeness loop
  -> optional refine_config_artifacts
  -> validation / standards / risk / CLI completeness loop
  -> frontier_model_judge
  -> optional autonomous_repair_config_artifacts
  -> optional revise_after_judge or regenerate_config_after_judge
  -> optional human_review_checkpoint
  -> finalize_package
```

The judge uses `FORTIGATE_JUDGE_MODEL_NAME`; judge calls bypass LiteLLM's web-search interception so the full review packet is judged directly. By default, the graph builds a structured implementation intent contract before CLI generation, covering VLANs, DHCP decisions, SD-WAN behavior, FortiSwitch, WiFi, object inventory, and the firewall policy matrix. Deterministic completeness gates inspect both intent and CLI output before Qwen sees the package.

Set `FORTIGATE_SECTIONAL_GENERATION_ENABLED=true` to replace the single initial CLI build with dedicated Gemma section builders for interfaces/DHCP, FortiSwitch, WiFi, SD-WAN/routing, objects/services, and firewall policies; Python then merges the sections into the usual `config_artifacts` package. The graph runs a thinking-enabled, higher-temperature builder review/refactor pass with `FORTIGATE_BUILDER_REVIEW_MODEL_NAME`, a thinking-enabled pre-judge config refiner with `FORTIGATE_CONFIG_REFINER_MODEL_NAME`, and up to `FORTIGATE_AUTONOMOUS_REPAIR_LIMIT` thinking-enabled autonomous repair passes for fixable engineering issues before asking for human review. Set `FORTIGATE_BUILDER_REVIEW_MODE=off` or `FORTIGATE_CONFIG_REFINEMENT_MODE=off` to skip either model pass, and tune `FORTIGATE_BUILDER_REVIEW_TEMPERATURE` to change how aggressively the builder review explores network, firewall policy, and security improvements. Reasoning traces returned by the model are not saved or rendered. Saved FortiGate runs expose human review questions derived from remaining judge items; reviewer answers are interpreted by the generation model into config instructions, accepted-risk notes, or requests for more detail. The package is marked final only when the judge returns `pass`.

## Standards Library

Source standards live under:

```text
/data/fortigate-standards/raw
```

The index is written to:

```text
/data/fortigate-standards/index.json
```

Supported source types include Markdown/text/config files, HTML, PDF, Word, PowerPoint, and Excel. The ingester skips `.DS_Store`, `._*`, and `__MACOSX` files.

Ingestion writes the JSON fallback index and attempts to create a Redis Search index with full-text fields plus vector embeddings. Standards search runs Redis keyword search and Redis vector search, fuses the ranked lists with Reciprocal Rank Fusion, optionally reranks fused candidates when `STANDARDS_RERANK_ENABLED=true`, and falls back to JSON keyword search if Redis or embeddings are unavailable.

Neo4j is not the first standards lookup engine in this version. Redis handles fast retrieval; Neo4j remains the right next layer for standards relationships, applicability, exceptions, and long-term audit graph traceability.

Rebuild the standards index with:

```bash
curl -sS http://localhost:8001/fortigate/standards/ingest \
  -H 'Content-Type: application/json' \
  -d '{"source_dir":"/data/fortigate-standards/raw"}'
```

## Persistence And Memory

### Redis

Redis is used for LangGraph checkpointing:

- `/chat` conversation state.
- `/research` graph checkpoints.
- `/research/interactive` interrupt/resume checkpoints.
- `/network-design` graph checkpoints.
- `/fortigate` graph checkpoints.

### JSON Run Archive

Every completed research run is saved as JSON in:

```text
/data/research-runs
```

Network Design Helper and FortiGate runs are saved as JSON in:

```text
/data/network-design-runs
/data/fortigate-runs
```

These JSON archives remain the fallback source of record.

### Neo4j Research Memory

Neo4j is used as a queryable graph memory layer. Writes are best-effort: if Neo4j is unavailable, research still completes and saves JSON.

Graph shape:

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

Neo4j supports:

- Retrieval before web search.
- Research backlog: weak citations, unsupported claims, missing perspectives, policy warnings, `needs_work` reviews.
- Human review audit graph.
- Cross-run source/claim deduplication.
- Source-to-run lookup.
- Follow-up chain lookup.

## API Endpoints

### Platform

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | App health and model name |
| `GET` | `/graph` | Chat graph visualizer |
| `GET` | `/graph/mermaid` | Chat graph Mermaid source |
| `GET` | `/research/graph` | Research graph visualizer |
| `GET` | `/research/graph/mermaid` | Research graph Mermaid source |
| `GET` | `/network-design/graph` | Network Design graph visualizer |
| `GET` | `/network-design/graph/mermaid` | Network Design graph Mermaid source |
| `GET` | `/fortigate/graph` | FortiGate graph visualizer |
| `GET` | `/fortigate/graph/mermaid` | FortiGate graph Mermaid source |

### Chat

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/chat` | Checkpointed chat |

Example:

```bash
curl -sS http://localhost:8001/chat \
  -H 'Content-Type: application/json' \
  -d '{"thread_id":"demo","message":"Say hello from LangGraph."}'
```

### Research

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/research` | Run structured research to completion |
| `GET` | `/research/runs` | List saved research runs |
| `GET` | `/research/runs/{run_id}` | Load saved research run |
| `POST` | `/research/runs/{run_id}/review` | Save human review decision |
| `POST` | `/research/runs/{run_id}/follow-up` | Run one more targeted follow-up pass |
| `POST` | `/research/interactive` | Start interactive run and pause at human review |
| `POST` | `/research/interactive/{thread_id}/review` | Resume paused interactive run |

Example:

```bash
curl -sS http://localhost:8001/research \
  -H 'Content-Type: application/json' \
  -d '{"thread_id":"research-demo","mode":"quick","question":"What is LangGraph interrupt/resume useful for?","constraints":"Prefer official docs."}'
```

### Network Design

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/network-design/message` | Run one conversational chat turn |
| `POST` | `/network-design/chat` | Generate a standards-aware design package |
| `GET` | `/network-design/runs` | List saved Network Design runs |
| `GET` | `/network-design/runs/{run_id}` | Load a saved Network Design run |
| `GET` | `/network-design/standards/search?q=...` | Search uploaded standards |

The package response includes `standard_requirements`, `compliance_matrix`, `fortigate_handoff`, and `markdown`.

### FortiGate

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/fortigate/design` | Generate an artifact-only FortiGate package |
| `POST` | `/fortigate/interactive` | Start interactive FortiGate workflow |
| `POST` | `/fortigate/interactive/{thread_id}/resume` | Resume interactive FortiGate workflow |
| `POST` | `/fortigate/configs/parse` | Parse FortiGate config text |
| `POST` | `/fortigate/changes/analyze` | Analyze a requested change against config text |
| `POST` | `/fortigate/runs/{run_id}/judge` | Re-run judge for a saved FortiGate package |
| `POST` | `/fortigate/runs/{run_id}/review` | Save human review decision |
| `POST` | `/fortigate/standards/ingest` | Rebuild standards index |
| `GET` | `/fortigate/standards/search?q=...` | Search uploaded standards |
| `GET` | `/fortigate/runs` | List saved FortiGate runs |
| `GET` | `/fortigate/runs/{run_id}` | Load saved FortiGate run |

### Neo4j Memory

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/memory/health` | Neo4j memory health |
| `GET` | `/memory/research/search?q=...` | Search prior research runs |
| `GET` | `/memory/research/runs/{run_id}` | Current graph memory for a run |
| `GET` | `/memory/research/runs/{run_id}/audit` | Human review, policy, quality, parent/follow-up audit |
| `GET` | `/memory/research/sources/{source_hash}/runs` | Runs that used the source |
| `GET` | `/memory/research/backlog` | Unresolved research issues |
| `GET` | `/memory/research/dedup?q=...` | Repeated claims and repeated sources |

Frontend callers use the same backend paths with `/api` prepended, for example `POST /api/network-design/message` and `POST /api/fortigate/design`. The frontend server strips `/api` and proxies to `API_PROXY_TARGET`.

Examples:

```bash
curl -sS http://localhost:8001/memory/health
curl -sS 'http://localhost:8001/memory/research/search?q=human&limit=5'
curl -sS 'http://localhost:8001/memory/research/backlog?limit=10'
```

## Useful Cypher Queries

Runs and sources:

```cypher
MATCH (r:ResearchRun)-[:USED_SOURCE]->(s:Source)
RETURN r.run_id, r.question, s.title, s.host
LIMIT 25;
```

Follow-up chains:

```cypher
MATCH (r:ResearchRun)-[:FOLLOWED_BY]->(f:ResearchRun)
RETURN r.run_id, r.question, f.run_id, f.question
LIMIT 25;
```

Repeated sources:

```cypher
MATCH (r:ResearchRun)-[:USED_SOURCE]->(s:Source)
WITH s, count(DISTINCT r) AS runs
WHERE runs > 1
RETURN s.host, s.url, runs
ORDER BY runs DESC
LIMIT 25;
```

Backlog:

```cypher
MATCH (r:ResearchRun)-[:HAS_POLICY]->(p:PolicyReport)
OPTIONAL MATCH (r)-[:HAS_QUALITY]->(q:QualityReport)
WHERE p.warning_count > 0
   OR p.blocking_issue_count > 0
   OR q.weak_citation_count > 0
   OR q.missing_perspective_count > 0
RETURN r.run_id, r.question, p.warning_count, p.blocking_issue_count,
       q.weak_citation_count, q.missing_perspective_count
ORDER BY r.created_at DESC
LIMIT 25;
```

## Helper Commands

Agent platform:

```bash
cd ~/dev/agent-platform
docker-compose ps
docker-compose logs -f langgraph-app
docker-compose up -d --build
```

If helper symlinks are present:

```bash
~/agent-platform status
~/agent-platform health
~/agent-platform frontend
~/agent-platform open
~/agent-platform graph
~/agent-platform research "What should we research next?"
~/agent-platform logs
~/agent-platform rebuild
```

Langfuse:

```bash
~/langfuse-platform status
~/langfuse-platform health
~/langfuse-platform logs
~/langfuse-platform creds
```

## Verified Behavior

Verified behavior as of this update:

- FastAPI health returns `ok` and model `gemma-local`.
- Password-protected frontend is available on port `8080`.
- `/api/health` proxies through the frontend to the internal FastAPI health endpoint after login.
- Network Design is the default frontend; the Research UI remains available at `/research`.
- LangGraph app calls LiteLLM on `agent-host.example:4010/v1`.
- Langfuse traces are enabled.
- Redis checkpointing supports chat and interactive graph state.
- Interactive research pauses at `human_review_checkpoint` and resumes with `Command(resume=...)`.
- Neo4j memory is connected and receives research run writes.
- Memory retrieval runs before research planning.
- Backlog, dedup, audit, source lookup, and memory search endpoints work.

## Operational Notes And Caveats

- Host networking is a Linux-focused deployment choice. If you run the stack on Docker Desktop, adjust the Compose files and URLs for that environment.
- For public access, publish only the frontend. Keep `:8001`, Redis, Neo4j, LiteLLM, and Langfuse internal unless they are separately secured.
- If the UI loads but the first chat/design interaction does not respond, confirm the page is using `apiBase = "/api"` and that `API_PROXY_TARGET` resolves from inside the frontend container.
- Neo4j memory writes are intentionally best-effort while JSON remains the fallback archive.
- Tavily is preferred for search when configured; DuckDuckGo HTML is a fallback and can be less reliable.
- The password gate is intentionally simple. Cloudflare Access, SSO, user roles, and multi-user approval attribution are future work.
- OPA is not wired yet. Policy checks are currently deterministic Python checks inside the research graph.

## Good Next Steps

High-value next upgrades:

- Backfill all existing JSON research runs into Neo4j, not just new runs.
- Add source reputation scoring over time.
- Add stale-source detection and recency checks.
- Add user identity/roles for human review approvals.
- Add OPA as an external policy gate once policy requirements stabilize.
- Add richer graph visualizations for Neo4j memory paths inside the frontend.
