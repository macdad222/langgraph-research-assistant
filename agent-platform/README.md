# Agent Platform

Local AI agent platform running on a Linux agent host with LangGraph, LiteLLM, Redis, Langfuse, Neo4j, a password-protected web frontend, a Network Design Helper, and an artifact-only FortiGate provisioning agent.

For the full architecture and operations guide, see [`PLATFORM_REFERENCE.md`](PLATFORM_REFERENCE.md).

## Services

| Service | URL | Purpose |
| --- | --- | --- |
| Admin Portal | `http://agent-host.example` | Landing page for service UIs |
| Network Design Helper | `http://agent-host.example:8080/network-design` | Default protected UI for chat-first design, standards, handoff, compliance matrix, and exports |
| FortiGate Agent | `http://agent-host.example:8080/fortigate` | Draft FortiGate design/config package generation |
| Research Assistant | `http://agent-host.example:8080/research` | Research UI, still available but not linked from the FortiGate design frontend |
| Frontend API Proxy | `http://agent-host.example:8080/api/*` | Same-origin proxy from the frontend to the internal FastAPI service |
| Agent API | `http://agent-host.example:8001` | Internal FastAPI + LangGraph service; do not expose directly when publishing through Cloudflare |
| Neo4j Browser | `http://agent-host.example:7474` | Research memory graph |
| Langfuse | `http://agent-host.example:3001` | Traces and observability |
| LiteLLM | `http://agent-host.example:4010/v1` | Model gateway |
| vLLM | `http://vllm-host.example:8000/v1` | Dedicated inference server |

## End-To-End Request Path

The public web path should expose only the frontend service:

```text
Browser
  -> Cloudflare Tunnel / LAN reverse proxy
  -> research-frontend on :8080
  -> password gate
  -> static UI pages
  -> same-origin /api proxy
  -> internal FastAPI + LangGraph app on :8001
  -> Redis checkpoints, Neo4j memory, standards index, Langfuse traces
  -> LiteLLM on :4010/v1
  -> vLLM model server
```

The browser never needs to call `:8001` directly. `network-design.html`, `fortigate.html`, and `index.html` default to `apiBase = "/api"`, and `frontend/server.js` proxies `/api/*` to `API_PROXY_TARGET`.

## Frontend Access

The frontend has a simple shared password gate.

```env
FRONTEND_PASSWORD=replace-with-frontend-password
API_PROXY_TARGET=http://agent-host.example:8001
```

The default landing page is the Network Design Helper. The Research Assistant remains available at `/research`, but links to it are intentionally removed from the FortiGate design UI.

## Run

```bash
cd ~/dev/agent-platform
docker compose up -d --build
```

Check status:

```bash
docker compose ps
curl http://localhost:8080/health
curl -H 'Cookie: fortigate_frontend_auth=1' http://localhost:8080/api/health
curl http://localhost:8001/health
```

## Linux Host Networking

The Linux template runs `langgraph-app` with `network_mode: host`.

That is intentional: it lets LiteLLM see LangGraph requests from the real agent host/LAN source address instead of a Docker bridge IP. Because of this, the app uses host-local service URLs such as `redis://127.0.0.1:6379/0` and `bolt://127.0.0.1:7687`.

The frontend container stays on normal Docker networking and reaches the backend through `API_PROXY_TARGET`. For the current LAN deployment this is usually `http://agent.lab.internal:8001`.

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
- Optionally build draft CLI artifacts section-by-section for interfaces/DHCP, FortiSwitch, WiFi, SD-WAN/routing, objects/services, and firewall policies.
- Run an optional thinking-enabled, higher-temperature builder review/refactor pass with `FORTIGATE_BUILDER_REVIEW_MODEL_NAME`.
- Run deterministic validation and standards checks.
- Optionally refactor the draft config with `FORTIGATE_CONFIG_REFINER_MODEL_NAME` before judge review.
- Run a Qwen-configured model judge before human review.
- Regenerate draft config artifacts when the judge finds more than three review items.
- Generate human review questions from judge findings, accept reviewer answers, revise the config, and send it back through validation and Qwen judging.
- Interpret each reviewer answer with an LLM first, so "use best practice" becomes concrete config guidance and "this is okay" becomes an accepted-risk package note before Qwen re-judges.
- Save packages under `/data/fortigate-runs`.

No live device changes are made.

The default FortiGate config flow is Gemma-first with Qwen refinement before Qwen judge:

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
  -> generate_config_artifacts
     or build_interfaces_dhcp_section
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
  -> builder_review_config_artifacts
  -> validate_config
  -> check_standards
  -> risk_review
  -> check_cli_contract
  -> refine_config_artifacts
  -> validate_config
  -> check_standards
  -> risk_review
  -> check_cli_contract
  -> optional autonomous_repair_config_artifacts
  -> frontier_model_judge
  -> optional autonomous_repair_config_artifacts
  -> optional revise_after_judge or regenerate_config_after_judge
  -> optional human_review_checkpoint
  -> finalize_package
```

The graph now builds an `implementation_intent` contract before CLI generation. That contract covers VLANs, DHCP decisions, SD-WAN behavior, FortiSwitch, WiFi, object inventory, and the firewall policy matrix. Deterministic completeness gates check both the intent and generated CLI before Qwen judge review.

Set `FORTIGATE_SECTIONAL_GENERATION_ENABLED=true` to use the section-by-section initial build. In that mode, Gemma builds dedicated sections for interfaces/DHCP, FortiSwitch, WiFi, SD-WAN/routing, objects/services, and firewall policies, then Python merges the sections into the standard `config_artifacts` package. Set `FORTIGATE_BUILDER_REVIEW_MODE=off` to skip the builder self-review pass. Set `FORTIGATE_BUILDER_REVIEW_TEMPERATURE=1.0` or another value to tune how aggressively that pass explores network, firewall policy, and security improvements. The builder review, autonomous repair, Qwen refiner, and Qwen judge calls use vLLM template thinking via `chat_template_kwargs.enable_thinking=true`; reasoning traces are not saved or rendered. Set `FORTIGATE_CONFIG_REFINEMENT_MODE=off` to skip the pre-judge refinement step, or set `FORTIGATE_CONFIG_REFINER_MODEL_NAME` to test a different refiner model. Set `FORTIGATE_AUTONOMOUS_REPAIR_LIMIT=1` to control how many autonomous repair passes can run before the system asks for human review.

Use model-role profiles to A/B test the Gemma/Qwen split without hand-editing `.env`:

```bash
scripts/agent-platform model-profile list
scripts/agent-platform model-profile apply swapped --restart
scripts/agent-platform model-profile apply openrouter-qwen-builder --restart
scripts/agent-platform model-profile apply openrouter-grok-judge --restart
scripts/agent-platform model-profile apply openrouter-frontier --restart
scripts/agent-platform model-profile apply openrouter-grok-all --restart
scripts/agent-platform model-profile apply openrouter-qwen3.7-all --restart
scripts/agent-platform model-profile apply qwen36-gemma-chat --restart
scripts/agent-platform model-profile apply current --restart
```

The profile switcher edits approved model-role keys, including Network Design chat/intake/package/handoff model and temperature keys, and writes a timestamped `.env` backup before each change.

Network Design can split model calls by phase:

```env
NETWORK_CHAT_MODEL_NAME=extl-gemma-4-31b
NETWORK_INTAKE_MODEL_NAME=extl-gemma-4-31b
NETWORK_PACKAGE_MODEL_NAME=Qwen3.6-27B
NETWORK_HANDOFF_MODEL_NAME=Qwen3.6-27B
NETWORK_CHAT_TEMPERATURE=0.7
NETWORK_CHAT_ENABLE_THINKING=false
NETWORK_INTAKE_TEMPERATURE=0.0
NETWORK_PACKAGE_TEMPERATURE=0.2
NETWORK_HANDOFF_TEMPERATURE=0.0
```

FortiGate generation can fall back to Gemma when a large prompt exceeds the primary model's context budget:

```env
FORTIGATE_CONTEXT_FALLBACK_MODEL_NAME=extl-gemma-4-31b
FORTIGATE_CONTEXT_FALLBACK_MAX_OUTPUT_TOKENS=32768
FORTIGATE_CONTEXT_FALLBACK_TEMPERATURE=0.2
```

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

When using the frontend, these are called through `/api`, for example:

```text
POST /api/network-design/message
POST /api/fortigate/design
GET  /api/health
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
- Publish only the frontend through Cloudflare or another public proxy. Keep `:8001`, Redis, Neo4j, LiteLLM, and Langfuse internal unless explicitly secured.
- If the public UI loads but chat/design actions do not respond, check that the frontend is using `/api` and that `API_PROXY_TARGET` points to the reachable internal FastAPI URL.
- Neo4j writes are best-effort; research runs still save to JSON if Neo4j is temporarily unavailable.
- OPA is not wired yet. Current policy checks are Python checks inside the research graph.
- Do not commit or paste active API keys from `.env`.
