import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.graph import build_graph, initial_messages
from app.memory import Neo4jResearchMemory
from app.research import build_research_graph

try:
    from langfuse import get_client
    from langfuse.langchain import CallbackHandler
except ImportError:  # pragma: no cover - optional dependency guard
    get_client = None
    CallbackHandler = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    thread_id: Optional[str] = Field(default=None, description="Conversation thread id for checkpointed memory")


class ChatResponse(BaseModel):
    response: str
    model: str
    thread_id: str


class CitationItem(BaseModel):
    claim: str = ""
    title: str = ""
    url: str = ""
    evidence: str = ""
    confidence: str = "medium"


class CitationVerificationItem(BaseModel):
    claim: str = ""
    url: str = ""
    verdict: str = "unclear"
    rationale: str = ""


class SourceScoreItem(BaseModel):
    title: str = ""
    url: str = ""
    authority: str = "unknown"
    relevance: str = "unknown"
    risk: str = "unknown"
    score: int = 0
    rationale: str = ""


class SourceItem(BaseModel):
    title: str = ""
    url: str = ""
    snippet: str = ""
    status: str = ""


class QualityReportItem(BaseModel):
    unsupported_claims: list[str] = Field(default_factory=list)
    weak_citations: list[str] = Field(default_factory=list)
    missing_perspectives: list[str] = Field(default_factory=list)
    source_risks: list[str] = Field(default_factory=list)
    confidence_rationale: str = "Not provided."
    revision_instructions: list[str] = Field(default_factory=list)


class ExecutionTraceItem(BaseModel):
    node: str = ""
    started_at: str = ""
    duration_ms: int = 0
    summary: str = ""
    branch: str = ""


class PolicyReportItem(BaseModel):
    passed: bool = True
    warnings: list[str] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)


class HumanReviewItem(BaseModel):
    decision: str = "pending"
    reviewer_notes: str = ""
    selected_issues: list[str] = Field(default_factory=list)
    reviewed_at: str = ""
    follow_up_run_id: Optional[str] = None


class HumanReviewRequest(BaseModel):
    decision: str = Field(..., pattern="^(approved|needs_work)$")
    reviewer_notes: str = ""
    selected_issues: list[str] = Field(default_factory=list)


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: str = Field(default="quick")
    constraints: str = Field(default="")
    thread_id: Optional[str] = Field(default=None, description="Research thread id for checkpointed state")


class ResearchFollowUpRequest(BaseModel):
    mode: str = Field(default="quick")
    instruction: str = Field(default="Run one more targeted pass to address remaining policy and quality issues.")


class InteractiveResearchResponse(BaseModel):
    status: str
    thread_id: str
    checkpoint_thread_id: str
    review: dict[str, Any] = Field(default_factory=dict)
    run: Optional["ResearchResponse"] = None


class ResearchRunSummary(BaseModel):
    run_id: str
    created_at: str
    thread_id: str
    mode: str
    question: str
    confidence: str
    source_count: int
    citation_count: int
    review_decision: str = "pending"


class ResearchResponse(BaseModel):
    run_id: str
    created_at: str
    thread_id: str
    model: str
    mode: str
    question: str
    plan: list[str]
    source_strategy: list[str]
    sources: list[SourceItem]
    evidence: list[SourceItem]
    source_scores: list[SourceScoreItem]
    gap_analysis: list[str] = Field(default_factory=list)
    deepening_queries: list[str] = Field(default_factory=list)
    deepened_sources: list[SourceItem] = Field(default_factory=list)
    deepened_evidence: list[SourceItem] = Field(default_factory=list)
    deepened_source_scores: list[SourceScoreItem] = Field(default_factory=list)
    deepened_areas: list[str] = Field(default_factory=list)
    repair_iterations: int = 0
    repair_queries: list[str] = Field(default_factory=list)
    repair_sources: list[SourceItem] = Field(default_factory=list)
    repair_evidence: list[SourceItem] = Field(default_factory=list)
    repair_source_scores: list[SourceScoreItem] = Field(default_factory=list)
    plan_results: list[str]
    findings: list[str]
    citations: list[CitationItem]
    citation_verifications: list[CitationVerificationItem]
    quality_report: QualityReportItem = Field(default_factory=QualityReportItem)
    execution_trace: list[ExecutionTraceItem] = Field(default_factory=list)
    policy_report: PolicyReportItem = Field(default_factory=PolicyReportItem)
    memory_context: dict[str, Any] = Field(default_factory=dict)
    answer: str
    confidence: str
    follow_up_questions: list[str]
    limitations: list[str]
    markdown: str
    human_review: Optional[HumanReviewItem] = None
    parent_run_id: Optional[str] = None


def _bullet_list(items: list[str]) -> str:
    if not items:
        return "- None"
    return "\n".join(f"- {item}" for item in items)


def render_research_markdown(response: ResearchResponse) -> str:
    lines: list[str] = [
        f"# Research Brief: {response.question}",
        "",
        f"- Mode: `{response.mode}`",
        f"- Model: `{response.model}`",
        f"- Thread: `{response.thread_id}`",
        f"- Run ID: `{response.run_id}`",
        f"- Confidence: `{response.confidence}`",
        "",
        "## Answer",
        "",
        response.answer or "No answer returned.",
        "",
        "## Policy Report",
        "",
        f"- Passed: `{response.policy_report.passed}`",
        "",
        "### Blocking Issues",
        "",
        _bullet_list(response.policy_report.blocking_issues),
        "",
        "### Warnings",
        "",
        _bullet_list(response.policy_report.warnings),
        "",
        "### Passed Checks",
        "",
        _bullet_list(response.policy_report.checks),
        "",
        "## Quality Report",
        "",
        f"- Confidence rationale: {response.quality_report.confidence_rationale}",
        "",
        "### Unsupported Claims",
        "",
        _bullet_list(response.quality_report.unsupported_claims),
        "",
        "### Weak Citations",
        "",
        _bullet_list(response.quality_report.weak_citations),
        "",
        "### Missing Perspectives",
        "",
        _bullet_list(response.quality_report.missing_perspectives),
        "",
        "### Source Risks",
        "",
        _bullet_list(response.quality_report.source_risks),
        "",
        "### Revision Instructions Applied",
        "",
        _bullet_list(response.quality_report.revision_instructions),
        "",
        "## Retrieved Research Memory",
        "",
        f"- Related runs: `{len(response.memory_context.get('related_runs', []))}`",
        f"- Prior claims: `{len(response.memory_context.get('prior_claims', []))}`",
        f"- Reusable sources: `{len(response.memory_context.get('trusted_sources', []))}`",
        "",
        "## Detailed Synthesis From Executed Plan",
        "",
        _bullet_list(response.plan_results),
        "",
        "## Deepened Areas",
        "",
        _bullet_list(response.deepened_areas),
        "",
        "## Findings",
        "",
        _bullet_list(response.findings),
        "",
        "## Citations",
        "",
    ]

    verification_by_key = {
        (item.claim, item.url): item for item in response.citation_verifications
    }
    if response.citations:
        for idx, citation in enumerate(response.citations, start=1):
            verification = verification_by_key.get((citation.claim, citation.url))
            lines.extend(
                [
                    f"{idx}. {citation.claim}",
                    f"   - Source: [{citation.title or citation.url}]({citation.url})",
                    f"   - Evidence: {citation.evidence or 'Not provided'}",
                    f"   - Confidence: `{citation.confidence}`",
                    f"   - Verification: `{verification.verdict if verification else 'not_verified'}`",
                    f"   - Rationale: {verification.rationale if verification and verification.rationale else 'Not provided'}",
                ]
            )
    else:
        lines.append("- No citations returned.")

    lines.extend(["", "## Source Quality", ""])
    if response.source_scores:
        lines.append("| Score | Authority | Relevance | Risk | Source | Rationale |")
        lines.append("| ---: | --- | --- | --- | --- | --- |")
        for score in response.source_scores:
            source = f"[{score.title or score.url}]({score.url})"
            rationale = score.rationale.replace("|", "\\|") if score.rationale else ""
            lines.append(
                f"| {score.score} | {score.authority} | {score.relevance} | {score.risk} | {source} | {rationale} |"
            )
    else:
        lines.append("- No source scores returned.")

    lines.extend(["", "## Execution Trace", ""])
    if response.execution_trace:
        lines.append("| Node | Duration | Branch | Summary |")
        lines.append("| --- | ---: | --- | --- |")
        for item in response.execution_trace:
            summary = item.summary.replace("|", "\\|") if item.summary else ""
            branch = item.branch or ""
            lines.append(f"| `{item.node}` | {item.duration_ms} ms | {branch} | {summary} |")
    else:
        lines.append("- No execution trace returned.")

    lines.extend(
        [
            "",
            "## Execution Plan / Audit Trail",
            "",
            _bullet_list(response.plan),
            "",
            "## Gap Analysis",
            "",
            _bullet_list(response.gap_analysis),
            "",
            "## Deepening Queries",
            "",
            _bullet_list(response.deepening_queries),
            "",
            "## Policy Repair",
            "",
            f"- Repair iterations: `{response.repair_iterations}`",
            "",
            "### Repair Queries",
            "",
            _bullet_list(response.repair_queries),
            "",
            "## Source Strategy",
            "",
            _bullet_list(response.source_strategy),
            "",
            "## Limitations",
            "",
            _bullet_list(response.limitations),
            "",
            "## Follow-Up Questions",
            "",
            _bullet_list(response.follow_up_questions),
            "",
        ]
    )
    if response.human_review:
        lines.extend([
            "## Human Review",
            "",
            f"- Decision: `{response.human_review.decision}`",
            f"- Reviewed At: {response.human_review.reviewed_at or 'Not recorded'}",
            f"- Reviewer Notes: {response.human_review.reviewer_notes or 'None'}",
            f"- Follow-Up Run: {response.human_review.follow_up_run_id or 'None'}",
            "- Selected Issues:",
            _bullet_list(response.human_review.selected_issues),
            "",
        ])
    if response.parent_run_id:
        lines.extend(["## Parent Run", "", f"- Follow-up generated from `{response.parent_run_id}`", ""])
    return "\n".join(lines)


RUNS_DIR = Path(os.getenv("RESEARCH_RUNS_DIR", "/data/research-runs"))


def _safe_run_id(run_id: str) -> str:
    if not run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in run_id):
        raise HTTPException(status_code=404, detail="Research run not found")
    return run_id


def _run_path(run_id: str) -> Path:
    return RUNS_DIR / f"{_safe_run_id(run_id)}.json"


def save_research_run(response: ResearchResponse) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _run_path(response.run_id).write_text(response.model_dump_json(indent=2))


async def save_research_memory(response: ResearchResponse) -> None:
    memory = getattr(app.state, "research_memory", None)
    if memory is None:
        return
    try:
        await memory.save_research_run(response)
    except Exception:
        # Memory is best-effort while JSON remains the fallback source of record.
        return


async def persist_research_run(response: ResearchResponse) -> None:
    save_research_run(response)
    await save_research_memory(response)


def load_research_run(run_id: str) -> ResearchResponse:
    path = _run_path(run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Research run not found")
    return ResearchResponse.model_validate_json(path.read_text())


def summarize_research_run(response: ResearchResponse) -> ResearchRunSummary:
    return ResearchRunSummary(
        run_id=response.run_id,
        created_at=response.created_at,
        thread_id=response.thread_id,
        mode=response.mode,
        question=response.question,
        confidence=response.confidence,
        source_count=len(response.sources),
        citation_count=len(response.citations),
        review_decision=response.human_review.decision if response.human_review else "pending",
    )


def list_research_runs(limit: int = 20) -> list[ResearchRunSummary]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    runs: list[ResearchResponse] = []
    for path in RUNS_DIR.glob("*.json"):
        try:
            runs.append(ResearchResponse.model_validate_json(path.read_text()))
        except Exception:
            continue
    runs.sort(key=lambda run: run.created_at, reverse=True)
    return [summarize_research_run(run) for run in runs[:limit]]


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def langfuse_enabled() -> bool:
    return all(os.getenv(name) for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    base_url = get_required_env("LITELLM_BASE_URL")
    api_key = get_required_env("LITELLM_API_KEY")
    redis_url = get_required_env("REDIS_URL")
    model_name = os.getenv("MODEL_NAME", "gemma-local")

    llm = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=0.2,
    )

    async with AsyncRedisSaver.from_conn_string(redis_url) as checkpointer:
        await checkpointer.asetup()
        app.state.model_name = model_name
        app.state.langfuse = None
        app.state.langfuse_handler = None
        app.state.research_memory = Neo4jResearchMemory.from_env()
        if app.state.research_memory is not None:
            try:
                await app.state.research_memory.init_schema()
            except Exception:
                app.state.research_memory = None
        if langfuse_enabled() and get_client is not None and CallbackHandler is not None:
            app.state.langfuse = get_client()
            app.state.langfuse_handler = CallbackHandler()
        async def retrieve_research_memory(query: str) -> dict[str, Any]:
            memory = getattr(app.state, "research_memory", None)
            if memory is None:
                return {"related_runs": [], "prior_claims": [], "trusted_sources": []}
            return await memory.retrieve_context(query, limit=5)

        app.state.graph = build_graph(llm, checkpointer)
        app.state.research_graph = build_research_graph(llm, checkpointer, memory_retriever=retrieve_research_memory)
        app.state.interactive_research_graph = build_research_graph(
            llm,
            checkpointer,
            human_review_interrupt=True,
            memory_retriever=retrieve_research_memory,
        )
        try:
            yield
        finally:
            if app.state.research_memory is not None:
                await app.state.research_memory.close()
        if app.state.langfuse is not None:
            app.state.langfuse.flush()


app = FastAPI(title="LangGraph Agent", version="0.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("FRONTEND_ORIGIN", "http://localhost:8080"),
        os.getenv("FRONTEND_ORIGIN", "http://localhost:8080"),
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": app.state.model_name,
        "langfuse_enabled": app.state.langfuse_handler is not None,
    }


def graph_mermaid() -> str:
    return app.state.graph.get_graph().draw_mermaid()


def research_mermaid() -> str:
    return app.state.research_graph.get_graph().draw_mermaid()


@app.get("/graph/mermaid", response_class=PlainTextResponse)
async def graph_mermaid_endpoint():
    return graph_mermaid()


@app.get("/research/graph/mermaid", response_class=PlainTextResponse)
async def research_graph_mermaid_endpoint():
    return research_mermaid()


@app.get("/graph", response_class=HTMLResponse)
async def graph_page():
    mermaid = graph_mermaid()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LangGraph Agent Graph</title>
  <style>
    :root {{ color-scheme: light dark; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f172a;
      color: #e2e8f0;
    }}
    header {{
      padding: 20px 28px;
      border-bottom: 1px solid #334155;
      background: #111827;
    }}
    main {{ padding: 24px 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .meta {{ color: #94a3b8; }}
    .panel {{
      margin-top: 20px;
      padding: 20px;
      border: 1px solid #334155;
      border-radius: 12px;
      background: #020617;
      overflow-x: auto;
    }}
    .mermaid {{
      display: flex;
      justify-content: center;
      min-width: 480px;
    }}
    pre {{
      white-space: pre-wrap;
      color: #cbd5e1;
      background: #111827;
      padding: 16px;
      border-radius: 8px;
    }}
    a {{ color: #93c5fd; }}
  </style>
</head>
<body>
  <header>
    <h1>LangGraph Agent Graph</h1>
    <div class="meta">Model: {app.state.model_name} | Raw Mermaid: <a href="/graph/mermaid">/graph/mermaid</a> | Health: <a href="/health">/health</a></div>
  </header>
  <main>
    <section class="panel">
      <div class="mermaid">
{mermaid}
      </div>
    </section>
    <section class="panel">
      <h2>Mermaid Source</h2>
      <pre>{mermaid}</pre>
    </section>
  </main>
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
    mermaid.initialize({{ startOnLoad: true, theme: "dark", securityLevel: "loose" }});
  </script>
</body>
</html>"""


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    thread_id = request.thread_id or str(uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {
            "langfuse_trace_name": "langgraph-chat",
            "langfuse_session_id": thread_id,
            "langfuse_user_id": "local-dev",
        },
    }
    if app.state.langfuse_handler is not None:
        config["callbacks"] = [app.state.langfuse_handler]

    try:
        result = await app.state.graph.ainvoke(
            {"messages": initial_messages(request.message)},
            config=config,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if app.state.langfuse is not None:
        app.state.langfuse.flush()

    final_message = result["messages"][-1]
    return ChatResponse(response=final_message.content, model=app.state.model_name, thread_id=thread_id)


def _research_response_from_state(result: dict[str, Any], thread_id: str, mode: str, question: str) -> ResearchResponse:
    human_review = None
    if result.get("human_review_decision"):
        human_review = HumanReviewItem(
            decision=result.get("human_review_decision", "pending"),
            reviewer_notes=result.get("human_review_notes", ""),
            selected_issues=result.get("human_selected_issues", []),
            reviewed_at=datetime.now(timezone.utc).isoformat(),
            follow_up_run_id=None,
        )
    response = ResearchResponse(
        run_id=str(uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
        thread_id=thread_id,
        model=app.state.model_name,
        mode=mode,
        question=question,
        plan=result.get("plan", []),
        source_strategy=result.get("source_strategy", []),
        sources=[SourceItem(**item) for item in result.get("sources", [])],
        evidence=[
            SourceItem(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=(item.get("text") or item.get("snippet") or "")[:600],
                status=item.get("status", ""),
            )
            for item in result.get("evidence", [])
        ],
        source_scores=[SourceScoreItem(**item) for item in result.get("source_scores", [])],
        gap_analysis=result.get("gap_analysis", []),
        deepening_queries=result.get("deepening_queries", []),
        deepened_sources=[SourceItem(**item) for item in result.get("deepened_sources", [])],
        deepened_evidence=[
            SourceItem(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=(item.get("text") or item.get("snippet") or "")[:600],
                status=item.get("status", ""),
            )
            for item in result.get("deepened_evidence", [])
        ],
        deepened_source_scores=[SourceScoreItem(**item) for item in result.get("deepened_source_scores", [])],
        deepened_areas=result.get("deepened_areas", []),
        repair_iterations=result.get("repair_iterations", 0),
        repair_queries=result.get("repair_queries", []),
        repair_sources=[SourceItem(**item) for item in result.get("repair_sources", [])],
        repair_evidence=[
            SourceItem(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=(item.get("text") or item.get("snippet") or "")[:600],
                status=item.get("status", ""),
            )
            for item in result.get("repair_evidence", [])
        ],
        repair_source_scores=[SourceScoreItem(**item) for item in result.get("repair_source_scores", [])],
        plan_results=result.get("plan_results", []),
        findings=result.get("findings", []),
        citations=[CitationItem(**item) for item in result.get("citations", [])],
        citation_verifications=[CitationVerificationItem(**item) for item in result.get("citation_verifications", [])],
        quality_report=QualityReportItem(**result.get("quality_report", {})),
        execution_trace=[ExecutionTraceItem(**item) for item in result.get("execution_trace", [])],
        policy_report=PolicyReportItem(**result.get("policy_report", {})),
        memory_context=result.get("memory_context", {}),
        answer=result.get("answer", ""),
        confidence=result.get("confidence", "medium"),
        follow_up_questions=result.get("follow_up_questions", []),
        limitations=result.get("limitations", []),
        markdown="",
        human_review=human_review,
        parent_run_id=None,
    )
    response.markdown = render_research_markdown(response)
    return response


def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any]:
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if not interrupts:
        return {}
    item = interrupts[0] if isinstance(interrupts, (list, tuple)) else interrupts
    value = getattr(item, "value", item)
    return value if isinstance(value, dict) else {"value": value}


def _interactive_config(thread_id: str, mode: str) -> dict[str, Any]:
    config = {
        "configurable": {"thread_id": f"research-interactive:{thread_id}"},
        "metadata": {
            "langfuse_trace_name": "research-pipeline-interactive",
            "langfuse_session_id": thread_id,
            "langfuse_user_id": "local-dev",
            "research_mode": mode,
        },
    }
    if app.state.langfuse_handler is not None:
        config["callbacks"] = [app.state.langfuse_handler]
    return config


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    thread_id = request.thread_id or str(uuid4())
    checkpoint_thread_id = f"research:{thread_id}"
    config = {
        "configurable": {"thread_id": checkpoint_thread_id},
        "metadata": {
            "langfuse_trace_name": "research-pipeline",
            "langfuse_session_id": thread_id,
            "langfuse_user_id": "local-dev",
            "research_mode": request.mode,
        },
    }
    if app.state.langfuse_handler is not None:
        config["callbacks"] = [app.state.langfuse_handler]

    try:
        result = await app.state.research_graph.ainvoke(
            {
                "question": request.question,
                "mode": request.mode,
                "constraints": request.constraints,
            },
            config=config,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if app.state.langfuse is not None:
        app.state.langfuse.flush()

    response = ResearchResponse(
        run_id=str(uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
        thread_id=thread_id,
        model=app.state.model_name,
        mode=request.mode,
        question=request.question,
        plan=result.get("plan", []),
        source_strategy=result.get("source_strategy", []),
        sources=[SourceItem(**item) for item in result.get("sources", [])],
        evidence=[
            SourceItem(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=(item.get("text") or item.get("snippet") or "")[:600],
                status=item.get("status", ""),
            )
            for item in result.get("evidence", [])
        ],
        source_scores=[SourceScoreItem(**item) for item in result.get("source_scores", [])],
        gap_analysis=result.get("gap_analysis", []),
        deepening_queries=result.get("deepening_queries", []),
        deepened_sources=[SourceItem(**item) for item in result.get("deepened_sources", [])],
        deepened_evidence=[
            SourceItem(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=(item.get("text") or item.get("snippet") or "")[:600],
                status=item.get("status", ""),
            )
            for item in result.get("deepened_evidence", [])
        ],
        deepened_source_scores=[SourceScoreItem(**item) for item in result.get("deepened_source_scores", [])],
        deepened_areas=result.get("deepened_areas", []),
        repair_iterations=result.get("repair_iterations", 0),
        repair_queries=result.get("repair_queries", []),
        repair_sources=[SourceItem(**item) for item in result.get("repair_sources", [])],
        repair_evidence=[
            SourceItem(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=(item.get("text") or item.get("snippet") or "")[:600],
                status=item.get("status", ""),
            )
            for item in result.get("repair_evidence", [])
        ],
        repair_source_scores=[SourceScoreItem(**item) for item in result.get("repair_source_scores", [])],
        plan_results=result.get("plan_results", []),
        findings=result.get("findings", []),
        citations=[CitationItem(**item) for item in result.get("citations", [])],
        citation_verifications=[
            CitationVerificationItem(**item) for item in result.get("citation_verifications", [])
        ],
        quality_report=QualityReportItem(**result.get("quality_report", {})),
        execution_trace=[ExecutionTraceItem(**item) for item in result.get("execution_trace", [])],
        policy_report=PolicyReportItem(**result.get("policy_report", {})),
        memory_context=result.get("memory_context", {}),
        answer=result.get("answer", ""),
        confidence=result.get("confidence", "medium"),
        follow_up_questions=result.get("follow_up_questions", []),
        limitations=result.get("limitations", []),
        markdown="",
        human_review=None,
        parent_run_id=None,
    )
    response.markdown = render_research_markdown(response)
    await persist_research_run(response)
    return response


@app.get("/memory/health")
async def memory_health():
    memory = getattr(app.state, "research_memory", None)
    if memory is None:
        return {"enabled": False, "ok": False, "detail": "Neo4j memory is not configured or failed initialization."}
    return await memory.ping()


@app.get("/memory/research/runs/{run_id}")
async def memory_research_run(run_id: str):
    memory = getattr(app.state, "research_memory", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="Neo4j memory is not available")
    result = await memory.get_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Research run memory not found")
    return result


@app.get("/memory/research/search")
async def memory_research_search(q: str, limit: int = 10):
    memory = getattr(app.state, "research_memory", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="Neo4j memory is not available")
    query = q.strip()
    if not query:
        return []
    return await memory.search(query, limit=max(1, min(limit, 50)))


@app.get("/memory/research/sources/{source_hash}/runs")
async def memory_research_source_runs(source_hash: str, limit: int = 25):
    memory = getattr(app.state, "research_memory", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="Neo4j memory is not available")
    result = await memory.source_runs(source_hash, limit=max(1, min(limit, 100)))
    if result is None:
        raise HTTPException(status_code=404, detail="Source memory not found")
    return result


@app.get("/memory/research/backlog")
async def memory_research_backlog(limit: int = 25):
    memory = getattr(app.state, "research_memory", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="Neo4j memory is not available")
    return await memory.backlog(limit=max(1, min(limit, 100)))


@app.get("/memory/research/runs/{run_id}/audit")
async def memory_research_audit(run_id: str):
    memory = getattr(app.state, "research_memory", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="Neo4j memory is not available")
    result = await memory.audit(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Research run audit not found")
    return result


@app.get("/memory/research/dedup")
async def memory_research_dedup(q: str = "", limit: int = 10):
    memory = getattr(app.state, "research_memory", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="Neo4j memory is not available")
    return await memory.dedup(q.strip(), limit=max(1, min(limit, 50)))


@app.post("/research/interactive", response_model=InteractiveResearchResponse)
async def research_interactive(request: ResearchRequest):
    thread_id = request.thread_id or str(uuid4())
    config = _interactive_config(thread_id, request.mode)
    try:
        result = await app.state.interactive_research_graph.ainvoke(
            {"question": request.question, "mode": request.mode, "constraints": request.constraints},
            config=config,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if app.state.langfuse is not None:
        app.state.langfuse.flush()

    review = _interrupt_payload(result)
    checkpoint_thread_id = f"research-interactive:{thread_id}"
    if review:
        return InteractiveResearchResponse(
            status="awaiting_review",
            thread_id=thread_id,
            checkpoint_thread_id=checkpoint_thread_id,
            review=review,
            run=None,
        )
    response = _research_response_from_state(result, thread_id, request.mode, request.question)
    await persist_research_run(response)
    return InteractiveResearchResponse(
        status="completed",
        thread_id=thread_id,
        checkpoint_thread_id=checkpoint_thread_id,
        review={},
        run=response,
    )


@app.post("/research/interactive/{thread_id}/review", response_model=InteractiveResearchResponse)
async def research_interactive_review(thread_id: str, request: HumanReviewRequest):
    config = _interactive_config(thread_id, "interactive")
    try:
        result = await app.state.interactive_research_graph.ainvoke(Command(resume=request.model_dump()), config=config)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if app.state.langfuse is not None:
        app.state.langfuse.flush()

    review = _interrupt_payload(result)
    checkpoint_thread_id = f"research-interactive:{thread_id}"
    if review:
        return InteractiveResearchResponse(
            status="awaiting_review",
            thread_id=thread_id,
            checkpoint_thread_id=checkpoint_thread_id,
            review=review,
            run=None,
        )

    question = result.get("question", "")
    mode = result.get("mode", "quick")
    response = _research_response_from_state(result, thread_id, mode, question)
    response.human_review = HumanReviewItem(
        decision=request.decision,
        reviewer_notes=request.reviewer_notes,
        selected_issues=request.selected_issues,
        reviewed_at=datetime.now(timezone.utc).isoformat(),
        follow_up_run_id=None,
    )
    response.markdown = render_research_markdown(response)
    await persist_research_run(response)
    return InteractiveResearchResponse(
        status="completed",
        thread_id=thread_id,
        checkpoint_thread_id=checkpoint_thread_id,
        review={},
        run=response,
    )


def _follow_up_constraints(run: ResearchResponse, instruction: str) -> str:
    policy = run.policy_report
    quality = run.quality_report
    citation_urls = [item.url for item in run.citations if item.url]
    source_urls = [item.url for item in [*run.sources, *run.deepened_sources, *run.repair_sources] if item.url]
    sections = [
        f"Human-in-the-loop follow-up for prior run `{run.run_id}`.",
        f"Instruction: {instruction}",
        "Focus the next pass on unresolved policy and quality issues. Do not simply repeat the prior answer.",
        "Prior policy blocking issues: " + ("; ".join(policy.blocking_issues) if policy.blocking_issues else "none"),
        "Prior policy warnings: " + ("; ".join(policy.warnings) if policy.warnings else "none"),
        "Prior unsupported claims: " + ("; ".join(quality.unsupported_claims) if quality.unsupported_claims else "none"),
        "Prior weak citations: " + ("; ".join(quality.weak_citations) if quality.weak_citations else "none"),
        "Prior missing perspectives: " + ("; ".join(quality.missing_perspectives) if quality.missing_perspectives else "none"),
        "Previously cited URLs: " + (", ".join(citation_urls[:12]) if citation_urls else "none"),
        "Previously used source URLs: " + (", ".join(source_urls[:16]) if source_urls else "none"),
    ]
    return "\n".join(sections)


@app.post("/research/runs/{run_id}/review", response_model=ResearchResponse)
async def research_run_review(run_id: str, request: HumanReviewRequest):
    run = load_research_run(run_id)
    run.human_review = HumanReviewItem(
        decision=request.decision,
        reviewer_notes=request.reviewer_notes,
        selected_issues=request.selected_issues,
        reviewed_at=datetime.now(timezone.utc).isoformat(),
        follow_up_run_id=run.human_review.follow_up_run_id if run.human_review else None,
    )
    run.markdown = render_research_markdown(run)
    await persist_research_run(run)
    return run


@app.post("/research/runs/{run_id}/follow-up", response_model=ResearchResponse)
async def research_run_follow_up(run_id: str, request: ResearchFollowUpRequest = ResearchFollowUpRequest()):
    prior_run = load_research_run(run_id)
    follow_up_request = ResearchRequest(
        question=prior_run.question,
        mode=request.mode,
        constraints=_follow_up_constraints(prior_run, request.instruction),
        thread_id=f"{prior_run.thread_id}-followup-{uuid4()}",
    )
    follow_up_run = await research(follow_up_request)
    follow_up_run.parent_run_id = prior_run.run_id
    follow_up_run.markdown = render_research_markdown(follow_up_run)
    await persist_research_run(follow_up_run)

    prior_run.human_review = HumanReviewItem(
        decision="needs_work",
        reviewer_notes=request.instruction,
        selected_issues=[*prior_run.policy_report.blocking_issues, *prior_run.policy_report.warnings, *prior_run.quality_report.missing_perspectives],
        reviewed_at=datetime.now(timezone.utc).isoformat(),
        follow_up_run_id=follow_up_run.run_id,
    )
    prior_run.markdown = render_research_markdown(prior_run)
    await persist_research_run(prior_run)
    return follow_up_run


@app.get("/research/runs", response_model=list[ResearchRunSummary])
async def research_runs(limit: int = 20):
    return list_research_runs(limit=max(1, min(limit, 100)))


@app.get("/research/runs/{run_id}", response_model=ResearchResponse)
async def research_run(run_id: str):
    return load_research_run(run_id)
