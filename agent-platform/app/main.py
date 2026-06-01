import asyncio
import json
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.fortigate import build_fortigate_graph, parse_fortigate_config
from app.fortigate_judge import run_frontier_judge
from app.fortigate_models import (
    FortiGateConfigSummary,
    FortiGateDesignJobResponse,
    FortiGateDesignRequest,
    FortiGateHumanReview,
    FortiGateInteractiveResponse,
    FortiGateJudgeReport,
    FortiGateReviewQuestion,
    FortiGateReviewRequest,
    FortiGateRunResponse,
    FortiGateRunSummary,
    FortiGateStandardsIngestRequest,
    FortiGateStandardsIngestResponse,
    FortiGateStandardsSearchResponse,
    FortiGateValidationReport,
)
from app.fortigate_policy import check_cli_completeness, check_standards_compliance, review_risk, validate_config_artifacts
from app.graph import build_graph, initial_messages
from app.memory import Neo4jResearchMemory
from app.network_design import build_network_design_graph
from app.network_design_models import (
    FortiGateHandoffPayload,
    NetworkDesignChatResponse,
    NetworkDesignJobResponse,
    NetworkDesignMessage,
    NetworkDesignRequest,
    NetworkDesignReadinessReport,
    NetworkDesignRunResponse,
    NetworkDesignRunSummary,
    NetworkDesignStandardsSearchResponse,
    NetworkDesignValidationReport,
)
from app.research import build_research_graph
from app.standards import DEFAULT_INDEX_PATH, ingest_standards, search_standards

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


class FortiGateConfigParseRequest(BaseModel):
    config_text: str = Field(default="")


class FortiGateChangeAnalysisRequest(BaseModel):
    current_config: str = Field(default="")
    requested_change: str = Field(..., min_length=1)


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


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return {}


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
FORTIGATE_RUNS_DIR = Path(os.getenv("FORTIGATE_RUNS_DIR", "/data/fortigate-runs"))
NETWORK_DESIGN_RUNS_DIR = Path(os.getenv("NETWORK_DESIGN_RUNS_DIR", "/data/network-design-runs"))


def _safe_run_id(run_id: str) -> str:
    if not run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in run_id):
        raise HTTPException(status_code=404, detail="Research run not found")
    return run_id


def _run_path(run_id: str) -> Path:
    return RUNS_DIR / f"{_safe_run_id(run_id)}.json"


def _fortigate_run_path(run_id: str) -> Path:
    return FORTIGATE_RUNS_DIR / f"{_safe_run_id(run_id)}.json"


def _fortigate_job_path(job_id: str) -> Path:
    return FORTIGATE_RUNS_DIR / "jobs" / f"{_safe_run_id(job_id)}.json"


def _network_design_run_path(run_id: str) -> Path:
    return NETWORK_DESIGN_RUNS_DIR / f"{_safe_run_id(run_id)}.json"


def _network_design_job_path(job_id: str) -> Path:
    return NETWORK_DESIGN_RUNS_DIR / "jobs" / f"{_safe_run_id(job_id)}.json"


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


def render_fortigate_markdown(response: FortiGateRunResponse) -> str:
    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "- None"

    artifacts = response.config_artifacts or {}
    lines = [
        f"# FortiGate Artifact Package: {response.intake.site_name or response.run_id}",
        "",
        f"- Status: `{response.status}`",
        f"- Request type: `{response.intake.request_type}`",
        f"- Thread: `{response.thread_id}`",
        f"- Run ID: `{response.run_id}`",
        "- Safety: `DRAFT ONLY - NOT APPLIED TO DEVICE`",
        "",
        "## Requirements Summary",
        "",
        response.intake.business_intent,
        "",
        "## Parsed Current Configuration",
        "",
        f"- Hostname: `{response.current_config_summary.hostname or 'unknown'}`",
        f"- Interfaces: `{len(response.current_config_summary.interfaces)}`",
        f"- VLANs: `{len(response.current_config_summary.vlans)}`",
        f"- Firewall policies: `{len(response.current_config_summary.firewall_policies)}`",
        f"- Address objects: `{len(response.current_config_summary.address_objects)}`",
        "",
        "## Logical Design",
        "",
        "```json",
        json.dumps(response.logical_design, indent=2),
        "```",
        "",
        "## FortiGate Design",
        "",
        "```json",
        json.dumps(response.fortigate_design, indent=2),
        "```",
        "",
        "## Implementation Intent",
        "",
        "```json",
        json.dumps(response.implementation_intent, indent=2),
        "```",
        "",
        "## Change Impact",
        "",
        "```json",
        json.dumps(response.change_impact, indent=2),
        "```",
        "",
        "## Draft CLI Configuration",
        "",
        "```text",
        str(artifacts.get("cli_config") or "No CLI config generated."),
        "```",
        "",
        "## Sectional Generation",
        "",
        "### Config Sections",
        "```json",
        json.dumps(response.config_sections, indent=2),
        "```",
        "### Section Validation",
        "```json",
        json.dumps(response.section_validation_report, indent=2),
        "```",
        "",
        "## Rollback Plan",
        "",
        str(artifacts.get("rollback_plan") or "No rollback plan generated."),
        "",
        "## Validation Report",
        "",
        f"- Passed: `{response.validation_report.passed}`",
        "### Blocking Issues",
        bullets(response.validation_report.blocking_issues),
        "### Warnings",
        bullets(response.validation_report.warnings),
        "",
        "## Completeness Reports",
        "",
        "### Intent Completeness",
        "```json",
        json.dumps(response.intent_completeness_report, indent=2),
        "```",
        "### CLI Completeness",
        "```json",
        json.dumps(response.cli_completeness_report, indent=2),
        "```",
        "### Autonomous Fixes",
        bullets(response.auto_fixed_items),
        "### Remaining Human Inputs",
        bullets(response.requires_human_input),
        "",
        "## Standards Report",
        "",
        f"- Passed: `{response.standards_report.passed}`",
        "### Warnings",
        bullets(response.standards_report.warnings),
        "",
        "## Frontier Model Judge",
        "",
        f"- Verdict: `{response.judge_report.verdict}`",
        f"- Model: `{response.judge_report.model or 'configured default'}`",
        "### Blocking Issues",
        bullets(response.judge_report.blocking_issues),
        "### Recommended Revisions",
        bullets(response.judge_report.recommended_revisions),
        "### Human Reviewer Focus",
        bullets(response.judge_report.human_reviewer_focus),
        "",
        "## Human Review Questions",
        "",
    ]
    if response.review_questions:
        for question in response.review_questions:
            lines.extend(
                [
                    f"### `{question.question_id}`",
                    "",
                    f"- Source: `{question.source}`",
                    f"- Required: `{question.required}`",
                    f"- Context: {question.context or 'None'}",
                    "",
                    question.question,
                    "",
                ]
            )
    else:
        lines.append("- No human review questions generated.")
    lines.extend(
        [
            "",
        "## Standards Evidence",
        "",
        ]
    )
    if response.standards:
        for chunk in response.standards:
            lines.extend([f"- `{chunk.chunk_id}` from `{chunk.document}` ({chunk.topic})"])
    else:
        lines.append("- No standards chunks retrieved.")
    if response.human_review:
        lines.extend(
            [
                "",
                "## Human Review",
                "",
                f"- Decision: `{response.human_review.decision}`",
                f"- Reviewed at: {response.human_review.reviewed_at or 'not recorded'}",
                f"- Notes: {response.human_review.reviewer_notes or 'None'}",
                "### Selected Issues",
                bullets(response.human_review.selected_issues),
                "### Answers",
                "```json",
                json.dumps(response.human_review.answers, indent=2),
                "```",
                "### Interpreted Answers",
                "```json",
                json.dumps(response.human_review.interpreted_answers, indent=2),
                "```",
            ]
        )
    return "\n".join(lines)


def _judge_items(report: FortiGateJudgeReport) -> list[tuple[str, str]]:
    fields = (
        ("blocking_issues", report.blocking_issues),
        ("warnings", report.warnings),
        ("standards_concerns", report.standards_concerns),
        ("config_risks", report.config_risks),
        ("missing_questions", report.missing_questions),
        ("recommended_revisions", report.recommended_revisions),
        ("human_reviewer_focus", report.human_reviewer_focus),
    )
    items: list[tuple[str, str]] = []
    seen: set[str] = set()
    for source, values in fields:
        for value in values:
            text = str(value).strip()
            fingerprint = f"{source}:{text.lower()}"
            if text and fingerprint not in seen:
                seen.add(fingerprint)
                items.append((source, text))
    return items


def build_fortigate_review_questions(report: FortiGateJudgeReport) -> list[FortiGateReviewQuestion]:
    questions: list[FortiGateReviewQuestion] = []
    for index, (source, text) in enumerate(_judge_items(report), start=1):
        if source == "missing_questions":
            prompt = text.rstrip("?") + "?"
        elif source == "recommended_revisions":
            prompt = f"Should this recommended revision be applied? If yes, provide the exact design/config direction: {text}"
        elif source == "human_reviewer_focus":
            prompt = f"Please review this focus area and provide approval notes or correction details: {text}"
        else:
            prompt = f"How should the draft address this {source.replace('_', ' ')} item: {text}"
        questions.append(
            FortiGateReviewQuestion(
                question_id=f"judge-{index:02d}",
                source=source,
                question=prompt,
                context=text,
                required=source in {"blocking_issues", "standards_concerns", "config_risks", "missing_questions"},
            )
        )
    return questions[:12]


def fortigate_response_from_state(result: dict[str, Any], thread_id: str, model_name: str) -> FortiGateRunResponse:
    judge_report = FortiGateJudgeReport(**result.get("judge_report", {}))
    response = FortiGateRunResponse(
        run_id=str(uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
        thread_id=thread_id,
        model=model_name,
        status=result.get("status", "needs_review"),
        intake=result.get("intake", {}),
        current_config_summary=FortiGateConfigSummary(**result.get("current_config_summary", {})),
        standards=result.get("standards", []),
        missing_questions=result.get("missing_questions", []),
        logical_design=result.get("logical_design", {}),
        fortigate_design=result.get("fortigate_design", {}),
        implementation_intent=result.get("implementation_intent", {}),
        intent_completeness_report=result.get("intent_completeness_report", {}),
        change_impact=result.get("change_impact", {}),
        config_sections=result.get("config_sections", {}),
        config_artifacts=result.get("config_artifacts", {}),
        section_validation_report=result.get("section_validation_report", {}),
        cli_completeness_report=result.get("cli_completeness_report", {}),
        validation_report=FortiGateValidationReport(**result.get("validation_report", {})),
        standards_report=FortiGateValidationReport(**result.get("standards_report", {})),
        risk_report=FortiGateValidationReport(**result.get("risk_report", {})),
        autonomous_repair_iterations=int(result.get("autonomous_repair_iterations", 0) or 0),
        auto_fixed_items=result.get("auto_fixed_items", []),
        requires_human_input=result.get("requires_human_input", []),
        judge_report=judge_report,
        review_questions=build_fortigate_review_questions(judge_report),
        human_review=(
            FortiGateHumanReview(
                decision=result.get("human_review_decision", "pending"),
                reviewer_notes=result.get("human_review_notes", ""),
                selected_issues=result.get("human_selected_issues", []),
                answers=result.get("human_review_answers", {}),
                interpreted_answers=result.get("human_review_interpreted_answers", []),
                reviewed_at=datetime.now(timezone.utc).isoformat(),
            )
            if result.get("human_review_decision")
            else None
        ),
        execution_trace=result.get("execution_trace", []),
    )
    response.markdown = render_fortigate_markdown(response)
    return response


def save_fortigate_run(response: FortiGateRunResponse) -> None:
    FORTIGATE_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _fortigate_run_path(response.run_id).write_text(response.model_dump_json(indent=2))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_fortigate_design_job(job: FortiGateDesignJobResponse) -> None:
    path = _fortigate_job_path(job.job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(job.model_dump_json(indent=2))


def load_fortigate_design_job(job_id: str) -> FortiGateDesignJobResponse:
    path = _fortigate_job_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="FortiGate design job not found")
    return FortiGateDesignJobResponse.model_validate_json(path.read_text())


def update_fortigate_design_job(
    job_id: str,
    *,
    status: str,
    run_id: str | None = None,
    error: str = "",
) -> FortiGateDesignJobResponse:
    job = load_fortigate_design_job(job_id)
    job.status = status
    job.updated_at = _utc_now()
    if run_id is not None:
        job.run_id = run_id
    if error:
        job.error = error
    save_fortigate_design_job(job)
    return job


def load_fortigate_run(run_id: str) -> FortiGateRunResponse:
    path = _fortigate_run_path(run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="FortiGate run not found")
    run = FortiGateRunResponse.model_validate_json(path.read_text())
    if not run.review_questions and run.judge_report.verdict != "pass":
        run.review_questions = build_fortigate_review_questions(run.judge_report)
    return run


def summarize_fortigate_run(response: FortiGateRunResponse) -> FortiGateRunSummary:
    return FortiGateRunSummary(
        run_id=response.run_id,
        created_at=response.created_at,
        thread_id=response.thread_id,
        request_type=response.intake.request_type,
        site_name=response.intake.site_name,
        status=response.status,
        judge_verdict=response.judge_report.verdict,
        validation_passed=response.validation_report.passed,
    )


def _fortigate_judge_packet(run: FortiGateRunResponse) -> dict[str, Any]:
    return {
        "intake": run.intake.model_dump(),
        "current_config_summary": run.current_config_summary.model_dump(),
        "logical_design": run.logical_design,
        "fortigate_design": run.fortigate_design,
        "implementation_intent": run.implementation_intent,
        "intent_completeness_report": run.intent_completeness_report,
        "change_impact": run.change_impact,
        "config_sections": run.config_sections,
        "section_validation_report": run.section_validation_report,
        "config_artifacts": run.config_artifacts,
        "cli_completeness_report": run.cli_completeness_report,
        "validation_report": run.validation_report.model_dump(),
        "standards_report": run.standards_report.model_dump(),
        "risk_report": run.risk_report.model_dump(),
        "autonomous_repair_iterations": run.autonomous_repair_iterations,
        "auto_fixed_items": run.auto_fixed_items,
        "requires_human_input": run.requires_human_input,
        "standards": [item.model_dump() for item in run.standards],
    }


def _fortigate_policy_state(run: FortiGateRunResponse) -> dict[str, Any]:
    return {
        "intake": run.intake.model_dump(),
        "current_config_summary": run.current_config_summary.model_dump(),
        "standards": [item.model_dump() for item in run.standards],
        "implementation_intent": run.implementation_intent,
        "config_sections": run.config_sections,
        "section_validation_report": run.section_validation_report,
        "config_artifacts": run.config_artifacts,
    }


async def _revise_fortigate_config_from_review(run: FortiGateRunResponse, review: FortiGateHumanReview) -> None:
    interpretation = await _interpret_fortigate_review_answers(run, review)
    review.interpreted_answers = interpretation
    response = await app.state.fortigate_generation_model.ainvoke(
        [
            SystemMessage(
                content=(
                    "Return only JSON with key config_artifacts. Update the FortiGate draft artifacts using the interpreted "
                    "human review answers and the prior judge report. Apply items where config_change_required is true. "
                    "For accepted_risk items, do not force a config change; add clear accepted-risk/package notes so the "
                    "next judge and human reviewer can see the exception. Preserve artifact-only safety labels, standards "
                    "citations, rollback details, and full CLI context. Do not apply changes to any device."
                )
            ),
            HumanMessage(
                content=json.dumps(
                    {
                        "intake": run.intake.model_dump(),
                        "current_config_summary": run.current_config_summary.model_dump(),
                        "logical_design": run.logical_design,
                        "fortigate_design": run.fortigate_design,
                        "change_impact": run.change_impact,
                        "previous_config_artifacts": run.config_artifacts,
                        "validation_report": run.validation_report.model_dump(),
                        "standards_report": run.standards_report.model_dump(),
                        "risk_report": run.risk_report.model_dump(),
                        "judge_report": run.judge_report.model_dump(),
                        "review_questions": [item.model_dump() for item in run.review_questions],
                        "human_review": review.model_dump(),
                        "interpreted_answers": interpretation,
                        "standards": [item.model_dump() for item in run.standards],
                    },
                    indent=2,
                )
            ),
        ]
    )
    data = _extract_json_object(str(response.content))
    artifacts = data.get("config_artifacts") if isinstance(data.get("config_artifacts"), dict) else data
    if not isinstance(artifacts, dict) or not artifacts:
        raise HTTPException(status_code=502, detail="Model did not return revised FortiGate config artifacts.")
    artifacts.setdefault("safety_label", "DRAFT ONLY - NOT APPLIED TO DEVICE")
    run.config_artifacts = artifacts
    policy_state = _fortigate_policy_state(run)
    run.validation_report = validate_config_artifacts(policy_state)
    run.standards_report = check_standards_compliance(policy_state)
    run.risk_report = review_risk(policy_state)
    run.cli_completeness_report = check_cli_completeness(policy_state).model_dump()


async def _interpret_fortigate_review_answers(run: FortiGateRunResponse, review: FortiGateHumanReview) -> list[dict[str, Any]]:
    if not review.answers:
        return []
    response = await app.state.fortigate_generation_model.ainvoke(
        [
            SystemMessage(
                content=(
                    "Return only JSON with key interpreted_answers. Interpret each human review answer for a FortiGate "
                    "artifact update workflow. For answers like 'use best practice', produce a concrete FortiGate-oriented "
                    "recommendation and config instructions. For answers like 'this is ok', 'accept risk', or 'do not change', "
                    "set accepted_risk true, config_change_required false, and write a package_note explaining that Qwen may "
                    "still flag it. For unclear answers, set needs_more_info true and explain the missing detail. Do not remove "
                    "artifact-only safety constraints."
                )
            ),
            HumanMessage(
                content=json.dumps(
                    {
                        "intake": run.intake.model_dump(),
                        "current_config_summary": run.current_config_summary.model_dump(),
                        "fortigate_design": run.fortigate_design,
                        "current_config_artifacts": run.config_artifacts,
                        "judge_report": run.judge_report.model_dump(),
                        "review_questions": [item.model_dump() for item in run.review_questions],
                        "human_review_answers": review.answers,
                        "reviewer_notes": review.reviewer_notes,
                        "standards": [item.model_dump() for item in run.standards],
                        "required_schema": {
                            "interpreted_answers": [
                                {
                                    "question_id": "judge-01",
                                    "source": "blocking_issues",
                                    "human_answer": "raw reviewer answer",
                                    "intent": "apply_change | accepted_risk | no_change | needs_more_info",
                                    "config_change_required": True,
                                    "accepted_risk": False,
                                    "needs_more_info": False,
                                    "recommendation": "specific engineering recommendation",
                                    "config_instructions": ["specific FortiGate artifact or CLI update"],
                                    "package_note": "note to preserve in the package/audit",
                                }
                            ]
                        },
                    },
                    indent=2,
                )
            ),
        ]
    )
    data = _extract_json_object(str(response.content))
    interpreted = data.get("interpreted_answers") if isinstance(data, dict) else None
    if not isinstance(interpreted, list):
        interpreted = []
    normalized: list[dict[str, Any]] = []
    question_lookup = {item.question_id: item for item in run.review_questions}
    for question_id, answer in review.answers.items():
        match = next((item for item in interpreted if str(item.get("question_id")) == question_id), None)
        question = question_lookup.get(question_id)
        if not isinstance(match, dict):
            match = {
                "question_id": question_id,
                "source": question.source if question else "human_review",
                "human_answer": answer,
                "intent": "needs_more_info" if not str(answer).strip() else "apply_change",
                "config_change_required": bool(str(answer).strip()),
                "accepted_risk": False,
                "needs_more_info": not bool(str(answer).strip()),
                "recommendation": str(answer),
                "config_instructions": [str(answer)] if str(answer).strip() else [],
                "package_note": "Fallback interpretation created because the model did not return this answer.",
            }
        match.setdefault("question_id", question_id)
        match.setdefault("source", question.source if question else "human_review")
        match.setdefault("human_answer", answer)
        match.setdefault("package_note", "")
        normalized.append(match)
    return normalized


def list_fortigate_runs(limit: int = 20) -> list[FortiGateRunSummary]:
    FORTIGATE_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    runs: list[FortiGateRunResponse] = []
    for path in FORTIGATE_RUNS_DIR.glob("*.json"):
        try:
            runs.append(FortiGateRunResponse.model_validate_json(path.read_text()))
        except Exception:
            continue
    runs.sort(key=lambda run: run.created_at, reverse=True)
    return [summarize_fortigate_run(run) for run in runs[:limit]]


def render_network_design_markdown(response: NetworkDesignRunResponse) -> str:
    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "- None"

    lines = [
        f"# Network Design Package: {response.intake.site_name or response.intake.customer_name or response.run_id}",
        "",
        f"- Status: `{response.status}`",
        f"- Customer: `{response.intake.customer_name or 'not provided'}`",
        f"- Site: `{response.intake.site_name or 'not provided'}`",
        f"- Thread: `{response.thread_id}`",
        f"- Run ID: `{response.run_id}`",
        "- Safety: `DESIGN ARTIFACT ONLY - NOT APPLIED TO DEVICE`",
        "",
        "## Design Goal",
        "",
        response.intake.design_goal,
        "",
        "## Requirements Summary",
        "",
        "```json",
        json.dumps(response.requirements_summary, indent=2),
        "```",
        "",
        "## Design Package",
        "",
        "```json",
        json.dumps(response.design_package, indent=2),
        "```",
        "",
        "## Open Questions",
        "",
        bullets(response.missing_questions),
        "",
        "## Extracted Standard Requirements",
        "",
    ]
    if response.standard_requirements:
        lines.append("| ID | Priority | Topic | Source | Requirement |")
        lines.append("| --- | --- | --- | --- | --- |")
        for requirement in response.standard_requirements:
            text = requirement.requirement.replace("|", "\\|")
            source = requirement.source_document.replace("|", "\\|")
            lines.append(f"| `{requirement.requirement_id}` | {requirement.priority} | {requirement.topic} | {source} | {text} |")
    else:
        lines.append("- No structured requirements extracted.")
    lines.extend(
        [
            "",
            "## Compliance Matrix",
            "",
        ]
    )
    if response.compliance_matrix:
        lines.append("| Requirement | Status | Design Evidence | Handoff Evidence | Rationale |")
        lines.append("| --- | --- | --- | --- | --- |")
        for item in response.compliance_matrix:
            design = "<br>".join(item.design_evidence).replace("|", "\\|") or "None"
            handoff = "<br>".join(item.handoff_evidence).replace("|", "\\|") or "None"
            rationale = item.rationale.replace("|", "\\|")
            lines.append(f"| `{item.requirement_id}` | {item.status} | {design} | {handoff} | {rationale} |")
    else:
        lines.append("- No compliance matrix generated.")
    lines.extend(
        [
            "",
            "## FortiGate Handoff Payload",
            "",
            "```json",
            json.dumps(response.fortigate_handoff.model_dump(), indent=2),
            "```",
            "",
            "## Validation Report",
            "",
            f"- Passed: `{response.validation_report.passed}`",
            "",
            "### Blocking Issues",
            bullets(response.validation_report.blocking_issues),
            "",
            "### Warnings",
            bullets(response.validation_report.warnings),
            "",
            "### Checks",
            bullets(response.validation_report.checks),
            "",
            "## Standards Evidence",
            "",
        ]
    )
    if response.standards:
        for chunk in response.standards:
            retrieval = chunk.retrieval_backend or "unknown"
            ranks = []
            if chunk.keyword_rank is not None:
                ranks.append(f"keyword rank {chunk.keyword_rank}")
            if chunk.vector_rank is not None:
                ranks.append(f"vector rank {chunk.vector_rank}")
            if chunk.rrf_score is not None:
                ranks.append(f"RRF {chunk.rrf_score}")
            rank_text = f"; {', '.join(ranks)}" if ranks else ""
            lines.append(f"- `{chunk.chunk_id}` from `{chunk.document}` ({chunk.topic}); retrieval `{retrieval}`{rank_text}")
    else:
        lines.append("- No standards chunks retrieved.")
    lines.extend(["", "## Execution Trace", ""])
    if response.execution_trace:
        lines.append("| Node | Duration | Branch | Summary |")
        lines.append("| --- | ---: | --- | --- |")
        for item in response.execution_trace:
            summary = str(item.get("summary", "")).replace("|", "\\|")
            lines.append(f"| `{item.get('node', '')}` | {item.get('duration_ms', 0)} ms | {item.get('branch', '')} | {summary} |")
    else:
        lines.append("- No execution trace returned.")
    return "\n".join(lines)


def network_design_response_from_state(result: dict[str, Any], thread_id: str, model_name: str) -> NetworkDesignRunResponse:
    response = NetworkDesignRunResponse(
        run_id=str(uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
        thread_id=thread_id,
        model=model_name,
        status=result.get("status", "needs_design_review"),
        intake=result.get("intake", {}),
        structured_intake=result.get("structured_intake", {}),
        readiness_report=result.get("readiness_report", {}),
        messages=result.get("messages", []),
        standards=result.get("standards", []),
        standard_requirements=result.get("standard_requirements", []),
        requirements_summary=result.get("requirements_summary", {}),
        missing_questions=result.get("missing_questions", []),
        design_package=result.get("design_package", {}),
        fortigate_handoff=FortiGateHandoffPayload(**result.get("fortigate_handoff", {})),
        compliance_matrix=result.get("compliance_matrix", []),
        validation_report=NetworkDesignValidationReport(**result.get("validation_report", {})),
        execution_trace=result.get("execution_trace", []),
    )
    response.markdown = render_network_design_markdown(response)
    return response


def save_network_design_run(response: NetworkDesignRunResponse) -> None:
    NETWORK_DESIGN_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _network_design_run_path(response.run_id).write_text(response.model_dump_json(indent=2))


def save_network_design_job(job: NetworkDesignJobResponse) -> None:
    path = _network_design_job_path(job.job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(job.model_dump_json(indent=2))


def load_network_design_job(job_id: str) -> NetworkDesignJobResponse:
    path = _network_design_job_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Network design job not found")
    return NetworkDesignJobResponse.model_validate_json(path.read_text())


def update_network_design_job(
    job_id: str,
    *,
    status: str,
    run_id: str | None = None,
    error: str = "",
) -> NetworkDesignJobResponse:
    job = load_network_design_job(job_id)
    job.status = status
    job.updated_at = _utc_now()
    if run_id is not None:
        job.run_id = run_id
    if error:
        job.error = error
    save_network_design_job(job)
    return job


def load_network_design_run(run_id: str) -> NetworkDesignRunResponse:
    path = _network_design_run_path(run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Network design run not found")
    return NetworkDesignRunResponse.model_validate_json(path.read_text())


def summarize_network_design_run(response: NetworkDesignRunResponse) -> NetworkDesignRunSummary:
    return NetworkDesignRunSummary(
        run_id=response.run_id,
        created_at=response.created_at,
        thread_id=response.thread_id,
        customer_name=response.intake.customer_name,
        site_name=response.intake.site_name,
        status=response.status,
        standards_count=len(response.standards),
        validation_passed=response.validation_report.passed,
    )


def list_network_design_runs(limit: int = 20) -> list[NetworkDesignRunSummary]:
    NETWORK_DESIGN_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    runs: list[NetworkDesignRunResponse] = []
    for path in NETWORK_DESIGN_RUNS_DIR.glob("*.json"):
        try:
            runs.append(NetworkDesignRunResponse.model_validate_json(path.read_text()))
        except Exception:
            continue
    runs.sort(key=lambda run: run.created_at, reverse=True)
    return [summarize_network_design_run(run) for run in runs[:limit]]


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def fortigate_model_extra_body(enable_thinking: bool = False, passthrough: bool = False) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if passthrough:
        body["metadata"] = {"agentic_gateway_openclaw_passthrough": True}
    if enable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": True}
    return body


def chat_model_kwargs(extra_body: dict[str, Any] | None = None, max_tokens: int | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "timeout": get_float_env("LITELLM_REQUEST_TIMEOUT_SECONDS", 600.0),
    }
    token_limit = get_int_env("MODEL_MAX_OUTPUT_TOKENS", 32768) if max_tokens is None else max_tokens
    if token_limit > 0:
        kwargs["max_tokens"] = token_limit
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


def make_chat_model(
    *,
    model_name: str,
    base_url: str,
    api_key: str,
    temperature: float,
    extra_body: dict[str, Any] | None = None,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        **chat_model_kwargs(extra_body, max_tokens=max_tokens),
    )


def langfuse_enabled() -> bool:
    return all(os.getenv(name) for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    base_url = get_required_env("LITELLM_BASE_URL")
    api_key = get_required_env("LITELLM_API_KEY")
    redis_url = get_required_env("REDIS_URL")
    model_name = os.getenv("MODEL_NAME", "gemma-local")
    network_chat_model_name = os.getenv("NETWORK_CHAT_MODEL_NAME", model_name)
    network_intake_model_name = os.getenv("NETWORK_INTAKE_MODEL_NAME", network_chat_model_name)
    network_package_model_name = os.getenv("NETWORK_PACKAGE_MODEL_NAME", model_name)
    network_handoff_model_name = os.getenv("NETWORK_HANDOFF_MODEL_NAME", network_package_model_name)
    default_temperature = get_float_env("MODEL_TEMPERATURE", 0.2)
    network_chat_temperature = get_float_env("NETWORK_CHAT_TEMPERATURE", default_temperature)
    network_intake_temperature = get_float_env("NETWORK_INTAKE_TEMPERATURE", 0.0)
    network_package_temperature = get_float_env("NETWORK_PACKAGE_TEMPERATURE", default_temperature)
    network_handoff_temperature = get_float_env("NETWORK_HANDOFF_TEMPERATURE", 0.0)
    network_chat_enable_thinking = get_bool_env("NETWORK_CHAT_ENABLE_THINKING", False)

    fortigate_design_model_name = os.getenv("FORTIGATE_DESIGN_MODEL_NAME", model_name)
    # implementation_intent has its own model knob (bake-off winner differs from design);
    # defaults to the design model when unset.
    fortigate_intent_model_name = os.getenv("FORTIGATE_INTENT_MODEL_NAME", fortigate_design_model_name)
    fortigate_render_model_name = os.getenv("FORTIGATE_RENDER_MODEL_NAME", fortigate_intent_model_name)
    fortigate_autonomous_repair_model_name = os.getenv("FORTIGATE_AUTONOMOUS_REPAIR_MODEL_NAME", model_name)
    fortigate_autonomous_repair_temperature = get_float_env("FORTIGATE_AUTONOMOUS_REPAIR_TEMPERATURE", 0.4)
    # Global toggle for reasoning/thinking on the FortiGate review/repair/judge roles. Defaults to
    # on to preserve prior behavior; set false to drop thinking tokens for speed.
    fortigate_enable_thinking = get_bool_env("FORTIGATE_ENABLE_THINKING", True)
    # Per-role override for the judge (gatekeeper): thinking-on gives stronger scoring even when
    # the rest of the flow runs thinking-off. Defaults to the global flag.
    fortigate_judge_enable_thinking = get_bool_env("FORTIGATE_JUDGE_ENABLE_THINKING", fortigate_enable_thinking)

    # Per-role output token caps. Structuring/design/judge/repair stay tight; CLI-producing
    # roles (generation, builder review, refine, package) stay generous.
    generation_max_output_tokens = get_int_env("FORTIGATE_GENERATION_MAX_OUTPUT_TOKENS", get_int_env("MODEL_MAX_OUTPUT_TOKENS", 32768))
    design_max_output_tokens = get_int_env("FORTIGATE_DESIGN_MAX_OUTPUT_TOKENS", 8192)
    judge_max_output_tokens = get_int_env("FORTIGATE_JUDGE_MAX_OUTPUT_TOKENS", 8192)
    builder_review_max_output_tokens = get_int_env("FORTIGATE_BUILDER_REVIEW_MAX_OUTPUT_TOKENS", generation_max_output_tokens)
    config_refiner_max_output_tokens = get_int_env("FORTIGATE_CONFIG_REFINER_MAX_OUTPUT_TOKENS", generation_max_output_tokens)
    autonomous_repair_max_output_tokens = get_int_env("FORTIGATE_AUTONOMOUS_REPAIR_MAX_OUTPUT_TOKENS", 8192)
    network_chat_max_output_tokens = get_int_env("NETWORK_CHAT_MAX_OUTPUT_TOKENS", 4096)
    network_intake_max_output_tokens = get_int_env("NETWORK_INTAKE_MAX_OUTPUT_TOKENS", 4096)
    network_package_max_output_tokens = get_int_env("NETWORK_PACKAGE_MAX_OUTPUT_TOKENS", generation_max_output_tokens)
    network_handoff_max_output_tokens = get_int_env("NETWORK_HANDOFF_MAX_OUTPUT_TOKENS", 8192)

    llm = make_chat_model(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=default_temperature,
    )
    network_chat_model = make_chat_model(
        model_name=network_chat_model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=network_chat_temperature,
        extra_body=fortigate_model_extra_body(enable_thinking=network_chat_enable_thinking),
        max_tokens=network_chat_max_output_tokens,
    )
    network_intake_model = make_chat_model(
        model_name=network_intake_model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=network_intake_temperature,
        max_tokens=network_intake_max_output_tokens,
    )
    network_package_model = make_chat_model(
        model_name=network_package_model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=network_package_temperature,
        max_tokens=network_package_max_output_tokens,
    )
    network_handoff_model = make_chat_model(
        model_name=network_handoff_model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=network_handoff_temperature,
        max_tokens=network_handoff_max_output_tokens,
    )
    fortigate_generation_model = make_chat_model(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=default_temperature,
        extra_body=fortigate_model_extra_body(passthrough=True),
        max_tokens=generation_max_output_tokens,
    )
    fortigate_design_model = make_chat_model(
        model_name=fortigate_design_model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=default_temperature,
        extra_body=fortigate_model_extra_body(passthrough=True),
        max_tokens=design_max_output_tokens,
    )
    fortigate_intent_model = make_chat_model(
        model_name=fortigate_intent_model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=default_temperature,
        extra_body=fortigate_model_extra_body(passthrough=True),
        max_tokens=design_max_output_tokens,
    )
    # Renderer path (M2): structuring model that fills the typed FortiGateConfigModel.
    # Defaults to the intent model unless FORTIGATE_RENDER_MODEL_NAME overrides it.
    fortigate_render_model = fortigate_intent_model
    if fortigate_render_model_name != fortigate_intent_model_name:
        fortigate_render_model = make_chat_model(
            model_name=fortigate_render_model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=default_temperature,
            extra_body=fortigate_model_extra_body(passthrough=True),
            max_tokens=design_max_output_tokens,
        )

    async with AsyncRedisSaver.from_conn_string(redis_url) as checkpointer:
        await checkpointer.asetup()
        app.state.model_name = model_name
        app.state.network_chat_model_name = network_chat_model_name
        app.state.network_intake_model_name = network_intake_model_name
        app.state.network_package_model_name = network_package_model_name
        app.state.network_handoff_model_name = network_handoff_model_name
        app.state.fortigate_judge_model_name = os.getenv("FORTIGATE_JUDGE_MODEL_NAME", model_name)
        app.state.fortigate_builder_review_mode = os.getenv("FORTIGATE_BUILDER_REVIEW_MODE", "pre_refine")
        app.state.fortigate_builder_review_model_name = os.getenv("FORTIGATE_BUILDER_REVIEW_MODEL_NAME", model_name)
        app.state.fortigate_builder_review_temperature = get_float_env("FORTIGATE_BUILDER_REVIEW_TEMPERATURE", 1.0)
        app.state.fortigate_config_refinement_mode = os.getenv("FORTIGATE_CONFIG_REFINEMENT_MODE", "pre_judge")
        app.state.fortigate_config_refiner_model_name = os.getenv("FORTIGATE_CONFIG_REFINER_MODEL_NAME", app.state.fortigate_judge_model_name)
        app.state.fortigate_design_model_name = fortigate_design_model_name
        app.state.fortigate_intent_model_name = fortigate_intent_model_name
        app.state.fortigate_autonomous_repair_model_name = fortigate_autonomous_repair_model_name
        app.state.fortigate_context_fallback_model_name = os.getenv("FORTIGATE_CONTEXT_FALLBACK_MODEL_NAME", "extl-gemma-4-31b")
        app.state.fortigate_context_fallback_max_output_tokens = get_int_env("FORTIGATE_CONTEXT_FALLBACK_MAX_OUTPUT_TOKENS", 8192)
        app.state.fortigate_context_fallback_temperature = get_float_env("FORTIGATE_CONTEXT_FALLBACK_TEMPERATURE", 0.2)
        app.state.fortigate_autonomous_repair_limit = int(get_float_env("FORTIGATE_AUTONOMOUS_REPAIR_LIMIT", 1))
        app.state.fortigate_sectional_generation_enabled = get_bool_env("FORTIGATE_SECTIONAL_GENERATION_ENABLED", False)
        app.state.fortigate_renderer_enabled = get_bool_env("FORTIGATE_RENDERER_ENABLED", False)
        app.state.fortigate_render_model_name = fortigate_render_model_name
        app.state.fortigate_design_timeout_seconds = get_float_env("FORTIGATE_DESIGN_TIMEOUT_SECONDS", 1800.0)
        app.state.fortigate_design_job_semaphore = asyncio.Semaphore(max(1, get_int_env("FORTIGATE_DESIGN_CONCURRENCY", 1)))
        app.state.fortigate_design_job_tasks = {}
        app.state.network_design_timeout_seconds = get_float_env("NETWORK_DESIGN_TIMEOUT_SECONDS", 1200.0)
        app.state.network_design_job_semaphore = asyncio.Semaphore(max(1, get_int_env("NETWORK_DESIGN_CONCURRENCY", 1)))
        app.state.network_design_job_tasks = {}
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
        app.state.network_design_chat_model = network_chat_model
        app.state.network_design_intake_model = network_intake_model
        app.state.fortigate_generation_model = fortigate_generation_model
        app.state.fortigate_design_model = fortigate_design_model
        app.state.fortigate_intent_model = fortigate_intent_model
        app.state.fortigate_render_model = fortigate_render_model
        app.state.research_graph = build_research_graph(llm, checkpointer, memory_retriever=retrieve_research_memory)
        app.state.interactive_research_graph = build_research_graph(
            llm,
            checkpointer,
            human_review_interrupt=True,
            memory_retriever=retrieve_research_memory,
        )
        judge_model = ChatOpenAI(
            model=app.state.fortigate_judge_model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=0,
            **chat_model_kwargs(fortigate_model_extra_body(enable_thinking=fortigate_judge_enable_thinking, passthrough=True), max_tokens=judge_max_output_tokens),
        )
        app.state.fortigate_judge_model = judge_model
        builder_review_model = ChatOpenAI(
            model=app.state.fortigate_builder_review_model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=app.state.fortigate_builder_review_temperature,
            **chat_model_kwargs(fortigate_model_extra_body(enable_thinking=fortigate_enable_thinking), max_tokens=builder_review_max_output_tokens),
        )
        app.state.fortigate_builder_review_model = builder_review_model
        autonomous_repair_model = ChatOpenAI(
            model=fortigate_autonomous_repair_model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=fortigate_autonomous_repair_temperature,
            **chat_model_kwargs(fortigate_model_extra_body(enable_thinking=fortigate_enable_thinking, passthrough=True), max_tokens=autonomous_repair_max_output_tokens),
        )
        app.state.fortigate_autonomous_repair_model = autonomous_repair_model
        refiner_model = judge_model
        if app.state.fortigate_config_refiner_model_name != app.state.fortigate_judge_model_name:
            refiner_model = ChatOpenAI(
                model=app.state.fortigate_config_refiner_model_name,
                base_url=base_url,
                api_key=api_key,
                temperature=0,
                **chat_model_kwargs(fortigate_model_extra_body(enable_thinking=fortigate_enable_thinking, passthrough=True), max_tokens=config_refiner_max_output_tokens),
            )
        app.state.fortigate_config_refiner_model = refiner_model
        context_fallback_model = make_chat_model(
            model_name=app.state.fortigate_context_fallback_model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=app.state.fortigate_context_fallback_temperature,
            extra_body=fortigate_model_extra_body(enable_thinking=fortigate_enable_thinking, passthrough=True),
            max_tokens=app.state.fortigate_context_fallback_max_output_tokens,
        )
        app.state.fortigate_context_fallback_model = context_fallback_model
        app.state.fortigate_graph = build_fortigate_graph(
            fortigate_generation_model,
            checkpointer,
            design_model=fortigate_design_model,
            design_model_name=fortigate_design_model_name,
            intent_model=fortigate_intent_model,
            intent_model_name=fortigate_intent_model_name,
            judge_model=judge_model,
            judge_model_name=app.state.fortigate_judge_model_name,
            builder_review_model=builder_review_model,
            builder_review_model_name=app.state.fortigate_builder_review_model_name,
            builder_review_mode=app.state.fortigate_builder_review_mode,
            autonomous_repair_model=autonomous_repair_model,
            autonomous_repair_model_name=fortigate_autonomous_repair_model_name,
            config_refiner_model=refiner_model,
            config_refiner_model_name=app.state.fortigate_config_refiner_model_name,
            config_refinement_mode=app.state.fortigate_config_refinement_mode,
            context_fallback_model=context_fallback_model,
            context_fallback_model_name=app.state.fortigate_context_fallback_model_name,
            autonomous_repair_limit=app.state.fortigate_autonomous_repair_limit,
            sectional_generation_enabled=app.state.fortigate_sectional_generation_enabled,
            renderer_enabled=app.state.fortigate_renderer_enabled,
            config_model_model=fortigate_render_model,
            config_model_model_name=fortigate_render_model_name,
        )
        app.state.interactive_fortigate_graph = build_fortigate_graph(
            fortigate_generation_model,
            checkpointer,
            clarification_interrupt=True,
            human_review_interrupt=True,
            design_model=fortigate_design_model,
            design_model_name=fortigate_design_model_name,
            intent_model=fortigate_intent_model,
            intent_model_name=fortigate_intent_model_name,
            judge_model=judge_model,
            judge_model_name=app.state.fortigate_judge_model_name,
            builder_review_model=builder_review_model,
            builder_review_model_name=app.state.fortigate_builder_review_model_name,
            builder_review_mode=app.state.fortigate_builder_review_mode,
            autonomous_repair_model=autonomous_repair_model,
            autonomous_repair_model_name=fortigate_autonomous_repair_model_name,
            config_refiner_model=refiner_model,
            config_refiner_model_name=app.state.fortigate_config_refiner_model_name,
            config_refinement_mode=app.state.fortigate_config_refinement_mode,
            context_fallback_model=context_fallback_model,
            context_fallback_model_name=app.state.fortigate_context_fallback_model_name,
            autonomous_repair_limit=app.state.fortigate_autonomous_repair_limit,
            sectional_generation_enabled=app.state.fortigate_sectional_generation_enabled,
            renderer_enabled=app.state.fortigate_renderer_enabled,
            config_model_model=fortigate_render_model,
            config_model_model_name=fortigate_render_model_name,
        )
        app.state.network_design_graph = build_network_design_graph(network_package_model, checkpointer, handoff_model=network_handoff_model)
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


def fortigate_mermaid() -> str:
    return app.state.fortigate_graph.get_graph().draw_mermaid()


def network_design_mermaid() -> str:
    return app.state.network_design_graph.get_graph().draw_mermaid()


@app.get("/graph/mermaid", response_class=PlainTextResponse)
async def graph_mermaid_endpoint():
    return graph_mermaid()


@app.get("/research/graph/mermaid", response_class=PlainTextResponse)
async def research_graph_mermaid_endpoint():
    return research_mermaid()


@app.get("/fortigate/graph/mermaid", response_class=PlainTextResponse)
async def fortigate_graph_mermaid_endpoint():
    return fortigate_mermaid()


@app.get("/network-design/graph/mermaid", response_class=PlainTextResponse)
async def network_design_graph_mermaid_endpoint():
    return network_design_mermaid()


def graph_viewer_page(title: str, description: str, mermaid: str, raw_path: str) -> str:
    title_json = json.dumps(title)
    mermaid_json = json.dumps(mermaid)
    raw_path_json = json.dumps(raw_path)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b1020;
      --panel: #111827;
      --panel-soft: #0f172a;
      --line: #334155;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #38bdf8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(56, 189, 248, .16), transparent 34rem),
        var(--bg);
      color: var(--text);
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: flex-start;
      padding: 20px 28px;
      border-bottom: 1px solid var(--line);
      background: rgba(17, 24, 39, .92);
      position: sticky;
      top: 0;
      z-index: 3;
      backdrop-filter: blur(10px);
    }}
    main {{ padding: 20px 28px 32px; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    .meta {{ color: var(--muted); line-height: 1.45; }}
    .actions {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    button, a.button {{
      border: 0;
      border-radius: 12px;
      padding: 10px 12px;
      color: #06111f;
      background: var(--accent);
      font-weight: 800;
      cursor: pointer;
      text-decoration: none;
      font-size: 14px;
    }}
    button.secondary, a.button.secondary {{
      background: #1f2937;
      color: var(--text);
      border: 1px solid var(--line);
    }}
    .panel {{
      border: 1px solid rgba(148, 163, 184, .22);
      border-radius: 18px;
      background: rgba(15, 23, 42, .92);
      box-shadow: 0 18px 55px rgba(0, 0, 0, .28);
      overflow: hidden;
    }}
    .viewer-toolbar {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      padding: 12px;
      border-bottom: 1px solid var(--line);
      background: rgba(2, 6, 23, .52);
    }}
    .viewer-toolbar .left, .viewer-toolbar .right {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
    .zoom-label {{ color: var(--muted); font-size: 13px; min-width: 52px; text-align: center; }}
    .viewport {{
      height: calc(100vh - 210px);
      min-height: 520px;
      overflow: hidden;
      background:
        linear-gradient(rgba(148, 163, 184, .055) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148, 163, 184, .055) 1px, transparent 1px),
        #020617;
      background-size: 28px 28px;
      cursor: grab;
      position: relative;
    }}
    .viewport.dragging {{ cursor: grabbing; }}
    #graph-canvas {{
      transform-origin: 0 0;
      padding: 28px;
      min-width: 100%;
      width: max-content;
    }}
    #graph-canvas svg {{
      max-width: none !important;
      height: auto;
      filter: drop-shadow(0 18px 30px rgba(0,0,0,.25));
    }}
    .source-panel {{
      margin-top: 18px;
      padding: 16px;
      border: 1px solid rgba(148, 163, 184, .22);
      border-radius: 16px;
      background: rgba(15, 23, 42, .84);
    }}
    pre {{
      white-space: pre-wrap;
      color: #cbd5e1;
      background: #020617;
      padding: 16px;
      border-radius: 12px;
      border: 1px solid var(--line);
      max-height: 360px;
      overflow: auto;
    }}
    a {{ color: #7dd3fc; }}
    @media (max-width: 820px) {{
      header {{ flex-direction: column; }}
      .actions {{ justify-content: flex-start; }}
      .viewport {{ height: 70vh; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>{title}</h1>
      <div class="meta">{description}<br />Model: {app.state.model_name} | Raw Mermaid: <a href="{raw_path}">{raw_path}</a></div>
    </div>
    <div class="actions">
      <a class="button secondary" href="/network-design/graph">Network Design</a>
      <a class="button secondary" href="/fortigate/graph">FortiGate</a>
      <a class="button secondary" href="/research/graph">Research</a>
      <a class="button secondary" href="/graph">Chat</a>
    </div>
  </header>
  <main>
    <section class="panel">
      <div class="viewer-toolbar">
        <div class="left">
          <button id="zoom-out" class="secondary">-</button>
          <span id="zoom-label" class="zoom-label">100%</span>
          <button id="zoom-in" class="secondary">+</button>
          <button id="fit" class="secondary">Fit</button>
          <button id="reset" class="secondary">Reset</button>
        </div>
        <div class="right">
          <button id="download-svg">Download SVG</button>
          <button id="download-mmd" class="secondary">Download Mermaid</button>
        </div>
      </div>
      <div id="viewport" class="viewport">
        <div id="graph-canvas"></div>
      </div>
    </section>
    <section class="source-panel">
      <h2>Mermaid Source</h2>
      <pre id="source"></pre>
    </section>
  </main>
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
    const title = {title_json};
    const rawPath = {raw_path_json};
    const source = {mermaid_json};
    const canvas = document.getElementById("graph-canvas");
    const viewport = document.getElementById("viewport");
    const zoomLabel = document.getElementById("zoom-label");
    let scale = 1;
    let translateX = 0;
    let translateY = 0;
    let dragging = false;
    let startX = 0;
    let startY = 0;

    mermaid.initialize({{
      startOnLoad: false,
      theme: "dark",
      securityLevel: "loose",
      flowchart: {{
        curve: "basis",
        padding: 18,
        nodeSpacing: 44,
        rankSpacing: 54,
        htmlLabels: true
      }},
      themeVariables: {{
        background: "#020617",
        primaryColor: "#0f172a",
        primaryTextColor: "#e5e7eb",
        primaryBorderColor: "#38bdf8",
        lineColor: "#7dd3fc",
        secondaryColor: "#111827",
        tertiaryColor: "#1f2937",
        noteBkgColor: "#111827",
        noteTextColor: "#e5e7eb",
        fontFamily: "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
      }}
    }});

    document.getElementById("source").textContent = source;

    function applyTransform() {{
      canvas.style.transform = `translate(${{translateX}}px, ${{translateY}}px) scale(${{scale}})`;
      zoomLabel.textContent = `${{Math.round(scale * 100)}}%`;
    }}

    function fitToView() {{
      const svg = canvas.querySelector("svg");
      if (!svg) return;
      const viewportRect = viewport.getBoundingClientRect();
      const graphRect = svg.getBoundingClientRect();
      const rawWidth = graphRect.width / scale;
      const rawHeight = graphRect.height / scale;
      scale = Math.min(1.4, Math.max(0.25, Math.min((viewportRect.width - 80) / rawWidth, (viewportRect.height - 80) / rawHeight)));
      translateX = 28;
      translateY = 28;
      applyTransform();
    }}

    function downloadText(filename, content, type) {{
      const blob = new Blob([content], {{ type }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }}

    const rendered = await mermaid.render("rendered-graph", source);
    canvas.innerHTML = rendered.svg;
    applyTransform();
    requestAnimationFrame(fitToView);

    document.getElementById("zoom-in").addEventListener("click", () => {{
      scale = Math.min(3, scale + 0.1);
      applyTransform();
    }});
    document.getElementById("zoom-out").addEventListener("click", () => {{
      scale = Math.max(0.15, scale - 0.1);
      applyTransform();
    }});
    document.getElementById("fit").addEventListener("click", fitToView);
    document.getElementById("reset").addEventListener("click", () => {{
      scale = 1;
      translateX = 0;
      translateY = 0;
      applyTransform();
    }});
    document.getElementById("download-mmd").addEventListener("click", () => {{
      downloadText(`${{title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "graph"}}.mmd`, source, "text/plain");
    }});
    document.getElementById("download-svg").addEventListener("click", () => {{
      const svg = canvas.querySelector("svg");
      if (!svg) return;
      downloadText(`${{title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "graph"}}.svg`, new XMLSerializer().serializeToString(svg), "image/svg+xml");
    }});

    viewport.addEventListener("pointerdown", (event) => {{
      dragging = true;
      viewport.classList.add("dragging");
      startX = event.clientX - translateX;
      startY = event.clientY - translateY;
      viewport.setPointerCapture(event.pointerId);
    }});
    viewport.addEventListener("pointermove", (event) => {{
      if (!dragging) return;
      translateX = event.clientX - startX;
      translateY = event.clientY - startY;
      applyTransform();
    }});
    viewport.addEventListener("pointerup", () => {{
      dragging = false;
      viewport.classList.remove("dragging");
    }});
    viewport.addEventListener("wheel", (event) => {{
      event.preventDefault();
      const delta = event.deltaY < 0 ? 0.08 : -0.08;
      scale = Math.max(0.15, Math.min(3, scale + delta));
      applyTransform();
    }}, {{ passive: false }});
  </script>
</body>
</html>"""


@app.get("/graph", response_class=HTMLResponse)
async def graph_page():
    return graph_viewer_page("Chat Graph", "Base checkpointed chat workflow.", graph_mermaid(), "/graph/mermaid")


@app.get("/research/graph", response_class=HTMLResponse)
async def research_graph_page():
    return graph_viewer_page("Research Graph", "Structured research workflow with memory, critique, repair, review, and citations.", research_mermaid(), "/research/graph/mermaid")


@app.get("/fortigate/graph", response_class=HTMLResponse)
async def fortigate_graph_page():
    return graph_viewer_page("FortiGate Provisioning Graph", "Artifact-only FortiGate design, validation, standards checks, judge review, and package finalization.", fortigate_mermaid(), "/fortigate/graph/mermaid")


@app.get("/network-design/graph", response_class=HTMLResponse)
async def network_design_graph_page():
    return graph_viewer_page("Network Design Helper Graph", "Standards-aware design chat workflow that creates a package and FortiGate handoff.", network_design_mermaid(), "/network-design/graph/mermaid")


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


def _fortigate_config(thread_id: str, mode: str = "artifact") -> dict[str, Any]:
    config = {
        "configurable": {"thread_id": f"fortigate:{thread_id}"},
        "metadata": {
            "langfuse_trace_name": "fortigate-provisioning-agent",
            "langfuse_session_id": thread_id,
            "langfuse_user_id": "local-dev",
            "fortigate_mode": mode,
        },
    }
    if app.state.langfuse_handler is not None:
        config["callbacks"] = [app.state.langfuse_handler]
    return config


def _network_design_config(thread_id: str, run_name: str = "") -> dict[str, Any]:
    config = {
        "configurable": {"thread_id": f"network-design:{thread_id}"},
        "metadata": {
            "langfuse_trace_name": "network-design-helper",
            "langfuse_session_id": thread_id,
            "langfuse_user_id": "local-dev",
        },
    }
    if run_name:
        config["run_name"] = run_name
    if app.state.langfuse_handler is not None:
        config["callbacks"] = [app.state.langfuse_handler]
    return config


def _network_design_chat_query(request: NetworkDesignRequest) -> str:
    message_text = " ".join(message.content for message in request.messages[-12:])
    parts = [
        request.intake.customer_name,
        request.intake.site_name,
        request.intake.design_goal,
        request.intake.business_context,
        request.intake.constraints,
        message_text,
    ]
    return " ".join(part for part in parts if part).strip() or "network design fortigate sd-wan firewall standards"


def _network_design_standards_payload(standards: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": item.chunk_id,
            "document": item.document,
            "topic": item.topic,
            "text": item.text[:1200],
        }
        for item in standards[:8]
    ]


NETWORK_DESIGN_CRITICAL_FIELDS = [
    {
        "field": "site_and_goal",
        "label": "Site and design goal",
        "question": "What site are we designing for, and what is the main business goal for this network design?",
    },
    {
        "field": "wan",
        "label": "WAN circuits and failover intent",
        "question": "What WAN circuits should we plan for, including provider, bandwidth, addressing type, and primary/backup or load-sharing preference?",
    },
    {
        "field": "lan",
        "label": "VLANs, subnets, and DHCP",
        "question": "What VLANs or LAN segments should be included, with subnet, gateway, and DHCP expectations for each?",
    },
    {
        "field": "security",
        "label": "Security zones and traffic intent",
        "question": "What security zones are needed, and what traffic should be allowed or blocked between them?",
    },
    {
        "field": "routing_sdwan",
        "label": "Routing and SD-WAN behavior",
        "question": "What routing or SD-WAN behavior should we design for, including failover, application steering, static routes, or dynamic routing?",
    },
    {
        "field": "remote_access",
        "label": "Remote access and VPN",
        "question": "Do you need remote access or site-to-site VPN, and who or what networks should be allowed to use it?",
    },
    {
        "field": "operations",
        "label": "Logging, monitoring, change, and rollback",
        "question": "What logging, monitoring, change window, approval, and rollback expectations should be included?",
    },
    {
        "field": "fortigate_target",
        "label": "FortiGate model and FortiOS target",
        "question": "Which FortiGate model and FortiOS version should this design target?",
    },
]


def _normalize_critical_status(value: Any) -> str:
    status = str(value or "missing").strip().lower()
    return status if status in {"missing", "partial", "complete"} else "missing"


def _network_design_field_alias(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _guided_intake_field_updates(request: NetworkDesignRequest) -> dict[str, dict[str, Any]]:
    aliases: dict[str, str] = {}
    for spec in NETWORK_DESIGN_CRITICAL_FIELDS:
        aliases[_network_design_field_alias(spec["field"])] = spec["field"]
        aliases[_network_design_field_alias(spec["label"])] = spec["field"]

    updates: dict[str, dict[str, Any]] = {}
    for message in request.messages:
        if message.role != "user":
            continue
        for raw_line in message.content.splitlines():
            line = raw_line.strip().lstrip("-*").strip()
            if ":" not in line:
                continue
            label, _, answer = line.partition(":")
            field = aliases.get(_network_design_field_alias(label))
            summary = answer.strip()
            if not field or not summary:
                continue
            updates[field] = {
                "status": "complete",
                "summary": summary[:500],
                "evidence": [line[:500]],
            }
    return updates


def _normalize_network_design_readiness(
    request: NetworkDesignRequest,
    structured_intake: dict[str, Any],
    readiness_report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    structured = structured_intake if isinstance(structured_intake, dict) else {}
    readiness = readiness_report if isinstance(readiness_report, dict) else {}
    structured_fields = structured.get("critical_fields") if isinstance(structured.get("critical_fields"), dict) else {}
    readiness_fields = readiness.get("critical_fields") if isinstance(readiness.get("critical_fields"), dict) else {}
    guided_fields = _guided_intake_field_updates(request)
    # Layer prior critical_fields as the base so the intake LLM can return only the fields it
    # changed this turn (deltas) without losing previously completed answers.
    prior_intake = request.structured_intake if isinstance(request.structured_intake, dict) else {}
    prior_fields = prior_intake.get("critical_fields") if isinstance(prior_intake.get("critical_fields"), dict) else {}
    raw_fields = {**prior_fields, **readiness_fields, **structured_fields, **guided_fields}
    critical_fields: dict[str, dict[str, Any]] = {}

    for spec in NETWORK_DESIGN_CRITICAL_FIELDS:
        field = spec["field"]
        raw = raw_fields.get(field) if isinstance(raw_fields.get(field), dict) else {}
        status = _normalize_critical_status(raw.get("status"))
        summary = str(raw.get("summary") or "").strip()
        evidence = [str(item).strip() for item in raw.get("evidence", []) if str(item).strip()] if isinstance(raw.get("evidence"), list) else []

        if field == "site_and_goal" and request.intake.site_name.strip() and request.intake.design_goal.strip():
            status = "complete"
            summary = summary or f"{request.intake.site_name}: {request.intake.design_goal}"
        if field == "fortigate_target" and request.intake.preferred_fortigate_model.strip() and request.intake.fortios_version.strip():
            status = "complete"
            summary = summary or f"{request.intake.preferred_fortigate_model} on FortiOS {request.intake.fortios_version}"

        critical_fields[field] = {
            "field": field,
            "label": spec["label"],
            "status": status,
            "summary": summary,
            "evidence": evidence[:5],
            "question": str(raw.get("question") or spec["question"]).strip(),
        }

    missing = [spec["field"] for spec in NETWORK_DESIGN_CRITICAL_FIELDS if critical_fields[spec["field"]]["status"] != "complete"]
    next_question = ""
    if missing:
        next_question = critical_fields[missing[0]]["question"]

    assumptions = []
    for source in (structured.get("assumptions"), readiness.get("assumptions")):
        if isinstance(source, list):
            assumptions.extend(str(item).strip() for item in source if str(item).strip())
    assumptions = list(dict.fromkeys(assumptions))[:12]
    complete_count = len(NETWORK_DESIGN_CRITICAL_FIELDS) - len(missing)

    structured = {
        **prior_intake,
        **structured,
        "critical_fields": critical_fields,
        "assumptions": assumptions,
        "open_questions": [next_question] if next_question else [],
    }
    normalized_readiness = NetworkDesignReadinessReport(
        readiness_score=round(complete_count / len(NETWORK_DESIGN_CRITICAL_FIELDS), 2),
        ready_for_package=not missing,
        missing_critical_fields=missing,
        next_question=next_question,
        critical_fields=critical_fields,
        assumptions=assumptions,
    ).model_dump()
    return structured, normalized_readiness


def _network_design_gate_detail(readiness_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "message": "Network design package generation is blocked until the required intake details are complete.",
        "missing_critical_fields": readiness_report.get("missing_critical_fields", []),
        "next_question": readiness_report.get("next_question", ""),
        "readiness_report": readiness_report,
    }


def _ensure_network_design_package_ready(request: NetworkDesignRequest) -> None:
    structured_intake, readiness_report = _normalize_network_design_readiness(
        request,
        request.structured_intake,
        request.readiness_report,
    )
    request.structured_intake = structured_intake
    request.readiness_report = readiness_report
    if not readiness_report.get("ready_for_package"):
        raise HTTPException(status_code=409, detail=_network_design_gate_detail(readiness_report))


def _network_design_fallback_reply(request: NetworkDesignRequest, standards: list[Any]) -> str:
    standards_note = ""
    if standards:
        standards_note = f"\n\nI found {len(standards)} relevant standards/documentation chunks and will use them when we generate the package."
    site = request.intake.site_name or "this site"
    return (
        f"Let's work through the network design for {site}. I have the high-level goal. "
        "Before generating the design package, I need a few details:\n\n"
        "1. WAN circuits: provider, bandwidth, static/DHCP addressing, gateways, and primary/backup or load-sharing preference.\n"
        "2. LAN networks: VLAN names, subnets, gateway IPs, DHCP expectations, and any voice/guest/corp separation.\n"
        "3. Security zones and policy intent: what each zone can reach, especially guest-to-corporate restrictions.\n"
        "4. Routing and SD-WAN: performance SLA targets, preferred apps, failover behavior, and any BGP/static routing needs.\n"
        "5. Operations: logging destination, monitoring/SNMP, change window, rollback expectation, and who approves the design.\n\n"
        "Answer any subset of those, and I will keep refining the design conversation."
        f"{standards_note}"
    )


def _looks_off_topic_network_design(text: str) -> bool:
    lowered = text.lower()
    network_terms = [
        "wan",
        "lan",
        "sd-wan",
        "firewall",
        "fortigate",
        "vlan",
        "subnet",
        "routing",
        "logging",
        "design",
        "network",
        "security",
    ]
    off_topic_terms = ["nba", "nhl", "playoff", "stanley cup", "espn"]
    return any(term in lowered for term in off_topic_terms) and not any(term in lowered for term in network_terms)


def _has_off_topic_network_design_reference(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in ["nba", "nhl", "playoff", "stanley cup", "espn"])


def _readiness_network_design_reply(readiness_report: dict[str, Any]) -> str:
    if readiness_report.get("ready_for_package"):
        return "The required intake is complete. You can generate the design package and FortiGate handoff."
    question = str(readiness_report.get("next_question") or "").strip()
    if question:
        return f"I captured the latest design details.\n\n{question}"
    return "I captured the latest design details. Please add the next required network design detail so we can complete the intake."


def _clean_network_design_reply(text: str) -> str:
    cleaned = re.sub(
        r"^\s*I cannot fulfill the request to search for sports schedules\.\s*"
        r"This conversation is strictly limited to network design\.\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned or text


async def structure_network_design_intake(
    request: NetworkDesignRequest,
    assistant_reply: str,
    standards: list[Any],
    thread_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prior = request.structured_intake if isinstance(request.structured_intake, dict) else {}
    # Deterministic-first: if the form fields, guided answers, and prior structured state already
    # satisfy every critical field, skip the intake LLM entirely.
    pre_structured, pre_readiness = _normalize_network_design_readiness(
        request,
        prior,
        request.readiness_report if isinstance(request.readiness_report, dict) else {},
    )
    if pre_readiness.get("ready_for_package"):
        return pre_structured, pre_readiness
    transcript = "\n".join(f"{message.role}: {message.content}" for message in request.messages[-20:])
    try:
        response = await app.state.network_design_intake_model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "Return only JSON with key structured_intake. Maintain a concise structured state for a "
                        "FortiGate/network design conversation. Return critical_fields as a DELTA: include only the "
                        "critical fields whose status or summary changed because of the latest user message; previously "
                        "captured fields are preserved automatically, so do not repeat unchanged fields. structured_intake "
                        "may also include intake_updates, known_requirements, assumptions, constraints, and evidence_notes. "
                        "critical_fields keys must come from this exact set: site_and_goal, wan, lan, security, "
                        "routing_sdwan, remote_access, operations, fortigate_target. Each returned critical field must "
                        "include field, label, status, summary, evidence, and question. status must be missing, partial, or "
                        "complete. intake_updates may only use NetworkDesignIntake keys: customer_name, site_name, "
                        "design_goal, business_context, constraints, preferred_fortigate_model, fortios_version. Do not "
                        "return a readiness_report; readiness is computed deterministically. Do not invent site-specific values."
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        {
                            "thread_id": thread_id,
                            "form_intake": request.intake.model_dump(),
                            "prior_structured_intake": prior,
                            "required_field_priority": NETWORK_DESIGN_CRITICAL_FIELDS,
                            "conversation": transcript,
                            "assistant_reply": assistant_reply,
                            "standards": _network_design_standards_payload(standards),
                        },
                        indent=2,
                    )
                ),
            ],
            config=_network_design_config(f"{thread_id}:intake", run_name="network_design_intake"),
        )
        data = _extract_json_object(str(response.content))
    except Exception:
        data = {}
    structured = data.get("structured_intake") if isinstance(data.get("structured_intake"), dict) else prior
    return _normalize_network_design_readiness(request, structured or {}, {})


@app.post("/network-design/message", response_model=NetworkDesignChatResponse)
async def network_design_message(request: NetworkDesignRequest):
    thread_id = request.thread_id or str(uuid4())
    standards = search_standards(_network_design_chat_query(request), limit=8)
    transcript = "\n".join(f"{message.role}: {message.content}" for message in request.messages[-16:])
    structured_intake, readiness_report = await structure_network_design_intake(request, "", standards, thread_id)
    if readiness_report.get("ready_for_package"):
        return NetworkDesignChatResponse(
            thread_id=thread_id,
            model=getattr(app.state, "network_chat_model_name", app.state.model_name),
            message=NetworkDesignMessage(role="assistant", content=_readiness_network_design_reply(readiness_report)),
            standards=standards,
            structured_intake=structured_intake,
            readiness_report=readiness_report,
        )
    try:
        result = await app.state.network_design_chat_model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are the Network Design Helper, a standards-aware assistant for network design engineers. "
                        "Keep the conversation focused on network design, site requirements, standards, and FortiGate handoff details. "
                        "Only redirect if the latest user message is clearly unrelated to network design; never tell the user to stay focused when they provide valid network details. "
                        "Have a normal conversational back-and-forth, but collect required design details one question at a time. "
                        "If readiness_report.ready_for_package is false, ask exactly the readiness_report.next_question and no other questions. "
                        "Acknowledge any newly supplied detail briefly before asking the next question. "
                        "Use the provided standards context when relevant, but do not over-cite. Do not claim any device changes were made. "
                        "Do not tell the user to generate the design package unless readiness_report.ready_for_package is true. "
                        "When ready_for_package is true, say the intake is complete and they can generate the design package and FortiGate handoff. "
                        "Your response must directly address the current network design request."
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        {
                            "intake": request.intake.model_dump(),
                            "structured_intake": structured_intake,
                            "readiness_report": readiness_report,
                            "conversation": transcript,
                            "standards": _network_design_standards_payload(standards),
                        },
                        indent=2,
                    )
                ),
            ],
            config=_network_design_config(thread_id, run_name="network_design_chat"),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if app.state.langfuse is not None:
        app.state.langfuse.flush()
    content = _clean_network_design_reply(str(result.content))
    if _looks_off_topic_network_design(content) or _has_off_topic_network_design_reference(content):
        content = _readiness_network_design_reply(readiness_report)
    if not readiness_report.get("ready_for_package") and readiness_report.get("next_question"):
        question = str(readiness_report["next_question"]).strip()
        if question and question.lower() not in content.lower():
            content = f"{content.rstrip()}\n\n{question}"
    return NetworkDesignChatResponse(
        thread_id=thread_id,
        model=getattr(app.state, "network_chat_model_name", app.state.model_name),
        message=NetworkDesignMessage(role="assistant", content=content),
        standards=standards,
        structured_intake=structured_intake,
        readiness_report=readiness_report,
    )


async def _generate_network_design_run(request: NetworkDesignRequest, thread_id: str) -> NetworkDesignRunResponse:
    _ensure_network_design_package_ready(request)
    timeout_seconds = float(getattr(app.state, "network_design_timeout_seconds", 1200.0) or 0)
    invoke = app.state.network_design_graph.ainvoke(
        {
            "intake": request.intake.model_dump(),
            "structured_intake": request.structured_intake,
            "readiness_report": request.readiness_report,
            "messages": [message.model_dump() for message in request.messages],
        },
        config=_network_design_config(thread_id),
    )
    try:
        result = await asyncio.wait_for(invoke, timeout=timeout_seconds) if timeout_seconds > 0 else await invoke
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail=f"Network design exceeded {int(timeout_seconds)} seconds") from exc
    if app.state.langfuse is not None:
        app.state.langfuse.flush()
    response = network_design_response_from_state(result, thread_id, app.state.network_package_model_name)
    save_network_design_run(response)
    return response


async def _run_network_design_job(job_id: str, design_request: NetworkDesignRequest, thread_id: str) -> None:
    semaphore = getattr(app.state, "network_design_job_semaphore", None)
    try:
        if semaphore is None:
            update_network_design_job(job_id, status="running")
            response = await _generate_network_design_run(design_request, thread_id)
        else:
            async with semaphore:
                update_network_design_job(job_id, status="running")
                response = await _generate_network_design_run(design_request, thread_id)
        update_network_design_job(job_id, status="completed", run_id=response.run_id)
    except asyncio.CancelledError:
        update_network_design_job(job_id, status="cancelled", error="Job cancelled before completion.")
        raise
    except HTTPException as exc:
        update_network_design_job(job_id, status="failed", error=str(exc.detail))
    except Exception as exc:
        update_network_design_job(job_id, status="failed", error=str(exc))


@app.post("/network-design/chat/jobs", response_model=NetworkDesignJobResponse, status_code=202)
async def network_design_chat_job_create(request: NetworkDesignRequest):
    _ensure_network_design_package_ready(request)
    thread_id = request.thread_id or str(uuid4())
    now = _utc_now()
    job = NetworkDesignJobResponse(
        job_id=str(uuid4()),
        thread_id=thread_id,
        status="queued",
        created_at=now,
        updated_at=now,
    )
    save_network_design_job(job)
    task = asyncio.create_task(_run_network_design_job(job.job_id, request, thread_id))
    getattr(app.state, "network_design_job_tasks", {})[job.job_id] = task
    task.add_done_callback(lambda _: getattr(app.state, "network_design_job_tasks", {}).pop(job.job_id, None))
    return job


@app.get("/network-design/chat/jobs/{job_id}", response_model=NetworkDesignJobResponse)
async def network_design_chat_job(job_id: str):
    return load_network_design_job(job_id)


@app.post("/network-design/chat", response_model=NetworkDesignRunResponse)
async def network_design_chat(request: NetworkDesignRequest, client_request: Request):
    _ensure_network_design_package_ready(request)
    thread_id = request.thread_id or str(uuid4())
    try:
        return await _run_with_client_disconnect_cancel(_generate_network_design_run(request, thread_id), client_request)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/network-design/standards/search", response_model=NetworkDesignStandardsSearchResponse)
async def network_design_standards_search(q: str, limit: int = 10):
    results = search_standards(q.strip(), limit=max(1, min(limit, 25)))
    return NetworkDesignStandardsSearchResponse(query=q, results=results)


@app.get("/network-design/runs", response_model=list[NetworkDesignRunSummary])
async def network_design_runs(limit: int = 20):
    return list_network_design_runs(limit=max(1, min(limit, 100)))


@app.get("/network-design/runs/{run_id}", response_model=NetworkDesignRunResponse)
async def network_design_run(run_id: str):
    return load_network_design_run(run_id)


async def _generate_fortigate_run(request: FortiGateDesignRequest, thread_id: str) -> FortiGateRunResponse:
    timeout_seconds = float(getattr(app.state, "fortigate_design_timeout_seconds", 1800.0) or 0)
    initial_state: dict[str, Any] = {
        "intake": request.intake.model_dump(),
        "existing_config": request.existing_config,
    }
    if request.seed_standards:
        initial_state["standards"] = request.seed_standards
    invoke = app.state.fortigate_graph.ainvoke(
        initial_state,
        config=_fortigate_config(thread_id, request.mode),
    )
    try:
        result = await asyncio.wait_for(invoke, timeout=timeout_seconds) if timeout_seconds > 0 else await invoke
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail=f"FortiGate design exceeded {int(timeout_seconds)} seconds") from exc
    if app.state.langfuse is not None:
        app.state.langfuse.flush()
    response = fortigate_response_from_state(result, thread_id, app.state.model_name)
    save_fortigate_run(response)
    return response


async def _run_with_client_disconnect_cancel(task_coro: Any, request: Request) -> Any:
    task = asyncio.create_task(task_coro)
    try:
        while not task.done():
            if await request.is_disconnected():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                raise HTTPException(status_code=499, detail="Client disconnected; FortiGate generation cancelled")
            await asyncio.sleep(1)
        return await task
    finally:
        if not task.done():
            task.cancel()


async def _run_fortigate_design_job(job_id: str, design_request: FortiGateDesignRequest, thread_id: str) -> None:
    semaphore = getattr(app.state, "fortigate_design_job_semaphore", None)
    try:
        if semaphore is None:
            update_fortigate_design_job(job_id, status="running")
            response = await _generate_fortigate_run(design_request, thread_id)
        else:
            async with semaphore:
                update_fortigate_design_job(job_id, status="running")
                response = await _generate_fortigate_run(design_request, thread_id)
        update_fortigate_design_job(job_id, status="completed", run_id=response.run_id)
    except asyncio.CancelledError:
        update_fortigate_design_job(job_id, status="cancelled", error="Job cancelled before completion.")
        raise
    except HTTPException as exc:
        update_fortigate_design_job(job_id, status="failed", error=str(exc.detail))
    except Exception as exc:
        update_fortigate_design_job(job_id, status="failed", error=str(exc))


@app.post("/fortigate/design/jobs", response_model=FortiGateDesignJobResponse, status_code=202)
async def fortigate_design_job_create(request: FortiGateDesignRequest):
    thread_id = request.thread_id or str(uuid4())
    now = _utc_now()
    job = FortiGateDesignJobResponse(
        job_id=str(uuid4()),
        thread_id=thread_id,
        status="queued",
        created_at=now,
        updated_at=now,
    )
    save_fortigate_design_job(job)
    task = asyncio.create_task(_run_fortigate_design_job(job.job_id, request, thread_id))
    getattr(app.state, "fortigate_design_job_tasks", {})[job.job_id] = task
    task.add_done_callback(lambda _: getattr(app.state, "fortigate_design_job_tasks", {}).pop(job.job_id, None))
    return job


@app.get("/fortigate/design/jobs/{job_id}", response_model=FortiGateDesignJobResponse)
async def fortigate_design_job(job_id: str):
    return load_fortigate_design_job(job_id)


@app.post("/fortigate/design", response_model=FortiGateRunResponse)
async def fortigate_design(request: FortiGateDesignRequest, client_request: Request):
    thread_id = request.thread_id or str(uuid4())
    try:
        return await _run_with_client_disconnect_cancel(_generate_fortigate_run(request, thread_id), client_request)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/fortigate/interactive", response_model=FortiGateInteractiveResponse)
async def fortigate_interactive(request: FortiGateDesignRequest):
    thread_id = request.thread_id or str(uuid4())
    checkpoint_thread_id = f"fortigate:{thread_id}"
    try:
        interactive_initial_state: dict[str, Any] = {"intake": request.intake.model_dump(), "existing_config": request.existing_config}
        if request.seed_standards:
            interactive_initial_state["standards"] = request.seed_standards
        result = await app.state.interactive_fortigate_graph.ainvoke(
            interactive_initial_state,
            config=_fortigate_config(thread_id, request.mode),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if app.state.langfuse is not None:
        app.state.langfuse.flush()
    review = _interrupt_payload(result)
    if review and review.get("stage") == "fortigate_clarification":
        return FortiGateInteractiveResponse(
            status="awaiting_clarification",
            thread_id=thread_id,
            checkpoint_thread_id=checkpoint_thread_id,
            questions=review.get("questions", []),
            run=None,
        )
    if review and review.get("stage") == "fortigate_input_review":
        return FortiGateInteractiveResponse(
            status="awaiting_input_review",
            thread_id=thread_id,
            checkpoint_thread_id=checkpoint_thread_id,
            questions=[],
            input_review=review.get("items", []),
            run=None,
        )
    if review:
        return FortiGateInteractiveResponse(
            status="awaiting_review",
            thread_id=thread_id,
            checkpoint_thread_id=checkpoint_thread_id,
            questions=[],
            run=None,
        )
    response = fortigate_response_from_state(result, thread_id, app.state.model_name)
    save_fortigate_run(response)
    return FortiGateInteractiveResponse(status="completed", thread_id=thread_id, checkpoint_thread_id=checkpoint_thread_id, questions=[], run=response)


@app.post("/fortigate/interactive/{thread_id}/resume", response_model=FortiGateInteractiveResponse)
async def fortigate_interactive_resume(thread_id: str, payload: dict[str, Any]):
    checkpoint_thread_id = f"fortigate:{thread_id}"
    try:
        result = await app.state.interactive_fortigate_graph.ainvoke(Command(resume=payload), config=_fortigate_config(thread_id, "interactive"))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if app.state.langfuse is not None:
        app.state.langfuse.flush()
    review = _interrupt_payload(result)
    if review and review.get("stage") == "fortigate_clarification":
        return FortiGateInteractiveResponse(status="awaiting_clarification", thread_id=thread_id, checkpoint_thread_id=checkpoint_thread_id, questions=review.get("questions", []), run=None)
    if review and review.get("stage") == "fortigate_input_review":
        return FortiGateInteractiveResponse(status="awaiting_input_review", thread_id=thread_id, checkpoint_thread_id=checkpoint_thread_id, questions=[], input_review=review.get("items", []), run=None)
    if review:
        return FortiGateInteractiveResponse(status="awaiting_review", thread_id=thread_id, checkpoint_thread_id=checkpoint_thread_id, questions=[], run=None)
    response = fortigate_response_from_state(result, thread_id, app.state.model_name)
    save_fortigate_run(response)
    return FortiGateInteractiveResponse(status="completed", thread_id=thread_id, checkpoint_thread_id=checkpoint_thread_id, questions=[], run=response)


@app.post("/fortigate/configs/parse", response_model=FortiGateConfigSummary)
async def fortigate_config_parse(request: FortiGateConfigParseRequest):
    return parse_fortigate_config(request.config_text)


@app.post("/fortigate/changes/analyze")
async def fortigate_change_analyze(request: FortiGateChangeAnalysisRequest):
    summary = parse_fortigate_config(request.current_config)
    lower_change = request.requested_change.lower()
    impacted = {
        "interfaces": [item for item in summary.interfaces if item.get("name", "").lower() in lower_change or item.get("alias", "").lower() in lower_change],
        "policies": [item for item in summary.firewall_policies if item.get("name", "").lower() in lower_change or item.get("id", "").lower() in lower_change],
        "address_objects": [item for item in summary.address_objects if item.get("name", "").lower() in lower_change],
    }
    return {
        "requested_change": request.requested_change,
        "current_config_summary": summary,
        "impacted": impacted,
        "requires_rollback_plan": True,
        "safety": "Analysis only. No device changes were made.",
    }


@app.post("/fortigate/runs/{run_id}/judge", response_model=FortiGateRunResponse)
async def fortigate_run_judge(run_id: str):
    run = load_fortigate_run(run_id)
    judge = await run_frontier_judge(
        app.state.fortigate_judge_model,
        _fortigate_judge_packet(run),
        judge_model_name=app.state.fortigate_judge_model_name,
    )
    run.judge_report = judge
    run.review_questions = build_fortigate_review_questions(judge)
    run.status = "approved_artifact" if judge.verdict == "pass" else ("blocked" if judge.verdict == "block" else "needs_review")
    run.markdown = render_fortigate_markdown(run)
    save_fortigate_run(run)
    return run


@app.post("/fortigate/standards/ingest", response_model=FortiGateStandardsIngestResponse)
async def fortigate_standards_ingest(request: FortiGateStandardsIngestRequest):
    result = ingest_standards(request.source_dir, DEFAULT_INDEX_PATH)
    return FortiGateStandardsIngestResponse(**result)


@app.get("/fortigate/standards/search", response_model=FortiGateStandardsSearchResponse)
async def fortigate_standards_search(q: str, limit: int = 10):
    results = search_standards(q.strip(), limit=max(1, min(limit, 25)))
    return FortiGateStandardsSearchResponse(query=q, results=results)


@app.get("/fortigate/runs", response_model=list[FortiGateRunSummary])
async def fortigate_runs(limit: int = 20):
    return list_fortigate_runs(limit=max(1, min(limit, 100)))


@app.get("/fortigate/runs/{run_id}", response_model=FortiGateRunResponse)
async def fortigate_run(run_id: str):
    return load_fortigate_run(run_id)


@app.post("/fortigate/runs/{run_id}/review", response_model=FortiGateRunResponse)
async def fortigate_run_review(run_id: str, request: FortiGateReviewRequest):
    run = load_fortigate_run(run_id)
    review = FortiGateHumanReview(
        decision=request.decision,
        reviewer_notes=request.reviewer_notes,
        selected_issues=request.selected_issues,
        answers=request.answers,
        reviewed_at=datetime.now(timezone.utc).isoformat(),
    )
    run.human_review = review
    if request.decision == "rejected":
        run.status = "rejected"
    elif request.answers or request.decision == "needs_work":
        await _revise_fortigate_config_from_review(run, review)
        judge = await run_frontier_judge(
            app.state.fortigate_judge_model,
            _fortigate_judge_packet(run),
            judge_model_name=app.state.fortigate_judge_model_name,
        )
        run.judge_report = judge
        run.review_questions = [] if judge.verdict == "pass" else build_fortigate_review_questions(judge)
        run.status = "approved_artifact" if judge.verdict == "pass" else ("blocked" if judge.verdict == "block" else "needs_review")
    elif request.decision == "approved" and run.judge_report.verdict == "pass":
        run.status = "approved_artifact"
        run.review_questions = []
    else:
        run.status = "needs_review"
        run.review_questions = build_fortigate_review_questions(run.judge_report)
    run.markdown = render_fortigate_markdown(run)
    save_fortigate_run(run)
    return run


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
