from typing import Any, Optional

from pydantic import BaseModel, Field


class FortiGateStandardChunk(BaseModel):
    chunk_id: str
    document: str
    topic: str = "general"
    text: str
    source_path: str = ""
    score: int = 0
    retrieval_backend: str = ""
    keyword_rank: Optional[int] = None
    vector_rank: Optional[int] = None
    rrf_score: Optional[float] = None


class FortiGateQuestion(BaseModel):
    field: str
    question: str
    reason: str = ""
    required: bool = True


class FortiGateConfigSummary(BaseModel):
    hostname: str = ""
    interfaces: list[dict[str, Any]] = Field(default_factory=list)
    vlans: list[dict[str, Any]] = Field(default_factory=list)
    static_routes: list[dict[str, Any]] = Field(default_factory=list)
    sdwan: dict[str, Any] = Field(default_factory=dict)
    firewall_policies: list[dict[str, Any]] = Field(default_factory=list)
    address_objects: list[dict[str, Any]] = Field(default_factory=list)
    service_objects: list[dict[str, Any]] = Field(default_factory=list)
    vips: list[dict[str, Any]] = Field(default_factory=list)
    vpn: list[dict[str, Any]] = Field(default_factory=list)
    ha: dict[str, Any] = Field(default_factory=dict)
    raw_line_count: int = 0
    parser_warnings: list[str] = Field(default_factory=list)


class FortiGateIntake(BaseModel):
    request_type: str = Field(default="new_build", description="new_build or modify_existing")
    site_name: str = ""
    business_intent: str = Field(..., min_length=1)
    fortigate_model: str = ""
    fortios_version: str = ""
    wan_circuits: list[dict[str, Any]] = Field(default_factory=list)
    lan_networks: list[dict[str, Any]] = Field(default_factory=list)
    security_zones: list[str] = Field(default_factory=list)
    firewall_policy_intent: str = ""
    nat_requirements: str = ""
    vpn_requirements: str = ""
    logging_requirements: str = ""
    ha_requirements: str = ""
    change_window: str = ""
    rollback_expectations: str = ""
    additional_context: str = ""


class FortiGateDesignRequest(BaseModel):
    intake: FortiGateIntake
    existing_config: str = ""
    thread_id: Optional[str] = None
    mode: str = "artifact"
    # Standards already retrieved upstream (e.g. by the Network Design handoff). When provided,
    # the FortiGate graph reuses them instead of retrieving a third time.
    seed_standards: list[dict[str, Any]] = Field(default_factory=list)


class FortiGateInteractiveResponse(BaseModel):
    status: str
    thread_id: str
    checkpoint_thread_id: str
    questions: list[FortiGateQuestion] = Field(default_factory=list)
    run: Optional["FortiGateRunResponse"] = None


class FortiGateReviewRequest(BaseModel):
    decision: str = Field(..., pattern="^(approved|needs_work|rejected)$")
    reviewer_notes: str = ""
    selected_issues: list[str] = Field(default_factory=list)
    answers: dict[str, str] = Field(default_factory=dict)


class FortiGateReviewQuestion(BaseModel):
    question_id: str
    source: str = ""
    question: str
    context: str = ""
    required: bool = True


class FortiGateJudgeReport(BaseModel):
    verdict: str = "needs_revision"
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    standards_concerns: list[str] = Field(default_factory=list)
    config_risks: list[str] = Field(default_factory=list)
    missing_questions: list[str] = Field(default_factory=list)
    recommended_revisions: list[str] = Field(default_factory=list)
    human_reviewer_focus: list[str] = Field(default_factory=list)
    model: str = ""


class FortiGateValidationReport(BaseModel):
    passed: bool = True
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)


class FortiGateHumanReview(BaseModel):
    decision: str = "pending"
    reviewer_notes: str = ""
    selected_issues: list[str] = Field(default_factory=list)
    answers: dict[str, str] = Field(default_factory=dict)
    interpreted_answers: list[dict[str, Any]] = Field(default_factory=list)
    reviewed_at: str = ""


class FortiGateRunSummary(BaseModel):
    run_id: str
    created_at: str
    thread_id: str
    request_type: str
    site_name: str
    status: str
    judge_verdict: str
    validation_passed: bool


class FortiGateDesignJobResponse(BaseModel):
    job_id: str
    thread_id: str
    status: str = Field(default="queued", pattern="^(queued|running|completed|failed|cancelled)$")
    created_at: str
    updated_at: str
    run_id: Optional[str] = None
    error: str = ""


class FortiGateRunResponse(BaseModel):
    run_id: str
    created_at: str
    thread_id: str
    model: str
    status: str
    intake: FortiGateIntake
    current_config_summary: FortiGateConfigSummary = Field(default_factory=FortiGateConfigSummary)
    standards: list[FortiGateStandardChunk] = Field(default_factory=list)
    missing_questions: list[FortiGateQuestion] = Field(default_factory=list)
    logical_design: dict[str, Any] = Field(default_factory=dict)
    fortigate_design: dict[str, Any] = Field(default_factory=dict)
    implementation_intent: dict[str, Any] = Field(default_factory=dict)
    intent_completeness_report: dict[str, Any] = Field(default_factory=dict)
    change_impact: dict[str, Any] = Field(default_factory=dict)
    config_sections: dict[str, Any] = Field(default_factory=dict)
    config_artifacts: dict[str, Any] = Field(default_factory=dict)
    section_validation_report: dict[str, Any] = Field(default_factory=dict)
    cli_completeness_report: dict[str, Any] = Field(default_factory=dict)
    validation_report: FortiGateValidationReport = Field(default_factory=FortiGateValidationReport)
    standards_report: FortiGateValidationReport = Field(default_factory=FortiGateValidationReport)
    risk_report: FortiGateValidationReport = Field(default_factory=FortiGateValidationReport)
    autonomous_repair_iterations: int = 0
    auto_fixed_items: list[str] = Field(default_factory=list)
    requires_human_input: list[str] = Field(default_factory=list)
    judge_report: FortiGateJudgeReport = Field(default_factory=FortiGateJudgeReport)
    review_questions: list[FortiGateReviewQuestion] = Field(default_factory=list)
    human_review: Optional[FortiGateHumanReview] = None
    execution_trace: list[dict[str, Any]] = Field(default_factory=list)
    markdown: str = ""


class FortiGateStandardsIngestRequest(BaseModel):
    source_dir: str = "/data/fortigate-standards/raw"
    rebuild: bool = True


class FortiGateStandardsIngestResponse(BaseModel):
    source_dir: str
    index_path: str
    document_count: int
    chunk_count: int
    redis_index: dict[str, Any] = Field(default_factory=dict)


class FortiGateStandardsSearchResponse(BaseModel):
    query: str
    results: list[FortiGateStandardChunk]
