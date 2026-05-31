from typing import Any, Optional

from pydantic import BaseModel, Field

from app.fortigate_models import FortiGateStandardChunk


class NetworkDesignMessage(BaseModel):
    role: str = Field(default="user", pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)


class NetworkDesignIntake(BaseModel):
    customer_name: str = ""
    site_name: str = ""
    design_goal: str = Field(..., min_length=1)
    business_context: str = ""
    constraints: str = ""
    preferred_fortigate_model: str = ""
    fortios_version: str = ""


class NetworkDesignRequest(BaseModel):
    intake: NetworkDesignIntake
    structured_intake: dict[str, Any] = Field(default_factory=dict)
    readiness_report: dict[str, Any] = Field(default_factory=dict)
    messages: list[NetworkDesignMessage] = Field(default_factory=list)
    thread_id: Optional[str] = None


class NetworkDesignCriticalField(BaseModel):
    field: str
    label: str
    status: str = Field(default="missing", pattern="^(missing|partial|complete)$")
    summary: str = ""
    evidence: list[str] = Field(default_factory=list)
    question: str = ""


class NetworkDesignReadinessReport(BaseModel):
    readiness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    ready_for_package: bool = False
    missing_critical_fields: list[str] = Field(default_factory=list)
    next_question: str = ""
    critical_fields: dict[str, NetworkDesignCriticalField] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)


class NetworkDesignChatResponse(BaseModel):
    thread_id: str
    model: str
    message: NetworkDesignMessage
    standards: list[FortiGateStandardChunk] = Field(default_factory=list)
    structured_intake: dict[str, Any] = Field(default_factory=dict)
    readiness_report: dict[str, Any] = Field(default_factory=dict)


class NetworkDesignValidationReport(BaseModel):
    passed: bool = True
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)


class StandardRequirement(BaseModel):
    requirement_id: str
    topic: str = "general"
    requirement: str
    source_document: str
    source_chunk_id: str
    source_excerpt: str = ""
    keywords: list[str] = Field(default_factory=list)
    priority: str = "medium"


class ComplianceMatrixItem(BaseModel):
    requirement_id: str
    topic: str = "general"
    requirement: str
    source_document: str
    status: str = "needs_review"
    design_evidence: list[str] = Field(default_factory=list)
    handoff_evidence: list[str] = Field(default_factory=list)
    config_evidence: list[str] = Field(default_factory=list)
    rationale: str = ""


class FortiGateHandoffPayload(BaseModel):
    request_type: str = "new_build"
    site_name: str = ""
    business_intent: str = ""
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


class NetworkDesignRunSummary(BaseModel):
    run_id: str
    created_at: str
    thread_id: str
    customer_name: str
    site_name: str
    status: str
    standards_count: int
    validation_passed: bool


class NetworkDesignJobResponse(BaseModel):
    job_id: str
    thread_id: str
    status: str = Field(default="queued", pattern="^(queued|running|completed|failed|cancelled)$")
    created_at: str
    updated_at: str
    run_id: Optional[str] = None
    error: str = ""


class NetworkDesignRunResponse(BaseModel):
    run_id: str
    created_at: str
    thread_id: str
    model: str
    status: str
    intake: NetworkDesignIntake
    structured_intake: dict[str, Any] = Field(default_factory=dict)
    readiness_report: dict[str, Any] = Field(default_factory=dict)
    messages: list[NetworkDesignMessage] = Field(default_factory=list)
    standards: list[FortiGateStandardChunk] = Field(default_factory=list)
    standard_requirements: list[StandardRequirement] = Field(default_factory=list)
    requirements_summary: dict[str, Any] = Field(default_factory=dict)
    missing_questions: list[str] = Field(default_factory=list)
    design_package: dict[str, Any] = Field(default_factory=dict)
    fortigate_handoff: FortiGateHandoffPayload = Field(default_factory=FortiGateHandoffPayload)
    compliance_matrix: list[ComplianceMatrixItem] = Field(default_factory=list)
    validation_report: NetworkDesignValidationReport = Field(default_factory=NetworkDesignValidationReport)
    execution_trace: list[dict[str, Any]] = Field(default_factory=list)
    markdown: str = ""


class NetworkDesignStandardsSearchResponse(BaseModel):
    query: str
    results: list[FortiGateStandardChunk]
