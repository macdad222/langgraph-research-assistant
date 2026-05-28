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
    messages: list[NetworkDesignMessage] = Field(default_factory=list)
    thread_id: Optional[str] = None


class NetworkDesignChatResponse(BaseModel):
    thread_id: str
    model: str
    message: NetworkDesignMessage
    standards: list[FortiGateStandardChunk] = Field(default_factory=list)


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


class NetworkDesignRunResponse(BaseModel):
    run_id: str
    created_at: str
    thread_id: str
    model: str
    status: str
    intake: NetworkDesignIntake
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
