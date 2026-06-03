import json
import re
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import END, StateGraph

from app.network_design_models import NetworkDesignIntake, NetworkDesignMessage, NetworkDesignValidationReport
from app.standards import extract_standard_requirements, search_standards


class NetworkDesignState(TypedDict, total=False):
    intake: dict[str, Any]
    structured_intake: dict[str, Any]
    readiness_report: dict[str, Any]
    messages: list[dict[str, Any]]
    standards_query: str
    standards_query_used: str
    standards: list[dict[str, Any]]
    standard_requirements: list[dict[str, Any]]
    requirements_summary: dict[str, Any]
    missing_questions: list[str]
    design_package: dict[str, Any]
    fortigate_handoff: dict[str, Any]
    compliance_matrix: list[dict[str, Any]]
    validation_report: dict[str, Any]
    status: str
    execution_trace: list[dict[str, Any]]


def _trace_start() -> tuple[str, float]:
    return datetime.now(UTC).isoformat(), perf_counter()


def _trace_update(
    state: NetworkDesignState,
    updates: NetworkDesignState,
    node: str,
    started_at: str,
    started_perf: float,
    summary: str,
    branch: str = "",
) -> NetworkDesignState:
    item = {
        "node": node,
        "started_at": started_at,
        "duration_ms": int((perf_counter() - started_perf) * 1000),
        "summary": summary,
        "branch": branch,
    }
    return {**updates, "execution_trace": [*state.get("execution_trace", []), item]}


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(stripped[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    end = stripped.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return {}


def _conversation_text(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages[-20:]:
        role = str(message.get("role") or "user")
        content = str(message.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _standards_payload(standards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": item.get("chunk_id"),
            "document": item.get("document"),
            "topic": item.get("topic"),
            "text": item.get("text"),
        }
        for item in standards[:10]
    ]


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _flatten_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2)
    return str(value or "")


def _evidence_snippets(haystack: str, keywords: list[str], limit: int = 3) -> list[str]:
    lines = [line.strip() for line in re.split(r"\n+|(?<=[.!?])\s+", haystack) if line.strip()]
    evidence: list[str] = []
    lowered_keywords = [keyword.lower() for keyword in keywords if len(keyword) > 2]
    for line in lines:
        lower = line.lower()
        if any(keyword in lower for keyword in lowered_keywords):
            evidence.append(line[:320])
        if len(evidence) >= limit:
            break
    return evidence


def build_compliance_matrix(state: NetworkDesignState) -> list[dict[str, Any]]:
    design_text = _flatten_text(state.get("design_package", {}))
    handoff_text = _flatten_text(state.get("fortigate_handoff", {}))
    matrix: list[dict[str, Any]] = []
    for requirement in state.get("standard_requirements", []):
        keywords = [str(item) for item in requirement.get("keywords", [])]
        design_evidence = _evidence_snippets(design_text, keywords)
        handoff_evidence = _evidence_snippets(handoff_text, keywords)
        if design_evidence and handoff_evidence:
            status = "mapped"
            rationale = "Requirement has evidence in both the design package and FortiGate handoff."
        elif design_evidence or handoff_evidence:
            status = "partial"
            rationale = "Requirement has some evidence, but should be reviewed for completeness."
        else:
            status = "needs_review"
            rationale = "No direct keyword evidence found; human review should verify applicability."
        matrix.append(
            {
                "requirement_id": requirement.get("requirement_id", ""),
                "topic": requirement.get("topic", "general"),
                "requirement": requirement.get("requirement", ""),
                "source_document": requirement.get("source_document", ""),
                "status": status,
                "design_evidence": design_evidence,
                "handoff_evidence": handoff_evidence,
                "config_evidence": [],
                "rationale": rationale,
            }
        )
    return matrix


def _fallback_requirements(intake: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "business_goal": intake.get("design_goal", ""),
        "customer": intake.get("customer_name", ""),
        "site": intake.get("site_name", ""),
        "constraints": [intake.get("constraints", "")] if intake.get("constraints") else [],
        "conversation_summary": _conversation_text(messages),
        "network_requirements": [],
        "security_requirements": [],
        "operational_requirements": [],
        "assumptions": ["Generated from limited structured input and chat transcript."],
    }


def _fallback_handoff(intake: dict[str, Any], requirements: dict[str, Any], design: dict[str, Any]) -> dict[str, Any]:
    details = [
        intake.get("design_goal", ""),
        str(requirements.get("conversation_summary") or ""),
        "Design package:",
        json.dumps(design, indent=2)[:6000],
    ]
    return {
        "request_type": "new_build",
        "site_name": intake.get("site_name", ""),
        "business_intent": "\n\n".join(item for item in details if item),
        "fortigate_model": intake.get("preferred_fortigate_model", ""),
        "fortios_version": intake.get("fortios_version", ""),
        "wifi_requirements": [],
        "fortiswitch_requirements": [],
        "system_hardening_requirements": {},
        "admin_access_requirements": {},
        "security_profile_requirements": {},
        "additional_context": "Generated by Network Design Helper. No device changes have been made.",
    }


def validate_network_design(state: NetworkDesignState) -> NetworkDesignValidationReport:
    intake = state.get("intake", {})
    design = state.get("design_package", {})
    handoff = state.get("fortigate_handoff", {})
    blocking: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []

    if not intake.get("design_goal"):
        blocking.append("Design goal is required.")
    else:
        checks.append("Design goal captured.")
    if not design:
        blocking.append("No design package was generated.")
    else:
        checks.append("Design package generated.")
    if not handoff.get("business_intent"):
        blocking.append("FortiGate handoff is missing business intent.")
    else:
        checks.append("FortiGate handoff business intent generated.")
    if not state.get("standards"):
        warnings.append("No standards chunks were retrieved for this conversation.")
    else:
        checks.append(f"Retrieved {len(state.get('standards', []))} standards chunks.")
    if state.get("missing_questions"):
        warnings.append("The design has open clarification questions.")
    if not state.get("standard_requirements"):
        warnings.append("No structured standard requirements were extracted from retrieved standards.")
    else:
        checks.append(f"Extracted {len(state.get('standard_requirements', []))} structured standard requirements.")
    if state.get("compliance_matrix"):
        mapped = sum(1 for item in state.get("compliance_matrix", []) if item.get("status") == "mapped")
        checks.append(f"Compliance matrix created with {mapped} fully mapped requirement(s).")

    return NetworkDesignValidationReport(passed=not blocking, blocking_issues=blocking, warnings=warnings, checks=checks)


def build_network_design_graph(
    model: ChatOpenAI,
    checkpointer: AsyncRedisSaver,
    handoff_model: ChatOpenAI | None = None,
):
    async def intake_conversation(state: NetworkDesignState) -> NetworkDesignState:
        started_at, started_perf = _trace_start()
        intake = NetworkDesignIntake(**state.get("intake", {})).model_dump()
        structured_intake = state.get("structured_intake", {})
        if isinstance(structured_intake, dict):
            for key, value in structured_intake.get("intake_updates", {}).items():
                if key in intake and value not in (None, "", [], {}):
                    intake[key] = value
        messages = [NetworkDesignMessage(**item).model_dump() for item in state.get("messages", [])]
        query = " ".join(
            part
            for part in [
                intake.get("customer_name", ""),
                intake.get("site_name", ""),
                intake.get("design_goal", ""),
                intake.get("business_context", ""),
                intake.get("constraints", ""),
                _conversation_text(messages),
            ]
            if part
        )
        return _trace_update(
            state,
            {
                "intake": intake,
                "structured_intake": structured_intake if isinstance(structured_intake, dict) else {},
                "readiness_report": state.get("readiness_report", {}),
                "messages": messages,
                "standards_query": query or "network design fortigate sd-wan firewall standards",
            },
            "intake_conversation",
            started_at,
            started_perf,
            f"Captured {len(messages)} conversation message(s).",
        )

    async def retrieve_standards(state: NetworkDesignState) -> NetworkDesignState:
        started_at, started_perf = _trace_start()
        query = state.get("standards_query", "")
        # Per-thread reuse: the chat graph re-runs every turn, but standards only change
        # when the query changes. Reuse the checkpointed result otherwise.
        if state.get("standards") and state.get("standards_query_used") == query:
            return _trace_update(state, {}, "retrieve_standards", started_at, started_perf, "Reused cached standards for unchanged query.", branch="cached")
        chunks = [chunk.model_dump() for chunk in search_standards(query, limit=10)]
        return _trace_update(state, {"standards": chunks, "standards_query_used": query, "standard_requirements": []}, "retrieve_standards", started_at, started_perf, f"Retrieved {len(chunks)} standards chunk(s).")

    async def curate_standards(state: NetworkDesignState) -> NetworkDesignState:
        started_at, started_perf = _trace_start()
        if state.get("standard_requirements"):
            return _trace_update(state, {}, "curate_standards", started_at, started_perf, "Reused cached standard requirements.", branch="cached")
        requirements = extract_standard_requirements(state.get("standards", []), max_requirements=24)
        return _trace_update(
            state,
            {"standard_requirements": requirements},
            "curate_standards",
            started_at,
            started_perf,
            f"Extracted {len(requirements)} structured standard requirement(s).",
        )

    async def summarize_requirements(state: NetworkDesignState) -> NetworkDesignState:
        started_at, started_perf = _trace_start()
        response = await model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "Return only JSON with key requirements_summary. Summarize the design discussion for a network design engineer. "
                        "Include business_goal, sites, wan_requirements, lan_requirements, security_requirements, routing_requirements, "
                        "operations_requirements, constraints, assumptions, and open_questions."
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        {
                            "intake": state.get("intake", {}),
                            "structured_intake": state.get("structured_intake", {}),
                            "conversation": state.get("messages", []),
                            "standards": _standards_payload(state.get("standards", [])),
                        },
                        indent=2,
                    )
                ),
            ],
            config={"run_name": "summarize_requirements"},
        )
        data = _extract_json_object(str(response.content))
        requirements = data.get("requirements_summary") if isinstance(data.get("requirements_summary"), dict) else data
        if not requirements:
            requirements = _fallback_requirements(state.get("intake", {}), state.get("messages", []))
        return _trace_update(state, {"requirements_summary": requirements}, "summarize_requirements", started_at, started_perf, "Summarized conversation into network requirements.")

    async def identify_gaps(state: NetworkDesignState) -> NetworkDesignState:
        started_at, started_perf = _trace_start()
        requirements = state.get("requirements_summary", {})
        open_questions = [str(item) for item in _safe_list(requirements.get("open_questions")) if str(item).strip()]
        defaults = [
            ("wan_requirements", "What WAN circuits, bandwidths, IP assignment, and failover behavior should be assumed?"),
            ("lan_requirements", "What LAN VLANs, subnets, gateways, and DHCP expectations should be included?"),
            ("security_requirements", "What security zones and policy boundaries are required?"),
            ("operations_requirements", "What logging, monitoring, change window, and rollback expectations should be included?"),
        ]
        for key, question in defaults:
            if not requirements.get(key):
                open_questions.append(question)
        deduped = list(dict.fromkeys(open_questions))[:8]
        return _trace_update(state, {"missing_questions": deduped}, "identify_gaps", started_at, started_perf, f"Identified {len(deduped)} open design question(s).")

    async def build_design_package(state: NetworkDesignState) -> NetworkDesignState:
        started_at, started_perf = _trace_start()
        response = await model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "Return only JSON with key design_package. Build a standards-grounded network design package. Include executive_summary, "
                        "topology, wan_design, lan_design, security_zone_design, routing_design, sdwan_design, firewall_policy_intent, "
                        "logging_monitoring, implementation_notes, risks, assumptions, standards_citations, and next_questions. "
                        "Keep it artifact-only and do not claim any changes were made."
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        {
                            "intake": state.get("intake", {}),
                            "structured_intake": state.get("structured_intake", {}),
                            "requirements_summary": state.get("requirements_summary", {}),
                            "missing_questions": state.get("missing_questions", []),
                            "standards": _standards_payload(state.get("standards", [])),
                            "standard_requirements": state.get("standard_requirements", []),
                        },
                        indent=2,
                    )
                ),
            ],
            config={"run_name": "build_design_package"},
        )
        data = _extract_json_object(str(response.content))
        design = data.get("design_package") if isinstance(data.get("design_package"), dict) else data
        if not design:
            design = {
                "executive_summary": state.get("intake", {}).get("design_goal", ""),
                "requirements_summary": state.get("requirements_summary", {}),
                "standards_citations": _standards_payload(state.get("standards", []))[:5],
                "assumptions": ["Generated from available structured input and chat transcript."],
                "next_questions": state.get("missing_questions", []),
            }
        return _trace_update(state, {"design_package": design}, "build_design_package", started_at, started_perf, "Built standards-grounded design package.")

    async def build_fortigate_handoff(state: NetworkDesignState) -> NetworkDesignState:
        started_at, started_perf = _trace_start()
        mapper = handoff_model or model
        response = await mapper.ainvoke(
            [
                SystemMessage(
                    content=(
                        "Return only JSON with key fortigate_handoff. Map the network design into the FortiGateDesignRequest intake shape. "
                        "Use keys request_type, site_name, business_intent, fortigate_model, fortios_version, wan_circuits, lan_networks, "
                        "security_zones, wifi_requirements, fortiswitch_requirements, system_hardening_requirements, "
                        "admin_access_requirements, security_profile_requirements, firewall_policy_intent, nat_requirements, "
                        "vpn_requirements, logging_requirements, ha_requirements, change_window, rollback_expectations, and "
                        "additional_context. wifi_requirements and fortiswitch_requirements are lists of objects (SSIDs/AP "
                        "profiles; FortiLink/managed switches/VLANs). system_hardening_requirements, admin_access_requirements, "
                        "and security_profile_requirements are objects (NTP/password-policy/timezone; admins+MFA+trusted hosts; "
                        "AV/IPS/web-filter/SSL inspection profiles). Do NOT invent secrets, serials, or real site addresses. "
                        "This is a handoff only, not a device change."
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        {
                            "intake": state.get("intake", {}),
                            "structured_intake": state.get("structured_intake", {}),
                            "requirements_summary": state.get("requirements_summary", {}),
                            "design_package": state.get("design_package", {}),
                            "missing_questions": state.get("missing_questions", []),
                        },
                        indent=2,
                    )
                ),
            ],
            config={"run_name": "build_fortigate_handoff"},
        )
        data = _extract_json_object(str(response.content))
        handoff = data.get("fortigate_handoff") if isinstance(data.get("fortigate_handoff"), dict) else data
        if not handoff:
            handoff = _fallback_handoff(state.get("intake", {}), state.get("requirements_summary", {}), state.get("design_package", {}))
        handoff.setdefault("request_type", "new_build")
        handoff.setdefault("site_name", state.get("intake", {}).get("site_name", ""))
        handoff.setdefault("business_intent", state.get("intake", {}).get("design_goal", ""))
        handoff.setdefault("additional_context", "Generated by Network Design Helper. No device changes have been made.")
        for key in (
            "request_type",
            "site_name",
            "business_intent",
            "fortigate_model",
            "fortios_version",
            "firewall_policy_intent",
            "nat_requirements",
            "vpn_requirements",
            "logging_requirements",
            "ha_requirements",
            "change_window",
            "rollback_expectations",
            "additional_context",
        ):
            value = handoff.get(key, "")
            if isinstance(value, (dict, list)):
                handoff[key] = json.dumps(value, indent=2)
            elif value is None:
                handoff[key] = ""
            else:
                handoff[key] = str(value)
        for key in ("wan_circuits", "lan_networks", "security_zones", "wifi_requirements", "fortiswitch_requirements"):
            value = handoff.get(key)
            if value in (None, ""):
                handoff[key] = []
            elif not isinstance(value, list):
                handoff[key] = [{"description": str(value)}] if key != "security_zones" else [str(value)]
            elif key != "security_zones":
                handoff[key] = [item if isinstance(item, dict) else {"description": str(item)} for item in value]
            else:
                handoff[key] = [str(item) for item in value if str(item).strip()]
        # Structured topic objects: default-empty dicts, additive and back-compatible.
        for key in ("system_hardening_requirements", "admin_access_requirements", "security_profile_requirements"):
            value = handoff.get(key)
            if isinstance(value, dict):
                continue
            if value in (None, ""):
                handoff[key] = {}
            else:
                handoff[key] = {"description": str(value)}
        return _trace_update(state, {"fortigate_handoff": handoff}, "build_fortigate_handoff", started_at, started_perf, "Prepared FortiGate handoff payload.")

    async def check_compliance(state: NetworkDesignState) -> NetworkDesignState:
        started_at, started_perf = _trace_start()
        matrix = build_compliance_matrix(state)
        needs_review = sum(1 for item in matrix if item.get("status") == "needs_review")
        return _trace_update(
            state,
            {"compliance_matrix": matrix},
            "check_compliance",
            started_at,
            started_perf,
            f"Created compliance matrix with {needs_review} requirement(s) needing review.",
        )

    async def validate_design(state: NetworkDesignState) -> NetworkDesignState:
        started_at, started_perf = _trace_start()
        report = validate_network_design(state).model_dump()
        return _trace_update(state, {"validation_report": report}, "validate_design", started_at, started_perf, f"Validation produced {len(report.get('blocking_issues', []))} blocking issue(s).")

    async def finalize_package(state: NetworkDesignState) -> NetworkDesignState:
        started_at, started_perf = _trace_start()
        report = state.get("validation_report", {})
        status = "ready_for_fortigate_handoff" if report.get("passed", False) else "needs_design_input"
        if state.get("missing_questions"):
            status = "needs_design_review"
        return _trace_update(state, {"status": status}, "finalize_package", started_at, started_perf, f"Final status: {status}.")

    graph = StateGraph(NetworkDesignState)
    graph.add_node("intake_conversation", intake_conversation)
    graph.add_node("retrieve_standards", retrieve_standards)
    graph.add_node("curate_standards", curate_standards)
    graph.add_node("summarize_requirements", summarize_requirements)
    graph.add_node("identify_gaps", identify_gaps)
    graph.add_node("build_design_package", build_design_package)
    graph.add_node("build_fortigate_handoff", build_fortigate_handoff)
    graph.add_node("check_compliance", check_compliance)
    graph.add_node("validate_design", validate_design)
    graph.add_node("finalize_package", finalize_package)

    graph.set_entry_point("intake_conversation")
    graph.add_edge("intake_conversation", "retrieve_standards")
    graph.add_edge("retrieve_standards", "curate_standards")
    graph.add_edge("curate_standards", "summarize_requirements")
    graph.add_edge("summarize_requirements", "identify_gaps")
    graph.add_edge("identify_gaps", "build_design_package")
    graph.add_edge("build_design_package", "build_fortigate_handoff")
    graph.add_edge("build_fortigate_handoff", "check_compliance")
    graph.add_edge("check_compliance", "validate_design")
    graph.add_edge("validate_design", "finalize_package")
    graph.add_edge("finalize_package", END)
    return graph.compile(checkpointer=checkpointer)
