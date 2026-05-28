import json
import re
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from app.fortigate_judge import run_frontier_judge
from app.fortigate_models import (
    FortiGateConfigSummary,
    FortiGateIntake,
    FortiGateQuestion,
    FortiGateStandardChunk,
)
from app.fortigate_policy import check_standards_compliance, review_risk, validate_config_artifacts
from app.standards import retrieve_fortigate_standards


class FortiGateState(TypedDict, total=False):
    intake: dict[str, Any]
    existing_config: str
    current_config_summary: dict[str, Any]
    request_classification: dict[str, Any]
    standards: list[dict[str, Any]]
    missing_questions: list[dict[str, Any]]
    logical_design: dict[str, Any]
    fortigate_design: dict[str, Any]
    change_impact: dict[str, Any]
    config_artifacts: dict[str, Any]
    validation_report: dict[str, Any]
    standards_report: dict[str, Any]
    risk_report: dict[str, Any]
    judge_report: dict[str, Any]
    judge_iterations: int
    human_review_decision: str
    human_review_notes: str
    human_selected_issues: list[str]
    status: str
    execution_trace: list[dict[str, Any]]


def _trace_start() -> tuple[str, float]:
    return datetime.now(UTC).isoformat(), perf_counter()


def _trace_update(
    state: FortiGateState,
    updates: FortiGateState,
    node: str,
    started_at: str,
    started_perf: float,
    summary: str,
    branch: str = "",
) -> FortiGateState:
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
    end = stripped.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return {}


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _judge_review_item_count(report: dict[str, Any]) -> int:
    review_fields = (
        "blocking_issues",
        "warnings",
        "standards_concerns",
        "config_risks",
        "missing_questions",
        "recommended_revisions",
        "human_reviewer_focus",
    )
    items: set[str] = set()
    for field in review_fields:
        for item in _as_list(report.get(field)):
            items.add(f"{field}:{item.lower()}")
    return len(items)


def _object_name(block: str) -> str:
    match = re.search(r'edit\s+"?([^"\n]+)"?', block)
    return match.group(1).strip() if match else ""


def _setting(block: str, key: str) -> str:
    match = re.search(rf"set\s+{re.escape(key)}\s+(.+)", block)
    if not match:
        return ""
    return match.group(1).strip().strip('"')


def _blocks(config: str, section: str) -> list[str]:
    pattern = rf"config\s+{re.escape(section)}\n(.*?)(?=\nconfig\s+|\Z)"
    match = re.search(pattern, config, re.DOTALL)
    if not match:
        return []
    body = match.group(1)
    return [f"edit {part}".strip() for part in re.split(r"\n\s*edit\s+", body) if part.strip() and not part.strip().startswith("end")]


def parse_fortigate_config(config: str) -> FortiGateConfigSummary:
    warnings: list[str] = []
    if not config.strip():
        return FortiGateConfigSummary(parser_warnings=["No configuration text supplied."])

    hostname = ""
    global_match = re.search(r"config system global\n(.*?)(?=\nconfig\s+|\Z)", config, re.DOTALL)
    if global_match:
        hostname = _setting(global_match.group(1), "hostname")

    interfaces = []
    vlans = []
    for block in _blocks(config, "system interface"):
        item = {
            "name": _object_name(block),
            "alias": _setting(block, "alias"),
            "ip": _setting(block, "ip"),
            "role": _setting(block, "role"),
            "interface": _setting(block, "interface"),
            "vlanid": _setting(block, "vlanid"),
        }
        if item["vlanid"]:
            vlans.append(item)
        else:
            interfaces.append(item)

    policies = []
    for block in _blocks(config, "firewall policy"):
        policies.append(
            {
                "id": _object_name(block),
                "name": _setting(block, "name"),
                "srcintf": _setting(block, "srcintf"),
                "dstintf": _setting(block, "dstintf"),
                "srcaddr": _setting(block, "srcaddr"),
                "dstaddr": _setting(block, "dstaddr"),
                "service": _setting(block, "service"),
                "action": _setting(block, "action"),
                "nat": _setting(block, "nat"),
                "logtraffic": _setting(block, "logtraffic"),
            }
        )

    addresses = [{"name": _object_name(block), "subnet": _setting(block, "subnet"), "fqdn": _setting(block, "fqdn")} for block in _blocks(config, "firewall address")]
    services = [{"name": _object_name(block), "tcp": _setting(block, "tcp-portrange"), "udp": _setting(block, "udp-portrange")} for block in _blocks(config, "firewall service custom")]
    routes = [{"id": _object_name(block), "dst": _setting(block, "dst"), "gateway": _setting(block, "gateway"), "device": _setting(block, "device")} for block in _blocks(config, "router static")]
    vips = [{"name": _object_name(block), "extip": _setting(block, "extip"), "mappedip": _setting(block, "mappedip"), "extintf": _setting(block, "extintf")} for block in _blocks(config, "firewall vip")]
    vpn = [{"name": _object_name(block), "interface": _setting(block, "interface"), "remote-gw": _setting(block, "remote-gw")} for block in _blocks(config, "vpn ipsec phase1-interface")]

    sdwan = {"present": "config system sdwan" in config, "members": len(_blocks(config, "system sdwan"))}
    ha = {"present": "config system ha" in config}
    if not interfaces and not policies:
        warnings.append("Parser found few FortiGate sections; config may be partial or unsupported.")

    return FortiGateConfigSummary(
        hostname=hostname,
        interfaces=interfaces,
        vlans=vlans,
        static_routes=routes,
        sdwan=sdwan,
        firewall_policies=policies,
        address_objects=addresses,
        service_objects=services,
        vips=vips,
        vpn=vpn,
        ha=ha,
        raw_line_count=len(config.splitlines()),
        parser_warnings=warnings,
    )


def _default_questions(intake: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, Any]]:
    questions: list[FortiGateQuestion] = []
    checks = [
        ("site_name", "What site name should be used in object names, comments, and documentation?", "Needed for naming and auditability."),
        ("fortigate_model", "What FortiGate model is this for?", "Model can affect interface and feature assumptions."),
        ("fortios_version", "What FortiOS version should the draft target?", "CLI syntax and SD-WAN behavior vary by version."),
        ("change_window", "What change window or maintenance window should the plan assume?", "Needed for risk and rollback planning."),
        ("rollback_expectations", "What rollback expectation should be included?", "Needed before any operational change package is approved."),
    ]
    for field, question, reason in checks:
        if not intake.get(field):
            questions.append(FortiGateQuestion(field=field, question=question, reason=reason).model_dump())
    if intake.get("request_type") == "modify_existing" and not summary.get("raw_line_count"):
        questions.append(
            FortiGateQuestion(
                field="existing_config",
                question="Please upload or paste the current FortiGate configuration for this change workflow.",
                reason="Existing-state analysis is required for modification requests.",
            ).model_dump()
        )
    return questions[:6]


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


def _fallback_cli(intake: dict[str, Any]) -> str:
    site = intake.get("site_name") or "site"
    lines = [
        "config system interface",
        f'    edit "wan1"',
        '        set role wan',
        "    next",
        f'    edit "{site}_lan"',
        "        set role lan",
        "    next",
        "end",
        "config system sdwan",
        "    set status enable",
        "end",
        "config firewall policy",
        "    edit 0",
        f'        set name "{site}_outbound_internet"',
        '        set srcintf "lan"',
        '        set dstintf "virtual-wan-link"',
        '        set srcaddr "all"',
        '        set dstaddr "all"',
        "        set action accept",
        "        set schedule always",
        '        set service "ALL"',
        "        set nat enable",
        "        set logtraffic all",
        "    next",
        "end",
    ]
    return "\n".join(lines)


def build_fortigate_graph(
    model: ChatOpenAI,
    checkpointer: AsyncRedisSaver,
    clarification_interrupt: bool = False,
    human_review_interrupt: bool = False,
    judge_model: ChatOpenAI | None = None,
    judge_model_name: str = "",
):
    async def intake_request(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        intake = FortiGateIntake(**state.get("intake", {})).model_dump()
        request_type = intake.get("request_type") or ("modify_existing" if state.get("existing_config") else "new_build")
        classification = {
            "request_type": request_type,
            "is_change": request_type == "modify_existing",
            "areas": [
                area
                for area, present in {
                    "sdwan": bool(intake.get("wan_circuits")),
                    "firewall": bool(intake.get("firewall_policy_intent")),
                    "nat": bool(intake.get("nat_requirements")),
                    "vpn": bool(intake.get("vpn_requirements")),
                    "ha": bool(intake.get("ha_requirements")),
                }.items()
                if present
            ],
        }
        intake["request_type"] = request_type
        return _trace_update(state, {"intake": intake, "request_classification": classification}, "intake_request", started_at, started_perf, f"Classified request as {request_type}.")

    async def parse_existing_config(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        summary = parse_fortigate_config(state.get("existing_config", "")).model_dump()
        return _trace_update(state, {"current_config_summary": summary}, "parse_existing_config", started_at, started_perf, f"Parsed config with {summary.get('raw_line_count', 0)} line(s).")

    async def retrieve_standards(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        standards = [chunk.model_dump() for chunk in retrieve_fortigate_standards(state.get("intake", {}), limit=10)]
        return _trace_update(state, {"standards": standards}, "retrieve_standards", started_at, started_perf, f"Retrieved {len(standards)} company standard chunk(s).")

    async def identify_missing_inputs(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        questions = _default_questions(state.get("intake", {}), state.get("current_config_summary", {}))
        return _trace_update(state, {"missing_questions": questions}, "identify_missing_inputs", started_at, started_perf, f"Identified {len(questions)} missing input question(s).", branch="questions" if questions else "complete")

    async def human_clarification_checkpoint(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        if not state.get("missing_questions"):
            return _trace_update(state, {}, "human_clarification_checkpoint", started_at, started_perf, "No clarification required.", branch="skip")
        answers = interrupt({"stage": "fortigate_clarification", "questions": state.get("missing_questions", []), "intake": state.get("intake", {})})
        intake = dict(state.get("intake", {}))
        if isinstance(answers, dict):
            for key, value in (answers.get("answers") or answers).items():
                if value not in (None, ""):
                    intake[key] = value
        return _trace_update(state, {"intake": intake, "missing_questions": []}, "human_clarification_checkpoint", started_at, started_perf, "Merged human clarification answers.", branch="resumed")

    async def build_logical_design(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        response = await model.ainvoke(
            [
                SystemMessage(content="Return only JSON with key logical_design. Build a vendor-neutral network/security design from the intake, current config summary, and company standards."),
                HumanMessage(content=json.dumps({"intake": state.get("intake", {}), "current_config_summary": state.get("current_config_summary", {}), "standards": _standards_payload(state.get("standards", []))}, indent=2)),
            ]
        )
        data = _extract_json_object(str(response.content))
        design = data.get("logical_design") if isinstance(data.get("logical_design"), dict) else data
        return _trace_update(state, {"logical_design": design or {}}, "build_logical_design", started_at, started_perf, "Built logical design.")

    async def build_fortigate_design(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        response = await model.ainvoke(
            [
                SystemMessage(content="Return only JSON with key fortigate_design. Map the logical design to FortiGate constructs: interfaces, zones, SD-WAN, routes, objects, policies, NAT/VIP, VPN, logging, HA."),
                HumanMessage(content=json.dumps({"intake": state.get("intake", {}), "logical_design": state.get("logical_design", {}), "current_config_summary": state.get("current_config_summary", {}), "standards": _standards_payload(state.get("standards", []))}, indent=2)),
            ]
        )
        data = _extract_json_object(str(response.content))
        design = data.get("fortigate_design") if isinstance(data.get("fortigate_design"), dict) else data
        return _trace_update(state, {"fortigate_design": design or {}}, "build_fortigate_design", started_at, started_perf, "Mapped logical design to FortiGate constructs.")

    async def analyze_change_impact(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        summary = state.get("current_config_summary", {})
        impact = {
            "mode": "modify_existing" if state.get("intake", {}).get("request_type") == "modify_existing" else "new_build",
            "impacted_interfaces": [item.get("name") for item in summary.get("interfaces", [])[:8]],
            "impacted_policies": [item.get("id") for item in summary.get("firewall_policies", [])[:12]],
            "rollback_required": True,
            "notes": ["Artifact-only analysis. No device changes will be made."],
        }
        return _trace_update(state, {"change_impact": impact}, "analyze_change_impact", started_at, started_perf, "Created change impact summary.")

    async def generate_config_artifacts(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        response = await model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "Return only JSON with key config_artifacts. Include cli_config, object_tables, rollback_plan, assumptions, "
                        "standards_citations. The CLI must be a draft only and must not claim it was applied."
                    )
                ),
                HumanMessage(content=json.dumps({"intake": state.get("intake", {}), "current_config_summary": state.get("current_config_summary", {}), "fortigate_design": state.get("fortigate_design", {}), "change_impact": state.get("change_impact", {}), "standards": _standards_payload(state.get("standards", []))}, indent=2)),
            ]
        )
        data = _extract_json_object(str(response.content))
        artifacts = data.get("config_artifacts") if isinstance(data.get("config_artifacts"), dict) else data
        if not artifacts.get("cli_config"):
            artifacts["cli_config"] = _fallback_cli(state.get("intake", {}))
        artifacts.setdefault("safety_label", "DRAFT ONLY - NOT APPLIED TO DEVICE")
        return _trace_update(state, {"config_artifacts": artifacts}, "generate_config_artifacts", started_at, started_perf, "Generated draft config artifacts.")

    async def validate_config(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        report = validate_config_artifacts(state).model_dump()
        return _trace_update(state, {"validation_report": report}, "validate_config", started_at, started_perf, f"Validation produced {len(report.get('blocking_issues', []))} blocking issue(s).")

    async def check_standards(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        report = check_standards_compliance(state).model_dump()
        return _trace_update(state, {"standards_report": report}, "check_standards", started_at, started_perf, f"Standards check produced {len(report.get('warnings', []))} warning(s).")

    async def risk_review(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        report = review_risk(state).model_dump()
        return _trace_update(state, {"risk_report": report}, "risk_review", started_at, started_perf, f"Risk review produced {len(report.get('warnings', []))} warning(s).")

    async def frontier_model_judge(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        packet = {
            "intake": state.get("intake", {}),
            "current_config_summary": state.get("current_config_summary", {}),
            "logical_design": state.get("logical_design", {}),
            "fortigate_design": state.get("fortigate_design", {}),
            "change_impact": state.get("change_impact", {}),
            "config_artifacts": state.get("config_artifacts", {}),
            "validation_report": state.get("validation_report", {}),
            "standards_report": state.get("standards_report", {}),
            "risk_report": state.get("risk_report", {}),
            "standards": _standards_payload(state.get("standards", [])),
        }
        report = await run_frontier_judge(judge_model or model, packet, judge_model_name=judge_model_name or getattr(model, "model_name", "configured-model"))
        return _trace_update(state, {"judge_report": report.model_dump(), "judge_iterations": int(state.get("judge_iterations", 0) or 0) + 1}, "frontier_model_judge", started_at, started_perf, f"Judge verdict: {report.verdict}.", branch=report.verdict)

    async def revise_after_judge(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        response = await model.ainvoke(
            [
                SystemMessage(content="Return only JSON with key config_artifacts. Revise the draft FortiGate artifacts to address the judge report while preserving safety labels and standards citations."),
                HumanMessage(content=json.dumps({"config_artifacts": state.get("config_artifacts", {}), "judge_report": state.get("judge_report", {}), "standards": _standards_payload(state.get("standards", []))}, indent=2)),
            ]
        )
        data = _extract_json_object(str(response.content))
        artifacts = data.get("config_artifacts") if isinstance(data.get("config_artifacts"), dict) else data
        if artifacts:
            artifacts.setdefault("safety_label", "DRAFT ONLY - NOT APPLIED TO DEVICE")
            return _trace_update(state, {"config_artifacts": artifacts}, "revise_after_judge", started_at, started_perf, "Revised artifacts after judge feedback.")
        return _trace_update(state, {}, "revise_after_judge", started_at, started_perf, "No structured revision returned; kept prior artifacts.", branch="no-op")

    async def regenerate_config_after_judge(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        response = await model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "Return only JSON with key config_artifacts. Recreate the FortiGate draft artifacts from scratch "
                        "because the independent judge found more than three review items. Do not patch the previous CLI; "
                        "use it only as negative context. Preserve safety labels, standards citations, rollback details, "
                        "and artifact-only language."
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        {
                            "intake": state.get("intake", {}),
                            "current_config_summary": state.get("current_config_summary", {}),
                            "fortigate_design": state.get("fortigate_design", {}),
                            "change_impact": state.get("change_impact", {}),
                            "validation_report": state.get("validation_report", {}),
                            "standards_report": state.get("standards_report", {}),
                            "risk_report": state.get("risk_report", {}),
                            "judge_report": state.get("judge_report", {}),
                            "previous_config_artifacts": state.get("config_artifacts", {}),
                            "standards": _standards_payload(state.get("standards", [])),
                        },
                        indent=2,
                    )
                ),
            ]
        )
        data = _extract_json_object(str(response.content))
        artifacts = data.get("config_artifacts") if isinstance(data.get("config_artifacts"), dict) else data
        if artifacts:
            artifacts.setdefault("safety_label", "DRAFT ONLY - NOT APPLIED TO DEVICE")
            return _trace_update(
                state,
                {"config_artifacts": artifacts},
                "regenerate_config_after_judge",
                started_at,
                started_perf,
                "Regenerated config artifacts from scratch after broad judge feedback.",
                branch="regenerated",
            )
        return _trace_update(state, {}, "regenerate_config_after_judge", started_at, started_perf, "No structured regeneration returned; kept prior artifacts.", branch="no-op")

    def route_after_judge(state: FortiGateState) -> str:
        report = state.get("judge_report", {})
        verdict = report.get("verdict", "needs_revision")
        review_item_count = _judge_review_item_count(report)
        if verdict == "needs_revision" and int(state.get("judge_iterations", 0) or 0) < 2:
            if review_item_count > 3:
                return "regenerate"
            return "revise"
        return "human_review" if human_review_interrupt else "finalize"

    async def human_review_checkpoint(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        report = state.get("judge_report", {})
        review = interrupt(
            {
                "stage": "fortigate_human_review",
                "status": state.get("status", "draft"),
                "judge_report": report,
                "validation_report": state.get("validation_report", {}),
                "standards_report": state.get("standards_report", {}),
                "risk_report": state.get("risk_report", {}),
                "config_artifacts": state.get("config_artifacts", {}),
                "instructions": "Approve, reject, or request revisions. No device changes will be made.",
            }
        )
        if not isinstance(review, dict):
            review = {"decision": "approved", "reviewer_notes": str(review or ""), "selected_issues": []}
        decision = str(review.get("decision", "approved")).lower()
        if decision not in {"approved", "needs_work", "rejected"}:
            decision = "approved"
        return _trace_update(
            state,
            {
                "human_review_decision": decision,
                "human_review_notes": str(review.get("reviewer_notes") or ""),
                "human_selected_issues": _as_list(review.get("selected_issues")),
            },
            "human_review_checkpoint",
            started_at,
            started_perf,
            f"Human review decision: {decision}.",
            branch=decision,
        )

    async def finalize_package(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        decision = state.get("human_review_decision") or "pending"
        judge_verdict = state.get("judge_report", {}).get("verdict", "needs_revision")
        status = "approved_artifact" if decision == "approved" else ("blocked" if judge_verdict == "block" or decision == "rejected" else "needs_review")
        return _trace_update(state, {"status": status}, "finalize_package", started_at, started_perf, f"Final package status: {status}.")

    graph = StateGraph(FortiGateState)
    graph.add_node("intake_request", intake_request)
    graph.add_node("parse_existing_config", parse_existing_config)
    graph.add_node("retrieve_standards", retrieve_standards)
    graph.add_node("identify_missing_inputs", identify_missing_inputs)
    if clarification_interrupt:
        graph.add_node("human_clarification_checkpoint", human_clarification_checkpoint)
    graph.add_node("build_logical_design", build_logical_design)
    graph.add_node("build_fortigate_design", build_fortigate_design)
    graph.add_node("analyze_change_impact", analyze_change_impact)
    graph.add_node("generate_config_artifacts", generate_config_artifacts)
    graph.add_node("validate_config", validate_config)
    graph.add_node("check_standards", check_standards)
    graph.add_node("risk_review", risk_review)
    graph.add_node("frontier_model_judge", frontier_model_judge)
    graph.add_node("revise_after_judge", revise_after_judge)
    graph.add_node("regenerate_config_after_judge", regenerate_config_after_judge)
    if human_review_interrupt:
        graph.add_node("human_review_checkpoint", human_review_checkpoint)
    graph.add_node("finalize_package", finalize_package)

    graph.set_entry_point("intake_request")
    graph.add_edge("intake_request", "parse_existing_config")
    graph.add_edge("parse_existing_config", "retrieve_standards")
    graph.add_edge("retrieve_standards", "identify_missing_inputs")
    if clarification_interrupt:
        graph.add_edge("identify_missing_inputs", "human_clarification_checkpoint")
        graph.add_edge("human_clarification_checkpoint", "build_logical_design")
    else:
        graph.add_edge("identify_missing_inputs", "build_logical_design")
    graph.add_edge("build_logical_design", "build_fortigate_design")
    graph.add_edge("build_fortigate_design", "analyze_change_impact")
    graph.add_edge("analyze_change_impact", "generate_config_artifacts")
    graph.add_edge("generate_config_artifacts", "validate_config")
    graph.add_edge("validate_config", "check_standards")
    graph.add_edge("check_standards", "risk_review")
    graph.add_edge("risk_review", "frontier_model_judge")
    judge_routes = {"revise": "revise_after_judge", "regenerate": "regenerate_config_after_judge", "finalize": "finalize_package"}
    if human_review_interrupt:
        judge_routes["human_review"] = "human_review_checkpoint"
    graph.add_conditional_edges("frontier_model_judge", route_after_judge, judge_routes)
    graph.add_edge("revise_after_judge", "validate_config")
    graph.add_edge("regenerate_config_after_judge", "validate_config")
    if human_review_interrupt:
        graph.add_edge("human_review_checkpoint", "finalize_package")
    graph.add_edge("finalize_package", END)
    return graph.compile(checkpointer=checkpointer)
