import asyncio
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
from app.fortigate_policy import check_cli_completeness, check_intent_completeness, check_section_completeness, check_standards_compliance, review_risk, validate_config_artifacts
from app.standards import retrieve_fortigate_standards
from app.fortigate_render_models import FortiGateConfigModel
from app.fortigate_renderer import render_config
from app.fortigate_config_builder import build_config_model as build_fortigate_config_model
from app.fortigate_config_builder import normalize_fortios_version


class FortiGateState(TypedDict, total=False):
    intake: dict[str, Any]
    existing_config: str
    current_config_summary: dict[str, Any]
    request_classification: dict[str, Any]
    standards: list[dict[str, Any]]
    missing_questions: list[dict[str, Any]]
    logical_design: dict[str, Any]
    fortigate_design: dict[str, Any]
    implementation_intent: dict[str, Any]
    intent_completeness_report: dict[str, Any]
    intent_repaired: bool
    change_impact: dict[str, Any]
    config_sections: dict[str, Any]
    config_model: dict[str, Any]
    renderer_active: bool
    renderer_fallback: bool
    config_artifacts: dict[str, Any]
    section_validation_report: dict[str, Any]
    cli_completeness_report: dict[str, Any]
    validation_report: dict[str, Any]
    standards_report: dict[str, Any]
    risk_report: dict[str, Any]
    config_builder_review: dict[str, Any]
    config_refinement: dict[str, Any]
    judge_report: dict[str, Any]
    judge_iterations: int
    autonomous_repair_iterations: int
    autonomous_repair_tasks: list[dict[str, Any]]
    auto_fixed_items: list[str]
    requires_human_input: list[str]
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


def _is_context_window_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "contextwindowexceeded",
            "context window",
            "maximum context length",
            "input_tokens",
        )
    )


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


def _judge_item_texts(report: dict[str, Any]) -> list[str]:
    fields = (
        "blocking_issues",
        "warnings",
        "standards_concerns",
        "config_risks",
        "missing_questions",
        "recommended_revisions",
        "human_reviewer_focus",
    )
    items: list[str] = []
    for field in fields:
        for item in _as_list(report.get(field)):
            text = str(item).strip()
            if text:
                items.append(f"{field}: {text}")
    return items


def _classify_judge_items(report: dict[str, Any]) -> tuple[list[str], list[str]]:
    auto_keywords = (
        "syntax",
        "invalid",
        "missing",
        "undefined",
        "not defined",
        "dhcp",
        "sd-wan",
        "sdwan",
        "health check",
        "route",
        "object",
        "profile",
        "pool",
        "policy",
        "logtraffic",
        "logging",
        "segmentation",
        "deny",
    )
    human_keywords = (
        "actual value",
        "specific source",
        "source ip",
        "public ip",
        "psk",
        "password",
        "secret",
        "community",
        "syslog",
        "snmp",
        "dns server",
        "confirm with",
        "client",
        "preference",
        "business decision",
        "accept risk",
    )
    auto_fixable: list[str] = []
    requires_human: list[str] = []
    for item in _judge_item_texts(report):
        lowered = item.lower()
        if any(keyword in lowered for keyword in human_keywords):
            requires_human.append(item)
        elif any(keyword in lowered for keyword in auto_keywords):
            auto_fixable.append(item)
        else:
            requires_human.append(item)
    return auto_fixable, requires_human


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


def _summarized_findings(state: FortiGateState, *, include_judge: bool = False) -> dict[str, Any]:
    """Collapse the verbose per-stage reports into a single deduplicated findings digest.

    The reviewer/refiner/judge only need the actionable issues, not the full nested report
    objects, so this keeps the LLM packet focused on the CLI under review.
    """
    report_keys = [
        "intent_completeness_report",
        "section_validation_report",
        "cli_completeness_report",
        "validation_report",
        "standards_report",
        "risk_report",
    ]
    if include_judge:
        report_keys.append("judge_report")
    blocking: list[str] = []
    warnings: list[str] = []
    for key in report_keys:
        report = state.get(key, {}) or {}
        if not isinstance(report, dict):
            continue
        for item in [*_as_list(report.get("blocking_issues")), *_as_list(report.get("issues"))]:
            blocking.append(f"{key}: {item}")
        for item in _as_list(report.get("warnings")):
            warnings.append(f"{key}: {item}")
    return {
        "blocking_issues": list(dict.fromkeys(blocking))[:40],
        "warnings": list(dict.fromkeys(warnings))[:60],
    }


def _review_payload(state: FortiGateState) -> dict[str, Any]:
    """Compact packet for CLI-improving roles (builder review, refine).

    Drops the duplicated config_sections (already merged into config_artifacts) and the
    redundant logical_design, and replaces the full per-stage reports with summarized
    findings. Keeps the design intent and the artifacts being improved.
    """
    payload: dict[str, Any] = {
        "intake": state.get("intake", {}),
        "fortigate_design": state.get("fortigate_design", {}),
        "implementation_intent": state.get("implementation_intent", {}),
        "change_impact": state.get("change_impact", {}),
        "config_artifacts": state.get("config_artifacts", {}),
        "findings": _summarized_findings(state),
        "requires_human_input": state.get("requires_human_input", []),
        "standards": _standards_payload(state.get("standards", [])),
    }
    if state.get("intake", {}).get("request_type") == "modify_existing":
        payload["current_config_summary"] = state.get("current_config_summary", {})
    return payload


def _judge_payload(state: FortiGateState) -> dict[str, Any]:
    """Compact judge packet: review artifacts plus builder/refine/repair provenance."""
    payload = _review_payload(state)
    payload.update(
        {
            "config_builder_review": state.get("config_builder_review", {}),
            "config_refinement": state.get("config_refinement", {}),
            "autonomous_repair_iterations": state.get("autonomous_repair_iterations", 0),
            "auto_fixed_items": state.get("auto_fixed_items", []),
        }
    )
    return payload


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


SECTION_ORDER = [
    "interfaces_dhcp",
    "fortiswitch",
    "wifi",
    "sdwan_routing",
    "objects_services",
    "firewall_policies",
]


SECTION_PROMPTS = {
    "interfaces_dhcp": (
        "Build only FortiGate physical interface, VLAN interface, zone, and DHCP server CLI blocks. "
        "Every LAN/VLAN in implementation_intent must have an interface decision and DHCP decision. "
        "Use placeholders and requires_human_input for unknown addressing, parent interfaces, DHCP ranges, or DNS values."
    ),
    "fortiswitch": (
        "Build only FortiSwitch/FortiLink managed switch CLI blocks and notes. Include FortiLink interface assumptions, "
        "switch-controller VLAN mappings, port profile intent, native/allowed VLAN placeholders, LLDP/STP/PoE notes, "
        "and requires_human_input for switch serials, uplink ports, port maps, or PoE assignments."
    ),
    "wifi": (
        "Build only FortiAP/wireless controller CLI blocks and notes. Include SSIDs, tunnel/bridge mode, VLAN mappings, "
        "guest isolation, AP profiles, and firewall policy dependencies. Put WPA keys, RADIUS servers, captive portal "
        "details, AP groups, and AP serials in requires_human_input when not supplied."
    ),
    "sdwan_routing": (
        "Build only SD-WAN and routing CLI blocks. Include SD-WAN members, zones, health checks, steering services, "
        "default route behavior, and static routes. Do not invent public gateways or circuit details."
    ),
    "objects_services": (
        "Build only firewall address objects, address groups, service objects, VIPs, and IP pools required by the intent "
        "and policy matrix. Prefer named objects over raw all, and list every created object in objects_defined."
    ),
    "firewall_policies": (
        "Build only firewall policy CLI blocks from the policy matrix. Use least privilege, explicit logging, NAT when "
        "required, inspection profiles where appropriate, and explicit deny/segmentation policies for guest/IoT/WiFi."
    ),
}


def _section_contract_prompt(section_name: str) -> str:
    return (
        "Return only JSON with keys section_name, cli_blocks, objects_defined, references_required, assumptions, "
        "requires_human_input, validation_notes. Build one FortiGate configuration section only; do not generate unrelated "
        f"sections. section_name must be {section_name}. Use implementation_intent as the source of truth. Do not invent "
        "site-specific values, secrets, public IPs, gateways, passwords, PSKs, RADIUS servers, syslog/SNMP/DNS targets, "
        "switch serials, AP serials, or port maps. Put unknowns in requires_human_input and use clear placeholders in CLI. "
        "Every CLI reference should be listed in references_required, and every object/interface/service created here should "
        f"be listed in objects_defined. {SECTION_PROMPTS[section_name]}"
    )


def _normalize_section(section_name: str, data: dict[str, Any]) -> dict[str, Any]:
    section = data.get("config_section") if isinstance(data.get("config_section"), dict) else data
    return {
        "section_name": str(section.get("section_name") or section_name),
        "cli_blocks": _as_list(section.get("cli_blocks")),
        "objects_defined": _as_list(section.get("objects_defined")),
        "references_required": _as_list(section.get("references_required")),
        "assumptions": _as_list(section.get("assumptions")),
        "requires_human_input": _as_list(section.get("requires_human_input")),
        "validation_notes": _as_list(section.get("validation_notes")),
        "repair_notes": _as_list(section.get("repair_notes")),
    }


def _local_repair_findings(state: FortiGateState) -> list[str]:
    return [
        *state.get("section_validation_report", {}).get("blocking_issues", []),
        *state.get("section_validation_report", {}).get("warnings", []),
        *state.get("cli_completeness_report", {}).get("blocking_issues", []),
        *state.get("cli_completeness_report", {}).get("warnings", []),
        *state.get("validation_report", {}).get("blocking_issues", []),
        *state.get("validation_report", {}).get("warnings", []),
        *state.get("standards_report", {}).get("warnings", []),
        *state.get("risk_report", {}).get("warnings", []),
    ]


def _finding_sections(finding: str) -> list[str]:
    lowered = finding.lower()
    matches: list[str] = []
    keyword_map = {
        "interfaces_dhcp": ("interface", "dhcp", "vlan", "zone", "subnet", "netmask", "gateway"),
        "fortiswitch": ("fortiswitch", "fortilink", "switch", "port"),
        "wifi": ("wifi", "wireless", "ssid", "ap_", "ap ", "radius", "wpa"),
        "sdwan_routing": ("sd-wan", "sdwan", "route", "routing", "health", "sla", "wan"),
        "objects_services": ("object", "address", "service", "vip", "pool", "profile"),
        "firewall_policies": ("policy", "policies", "srcaddr", "dstaddr", "srcintf", "dstintf", "logging", "deny"),
    }
    for section_name, keywords in keyword_map.items():
        if any(keyword in lowered for keyword in keywords):
            matches.append(section_name)
    if "references that are not listed" in lowered or "undefined" in lowered or "not defined" in lowered:
        for section_name in SECTION_ORDER:
            if section_name in lowered and section_name not in matches:
                matches.append(section_name)
    return matches or ["firewall_policies"]


def _is_placeholder_only_finding(finding: str) -> bool:
    lowered = finding.lower()
    placeholders = re.findall(r"<[^>\n]+>", finding)
    return bool(placeholders) and not any(marker in lowered for marker in ("syntax", "invalid", "missing section", "required section"))


def _plan_repair_tasks(state: FortiGateState) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    judge_auto_fixable, judge_requires_human = _classify_judge_items(state.get("judge_report", {}))
    findings = [
        *_local_repair_findings(state),
        *judge_auto_fixable,
        *[f"requires human input: {item}" for item in judge_requires_human],
    ]
    for finding in findings:
        text = str(finding).strip()
        if not text:
            continue
        deterministic = _is_placeholder_only_finding(text) or text.lower().startswith("requires human input:")
        placeholders = sorted(set(re.findall(r"<[^>\n]+>", text)))
        for section_name in _finding_sections(text):
            key = (section_name, text.lower())
            if key in seen:
                continue
            seen.add(key)
            tasks.append(
                {
                    "task_id": f"repair-{len(tasks) + 1}",
                    "section_name": section_name,
                    "finding": text,
                    "repair_type": "placeholder_review" if deterministic else "section_repair",
                    "requires_model": not deterministic,
                    "requires_human_input": [f"{section_name}: confirm value for {item}" for item in placeholders] or ([text] if deterministic else []),
                }
            )
    return tasks


def _section_intent_slice(state: FortiGateState, section_name: str) -> dict[str, Any]:
    intent = state.get("implementation_intent", {})
    if section_name == "interfaces_dhcp":
        keys = ("interface_inventory", "dhcp_plan", "assumptions", "requires_human_input")
    elif section_name == "fortiswitch":
        keys = ("fortiswitch_plan", "interface_inventory", "assumptions", "requires_human_input")
    elif section_name == "wifi":
        keys = ("wifi_plan", "interface_inventory", "policy_matrix", "assumptions", "requires_human_input")
    elif section_name == "sdwan_routing":
        keys = ("sdwan_plan", "interface_inventory", "assumptions", "requires_human_input")
    elif section_name == "objects_services":
        keys = ("object_inventory", "policy_matrix", "assumptions", "requires_human_input")
    else:
        keys = ("policy_matrix", "object_inventory", "logging_plan", "assumptions", "requires_human_input")
    return {key: intent.get(key) for key in keys if key in intent}


def _section_dependency_context(state: FortiGateState, section_name: str) -> dict[str, Any]:
    sections = state.get("config_sections", {})
    dependencies: dict[str, Any] = {}
    if section_name == "firewall_policies":
        dependencies["interfaces_dhcp"] = sections.get("interfaces_dhcp", {})
        dependencies["objects_services"] = sections.get("objects_services", {})
    elif section_name == "objects_services":
        dependencies["interfaces_dhcp"] = sections.get("interfaces_dhcp", {})
        dependencies["firewall_policy_intent"] = state.get("intake", {}).get("firewall_policy_intent", "")
    elif section_name in {"wifi", "fortiswitch", "sdwan_routing"}:
        dependencies["interfaces_dhcp"] = sections.get("interfaces_dhcp", {})
    return dependencies


def _section_standards_payload(state: FortiGateState, section_name: str) -> list[dict[str, Any]]:
    topic_keywords = {
        "interfaces_dhcp": ("lan", "interface", "dhcp", "segmentation"),
        "fortiswitch": ("switch", "fortilink", "lan"),
        "wifi": ("wifi", "wireless", "guest"),
        "sdwan_routing": ("sdwan", "wan", "route"),
        "objects_services": ("object", "service", "firewall"),
        "firewall_policies": ("policy", "firewall", "segmentation"),
    }.get(section_name, ())
    selected = []
    for item in state.get("standards", []):
        haystack = f"{item.get('topic', '')} {item.get('document', '')} {item.get('text', '')}".lower()
        if any(keyword in haystack for keyword in topic_keywords):
            selected.append(item)
        if len(selected) >= 3:
            break
    return _standards_payload(selected or state.get("standards", [])[:2])


def _merge_repaired_sections(state: FortiGateState, repaired_sections: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    sections = {**state.get("config_sections", {})}
    for section_name, section in repaired_sections.items():
        if section.get("cli_blocks"):
            sections[section_name] = section
    merged_state = {**state, "config_sections": sections}
    return sections, _merge_section_artifacts(merged_state)


def _merge_section_artifacts(state: FortiGateState) -> dict[str, Any]:
    sections = state.get("config_sections", {})
    cli_blocks: list[str] = []
    assumptions: list[str] = []
    human_inputs: list[str] = []
    implementation_notes: list[str] = []
    object_tables: dict[str, Any] = {}
    for section_name in SECTION_ORDER:
        section = sections.get(section_name)
        if not isinstance(section, dict):
            continue
        section_cli = [block for block in _as_list(section.get("cli_blocks")) if block.strip()]
        if section_cli:
            cli_blocks.append(f"# --- {section_name} ---")
            cli_blocks.extend(section_cli)
        object_tables[section_name] = section.get("objects_defined", [])
        assumptions.extend(f"{section_name}: {item}" for item in _as_list(section.get("assumptions")))
        human_inputs.extend(f"{section_name}: {item}" for item in _as_list(section.get("requires_human_input")))
        implementation_notes.extend(f"{section_name}: {item}" for item in _as_list(section.get("validation_notes")))
    return {
        "cli_config": "\n".join(cli_blocks).strip() or _fallback_cli(state.get("intake", {})),
        "object_tables": object_tables,
        "policy_table": sections.get("firewall_policies", {}).get("validation_notes", []) if isinstance(sections.get("firewall_policies"), dict) else [],
        "rollback_plan": [
            "Review generated section boundaries before implementation.",
            "Back up the current FortiGate configuration before applying any block.",
            "Apply changes during an approved maintenance window and revert by restoring the prior configuration if validation fails.",
        ],
        "assumptions": assumptions,
        "standards_citations": [],
        "implementation_notes": implementation_notes,
        "sectional_generation": True,
        "safety_label": "DRAFT ONLY - NOT APPLIED TO DEVICE",
        "requires_human_input": human_inputs,
    }


def build_fortigate_graph(
    model: ChatOpenAI,
    checkpointer: AsyncRedisSaver,
    clarification_interrupt: bool = False,
    human_review_interrupt: bool = False,
    design_model: ChatOpenAI | None = None,
    design_model_name: str = "",
    intent_model: ChatOpenAI | None = None,
    intent_model_name: str = "",
    judge_model: ChatOpenAI | None = None,
    judge_model_name: str = "",
    builder_review_model: ChatOpenAI | None = None,
    builder_review_model_name: str = "",
    builder_review_mode: str = "pre_refine",
    autonomous_repair_model: ChatOpenAI | None = None,
    autonomous_repair_model_name: str = "",
    config_refiner_model: ChatOpenAI | None = None,
    config_refiner_model_name: str = "",
    config_refinement_mode: str = "pre_judge",
    context_fallback_model: ChatOpenAI | None = None,
    context_fallback_model_name: str = "",
    autonomous_repair_limit: int = 2,
    sectional_generation_enabled: bool = False,
    renderer_enabled: bool = False,
    config_model_model: ChatOpenAI | None = None,
    config_model_model_name: str = "",
):
    normalized_builder_review_mode = builder_review_mode if builder_review_mode in {"off", "pre_refine"} else "pre_refine"
    normalized_refinement_mode = config_refinement_mode if config_refinement_mode in {"off", "pre_judge"} else "pre_judge"
    # Structuring/design nodes use a tighter-token design model when provided; CLI generation
    # nodes keep using the generous generation model.
    design_llm = design_model or model
    active_design_model_name = design_model_name or getattr(design_llm, "model_name", "configured-model")
    # implementation_intent can run on its own model knob (bake-off winner differs from design);
    # falls back to the design model when no dedicated intent model is configured.
    intent_llm = intent_model or design_llm
    active_intent_model_name = intent_model_name or active_design_model_name or getattr(intent_llm, "model_name", "configured-model")
    # The renderer path builds a typed config model with a structuring LLM (defaults to the
    # intent model). Rendering itself is deterministic and uses no model.
    render_llm = config_model_model or intent_llm
    active_render_model_name = config_model_model_name or getattr(render_llm, "model_name", active_intent_model_name)

    async def ainvoke_with_context_fallback(model_to_use: ChatOpenAI, messages: list[Any], model_name: str, run_name: str | None = None) -> tuple[Any, str]:
        # run_name names this generation in Langfuse (otherwise everything logs as "ChatOpenAI").
        config = {"run_name": run_name} if run_name else None
        try:
            return await model_to_use.ainvoke(messages, config=config), model_name
        except Exception as exc:
            if context_fallback_model is None or not _is_context_window_error(exc):
                raise
            fallback_name = context_fallback_model_name or getattr(context_fallback_model, "model_name", "context-fallback-model")
            return await context_fallback_model.ainvoke(messages, config=config), fallback_name

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
        # Config parsing and standards retrieval are independent; run them concurrently.
        summary_obj, standards_chunks = await asyncio.gather(
            asyncio.to_thread(parse_fortigate_config, state.get("existing_config", "")),
            asyncio.to_thread(retrieve_fortigate_standards, state.get("intake", {}), 10),
        )
        summary = summary_obj.model_dump()
        standards = [chunk.model_dump() for chunk in standards_chunks]
        return _trace_update(
            state,
            {"current_config_summary": summary, "standards": standards},
            "parse_existing_config",
            started_at,
            started_perf,
            f"Parsed config with {summary.get('raw_line_count', 0)} line(s); retrieved {len(standards)} standard chunk(s).",
        )

    async def retrieve_standards(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        if state.get("standards"):
            return _trace_update(state, {}, "retrieve_standards", started_at, started_perf, "Standards already retrieved upstream.", branch="cached")
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
        response, _ = await ainvoke_with_context_fallback(
            design_llm,
            [
                SystemMessage(content="Return only JSON with key logical_design. Build a vendor-neutral network/security design from the intake, current config summary, and company standards."),
                HumanMessage(content=json.dumps({"intake": state.get("intake", {}), "current_config_summary": state.get("current_config_summary", {}), "standards": _standards_payload(state.get("standards", []))}, indent=2)),
            ],
            active_design_model_name,
            run_name="build_logical_design",
        )
        data = _extract_json_object(str(response.content))
        design = data.get("logical_design") if isinstance(data.get("logical_design"), dict) else data
        return _trace_update(state, {"logical_design": design or {}}, "build_logical_design", started_at, started_perf, "Built logical design.")

    async def build_fortigate_design(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        response, _ = await ainvoke_with_context_fallback(
            design_llm,
            [
                SystemMessage(content="Return only JSON with key fortigate_design. Map the logical design to FortiGate constructs: interfaces, zones, SD-WAN, routes, objects, policies, NAT/VIP, VPN, logging, HA."),
                HumanMessage(content=json.dumps({"intake": state.get("intake", {}), "logical_design": state.get("logical_design", {}), "current_config_summary": state.get("current_config_summary", {}), "standards": _standards_payload(state.get("standards", []))}, indent=2)),
            ],
            active_design_model_name,
            run_name="build_fortigate_design",
        )
        data = _extract_json_object(str(response.content))
        design = data.get("fortigate_design") if isinstance(data.get("fortigate_design"), dict) else data
        return _trace_update(state, {"fortigate_design": design or {}}, "build_fortigate_design", started_at, started_perf, "Mapped logical design to FortiGate constructs.")

    async def build_implementation_intent(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        response, _ = await ainvoke_with_context_fallback(
            intent_llm,
            [
                SystemMessage(
                    content=(
                        "Return only JSON with key implementation_intent. Build a FortiGate implementation contract that the CLI "
                        "must satisfy before any config is generated. Include keys: interface_inventory, dhcp_plan, sdwan_plan, "
                        "fortiswitch_plan, wifi_plan, policy_matrix, object_inventory, logging_plan, assumptions, "
                        "requires_human_input. For every LAN/VLAN, "
                        "include name, vlan_id when known, subnet, gateway, zone, role, and dhcp_mode. For DHCP, decide enable, "
                        "disable, or needs_human_input; enable DHCP for user/guest VLANs when subnet and gateway are known unless "
                        "the intake says otherwise. For dual-WAN, include SD-WAN members, health_checks, steering_rules, and "
                        "default_route_behavior. For FortiSwitch, include FortiLink, VLAN/port profile, and port-map decisions "
                        "or explicit human inputs. For WiFi, include SSIDs, security mode, VLAN mappings, guest isolation, AP "
                        "profiles, and required human inputs such as PSKs/RADIUS/AP serials. For firewall policies, build a complete policy matrix with source, destination, "
                        "service, action, nat, logging, inspection_profile, and rationale. Put missing site-specific values in "
                        "requires_human_input rather than inventing them."
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        {
                            "intake": state.get("intake", {}),
                            "logical_design": state.get("logical_design", {}),
                            "fortigate_design": state.get("fortigate_design", {}),
                            "current_config_summary": state.get("current_config_summary", {}),
                            "standards": _standards_payload(state.get("standards", [])),
                        },
                        indent=2,
                    )
                ),
            ],
            active_intent_model_name,
            run_name="build_implementation_intent",
        )
        data = _extract_json_object(str(response.content))
        intent = data.get("implementation_intent") if isinstance(data.get("implementation_intent"), dict) else data
        return _trace_update(
            state,
            {
                "implementation_intent": intent or {},
                "requires_human_input": [str(item) for item in _as_list((intent or {}).get("requires_human_input"))],
            },
            "build_implementation_intent",
            started_at,
            started_perf,
            "Built structured implementation intent contract.",
        )

    async def check_intent_contract(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        report = check_intent_completeness(state).model_dump()
        return _trace_update(
            state,
            {"intent_completeness_report": report},
            "check_intent_contract",
            started_at,
            started_perf,
            f"Intent completeness produced {len(report.get('warnings', []))} warning(s).",
        )

    async def repair_implementation_intent(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        response = await intent_llm.ainvoke(
            [
                SystemMessage(
                    content=(
                        "Return only JSON with key implementation_intent. Repair the FortiGate implementation contract using "
                        "the completeness report. Fill auto-fixable gaps such as DHCP decisions, SD-WAN members/health checks, "
                        "policy matrix rows, object inventory, and logging plan. Do not invent real site values; keep those in "
                        "requires_human_input."
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        {
                            "intake": state.get("intake", {}),
                            "logical_design": state.get("logical_design", {}),
                            "fortigate_design": state.get("fortigate_design", {}),
                            "implementation_intent": state.get("implementation_intent", {}),
                            "intent_completeness_report": state.get("intent_completeness_report", {}),
                            "standards": _standards_payload(state.get("standards", [])),
                        },
                        indent=2,
                    )
                ),
            ],
            config={"run_name": "repair_implementation_intent"},
        )
        data = _extract_json_object(str(response.content))
        intent = data.get("implementation_intent") if isinstance(data.get("implementation_intent"), dict) else data
        fixed_items = [f"intent: {item}" for item in state.get("intent_completeness_report", {}).get("warnings", [])]
        return _trace_update(
            state,
            {
                "implementation_intent": intent or state.get("implementation_intent", {}),
                "auto_fixed_items": [*state.get("auto_fixed_items", []), *fixed_items],
                "requires_human_input": [str(item) for item in _as_list((intent or {}).get("requires_human_input"))] or state.get("requires_human_input", []),
                "intent_repaired": True,
            },
            "repair_implementation_intent",
            started_at,
            started_perf,
            f"Repaired implementation intent for {len(fixed_items)} item(s).",
        )

    def route_after_intent_check(state: FortiGateState) -> str:
        report = state.get("intent_completeness_report", {})
        warnings = report.get("warnings", [])
        if warnings and not state.get("intent_repaired"):
            return "repair"
        return "continue"

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
        response, used_model_name = await ainvoke_with_context_fallback(
            model,
            [
                SystemMessage(
                    content=(
                        "Return only JSON with key config_artifacts. Build a high-quality draft FortiGate configuration package. "
                        "config_artifacts must include cli_config, object_tables, policy_table, rollback_plan, assumptions, "
                        "standards_citations, and implementation_notes. Generate ordered FortiOS CLI sections that match the "
                        "FortiGate design: system settings, interfaces/VLANs, zones, address and service objects, SD-WAN, routes, "
                        "NAT/VIP, VPN, HA, logging, and firewall policies when applicable. Prefer named objects and zones over raw "
                        "'all', use least-privilege policies, enable logging on allow policies, and make object names traceable to "
                        "the site and design intent. For modify_existing requests, avoid destructive changes unless explicitly "
                        "requested and include additive change blocks where possible. Do not invent secrets, public IPs, gateways, "
                        "VPN PSKs, DNS servers, syslog servers, or SNMP communities; place unknown values in assumptions and use "
                        "clear placeholders in the CLI. The CLI must be a draft only and must not claim it was applied."
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        {
                            "intake": state.get("intake", {}),
                            "current_config_summary": state.get("current_config_summary", {}),
                            "fortigate_design": state.get("fortigate_design", {}),
                            "implementation_intent": state.get("implementation_intent", {}),
                            "intent_completeness_report": state.get("intent_completeness_report", {}),
                            "change_impact": state.get("change_impact", {}),
                            "standards": _standards_payload(state.get("standards", [])),
                        },
                        indent=2,
                    )
                ),
            ]
            ,
            getattr(model, "model_name", "configured-model"),
            run_name="generate_config_artifacts",
        )
        data = _extract_json_object(str(response.content))
        artifacts = data.get("config_artifacts") if isinstance(data.get("config_artifacts"), dict) else data
        if not artifacts.get("cli_config"):
            artifacts["cli_config"] = _fallback_cli(state.get("intake", {}))
        artifacts.setdefault("safety_label", "DRAFT ONLY - NOT APPLIED TO DEVICE")
        return _trace_update(state, {"config_artifacts": artifacts}, "generate_config_artifacts", started_at, started_perf, f"Generated draft config artifacts with {used_model_name}.")

    async def build_config_section(state: FortiGateState, section_name: str) -> FortiGateState:
        started_at, started_perf = _trace_start()
        response, used_model_name = await ainvoke_with_context_fallback(
            model,
            [
                SystemMessage(content=_section_contract_prompt(section_name)),
                HumanMessage(
                    content=json.dumps(
                        {
                            "section_name": section_name,
                            "intake": state.get("intake", {}),
                            "current_config_summary": state.get("current_config_summary", {}),
                            "fortigate_design": state.get("fortigate_design", {}),
                            "implementation_intent": _section_intent_slice(state, section_name),
                            "dependency_context": _section_dependency_context(state, section_name),
                            "standards": _section_standards_payload(state, section_name),
                        },
                        indent=2,
                    )
                ),
            ]
            ,
            getattr(model, "model_name", "configured-model"),
            run_name=f"build_{section_name}_section",
        )
        data = _extract_json_object(str(response.content))
        section = _normalize_section(section_name, data)
        sections = {**state.get("config_sections", {}), section_name: section}
        human_inputs = [str(item) for item in section.get("requires_human_input", [])]
        return _trace_update(
            state,
            {
                "config_sections": sections,
                "requires_human_input": list(dict.fromkeys([*state.get("requires_human_input", []), *human_inputs])),
            },
            f"build_{section_name}_section",
            started_at,
            started_perf,
            f"Built sectional FortiGate config output for {section_name} with {used_model_name}.",
        )

    async def build_interfaces_dhcp_section(state: FortiGateState) -> FortiGateState:
        return await build_config_section(state, "interfaces_dhcp")

    async def build_fortiswitch_section(state: FortiGateState) -> FortiGateState:
        return await build_config_section(state, "fortiswitch")

    async def build_wifi_section(state: FortiGateState) -> FortiGateState:
        return await build_config_section(state, "wifi")

    async def build_sdwan_routing_section(state: FortiGateState) -> FortiGateState:
        return await build_config_section(state, "sdwan_routing")

    async def build_objects_services_section(state: FortiGateState) -> FortiGateState:
        return await build_config_section(state, "objects_services")

    async def build_firewall_policies_section(state: FortiGateState) -> FortiGateState:
        return await build_config_section(state, "firewall_policies")

    async def validate_config_sections(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        report = check_section_completeness(state).model_dump()
        return _trace_update(
            state,
            {"section_validation_report": report},
            "validate_config_sections",
            started_at,
            started_perf,
            f"Section validation produced {len(report.get('warnings', []))} warning(s) and {len(report.get('blocking_issues', []))} blocker(s).",
        )

    async def assemble_sectional_config_artifacts(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        artifacts = _merge_section_artifacts(state)
        return _trace_update(
            state,
            {"config_artifacts": artifacts},
            "assemble_sectional_config_artifacts",
            started_at,
            started_perf,
            f"Deterministically assembled {len(state.get('config_sections', {}))} FortiGate config section(s).",
        )

    async def _render_invoke(system_text: str, human_text: str, run_name: str) -> str:
        response, _ = await ainvoke_with_context_fallback(
            render_llm,
            [SystemMessage(content=system_text), HumanMessage(content=human_text)],
            active_render_model_name,
            run_name=run_name,
        )
        return str(response.content)

    def _render_context(state: FortiGateState, *, judge: bool = False) -> dict[str, Any]:
        context = {
            "intake": state.get("intake", {}),
            "fortigate_design": state.get("fortigate_design", {}),
            "implementation_intent": state.get("implementation_intent", {}),
            "intent_completeness_report": state.get("intent_completeness_report", {}),
            "change_impact": state.get("change_impact", {}),
            "standards": _standards_payload(state.get("standards", [])),
        }
        if judge:
            context["previous_config_model"] = state.get("config_model", {})
            context["judge_report"] = state.get("judge_report", {})
            context["validation_report"] = state.get("validation_report", {})
            context["standards_report"] = state.get("standards_report", {})
            context["risk_report"] = state.get("risk_report", {})
        return context

    def _render_fortios_version(state: FortiGateState) -> str:
        return normalize_fortios_version(str(state.get("intake", {}).get("fortios_version", "")))

    async def build_config_model_node(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        model_obj, errors = await build_fortigate_config_model(
            _render_invoke,
            _render_context(state),
            fortios_version=_render_fortios_version(state),
            retries=1,
        )
        if model_obj is None:
            return _trace_update(
                state,
                {"renderer_fallback": True, "renderer_active": False},
                "build_config_model",
                started_at,
                started_perf,
                f"Config model build failed after retries ({len(errors)} error(s)); falling back to LLM CLI path.",
                branch="fallback",
            )
        return _trace_update(
            state,
            {"config_model": model_obj.model_dump(mode="json"), "renderer_active": True, "renderer_fallback": False},
            "build_config_model",
            started_at,
            started_perf,
            "Built structured FortiGate config model from implementation intent.",
        )

    async def render_config_artifacts(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        try:
            model_obj = FortiGateConfigModel(**state.get("config_model", {}))
            artifacts = render_config(model_obj)
        except Exception as exc:
            artifacts = {
                "cli_config": _fallback_cli(state.get("intake", {})),
                "safety_label": "DRAFT ONLY - NOT APPLIED TO DEVICE",
                "rendered_by": "deterministic-renderer@error",
            }
            return _trace_update(
                state,
                {"config_artifacts": artifacts, "renderer_fallback": True},
                "render_config_artifacts",
                started_at,
                started_perf,
                f"Deterministic render failed unexpectedly: {exc}",
                branch="fallback",
            )
        human_inputs = [str(item) for item in artifacts.get("requires_human_input", [])]
        return _trace_update(
            state,
            {
                "config_artifacts": artifacts,
                "requires_human_input": list(dict.fromkeys([*state.get("requires_human_input", []), *human_inputs])),
            },
            "render_config_artifacts",
            started_at,
            started_perf,
            f"Rendered FortiGate CLI deterministically ({artifacts.get('rendered_by', '')}).",
        )

    def route_after_change_impact(state: FortiGateState) -> str:
        if renderer_enabled:
            return "renderer"
        return "sectional" if sectional_generation_enabled else "monolith"

    def route_after_build_model(state: FortiGateState) -> str:
        if state.get("renderer_fallback"):
            return "sectional" if sectional_generation_enabled else "monolith"
        return "render"

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

    async def check_cli_contract(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        report = check_cli_completeness(state).model_dump()
        return _trace_update(
            state,
            {"cli_completeness_report": report},
            "check_cli_contract",
            started_at,
            started_perf,
            f"CLI completeness produced {len(report.get('warnings', []))} warning(s) and {len(report.get('blocking_issues', []))} blocker(s).",
        )

    async def _repair_section_task(
        state: FortiGateState,
        task: dict[str, Any],
        repair_model: ChatOpenAI,
        repair_model_name: str,
    ) -> tuple[dict[str, Any], str]:
        section_name = str(task.get("section_name") or "")
        existing_section = state.get("config_sections", {}).get(section_name, {})
        response, used_model_name = await ainvoke_with_context_fallback(
            repair_model,
            [
                SystemMessage(
                    content=(
                        "Return only JSON with keys section_name, cli_blocks, objects_defined, references_required, "
                        "assumptions, requires_human_input, validation_notes, repair_notes. Repair only the requested "
                        "FortiGate section. Do not rewrite unrelated sections or return full config_artifacts. Preserve "
                        "unknown real-world values as placeholders and list them in requires_human_input."
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        {
                            "task": task,
                            "section_name": section_name,
                            "existing_section": existing_section,
                            "implementation_intent": _section_intent_slice(state, section_name),
                            "dependency_context": _section_dependency_context(state, section_name),
                            "reports": {
                                "section_validation_report": state.get("section_validation_report", {}),
                                "cli_completeness_report": state.get("cli_completeness_report", {}),
                                "validation_report": state.get("validation_report", {}),
                                "standards_report": state.get("standards_report", {}),
                                "risk_report": state.get("risk_report", {}),
                            },
                            "standards": _section_standards_payload(state, section_name),
                        },
                        indent=2,
                    )
                ),
            ],
            repair_model_name,
            run_name=f"repair_{section_name}_section",
        )
        data = _extract_json_object(str(response.content))
        return _normalize_section(section_name, data), used_model_name

    async def autonomous_repair_config_artifacts(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        iteration = int(state.get("autonomous_repair_iterations", 0) or 0) + 1
        repair_model = autonomous_repair_model or model
        repair_model_name = autonomous_repair_model_name or getattr(model, "model_name", "configured-model")
        report = state.get("judge_report", {})
        auto_fixable, requires_human = _classify_judge_items(report)
        tasks = _plan_repair_tasks(state)
        model_tasks = [task for task in tasks if task.get("requires_model")][:4]
        deterministic_human_inputs = [
            item
            for task in tasks
            if not task.get("requires_model")
            for item in _as_list(task.get("requires_human_input"))
        ]
        unique_tasks: list[dict[str, Any]] = []
        seen_sections: set[str] = set()
        for task in model_tasks:
            section_name = str(task.get("section_name") or "")
            if section_name in seen_sections:
                continue
            seen_sections.add(section_name)
            unique_tasks.append(task)
        repaired_sections: dict[str, dict[str, Any]] = {}
        used_models: list[str] = []
        if unique_tasks:
            # Section repairs are independent; run them concurrently to cut wall-clock time.
            results = await asyncio.gather(
                *(_repair_section_task(state, task, repair_model, repair_model_name) for task in unique_tasks)
            )
            for task, (section, used_model_name) in zip(unique_tasks, results):
                section_name = str(task.get("section_name") or "")
                if section.get("cli_blocks"):
                    section["repaired_by_model"] = used_model_name
                    repaired_sections[section_name] = section
                    used_models.append(used_model_name)
        if repaired_sections:
            sections, artifacts = _merge_repaired_sections(state, repaired_sections)
            repaired_notes = [
                f"{task.get('section_name')}: {task.get('finding')}"
                for task in model_tasks
                if task.get("section_name") in repaired_sections
            ]
            return _trace_update(
                state,
                {
                    "config_sections": sections,
                    "config_artifacts": artifacts,
                    "autonomous_repair_iterations": iteration,
                    "autonomous_repair_tasks": tasks,
                    "auto_fixed_items": [*state.get("auto_fixed_items", []), *(repaired_notes or auto_fixable)],
                    "requires_human_input": list(dict.fromkeys([*state.get("requires_human_input", []), *requires_human, *deterministic_human_inputs])),
                },
                "autonomous_repair_config_artifacts",
                started_at,
                started_perf,
                f"Autonomous repair pass {iteration} repaired {len(repaired_sections)} section task(s).",
                branch=",".join(sorted(set(used_models))) or repair_model_name,
            )
        human_inputs = [
            *requires_human,
            *deterministic_human_inputs,
            *[
                item
                for task in tasks
                for item in _as_list(task.get("requires_human_input"))
                if item not in deterministic_human_inputs
            ],
        ]
        return _trace_update(
            state,
            {
                "autonomous_repair_iterations": iteration,
                "autonomous_repair_tasks": tasks,
                "auto_fixed_items": [*state.get("auto_fixed_items", []), *auto_fixable],
                "requires_human_input": list(dict.fromkeys([*state.get("requires_human_input", []), *human_inputs])),
            },
            "autonomous_repair_config_artifacts",
            started_at,
            started_perf,
            f"Autonomous repair pass {iteration} planned {len(tasks)} task(s); no model section rewrite was required.",
            branch="deterministic" if tasks else "no-op",
        )

    async def builder_review_config_artifacts(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        reviewer = builder_review_model or model
        reviewer_name = builder_review_model_name or getattr(model, "model_name", "configured-model")
        response, used_model_name = await ainvoke_with_context_fallback(
            reviewer,
            [
                SystemMessage(
                    content=(
                        "Return only JSON with keys config_artifacts and config_builder_review. You are doing a thorough "
                        "network configuration, firewall policy, and security refactor of the initial FortiGate draft. Look for "
                        "missing objects, overbroad policies, missing logging, weak segmentation, incomplete SD-WAN/route behavior, "
                        "unsafe assumptions, and mismatch with the design or standards. Improve the CLI and package where you can, "
                        "but keep it artifact-only, preserve safety labels and rollback details, and do not invent secrets or site "
                        "values. Explain material improvements, unresolved assumptions, and any security tradeoffs in "
                        "config_builder_review."
                    )
                ),
                HumanMessage(content=json.dumps(_review_payload(state), indent=2)),
            ]
            ,
            reviewer_name,
            run_name="builder_review_config_artifacts",
        )
        data = _extract_json_object(str(response.content))
        artifacts = data.get("config_artifacts") if isinstance(data.get("config_artifacts"), dict) else {}
        review = data.get("config_builder_review") if isinstance(data.get("config_builder_review"), dict) else {}
        if artifacts.get("cli_config"):
            artifacts.setdefault("safety_label", "DRAFT ONLY - NOT APPLIED TO DEVICE")
            artifacts["builder_reviewed_by_model"] = used_model_name
            review.setdefault("model", used_model_name)
            review.setdefault("mode", normalized_builder_review_mode)
            return _trace_update(
                state,
                {"config_artifacts": artifacts, "config_builder_review": review},
                "builder_review_config_artifacts",
                started_at,
                started_perf,
                "Builder model thoroughly reviewed and refactored draft config artifacts.",
                branch=used_model_name,
            )
        return _trace_update(
            state,
            {"config_builder_review": {"model": reviewer_name, "mode": normalized_builder_review_mode, "status": "no_structured_review"}},
            "builder_review_config_artifacts",
            started_at,
            started_perf,
            "No structured builder review returned; kept prior artifacts.",
            branch="no-op",
        )

    async def refine_config_artifacts(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        refiner = config_refiner_model or judge_model or model
        refiner_name = config_refiner_model_name or judge_model_name or getattr(model, "model_name", "configured-model")
        response, used_model_name = await ainvoke_with_context_fallback(
            refiner,
            [
                SystemMessage(
                    content=(
                        "Return only JSON with keys config_artifacts and config_refinement. You are a senior FortiGate "
                        "configuration refiner. Improve the draft CLI before independent judge review by addressing validation, "
                        "standards, and risk findings. Preserve artifact-only language, rollback details, safety labels, standards "
                        "citations, and the user's design intent. Do not claim the config was applied. If the draft is already "
                        "appropriate, return it unchanged and explain that in config_refinement."
                    )
                ),
                HumanMessage(content=json.dumps(_review_payload(state), indent=2)),
            ]
            ,
            refiner_name,
            run_name="refine_config_artifacts",
        )
        data = _extract_json_object(str(response.content))
        artifacts = data.get("config_artifacts") if isinstance(data.get("config_artifacts"), dict) else {}
        refinement = data.get("config_refinement") if isinstance(data.get("config_refinement"), dict) else {}
        if artifacts.get("cli_config"):
            artifacts.setdefault("safety_label", "DRAFT ONLY - NOT APPLIED TO DEVICE")
            artifacts["refined_by_model"] = used_model_name
            refinement.setdefault("model", used_model_name)
            refinement.setdefault("mode", normalized_refinement_mode)
            return _trace_update(
                state,
                {"config_artifacts": artifacts, "config_refinement": refinement},
                "refine_config_artifacts",
                started_at,
                started_perf,
                "Refined draft config artifacts before judge review.",
                branch=used_model_name,
            )
        return _trace_update(
            state,
            {"config_refinement": {"model": refiner_name, "mode": normalized_refinement_mode, "status": "no_structured_refinement"}},
            "refine_config_artifacts",
            started_at,
            started_perf,
            "No structured refinement returned; kept prior artifacts.",
            branch="no-op",
        )

    async def frontier_model_judge(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        packet = _judge_payload(state)
        active_judge = judge_model or model
        active_judge_name = judge_model_name or getattr(model, "model_name", "configured-model")
        try:
            report = await run_frontier_judge(active_judge, packet, judge_model_name=active_judge_name)
        except Exception as exc:
            if context_fallback_model is None or not _is_context_window_error(exc):
                raise
            fallback_name = context_fallback_model_name or getattr(context_fallback_model, "model_name", "context-fallback-model")
            report = await run_frontier_judge(context_fallback_model, packet, judge_model_name=fallback_name)
        _, requires_human = _classify_judge_items(report.model_dump())
        return _trace_update(
            state,
            {
                "judge_report": report.model_dump(),
                "judge_iterations": int(state.get("judge_iterations", 0) or 0) + 1,
                "requires_human_input": list(dict.fromkeys([*state.get("requires_human_input", []), *requires_human])),
            },
            "frontier_model_judge",
            started_at,
            started_perf,
            f"Judge verdict: {report.verdict}.",
            branch=report.verdict,
        )

    def route_after_risk_review(state: FortiGateState) -> str:
        # Renderer path: syntax is guaranteed by construction, so skip builder review,
        # text refinement, and section repair (which all edit CLI text). Go straight to judge.
        if state.get("renderer_active"):
            return "judge"
        if normalized_builder_review_mode == "pre_refine" and not state.get("config_builder_review"):
            return "builder_review"
        if normalized_refinement_mode == "pre_judge" and not state.get("config_refinement"):
            return "refine"
        repair_tasks = _plan_repair_tasks(state)
        if repair_tasks and int(state.get("autonomous_repair_iterations", 0) or 0) < autonomous_repair_limit:
            return "autonomous_repair"
        return "judge"

    async def revise_after_judge(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        response, used_model_name = await ainvoke_with_context_fallback(
            model,
            [
                SystemMessage(content="Return only JSON with key config_artifacts. Revise the draft FortiGate artifacts to address the judge report while preserving safety labels and standards citations."),
                HumanMessage(content=json.dumps({"config_artifacts": state.get("config_artifacts", {}), "judge_report": state.get("judge_report", {}), "standards": _standards_payload(state.get("standards", []))}, indent=2)),
            ]
            ,
            getattr(model, "model_name", "configured-model"),
            run_name="revise_after_judge",
        )
        data = _extract_json_object(str(response.content))
        artifacts = data.get("config_artifacts") if isinstance(data.get("config_artifacts"), dict) else data
        if artifacts:
            artifacts.setdefault("safety_label", "DRAFT ONLY - NOT APPLIED TO DEVICE")
            return _trace_update(state, {"config_artifacts": artifacts}, "revise_after_judge", started_at, started_perf, f"Revised artifacts after judge feedback with {used_model_name}.")
        return _trace_update(state, {}, "revise_after_judge", started_at, started_perf, "No structured revision returned; kept prior artifacts.", branch="no-op")

    async def regenerate_config_after_judge(state: FortiGateState) -> FortiGateState:
        started_at, started_perf = _trace_start()
        # Renderer path: patch the structured model from judge feedback and re-render
        # deterministically. CLI text is never hand-edited, so the syntax guarantee holds.
        if state.get("renderer_active") and state.get("config_model"):
            model_obj, errors = await build_fortigate_config_model(
                _render_invoke,
                _render_context(state, judge=True),
                fortios_version=_render_fortios_version(state),
                retries=1,
                judge_context=state.get("judge_report", {}),
                run_name="patch_config_model",
            )
            if model_obj is None:
                return _trace_update(
                    state,
                    {},
                    "regenerate_config_after_judge",
                    started_at,
                    started_perf,
                    f"Patch-model failed ({len(errors)} error(s)); kept prior rendered artifacts.",
                    branch="no-op",
                )
            artifacts = render_config(model_obj)
            human_inputs = [str(item) for item in artifacts.get("requires_human_input", [])]
            return _trace_update(
                state,
                {
                    "config_model": model_obj.model_dump(mode="json"),
                    "config_artifacts": artifacts,
                    "requires_human_input": list(dict.fromkeys([*state.get("requires_human_input", []), *human_inputs])),
                },
                "regenerate_config_after_judge",
                started_at,
                started_perf,
                "Patched config model from judge feedback and re-rendered deterministically.",
                branch="patched-rerender",
            )
        response, used_model_name = await ainvoke_with_context_fallback(
            model,
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
            ,
            getattr(model, "model_name", "configured-model"),
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
                f"Regenerated config artifacts from scratch after broad judge feedback with {used_model_name}.",
                branch="regenerated",
            )
        return _trace_update(state, {}, "regenerate_config_after_judge", started_at, started_perf, "No structured regeneration returned; kept prior artifacts.", branch="no-op")

    def route_after_judge(state: FortiGateState) -> str:
        report = state.get("judge_report", {})
        verdict = report.get("verdict", "needs_revision")
        # Renderer path: all design fixes go through patch-model -> re-render (the "regenerate"
        # node is renderer-aware). Never use the CLI-editing repair/revise nodes.
        if state.get("renderer_active"):
            if verdict == "needs_revision" and int(state.get("judge_iterations", 0) or 0) < 2:
                return "regenerate"
            return "human_review" if human_review_interrupt else "finalize"
        review_item_count = _judge_review_item_count(report)
        auto_fixable, requires_human = _classify_judge_items(report)
        if verdict == "needs_revision" and auto_fixable and int(state.get("autonomous_repair_iterations", 0) or 0) < autonomous_repair_limit:
            if len(requires_human) < review_item_count:
                return "autonomous_repair"
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
    graph.add_node("build_implementation_intent", build_implementation_intent)
    graph.add_node("check_intent_contract", check_intent_contract)
    graph.add_node("repair_implementation_intent", repair_implementation_intent)
    graph.add_node("analyze_change_impact", analyze_change_impact)
    graph.add_node("generate_config_artifacts", generate_config_artifacts)
    graph.add_node("build_interfaces_dhcp_section", build_interfaces_dhcp_section)
    graph.add_node("build_fortiswitch_section", build_fortiswitch_section)
    graph.add_node("build_wifi_section", build_wifi_section)
    graph.add_node("build_sdwan_routing_section", build_sdwan_routing_section)
    graph.add_node("build_objects_services_section", build_objects_services_section)
    graph.add_node("build_firewall_policies_section", build_firewall_policies_section)
    graph.add_node("validate_config_sections", validate_config_sections)
    graph.add_node("assemble_sectional_config_artifacts", assemble_sectional_config_artifacts)
    graph.add_node("build_config_model", build_config_model_node)
    graph.add_node("render_config_artifacts", render_config_artifacts)
    graph.add_node("validate_config", validate_config)
    graph.add_node("check_standards", check_standards)
    graph.add_node("risk_review", risk_review)
    graph.add_node("check_cli_contract", check_cli_contract)
    graph.add_node("autonomous_repair_config_artifacts", autonomous_repair_config_artifacts)
    graph.add_node("builder_review_config_artifacts", builder_review_config_artifacts)
    graph.add_node("refine_config_artifacts", refine_config_artifacts)
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
    graph.add_edge("build_fortigate_design", "build_implementation_intent")
    graph.add_edge("build_implementation_intent", "check_intent_contract")
    graph.add_conditional_edges("check_intent_contract", route_after_intent_check, {"repair": "repair_implementation_intent", "continue": "analyze_change_impact"})
    graph.add_edge("repair_implementation_intent", "check_intent_contract")
    graph.add_conditional_edges("analyze_change_impact", route_after_change_impact, {"monolith": "generate_config_artifacts", "sectional": "build_interfaces_dhcp_section", "renderer": "build_config_model"})
    graph.add_conditional_edges("build_config_model", route_after_build_model, {"render": "render_config_artifacts", "sectional": "build_interfaces_dhcp_section", "monolith": "generate_config_artifacts"})
    graph.add_edge("render_config_artifacts", "validate_config")
    graph.add_edge("build_interfaces_dhcp_section", "build_fortiswitch_section")
    graph.add_edge("build_fortiswitch_section", "build_wifi_section")
    graph.add_edge("build_wifi_section", "build_sdwan_routing_section")
    graph.add_edge("build_sdwan_routing_section", "build_objects_services_section")
    graph.add_edge("build_objects_services_section", "build_firewall_policies_section")
    graph.add_edge("build_firewall_policies_section", "validate_config_sections")
    graph.add_edge("validate_config_sections", "assemble_sectional_config_artifacts")
    graph.add_edge("assemble_sectional_config_artifacts", "validate_config")
    graph.add_edge("generate_config_artifacts", "validate_config")
    graph.add_edge("validate_config", "check_standards")
    graph.add_edge("check_standards", "risk_review")
    graph.add_edge("risk_review", "check_cli_contract")
    graph.add_conditional_edges(
        "check_cli_contract",
        route_after_risk_review,
        {
            "builder_review": "builder_review_config_artifacts",
            "refine": "refine_config_artifacts",
            "autonomous_repair": "autonomous_repair_config_artifacts",
            "judge": "frontier_model_judge",
        },
    )
    graph.add_edge("builder_review_config_artifacts", "validate_config")
    graph.add_edge("refine_config_artifacts", "validate_config")
    graph.add_edge("autonomous_repair_config_artifacts", "validate_config")
    judge_routes = {
        "autonomous_repair": "autonomous_repair_config_artifacts",
        "revise": "revise_after_judge",
        "regenerate": "regenerate_config_after_judge",
        "finalize": "finalize_package",
    }
    if human_review_interrupt:
        judge_routes["human_review"] = "human_review_checkpoint"
    graph.add_conditional_edges("frontier_model_judge", route_after_judge, judge_routes)
    graph.add_edge("revise_after_judge", "validate_config")
    graph.add_edge("regenerate_config_after_judge", "validate_config")
    if human_review_interrupt:
        graph.add_edge("human_review_checkpoint", "finalize_package")
    graph.add_edge("finalize_package", END)
    return graph.compile(checkpointer=checkpointer)
