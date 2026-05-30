from __future__ import annotations

from typing import Any

from app.fortigate_models import FortiGateConfigSummary, FortiGateValidationReport


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return str(value)
    return str(value or "")


def _intent_vlans(intent: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = (
        intent.get("vlans")
        or intent.get("lan_vlans")
        or intent.get("interface_inventory")
        or intent.get("interfaces")
        or []
    )
    return [item for item in _as_list(candidates) if isinstance(item, dict)]


def check_intent_completeness(state: dict[str, Any]) -> FortiGateValidationReport:
    intent = state.get("implementation_intent") or {}
    intake = state.get("intake") or {}
    blocking: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []

    if not intent:
        blocking.append("No implementation intent contract was produced before CLI generation.")
        return FortiGateValidationReport(passed=False, blocking_issues=blocking, warnings=warnings, checks=checks)

    vlans = _intent_vlans(intent)
    if intake.get("lan_networks") and not vlans:
        warnings.append("LAN networks were provided, but implementation intent has no VLAN/interface inventory.")
    else:
        checks.append(f"Implementation intent includes {len(vlans)} interface/VLAN item(s).")

    for vlan in vlans:
        role = _as_text(vlan.get("role") or vlan.get("zone") or vlan.get("name")).lower()
        subnet = vlan.get("subnet") or vlan.get("network")
        gateway = vlan.get("gateway") or vlan.get("ip")
        dhcp = _as_text(vlan.get("dhcp") or vlan.get("dhcp_mode") or vlan.get("dhcp_service")).lower()
        is_user_vlan = any(token in role for token in ("corp", "staff", "user", "guest", "lan"))
        is_mgmt = "mgmt" in role or "management" in role
        if is_user_vlan and not is_mgmt and subnet and gateway and not dhcp:
            warnings.append(f"VLAN/interface {vlan.get('name') or vlan.get('vlan_id') or subnet} has subnet/gateway but no DHCP service decision.")
        if is_mgmt and not dhcp:
            checks.append(f"Management VLAN/interface {vlan.get('name') or subnet} has no DHCP decision; this may be intentional.")

    wan_count = len(_as_list(intake.get("wan_circuits")))
    sdwan = intent.get("sdwan_plan") or intent.get("sd_wan_plan") or {}
    if wan_count > 1:
        if not sdwan:
            warnings.append("Multiple WAN circuits were provided, but implementation intent has no SD-WAN plan.")
        else:
            checks.append("Implementation intent includes an SD-WAN plan.")
            for key in ("members", "health_checks", "steering_rules", "default_route_behavior"):
                if not sdwan.get(key):
                    warnings.append(f"SD-WAN plan is missing {key}.")

    policy_matrix = _as_list(intent.get("policy_matrix") or intent.get("firewall_policy_matrix"))
    if not policy_matrix:
        warnings.append("Implementation intent has no firewall policy matrix.")
    else:
        checks.append(f"Implementation intent includes {len(policy_matrix)} firewall policy matrix row(s).")
        for index, policy in enumerate(policy_matrix, start=1):
            if not isinstance(policy, dict):
                continue
            missing = [key for key in ("source", "destination", "service", "action", "logging") if not policy.get(key)]
            if missing:
                warnings.append(f"Policy matrix row {index} is missing: {', '.join(missing)}.")

    return FortiGateValidationReport(
        passed=not blocking,
        blocking_issues=blocking,
        warnings=warnings,
        checks=checks,
    )


def validate_config_artifacts(state: dict[str, Any]) -> FortiGateValidationReport:
    artifacts = state.get("config_artifacts") or {}
    intake = state.get("intake") or {}
    cli = str(artifacts.get("cli_config") or "")
    blocking: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []

    if not cli.strip():
        blocking.append("No FortiGate CLI configuration draft was produced.")
    else:
        checks.append("CLI configuration draft is present.")

    required_sections = ["config system interface", "config firewall policy"]
    if intake.get("wan_circuits"):
        required_sections.append("config system sdwan")
    for section in required_sections:
        if section in cli:
            checks.append(f"Required section found: {section}.")
        else:
            warnings.append(f"Expected section may be missing: {section}.")

    if "set action accept" in cli and "set logtraffic" not in cli:
        warnings.append("Accept firewall policies should explicitly configure logging.")
    if "set srcaddr all" in cli and "set dstaddr all" in cli and "set service ALL" in cli:
        warnings.append("Broad all-to-all policy detected; human reviewer should confirm this is intentional.")
    if "set admin-sport" not in cli and "config system global" in cli:
        warnings.append("Admin HTTPS port is not explicitly set in the generated global settings.")

    return FortiGateValidationReport(
        passed=not blocking,
        blocking_issues=blocking,
        warnings=warnings,
        checks=checks,
    )


def check_standards_compliance(state: dict[str, Any]) -> FortiGateValidationReport:
    standards = state.get("standards") or []
    artifacts = state.get("config_artifacts") or {}
    cli = str(artifacts.get("cli_config") or "")
    blocking: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []

    if standards:
        checks.append(f"Retrieved {len(standards)} standard chunk(s) for this design.")
    else:
        warnings.append("No company standards were retrieved; output must be treated as generic FortiGate guidance.")

    lower_standards = "\n".join(str(item.get("text") or "") for item in standards).lower()
    if "logging" in lower_standards and "set logtraffic" not in cli:
        warnings.append("Retrieved standards mention logging, but generated policies do not explicitly set logtraffic.")
    if ("sd-wan" in lower_standards or "sdwan" in lower_standards) and "config system sdwan" not in cli:
        warnings.append("Retrieved standards mention SD-WAN, but generated config does not include an SD-WAN section.")
    if "rollback" in lower_standards and not artifacts.get("rollback_plan"):
        warnings.append("Retrieved standards mention rollback, but no rollback plan was produced.")

    return FortiGateValidationReport(
        passed=not blocking,
        blocking_issues=blocking,
        warnings=warnings,
        checks=checks,
    )


def review_risk(state: dict[str, Any]) -> FortiGateValidationReport:
    intake = state.get("intake") or {}
    summary = state.get("current_config_summary") or {}
    if isinstance(summary, FortiGateConfigSummary):
        summary = summary.model_dump()
    blocking: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []

    if intake.get("request_type") == "modify_existing":
        if summary.get("raw_line_count"):
            checks.append("Existing configuration was parsed before change analysis.")
        else:
            blocking.append("Modify-existing request does not include a parsed existing configuration.")

    if not intake.get("change_window"):
        warnings.append("No change window was provided.")
    if not intake.get("rollback_expectations"):
        warnings.append("No rollback expectation was provided.")
    if intake.get("vpn_requirements") and "vpn" not in str(state.get("config_artifacts", {})).lower():
        warnings.append("VPN requirements were provided, but generated artifacts may not include VPN configuration.")

    return FortiGateValidationReport(
        passed=not blocking,
        blocking_issues=blocking,
        warnings=warnings,
        checks=checks,
    )


def check_cli_completeness(state: dict[str, Any]) -> FortiGateValidationReport:
    intake = state.get("intake") or {}
    intent = state.get("implementation_intent") or {}
    artifacts = state.get("config_artifacts") or {}
    cli = str(artifacts.get("cli_config") or "")
    cli_lower = cli.lower()
    blocking: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []

    if not cli.strip():
        blocking.append("No CLI is available for completeness review.")
        return FortiGateValidationReport(passed=False, blocking_issues=blocking, warnings=warnings, checks=checks)

    vlans = _intent_vlans(intent)
    user_vlans = []
    for vlan in vlans:
        role = _as_text(vlan.get("role") or vlan.get("zone") or vlan.get("name")).lower()
        is_user_vlan = any(token in role for token in ("corp", "staff", "user", "guest", "lan"))
        is_mgmt = "mgmt" in role or "management" in role
        if is_user_vlan and not is_mgmt:
            user_vlans.append(vlan)
    if user_vlans:
        if "config system dhcp server" in cli_lower:
            checks.append("DHCP server section is present for user/guest VLAN review.")
        else:
            warnings.append("User/guest VLANs are present in intent, but CLI has no DHCP server section.")

    if len(_as_list(intake.get("wan_circuits"))) > 1:
        if "config system sdwan" not in cli_lower:
            warnings.append("Multiple WAN circuits were provided, but CLI has no SD-WAN section.")
        else:
            checks.append("SD-WAN section is present.")
            for token, label in (("health-check", "health checks"), ("service", "steering services"), ("members", "members")):
                if token not in cli_lower:
                    warnings.append(f"SD-WAN CLI may be missing {label}.")
        if "config router static" not in cli_lower:
            warnings.append("Multiple WAN/SD-WAN design has no static/default route section.")

    policy_matrix = _as_list(intent.get("policy_matrix") or intent.get("firewall_policy_matrix"))
    policies = cli_lower.count("config firewall policy") and cli_lower.count("edit ")
    if policy_matrix and "config firewall policy" not in cli_lower:
        blocking.append("Policy matrix exists, but CLI has no firewall policy section.")
    elif policy_matrix:
        checks.append("Firewall policy section is present for policy matrix review.")

    object_keywords = ("set srcaddr", "set dstaddr", "set service", "set utm-status enable")
    if "config firewall policy" in cli_lower:
        for keyword in object_keywords:
            if keyword not in cli_lower:
                warnings.append(f"Firewall policy CLI may be missing expected directive: {keyword}.")
    if "set logtraffic" not in cli_lower and "set action accept" in cli_lower:
        warnings.append("Allow policies are present but policy logging is not consistently configured.")

    if "guest" in cli_lower and "deny" not in cli_lower and "set action deny" not in cli_lower:
        warnings.append("Guest network appears in CLI, but no explicit deny policy was found.")

    if "set mfa-enabled" in cli_lower and "config vpn ssl settings" not in cli_lower:
        warnings.append("MFA appears referenced without a complete SSL-VPN settings section.")

    _ = policies
    return FortiGateValidationReport(
        passed=not blocking,
        blocking_issues=blocking,
        warnings=warnings,
        checks=checks,
    )
