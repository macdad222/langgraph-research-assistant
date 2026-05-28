from __future__ import annotations

from typing import Any

from app.fortigate_models import FortiGateConfigSummary, FortiGateValidationReport


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
