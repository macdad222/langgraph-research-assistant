"""Deterministic FortiOS CLI renderer (M1).

Turns a validated :class:`FortiGateConfigModel` into syntactically valid FortiOS 7.4 CLI
using Jinja2 templates. Pure + deterministic: no network, no LLM. Stanzas are emitted in
dependency order (system global -> interfaces -> zones -> dhcp -> objects/services ->
sd-wan -> routing -> policies -> logging) so ordering classes of errors (e.g. a zone
referencing an interface defined later) are impossible by construction.

Imports only jinja2 + the render models, so it is cheap to import and test in isolation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .fortigate_render_models import FortiGateConfigModel, render_value

SAFETY_LABEL = "DRAFT ONLY - NOT APPLIED TO DEVICE"
TEMPLATE_ROOT = Path(__file__).parent / "templates" / "fortios"

# Section order: (display header, template file, selector). Empty selections are skipped.
SECTION_PLAN: list[tuple[str, str, Callable[[FortiGateConfigModel], Any]]] = [
    ("system", "system.j2", lambda m: m.hostname),
    ("interfaces", "interfaces.j2", lambda m: m.interfaces),
    ("zones", "zones.j2", lambda m: m.zones),
    ("dhcp", "dhcp.j2", lambda m: m.dhcp_servers),
    ("address-objects", "address.j2", lambda m: m.address_objects),
    ("address-groups", "addrgrp.j2", lambda m: m.address_groups),
    ("services", "service.j2", lambda m: m.service_objects),
    ("service-groups", "servicegrp.j2", lambda m: m.service_groups),
    ("vips", "vip.j2", lambda m: m.vips),
    ("sdwan", "sdwan.j2", lambda m: m.sdwan),
    ("routing", "route.j2", lambda m: m.static_routes),
    ("firewall-policies", "policy.j2", lambda m: m.firewall_policies),
    ("logging", "logging.j2", lambda m: m.logging),
]

_ENV_CACHE: dict[str, Environment] = {}


def _names(items: list[Any]) -> str:
    return " ".join('"%s"' % render_value(i) for i in items)


def _qvals(items: list[Any]) -> str:
    return " ".join('"%s"' % render_value(i) for i in items)


def _get_env(version: str) -> Environment:
    if version not in _ENV_CACHE:
        tdir = TEMPLATE_ROOT / version
        if not tdir.is_dir():
            raise ValueError(f"No FortiOS templates for version '{version}' at {tdir}")
        env = Environment(
            loader=FileSystemLoader(str(tdir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
            autoescape=False,
            undefined=StrictUndefined,
        )
        env.filters["val"] = render_value
        env.filters["names"] = _names
        env.filters["qvals"] = _qvals
        _ENV_CACHE[version] = env
    return _ENV_CACHE[version]


def _context(model: FortiGateConfigModel) -> dict[str, Any]:
    return {
        "hostname": model.hostname,
        "interfaces": model.interfaces,
        "zones": model.zones,
        "dhcp_servers": model.dhcp_servers,
        "address_objects": model.address_objects,
        "address_groups": model.address_groups,
        "service_objects": model.service_objects,
        "service_groups": model.service_groups,
        "vips": model.vips,
        "sdwan": model.sdwan,
        "static_routes": model.static_routes,
        "firewall_policies": model.firewall_policies,
        "logging": model.logging,
    }


def _object_tables(model: FortiGateConfigModel) -> dict[str, list[str]]:
    sdwan_zones = [z.name for z in model.sdwan.zones] if model.sdwan else []
    return {
        "interfaces_dhcp": [i.name for i in model.interfaces]
        + [f"dhcp:{d.interface}" for d in model.dhcp_servers],
        "fortiswitch": [],
        "wifi": [],
        "sdwan_routing": sdwan_zones + [f"route:{r.dst}" for r in model.static_routes],
        "objects_services": [a.name for a in model.address_objects]
        + [g.name for g in model.address_groups]
        + [s.name for s in model.service_objects]
        + [g.name for g in model.service_groups]
        + [v.name for v in model.vips],
        "firewall_policies": [p.name for p in model.firewall_policies],
    }


def render_cli(model: FortiGateConfigModel) -> str:
    """Render only the CLI text (no artifact wrapper)."""
    env = _get_env(model.fortios_version)
    ctx = _context(model)
    blocks: list[str] = []
    for header, template_name, selector in SECTION_PLAN:
        selection = selector(model)
        if not selection:
            continue
        body = env.get_template(template_name).render(**ctx).strip()
        if body:
            blocks.append(f"# --- {header} ---\n{body}")
    if model.raw_cli_appendix:
        appendix = "\n".join(b.strip() for b in model.raw_cli_appendix if b.strip())
        if appendix:
            blocks.append(
                "# --- raw-cli-appendix (UNVERIFIED - human review required) ---\n" + appendix
            )
    return "\n\n".join(blocks).strip()


_CONFIG_RE = re.compile(r"^\s*config\b")
_END_RE = re.compile(r"^\s*end\s*$")
_EDIT_RE = re.compile(r"^\s*edit\b")
_NEXT_RE = re.compile(r"^\s*next\s*$")


def check_block_balance(cli: str) -> list[str]:
    """Structural check: config/end and edit/next must nest like balanced brackets.

    FortiOS legitimately nests `config ... end` inside an open `edit ... next` block, so a
    stack is required. Returns a list of issue strings (empty == balanced). Precursor to
    the M3 linter.
    """
    stack: list[tuple[str, int]] = []  # (kind, lineno)
    issues: list[str] = []
    for lineno, line in enumerate(cli.splitlines(), start=1):
        if _CONFIG_RE.match(line):
            stack.append(("config", lineno))
        elif _EDIT_RE.match(line):
            stack.append(("edit", lineno))
        elif _NEXT_RE.match(line):
            if not stack or stack[-1][0] != "edit":
                issues.append(f"line {lineno}: 'next' without an open 'edit'")
            else:
                stack.pop()
        elif _END_RE.match(line):
            if not stack or stack[-1][0] != "config":
                issues.append(f"line {lineno}: 'end' without an open 'config'")
            else:
                stack.pop()
    for kind, lineno in stack:
        closer = "end" if kind == "config" else "next"
        issues.append(f"line {lineno}: '{kind}' never closed with '{closer}'")
    return issues


def render_config(model: FortiGateConfigModel) -> dict[str, Any]:
    """Render the model and return artifacts shaped like the legacy config_artifacts dict."""
    cli = render_cli(model)
    return {
        "cli_config": cli,
        "object_tables": _object_tables(model),
        "policy_table": [p.name for p in model.firewall_policies],
        "rollback_plan": [
            "Back up the current FortiGate configuration before applying any block.",
            "Apply changes during an approved maintenance window.",
            "Revert by restoring the prior configuration if validation fails.",
        ],
        "assumptions": [],
        "standards_citations": [],
        "implementation_notes": [],
        "sectional_generation": False,
        "rendered_by": f"deterministic-renderer@{model.fortios_version}",
        "safety_label": SAFETY_LABEL,
        "requires_human_input": model.human_input_items(cli),
        "structure_issues": check_block_balance(cli),
    }
