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
    ("system-hardening", "hardening.j2", lambda m: m.system_hardening),
    ("admin-access", "admin.j2", lambda m: m.admin_access.admins if m.admin_access else None),
    ("interfaces", "interfaces.j2", lambda m: m.interfaces),
    ("fortiswitch", "fortiswitch.j2", lambda m: m.managed_switches or m.switch_vlans),
    (
        "wifi",
        "wifi.j2",
        lambda m: (m.wifi.vaps or m.wifi.wtp_profiles or m.wifi.managed_aps) if m.wifi else None,
    ),
    ("zones", "zones.j2", lambda m: m.zones),
    ("dhcp", "dhcp.j2", lambda m: m.dhcp_servers),
    ("address-objects", "address.j2", lambda m: m.address_objects),
    ("address-groups", "addrgrp.j2", lambda m: m.address_groups),
    ("services", "service.j2", lambda m: m.service_objects),
    ("service-groups", "servicegrp.j2", lambda m: m.service_groups),
    ("vips", "vip.j2", lambda m: m.vips),
    ("sdwan", "sdwan.j2", lambda m: m.sdwan),
    ("routing", "route.j2", lambda m: m.static_routes),
    ("vpn", "vpn.j2", lambda m: m.vpn),
    (
        "utm-profiles",
        "utm.j2",
        lambda m: (
            m.utm_profiles.antivirus
            or m.utm_profiles.ips_sensors
            or m.utm_profiles.webfilter
            or m.utm_profiles.ssl_ssh
        )
        if m.utm_profiles
        else None,
    ),
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
        "system_hardening": model.system_hardening,
        "admin_access": model.admin_access,
        "interfaces": model.interfaces,
        "managed_switches": model.managed_switches,
        "switch_vlans": model.switch_vlans,
        "wifi": model.wifi,
        "utm_profiles": model.utm_profiles,
        "zones": model.zones,
        "dhcp_servers": model.dhcp_servers,
        "address_objects": model.address_objects,
        "address_groups": model.address_groups,
        "service_objects": model.service_objects,
        "service_groups": model.service_groups,
        "vips": model.vips,
        "sdwan": model.sdwan,
        "static_routes": model.static_routes,
        "vpn": model.vpn,
        "firewall_policies": model.firewall_policies,
        "logging": model.logging,
    }


def _object_tables(model: FortiGateConfigModel) -> dict[str, list[str]]:
    sdwan_zones = [z.name for z in model.sdwan.zones] if model.sdwan else []
    vpn_names = []
    if model.vpn:
        vpn_names = [p.name for p in model.vpn.ipsec_phase1] + [p.name for p in model.vpn.ssl_portals]
    fortiswitch_objs = (
        [render_value(s.switch_id) for s in model.managed_switches]
        + [v.name for v in model.switch_vlans]
    )
    wifi_objs: list[str] = []
    if model.wifi:
        wifi_objs = (
            [v.name for v in model.wifi.vaps]
            + [p.name for p in model.wifi.wtp_profiles]
            + [render_value(a.serial) for a in model.wifi.managed_aps]
        )
    return {
        "vpn": vpn_names,
        "interfaces_dhcp": [i.name for i in model.interfaces]
        + [f"dhcp:{d.interface}" for d in model.dhcp_servers],
        "fortiswitch": fortiswitch_objs,
        "wifi": wifi_objs,
        "sdwan_routing": sdwan_zones + [f"route:{r.dst}" for r in model.static_routes],
        "objects_services": [a.name for a in model.address_objects]
        + [g.name for g in model.address_groups]
        + [s.name for s in model.service_objects]
        + [g.name for g in model.service_groups]
        + [v.name for v in model.vips],
        "firewall_policies": [p.name for p in model.firewall_policies],
    }


# --- pre-deployment secret annotation -------------------------------------------------
# Centralized post-render pass: credential/secret `set` values get an OWN-LINE `#`
# REPLACE marker directly above them, plus a checklist block prepended to the CLI. Only
# true secret-bearing fields that exist in the FortiOS 7.4 schema are flagged (admin
# password, wireless VAP passphrase, IPsec phase1 psksecret). Non-secret site values
# (WAN IPs, subnets, VLAN IDs, trusted hosts) and numeric look-alikes such as the
# `config system password-policy` knobs are deliberately left untouched. All inserted
# lines are `#` comments, so FortiOS config/edit/next/end balance and CLI-paste/restore
# safety are preserved.

REPLACE_NOTE = "# >>> REPLACE WITH ACTUAL CUSTOMER VALUE BEFORE DEPLOYMENT"
CHECKLIST_TITLE = (
    "# ===== PRE-DEPLOYMENT CHECKLIST \u2014 replace the following secret values "
    "with actual customer values before deployment ====="
)
CHECKLIST_END = "# ===== end pre-deployment checklist ====="

# (enclosing-config-stanza regex, secret `set` key, label builder). A line is flagged
# only when the `set` key matches AND its nearest enclosing `config` stanza matches, so
# look-alike keys in unrelated stanzas can never produce a false positive.
_SECRET_FIELDS: list[tuple[re.Pattern, str, Callable[[str], str]]] = [
    (
        re.compile(r"^config\s+system\s+admin$"),
        "password",
        lambda edit: f"system admin {edit!r} password",
    ),
    (
        re.compile(r"^config\s+wireless-controller\s+vap$"),
        "passphrase",
        lambda edit: f"wireless vap {edit!r} passphrase",
    ),
    (
        re.compile(r"^config\s+vpn\s+ipsec\s+phase1-interface$"),
        "psksecret",
        lambda edit: f"vpn ipsec phase1 {edit!r} pre-shared key (psksecret)",
    ),
]

_SET_LINE_RE = re.compile(r"^(\s*)set\s+(\S+)")
_EDIT_NAME_RE = re.compile(r'^\s*edit\s+"?(.*?)"?\s*$')


def _secret_label(stanza: str, set_key: str, edit_name: str | None) -> str | None:
    for stanza_re, key, label in _SECRET_FIELDS:
        if key == set_key and stanza_re.match(stanza):
            return label(edit_name or "?")
    return None


def _strip_secret_annotations(cli: str) -> str:
    """Remove previously-inserted REPLACE markers and checklist block (idempotency)."""
    out: list[str] = []
    in_checklist = False
    for line in cli.splitlines():
        s = line.strip()
        if s == CHECKLIST_TITLE:
            in_checklist = True
            continue
        if in_checklist:
            if s == CHECKLIST_END:
                in_checklist = False
            continue
        if s == REPLACE_NOTE:
            continue
        out.append(line)
    return "\n".join(out).lstrip("\n")


def annotate_secrets(cli: str) -> str:
    """Insert own-line REPLACE markers above secret `set` lines and prepend a checklist.

    Idempotent: existing markers/checklist are stripped first, then re-applied. A
    config-stanza stack ensures secret keys are matched only inside their true stanza.
    """
    cli = _strip_secret_annotations(cli)
    config_stack: list[str] = []
    edit_stack: list[str | None] = []
    out: list[str] = []
    checklist: list[str] = []
    for line in cli.splitlines():
        stripped = line.strip()
        if stripped.startswith("config "):
            config_stack.append(stripped)
            edit_stack.append(None)
            out.append(line)
            continue
        if stripped == "end":
            if config_stack:
                config_stack.pop()
                edit_stack.pop()
            out.append(line)
            continue
        if stripped == "next":
            if edit_stack:
                edit_stack[-1] = None
            out.append(line)
            continue
        m_edit = _EDIT_NAME_RE.match(line) if stripped.startswith("edit") else None
        if m_edit:
            if edit_stack:
                edit_stack[-1] = m_edit.group(1)
            out.append(line)
            continue
        m_set = _SET_LINE_RE.match(line)
        if m_set and config_stack:
            indent, key = m_set.group(1), m_set.group(2)
            label = _secret_label(config_stack[-1], key, edit_stack[-1] if edit_stack else None)
            if label:
                out.append(indent + REPLACE_NOTE)
                checklist.append(label)
        out.append(line)
    annotated = "\n".join(out)
    if checklist:
        block = "\n".join([CHECKLIST_TITLE] + ["# - " + item for item in checklist] + [CHECKLIST_END])
        annotated = block + "\n\n" + annotated
    return annotated



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
    cli = "\n\n".join(blocks).strip()
    return annotate_secrets(cli)


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
