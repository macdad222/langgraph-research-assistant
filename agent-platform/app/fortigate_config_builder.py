"""LLM -> structured config model builder for the deterministic renderer (M2).

Maps the existing `implementation_intent` (plus design/intake context) into a validated
:class:`FortiGateConfigModel`. The renderer then turns that model into FortiOS CLI. This
is the only "intelligence" step in the renderer path; everything after it is deterministic.

Decoupled from the LangGraph closure so it can be reused by the shadow/eval harness: the
caller passes an async `invoke(system_text, human_text, run_name) -> content_text`.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Optional

from .fortigate_render_models import FortiGateConfigModel

InvokeFn = Callable[[str, str, str], Awaitable[str]]

SUPPORTED_VERSIONS = {"7.4"}
DEFAULT_VERSION = "7.4"


def normalize_fortios_version(version: str) -> str:
    """Map an intake fortios_version to a template dialect we ship. Defaults to 7.4."""
    v = (version or "").strip()
    if not v:
        return DEFAULT_VERSION
    # "7.4.5" -> "7.4"; "v7.2" -> "7.2"
    v = v.lstrip("vV")
    parts = v.split(".")
    if len(parts) >= 2:
        major_minor = f"{parts[0]}.{parts[1]}"
        if major_minor in SUPPORTED_VERSIONS:
            return major_minor
    return DEFAULT_VERSION


SCHEMA_GUIDE = """You convert a FortiGate implementation intent into a STRICT JSON object that a
deterministic renderer turns into FortiOS CLI. Return ONLY JSON with a single top-level key
"fortigate_config". Do not emit any CLI text yourself - only structured data.

"fortigate_config" fields (omit a field to leave it empty):
- hostname: string
- interfaces: [{name, role(lan|wan|dmz|mgmt|undefined), vlan_id(int, omit if physical),
  parent_interface(required when vlan_id set), mode(static|dhcp|pppoe), ip("A.B.C.D M.M.M.M"),
  allowaccess(["ping","https","ssh"]), description}]
- zones: [{name, interfaces:[interface names], intrazone(allow|deny)}]
- dhcp_servers: [{interface, gateway, netmask, dns:[...], range_start, range_end, lease_time(int)}]
- address_objects: [{name, type(ipmask|iprange|fqdn|geography), subnet("ip mask"),
  start_ip, end_ip, fqdn, comment}]
- address_groups: [{name, members:[address names], comment}]
- service_objects: [{name, protocol(TCP|UDP|SCTP|ICMP|IP), tcp_portrange, udp_portrange, comment}]
- service_groups: [{name, members:[service names]}]
- vips: [{name, extip, mappedip, extintf, portforward(bool), protocol, extport, mappedport}]
- sdwan: {status(bool), load_balance_mode, zones:[{name}], members:[{seq_num(int), interface,
  gateway, zone, source}], health_checks:[{name, server:[...], protocol(ping|tcp-echo|udp-echo|http),
  members:[seq_num ints], sla:[{id(int), latency_threshold, jitter_threshold, packetloss_threshold}]}],
  services:[{id(int), name, dst:[address names], src:[address names], mode(sla|manual|priority|load-balance),
  health_check, sla:[{health_check, id(int)}], priority_members:[seq_num ints]}]}
- static_routes: [{seq_num(int), dst("ip mask"), gateway, device(interface), sdwan_zone, distance(int), comment}]
  (For a route steered over SD-WAN, set sdwan_zone to the sd-wan zone name and LEAVE device empty.
   Use device ONLY for a plain physical/VLAN interface egress. Do NOT put an sd-wan zone in device.)
- vpn: {ipsec_phase1:[{name, interface, remote_gw, ike_version("1"|"2"), proposal, psksecret,
  peertype(any|one|dialup), comments}], ipsec_phase2:[{name, phase1name, proposal, src_subnet, dst_subnet, pfs}],
  ssl_portals:[{name, tunnel_mode(bool), split_tunneling(bool), ip_pools:[address names]}],
  ssl_settings:{listen_port(int), source_interface:[...], source_address:[address names], default_portal, tunnel_ip_pools:[address names]}}
- firewall_policies: [{name, policyid(int), srcintf:[zone/interface names], dstintf:[...],
  srcaddr:[address names or "all"], dstaddr:[...], service:[service names or built-ins like ALL/HTTPS/DNS/PING],
  action(accept|deny), schedule, nat(bool), poolname, logtraffic(all|utm|disable),
  av_profile, ips_sensor, application_list, webfilter_profile, ssl_ssh_profile, comments}]
- logging: {syslog_servers:[{server, port(int), mode(udp|reliable)}], fortianalyzer:{server, upload_option}}
- raw_cli_appendix: [strings] - ONLY for stanzas not covered above; flagged for human review.
- extra_human_input: [strings] - non-placeholder follow-ups.

CRITICAL RULES (referential integrity is enforced and will reject your output otherwise):
- Every zone member, dhcp interface, policy srcintf/dstintf, policy srcaddr/dstaddr/service,
  sd-wan member interface/service address, static route device/sdwan_zone, address group member,
  ipsec phase2 phase1name, and ssl reference MUST point at something you defined (or a physical
  port like wan1/port1, "all" for addresses, "any" for interfaces, or a FortiOS built-in service).
- A VLAN interface MUST set parent_interface.
- SD-WAN zone names MUST NOT equal any system zone name.
- NEVER invent secrets, public IPs, gateways, PSKs, DNS/syslog/RADIUS targets, or serials.
  For any unknown site value, use a placeholder object: {"token": "<DESCRIPTIVE_NAME>",
  "human_prompt": "what to ask the human"} in place of the string value.
"""


def _extract_json(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        # strip ```json ... ``` fences
        stripped = stripped.split("```", 2)[1] if stripped.count("```") >= 2 else stripped
        if stripped.lstrip().lower().startswith("json"):
            stripped = stripped.lstrip()[4:]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return {}


def _system_prompt(fortios_version: str, judge_context: Optional[dict[str, Any]]) -> str:
    prompt = SCHEMA_GUIDE + f"\nTarget FortiOS version: {fortios_version}.\n"
    if judge_context:
        prompt += (
            "\nThis is a REVISION. An independent reviewer found issues with the previous render. "
            "Return a corrected full fortigate_config that resolves them while preserving the parts "
            "that were correct. Do not regress working sections."
        )
    return prompt


def _human_prompt(context: dict[str, Any], prior_error: Optional[str]) -> str:
    payload = json.dumps(context, indent=2, default=str)
    if prior_error:
        payload += (
            "\n\nYour previous JSON failed strict validation with this error:\n"
            f"{prior_error}\n"
            "Return corrected JSON that fixes every listed reference/field problem."
        )
    return payload


async def build_config_model(
    invoke: InvokeFn,
    context: dict[str, Any],
    *,
    fortios_version: str = DEFAULT_VERSION,
    retries: int = 1,
    judge_context: Optional[dict[str, Any]] = None,
    run_name: str = "build_config_model",
) -> tuple[Optional[FortiGateConfigModel], list[str]]:
    """Return (model, errors). model is None when all attempts fail (caller should fall back)."""
    version = normalize_fortios_version(fortios_version)
    system = _system_prompt(version, judge_context)
    errors: list[str] = []
    prior_error: Optional[str] = None
    for _ in range(retries + 1):
        human = _human_prompt(context, prior_error)
        content = await invoke(system, human, run_name)
        data = _extract_json(content)
        cfg = data.get("fortigate_config") if isinstance(data.get("fortigate_config"), dict) else data
        if not isinstance(cfg, dict) or not cfg:
            prior_error = "Response did not contain a JSON object under 'fortigate_config'."
            errors.append(prior_error)
            continue
        cfg.setdefault("fortios_version", version)
        try:
            model = FortiGateConfigModel(**cfg)
            return model, errors
        except Exception as exc:  # pydantic ValidationError or our referential ValueError
            prior_error = str(exc)
            errors.append(prior_error)
    return None, errors
