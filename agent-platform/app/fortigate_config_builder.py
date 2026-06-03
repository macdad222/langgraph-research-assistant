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
- system_hardening: {admin_https_redirect(bool), admin_ssh_v1(bool, keep false), admintimeout(int, <=5),
  strong_crypto(bool), admin_https_ssl_versions(str "tlsv1-2 tlsv1-3"), pre_login_banner(bool), timezone(str id),
  ntp:{servers:[ip/fqdn], type(fortiguard|custom), sync_interval(int)},
  password_policy:{status(bool), minimum_length(int>=14), min_lower_case_letter(int), min_upper_case_letter(int),
  min_number(int), min_non_alphanumeric(int), expire_days(int|null), reuse_password(enable|disable)}}
  (Source: system_hardening plan in implementation_intent / fortigate_handoff. Use CIS-aligned defaults:
   minimum_length>=14 with complexity, admintimeout<=5, HTTPS/SSH only, disable ssh-v1, strong-crypto on.
   NEVER invent real NTP server addresses - use placeholders for them.)
- admin_access: {admins:[{name, accprofile(default super_admin), trusted_hosts:["ip mask", ...],
  two_factor(disable|fortitoken|email|sms), two_factor_email, password, comments}]}
  (Enable MFA (two_factor) for privileged admins. NEVER invent passwords or real trusted-host subnets -
   emit them as placeholder objects so the operator supplies them.)
- utm_profiles: {antivirus:[{name, comment, http(bool), ftp, smtp, pop3, imap, outbreak_prevention(bool)}],
  ips_sensors:[{name, comment, block_malicious_url(bool), extended_log(bool),
  entries:[{id(int), severity, location, protocol, status, action}]}],
  webfilter:[{name, comment, inspection_mode(flow|proxy), fortiguard_categories:[{category_id(int), action}]}],
  ssl_ssh:[{name, comment, inspect_all(certificate-inspection|deep-inspection|disable)}]}
  (Define a standard set named av-default, ips-default, web-default and reference the BUILT-IN ssl profile
   "certificate-inspection" (or "deep-inspection"). Attach these to OUTBOUND/internet-egress accept policies
   via the policy fields av_profile/ips_sensor/webfilter_profile/ssl_ssh_profile. Source: policy_matrix[].inspection_profile
   plus the security_profiles plan/standards. Built-in ssl profiles do NOT need a definition here.)
- managed_switches: [{switch_id(serial - usually a placeholder), fortilink(default "fortilink"),
  ports:[{port, native_vlan(interface name), allowed_vlans:[interface names], poe_status(enable|disable)}]}]
- switch_vlans: [{name, vlanid(int), interface(default "fortilink"), ip("A.B.C.D M.M.M.M")}]
  (Map fortiswitch_plan: if FortiLink is enabled, emit one managed switch with a PLACEHOLDER serial; put
   unknown VLAN/port-profile/port-map decisions into placeholders or extra_human_input. switch_vlans are
   interface-like and may be referenced by zones/dhcp/policies.)
- wifi: {vaps:[{name, ssid, security(wpa2-only-personal|wpa2-only-enterprise|wpa3-only-personal|open),
  passphrase(placeholder for PSK), auth(psk|radius|usergroup), radius_server(placeholder), vlanid(int),
  local_bridging(bool), mapped_interface(a defined VLAN interface name), guest_isolation(bool)}],
  wtp_profiles:[{name, comment, platform_type(e.g. FAP231F), country, vaps:[VAP names]}],
  managed_aps:[{name, serial(placeholder), wtp_profile, comment}]}
  (Map wifi_plan: each ssid -> a vap; security_mode -> security + auth (enterprise=>auth radius + radius_server
   placeholder; PSK=>auth psk + passphrase placeholder); vlan_mapping -> mapped_interface (the defined VLAN
   interface, e.g. "vlan_corp"); guest_isolation -> guest_isolation. Unknown PSKs/RADIUS/AP serials MUST be
   placeholders. If a vap sets vlanid + local_bridging it MUST also set mapped_interface (the bridge target).)
- raw_cli_appendix: [strings] - ONLY for stanzas not covered above; flagged for human review.
- extra_human_input: [strings] - non-placeholder follow-ups.

CRITICAL RULES (referential integrity is enforced and will reject your output otherwise):
- Every zone member, dhcp interface, policy srcintf/dstintf, policy srcaddr/dstaddr/service,
  sd-wan member interface/service address, static route device/sdwan_zone, address group member,
  ipsec phase2 phase1name, and ssl reference MUST point at something you defined (or a physical
  port like wan1/port1, "all" for addresses, "any" for interfaces, or a FortiOS built-in service).
- A VLAN interface MUST set parent_interface.
- switch_vlans names and wifi vap names are interface-like and count as "defined interfaces" for
  zone/dhcp/policy references. managed_switch.fortilink, switch_vlan.interface, port native_vlan/allowed_vlans,
  and wifi vap.mapped_interface MUST resolve to a defined or physical interface (fortilink is a valid port).
- A managed_ap.wtp_profile MUST reference a defined wtp_profile name.
- SD-WAN zone names MUST NOT equal any system zone name.
- NEVER invent secrets, public IPs, gateways, PSKs, DNS/syslog/RADIUS targets, or serials.
  For any unknown site value, use a placeholder object: {"token": "<DESCRIPTIVE_NAME>",
  "human_prompt": "what to ask the human"} in place of the string value.

PLACEHOLDER ENCODING (STRICT - read carefully):
- A placeholder MUST be emitted as a nested JSON OBJECT, never as a JSON string.
- The object goes directly in the field position where a value would otherwise be.
- CORRECT (nested object):
    "ip": {"token": "<CORP_IP_MASK>", "human_prompt": "Corp VLAN interface IP and mask"}
- WRONG (a JSON-encoded string - DO NOT DO THIS):
    "ip": "{ \\"token\\": \\"<CORP_IP_MASK>\\", \\"human_prompt\\": \\"...\\" }"
  The wrong form double-encodes the object as a string and will be misread as a literal
  value. Always emit the bare object, with no surrounding quotes and no escaping.
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


RECOMMEND_GUIDE = """You are a senior FortiGate network engineer filling in the unknown
site-specific values for a configuration that has already been designed. For EACH requested
value, propose one sensible, production-reasonable default that is consistent with the design
context and with the other values you propose. Return ONLY JSON in this exact shape:
{"recommendations": {"<TOKEN>": {"value": "<cli value>", "confidence": "high|medium|low"}}}

Rules:
- Use RFC1918 ranges for private subnets/IPs; keep per-network subnets non-overlapping.
- VLAN IDs: small distinct integers (e.g. 10, 20, 30) aligned to each network's purpose.
- Netmasks / route subnets in FortiOS form (e.g. "10.10.10.0 255.255.255.0").
- DHCP ranges must sit inside their matching subnet.
- DNS: reputable public resolvers (e.g. 1.1.1.1 / 8.8.8.8) unless the context says otherwise.
- SD-WAN SLA thresholds: typical latency(ms)/jitter(ms)/packet-loss(%) integers.
- Pre-shared keys / secrets: generate a strong random value and set confidence "low".
- Externally-assigned values (public WAN IP, ISP gateway, remote VPN peer): give a clearly
  in-range EXAMPLE and set confidence "low" so the operator knows to confirm it.
- The value MUST be only the literal CLI token(s), no commentary, no quotes around it."""


async def recommend_placeholder_values(
    invoke: InvokeFn,
    context: dict[str, Any],
    placeholders: dict[str, str],
    *,
    run_name: str = "recommend_input_values",
) -> list[dict[str, str]]:
    """Ask the model to recommend a value for every placeholder token.

    Returns a list of {token, prompt, recommended_value, confidence} preserving
    placeholder order. Never raises; missing recommendations fall back to blank."""
    if not placeholders:
        return []
    items = [{"token": t, "prompt": p} for t, p in placeholders.items()]
    human = json.dumps({"design_context": context, "values_needed": items}, indent=2, default=str)
    recs: dict[str, Any] = {}
    try:
        content = await invoke(RECOMMEND_GUIDE, human, run_name)
        data = _extract_json(content)
        candidate = data.get("recommendations")
        if isinstance(candidate, dict):
            recs = candidate
    except Exception:
        recs = {}
    out: list[dict[str, str]] = []
    for token, prompt in placeholders.items():
        entry = recs.get(token)
        if isinstance(entry, dict):
            value = str(entry.get("value", "") or "")
            confidence = str(entry.get("confidence", "medium") or "medium")
        elif entry not in (None, ""):
            value = str(entry)
            confidence = "medium"
        else:
            value = ""
            confidence = "low"
        out.append(
            {
                "token": token,
                "prompt": prompt,
                "recommended_value": value,
                "confidence": confidence,
            }
        )
    return out
