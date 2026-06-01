"""Typed schema for the deterministic FortiOS CLI renderer (M1).

The LLM fills this structured model (M2 wires `build_config_model`); the renderer
(`fortigate_renderer.py`) turns it into syntactically valid FortiOS 7.4 CLI. Pydantic
validators enforce referential integrity at construction time so dangling references
(zone members, policy interfaces, address objects, sd-wan zones) fail loudly here rather
than producing broken CLI.

This module imports only pydantic + stdlib so it stays cheap to import and test in
isolation (no langchain/redis/app graph dependencies).
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# --- placeholder handling -------------------------------------------------------------

PLACEHOLDER_TOKEN_RE = re.compile(r"<[^>\n]+>|\[[A-Z0-9_]{2,}\]")


class Placeholder(BaseModel):
    """An explicit unknown site value the LLM must not invent.

    Renders as its `token` (e.g. ``<DNS_SERVER_1>``) and is auto-collected into
    ``requires_human_input``.
    """

    token: str = Field(..., min_length=1)
    human_prompt: str = ""

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.token


# A field value that may be a known string ("10.0.0.1 255.255.255.0") or a Placeholder.
Value = Union[Placeholder, str]


def render_value(value: Optional[Value]) -> str:
    if value is None:
        return ""
    if isinstance(value, Placeholder):
        return value.token
    return str(value)


def collect_placeholders(obj: Any, found: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Walk a model graph collecting explicit Placeholder token -> human_prompt."""
    if found is None:
        found = {}
    if isinstance(obj, Placeholder):
        found.setdefault(obj.token, obj.human_prompt)
    elif isinstance(obj, BaseModel):
        for name in type(obj).model_fields:
            collect_placeholders(getattr(obj, name), found)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            collect_placeholders(item, found)
    elif isinstance(obj, dict):
        for item in obj.values():
            collect_placeholders(item, found)
    return found


def substitute_placeholder_dicts(node: Any, values: dict[str, Any]) -> Any:
    """Recursively replace serialized Placeholder dicts ({"token": ...}) whose
    token is present in ``values`` with the mapped value. Placeholders without a
    provided value are left intact so they remain visible as unresolved inputs.
    Returns a new structure; the input is not mutated."""
    if isinstance(node, dict):
        keys = set(node.keys())
        if "token" in node and keys <= {"token", "human_prompt"}:
            token = node.get("token")
            if token in values and values[token] not in (None, ""):
                return values[token]
            return dict(node)
        return {k: substitute_placeholder_dicts(v, values) for k, v in node.items()}
    if isinstance(node, list):
        return [substitute_placeholder_dicts(item, values) for item in node]
    return node


# --- reference helpers ----------------------------------------------------------------

PHYSICAL_PORT_RE = re.compile(
    r"^(port\d+|wan\d+|internal\d*|mgmt\d*|ha\d*|x\d+|lan\d*|dmz\d*|fortilink"
    r"|aggregate\d*|redundant\d*|npu\d+(_\d+)?|ssl\.root|modem\d*|wwan\d*)$",
    re.IGNORECASE,
)

# Common FortiOS built-in firewall services (subset; custom services are validated by name).
BUILTIN_SERVICES = {
    "ALL", "ALL_TCP", "ALL_UDP", "ALL_ICMP", "ALL_ICMP6",
    "HTTP", "HTTPS", "DNS", "PING", "PING6", "SSH", "TELNET", "SMTP", "SMTPS",
    "FTP", "FTP_GET", "FTP_PUT", "TFTP", "NTP", "SNMP", "SYSLOG", "DHCP", "DHCP6",
    "IKE", "ESP", "GRE", "L2TP", "PPTP", "IMAP", "IMAPS", "POP3", "POP3S",
    "LDAP", "LDAP_UDP", "KERBEROS", "RADIUS", "RDP", "VNC", "SAMBA", "SMB",
    "NETBIOS", "NETBIOS_SESSION", "TRACEROUTE", "WINS", "MS-SQL", "MYSQL",
    "RIP", "BGP", "OSPF", "SIP", "H323", "RTSP", "WEB", "WEB_PROXY",
}

INTERFACE_SPECIAL = {"any"}
ADDRESS_SPECIAL = {"all", "none"}
# FortiOS ships with these SSL-VPN web portals; a config may reference them without
# (re)defining a portal of the same name.
BUILTIN_SSL_PORTALS = {"full-access", "web-access", "tunnel-access", "default"}


def _is_interface_ref(name: str, declared: set[str]) -> bool:
    if name in declared or name in INTERFACE_SPECIAL:
        return True
    return bool(PHYSICAL_PORT_RE.match(name))


# --- leaf models ----------------------------------------------------------------------


class InterfaceModel(BaseModel):
    name: str = Field(..., min_length=1)
    role: Literal["lan", "wan", "dmz", "mgmt", "undefined"] = "lan"
    vlan_id: Optional[Union[int, Placeholder]] = None
    parent_interface: Optional[str] = None
    mode: Literal["static", "dhcp", "pppoe"] = "static"
    ip: Optional[Value] = None  # "10.0.0.1 255.255.255.0"
    allowaccess: list[str] = Field(default_factory=list)
    role_description: str = ""
    description: str = ""


class ZoneModel(BaseModel):
    name: str = Field(..., min_length=1)
    interfaces: list[str] = Field(default_factory=list)
    intrazone: Literal["allow", "deny"] = "deny"


class DhcpServerModel(BaseModel):
    interface: str = Field(..., min_length=1)
    gateway: Value
    netmask: Value = "255.255.255.0"
    dns: list[Value] = Field(default_factory=list)
    range_start: Value
    range_end: Value
    lease_time: int = 86400


class AddressObjectModel(BaseModel):
    name: str = Field(..., min_length=1)
    type: Literal["ipmask", "iprange", "fqdn", "geography"] = "ipmask"
    subnet: Optional[Value] = None  # "10.0.0.0 255.255.255.0"
    start_ip: Optional[Value] = None
    end_ip: Optional[Value] = None
    fqdn: str = ""
    comment: str = ""


class AddressGroupModel(BaseModel):
    name: str = Field(..., min_length=1)
    members: list[Value] = Field(default_factory=list)
    comment: str = ""


class ServiceObjectModel(BaseModel):
    name: str = Field(..., min_length=1)
    protocol: Literal["TCP", "UDP", "SCTP", "ICMP", "IP"] = "TCP"
    tcp_portrange: str = ""
    udp_portrange: str = ""
    comment: str = ""


class ServiceGroupModel(BaseModel):
    name: str = Field(..., min_length=1)
    members: list[Value] = Field(default_factory=list)


class VipModel(BaseModel):
    name: str = Field(..., min_length=1)
    extip: Value
    mappedip: Value
    extintf: str = "any"
    portforward: bool = False
    protocol: Literal["tcp", "udp", "sctp", "icmp"] = "tcp"
    extport: str = ""
    mappedport: str = ""


class SdwanZoneModel(BaseModel):
    name: str = Field(..., min_length=1)


class SdwanMemberModel(BaseModel):
    seq_num: int = Field(..., ge=1)
    interface: str = Field(..., min_length=1)
    gateway: Optional[Value] = None
    zone: str = ""
    source: Optional[Value] = None


class SdwanSlaThresholdModel(BaseModel):
    id: int = Field(default=1, ge=1)
    latency_threshold: Optional[Union[int, Placeholder]] = None
    jitter_threshold: Optional[Union[int, Placeholder]] = None
    packetloss_threshold: Optional[Union[int, Placeholder]] = None


class SdwanHealthCheckModel(BaseModel):
    name: str = Field(..., min_length=1)
    server: list[Value] = Field(default_factory=list)
    protocol: Literal["ping", "tcp-echo", "udp-echo", "http", "twamp", "dns"] = "ping"
    members: list[int] = Field(default_factory=list)
    sla: list[SdwanSlaThresholdModel] = Field(default_factory=list)


class SdwanServiceSlaModel(BaseModel):
    health_check: str = Field(..., min_length=1)
    id: int = Field(default=1, ge=1)


class SdwanServiceModel(BaseModel):
    id: int = Field(..., ge=1)
    name: str = Field(..., min_length=1)
    dst: list[str] = Field(default_factory=list)
    src: list[str] = Field(default_factory=list)
    mode: Literal["auto", "manual", "priority", "sla", "load-balance"] = "sla"
    health_check: str = ""
    sla: list[SdwanServiceSlaModel] = Field(default_factory=list)
    priority_members: list[int] = Field(default_factory=list)


class SdwanModel(BaseModel):
    status: bool = True
    load_balance_mode: str = ""
    zones: list[SdwanZoneModel] = Field(default_factory=list)
    members: list[SdwanMemberModel] = Field(default_factory=list)
    health_checks: list[SdwanHealthCheckModel] = Field(default_factory=list)
    services: list[SdwanServiceModel] = Field(default_factory=list)


class StaticRouteModel(BaseModel):
    seq_num: Optional[int] = Field(default=None, ge=1)
    dst: str = "0.0.0.0 0.0.0.0"
    gateway: Optional[Value] = None
    device: str = ""
    sdwan_zone: str = ""
    distance: int = 10
    comment: str = ""


class FirewallPolicyModel(BaseModel):
    name: str = Field(..., min_length=1)
    policyid: Optional[int] = Field(default=None, ge=0)
    srcintf: list[str] = Field(default_factory=list)
    dstintf: list[str] = Field(default_factory=list)
    srcaddr: list[str] = Field(default_factory=list)
    dstaddr: list[str] = Field(default_factory=list)
    service: list[str] = Field(default_factory=list)
    action: Literal["accept", "deny"] = "accept"
    schedule: str = "always"
    nat: bool = False
    poolname: str = ""
    logtraffic: Literal["all", "utm", "disable"] = "all"
    av_profile: str = ""
    ips_sensor: str = ""
    application_list: str = ""
    webfilter_profile: str = ""
    ssl_ssh_profile: str = ""
    comments: str = ""


class SyslogServerModel(BaseModel):
    server: Value
    port: Optional[Union[int, Placeholder]] = None
    mode: Literal["udp", "legacy-reliable", "reliable", ""] = ""


_FAZ_UPLOAD_OPTIONS = {"realtime", "1-minute", "5-minute"}


class FortiAnalyzerModel(BaseModel):
    # Optional so an LLM that emits an empty/placeholder FAZ block doesn't hard-fail;
    # LoggingModel drops the whole block when no real server is present.
    server: Optional[Value] = None
    upload_option: str = ""

    @field_validator("upload_option", mode="before")
    @classmethod
    def _normalize_upload_option(cls, v: Any) -> str:
        # FortiOS only accepts realtime/1-minute/5-minute here; anything else
        # (e.g. an LLM emitting "disable" to mean "off") is normalized away so we
        # never render an invalid `set upload-option` line.
        if isinstance(v, str) and v in _FAZ_UPLOAD_OPTIONS:
            return v
        return ""


class LoggingModel(BaseModel):
    syslog_servers: list[SyslogServerModel] = Field(default_factory=list)
    fortianalyzer: Optional[FortiAnalyzerModel] = None

    @model_validator(mode="after")
    def _drop_empty_fortianalyzer(self) -> "LoggingModel":
        # An empty/placeholder-less FAZ block (no real server) would render
        # `set server <missing>`; treat it as "not configured" instead.
        faz = self.fortianalyzer
        if faz is not None and (faz.server is None or render_value(faz.server).strip() == ""):
            self.fortianalyzer = None
        return self


# --- VPN models (M2) ------------------------------------------------------------------


class IPsecPhase1Model(BaseModel):
    name: str = Field(..., min_length=1)
    interface: str = Field(..., min_length=1)  # underlying WAN interface
    remote_gw: Value
    ike_version: Literal["1", "2"] = "2"
    proposal: str = "aes256-sha256"
    psksecret: Value  # almost always a placeholder
    peertype: Literal["any", "one", "dialup"] = "any"
    net_device: Literal["enable", "disable", ""] = ""
    comments: str = ""


class IPsecPhase2Model(BaseModel):
    name: str = Field(..., min_length=1)
    phase1name: str = Field(..., min_length=1)  # must reference a defined phase1
    proposal: str = "aes256-sha256"
    src_subnet: Value = "0.0.0.0 0.0.0.0"
    dst_subnet: Value = "0.0.0.0 0.0.0.0"
    pfs: Literal["enable", "disable", ""] = ""


class SslVpnPortalModel(BaseModel):
    name: str = Field(..., min_length=1)
    tunnel_mode: bool = True
    split_tunneling: bool = True
    ip_pools: list[str] = Field(default_factory=list)  # address object names


class SslVpnSettingsModel(BaseModel):
    listen_port: int = 10443
    source_interface: list[str] = Field(default_factory=list)
    source_address: list[str] = Field(default_factory=list)
    default_portal: str = ""
    tunnel_ip_pools: list[str] = Field(default_factory=list)


class VpnModel(BaseModel):
    ipsec_phase1: list[IPsecPhase1Model] = Field(default_factory=list)
    ipsec_phase2: list[IPsecPhase2Model] = Field(default_factory=list)
    ssl_portals: list[SslVpnPortalModel] = Field(default_factory=list)
    ssl_settings: Optional[SslVpnSettingsModel] = None


# --- top-level model ------------------------------------------------------------------


class FortiGateConfigModel(BaseModel):
    fortios_version: str = "7.4"
    hostname: Value = ""
    interfaces: list[InterfaceModel] = Field(default_factory=list)
    zones: list[ZoneModel] = Field(default_factory=list)
    dhcp_servers: list[DhcpServerModel] = Field(default_factory=list)
    address_objects: list[AddressObjectModel] = Field(default_factory=list)
    address_groups: list[AddressGroupModel] = Field(default_factory=list)
    service_objects: list[ServiceObjectModel] = Field(default_factory=list)
    service_groups: list[ServiceGroupModel] = Field(default_factory=list)
    vips: list[VipModel] = Field(default_factory=list)
    sdwan: Optional[SdwanModel] = None
    static_routes: list[StaticRouteModel] = Field(default_factory=list)
    vpn: Optional[VpnModel] = None
    firewall_policies: list[FirewallPolicyModel] = Field(default_factory=list)
    logging: Optional[LoggingModel] = None
    # Escape hatch: stanzas not yet modelled. Anything here is flagged
    # for mandatory human review and excluded from the "syntax guaranteed" claim.
    raw_cli_appendix: list[str] = Field(default_factory=list)
    # Non-token human follow-ups that are not placeholders (e.g. "confirm switch serials").
    extra_human_input: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_referential_integrity(self) -> "FortiGateConfigModel":
        errors: list[str] = []

        declared_ifaces = {i.name for i in self.interfaces}
        # IPsec phase1-interface entries create virtual interfaces policies/routes can use.
        if self.vpn:
            declared_ifaces |= {p.name for p in self.vpn.ipsec_phase1}
        zone_names = {z.name for z in self.zones}
        addr_names = (
            {a.name for a in self.address_objects}
            | {g.name for g in self.address_groups}
            | {v.name for v in self.vips}
        )
        service_names = {s.name for s in self.service_objects} | {g.name for g in self.service_groups}
        sdwan_zone_names = {z.name for z in self.sdwan.zones} if self.sdwan else set()
        health_check_names = {h.name for h in self.sdwan.health_checks} if self.sdwan else set()
        member_seqs = {m.seq_num for m in self.sdwan.members} if self.sdwan else set()

        # Tolerate a common LLM field confusion: an SD-WAN-routed route belongs in
        # `sdwan_zone` (renders `set sdwan-zone`), not `device`. If `device` names a
        # defined sd-wan zone and sdwan_zone is empty, move it before validating.
        for route in self.static_routes:
            if route.device and not route.sdwan_zone and route.device in sdwan_zone_names:
                route.sdwan_zone = route.device
                route.device = ""

        # Auto-resolve a FortiOS namespace collision: a firewall zone that shares a name
        # with an sd-wan zone cannot coexist. Drop the redundant firewall zone; policy
        # interface references still resolve because intf_refs includes sd-wan zone names.
        if sdwan_zone_names:
            kept_zones = [z for z in self.zones if z.name not in sdwan_zone_names]
            if len(kept_zones) != len(self.zones):
                self.zones = kept_zones
                zone_names = {z.name for z in self.zones}

        # Tolerate the LLM using the address special 'all' for a policy interface; FortiOS
        # uses 'any' to mean "every interface" in srcintf/dstintf.
        for pol in self.firewall_policies:
            pol.srcintf = ["any" if x == "all" else x for x in pol.srcintf]
            pol.dstintf = ["any" if x == "all" else x for x in pol.dstintf]

        # interfaces: vlan needs a parent; parent must resolve.
        for iface in self.interfaces:
            if iface.vlan_id is not None and not iface.parent_interface:
                errors.append(f"interface '{iface.name}': vlan_id set but no parent_interface")
            if iface.parent_interface and not _is_interface_ref(iface.parent_interface, declared_ifaces):
                errors.append(
                    f"interface '{iface.name}': parent_interface '{iface.parent_interface}' is not a "
                    "declared or physical interface"
                )

        # zones: members must resolve; a zone cannot share a member with another zone.
        seen_members: dict[str, str] = {}
        for zone in self.zones:
            for member in zone.interfaces:
                if not _is_interface_ref(member, declared_ifaces):
                    errors.append(f"zone '{zone.name}': member interface '{member}' is not defined")
                if member in seen_members and seen_members[member] != zone.name:
                    errors.append(
                        f"interface '{member}' assigned to both zone '{seen_members[member]}' and '{zone.name}'"
                    )
                seen_members[member] = zone.name

        # dhcp: bound interface must be one we configure (declared or physical).
        for dhcp in self.dhcp_servers:
            if not _is_interface_ref(dhcp.interface, declared_ifaces):
                errors.append(f"dhcp server: interface '{dhcp.interface}' is not defined")

        # address groups: members must resolve to address objects/groups.
        for grp in self.address_groups:
            for member in grp.members:
                if isinstance(member, Placeholder):
                    continue
                if member not in addr_names and member not in ADDRESS_SPECIAL:
                    errors.append(f"address group '{grp.name}': member '{member}' is not a defined address")

        # service groups: members must resolve.
        for grp in self.service_groups:
            for member in grp.members:
                if isinstance(member, Placeholder):
                    continue
                if member not in service_names and member.upper() not in BUILTIN_SERVICES:
                    errors.append(f"service group '{grp.name}': member '{member}' is not a defined service")

        # sd-wan: zone names must not collide with system zones; members/services must resolve.
        if self.sdwan:
            for z in self.sdwan.zones:
                if z.name in zone_names:
                    errors.append(
                        f"sd-wan zone '{z.name}' collides with a system zone of the same name; rename it"
                    )
            for m in self.sdwan.members:
                if not _is_interface_ref(m.interface, declared_ifaces):
                    errors.append(f"sd-wan member {m.seq_num}: interface '{m.interface}' is not defined")
                if m.zone and m.zone not in sdwan_zone_names:
                    errors.append(f"sd-wan member {m.seq_num}: zone '{m.zone}' is not a defined sd-wan zone")
            for h in self.sdwan.health_checks:
                for seq in h.members:
                    if seq not in member_seqs:
                        errors.append(f"health-check '{h.name}': member {seq} is not a defined sd-wan member")
            for svc in self.sdwan.services:
                for ref in svc.dst:
                    if ref not in addr_names and ref not in ADDRESS_SPECIAL:
                        errors.append(f"sd-wan service '{svc.name}': dst '{ref}' is not a defined address")
                for ref in svc.src:
                    if ref not in addr_names and ref not in ADDRESS_SPECIAL:
                        errors.append(f"sd-wan service '{svc.name}': src '{ref}' is not a defined address")
                if svc.health_check and svc.health_check not in health_check_names:
                    errors.append(
                        f"sd-wan service '{svc.name}': health_check '{svc.health_check}' is not defined"
                    )
                for seq in svc.priority_members:
                    if seq not in member_seqs:
                        errors.append(
                            f"sd-wan service '{svc.name}': priority member {seq} is not a defined sd-wan member"
                        )

        # static routes: device must resolve; sdwan_zone must resolve.
        for route in self.static_routes:
            if route.sdwan_zone and route.sdwan_zone not in sdwan_zone_names:
                errors.append(f"static route to '{route.dst}': sdwan_zone '{route.sdwan_zone}' is not defined")
            if route.device and not _is_interface_ref(route.device, declared_ifaces):
                errors.append(f"static route to '{route.dst}': device '{route.device}' is not defined")

        # firewall policies: interfaces, addresses, services must resolve.
        intf_refs = declared_ifaces | zone_names | sdwan_zone_names
        for pol in self.firewall_policies:
            for ref in pol.srcintf + pol.dstintf:
                if ref not in intf_refs and ref not in INTERFACE_SPECIAL and not PHYSICAL_PORT_RE.match(ref):
                    errors.append(f"policy '{pol.name}': interface '{ref}' is not a defined zone/interface")
            for ref in pol.srcaddr + pol.dstaddr:
                if ref not in addr_names and ref not in ADDRESS_SPECIAL:
                    errors.append(f"policy '{pol.name}': address '{ref}' is not a defined object")
            for ref in pol.service:
                if ref not in service_names and ref.upper() not in BUILTIN_SERVICES:
                    errors.append(f"policy '{pol.name}': service '{ref}' is not a defined/built-in service")
            if pol.poolname and pol.poolname not in addr_names:
                # ippool names are not address objects, so only warn-by-not-erroring here.
                pass

        # vpn: phase2 must reference a phase1; ssl refs must resolve.
        if self.vpn:
            phase1_names = {p.name for p in self.vpn.ipsec_phase1}
            portal_names = {p.name for p in self.vpn.ssl_portals}
            for p1 in self.vpn.ipsec_phase1:
                if not _is_interface_ref(p1.interface, declared_ifaces):
                    errors.append(f"ipsec phase1 '{p1.name}': interface '{p1.interface}' is not defined")
            for p2 in self.vpn.ipsec_phase2:
                if p2.phase1name not in phase1_names:
                    errors.append(f"ipsec phase2 '{p2.name}': phase1name '{p2.phase1name}' is not a defined phase1")
            for portal in self.vpn.ssl_portals:
                for pool in portal.ip_pools:
                    if pool not in addr_names:
                        errors.append(f"ssl portal '{portal.name}': ip_pool '{pool}' is not a defined address")
            ssl = self.vpn.ssl_settings
            if ssl:
                for ref in ssl.source_interface:
                    if not _is_interface_ref(ref, declared_ifaces):
                        errors.append(f"ssl settings: source-interface '{ref}' is not defined")
                for ref in ssl.source_address:
                    if ref not in addr_names and ref not in ADDRESS_SPECIAL:
                        errors.append(f"ssl settings: source-address '{ref}' is not a defined address")
                for pool in ssl.tunnel_ip_pools:
                    if pool not in addr_names:
                        errors.append(f"ssl settings: tunnel-ip-pool '{pool}' is not a defined address")
                if (
                    ssl.default_portal
                    and ssl.default_portal not in portal_names
                    and ssl.default_portal not in BUILTIN_SSL_PORTALS
                ):
                    errors.append(f"ssl settings: default-portal '{ssl.default_portal}' is not a defined portal")

        if errors:
            raise ValueError("FortiGate config referential integrity failed:\n- " + "\n- ".join(errors))
        return self

    def human_input_items(self, rendered_cli: str = "") -> list[str]:
        """Aggregate requires_human_input: explicit placeholders + tokens found in CLI."""
        prompts = collect_placeholders(self)
        tokens: dict[str, str] = dict(prompts)
        for match in PLACEHOLDER_TOKEN_RE.findall(rendered_cli or ""):
            tokens.setdefault(match, prompts.get(match, ""))
        items = []
        for token in sorted(tokens):
            prompt = tokens[token]
            items.append(f"Provide value for {token}" + (f": {prompt}" if prompt else ""))
        items.extend(self.extra_human_input)
        if self.raw_cli_appendix:
            items.append("Review raw_cli_appendix block(s) manually (not covered by the renderer guarantee).")
        return items
