"""Shared helpers + fixtures for the M1 renderer tests (no pytest dependency)."""

import os
import sys

# Make the agent-platform root importable so `import app.*` works.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.fortigate_render_models import (  # noqa: E402
    AddressGroupModel,
    AddressObjectModel,
    DhcpServerModel,
    FirewallPolicyModel,
    FortiAnalyzerModel,
    FortiGateConfigModel,
    InterfaceModel,
    LoggingModel,
    Placeholder,
    SdwanHealthCheckModel,
    SdwanMemberModel,
    SdwanModel,
    SdwanServiceModel,
    SdwanServiceSlaModel,
    SdwanSlaThresholdModel,
    SdwanZoneModel,
    ServiceObjectModel,
    StaticRouteModel,
    SyslogServerModel,
    VipModel,
    ZoneModel,
)


def assert_raises(substr, fn):
    try:
        fn()
    except Exception as exc:  # pydantic ValidationError wraps our ValueError
        assert substr.lower() in str(exc).lower(), (
            f"expected error containing {substr!r}, got: {exc}"
        )
        return
    raise AssertionError(f"expected an error containing {substr!r} but none was raised")


def representative_model():
    """A realistic multi-section config exercising the riskier stanzas (sd-wan, dhcp, zones)."""
    return FortiGateConfigModel(
        fortios_version="7.4",
        hostname="site-fgt-01",
        interfaces=[
            InterfaceModel(name="port1", role="wan", ip="203.0.113.2 255.255.255.252",
                           allowaccess=["ping"]),
            InterfaceModel(name="port2", role="wan", ip="198.51.100.2 255.255.255.252",
                           allowaccess=["ping"]),
            InterfaceModel(name="vlan10", role="lan", vlan_id=10, parent_interface="port5",
                           ip="10.10.10.1 255.255.255.0", allowaccess=["ping", "https", "ssh"],
                           description="Corp LAN"),
            InterfaceModel(name="vlan20", role="lan", vlan_id=20, parent_interface="port5",
                           ip="10.10.20.1 255.255.255.0", allowaccess=["ping"],
                           description="Guest"),
        ],
        zones=[
            ZoneModel(name="LAN_ZONE", interfaces=["vlan10"], intrazone="allow"),
            ZoneModel(name="GUEST_ZONE", interfaces=["vlan20"], intrazone="deny"),
        ],
        dhcp_servers=[
            DhcpServerModel(interface="vlan10", gateway="10.10.10.1",
                            dns=[Placeholder(token="<DNS_PRIMARY>", human_prompt="Corp DNS server")],
                            range_start="10.10.10.100", range_end="10.10.10.200"),
        ],
        address_objects=[
            AddressObjectModel(name="corp-net", subnet="10.10.10.0 255.255.255.0"),
            AddressObjectModel(name="guest-net", subnet="10.10.20.0 255.255.255.0"),
        ],
        address_groups=[
            AddressGroupModel(name="internal-nets", members=["corp-net", "guest-net"]),
        ],
        service_objects=[
            ServiceObjectModel(name="web-svc", protocol="TCP", tcp_portrange="80 443"),
        ],
        vips=[],
        sdwan=SdwanModel(
            status=True,
            load_balance_mode="",
            zones=[SdwanZoneModel(name="SDWAN_OVERLAY")],
            members=[
                SdwanMemberModel(seq_num=1, interface="port1", gateway="203.0.113.1",
                                 zone="SDWAN_OVERLAY"),
                SdwanMemberModel(seq_num=2, interface="port2", gateway="198.51.100.1",
                                 zone="SDWAN_OVERLAY"),
            ],
            health_checks=[
                SdwanHealthCheckModel(name="hc-internet", server=["8.8.8.8", "1.1.1.1"],
                                      protocol="ping", members=[1, 2],
                                      sla=[SdwanSlaThresholdModel(id=1, latency_threshold=250,
                                                                  packetloss_threshold=5)]),
            ],
            services=[
                SdwanServiceModel(id=1, name="corp-traffic", dst=["all"], src=["corp-net"],
                                  mode="sla", health_check="hc-internet",
                                  sla=[SdwanServiceSlaModel(health_check="hc-internet", id=1)],
                                  priority_members=[1, 2]),
            ],
        ),
        static_routes=[
            StaticRouteModel(dst="0.0.0.0 0.0.0.0", sdwan_zone="SDWAN_OVERLAY", distance=1),
        ],
        firewall_policies=[
            FirewallPolicyModel(name="corp-to-internet", srcintf=["LAN_ZONE"],
                                dstintf=["SDWAN_OVERLAY"], srcaddr=["corp-net"], dstaddr=["all"],
                                service=["web-svc", "DNS", "PING"], action="accept", nat=True,
                                logtraffic="all"),
            FirewallPolicyModel(name="guest-isolation", srcintf=["GUEST_ZONE"],
                                dstintf=["LAN_ZONE"], srcaddr=["guest-net"],
                                dstaddr=["corp-net"], service=["ALL"], action="deny",
                                logtraffic="all"),
        ],
        logging=LoggingModel(
            syslog_servers=[SyslogServerModel(server="10.10.10.50", port=514, mode="udp")],
            fortianalyzer=FortiAnalyzerModel(server=Placeholder(token="<FAZ_IP>"),
                                             upload_option="realtime"),
        ),
    )
