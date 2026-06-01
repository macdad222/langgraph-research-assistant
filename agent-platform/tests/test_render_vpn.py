"""VPN model + rendering tests (M2)."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.fortigate_render_models import (  # noqa: E402
    AddressObjectModel,
    FortiGateConfigModel,
    InterfaceModel,
    IPsecPhase1Model,
    IPsecPhase2Model,
    Placeholder,
    SslVpnPortalModel,
    SslVpnSettingsModel,
    VpnModel,
)
from app.fortigate_renderer import check_block_balance, render_cli  # noqa: E402

from _m1_common import assert_raises  # noqa: E402


def _vpn_model():
    return FortiGateConfigModel(
        interfaces=[InterfaceModel(name="port1", role="wan", ip="203.0.113.2 255.255.255.252")],
        address_objects=[AddressObjectModel(name="ssl-pool", type="iprange",
                                            start_ip="10.212.134.200", end_ip="10.212.134.210")],
        vpn=VpnModel(
            ipsec_phase1=[IPsecPhase1Model(name="hub-vpn", interface="port1",
                                           remote_gw=Placeholder(token="<HUB_PUBLIC_IP>"),
                                           psksecret=Placeholder(token="<VPN_PSK>"))],
            ipsec_phase2=[IPsecPhase2Model(name="hub-vpn-p2", phase1name="hub-vpn",
                                           src_subnet="10.10.0.0 255.255.0.0",
                                           dst_subnet="10.20.0.0 255.255.0.0")],
            ssl_portals=[SslVpnPortalModel(name="full-access", ip_pools=["ssl-pool"])],
            ssl_settings=SslVpnSettingsModel(listen_port=10443, source_interface=["port1"],
                                             default_portal="full-access",
                                             tunnel_ip_pools=["ssl-pool"]),
        ),
    )


def test_vpn_renders_balanced():
    cli = render_cli(_vpn_model())
    assert check_block_balance(cli) == []


def test_ipsec_phase1_and_phase2_present():
    cli = render_cli(_vpn_model())
    assert "config vpn ipsec phase1-interface" in cli
    assert 'edit "hub-vpn"' in cli
    assert 'set interface "port1"' in cli
    assert "config vpn ipsec phase2-interface" in cli
    assert 'set phase1name "hub-vpn"' in cli


def test_ssl_vpn_present():
    cli = render_cli(_vpn_model())
    assert "config vpn ssl web portal" in cli
    assert "config vpn ssl settings" in cli
    assert 'set default-portal "full-access"' in cli


def test_vpn_secrets_are_placeholders():
    cli = render_cli(_vpn_model())
    assert "<VPN_PSK>" in cli
    assert "<HUB_PUBLIC_IP>" in cli


def test_phase1_interface_usable_in_policy():
    # An IPsec phase1 name should be a valid policy interface reference.
    model = FortiGateConfigModel(
        interfaces=[InterfaceModel(name="port1", role="wan", ip="1.1.1.1 255.255.255.252")],
        address_objects=[AddressObjectModel(name="net", subnet="10.0.0.0 255.255.255.0")],
        vpn=VpnModel(ipsec_phase1=[IPsecPhase1Model(name="hub-vpn", interface="port1",
                                                    remote_gw="198.51.100.1", psksecret="x")]),
        firewall_policies=[{"name": "to-hub", "srcintf": ["port1"], "dstintf": ["hub-vpn"],
                            "srcaddr": ["net"], "dstaddr": ["all"], "service": ["ALL"]}],
    )
    assert model.firewall_policies[0].dstintf == ["hub-vpn"]


def test_phase2_bad_phase1name_rejected():
    assert_raises(
        "is not a defined phase1",
        lambda: FortiGateConfigModel(
            interfaces=[InterfaceModel(name="port1", role="wan")],
            vpn=VpnModel(
                ipsec_phase1=[IPsecPhase1Model(name="hub-vpn", interface="port1",
                                               remote_gw="1.2.3.4", psksecret="x")],
                ipsec_phase2=[IPsecPhase2Model(name="p2", phase1name="ghost-vpn")],
            ),
        ),
    )


def test_ssl_portal_pool_must_be_defined():
    assert_raises(
        "is not a defined address",
        lambda: FortiGateConfigModel(
            vpn=VpnModel(ssl_portals=[SslVpnPortalModel(name="p", ip_pools=["missing-pool"])]),
        ),
    )


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok", t.__name__)
    print(f"PASSED {len(TESTS)} vpn tests")
