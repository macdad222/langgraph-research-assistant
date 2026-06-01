"""Referential-integrity + validation tests for fortigate_render_models (M1)."""

from _m1_common import (
    AddressGroupModel,
    AddressObjectModel,
    DhcpServerModel,
    FirewallPolicyModel,
    FortiGateConfigModel,
    InterfaceModel,
    Placeholder,
    SdwanModel,
    SdwanMemberModel,
    SdwanServiceModel,
    SdwanZoneModel,
    ZoneModel,
    assert_raises,
    representative_model,
)


def test_valid_model_builds():
    model = representative_model()
    assert model.hostname == "site-fgt-01"
    assert len(model.firewall_policies) == 2


def test_vlan_requires_parent():
    assert_raises(
        "no parent_interface",
        lambda: FortiGateConfigModel(
            interfaces=[InterfaceModel(name="vlan10", vlan_id=10)],
        ),
    )


def test_zone_member_must_be_defined():
    assert_raises(
        "is not defined",
        lambda: FortiGateConfigModel(
            interfaces=[InterfaceModel(name="vlan10", vlan_id=10, parent_interface="port1")],
            zones=[ZoneModel(name="Z", interfaces=["vlan99"])],
        ),
    )


def test_zone_member_physical_port_ok():
    # Physical ports are valid zone members even when not declared as InterfaceModel.
    model = FortiGateConfigModel(zones=[ZoneModel(name="Z", interfaces=["port3"])])
    assert model.zones[0].interfaces == ["port3"]


def test_dhcp_interface_must_be_defined():
    assert_raises(
        "is not defined",
        lambda: FortiGateConfigModel(
            dhcp_servers=[DhcpServerModel(interface="vlanX", gateway="10.0.0.1",
                                          range_start="10.0.0.10", range_end="10.0.0.20")],
        ),
    )


def test_address_group_member_must_be_defined():
    assert_raises(
        "is not a defined address",
        lambda: FortiGateConfigModel(
            address_groups=[AddressGroupModel(name="g", members=["ghost-net"])],
        ),
    )


def test_policy_address_must_be_defined():
    assert_raises(
        "is not a defined object",
        lambda: FortiGateConfigModel(
            interfaces=[InterfaceModel(name="vlan10", vlan_id=10, parent_interface="port1")],
            firewall_policies=[FirewallPolicyModel(name="p", srcintf=["vlan10"],
                                                   dstintf=["port1"], srcaddr=["missing"],
                                                   dstaddr=["all"], service=["ALL"])],
        ),
    )


def test_policy_builtin_service_ok():
    model = FortiGateConfigModel(
        address_objects=[AddressObjectModel(name="net", subnet="10.0.0.0 255.255.255.0")],
        firewall_policies=[FirewallPolicyModel(name="p", srcintf=["port1"], dstintf=["port2"],
                                               srcaddr=["net"], dstaddr=["all"],
                                               service=["HTTPS", "DNS"])],
    )
    assert model.firewall_policies[0].service == ["HTTPS", "DNS"]


def test_policy_unknown_service_rejected():
    assert_raises(
        "is not a defined/built-in service",
        lambda: FortiGateConfigModel(
            address_objects=[AddressObjectModel(name="net", subnet="10.0.0.0 255.255.255.0")],
            firewall_policies=[FirewallPolicyModel(name="p", srcintf=["port1"],
                                                   dstintf=["port2"], srcaddr=["net"],
                                                   dstaddr=["all"], service=["MADE_UP_SVC"])],
        ),
    )


def test_sdwan_zone_collision_detected():
    assert_raises(
        "collides with a system zone",
        lambda: FortiGateConfigModel(
            interfaces=[InterfaceModel(name="vlan10", vlan_id=10, parent_interface="port5")],
            zones=[ZoneModel(name="WAN", interfaces=["vlan10"])],
            sdwan=SdwanModel(zones=[SdwanZoneModel(name="WAN")],
                             members=[SdwanMemberModel(seq_num=1, interface="port1")]),
        ),
    )


def test_sdwan_service_addr_must_be_defined():
    assert_raises(
        "is not a defined address",
        lambda: FortiGateConfigModel(
            sdwan=SdwanModel(
                zones=[SdwanZoneModel(name="OVL")],
                members=[SdwanMemberModel(seq_num=1, interface="port1", zone="OVL")],
                services=[SdwanServiceModel(id=1, name="s", dst=["ghost"], mode="manual")],
            ),
        ),
    )


def test_placeholder_collected_in_human_input():
    model = representative_model()
    cli = ""
    items = model.human_input_items(cli)
    joined = "\n".join(items)
    assert "<DNS_PRIMARY>" in joined
    assert "<FAZ_IP>" in joined


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok", t.__name__)
    print(f"PASSED {len(TESTS)} model tests")
