"""Rendering + structural tests for fortigate_renderer (M1).

These lock in the specific FortiOS 7.4 syntax fixes that the jakes 2/5 run got wrong:
correct stanza ordering, real DHCP ip-range keys, real sd-wan service/health-check keys,
and no invented `config log tvlan setting`.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.fortigate_renderer import check_block_balance, render_cli, render_config  # noqa: E402

from _m1_common import representative_model  # noqa: E402


def test_renders_and_is_block_balanced():
    cli = render_cli(representative_model())
    assert cli, "expected non-empty CLI"
    assert check_block_balance(cli) == []


def test_interfaces_render_before_zones():
    cli = render_cli(representative_model())
    assert "config system interface" in cli
    assert "config system zone" in cli
    assert cli.index("config system interface") < cli.index("config system zone"), (
        "interfaces must be defined before zones reference them"
    )


def test_vlan_uses_type_vlan_and_parent():
    cli = render_cli(representative_model())
    assert "set type vlan" in cli
    assert 'set interface "port5"' in cli
    assert "set vlanid 10" in cli


def test_dhcp_uses_real_ip_range_keys():
    cli = render_cli(representative_model())
    assert "config ip-range" in cli
    assert "set start-ip 10.10.10.100" in cli
    assert "set end-ip 10.10.10.200" in cli
    # The jakes run invented these non-existent keys:
    assert "set range-start" not in cli
    assert "set range-end" not in cli


def test_sdwan_service_uses_dst_src_not_address_keys():
    cli = render_cli(representative_model())
    assert "config service" in cli
    assert "set dst " in cli
    assert "set src " in cli
    # jakes invented src-address / dst-address under sd-wan service:
    assert "dst-address" not in cli
    assert "src-address" not in cli


def test_sdwan_healthcheck_uses_protocol_not_server_type():
    cli = render_cli(representative_model())
    assert "set protocol ping" in cli
    assert "server-type" not in cli


def test_no_invented_log_tvlan_stanza():
    cli = render_cli(representative_model())
    assert "config log tvlan" not in cli
    # real syslog stanza is present instead:
    assert "config log syslogd setting" in cli


def test_policy_references_zones_and_objects():
    cli = render_cli(representative_model())
    assert 'set srcintf "LAN_ZONE"' in cli
    assert 'set dstintf "SDWAN_OVERLAY"' in cli
    assert 'set srcaddr "corp-net"' in cli
    assert "set action deny" in cli  # guest isolation policy


def test_artifacts_shape_and_human_input():
    artifacts = render_config(representative_model())
    for key in ("cli_config", "object_tables", "policy_table", "requires_human_input",
                "safety_label", "rendered_by", "structure_issues"):
        assert key in artifacts, f"missing artifact key {key}"
    assert artifacts["rendered_by"].startswith("deterministic-renderer@7.4")
    assert artifacts["structure_issues"] == []
    joined = "\n".join(artifacts["requires_human_input"])
    assert "<DNS_PRIMARY>" in joined and "<FAZ_IP>" in joined


def test_raw_appendix_flagged():
    model = representative_model()
    model.raw_cli_appendix = ["config vpn ipsec phase1-interface\n    edit \"vpn1\"\n    next\nend"]
    artifacts = render_config(model)
    assert "raw-cli-appendix" in artifacts["cli_config"]
    assert any("raw_cli_appendix" in item for item in artifacts["requires_human_input"])


def test_balance_detects_unclosed():
    issues = check_block_balance("config system interface\n    edit \"x\"\n    next")
    assert issues, "expected an imbalance issue for missing end"


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


if __name__ == "__main__":
    for t in TESTS:
        t()
        print("ok", t.__name__)
    print(f"PASSED {len(TESTS)} renderer tests")
