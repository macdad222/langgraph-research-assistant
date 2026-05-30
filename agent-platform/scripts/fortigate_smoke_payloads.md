# FortiGate Autonomous QA Smoke Payloads

Use these payloads against `POST /fortigate/design` to verify intent completeness, CLI completeness, autonomous repair, and remaining human-review behavior.

## Dual WAN Branch With DHCP

Expected behavior:

- Creates implementation intent for corp, guest, and management VLANs.
- Includes DHCP service decisions for corp and guest.
- Treats management DHCP as disabled or explicit design decision.
- Builds SD-WAN members, health checks, steering rules, and default route behavior.
- Produces policy matrix rows for corp internet, guest internet, guest deny to corp, and management access.

```json
{
  "thread_id": "smoke-dual-wan-dhcp",
  "mode": "artifact",
  "existing_config": "",
  "intake": {
    "request_type": "new_build",
    "site_name": "smoke-branch",
    "business_intent": "Build a FortiGate branch firewall with dual WAN, SD-WAN failover, corporate LAN, guest Wi-Fi, and a management VLAN.",
    "fortigate_model": "FortiGate 100F",
    "fortios_version": "7.4.x",
    "wan_circuits": [
      {"description": "wan1 Comcast DHCP primary 500/50"},
      {"description": "wan2 AT&T static backup 100/100"}
    ],
    "lan_networks": [
      {"description": "corp VLAN 10 subnet 10.10.10.0/24 gateway 10.10.10.1 DHCP enabled"},
      {"description": "guest VLAN 30 subnet 10.10.30.0/24 gateway 10.10.30.1 DHCP enabled"},
      {"description": "management VLAN 50 subnet 10.10.50.0/24 gateway 10.10.50.1 static admin devices"}
    ],
    "firewall_policy_intent": "Corp outbound internet with inspection. Guest internet only and explicitly blocked from corp and management. Management access only from trusted admin subnets.",
    "logging_requirements": "Log all allow and deny policies to syslog placeholder.",
    "change_window": "Sunday 02:00 local",
    "rollback_expectations": "Export current config and remove new objects/policies if validation fails."
  }
}
```

## Guest Failover Decision

Expected behavior:

- If guest WAN failover is unspecified, remaining human input should ask whether guests can fail over to primary WAN.
- If the model chooses a default, it should document the assumption.

```json
{
  "thread_id": "smoke-guest-failover-choice",
  "mode": "artifact",
  "existing_config": "",
  "intake": {
    "request_type": "new_build",
    "site_name": "smoke-guest",
    "business_intent": "Build guest Wi-Fi internet access on a FortiGate with two WAN links.",
    "wan_circuits": [
      {"description": "wan1 fiber primary"},
      {"description": "wan2 broadband secondary"}
    ],
    "lan_networks": [
      {"description": "guest VLAN 30 subnet 10.30.30.0/24 gateway 10.30.30.1 DHCP enabled"}
    ],
    "firewall_policy_intent": "Guest internet only. No access to internal networks.",
    "change_window": "Sunday 02:00 local",
    "rollback_expectations": "Remove guest VLAN and policies."
  }
}
```

## Missing Real Values Stay Human

Expected behavior:

- Does not invent syslog, SNMP, DNS, public IP, PSK, or admin source values.
- Adds placeholders and `requires_human_input` entries.

```json
{
  "thread_id": "smoke-human-values",
  "mode": "artifact",
  "existing_config": "",
  "intake": {
    "request_type": "new_build",
    "site_name": "smoke-values",
    "business_intent": "Build a branch FortiGate with VPN, syslog, SNMP monitoring, and secure admin access.",
    "wan_circuits": [
      {"description": "wan1 static public IP unknown"}
    ],
    "lan_networks": [
      {"description": "corp VLAN 10 subnet 10.40.10.0/24 gateway 10.40.10.1 DHCP enabled"}
    ],
    "vpn_requirements": "Site-to-site VPN to headquarters, remote peer and PSK not yet known.",
    "logging_requirements": "Send logs to syslog and support SNMP monitoring; server addresses are not yet known.",
    "firewall_policy_intent": "Corp outbound internet and VPN access to headquarters.",
    "change_window": "Sunday 02:00 local",
    "rollback_expectations": "Restore previous config from backup."
  }
}
```
