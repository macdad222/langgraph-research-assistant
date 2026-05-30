# Sectional FortiGate Generation Plan

## Goal

Improve first-pass FortiGate draft quality by replacing one broad CLI generation call with smaller, specialized section-building tasks. The current model set remains in use: Gemma 4 31B for generation, review, and repair; Qwen 3.6 27B for refinement and final judging.

## Current Status

This flow is implemented behind `FORTIGATE_SECTIONAL_GENERATION_ENABLED`. When enabled, the graph branches after `analyze_change_impact`, runs the section builders in order, validates the section outputs, deterministically assembles `config_artifacts`, and then rejoins the normal validation, review, repair, Qwen refinement, and judge pipeline.

## Initial Scope

Enable the flow behind `FORTIGATE_SECTIONAL_GENERATION_ENABLED=true`.

The first section builders are:

- `interfaces_dhcp`: physical interfaces, VLANs, zones, and DHCP server blocks.
- `fortiswitch`: FortiLink, switch-controller VLAN mappings, port/profile intent, and required placeholders.
- `wifi`: SSIDs, VLAN mappings, guest isolation, AP profiles, and required placeholders.
- `sdwan_routing`: SD-WAN members, health checks, steering rules, and routes.
- `objects_services`: address objects, groups, service objects, VIPs, and IP pools.
- `firewall_policies`: least-privilege policies, logging, NAT, inspection profiles, and explicit denies.

## Output Contract

Each section builder returns only JSON:

```json
{
  "section_name": "interfaces_dhcp",
  "cli_blocks": [],
  "objects_defined": [],
  "references_required": [],
  "assumptions": [],
  "requires_human_input": [],
  "validation_notes": []
}
```

## Model Settings

- Section builders: Gemma 4 31B, low temperature from the primary model, thinking off for schema stability.
- Builder review: Gemma 4 31B, temperature `1.0`, thinking on.
- Autonomous repair: Gemma 4 31B, temperature `0.4`, thinking on.
- Refiner and judge: Qwen 3.6 27B, temperature `0`, thinking on.

## Validation

Python should merge sections deterministically, preserve section provenance, and check:

- Required sections are present.
- WiFi VLANs/zones are represented in interface, object, and policy sections.
- FortiSwitch VLAN/port references point to defined VLANs or are marked human-required.
- Dual WAN requests include SD-WAN, health checks, steering, and route behavior.
- User/guest VLANs include DHCP decisions and CLI when enabled.
- Firewall policies reference defined zones, objects, and services where possible.
- Missing site values are recorded as `requires_human_input` instead of invented.

## Golden Config Future

When sanitized golden configs are available, ingest them as retrieved standards/examples by component type. They should guide patterns and validation rules, not be copied as whole-site templates.
