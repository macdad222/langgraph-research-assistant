# Deterministic FortiOS CLI Renderer — Implementation Plan

Status: PROPOSED (experiment)
Branch: `cursor/fortigate-cli-renderer` (forked from `cursor/fortigate-provisioning-agent`)
Owner: agent-platform / FortiGate flow

---

## 1. Why

The per-call model bake-off plus an end-to-end run (fixture: jakes 100-site Comcast
SD-WAN template, run `1e37d77f`) exposed two coupled problems:

1. **Syntax correctness is not guaranteed.** The final CLI scored 2/5 from a neutral
   Sonnet judge with ~7 *critical* FortiOS 7.4 syntax errors (zone defined before its
   member interfaces, SD-WAN zone "WAN" colliding with a system zone "WAN",
   `config log tvlan setting` which is not a real stanza, DHCP `set range-start`/
   `range-end`, IPsec `set dst-end`, SD-WAN service `src-address`/`dst-address` keys,
   health-check `server-type icmp`). The internal judge (gemma, thinking-on) caught
   *none* of them because all deterministic checks in `app/fortigate_policy.py` are
   substring/keyword based, not grammar based.

2. **The expensive nodes are the CLI authoring/review/repair nodes.** Of the 314s
   total run: builder_review 45s, regenerate_after_judge 37s, judge ×2 (34s + 29s),
   refine 23s, autonomous_repair 21s, generate + 6 section builders ~55s. Roughly
   **~190s of 314s** is spent authoring, reviewing, and repairing CLI *text*.

The renderer moves CLI syntax from "the LLM hopefully writes valid FortiOS" to
"a deterministic renderer always writes valid FortiOS." The LLM decides *what*
(structured intent); Jinja2 templates decide *how* (exact 7.4 syntax). This targets
**both** goals at once: higher quality (syntax ~100% valid by construction) and lower
latency (the long generative + review + repair work collapses).

---

## 2. Goals / Non-goals

### Goals
- A pure-Python, deterministic, sub-millisecond renderer: validated structured model -> FortiOS CLI text.
- Guaranteed syntactic validity for the stanzas we support (interfaces, zones, DHCP,
  SD-WAN, routing, objects/services, firewall policies, VPN, logging).
- Deterministic referential integrity (every referenced object/interface/zone is defined).
- Drop-in behind a feature flag with safe fallback to the current LLM path.
- Remove or shrink the most expensive LLM nodes (generate CLI, builder_review, refine,
  regenerate_after_judge, syntax-driven autonomous_repair).

### Non-goals (for this experiment)
- Covering 100% of the FortiOS option surface. Scope = what the platform actually emits today.
- Semantic correctness beyond referential integrity (a real FortiGate-VM gate is a
  separate, later track — see Section 9).
- Changing the network-design flow. This is FortiGate-side only.

---

## 3. Architecture

```
intake ─► logical_design ─► fortigate_design ─► implementation_intent (LLM, structured)
                                                      │
                                                      ▼
                                   ┌──────────────────────────────────────┐
                                   │  build_config_model  (LLM ► schema)   │  small, structured-only
                                   │  -> FortiGateConfigModel (Pydantic)   │
                                   └──────────────────────────────────────┘
                                                      │  validated
                                                      ▼
                                   ┌──────────────────────────────────────┐
                                   │  render_config  (DETERMINISTIC)       │  Jinja2, no LLM
                                   │  model -> cli_config + object tables  │
                                   └──────────────────────────────────────┘
                                                      │
                          ┌───────────────────────────┴───────────────────────────┐
                          ▼                                                         ▼
              deterministic validators                                  frontier_model_judge
              (schema + referential                                     (DESIGN quality only,
               integrity + lint rules)                                   not syntax)
```

Key idea: the LLM's last free-text CLI responsibility is removed. `implementation_intent`
already carries the structured fields we need (`interface_inventory`, `dhcp_plan`,
`sdwan_plan`, `fortiswitch_plan`, `wifi_plan`, `policy_matrix`, `object_inventory`,
`logging_plan`). We tighten those loose `dict[str, Any]` shapes into a typed model and
render from it.

---

## 4. The Structured Model (`app/fortigate_render_models.py`)

New Pydantic models — typed versions of the existing intent slices. These map 1:1 to
`SECTION_ORDER` (`interfaces_dhcp, fortiswitch, wifi, sdwan_routing, objects_services,
firewall_policies`). All fields support an explicit placeholder sentinel so unknown
site values flow into `requires_human_input` instead of being invented.

```python
class Placeholder(BaseModel):
    token: str            # e.g. "<DNS_SERVER_1>" or "[SITE_ID]"
    human_prompt: str     # "Corporate DNS server IP"

class InterfaceModel(BaseModel):
    name: str             # vlan10, wan1, lan1
    role: Literal["lan","wan","dmz","mgmt"] = "lan"
    vlan_id: int | None = None
    parent_interface: str | None = None      # required if vlan_id set
    ip: str | Placeholder | None = None       # "10.10.1.1 255.255.255.0"
    allowaccess: list[str] = []               # ping, https, ssh
    description: str = ""

class ZoneModel(BaseModel):
    name: str
    interfaces: list[str]                     # must reference defined InterfaceModel.name

class DhcpServerModel(BaseModel):
    interface: str                            # must reference a defined interface
    gateway: str | Placeholder
    dns: list[str | Placeholder] = []
    range_start: str | Placeholder
    range_end: str | Placeholder
    lease_time: int = 86400

class SdwanMemberModel(BaseModel): ...
class SdwanHealthCheckModel(BaseModel): ...
class SdwanServiceModel(BaseModel): ...
class StaticRouteModel(BaseModel): ...
class AddressObjectModel(BaseModel): ...
class ServiceObjectModel(BaseModel): ...
class FirewallPolicyModel(BaseModel):
    name: str
    srcintf: str; dstintf: str                # must reference a defined zone/interface
    srcaddr: list[str]; dstaddr: list[str]    # must reference defined address objects or 'all'
    service: list[str]
    action: Literal["accept","deny"]
    nat: bool = False
    logtraffic: Literal["all","utm","disable"] = "all"
    inspection: dict | None = None            # av/ips/webfilter/appctrl profile names

class FortiGateConfigModel(BaseModel):
    fortios_version: str                      # drives template dialect (7.4 vs 7.2)
    interfaces: list[InterfaceModel] = []
    zones: list[ZoneModel] = []
    dhcp_servers: list[DhcpServerModel] = []
    sdwan: SdwanModel | None = None
    static_routes: list[StaticRouteModel] = []
    address_objects: list[AddressObjectModel] = []
    service_objects: list[ServiceObjectModel] = []
    firewall_policies: list[FirewallPolicyModel] = []
    vpn: VpnModel | None = None
    logging: LoggingModel | None = None
    requires_human_input: list[str] = []      # auto-collected from Placeholder tokens
```

Pydantic validators enforce ordering/reference rules at construction time:
- A VLAN interface must declare `parent_interface`.
- Every `ZoneModel.interfaces` entry references a defined `InterfaceModel`.
- Every `DhcpServerModel.interface` references a defined interface.
- Every `FirewallPolicyModel` srcintf/dstintf references a defined zone/interface; every
  srcaddr/dstaddr references a defined address object/group or the literal `all`.
- This is the referential-integrity gate, computed deterministically (no LLM).

---

## 5. The Renderer (`app/fortigate_renderer.py` + `app/templates/fortios/`)

- One Jinja2 template per stanza family, organized by FortiOS dialect:
  `templates/fortios/7.4/{interfaces,zones,dhcp,sdwan,routing,objects,services,policies,vpn,logging}.j2`.
- The renderer emits stanzas in the **correct dependency order** (interfaces -> zones ->
  dhcp -> objects/services -> sdwan/routing -> policies -> vpn -> logging), eliminating the
  "zone before interface" class of errors by construction.
- Output preserves the existing artifact shape so downstream code/markdown is unchanged:
  ```python
  def render_config(model: FortiGateConfigModel) -> dict:
      return {
          "cli_config": "<rendered FortiOS CLI>",
          "object_tables": {...},      # same shape _merge_section_artifacts produces
          "policy_table": [...],
          "requires_human_input": [...],
          "safety_label": "DRAFT ONLY - NOT APPLIED TO DEVICE",
          "rendered_by": "deterministic-renderer@7.4",
      }
  ```
- Deterministic, no network, no LLM. Target < 5 ms for a full config.
- Golden-config exemplars (the separate `.conf` corpus from the standards plan) are used
  to author/verify the templates, not at runtime.

### FortiOS version handling
- `fortios_version` selects the template directory. Start with `7.4`; structure allows
  adding `7.2` later. A small capability map flags stanzas that differ across versions.

---

## 6. Graph integration (`app/fortigate.py`, `app/main.py`)

Behind feature flag `FORTIGATE_RENDERER_ENABLED` (default `false` initially).

When enabled, the FortiGate graph changes as follows:

| Current node | Under renderer |
|---|---|
| `build_implementation_intent` | unchanged (still the structured contract) |
| `generate_config_artifacts` / `build_config_section` ×6 | replaced by `build_config_model` (LLM emits typed JSON only) + `render_config` (deterministic) |
| `validate_config_sections` | replaced by Pydantic validation + referential-integrity check |
| `assemble_sectional_config_artifacts` | replaced by `render_config` output |
| `builder_review_config_artifacts` | removed (syntax guaranteed) — optional light design review kept |
| `refine_config_artifacts` | removed for syntax; kept only for design-level edits if needed |
| `frontier_model_judge` | kept, but rubric focuses on DESIGN/coverage, not CLI syntax |
| `regenerate_config_after_judge` | becomes "adjust the structured model + re-render" (cheap) |
| `autonomous_repair_config_artifacts` | only runs for *semantic* gaps (e.g., missing object), not syntax |

New nodes:
- `build_config_model`: one LLM call that fills `FortiGateConfigModel` from
  `implementation_intent` + dependency context. Smaller + faster than long CLI generation.
  Uses structured output / JSON-mode; on parse failure, retry once, then fall back to the
  legacy LLM-CLI path (flag-gated).
- `render_config`: deterministic node calling the renderer.

`main.py`: add `FORTIGATE_RENDERER_ENABLED` (and template dir path) env wiring; extend
`scripts/agent-platform` `allowed_keys` so it can be toggled per model profile.

---

## 7. Placeholders & requires_human_input
- The model uses the `Placeholder` type for any unknown site value (DNS, gateways, PSKs,
  AP/switch serials, public IPs). The renderer emits the token in CLI and auto-collects it
  into `requires_human_input`. This preserves today's behavior (35 placeholders in the
  jakes run) deterministically and guarantees we never invent secrets.

---

## 8. Deterministic validators (replace substring checks)
- **Pydantic validation**: structural correctness at model construction.
- **Referential integrity**: reuse the existing `objects_defined` / `references_required`
  concept (already in section JSON) — now computed from the typed model, so it is exact.
- **Lint rules** (small, curated, FortiOS 7.4): valid stanza names only (catches
  `config log tvlan setting`), no duplicate zone names across system/sdwan, DHCP uses
  `config ip-range`/correct keys, etc. These become impossible-by-construction once
  templates own the syntax, but the linter stays as a guard for the escape hatch.
- Keep `app/fortigate_policy.py` checks as a secondary safety net during rollout.

---

## 9. Optional later track — ground-truth validation
Rendering guarantees syntax; it cannot guarantee the device accepts every semantic combo.
A future, separate track can add a **FortiGate-VM gate**: load the rendered config into a
sandbox VM/VDOM and capture parse errors. Out of scope for this branch, but the renderer
makes it cheap to add because we already have clean, structured input.

---

## 10. Escape hatch
- `FortiGateConfigModel` includes an optional `raw_cli_appendix: list[str]` for stanzas the
  schema does not yet model. Anything emitted via the appendix is **flagged for mandatory
  human review** and excluded from the "syntax guaranteed" claim. This prevents the
  renderer from blocking rare/novel requirements while keeping the guarantee honest.

---

## 11. Rollout / migration strategy
1. Land schema + renderer + templates + unit tests with the flag **off**. Zero runtime change.
2. Add `build_config_model` + `render_config` nodes, flag-gated. Legacy path remains default.
3. Shadow mode: when flag on for a test profile, render AND run legacy, store both, compare.
4. Flip the test model profile (e.g. a `renderer-quality.env`) to renderer-on; benchmark.
5. Promote to default once metrics pass (Section 13). Keep `FORTIGATE_RENDERER_ENABLED=false`
   as an instant rollback.

Deploy via the existing flow: edit on Mac clone -> commit/push ->
`scripts/deploy-to-server.sh --apply renderer-quality`.

---

## 12. Testing strategy
- **Unit**: each template/stanza renders expected CLI from a fixture model (snapshot tests).
- **Referential integrity**: malformed models (dangling zone/object refs) must raise.
- **Round-trip**: render -> parse with the existing `parse_fortigate_config` -> assert
  the parsed summary matches the model (no loss/garble).
- **Golden replays**: re-run captured fixtures (`data/fortigate-runs/*`, incl. jakes
  `2041107c`/`1e37d77f`) through the renderer path; Sonnet-judge the final CLI and compare
  to the legacy 2/5 baseline.
- **Lint**: assert the curated FortiOS 7.4 linter passes on all rendered output and fails
  on known-bad samples (the 7 errors from the jakes run as regression fixtures).

---

## 13. Success metrics (promotion gate)
- Sonnet judge on final CLI: from 2/5 baseline to **>= 4/5** on the jakes fixture and
  >= 4/5 average across 3+ replayed fixtures.
- FortiOS 7.4 lint: **0 critical syntax errors** on rendered output.
- End-to-end latency: from ~314s to a target **<= ~120s** (design + intent + model +
  render + single design-only judge).
- No regression in coverage (all intake requirements still represented) or in the
  `requires_human_input` placeholder behavior.

---

## 14. Milestones / deliverables
- **M1 — Schema + renderer core**: `fortigate_render_models.py`, `fortigate_renderer.py`,
  `templates/fortios/7.4/*.j2`, unit + round-trip tests. Flag off. (No graph change.)
- **M2 — Graph integration**: `build_config_model` + `render_config` nodes, flag wiring in
  `main.py` + `scripts/agent-platform`, legacy fallback. Shadow-mode compare harness.
- **M3 — Validators**: typed referential-integrity + curated 7.4 linter; regression
  fixtures for the 7 jakes errors. Prune syntax-only LLM nodes when flag on.
- **M4 — Benchmark + promote**: replay fixtures, Sonnet-judge, latency measurement vs
  baseline; create `renderer-quality.env`; decide promote/iterate.

---

## 15. Proposed file layout
```
agent-platform/app/fortigate_render_models.py    # typed schema
agent-platform/app/fortigate_renderer.py          # render_config(model) -> artifacts
agent-platform/app/fortigate_lint.py              # curated FortiOS 7.4 block parser + rules
agent-platform/app/templates/fortios/7.4/*.j2     # stanza templates
agent-platform/tests/test_renderer_*.py           # unit / round-trip / regression
```

---

## 16. Risks & mitigations
- **Schema fidelity is the hard part** (not rendering). Mitigation: scope to current
  emitted stanzas; `raw_cli_appendix` escape hatch for the long tail (human-reviewed).
- **LLM JSON-mode parse failures.** Mitigation: retry once, then flag-gated fallback to
  legacy LLM-CLI path; never hard-fail the run.
- **FortiOS version drift.** Mitigation: version-keyed template dirs + capability map.
- **Hidden coupling in downstream markdown/UI.** Mitigation: renderer reproduces the exact
  `config_artifacts` dict shape (`cli_config`, `object_tables`, `policy_table`, etc.).
- **"Guaranteed valid" overclaim via escape hatch.** Mitigation: appendix output is
  excluded from the guarantee and always flagged for human review.

---

## 17. Open questions
- Do we model VPN (IPsec phase1/2, SSL-VPN portal) in M1 or defer to M2? (jakes needs it.)
- Single `build_config_model` call vs. keep the 6-section split feeding the model? (Leaning
  single call for latency; section split only if model output quality suffers.)
- Keep a thin design-only `refine` pass, or let the judge -> adjust-model loop cover it?
