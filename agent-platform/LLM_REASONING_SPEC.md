# LLM Reasoning Usage Spec

This spec defines when the agent platform should request model reasoning or thinking behavior for FortiGate, Network Design, and Research workflows.

The goal is to spend reasoning budget only where it improves engineering quality: ambiguous design decisions, multi-step configuration synthesis, policy/security review, and autonomous repair. Routine extraction, formatting, deterministic validation, and simple chat responses should stay fast and cheap.

## Current State

The FortiGate workflow explicitly enables vLLM/LiteLLM template thinking for the highest-value review and repair calls.

Current model controls are limited to:

- Primary model temperature, currently low for normal generation.
- Gemma builder-review pass at temperature `1.0` with thinking enabled.
- Gemma autonomous repair pass at temperature `0.4` with thinking enabled.
- Qwen judge/refiner temperature set to zero with thinking enabled.
- LiteLLM passthrough metadata for Qwen judge/refiner calls.

Direct LiteLLM probes showed that `reasoning_effort`, `thinking_budget`, and `reasoning_budget` are accepted by the API but did not change observed output/token behavior for the current vLLM-backed models. `chat_template_kwargs.enable_thinking` is the confirmed working control. The app does not store or render returned `reasoning_content`.

## Principles

Use reasoning when the model must reconcile multiple constraints before producing an artifact.

Do not use reasoning when a deterministic parser, validator, schema check, or low-cost model call is enough.

Never require visible chain-of-thought in user-facing output. If a provider can return reasoning traces, default to not storing or displaying them. Store structured summaries such as `rationale`, `checks`, `auto_fixed_items`, and `requires_human_input` instead.

Reasoning should be feature-flagged and model/provider aware. Different backends expose thinking differently, and unsupported params should not break the workflow.

## Reasoning Levels

Use four platform-level values:

- `off`: no explicit reasoning requested.
- `low`: small reasoning budget for light classification, schema repair, or intent extraction.
- `medium`: deeper reasoning for multi-step artifact generation and repair.
- `high`: maximum configured reasoning for independent judge/reviewer calls.

The app should map these platform levels to provider-specific parameters in one place, instead of scattering raw provider options through graph nodes.

## FortiGate Workflow Defaults

### No Reasoning

Use `off` for:

- `intake_request`
- `parse_existing_config`
- deterministic checks in `fortigate_policy.py`
- run loading/saving
- UI/API response formatting

Reason: these are classification, parsing, deterministic validation, or persistence operations.

### Low Reasoning

Use `low` for:

- human review answer interpretation
- standards-aware requirement extraction
- simple follow-up config updates when the human answer is explicit

Reason: these calls need context, but should not spend large reasoning budgets.

### Medium Reasoning

Use `medium` for:

- `build_logical_design`
- `build_fortigate_design`
- `build_implementation_intent`
- `generate_config_artifacts`
- sectional config builders when `FORTIGATE_SECTIONAL_GENERATION_ENABLED=true`
- `repair_implementation_intent`
- `autonomous_repair_config_artifacts`
- `revise_after_judge`
- `regenerate_config_after_judge`

Reason: these calls synthesize network design, DHCP behavior, SD-WAN behavior, object inventory, policy matrix logic, and FortiGate CLI from multiple inputs. They are complex enough to benefit from deliberate reasoning, but still bounded by deterministic checks and Qwen review.

The current implementation enables thinking for `autonomous_repair_config_artifacts`, but leaves initial design/config generation and sectional builders thinking off for stability and schema discipline.

### High Reasoning

Use `high` for:

- Qwen `refine_config_artifacts`
- Qwen `frontier_model_judge`
- any future independent security/compliance judge

Reason: these calls are the strongest quality gates before human review. They should deeply examine policy logic, security posture, config completeness, undefined references, standards gaps, and unresolved design decisions.

## Network Design Workflow Defaults

Use `medium` for:

- requirements summarization
- design package generation
- FortiGate handoff generation

Use `high` only if the workflow adds an independent design judge.

Use `off` or `low` for chat replies and standards search summaries.

## Research Workflow Defaults

Use `medium` for:

- research planning
- synthesis
- gap analysis
- policy repair planning

Use `high` for:

- critique
- citation verification
- final revision after critique

Use `off` for deterministic policy checks, source fetching, parsing, and persistence.

## Configuration Shape

Add global defaults:

```env
LLM_REASONING_ENABLED=false
LLM_REASONING_INCLUDE_TRACE=false
LLM_REASONING_DEFAULT_EFFORT=off
```

Add FortiGate-specific overrides:

```env
FORTIGATE_DESIGN_REASONING_EFFORT=medium
FORTIGATE_CONFIG_GENERATION_REASONING_EFFORT=medium
FORTIGATE_BUILDER_REVIEW_REASONING_EFFORT=medium
FORTIGATE_AUTONOMOUS_REPAIR_REASONING_EFFORT=medium
FORTIGATE_CONFIG_REFINER_REASONING_EFFORT=high
FORTIGATE_JUDGE_REASONING_EFFORT=high
FORTIGATE_HUMAN_REVIEW_REASONING_EFFORT=low
```

If `LLM_REASONING_ENABLED=false`, all effort settings are ignored.

For the current vLLM/LiteLLM stack, enabled thinking maps to:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": true
  }
}
```

Do not set `thinking_budget` or `reasoning_budget` until the backend is verified to honor those fields. If a backend does not support explicit reasoning params, calls should continue without them and record `reasoning_requested=false` or `reasoning_supported=false` in trace metadata.

## Provider Adapter

Implement a single helper that builds model kwargs from the platform effort level.

The helper should:

- Accept `effort`, `model_name`, and `provider_hint`.
- Return safe `extra_body` or model kwargs for the configured provider.
- Drop unsupported params when provider support is unknown.
- Keep visible reasoning traces disabled unless explicitly enabled.
- Add trace metadata showing requested effort and whether params were applied.

Do not put provider-specific reasoning syntax directly into graph node prompts.

## Observability

Each LLM node that requests reasoning should add trace metadata:

- `reasoning_enabled`
- `reasoning_effort`
- `reasoning_trace_included`
- `reasoning_provider_params_applied`

Saved FortiGate runs should expose the reasoning effort used for the major model passes, but should not expose chain-of-thought.

Useful visible fields are:

- `implementation_intent`
- `intent_completeness_report`
- `cli_completeness_report`
- `auto_fixed_items`
- `requires_human_input`
- judge/refiner rationale summaries

## Rollout Plan

1. Keep primary design/config generation thinking off until schema quality is measured with thinking enabled.
2. Enable thinking for Qwen judge/refiner and Gemma autonomous repair.
3. Use Gemma builder-review at temperature `1.0` with thinking enabled based on the blinded Qwen comparison.
4. Compare judge findings, pass rate, latency, and cost against non-thinking runs.
5. Keep visible reasoning traces disabled unless there is a specific internal debugging need.

## Success Criteria

Reasoning is worth enabling when it produces:

- fewer syntax and undefined-reference judge findings
- more complete DHCP, SD-WAN, and policy matrix coverage
- fewer human-review questions for auto-fixable issues
- no increase in invented site-specific values
- acceptable latency for the workflow

Reasoning should be disabled or reduced when it produces:

- longer runs without quality improvement
- more hallucinated IPs, PSKs, or monitoring values
- unstable JSON output
- repeated over-engineering of simple branches

