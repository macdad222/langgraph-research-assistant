================================================================================
MODEL TEST README — Per-Call Model Bake-off & Flow-Efficiency Refactor
================================================================================
Last updated: 2026-05-31
Scope: agent-platform FortiGate + Network-Design LangGraph flows

--------------------------------------------------------------------------------
1. GOAL
--------------------------------------------------------------------------------
Find the best local model for EACH major LLM call in the graph, balancing
quality and latency, then wire the winners in via per-role model knobs and a
named model profile. Quality was scored by a neutral frontier judge (Sonnet).

Candidate local models (served via LiteLLM -> vLLM):
  - extl-gemma-4-31b   (Gemma 4 31B)
  - qwen3-coder-next   (Qwen3 Coder Next)
  - Qwen3.6-27B        (Qwen 3.6 27B)

Neutral judge: anthropic-sonnet-4.6 (temperature not sent; not score-able by a
candidate under test).

Thinking/reasoning policy used during the sweep:
  - Qwen models: thinking OFF (they over-think; runaway risk on long generation)
  - Gemma: evaluated BOTH thinking OFF and ON

--------------------------------------------------------------------------------
2. METHODOLOGY
--------------------------------------------------------------------------------
"Reconstruct-and-replay" sweep harness, run INSIDE the langgraph-app container so
it imports the app's own prompt-building helpers for fidelity:
  - Golden inputs captured from prior real runs (data/fortigate-runs,
    data/network-design-runs).
  - For each graph call-class, the harness rebuilds the exact prompt the node
    would send (via app.fortigate / app.network_design helpers), runs each
    candidate model config N times, records median latency, JSON-validity %,
    finish reason, output tokens.
  - Sonnet judges each output 1-5 against a FortiOS-7.4 correctness rubric.
  - All candidate calls send metadata {agentic_gateway_openclaw_passthrough:true}
    to bypass the LiteLLM web-search interception callback (matches production).

Harness scripts (kept under /tmp in the container during testing):
  sweep.py      - FortiGate call-classes
  sweep_nd.py   - Network-design call-classes
  fulltrace.py  - End-to-end /fortigate/design run + timing + Sonnet judge
  rejudge.py    - Re-judge a saved run's final cli_config only

IMPORTANT GOTCHA (Qwen3.6 + web search):
  Qwen3.6-27B is a DB-defined LiteLLM alias whose backend is tool-capable. The
  auto_web_search_tool custom callback injects a web-search tool + system nudges
  for certain models (incl. Qwen3.6); the model then EXECUTES the search and the
  results inflate the prompt (e.g. ~29k vs ~6k tokens) and derail the output
  (off-topic "sports playoff" text). Fix: production design/intent/judge/refine/
  repair calls set passthrough=true to opt out. Bake-off harness does the same.

--------------------------------------------------------------------------------
3. PER-CALL WINNERS (applied in config/model-profiles/local-quality.env)
--------------------------------------------------------------------------------
Call (graph node)                 Model              Think  Judge  Notes
--------------------------------- ------------------ -----  -----  --------------
logical/fortigate design          extl-gemma-4-31b   off     4     fast, structured
implementation_intent             Qwen3.6-27B        off     5     dedicated knob*
generate + section builders       extl-gemma-4-31b   off     4     ~14s; Qwen runaway,
                                                                    coder slower (j3.5)
builder_review                    qwen3-coder-next   off     4     only 100%-valid j4
refine                            Qwen3.6-27B        off    4.5    beats coder + faster
judge (frontier_model_judge)      extl-gemma-4-31b   ON      5     gatekeeper; per-role*
autonomous repair                 extl-gemma-4-31b   off    4.5    ~7.5s fastest + best
revise_after_judge (MODEL_NAME)   extl-gemma-4-31b   off     4
nd summarize + package            qwen3-coder-next   off    5/4.5
nd handoff                        extl-gemma-4-31b   off     5     ~5s fastest + best
chat / intake / context-fallback  extl-gemma-4-31b   off     -

* New code knobs added (see section 4).

Headline takeaways:
  - Gemma 4 31B is the backbone: fast, strong on design/generation/handoff/repair.
  - Qwen3.6-27B excels at bounded EDITING (intent, refine) but RUNS AWAY on long
    cold generation -- never use it for the generate node.
  - qwen3-coder-next is the reliable syntax-aware reviewer (builder_review) and
    the best network-design package/summarize model; slower but 100% valid.

--------------------------------------------------------------------------------
4. CODE CHANGES (this iteration)
--------------------------------------------------------------------------------
a) FORTIGATE_INTENT_MODEL_NAME  (NEW)
   implementation_intent now has its own model knob (default = design model).
   Wired in app/main.py (fortigate_intent_model) and app/fortigate.py
   (build_fortigate_graph intent_model / intent_model_name; build_implementation_
   intent + repair_implementation_intent use it). Created with passthrough=true.

b) FORTIGATE_JUDGE_ENABLE_THINKING  (NEW)
   Per-role thinking override for the judge so it can run thinking-ON for max
   rigor even when the rest of the flow runs thinking-OFF. Defaults to the global
   FORTIGATE_ENABLE_THINKING. Wired in app/main.py (judge_model extra_body).

c) scripts/agent-platform allowed_keys extended with both new keys.

Apply / switch profiles:
   scripts/agent-platform model-profile apply local-quality --restart

--------------------------------------------------------------------------------
5. END-TO-END VALIDATION RUN (fixture: jakes / 100-site Comcast SD-WAN template)
--------------------------------------------------------------------------------
Source: network-design run 887c1a71 -> FortiGate design_handoff intake.
Run id: 1e37d77f-f548-486e-8d16-9bfe8cd2c4d5  (data/fortigate-runs/)

Timing: 5.2 min total (314s). Slowest nodes:
   builder_review 45s, regenerate_after_judge 37s, judge 35s + 29s (two passes),
   refine 23s, autonomous_repair 21s, intent+repair 21s+21s.
   The judge->regenerate->judge loop was ~100s of the total.

Quality: Sonnet scored the FINAL cli_config 2/5.
   + Good coverage/structure: all 5 firewall policies w/ zones+UTM, explicit
     denies, NAT, SD-WAN zone/members/health-check/service, IPsec AES-256/DH14,
     SSL VPN portal, address objects.
   - ~7 CRITICAL FortiOS 7.4 syntax errors: system zone defined before its
     member interfaces; SD-WAN zone "WAN" collides with system zone "WAN";
     'config log tvlan setting' is not real; DHCP range-start/range-end + dns
     sub-block invalid; ipsec phase2 'set dst-end'; sdwan service src/dst-address
     keys wrong; health-check 'server-type icmp' wrong.

KEY FINDING / OPEN ITEM:
   The internal judge (gemma thinking-ON) MISSED all of the above CLI syntax
   errors -- it only flagged design-level concerns (no HA, local logging not
   scalable, VPN bound to wan1) with 0 blocking issues, then passed a config
   Sonnet rates 2/5. The gatekeeper is not catching FortiOS CLI correctness, and
   generation (gemma) emits real syntax bugs that survive review/refine/repair.

   Candidate follow-ups (not yet applied):
   (a) run 1-2 more fixtures to confirm the pattern;
   (b) try qwen3-coder-next as the judge (it caught syntax in review/refine);
   (c) try qwen3-coder-next for generation (100% valid, slower) vs gemma;
   (d) add a deterministic FortiOS syntax-lint pass between refine and judge.

   NOTE: the per-section bake-off could not see cross-section assembly conflicts
   (zone ordering, duplicate WAN zone) -- these only appear in the full pipeline.

--------------------------------------------------------------------------------
6. PROFILES
--------------------------------------------------------------------------------
config/model-profiles/local-quality.env  <- the applied bake-off-winner profile.
Other profiles (anthropic, xai-grok, openrouter-*, qwen*, etc.) remain available
for quick switching via the model-profile apply command.
================================================================================
