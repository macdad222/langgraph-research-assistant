#!/usr/bin/env python
"""Shadow/compare harness for the deterministic renderer (M2).

Given a captured FortiGate run JSON (from /data/fortigate-runs/<run_id>.json), this:
  1. extracts the implementation_intent + design + intake the legacy run used,
  2. calls the same build_config_model -> render path the graph uses (real LLM via LiteLLM),
  3. prints the rendered CLI + deterministic structure check,
  4. compares it against the legacy run's cli_config (line counts + block balance).

Run inside the app image, e.g.:
  docker run --rm --env-file agent-platform/.env \
    -v ~/dev/langgraph-research-assistant/agent-platform:/work -v /data:/data \
    -w /work agent-platform-langgraph-app python scripts/renderer_eval.py <run_id|path>

Env used: LITELLM_BASE_URL, LITELLM_API_KEY, and FORTIGATE_RENDER_MODEL_NAME
(falls back to FORTIGATE_INTENT_MODEL_NAME, then MODEL_NAME).
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.fortigate_config_builder import build_config_model, normalize_fortios_version
from app.fortigate_renderer import check_block_balance, render_config


def _load_run(arg: str) -> dict:
    path = Path(arg)
    if not path.exists():
        path = Path(os.getenv("FORTIGATE_RUNS_DIR", "/data/fortigate-runs")) / f"{arg}.json"
    return json.loads(path.read_text())


def _model_name() -> str:
    return (
        os.getenv("FORTIGATE_RENDER_MODEL_NAME")
        or os.getenv("FORTIGATE_INTENT_MODEL_NAME")
        or os.getenv("MODEL_NAME", "gemma-local")
    )


async def _run(run: dict) -> None:
    llm = ChatOpenAI(
        model=_model_name(),
        base_url=os.environ["LITELLM_BASE_URL"],
        api_key=os.environ["LITELLM_API_KEY"],
        temperature=0.2,
        extra_body={"metadata": {"agentic_gateway_openclaw_passthrough": True}},
    )

    async def invoke(system_text: str, human_text: str, run_name: str) -> str:
        resp = await llm.ainvoke(
            [SystemMessage(content=system_text), HumanMessage(content=human_text)],
            config={"run_name": run_name},
        )
        return str(resp.content)

    intake = run.get("intake", {})
    context = {
        "intake": intake,
        "fortigate_design": run.get("fortigate_design", {}),
        "implementation_intent": run.get("implementation_intent", {}),
        "intent_completeness_report": run.get("intent_completeness_report", {}),
        "change_impact": run.get("change_impact", {}),
        "standards": [],
    }
    version = normalize_fortios_version(str(intake.get("fortios_version", "")))

    print(f"== build_config_model (model={_model_name()}, fortios={version}) ==")
    model, errors = await build_config_model(invoke, context, fortios_version=version, retries=1)
    if model is None:
        print(f"FAILED to build config model after retries. Errors:\n- " + "\n- ".join(errors))
        return

    artifacts = render_config(model)
    rendered = artifacts["cli_config"]
    legacy = (run.get("config_artifacts", {}) or {}).get("cli_config", "")

    print("\n===== RENDERED CLI =====\n")
    print(rendered)
    print("\n===== COMPARISON =====")
    print(f"rendered lines:        {len(rendered.splitlines())}")
    print(f"legacy lines:          {len(legacy.splitlines())}")
    print(f"rendered struct issues:{check_block_balance(rendered)}")
    print(f"legacy struct issues:  {check_block_balance(legacy)}")
    print(f"requires_human_input:  {len(artifacts.get('requires_human_input', []))} item(s)")
    if errors:
        print(f"build retries needed:  {len(errors)}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    run = _load_run(sys.argv[1])
    asyncio.run(_run(run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
