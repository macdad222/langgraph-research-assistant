import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.fortigate_models import FortiGateJudgeReport


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return {}


async def run_frontier_judge(model: ChatOpenAI, review_packet: dict[str, Any], judge_model_name: str = "", run_name: str = "frontier_model_judge") -> FortiGateJudgeReport:
    response = await model.ainvoke(
        [
            SystemMessage(
                content=(
                    "You are an independent senior FortiGate architecture reviewer. "
                    "Review the proposed artifact package before human approval. "
                    "Return only JSON with keys: verdict, blocking_issues, warnings, standards_concerns, "
                    "config_risks, missing_questions, recommended_revisions, human_reviewer_focus. "
                    "verdict must be one of: pass, needs_revision, block. "
                    "Be conservative: block only for material outage/security/compliance risks, "
                    "needs_revision for fixable concerns, pass when the draft is fit for human review."
                )
            ),
            HumanMessage(content=json.dumps(review_packet, indent=2)),
        ],
        config={"run_name": run_name} if run_name else None,
    )
    data = _extract_json_object(str(response.content))
    if not data:
        data = {
            "verdict": "needs_revision",
            "blocking_issues": ["Judge model did not return structured JSON."],
            "warnings": [],
            "standards_concerns": [],
            "config_risks": [],
            "missing_questions": [],
            "recommended_revisions": ["Re-run judge review or inspect the draft manually."],
            "human_reviewer_focus": ["Structured judge output was unavailable."],
        }
    verdict = str(data.get("verdict") or "needs_revision").strip().lower()
    if verdict not in {"pass", "needs_revision", "block"}:
        verdict = "needs_revision"
    return FortiGateJudgeReport(
        verdict=verdict,
        blocking_issues=[str(item) for item in data.get("blocking_issues") or []],
        warnings=[str(item) for item in data.get("warnings") or []],
        standards_concerns=[str(item) for item in data.get("standards_concerns") or []],
        config_risks=[str(item) for item in data.get("config_risks") or []],
        missing_questions=[str(item) for item in data.get("missing_questions") or []],
        recommended_revisions=[str(item) for item in data.get("recommended_revisions") or []],
        human_reviewer_focus=[str(item) for item in data.get("human_reviewer_focus") or []],
        model=judge_model_name,
    )
