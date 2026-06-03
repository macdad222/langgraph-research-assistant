import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.fortigate_models import FortiGateJudgeReport


def _as_str_list(value: Any) -> list[str]:
    """Normalize a judge list field into a list of non-empty strings.

    Guards the case where a model returns a single string (e.g. a one-line
    ``human_reviewer_focus`` like "Verify ...") for a field declared as a list:
    iterating that string yields one entry per CHARACTER. Wrap a bare string in a
    single-element list before stringifying each item."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    elif not isinstance(value, (list, tuple, set)):
        value = [value]
    return [str(item).strip() for item in value if str(item).strip()]


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
        blocking_issues=_as_str_list(data.get("blocking_issues")),
        warnings=_as_str_list(data.get("warnings")),
        standards_concerns=_as_str_list(data.get("standards_concerns")),
        config_risks=_as_str_list(data.get("config_risks")),
        missing_questions=_as_str_list(data.get("missing_questions")),
        recommended_revisions=_as_str_list(data.get("recommended_revisions")),
        human_reviewer_focus=_as_str_list(data.get("human_reviewer_focus")),
        model=judge_model_name,
    )
