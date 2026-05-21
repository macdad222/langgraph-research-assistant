import json
import os
import re
from datetime import UTC, datetime
from time import perf_counter
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict
from urllib.parse import quote_plus, unquote, urlparse, parse_qs

import httpx
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt


class SourceCandidate(TypedDict, total=False):
    title: str
    url: str
    snippet: str


class SourceEvidence(TypedDict, total=False):
    title: str
    url: str
    snippet: str
    text: str
    status: str


class SourceScore(TypedDict, total=False):
    title: str
    url: str
    authority: str
    relevance: str
    risk: str
    score: int
    rationale: str


class Citation(TypedDict, total=False):
    claim: str
    title: str
    url: str
    evidence: str
    confidence: str


class CitationVerification(TypedDict, total=False):
    claim: str
    url: str
    verdict: str
    rationale: str


class QualityReport(TypedDict, total=False):
    unsupported_claims: list[str]
    weak_citations: list[str]
    missing_perspectives: list[str]
    source_risks: list[str]
    confidence_rationale: str
    revision_instructions: list[str]


class ExecutionTraceItem(TypedDict, total=False):
    node: str
    started_at: str
    duration_ms: int
    summary: str
    branch: str


class PolicyReport(TypedDict, total=False):
    passed: bool
    warnings: list[str]
    blocking_issues: list[str]
    checks: list[str]


class ResearchState(TypedDict, total=False):
    question: str
    mode: str
    constraints: str
    plan: list[str]
    source_strategy: list[str]
    sources: list[SourceCandidate]
    evidence: list[SourceEvidence]
    source_scores: list[SourceScore]
    gap_analysis: list[str]
    deepening_queries: list[str]
    deepened_sources: list[SourceCandidate]
    deepened_evidence: list[SourceEvidence]
    deepened_source_scores: list[SourceScore]
    deepened_areas: list[str]
    repair_iterations: int
    repair_queries: list[str]
    repair_sources: list[SourceCandidate]
    repair_evidence: list[SourceEvidence]
    repair_source_scores: list[SourceScore]
    plan_results: list[str]
    findings: list[str]
    citations: list[Citation]
    citation_verifications: list[CitationVerification]
    quality_report: QualityReport
    confidence: str
    follow_up_questions: list[str]
    answer: str
    limitations: list[str]
    execution_trace: list[ExecutionTraceItem]
    policy_report: PolicyReport
    memory_context: dict[str, Any]


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model did not return a JSON object")
    return json.loads(stripped[start : end + 1])


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _as_verifications(value: Any) -> list[CitationVerification]:
    verifications: list[CitationVerification] = []
    if not isinstance(value, list):
        return verifications
    allowed = {"supported", "partially_supported", "unsupported", "unclear"}
    for item in value:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or "").strip()
        url = str(item.get("url") or "").strip()
        verdict = str(item.get("verdict") or "unclear").strip().lower()
        if verdict not in allowed:
            verdict = "unclear"
        if not claim or not url:
            continue
        verifications.append(
            {
                "claim": claim,
                "url": url,
                "verdict": verdict,
                "rationale": str(item.get("rationale") or "").strip(),
            }
        )
    return verifications


def _as_quality_report(value: Any) -> QualityReport:
    if not isinstance(value, dict):
        return {
            "unsupported_claims": [],
            "weak_citations": [],
            "missing_perspectives": [],
            "source_risks": [],
            "confidence_rationale": "Not provided.",
            "revision_instructions": [],
        }
    return {
        "unsupported_claims": _as_string_list(value.get("unsupported_claims")),
        "weak_citations": _as_string_list(value.get("weak_citations")),
        "missing_perspectives": _as_string_list(value.get("missing_perspectives")),
        "source_risks": _as_string_list(value.get("source_risks")),
        "confidence_rationale": str(value.get("confidence_rationale") or "Not provided.").strip(),
        "revision_instructions": _as_string_list(value.get("revision_instructions")),
    }


def _as_citations(value: Any) -> list[Citation]:
    citations: list[Citation] = []
    if not isinstance(value, list):
        return citations
    for item in value:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or "").strip()
        url = str(item.get("url") or "").strip()
        if not claim or not url:
            continue
        citations.append(
            {
                "claim": claim,
                "title": str(item.get("title") or "").strip(),
                "url": url,
                "evidence": str(item.get("evidence") or "").strip(),
                "confidence": str(item.get("confidence") or "medium").strip().lower(),
            }
        )
    return citations


def _clean_text(text: str, limit: int = 3500) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned[:limit]


def _normalize_duckduckgo_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        uddg = parse_qs(parsed.query).get("uddg")
        if uddg:
            return unquote(uddg[0])
    return url


async def _tavily_search(query: str, max_results: int = 5) -> list[SourceCandidate]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        response = await client.post("https://api.tavily.com/search", json=payload)
        response.raise_for_status()
    data = response.json()

    sources: list[SourceCandidate] = []
    for item in data.get("results", []):
        url = str(item.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        sources.append(
            {
                "title": str(item.get("title") or url).strip(),
                "url": url,
                "snippet": str(item.get("content") or "").strip(),
                "provider": "tavily",
            }
        )
        if len(sources) >= max_results:
            break
    return sources


async def _duckduckgo_search(query: str, max_results: int = 5) -> list[SourceCandidate]:
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {"User-Agent": "Mozilla/5.0 research-assistant/0.1"}
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    results: list[SourceCandidate] = []
    for result in soup.select(".result"):
        link = result.select_one(".result__a")
        if link is None:
            continue
        href = link.get("href") or ""
        title = link.get_text(" ", strip=True)
        snippet_el = result.select_one(".result__snippet")
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        normalized = _normalize_duckduckgo_url(href)
        if not normalized.startswith(("http://", "https://")):
            continue
        results.append({"title": title, "url": normalized, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


async def _search_sources(query: str, max_results: int = 5) -> tuple[list[SourceCandidate], str]:
    try:
        tavily_results = await _tavily_search(query, max_results=max_results)
        if tavily_results:
            return tavily_results, "tavily"
    except Exception:
        # Fall through to the free fallback. The graph records provider choice in limitations.
        pass

    return await _duckduckgo_search(query, max_results=max_results), "duckduckgo"


async def _fetch_source(source: SourceCandidate) -> SourceEvidence:
    headers = {"User-Agent": "Mozilla/5.0 research-assistant/0.1"}
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=headers) as client:
            response = await client.get(source["url"])
            response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        text = _clean_text(soup.get_text(" ", strip=True))
        return {
            "title": source.get("title", ""),
            "url": source["url"],
            "snippet": source.get("snippet", ""),
            "text": text or source.get("snippet", ""),
            "status": "ok" if (text or source.get("snippet")) else "empty",
        }
    except Exception as exc:
        return {
            "title": source.get("title", ""),
            "url": source.get("url", ""),
            "snippet": source.get("snippet", ""),
            "text": "",
            "status": f"error: {type(exc).__name__}: {exc}",
        }


def _score_source(source: SourceEvidence, question: str) -> SourceScore:
    url = source.get("url", "")
    title = source.get("title", "")
    text = f"{title} {source.get('snippet', '')} {source.get('text', '')}".lower()
    host = urlparse(url).netloc.lower().removeprefix("www.")
    question_terms = [term for term in re.findall(r"[a-z0-9]{4,}", question.lower()) if term not in {"what", "with", "from", "that", "this", "should"}]
    matched_terms = sorted({term for term in question_terms if term in text})

    authority = "unknown"
    score = 45
    rationale_parts: list[str] = []

    if any(host.endswith(domain) for domain in (".gov", ".edu")):
        authority = "primary"
        score += 25
        rationale_parts.append("government/education domain")
    elif any(name in host for name in ("docs.", "documentation", "github.com", "python.org", "langchain.com", "langgraph", "tavily.com", "exa.ai")):
        authority = "official_or_vendor"
        score += 22
        rationale_parts.append("official/vendor/developer source")
    elif any(name in host for name in ("medium.com", "substack.com", "dev.to", "reddit.com", "news.ycombinator.com")):
        authority = "community"
        score += 5
        rationale_parts.append("community source")
    elif any(name in host for name in ("firecrawl.dev", "composio.dev", "zapier.com")):
        authority = "vendor_or_industry"
        score += 12
        rationale_parts.append("vendor/industry source")

    if len(matched_terms) >= 4:
        relevance = "high"
        score += 20
        rationale_parts.append(f"matches query terms: {', '.join(matched_terms[:6])}")
    elif len(matched_terms) >= 2:
        relevance = "medium"
        score += 10
        rationale_parts.append(f"matches query terms: {', '.join(matched_terms[:4])}")
    else:
        relevance = "low"
        score -= 10
        rationale_parts.append("few direct query-term matches")

    if source.get("status") != "ok":
        risk = "high"
        score -= 25
        rationale_parts.append("source could not be fully read")
    elif authority in {"unknown", "community"}:
        risk = "medium"
    else:
        risk = "low"

    score = max(0, min(100, score))
    return {
        "title": title,
        "url": url,
        "authority": authority,
        "relevance": relevance,
        "risk": risk,
        "score": score,
        "rationale": "; ".join(rationale_parts),
    }


def _trace_update(
    state: ResearchState,
    updates: ResearchState,
    node: str,
    started_at: str,
    started_perf: float,
    summary: str,
    branch: str = "",
) -> ResearchState:
    trace_item: ExecutionTraceItem = {
        "node": node,
        "started_at": started_at,
        "duration_ms": int((perf_counter() - started_perf) * 1000),
        "summary": summary,
    }
    if branch:
        trace_item["branch"] = branch
    return {
        **updates,
        "execution_trace": [*state.get("execution_trace", []), trace_item],
    }


def _trace_start() -> tuple[str, float]:
    return datetime.now(UTC).isoformat(), perf_counter()


def build_research_graph(
    model: ChatOpenAI,
    checkpointer: AsyncRedisSaver,
    human_review_interrupt: bool = False,
    memory_retriever: Callable[[str], Awaitable[dict[str, Any]]] | None = None,
):
    async def retrieve_memory(state: ResearchState) -> ResearchState:
        started_at, started_perf = _trace_start()
        if memory_retriever is None:
            return _trace_update(
                state,
                {"memory_context": {"related_runs": [], "prior_claims": [], "trusted_sources": []}},
                "retrieve_memory",
                started_at,
                started_perf,
                "Skipped Neo4j retrieval because memory is unavailable.",
                branch="skip",
            )
        try:
            context = await memory_retriever(state["question"])
        except Exception:
            context = {"related_runs": [], "prior_claims": [], "trusted_sources": []}
            branch = "error"
            summary = "Neo4j retrieval failed; continuing without memory context."
        else:
            branch = "retrieved"
            summary = (
                f"Retrieved {len(context.get('related_runs', []))} related run(s), "
                f"{len(context.get('prior_claims', []))} prior claim(s), and "
                f"{len(context.get('trusted_sources', []))} reusable source(s)."
            )
        return _trace_update(
            state,
            {"memory_context": context},
            "retrieve_memory",
            started_at,
            started_perf,
            summary,
            branch=branch,
        )

    async def plan_research(state: ResearchState) -> ResearchState:
        started_at, started_perf = _trace_start()
        response = await model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are a research planning node. Return only JSON with keys: "
                        "plan, source_strategy, limitations. Each value must be an array of strings. "
                        "Plan for live web search and source reading. Use prior memory only as context: "
                        "reuse trustworthy prior sources or claims when relevant, but identify gaps or stale facts "
                        "that still need live verification. Do not fabricate citations."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Mode: {state.get('mode', 'quick')}\n"
                        f"Question: {state['question']}\n"
                        f"Constraints/context: {state.get('constraints') or 'none'}"
                    )
                ),
            ]
        )
        data = _extract_json_object(response.content)
        plan = _as_string_list(data.get("plan"))
        source_strategy = _as_string_list(data.get("source_strategy"))
        limitations = _as_string_list(data.get("limitations"))
        return _trace_update(
            state,
            {"plan": plan, "source_strategy": source_strategy, "limitations": limitations},
            "plan_research",
            started_at,
            started_perf,
            f"Created {len(plan)} plan steps and {len(source_strategy)} source strategy items.",
        )

    async def search_sources(state: ResearchState) -> ResearchState:
        started_at, started_perf = _trace_start()
        trusted_hosts = [source.get("host") for source in state.get("memory_context", {}).get("trusted_sources", []) if source.get("host")]
        memory_hint = " ".join(trusted_hosts[:3])
        query = f"{state['question']} {state.get('constraints') or ''} {memory_hint}".strip()
        sources, provider = await _search_sources(query, max_results=5)
        limitations = list(state.get("limitations", []))
        limitations.append(f"Search provider used: {provider}.")
        if not sources:
            limitations.append("Web search returned no candidate sources.")
        return _trace_update(
            state,
            {"sources": sources, "limitations": limitations},
            "search_sources",
            started_at,
            started_perf,
            f"Found {len(sources)} candidate sources via {provider}.",
        )

    async def read_sources(state: ResearchState) -> ResearchState:
        started_at, started_perf = _trace_start()
        evidence: list[SourceEvidence] = []
        for source in state.get("sources", [])[:5]:
            evidence.append(await _fetch_source(source))
        limitations = list(state.get("limitations", []))
        failed = [item for item in evidence if item.get("status") != "ok"]
        if failed:
            limitations.append(f"{len(failed)} source(s) could not be fully read.")
        return _trace_update(
            state,
            {"evidence": evidence, "limitations": limitations},
            "read_sources",
            started_at,
            started_perf,
            f"Read {len(evidence)} sources; {len(failed)} failed or were incomplete.",
        )

    async def score_sources(state: ResearchState) -> ResearchState:
        started_at, started_perf = _trace_start()
        scores = [_score_source(source, state["question"]) for source in state.get("evidence", [])]
        scores.sort(key=lambda item: item.get("score", 0), reverse=True)
        return _trace_update(
            state,
            {"source_scores": scores},
            "score_sources",
            started_at,
            started_perf,
            f"Scored {len(scores)} sources.",
        )

    async def synthesize_research(state: ResearchState) -> ResearchState:
        started_at, started_perf = _trace_start()
        response = await model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are a source-grounded research synthesis node. Return only JSON with keys: "
                        "answer, plan_results, findings, citations, confidence, follow_up_questions. "
                        "plan_results, findings, and follow_up_questions must be arrays of strings. "
                        "plan_results must summarize what was actually learned for each executed plan step using retrieved evidence. "
                        "citations must be an array of objects with claim, title, url, evidence, confidence. "
                        "Every source-grounded finding should have a citation object with a URL from the supplied evidence. "
                        "confidence must be one of: low, medium, high. "
                        "Use only the supplied evidence for source-grounded claims. Mention uncertainty when evidence is thin. "
                        "The plan has already been executed by previous graph nodes, so do not describe future steps or say what you would do. "
                        "The answer must synthesize the evidence that was actually retrieved and read. "
                        "Format answer as clean Markdown with short section headings (## Recommendation, ## Rationale, ## Caveats) "
                        "and concise bullets where useful. Do not include a top-level title."
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        {
                            "mode": state.get("mode", "quick"),
                            "question": state["question"],
                            "constraints": state.get("constraints") or "none",
                            "plan": state.get("plan", []),
                            "source_strategy": state.get("source_strategy", []),
                            "sources": state.get("sources", []),
                            "evidence": state.get("evidence", []),
                            "source_scores": state.get("source_scores", []),
                            "limitations": state.get("limitations", []),
                        },
                        indent=2,
                    )
                ),
            ]
        )
        data = _extract_json_object(response.content)
        findings = _as_string_list(data.get("findings"))
        citations = _as_citations(data.get("citations"))
        return _trace_update(
            state,
            {
                "answer": str(data.get("answer", "")).strip(),
                "plan_results": _as_string_list(data.get("plan_results")),
                "findings": findings,
                "citations": citations,
                "confidence": str(data.get("confidence", "medium")).strip().lower(),
                "follow_up_questions": _as_string_list(data.get("follow_up_questions")),
            },
            "synthesize_research",
            started_at,
            started_perf,
            f"Drafted answer with {len(findings)} findings and {len(citations)} citations.",
        )

    async def analyze_gaps(state: ResearchState) -> ResearchState:
        started_at, started_perf = _trace_start()
        mode = state.get("mode", "quick").lower()
        if mode not in {"deep", "comparison"}:
            return _trace_update(
                state,
                {"gap_analysis": [], "deepening_queries": [], "deepened_areas": []},
                "analyze_gaps",
                started_at,
                started_perf,
                "Skipped iterative deepening for quick mode.",
                branch="critique",
            )

        response = await model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are a research reflection node. Return only JSON with keys: "
                        "gap_analysis and deepening_queries. Both must be arrays of strings. "
                        "Identify the most valuable areas where one more targeted web search would add depth, "
                        "resolve ambiguity, cover a missing comparison dimension, or improve weak evidence. "
                        "Return at most 4 gap_analysis items and at most 3 concrete search queries."
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        {
                            "mode": mode,
                            "question": state["question"],
                            "constraints": state.get("constraints") or "none",
                            "plan": state.get("plan", []),
                            "answer": state.get("answer", ""),
                            "findings": state.get("findings", []),
                            "citations": state.get("citations", []),
                            "source_scores": state.get("source_scores", []),
                            "limitations": state.get("limitations", []),
                        },
                        indent=2,
                    )
                ),
            ]
        )
        data = _extract_json_object(response.content)
        gap_analysis = _as_string_list(data.get("gap_analysis"))[:4]
        deepening_queries = _as_string_list(data.get("deepening_queries"))[:3]
        branch = "deepen" if deepening_queries else "critique"
        return _trace_update(
            state,
            {"gap_analysis": gap_analysis, "deepening_queries": deepening_queries},
            "analyze_gaps",
            started_at,
            started_perf,
            f"Identified {len(gap_analysis)} gaps and {len(deepening_queries)} deepening queries.",
            branch=branch,
        )

    def route_after_gap_analysis(state: ResearchState) -> str:
        mode = state.get("mode", "quick").lower()
        if mode in {"deep", "comparison"} and state.get("deepening_queries"):
            return "deepen"
        return "critique"

    async def search_deeper_sources(state: ResearchState) -> ResearchState:
        started_at, started_perf = _trace_start()
        seen_urls = {
            source.get("url")
            for source in [*state.get("sources", []), *state.get("evidence", [])]
            if source.get("url")
        }
        deepened_sources: list[SourceCandidate] = []
        providers: list[str] = []
        for query in state.get("deepening_queries", [])[:3]:
            sources, provider = await _search_sources(query, max_results=3)
            providers.append(provider)
            for source in sources:
                url = source.get("url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                deepened_sources.append(source)
                if len(deepened_sources) >= 6:
                    break
            if len(deepened_sources) >= 6:
                break

        limitations = list(state.get("limitations", []))
        if providers:
            limitations.append(f"Iterative deepening search providers used: {', '.join(providers)}.")
        if not deepened_sources:
            limitations.append("Iterative deepening found no new candidate sources.")
        return _trace_update(
            state,
            {"deepened_sources": deepened_sources, "limitations": limitations},
            "search_deeper_sources",
            started_at,
            started_perf,
            f"Found {len(deepened_sources)} additional candidate sources.",
        )

    async def read_deepened_sources(state: ResearchState) -> ResearchState:
        started_at, started_perf = _trace_start()
        evidence: list[SourceEvidence] = []
        for source in state.get("deepened_sources", [])[:6]:
            evidence.append(await _fetch_source(source))
        limitations = list(state.get("limitations", []))
        failed = [item for item in evidence if item.get("status") != "ok"]
        if failed:
            limitations.append(f"{len(failed)} deepening source(s) could not be fully read.")
        return _trace_update(
            state,
            {"deepened_evidence": evidence, "limitations": limitations},
            "read_deepened_sources",
            started_at,
            started_perf,
            f"Read {len(evidence)} deepening sources; {len(failed)} failed or were incomplete.",
        )

    async def score_deepened_sources(state: ResearchState) -> ResearchState:
        started_at, started_perf = _trace_start()
        scores = [_score_source(source, state["question"]) for source in state.get("deepened_evidence", [])]
        scores.sort(key=lambda item: item.get("score", 0), reverse=True)
        return _trace_update(
            state,
            {"deepened_source_scores": scores},
            "score_deepened_sources",
            started_at,
            started_perf,
            f"Scored {len(scores)} deepening sources.",
        )

    async def resynthesize_research(state: ResearchState) -> ResearchState:
        started_at, started_perf = _trace_start()
        response = await model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are a final source-grounded research synthesis node after iterative deepening or policy repair. "
                        "Return only JSON with keys: answer, plan_results, deepened_areas, findings, citations, "
                        "confidence, follow_up_questions. plan_results, deepened_areas, findings, and "
                        "follow_up_questions must be arrays of strings. deepened_areas must explain what the second "
                        "research pass added, corrected, or made more nuanced, grouped by gap or follow-up query. "
                        "Use only supplied evidence for source-grounded claims. Every important claim should have a citation "
                        "object with a URL from supplied original or deepened evidence. Do not describe future steps. "
                        "Format answer as clean Markdown with short section headings (## Recommendation, ## Rationale, ## Caveats) "
                        "and concise bullets where useful. Do not include a top-level title."
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        {
                            "mode": state.get("mode", "quick"),
                            "question": state["question"],
                            "constraints": state.get("constraints") or "none",
                            "plan": state.get("plan", []),
                            "source_strategy": state.get("source_strategy", []),
                            "initial_answer": state.get("answer", ""),
                            "initial_findings": state.get("findings", []),
                            "gap_analysis": state.get("gap_analysis", []),
                            "deepening_queries": state.get("deepening_queries", []),
                            "evidence": state.get("evidence", []),
                            "deepened_evidence": state.get("deepened_evidence", []),
                            "repair_queries": state.get("repair_queries", []),
                            "repair_evidence": state.get("repair_evidence", []),
                            "policy_report": state.get("policy_report", {}),
                            "source_scores": state.get("source_scores", []),
                            "deepened_source_scores": state.get("deepened_source_scores", []),
                            "repair_source_scores": state.get("repair_source_scores", []),
                            "limitations": state.get("limitations", []),
                        },
                        indent=2,
                    )
                ),
            ]
        )
        data = _extract_json_object(response.content)
        deepened_areas = _as_string_list(data.get("deepened_areas"))
        findings = _as_string_list(data.get("findings"))
        citations = _as_citations(data.get("citations"))
        return _trace_update(
            state,
            {
                "answer": str(data.get("answer", "")).strip(),
                "plan_results": _as_string_list(data.get("plan_results")),
                "deepened_areas": deepened_areas,
                "findings": findings,
                "citations": citations,
                "confidence": str(data.get("confidence", "medium")).strip().lower(),
                "follow_up_questions": _as_string_list(data.get("follow_up_questions")),
            },
            "resynthesize_research",
            started_at,
            started_perf,
            f"Resynthesized with {len(deepened_areas)} deepened areas, {len(findings)} findings, and {len(citations)} citations.",
        )

    async def critique_research(state: ResearchState) -> ResearchState:
        started_at, started_perf = _trace_start()
        response = await model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are a skeptical research quality critic. Return only JSON with key: quality_report. "
                        "quality_report must be an object with keys: unsupported_claims, weak_citations, "
                        "missing_perspectives, source_risks, confidence_rationale, revision_instructions. "
                        "All keys except confidence_rationale must be arrays of strings. Identify concrete issues only. "
                        "Do not use outside knowledge; judge against supplied evidence, citations, source scores, and limitations."
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        {
                            "question": state["question"],
                            "mode": state.get("mode", "quick"),
                            "constraints": state.get("constraints") or "none",
                            "answer": state.get("answer", ""),
                            "plan_results": state.get("plan_results", []),
                            "deepened_areas": state.get("deepened_areas", []),
                            "findings": state.get("findings", []),
                            "citations": state.get("citations", []),
                            "confidence": state.get("confidence", "medium"),
                            "evidence": state.get("evidence", []),
                            "deepened_evidence": state.get("deepened_evidence", []),
                            "source_scores": state.get("source_scores", []),
                            "deepened_source_scores": state.get("deepened_source_scores", []),
                            "limitations": state.get("limitations", []),
                        },
                        indent=2,
                    )
                ),
            ]
        )
        data = _extract_json_object(response.content)
        quality_report = _as_quality_report(data.get("quality_report"))
        issue_count = sum(
            len(quality_report.get(key, []))
            for key in ["unsupported_claims", "weak_citations", "missing_perspectives", "source_risks"]
        )
        return _trace_update(
            state,
            {"quality_report": quality_report},
            "critique_research",
            started_at,
            started_perf,
            f"Produced quality report with {issue_count} issues and {len(quality_report.get('revision_instructions', []))} revision instructions.",
        )

    async def revise_final_answer(state: ResearchState) -> ResearchState:
        started_at, started_perf = _trace_start()
        response = await model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are a final research editor. Return only JSON with keys: answer, findings, citations, "
                        "confidence, follow_up_questions, limitations. findings, follow_up_questions, and limitations "
                        "must be arrays of strings. citations must be an array of objects with claim, title, url, "
                        "evidence, confidence. Revise the draft using the quality_report: remove unsupported claims, "
                        "downgrade confidence where needed, make uncertainty explicit, and preserve only claims grounded "
                        "in supplied evidence. Do not add new facts from outside the supplied evidence. "
                        "Format the final answer as polished Markdown with short section headings and bullets. "
                        "Avoid raw formatting artifacts; use headings instead of long bold lead-ins."
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        {
                            "question": state["question"],
                            "mode": state.get("mode", "quick"),
                            "constraints": state.get("constraints") or "none",
                            "draft_answer": state.get("answer", ""),
                            "draft_findings": state.get("findings", []),
                            "draft_citations": state.get("citations", []),
                            "draft_confidence": state.get("confidence", "medium"),
                            "quality_report": state.get("quality_report", {}),
                            "evidence": state.get("evidence", []),
                            "deepened_evidence": state.get("deepened_evidence", []),
                            "source_scores": state.get("source_scores", []),
                            "deepened_source_scores": state.get("deepened_source_scores", []),
                            "existing_limitations": state.get("limitations", []),
                        },
                        indent=2,
                    )
                ),
            ]
        )
        data = _extract_json_object(response.content)
        limitations = list(state.get("limitations", []))
        for item in _as_string_list(data.get("limitations")):
            if item not in limitations:
                limitations.append(item)
        findings = _as_string_list(data.get("findings"))
        citations = _as_citations(data.get("citations"))
        return _trace_update(
            state,
            {
                "answer": str(data.get("answer", "")).strip(),
                "findings": findings,
                "citations": citations,
                "confidence": str(data.get("confidence", state.get("confidence", "medium"))).strip().lower(),
                "follow_up_questions": _as_string_list(data.get("follow_up_questions")),
                "limitations": limitations,
            },
            "revise_final_answer",
            started_at,
            started_perf,
            f"Revised final answer with {len(findings)} findings and {len(citations)} citations.",
        )

    async def check_policy(state: ResearchState) -> ResearchState:
        started_at, started_perf = _trace_start()
        mode = state.get("mode", "quick").lower()
        citations = state.get("citations", [])
        quality_report = state.get("quality_report", {})
        confidence = state.get("confidence", "medium").lower()
        warnings: list[str] = []
        blocking_issues: list[str] = []
        checks: list[str] = []

        min_citations = 4 if mode in {"deep", "comparison"} else 2
        if len(citations) >= min_citations:
            checks.append(f"Citation count passed: {len(citations)} citation(s), minimum {min_citations}.")
        else:
            blocking_issues.append(f"Only {len(citations)} citation(s) returned; minimum for {mode} mode is {min_citations}.")

        unsupported_count = len(quality_report.get("unsupported_claims", []))
        if unsupported_count and confidence == "high":
            blocking_issues.append("High confidence is not allowed while unsupported claims remain in the quality report.")
        elif unsupported_count:
            warnings.append(f"Quality report identified {unsupported_count} unsupported or under-supported claim(s).")
        else:
            checks.append("No unsupported claims were flagged by the critic.")

        weak_citation_count = len(quality_report.get("weak_citations", []))
        if weak_citation_count >= 3:
            warnings.append(f"Quality report identified {weak_citation_count} weak citation(s); consider another source pass.")
        elif weak_citation_count:
            warnings.append(f"Quality report identified {weak_citation_count} weak citation(s).")
        else:
            checks.append("No weak citations were flagged by the critic.")

        cited_hosts = {
            urlparse(citation.get("url", "")).netloc.lower().removeprefix("www.")
            for citation in citations
            if citation.get("url")
        }
        if len(cited_hosts) >= 3 or len(citations) < 3:
            checks.append(f"Source diversity check observed {len(cited_hosts)} cited domain(s).")
        else:
            warnings.append(f"Low source diversity: citations come from only {len(cited_hosts)} domain(s).")

        if mode in {"deep", "comparison"}:
            if state.get("deepening_queries") and state.get("deepened_areas"):
                checks.append("Deepening policy passed: follow-up queries and deepened areas are present.")
            else:
                warnings.append("Deep/comparison mode did not produce a complete deepening pass.")

        policy_report: PolicyReport = {
            "passed": not blocking_issues,
            "warnings": warnings,
            "blocking_issues": blocking_issues,
            "checks": checks,
        }
        summary = f"Policy {'passed' if policy_report['passed'] else 'needs attention'} with {len(blocking_issues)} blocking issue(s) and {len(warnings)} warning(s)."
        return _trace_update(
            state,
            {"policy_report": policy_report},
            "check_policy",
            started_at,
            started_perf,
            summary,
        )

    def route_after_policy(state: ResearchState) -> str:
        mode = state.get("mode", "quick").lower()
        policy_report = state.get("policy_report", {})
        has_policy_findings = bool(policy_report.get("blocking_issues") or policy_report.get("warnings"))
        repair_iterations = int(state.get("repair_iterations", 0) or 0)
        if mode in {"deep", "comparison"} and has_policy_findings and repair_iterations < 1:
            return "repair"
        return "verify"

    async def plan_policy_repair(state: ResearchState) -> ResearchState:
        started_at, started_perf = _trace_start()
        response = await model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are a policy repair planning node. Return only JSON with key: repair_queries. "
                        "repair_queries must be an array of at most 3 concrete web search queries. "
                        "Generate targeted searches that address the policy warnings or blocking issues, especially "
                        "low source diversity, weak citations, missing perspectives, or too few citations."
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        {
                            "question": state["question"],
                            "mode": state.get("mode", "quick"),
                            "constraints": state.get("constraints") or "none",
                            "policy_report": state.get("policy_report", {}),
                            "quality_report": state.get("quality_report", {}),
                            "human_review_decision": state.get("human_review_decision", ""),
                            "human_review_notes": state.get("human_review_notes", ""),
                            "human_selected_issues": state.get("human_selected_issues", []),
                            "existing_citations": state.get("citations", []),
                            "existing_sources": state.get("sources", []),
                            "deepened_sources": state.get("deepened_sources", []),
                        },
                        indent=2,
                    )
                ),
            ]
        )
        data = _extract_json_object(response.content)
        repair_queries = _as_string_list(data.get("repair_queries"))[:3]
        return _trace_update(
            state,
            {
                "repair_queries": repair_queries,
                "repair_iterations": int(state.get("repair_iterations", 0) or 0) + 1,
            },
            "plan_policy_repair",
            started_at,
            started_perf,
            f"Generated {len(repair_queries)} policy repair queries.",
        )

    async def search_policy_repair_sources(state: ResearchState) -> ResearchState:
        started_at, started_perf = _trace_start()
        seen_urls = {
            source.get("url")
            for source in [
                *state.get("sources", []),
                *state.get("evidence", []),
                *state.get("deepened_sources", []),
                *state.get("deepened_evidence", []),
            ]
            if source.get("url")
        }
        repair_sources: list[SourceCandidate] = []
        providers: list[str] = []
        for query in state.get("repair_queries", [])[:3]:
            sources, provider = await _search_sources(query, max_results=3)
            providers.append(provider)
            for source in sources:
                url = source.get("url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                repair_sources.append(source)
                if len(repair_sources) >= 6:
                    break
            if len(repair_sources) >= 6:
                break
        limitations = list(state.get("limitations", []))
        if providers:
            limitations.append(f"Policy repair search providers used: {', '.join(providers)}.")
        if not repair_sources:
            limitations.append("Policy repair found no new candidate sources.")
        return _trace_update(
            state,
            {"repair_sources": repair_sources, "limitations": limitations},
            "search_policy_repair_sources",
            started_at,
            started_perf,
            f"Found {len(repair_sources)} policy repair candidate sources.",
        )

    async def read_policy_repair_sources(state: ResearchState) -> ResearchState:
        started_at, started_perf = _trace_start()
        evidence: list[SourceEvidence] = []
        for source in state.get("repair_sources", [])[:6]:
            evidence.append(await _fetch_source(source))
        limitations = list(state.get("limitations", []))
        failed = [item for item in evidence if item.get("status") != "ok"]
        if failed:
            limitations.append(f"{len(failed)} policy repair source(s) could not be fully read.")
        return _trace_update(
            state,
            {"repair_evidence": evidence, "limitations": limitations},
            "read_policy_repair_sources",
            started_at,
            started_perf,
            f"Read {len(evidence)} policy repair sources; {len(failed)} failed or were incomplete.",
        )

    async def score_policy_repair_sources(state: ResearchState) -> ResearchState:
        started_at, started_perf = _trace_start()
        scores = [_score_source(source, state["question"]) for source in state.get("repair_evidence", [])]
        scores.sort(key=lambda item: item.get("score", 0), reverse=True)
        return _trace_update(
            state,
            {"repair_source_scores": scores},
            "score_policy_repair_sources",
            started_at,
            started_perf,
            f"Scored {len(scores)} policy repair sources.",
        )

    async def human_review_checkpoint(state: ResearchState) -> ResearchState:
        started_at, started_perf = _trace_start()
        policy_report = state.get("policy_report", {})
        quality_report = state.get("quality_report", {})
        default_issues = [
            *policy_report.get("blocking_issues", []),
            *policy_report.get("warnings", []),
            *quality_report.get("unsupported_claims", []),
            *quality_report.get("weak_citations", []),
            *quality_report.get("missing_perspectives", []),
        ]
        review = interrupt(
            {
                "stage": "human_review",
                "question": state.get("question", ""),
                "mode": state.get("mode", "quick"),
                "answer": state.get("answer", ""),
                "confidence": state.get("confidence", "medium"),
                "policy_report": policy_report,
                "quality_report": quality_report,
                "suggested_issues": default_issues,
                "instructions": "Approve the final answer or request one more repair pass with reviewer notes.",
            }
        )
        if not isinstance(review, dict):
            review = {"decision": "approved", "reviewer_notes": str(review or ""), "selected_issues": []}
        decision = str(review.get("decision", "approved")).strip().lower()
        if decision not in {"approved", "needs_work"}:
            decision = "approved"
        selected_issues = _as_string_list(review.get("selected_issues")) or default_issues
        reviewer_notes = str(review.get("reviewer_notes", "")).strip()
        status = "approved" if decision == "approved" else "needs_human_requested_repair"
        return _trace_update(
            state,
            {
                "human_review_decision": decision,
                "human_review_notes": reviewer_notes,
                "human_selected_issues": selected_issues,
                "human_review_required": False,
                "final_status": status,
            },
            "human_review_checkpoint",
            started_at,
            started_perf,
            f"Human review decision: {decision} with {len(selected_issues)} selected issue(s).",
            branch=decision,
        )

    def route_after_human_review(state: ResearchState) -> str:
        if state.get("human_review_decision") == "needs_work":
            return "repair"
        return "verify"

    async def verify_citations(state: ResearchState) -> ResearchState:
        started_at, started_perf = _trace_start()
        citations = state.get("citations", [])
        if not citations:
            return _trace_update(
                state,
                {"citation_verifications": []},
                "verify_citations",
                started_at,
                started_perf,
                "Skipped citation verification because no citations were returned.",
            )

        response = await model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are a citation verification node. Return only JSON with key: citation_verifications. "
                        "citation_verifications must be an array of objects with claim, url, verdict, rationale. "
                        "verdict must be one of: supported, partially_supported, unsupported, unclear. "
                        "Judge only whether the provided evidence text supports the cited claim. Do not use outside knowledge."
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        {
                            "question": state["question"],
                            "citations": citations,
                            "evidence": state.get("evidence", []),
                            "deepened_evidence": state.get("deepened_evidence", []),
                            "repair_evidence": state.get("repair_evidence", []),
                        },
                        indent=2,
                    )
                ),
            ]
        )
        data = _extract_json_object(response.content)
        verifications = _as_verifications(data.get("citation_verifications"))
        return _trace_update(
            state,
            {"citation_verifications": verifications},
            "verify_citations",
            started_at,
            started_perf,
            f"Verified {len(verifications)} citations.",
        )

    graph = StateGraph(ResearchState)
    graph.add_node("retrieve_memory", retrieve_memory)
    graph.add_node("plan_research", plan_research)
    graph.add_node("search_sources", search_sources)
    graph.add_node("read_sources", read_sources)
    graph.add_node("score_sources", score_sources)
    graph.add_node("synthesize_research", synthesize_research)
    graph.add_node("analyze_gaps", analyze_gaps)
    graph.add_node("search_deeper_sources", search_deeper_sources)
    graph.add_node("read_deepened_sources", read_deepened_sources)
    graph.add_node("score_deepened_sources", score_deepened_sources)
    graph.add_node("resynthesize_research", resynthesize_research)
    graph.add_node("critique_research", critique_research)
    graph.add_node("revise_final_answer", revise_final_answer)
    graph.add_node("check_policy", check_policy)
    if human_review_interrupt:
        graph.add_node("human_review_checkpoint", human_review_checkpoint)
    graph.add_node("plan_policy_repair", plan_policy_repair)
    graph.add_node("search_policy_repair_sources", search_policy_repair_sources)
    graph.add_node("read_policy_repair_sources", read_policy_repair_sources)
    graph.add_node("score_policy_repair_sources", score_policy_repair_sources)
    graph.add_node("verify_citations", verify_citations)
    graph.set_entry_point("retrieve_memory")
    graph.add_edge("retrieve_memory", "plan_research")
    graph.add_edge("plan_research", "search_sources")
    graph.add_edge("search_sources", "read_sources")
    graph.add_edge("read_sources", "score_sources")
    graph.add_edge("score_sources", "synthesize_research")
    graph.add_edge("synthesize_research", "analyze_gaps")
    graph.add_conditional_edges(
        "analyze_gaps",
        route_after_gap_analysis,
        {"deepen": "search_deeper_sources", "critique": "critique_research"},
    )
    graph.add_edge("search_deeper_sources", "read_deepened_sources")
    graph.add_edge("read_deepened_sources", "score_deepened_sources")
    graph.add_edge("score_deepened_sources", "resynthesize_research")
    graph.add_edge("resynthesize_research", "critique_research")
    graph.add_edge("critique_research", "revise_final_answer")
    graph.add_edge("revise_final_answer", "check_policy")
    if human_review_interrupt:
        graph.add_edge("check_policy", "human_review_checkpoint")
        graph.add_conditional_edges(
            "human_review_checkpoint",
            route_after_human_review,
            {"repair": "plan_policy_repair", "verify": "verify_citations"},
        )
    else:
        graph.add_conditional_edges(
            "check_policy",
            route_after_policy,
            {"repair": "plan_policy_repair", "verify": "verify_citations"},
        )
    graph.add_edge("plan_policy_repair", "search_policy_repair_sources")
    graph.add_edge("search_policy_repair_sources", "read_policy_repair_sources")
    graph.add_edge("read_policy_repair_sources", "score_policy_repair_sources")
    graph.add_edge("score_policy_repair_sources", "resynthesize_research")
    graph.add_edge("verify_citations", END)
    return graph.compile(checkpointer=checkpointer)

