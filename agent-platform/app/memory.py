import hashlib
import json
import os
from typing import Any
from urllib.parse import urlparse

try:
    from neo4j import AsyncGraphDatabase
except ImportError:  # pragma: no cover - dependency may be unavailable during partial setup
    AsyncGraphDatabase = None


class Neo4jResearchMemory:
    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password)) if AsyncGraphDatabase else None

    @classmethod
    def from_env(cls) -> "Neo4jResearchMemory | None":
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        if not uri or not password or AsyncGraphDatabase is None:
            return None
        return cls(uri, user, password)

    @property
    def enabled(self) -> bool:
        return self._driver is not None

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()

    async def ping(self) -> dict[str, Any]:
        if not self._driver:
            return {"enabled": False, "ok": False, "detail": "Neo4j driver or configuration is unavailable."}
        try:
            async with self._driver.session() as session:
                record = await session.execute_read(_ping_tx)
            return {"enabled": True, "ok": bool(record and record["ok"] == 1), "uri": self.uri, "user": self.user}
        except Exception as exc:
            return {"enabled": True, "ok": False, "uri": self.uri, "user": self.user, "detail": str(exc)}

    async def init_schema(self) -> None:
        if not self._driver:
            return
        statements = [
            "CREATE CONSTRAINT research_run_id IF NOT EXISTS FOR (r:ResearchRun) REQUIRE r.run_id IS UNIQUE",
            "CREATE CONSTRAINT question_id IF NOT EXISTS FOR (q:Question) REQUIRE q.question_hash IS UNIQUE",
            "CREATE CONSTRAINT source_id IF NOT EXISTS FOR (s:Source) REQUIRE s.source_hash IS UNIQUE",
            "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.claim_hash IS UNIQUE",
            "CREATE CONSTRAINT review_id IF NOT EXISTS FOR (h:HumanReview) REQUIRE h.review_id IS UNIQUE",
            "CREATE CONSTRAINT policy_id IF NOT EXISTS FOR (p:PolicyReport) REQUIRE p.policy_id IS UNIQUE",
            "CREATE CONSTRAINT quality_id IF NOT EXISTS FOR (q:QualityReport) REQUIRE q.quality_id IS UNIQUE",
            "CREATE INDEX research_run_created IF NOT EXISTS FOR (r:ResearchRun) ON (r.created_at)",
            "CREATE INDEX research_run_review IF NOT EXISTS FOR (r:ResearchRun) ON (r.review_decision)",
            "CREATE INDEX source_host IF NOT EXISTS FOR (s:Source) ON (s.host)",
        ]
        async with self._driver.session() as session:
            for statement in statements:
                await session.run(statement)

    async def save_research_run(self, response: Any) -> None:
        if not self._driver:
            return
        data = _model_dump(response)
        payload = _run_payload(data)
        async with self._driver.session() as session:
            await session.execute_write(_upsert_research_run, payload)

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        if not self._driver:
            return None
        async with self._driver.session() as session:
            return await session.execute_read(_get_run_summary, run_id)

    async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        if not self._driver:
            return []
        async with self._driver.session() as session:
            return await session.execute_read(_search_runs, query.lower(), limit)

    async def retrieve_context(self, query: str, limit: int = 5) -> dict[str, Any]:
        if not self._driver:
            return {"related_runs": [], "trusted_sources": [], "prior_claims": []}
        async with self._driver.session() as session:
            return await session.execute_read(_retrieve_context, query.lower(), limit)

    async def backlog(self, limit: int = 25) -> list[dict[str, Any]]:
        if not self._driver:
            return []
        async with self._driver.session() as session:
            return await session.execute_read(_backlog_runs, limit)

    async def audit(self, run_id: str) -> dict[str, Any] | None:
        if not self._driver:
            return None
        async with self._driver.session() as session:
            return await session.execute_read(_audit_run, run_id)

    async def dedup(self, query: str, limit: int = 10) -> dict[str, Any]:
        if not self._driver:
            return {"claims": [], "repeated_sources": []}
        async with self._driver.session() as session:
            return await session.execute_read(_dedup_memory, query.lower(), limit)


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    raise TypeError(f"Unsupported research response type: {type(value)!r}")


def _hash(value: str) -> str:
    return hashlib.sha256(value.strip().lower().encode("utf-8")).hexdigest()[:24]


def source_hash(url: str) -> str:
    return _hash(url or "unknown-source")


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _json(value: Any) -> str:
    return json.dumps(value or [], ensure_ascii=True, sort_keys=True)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _run_payload(data: dict[str, Any]) -> dict[str, Any]:
    review = data.get("human_review") or {}
    policy = data.get("policy_report") or {}
    quality = data.get("quality_report") or {}
    sources_by_url: dict[str, dict[str, Any]] = {}

    def add_source(item: dict[str, Any], phase: str, score: dict[str, Any] | None = None) -> None:
        url = (item or {}).get("url") or ""
        if not url:
            return
        existing = sources_by_url.setdefault(
            url,
            {
                "source_hash": source_hash(url),
                "url": url,
                "title": (item or {}).get("title") or url,
                "host": _host(url),
                "snippet": ((item or {}).get("snippet") or "")[:800],
                "status": (item or {}).get("status") or "",
                "phases": set(),
                "score": None,
                "score_rationale": "",
            },
        )
        existing["phases"].add(phase)
        if score:
            existing["score"] = int(score.get("score") or 0)
            existing["score_rationale"] = score.get("rationale") or ""

    score_by_url = {item.get("url"): item for item in _as_list(data.get("source_scores")) if item.get("url")}
    deep_score_by_url = {item.get("url"): item for item in _as_list(data.get("deepened_source_scores")) if item.get("url")}
    repair_score_by_url = {item.get("url"): item for item in _as_list(data.get("repair_source_scores")) if item.get("url")}
    for item in [*_as_list(data.get("sources")), *_as_list(data.get("evidence"))]:
        add_source(item, "initial", score_by_url.get(item.get("url")))
    for item in [*_as_list(data.get("deepened_sources")), *_as_list(data.get("deepened_evidence"))]:
        add_source(item, "deepening", deep_score_by_url.get(item.get("url")))
    for item in [*_as_list(data.get("repair_sources")), *_as_list(data.get("repair_evidence"))]:
        add_source(item, "repair", repair_score_by_url.get(item.get("url")))

    citations = []
    for citation in _as_list(data.get("citations")):
        claim = citation.get("claim") or ""
        url = citation.get("url") or ""
        if not claim:
            continue
        citations.append(
            {
                "claim_hash": _hash(f"{claim}|{url}"),
                "claim": claim,
                "url": url,
                "source_hash": source_hash(url) if url else "",
                "title": citation.get("title") or "",
                "evidence": citation.get("evidence") or "",
                "confidence": citation.get("confidence") or "medium",
            }
        )

    return {
        "run": {
            "run_id": data.get("run_id"),
            "created_at": data.get("created_at"),
            "thread_id": data.get("thread_id"),
            "mode": data.get("mode"),
            "question": data.get("question"),
            "answer": data.get("answer"),
            "confidence": data.get("confidence"),
            "model": data.get("model"),
            "source_count": len(sources_by_url),
            "citation_count": len(citations),
            "review_decision": review.get("decision") if review else "pending",
            "parent_run_id": data.get("parent_run_id"),
        },
        "question": {"question_hash": _hash(data.get("question") or ""), "text": data.get("question") or ""},
        "sources": [{**item, "phases": sorted(item["phases"])} for item in sources_by_url.values()],
        "citations": citations,
        "review": {
            "review_id": f"{data.get('run_id')}:review",
            "decision": review.get("decision", "pending") if review else "pending",
            "reviewer_notes": review.get("reviewer_notes", "") if review else "",
            "selected_issues": review.get("selected_issues", []) if review else [],
            "reviewed_at": review.get("reviewed_at", "") if review else "",
            "follow_up_run_id": review.get("follow_up_run_id") if review else None,
        },
        "policy": {
            "policy_id": f"{data.get('run_id')}:policy",
            "passed": bool(policy.get("passed", True)),
            "warnings": policy.get("warnings", []),
            "blocking_issues": policy.get("blocking_issues", []),
            "checks": policy.get("checks", []),
        },
        "quality": {
            "quality_id": f"{data.get('run_id')}:quality",
            "unsupported_claims": quality.get("unsupported_claims", []),
            "weak_citations": quality.get("weak_citations", []),
            "missing_perspectives": quality.get("missing_perspectives", []),
            "source_risks": quality.get("source_risks", []),
            "confidence_rationale": quality.get("confidence_rationale", ""),
        },
    }


async def _ping_tx(tx: Any) -> Any:
    result = await tx.run("RETURN 1 AS ok")
    return await result.single()


async def _upsert_research_run(tx: Any, payload: dict[str, Any]) -> None:
    await tx.run(
        """
        MERGE (r:ResearchRun {run_id: $run.run_id})
        SET r += $run,
            r.updated_at = datetime()
        MERGE (q:Question {question_hash: $question.question_hash})
        SET q.text = $question.text
        MERGE (r)-[:ANSWERED]->(q)
        """,
        **payload,
    )
    if payload["run"].get("parent_run_id"):
        await tx.run(
            """
            MATCH (child:ResearchRun {run_id: $run_id})
            MERGE (parent:ResearchRun {run_id: $parent_run_id})
            MERGE (parent)-[:FOLLOWED_BY]->(child)
            """,
            run_id=payload["run"]["run_id"],
            parent_run_id=payload["run"]["parent_run_id"],
        )
    if payload["review"].get("follow_up_run_id"):
        await tx.run(
            """
            MATCH (parent:ResearchRun {run_id: $run_id})
            MERGE (child:ResearchRun {run_id: $follow_up_run_id})
            MERGE (parent)-[:FOLLOWED_BY]->(child)
            """,
            run_id=payload["run"]["run_id"],
            follow_up_run_id=payload["review"]["follow_up_run_id"],
        )
    for source in payload["sources"]:
        await tx.run(
            """
            MATCH (r:ResearchRun {run_id: $run_id})
            MERGE (s:Source {source_hash: $source.source_hash})
            SET s.url = $source.url,
                s.title = $source.title,
                s.host = $source.host,
                s.snippet = $source.snippet,
                s.updated_at = datetime()
            MERGE (r)-[rel:USED_SOURCE]->(s)
            SET rel.phases = $source.phases,
                rel.status = $source.status,
                rel.score = $source.score,
                rel.score_rationale = $source.score_rationale
            """,
            run_id=payload["run"]["run_id"],
            source=source,
        )
    for citation in payload["citations"]:
        await tx.run(
            """
            MATCH (r:ResearchRun {run_id: $run_id})
            MERGE (c:Claim {claim_hash: $citation.claim_hash})
            SET c.text = $citation.claim,
                c.confidence = $citation.confidence,
                c.evidence = $citation.evidence,
                c.updated_at = datetime()
            MERGE (r)-[:MADE_CLAIM]->(c)
            WITH c
            MATCH (s:Source {source_hash: $citation.source_hash})
            MERGE (c)-[:CITED]->(s)
            """,
            run_id=payload["run"]["run_id"],
            citation=citation,
        )
    await tx.run(
        """
        MATCH (r:ResearchRun {run_id: $run_id})
        MERGE (h:HumanReview {review_id: $review.review_id})
        SET h.decision = $review.decision,
            h.reviewer_notes = $review.reviewer_notes,
            h.selected_issues_json = $selected_issues_json,
            h.reviewed_at = $review.reviewed_at,
            h.follow_up_run_id = $review.follow_up_run_id
        MERGE (r)-[:HAS_REVIEW]->(h)
        MERGE (p:PolicyReport {policy_id: $policy.policy_id})
        SET p.passed = $policy.passed,
            p.warnings_json = $warnings_json,
            p.blocking_issues_json = $blocking_issues_json,
            p.checks_json = $checks_json,
            p.warning_count = size($policy.warnings),
            p.blocking_issue_count = size($policy.blocking_issues)
        MERGE (r)-[:HAS_POLICY]->(p)
        MERGE (q:QualityReport {quality_id: $quality.quality_id})
        SET q.unsupported_claims_json = $unsupported_claims_json,
            q.weak_citations_json = $weak_citations_json,
            q.missing_perspectives_json = $missing_perspectives_json,
            q.source_risks_json = $source_risks_json,
            q.confidence_rationale = $quality.confidence_rationale,
            q.unsupported_claim_count = size($quality.unsupported_claims),
            q.weak_citation_count = size($quality.weak_citations),
            q.missing_perspective_count = size($quality.missing_perspectives)
        MERGE (r)-[:HAS_QUALITY]->(q)
        """,
        run_id=payload["run"]["run_id"],
        review=payload["review"],
        policy=payload["policy"],
        quality=payload["quality"],
        selected_issues_json=_json(payload["review"].get("selected_issues")),
        warnings_json=_json(payload["policy"].get("warnings")),
        blocking_issues_json=_json(payload["policy"].get("blocking_issues")),
        checks_json=_json(payload["policy"].get("checks")),
        unsupported_claims_json=_json(payload["quality"].get("unsupported_claims")),
        weak_citations_json=_json(payload["quality"].get("weak_citations")),
        missing_perspectives_json=_json(payload["quality"].get("missing_perspectives")),
        source_risks_json=_json(payload["quality"].get("source_risks")),
    )


async def _get_run_summary(tx: Any, run_id: str) -> dict[str, Any] | None:
    result = await tx.run(
        """
        MATCH (r:ResearchRun {run_id: $run_id})
        OPTIONAL MATCH (r)-[:ANSWERED]->(q:Question)
        OPTIONAL MATCH (r)-[:USED_SOURCE]->(s:Source)
        OPTIONAL MATCH (r)-[:MADE_CLAIM]->(c:Claim)
        OPTIONAL MATCH (r)-[:HAS_REVIEW]->(h:HumanReview)
        OPTIONAL MATCH (r)-[:FOLLOWED_BY]->(child:ResearchRun)
        OPTIONAL MATCH (parent:ResearchRun)-[:FOLLOWED_BY]->(r)
        RETURN r, q.text AS question,
               collect(DISTINCT s { .source_hash, .url, .title, .host }) AS sources,
               collect(DISTINCT c { .claim_hash, .text, .confidence }) AS claims,
               h { .decision, .reviewer_notes, .reviewed_at, .follow_up_run_id } AS review,
               collect(DISTINCT child.run_id) AS follow_up_run_ids,
               collect(DISTINCT parent.run_id) AS parent_run_ids
        """,
        run_id=run_id,
    )
    record = await result.single()
    if not record:
        return None
    run = dict(record["r"])
    return {
        "run": run,
        "question": record["question"],
        "sources": record["sources"],
        "claims": record["claims"],
        "review": record["review"],
        "follow_up_run_ids": record["follow_up_run_ids"],
        "parent_run_ids": record["parent_run_ids"],
    }


async def _search_runs(tx: Any, search_text: str, limit: int) -> list[dict[str, Any]]:
    result = await tx.run(
        """
        MATCH (r:ResearchRun)
        WHERE toLower(r.question) CONTAINS $search_text
           OR toLower(r.answer) CONTAINS $search_text
           OR toLower(coalesce(r.review_decision, '')) CONTAINS $search_text
        OPTIONAL MATCH (r)-[:USED_SOURCE]->(s:Source)
        WITH r, collect(DISTINCT s.host)[0..5] AS hosts
        RETURN r.run_id AS run_id,
               r.created_at AS created_at,
               r.question AS question,
               r.confidence AS confidence,
               r.review_decision AS review_decision,
               r.source_count AS source_count,
               r.citation_count AS citation_count,
               hosts
        ORDER BY r.created_at DESC
        LIMIT $limit
        """,
        search_text=search_text,
        limit=limit,
    )
    return [dict(record) async for record in result]


async def _source_runs(tx: Any, source_hash_value: str, limit: int) -> dict[str, Any] | None:
    result = await tx.run(
        """
        MATCH (s:Source {source_hash: $source_hash})<-[:USED_SOURCE]-(r:ResearchRun)
        RETURN s { .source_hash, .url, .title, .host } AS source,
               collect(r { .run_id, .created_at, .question, .confidence, .review_decision })[0..$limit] AS runs
        """,
        source_hash=source_hash_value,
        limit=limit,
    )
    record = await result.single()
    if not record:
        return None
    return {"source": record["source"], "runs": record["runs"]}


async def _retrieve_context(tx: Any, search_text: str, limit: int) -> dict[str, Any]:
    related_result = await tx.run(
        """
        MATCH (r:ResearchRun)
        WHERE toLower(r.question) CONTAINS $search_text
           OR toLower(r.answer) CONTAINS $search_text
        OPTIONAL MATCH (r)-[:USED_SOURCE]->(s:Source)
        OPTIONAL MATCH (r)-[:MADE_CLAIM]->(c:Claim)
        RETURN r.run_id AS run_id,
               r.question AS question,
               r.confidence AS confidence,
               r.review_decision AS review_decision,
               r.created_at AS created_at,
               collect(DISTINCT s { .source_hash, .url, .title, .host })[0..5] AS sources,
               collect(DISTINCT c { .claim_hash, .text, .confidence })[0..5] AS claims
        ORDER BY r.created_at DESC
        LIMIT $limit
        """,
        search_text=search_text,
        limit=limit,
    )
    related_runs = [dict(record) async for record in related_result]

    claim_result = await tx.run(
        """
        MATCH (c:Claim)-[:CITED]->(s:Source)
        WHERE toLower(c.text) CONTAINS $search_text
        RETURN c.claim_hash AS claim_hash,
               c.text AS claim,
               c.confidence AS confidence,
               collect(DISTINCT s { .source_hash, .url, .title, .host })[0..3] AS sources
        LIMIT $limit
        """,
        search_text=search_text,
        limit=limit,
    )
    prior_claims = [dict(record) async for record in claim_result]

    source_result = await tx.run(
        """
        MATCH (r:ResearchRun)-[rel:USED_SOURCE]->(s:Source)
        WITH s, count(DISTINCT r) AS run_count, avg(coalesce(rel.score, 0)) AS avg_score
        WHERE run_count > 1 OR avg_score >= 70
        RETURN s.source_hash AS source_hash,
               s.url AS url,
               s.title AS title,
               s.host AS host,
               run_count,
               round(avg_score) AS avg_score
        ORDER BY run_count DESC, avg_score DESC
        LIMIT $limit
        """,
        limit=limit,
    )
    trusted_sources = [dict(record) async for record in source_result]
    return {"related_runs": related_runs, "prior_claims": prior_claims, "trusted_sources": trusted_sources}


async def _backlog_runs(tx: Any, limit: int) -> list[dict[str, Any]]:
    result = await tx.run(
        """
        MATCH (r:ResearchRun)
        OPTIONAL MATCH (r)-[:HAS_POLICY]->(p:PolicyReport)
        OPTIONAL MATCH (r)-[:HAS_QUALITY]->(q:QualityReport)
        OPTIONAL MATCH (r)-[:HAS_REVIEW]->(h:HumanReview)
        WHERE coalesce(r.review_decision, 'pending') = 'needs_work'
           OR coalesce(p.blocking_issue_count, 0) > 0
           OR coalesce(p.warning_count, 0) > 0
           OR coalesce(q.unsupported_claim_count, 0) > 0
           OR coalesce(q.weak_citation_count, 0) > 0
           OR coalesce(q.missing_perspective_count, 0) > 0
        RETURN r.run_id AS run_id,
               r.created_at AS created_at,
               r.question AS question,
               r.confidence AS confidence,
               coalesce(r.review_decision, 'pending') AS review_decision,
               coalesce(p.blocking_issue_count, 0) AS blocking_issues,
               coalesce(p.warning_count, 0) AS warnings,
               coalesce(q.unsupported_claim_count, 0) AS unsupported_claims,
               coalesce(q.weak_citation_count, 0) AS weak_citations,
               coalesce(q.missing_perspective_count, 0) AS missing_perspectives,
               h.follow_up_run_id AS follow_up_run_id
        ORDER BY r.created_at DESC
        LIMIT $limit
        """,
        limit=limit,
    )
    return [dict(record) async for record in result]


async def _audit_run(tx: Any, run_id: str) -> dict[str, Any] | None:
    result = await tx.run(
        """
        MATCH (r:ResearchRun {run_id: $run_id})
        OPTIONAL MATCH (r)-[:HAS_REVIEW]->(h:HumanReview)
        OPTIONAL MATCH (r)-[:HAS_POLICY]->(p:PolicyReport)
        OPTIONAL MATCH (r)-[:HAS_QUALITY]->(q:QualityReport)
        OPTIONAL MATCH (r)-[:FOLLOWED_BY]->(child:ResearchRun)
        OPTIONAL MATCH (parent:ResearchRun)-[:FOLLOWED_BY]->(r)
        RETURN r { .run_id, .question, .created_at, .confidence, .review_decision, .parent_run_id } AS run,
               h { .decision, .reviewer_notes, .reviewed_at, .follow_up_run_id, .selected_issues_json } AS review,
               p { .passed, .warning_count, .blocking_issue_count, .warnings_json, .blocking_issues_json } AS policy,
               q { .unsupported_claim_count, .weak_citation_count, .missing_perspective_count, .unsupported_claims_json, .weak_citations_json, .missing_perspectives_json } AS quality,
               collect(DISTINCT child { .run_id, .question, .created_at, .review_decision }) AS follow_ups,
               collect(DISTINCT parent { .run_id, .question, .created_at, .review_decision }) AS parents
        """,
        run_id=run_id,
    )
    record = await result.single()
    if not record:
        return None
    return {key: record[key] for key in ["run", "review", "policy", "quality", "follow_ups", "parents"]}


async def _dedup_memory(tx: Any, search_text: str, limit: int) -> dict[str, Any]:
    claim_result = await tx.run(
        """
        MATCH (r:ResearchRun)-[:MADE_CLAIM]->(c:Claim)-[:CITED]->(s:Source)
        WHERE $search_text = '' OR toLower(c.text) CONTAINS $search_text
        WITH c, collect(DISTINCT r { .run_id, .question, .created_at }) AS runs, collect(DISTINCT s { .source_hash, .url, .title, .host }) AS sources
        WHERE size(runs) > 1 OR size(sources) > 1
        RETURN c.claim_hash AS claim_hash,
               c.text AS claim,
               c.confidence AS confidence,
               runs[0..5] AS runs,
               sources[0..5] AS sources
        ORDER BY size(runs) DESC, size(sources) DESC
        LIMIT $limit
        """,
        search_text=search_text,
        limit=limit,
    )
    claims = [dict(record) async for record in claim_result]

    source_result = await tx.run(
        """
        MATCH (r:ResearchRun)-[:USED_SOURCE]->(s:Source)
        WITH s, collect(DISTINCT r { .run_id, .question, .created_at }) AS runs
        WHERE size(runs) > 1
        RETURN s.source_hash AS source_hash,
               s.url AS url,
               s.title AS title,
               s.host AS host,
               runs[0..8] AS runs,
               size(runs) AS run_count
        ORDER BY run_count DESC
        LIMIT $limit
        """,
        limit=limit,
    )
    repeated_sources = [dict(record) async for record in source_result]
    return {"claims": claims, "repeated_sources": repeated_sources}
