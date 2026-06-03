import hashlib
import html
import json
import os
import re
import struct
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from app.fortigate_models import FortiGateStandardChunk


DEFAULT_STANDARDS_DIR = Path(os.getenv("FORTIGATE_STANDARDS_DIR", "/data/fortigate-standards/raw"))
DEFAULT_INDEX_PATH = Path(os.getenv("FORTIGATE_STANDARDS_INDEX", "/data/fortigate-standards/index.json"))
STANDARDS_RETRIEVAL_BACKEND = os.getenv("STANDARDS_RETRIEVAL_BACKEND", "redis_hybrid").strip().lower()
STANDARDS_REDIS_URL = os.getenv("STANDARDS_REDIS_URL", os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"))
STANDARDS_REDIS_INDEX = os.getenv("STANDARDS_REDIS_INDEX", "idx:standards")
STANDARDS_REDIS_PREFIX = os.getenv("STANDARDS_REDIS_PREFIX", "std:chunk:")
STANDARDS_EMBEDDING_MODEL = os.getenv("STANDARDS_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
STANDARDS_VECTOR_DIM = int(os.getenv("STANDARDS_VECTOR_DIM", "384"))
STANDARDS_RRF_K = int(os.getenv("STANDARDS_RRF_K", "60"))
STANDARDS_RERANK_ENABLED = os.getenv("STANDARDS_RERANK_ENABLED", "false").lower() == "true"
STANDARDS_RERANK_MODEL = os.getenv("STANDARDS_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
SUPPORTED_SUFFIXES = {
    ".md",
    ".markdown",
    ".txt",
    ".conf",
    ".cfg",
    ".yaml",
    ".yml",
    ".json",
    ".html",
    ".htm",
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
}
MAX_EXTRACTED_CHARS_PER_FILE = int(os.getenv("FORTIGATE_STANDARDS_MAX_FILE_CHARS", "500000"))


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _redis_client():
    try:
        import redis
    except ImportError:
        return None
    try:
        client = redis.Redis.from_url(STANDARDS_REDIS_URL, decode_responses=False)
        client.ping()
        return client
    except Exception:
        return None


@lru_cache(maxsize=1)
def _embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None
    try:
        return SentenceTransformer(STANDARDS_EMBEDDING_MODEL)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _rerank_model():
    if not STANDARDS_RERANK_ENABLED:
        return None
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        return None
    try:
        return CrossEncoder(STANDARDS_RERANK_MODEL)
    except Exception:
        return None


def _normalize_vector(values: list[float]) -> list[float]:
    if not values:
        return [0.0] * STANDARDS_VECTOR_DIM
    if len(values) < STANDARDS_VECTOR_DIM:
        values = [*values, *([0.0] * (STANDARDS_VECTOR_DIM - len(values)))]
    elif len(values) > STANDARDS_VECTOR_DIM:
        values = values[:STANDARDS_VECTOR_DIM]
    norm = sum(item * item for item in values) ** 0.5
    if norm <= 0:
        return values
    return [item / norm for item in values]


def _hash_embedding(text: str) -> list[float]:
    values = [0.0] * STANDARDS_VECTOR_DIM
    tokens = re.findall(r"[a-zA-Z0-9_-]{3,}", text.lower())
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % STANDARDS_VECTOR_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        values[bucket] += sign
    return _normalize_vector(values)


def _embed_text(text: str) -> list[float]:
    model = _embedding_model()
    if model is None:
        return _hash_embedding(text)
    try:
        vector = model.encode(text[:8000], normalize_embeddings=True)
    except Exception:
        return _hash_embedding(text)
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return _normalize_vector([float(item) for item in vector])


def _pack_vector(values: list[float]) -> bytes:
    normalized = _normalize_vector(values)
    return struct.pack(f"{len(normalized)}f", *normalized)


def _chunk_id(document: str, ordinal: int, text: str) -> str:
    digest = hashlib.sha256(f"{document}:{ordinal}:{text[:200]}".encode("utf-8")).hexdigest()[:16]
    return f"{Path(document).stem}-{ordinal}-{digest}"


def _topic_for_text(path: Path, text: str) -> str:
    haystack = f"{path.name} {text[:500]}".lower()
    topics = {
        "security_profiles": ["utm", "security profile", "antivirus", "av profile", "ips", "intrusion prevention", "web filter", "application control", "dns filter", "ssl inspection", "deep inspection", "file filter", "fortiguard"],
        "authentication": ["fsso", "ldap", "radius", "saml", "two-factor", "2fa", "mfa", "user group", "authentication", "trusted host", "admin access"],
        "wifi": ["wifi", "wi-fi", "wireless", "ssid", "fortiap", "wlan", "captive portal"],
        "fortiswitch": ["fortiswitch", "fortilink", "switch-controller", "managed switch"],
        "ztna": ["ztna", "zero trust", "access proxy", "ztna proxy"],
        "certificates": ["certificate", "pki", "ca cert", "ssl certificate", "scep"],
        "qos": ["qos", "traffic shaping", "traffic shaper", "shaper", "bandwidth"],
        "sdwan": ["sd-wan", "sdwan", "sla", "performance"],
        "vpn": ["vpn", "ipsec", "ssl-vpn"],
        "nat": ["nat", "vip", "dnat", "snat"],
        "routing": ["route", "bgp", "ospf", "static route"],
        "dns_dhcp": ["dhcp", "dns server", "ddns", "dns filter"],
        "system_hardening": ["hardening", "firmware", "config backup", "ntp", "management interface", "secure access", "admin password"],
        "logging": ["log", "logging", "syslog", "fortianalyzer"],
        "ha": ["ha", "cluster", "failover"],
        "firewall_policy": ["firewall policy", "policy", "utm", "security profile"],
        "interfaces": ["interface", "vlan", "zone", "switch"],
    }
    for topic, needles in topics.items():
        if any(needle in haystack for needle in needles):
            return topic
    return "general"


def _split_chunks(text: str, max_chars: int = 1400) -> list[str]:
    cleaned = re.sub(r"\r\n?", "\n", text)
    sections = re.split(r"\n(?=#{1,4}\s+)", cleaned)
    chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        while len(section) > max_chars:
            split_at = section.rfind("\n", 0, max_chars)
            if split_at < 400:
                split_at = max_chars
            chunks.append(section[:split_at].strip())
            section = section[split_at:].strip()
        if section:
            chunks.append(section)
    return chunks


def _is_noise_file(path: Path) -> bool:
    parts = set(path.parts)
    return (
        "__MACOSX" in parts
        or path.name == ".DS_Store"
        or path.name.startswith("._")
        or path.suffix.lower() == ".ds_store"
    )


def _normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:MAX_EXTRACTED_CHARS_PER_FILE]


def _xml_texts(raw: bytes) -> list[str]:
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError:
        return []
    texts: list[str] = []
    for elem in root.iter():
        if elem.text and elem.text.strip():
            texts.append(elem.text.strip())
    return texts


def _read_office_zip(path: Path, prefixes: tuple[str, ...]) -> str:
    texts: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".xml"):
                continue
            if not any(name.startswith(prefix) for prefix in prefixes):
                continue
            texts.extend(_xml_texts(archive.read(name)))
            if sum(len(item) for item in texts) > MAX_EXTRACTED_CHARS_PER_FILE:
                break
    return _normalize_text("\n".join(texts))


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    texts: list[str] = []
    reader = PdfReader(str(path))
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            texts.append(text)
        if sum(len(item) for item in texts) > MAX_EXTRACTED_CHARS_PER_FILE:
            break
    return _normalize_text("\n".join(texts))


def _read_html(path: Path) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return _normalize_text(path.read_text(encoding="utf-8", errors="replace"))
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return _normalize_text(soup.get_text("\n", strip=True))


def _read_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt", ".conf", ".cfg", ".yaml", ".yml", ".json"}:
        return _normalize_text(path.read_text(encoding="utf-8", errors="replace"))
    if suffix in {".html", ".htm"}:
        return _read_html(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".docx":
        return _read_office_zip(path, ("word/document", "word/header", "word/footer"))
    if suffix == ".pptx":
        return _read_office_zip(path, ("ppt/slides/slide", "ppt/notesSlides/notesSlide"))
    if suffix == ".xlsx":
        return _read_office_zip(path, ("xl/sharedStrings", "xl/worksheets/sheet"))
    return ""


def _create_redis_index(client: Any) -> None:
    try:
        from redis.commands.search.field import TagField, TextField, VectorField
        try:
            from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        except Exception:
            from redis.commands.search.index_definition import IndexDefinition, IndexType
    except Exception as exc:
        raise RuntimeError("redis-py RediSearch helpers are unavailable.") from exc
    try:
        client.ft(STANDARDS_REDIS_INDEX).dropindex(delete_documents=False)
    except Exception:
        pass
    schema = (
        TextField("chunk_id"),
        TextField("document"),
        TagField("topic"),
        TextField("source_path"),
        TextField("text"),
        TextField("hash"),
        VectorField(
            "embedding",
            "HNSW",
            {
                "TYPE": "FLOAT32",
                "DIM": STANDARDS_VECTOR_DIM,
                "DISTANCE_METRIC": "COSINE",
            },
        ),
    )
    client.ft(STANDARDS_REDIS_INDEX).create_index(
        schema,
        definition=IndexDefinition(prefix=[STANDARDS_REDIS_PREFIX], index_type=IndexType.HASH),
    )


def _clear_redis_standard_chunks(client: Any) -> None:
    batch: list[bytes] = []
    for key in client.scan_iter(f"{STANDARDS_REDIS_PREFIX}*"):
        batch.append(key)
        if len(batch) >= 500:
            client.delete(*batch)
            batch = []
    if batch:
        client.delete(*batch)


def _index_standards_in_redis(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    if STANDARDS_RETRIEVAL_BACKEND not in {"redis_hybrid", "redis", "hybrid"}:
        return {"enabled": False, "indexed_count": 0, "error": "Redis hybrid retrieval disabled."}
    client = _redis_client()
    if client is None:
        return {"enabled": False, "indexed_count": 0, "error": "Redis is unavailable."}
    try:
        _clear_redis_standard_chunks(client)
        _create_redis_index(client)
        pipe = client.pipeline(transaction=False)
        for chunk in chunks:
            key = f"{STANDARDS_REDIS_PREFIX}{chunk['chunk_id']}"
            embedding = _pack_vector(_embed_text(f"{chunk.get('document', '')}\n{chunk.get('topic', '')}\n{chunk.get('text', '')}"))
            pipe.hset(
                key,
                mapping={
                    "chunk_id": chunk.get("chunk_id", ""),
                    "document": chunk.get("document", ""),
                    "topic": chunk.get("topic", "general"),
                    "source_path": chunk.get("source_path", ""),
                    "text": chunk.get("text", ""),
                    "hash": chunk.get("hash", ""),
                    "embedding": embedding,
                },
            )
        if chunks:
            pipe.execute()
        return {"enabled": True, "indexed_count": len(chunks), "index_name": STANDARDS_REDIS_INDEX}
    except Exception as exc:
        return {"enabled": False, "indexed_count": 0, "error": str(exc)}


def ingest_standards(source_dir: str | Path = DEFAULT_STANDARDS_DIR, index_path: str | Path = DEFAULT_INDEX_PATH) -> dict[str, Any]:
    source = Path(source_dir)
    index = Path(index_path)
    chunks: list[dict[str, Any]] = []
    document_count = 0

    if source.exists():
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            if _is_noise_file(path):
                continue
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            try:
                text = _read_document_text(path)
            except Exception:
                continue
            if not text.strip():
                continue
            document_count += 1
            for ordinal, chunk_text in enumerate(_split_chunks(text), start=1):
                chunks.append(
                    {
                        "chunk_id": _chunk_id(path.name, ordinal, chunk_text),
                        "document": path.name,
                        "topic": _topic_for_text(path, chunk_text),
                        "text": chunk_text,
                        "source_path": str(path),
                        "hash": _chunk_hash(chunk_text),
                        "score": 0,
                    }
                )

    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps({"chunks": chunks}, indent=2))
    redis_index = _index_standards_in_redis(chunks)
    return {
        "source_dir": str(source),
        "index_path": str(index),
        "document_count": document_count,
        "chunk_count": len(chunks),
        "redis_index": redis_index,
    }


def load_standard_chunks(index_path: str | Path = DEFAULT_INDEX_PATH) -> list[FortiGateStandardChunk]:
    index = Path(index_path)
    if not index.exists():
        return []
    try:
        data = json.loads(index.read_text())
    except Exception:
        return []
    return [FortiGateStandardChunk(**item) for item in data.get("chunks", []) if isinstance(item, dict)]


def _keyword_search_json(query: str, limit: int = 8, index_path: str | Path = DEFAULT_INDEX_PATH) -> list[FortiGateStandardChunk]:
    terms = [term.lower() for term in re.findall(r"[a-zA-Z0-9_-]{3,}", query)]
    phrase = " ".join(terms)
    chunks = load_standard_chunks(index_path)
    scored: list[FortiGateStandardChunk] = []
    for chunk in chunks:
        title = f"{chunk.document} {chunk.source_path}".lower()
        text = chunk.text.lower()
        haystack = f"{title} {chunk.topic} {text}"
        score = 0
        if phrase and phrase in title:
            score += 40
        elif phrase and phrase in text:
            score += 25
        matched_terms = 0
        for term in terms:
            term_score = 0
            if term in title:
                term_score += 12
            if term in text:
                term_score += 4
                term_score += min(text.count(term), 5)
            if term_score:
                matched_terms += 1
                score += term_score
        if terms:
            score += matched_terms * 4
            if matched_terms == len(terms):
                score += 20
        if score:
            scored.append(chunk.model_copy(update={"score": score, "retrieval_backend": "json_keyword"}))
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[: max(1, min(limit, 25))]


def _redis_chunk_from_doc(doc: Any, updates: dict[str, Any] | None = None) -> FortiGateStandardChunk:
    payload = {
        "chunk_id": _as_text(getattr(doc, "chunk_id", "")),
        "document": _as_text(getattr(doc, "document", "")),
        "topic": _as_text(getattr(doc, "topic", "general")) or "general",
        "text": _as_text(getattr(doc, "text", "")),
        "source_path": _as_text(getattr(doc, "source_path", "")),
        "score": 0,
    }
    payload.update(updates or {})
    return FortiGateStandardChunk(**payload)


def _escape_redis_term(term: str) -> str:
    return re.sub(r"([@{}\\[\\]\"'():|&!\\-~*?\\\\/])", r"\\\1", term)


def _redis_fulltext_search(client: Any, query: str, candidate_count: int) -> list[FortiGateStandardChunk]:
    try:
        from redis.commands.search.query import Query
    except Exception:
        return []
    terms = [term.lower() for term in re.findall(r"[a-zA-Z0-9_-]{3,}", query)][:16]
    if not terms:
        terms = ["fortigate", "standards"]
    query_text = " | ".join(_escape_redis_term(term) for term in terms)
    try:
        redis_query = (
            Query(query_text)
            .return_fields("chunk_id", "document", "topic", "source_path", "text")
            .paging(0, candidate_count)
            .dialect(2)
        )
        result = client.ft(STANDARDS_REDIS_INDEX).search(redis_query)
    except Exception:
        return []
    return [
        _redis_chunk_from_doc(doc, {"keyword_rank": rank, "retrieval_backend": "redis_keyword"})
        for rank, doc in enumerate(getattr(result, "docs", []), start=1)
        if _as_text(getattr(doc, "chunk_id", ""))
    ]


def _redis_vector_search(client: Any, query: str, candidate_count: int) -> list[FortiGateStandardChunk]:
    try:
        from redis.commands.search.query import Query
    except Exception:
        return []
    try:
        vector = _pack_vector(_embed_text(query or "fortigate firewall network standards"))
        redis_query = (
            Query(f"*=>[KNN {candidate_count} @embedding $vector AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("chunk_id", "document", "topic", "source_path", "text", "vector_distance")
            .paging(0, candidate_count)
            .dialect(2)
        )
        result = client.ft(STANDARDS_REDIS_INDEX).search(redis_query, query_params={"vector": vector})
    except Exception:
        return []
    return [
        _redis_chunk_from_doc(doc, {"vector_rank": rank, "retrieval_backend": "redis_vector"})
        for rank, doc in enumerate(getattr(result, "docs", []), start=1)
        if _as_text(getattr(doc, "chunk_id", ""))
    ]


def _rrf_fuse(
    keyword_results: list[FortiGateStandardChunk],
    vector_results: list[FortiGateStandardChunk],
    limit: int,
) -> list[FortiGateStandardChunk]:
    by_id: dict[str, FortiGateStandardChunk] = {}
    ranks: dict[str, dict[str, int]] = {}
    for rank, chunk in enumerate(keyword_results, start=1):
        by_id.setdefault(chunk.chunk_id, chunk)
        ranks.setdefault(chunk.chunk_id, {})["keyword_rank"] = rank
    for rank, chunk in enumerate(vector_results, start=1):
        by_id.setdefault(chunk.chunk_id, chunk)
        ranks.setdefault(chunk.chunk_id, {})["vector_rank"] = rank

    fused: list[FortiGateStandardChunk] = []
    for chunk_id, chunk in by_id.items():
        item_ranks = ranks.get(chunk_id, {})
        rrf_score = sum(1.0 / (STANDARDS_RRF_K + rank) for rank in item_ranks.values())
        fused.append(
            chunk.model_copy(
                update={
                    "score": int(rrf_score * 100000),
                    "retrieval_backend": "redis_hybrid",
                    "keyword_rank": item_ranks.get("keyword_rank"),
                    "vector_rank": item_ranks.get("vector_rank"),
                    "rrf_score": round(rrf_score, 6),
                }
            )
        )
    fused.sort(key=lambda item: (item.rrf_score or 0, item.score), reverse=True)
    return fused[: max(1, min(limit, 25))]


def _rerank_standards(query: str, chunks: list[FortiGateStandardChunk], limit: int) -> list[FortiGateStandardChunk]:
    model = _rerank_model()
    if model is None:
        return chunks[:limit]
    try:
        pairs = [(query, chunk.text[:4000]) for chunk in chunks]
        scores = model.predict(pairs)
    except Exception:
        return chunks[:limit]
    reranked: list[FortiGateStandardChunk] = []
    for rank, (chunk, raw_score) in enumerate(
        sorted(zip(chunks, scores, strict=False), key=lambda item: float(item[1]), reverse=True),
        start=1,
    ):
        reranked.append(
            chunk.model_copy(
                update={
                    "score": int(float(raw_score) * 1000),
                    "retrieval_backend": "redis_hybrid_reranked",
                    "rrf_score": chunk.rrf_score,
                }
            )
        )
        if rank >= limit:
            break
    return reranked


def hybrid_search_standards(query: str, limit: int = 8, index_path: str | Path = DEFAULT_INDEX_PATH) -> list[FortiGateStandardChunk]:
    if STANDARDS_RETRIEVAL_BACKEND not in {"redis_hybrid", "redis", "hybrid"}:
        return _keyword_search_json(query, limit=limit, index_path=index_path)
    client = _redis_client()
    if client is None:
        return _keyword_search_json(query, limit=limit, index_path=index_path)
    candidate_count = max(limit * 4, 30)
    keyword_results = _redis_fulltext_search(client, query, candidate_count)
    vector_results = _redis_vector_search(client, query, candidate_count)
    fused = _rrf_fuse(keyword_results, vector_results, limit=max(limit, 8))
    if not fused:
        return _keyword_search_json(query, limit=limit, index_path=index_path)
    if STANDARDS_RERANK_ENABLED:
        fused = _rerank_standards(query, fused, limit=max(limit, 8))
    return fused[: max(1, min(limit, 25))]


def search_standards(query: str, limit: int = 8, index_path: str | Path = DEFAULT_INDEX_PATH) -> list[FortiGateStandardChunk]:
    return hybrid_search_standards(query, limit=limit, index_path=index_path)


def _requirement_keywords(text: str, topic: str) -> list[str]:
    words = [word.lower() for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text)]
    stop = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "should",
        "must",
        "will",
        "need",
        "needs",
        "using",
        "use",
        "can",
        "not",
        "all",
        "any",
    }
    topic_terms = {
        "sdwan": ["sd-wan", "wan", "sla", "failover", "steering", "performance"],
        "firewall_policy": ["firewall", "policy", "zone", "security", "internet", "guest"],
        "interfaces": ["interface", "vlan", "subnet", "gateway", "dhcp"],
        "routing": ["route", "routing", "bgp", "ospf", "static"],
        "nat": ["nat", "vip", "snat", "dnat"],
        "vpn": ["vpn", "ipsec", "tunnel"],
        "ha": ["ha", "backup", "redundant", "failover"],
        "logging": ["log", "logging", "syslog", "snmp", "fortianalyzer", "monitoring"],
        "security_profiles": ["utm", "ips", "antivirus", "web-filter", "application-control", "ssl-inspection", "dns-filter"],
        "authentication": ["fsso", "ldap", "radius", "mfa", "user", "admin"],
        "wifi": ["wifi", "ssid", "fortiap", "wlan"],
        "fortiswitch": ["fortiswitch", "fortilink", "switch"],
        "ztna": ["ztna", "zero-trust", "proxy"],
        "certificates": ["certificate", "pki", "ssl"],
        "qos": ["qos", "shaping", "bandwidth"],
        "dns_dhcp": ["dhcp", "dns", "ddns"],
        "system_hardening": ["hardening", "firmware", "backup", "ntp", "management"],
    }
    selected = [word for word in words if word not in stop and len(word) > 3]
    merged = [*topic_terms.get(topic, []), *selected]
    return list(dict.fromkeys(merged))[:12]


def _requirement_priority(text: str) -> str:
    lower = text.lower()
    if any(term in lower for term in ("must", "always", "required", "do not", "never", "ensure", "critical")):
        return "high"
    if any(term in lower for term in ("should", "recommended", "best practice", "prefer")):
        return "medium"
    return "low"


def extract_standard_requirements(chunks: list[FortiGateStandardChunk | dict[str, Any]], max_requirements: int = 24) -> list[dict[str, Any]]:
    requirements: list[dict[str, Any]] = []
    seen: set[str] = set()
    normative = re.compile(
        r"\b(must|should|required|requires|requirement|ensure|always|never|do not|be sure|need to|needs to|recommended|best practice)\b",
        re.IGNORECASE,
    )
    for raw in chunks:
        chunk = raw if isinstance(raw, dict) else raw.model_dump()
        text = str(chunk.get("text") or "")
        document = str(chunk.get("document") or "unknown")
        chunk_id = str(chunk.get("chunk_id") or "")
        topic = str(chunk.get("topic") or "general")
        sentences = [part.strip(" -•\t\n") for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
        candidates = [sentence for sentence in sentences if normative.search(sentence)]
        if not candidates and topic in {"logging", "sdwan", "firewall_policy", "routing", "interfaces", "security_profiles", "authentication", "wifi", "fortiswitch", "ztna", "certificates", "qos", "dns_dhcp", "system_hardening"}:
            candidates = sentences[:2]
        for sentence in candidates[:3]:
            cleaned = re.sub(r"\s+", " ", sentence).strip()
            if len(cleaned) < 35 or len(cleaned) > 420:
                continue
            fingerprint = re.sub(r"[^a-z0-9]+", " ", cleaned.lower())[:180]
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            requirement_id = f"STD-{len(requirements) + 1:03d}"
            requirements.append(
                {
                    "requirement_id": requirement_id,
                    "topic": topic,
                    "requirement": cleaned,
                    "source_document": document,
                    "source_chunk_id": chunk_id,
                    "source_excerpt": text[:1000],
                    "keywords": _requirement_keywords(cleaned, topic),
                    "priority": _requirement_priority(cleaned),
                }
            )
            if len(requirements) >= max_requirements:
                return requirements
    return requirements


def retrieve_fortigate_standards(payload: dict[str, Any], limit: int = 10) -> list[FortiGateStandardChunk]:
    query_parts = [
        str(payload.get("business_intent") or ""),
        str(payload.get("request_type") or ""),
        str(payload.get("firewall_policy_intent") or ""),
        str(payload.get("nat_requirements") or ""),
        str(payload.get("vpn_requirements") or ""),
        str(payload.get("logging_requirements") or ""),
        str(payload.get("ha_requirements") or ""),
        " ".join(payload.get("security_zones") or []),
        " ".join(str(item) for item in payload.get("wan_circuits") or []),
        " ".join(str(item) for item in payload.get("lan_networks") or []),
    ]
    query = " ".join(part for part in query_parts if part.strip())
    return search_standards(query or "fortigate firewall sd-wan interface policy", limit=limit)
