import hashlib
import html
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from app.fortigate_models import FortiGateStandardChunk


DEFAULT_STANDARDS_DIR = Path(os.getenv("FORTIGATE_STANDARDS_DIR", "/data/fortigate-standards/raw"))
DEFAULT_INDEX_PATH = Path(os.getenv("FORTIGATE_STANDARDS_INDEX", "/data/fortigate-standards/index.json"))
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


def _chunk_id(document: str, ordinal: int, text: str) -> str:
    digest = hashlib.sha256(f"{document}:{ordinal}:{text[:200]}".encode("utf-8")).hexdigest()[:16]
    return f"{Path(document).stem}-{ordinal}-{digest}"


def _topic_for_text(path: Path, text: str) -> str:
    haystack = f"{path.name} {text[:500]}".lower()
    topics = {
        "sdwan": ["sd-wan", "sdwan", "sla", "performance"],
        "firewall_policy": ["firewall policy", "policy", "utm", "security profile"],
        "interfaces": ["interface", "vlan", "zone", "switch"],
        "routing": ["route", "bgp", "ospf", "static route"],
        "nat": ["nat", "vip", "dnat", "snat"],
        "vpn": ["vpn", "ipsec", "ssl-vpn"],
        "ha": ["ha", "cluster", "failover"],
        "logging": ["log", "logging", "syslog", "fortianalyzer"],
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
                        "score": 0,
                    }
                )

    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps({"chunks": chunks}, indent=2))
    return {
        "source_dir": str(source),
        "index_path": str(index),
        "document_count": document_count,
        "chunk_count": len(chunks),
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


def search_standards(query: str, limit: int = 8, index_path: str | Path = DEFAULT_INDEX_PATH) -> list[FortiGateStandardChunk]:
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
            scored.append(chunk.model_copy(update={"score": score}))
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[: max(1, min(limit, 25))]


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
