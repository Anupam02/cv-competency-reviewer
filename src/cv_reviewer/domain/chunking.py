from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


SECTION_HEADINGS = (
    "summary",
    "profile",
    "about",
    "experience",
    "work experience",
    "employment",
    "professional experience",
    "projects",
    "selected projects",
    "skills",
    "technical skills",
    "core skills",
    "education",
    "certifications",
    "publications",
    "research",
    "awards",
)


@dataclass
class Chunk:
    text: str
    section: str
    index: int


_HEADING_RE = re.compile(
    r"^(?:"
    + "|".join(re.escape(h) for h in sorted(SECTION_HEADINGS, key=len, reverse=True))
    + r")\s*:?\s*$",
    re.IGNORECASE,
)


def chunk_cv(text: str, max_chars: int = 900, overlap: int = 120) -> list[Chunk]:
    """Split a CV into overlapping windows, preserving detected section names."""
    sections = _split_sections(text)
    chunks: list[Chunk] = []
    idx = 0
    for section, body in sections:
        windows = _windows(body, max_chars=max_chars, overlap=overlap)
        if not windows:
            continue
        for window in windows:
            chunks.append(Chunk(text=window, section=section, index=idx))
            idx += 1
    if not chunks:
        chunks.append(Chunk(text=text.strip(), section="unknown", index=0))
    return chunks


def _split_sections(text: str) -> list[tuple[str, str]]:
    current = "preamble"
    buckets: dict[str, list[str]] = {current: []}
    order = [current]
    for raw in text.splitlines():
        line = raw.strip()
        if line and _HEADING_RE.match(line):
            current = line.rstrip(":").strip().lower()
            if current not in buckets:
                buckets[current] = []
                order.append(current)
            continue
        buckets[current].append(raw)
    return [(name, "\n".join(buckets[name]).strip()) for name in order if "".join(buckets[name]).strip()]


def _windows(text: str, max_chars: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            break_at = text.rfind("\n", start + max_chars // 2, end)
            if break_at == -1:
                break_at = text.rfind(" ", start + max_chars // 2, end)
            if break_at > start:
                end = break_at
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return out


def stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
