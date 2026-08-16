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

_BULLET_PREFIX_RE = re.compile(r"^(?:[-*+•●◦▪▸►–—·]|\d+[.)]|\(\d+\))\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_cv(text: str, max_chars: int = 900, overlap: int = 120) -> list[Chunk]:
    """Split a CV into per-claim chunks, preserving detected section names.

    Each non-empty line in a section is one chunk so classification can judge
    bullets independently. Oversized lines are split by sentence, then by
    character windows as a last resort.
    """
    sections = _split_sections(text)
    chunks: list[Chunk] = []
    idx = 0
    for section, body in sections:
        for piece in _chunk_section_body(body, max_chars=max_chars, overlap=overlap):
            chunks.append(Chunk(text=piece, section=section, index=idx))
            idx += 1
    if not chunks:
        chunks.append(Chunk(text=text.strip(), section="unknown", index=0))
    return chunks


def _strip_bullet_prefix(line: str) -> str:
    return _BULLET_PREFIX_RE.sub("", line.strip(), count=1)


def _chunk_section_body(body: str, max_chars: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    for raw_line in body.splitlines():
        line = _strip_bullet_prefix(raw_line)
        if not line:
            continue
        if len(line) <= max_chars:
            pieces.append(line)
            continue
        pieces.extend(_split_long_line(line, max_chars=max_chars, overlap=overlap))
    return pieces


def _split_long_line(line: str, max_chars: int, overlap: int) -> list[str]:
    sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(line) if part.strip()]
    if not sentences:
        return _windows(line, max_chars=max_chars, overlap=overlap)

    grouped: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                grouped.append(current)
                current = ""
            grouped.extend(_windows(sentence, max_chars=max_chars, overlap=overlap))
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            grouped.append(current)
            current = sentence
    if current:
        grouped.append(current)
    return grouped


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
