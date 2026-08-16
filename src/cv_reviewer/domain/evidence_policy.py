from __future__ import annotations

import re

from cv_reviewer.domain.chunking import Chunk
from cv_reviewer.domain.taxonomy import ADDITIONAL_TECH_KEYWORDS, REQUIRED_AREAS, CompetencyArea

MENTION_SECTIONS = {
    "skills",
    "technical skills",
    "core skills",
    "preamble",
}

ACTION_VERBS = (
    "built",
    "build",
    "designed",
    "design",
    "implemented",
    "implement",
    "developed",
    "develop",
    "deployed",
    "deploy",
    "trained",
    "fine-tuned",
    "finetuned",
    "evaluated",
    "evaluation",
    "integrated",
    "orchestrat",
    "indexed",
    "retrieved",
    "production",
    "shipped",
    "led",
    "owned",
    "maintained",
    "optimis",
    "optimiz",
    "served",
    "published",
    "researched",
    "benchmark",
)

WEAK_MARKERS = (
    "familiar with",
    "knowledge of",
    "exposure to",
    "interested in",
    "coursework",
    "basic understanding",
    "awareness of",
)

AMBIGUOUS_MARKERS = (
    "briefly",
    "touched",
    "pairing",
    "paired on",
    "helped with",
    "shadowed",
    "once used",
    "one-off",
    "introductory",
    "intro to",
    "minor involvement",
    "while pairing",
)


def keyword_hits(text: str, keywords: tuple[str, ...]) -> list[str]:
    lowered = text.lower()
    return [kw for kw in keywords if kw.lower() in lowered]


def classify_evidence_type(chunk: Chunk, matched_keywords: list[str]) -> str:
    """Domain policy: demonstrated, mentioned, or too thin to assess (ambiguous)."""
    section = chunk.section.lower()
    body = chunk.text.lower()
    if any(marker in body for marker in WEAK_MARKERS):
        return "mentioned"
    if section in MENTION_SECTIONS:
        if any(verb in body for verb in ACTION_VERBS):
            return "demonstrated"
        return "mentioned"
    if any(marker in body for marker in AMBIGUOUS_MARKERS):
        return "ambiguous"
    if any(verb in body for verb in ACTION_VERBS):
        return "demonstrated"
    if section in {
        "experience",
        "work experience",
        "employment",
        "professional experience",
        "projects",
        "selected projects",
        "research",
        "publications",
    } and matched_keywords:
        # A passing job-line reference without activity verbs is not enough to assess.
        return "ambiguous"
    if matched_keywords and len(" ".join(chunk.text.split())) < 80:
        return "ambiguous"
    return "mentioned"


def relevant_chunks(scored: list, area: CompetencyArea) -> list:
    kept = []
    for item in scored:
        hits = keyword_hits(item.chunk.text, area.keywords + area.related_tech)
        if hits:
            kept.append(item)
    return kept


class _LexicalScored:
    def __init__(self, chunk: Chunk, score: float = 0.0) -> None:
        self.chunk = chunk
        self.score = score


def passages_for_area(retrieved: list, chunks: list[Chunk], area: CompetencyArea) -> list:
    """Use retrieved hits plus any other chunk that names the area.

    Retrieval alone can miss a one-line passing reference; treating that as
    not_demonstrated is overconfident.
    """
    scored = list(relevant_chunks(retrieved, area))
    seen = {item.chunk.index for item in scored}
    for chunk in chunks:
        if chunk.index in seen:
            continue
        hits = keyword_hits(chunk.text, area.keywords + area.related_tech)
        if hits:
            scored.append(_LexicalScored(chunk))
            seen.add(chunk.index)
    return scored


def find_additional_technologies(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    required_terms = {
        kw.lower()
        for area in REQUIRED_AREAS
        for kw in area.keywords + area.related_tech
    }
    found: dict[str, list[Chunk]] = {}
    for tech in ADDITIONAL_TECH_KEYWORDS:
        if tech.lower() in required_terms:
            continue
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(tech.lower())}(?![a-z0-9])")
        matching = [c for c in chunks if pattern.search(c.text.lower())]
        if matching:
            found[tech] = matching
    return found
