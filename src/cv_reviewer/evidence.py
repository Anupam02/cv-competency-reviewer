from __future__ import annotations

import re

from cv_reviewer.chunking import Chunk
from cv_reviewer.taxonomy import ADDITIONAL_TECH_KEYWORDS, REQUIRED_AREAS, CompetencyArea
from cv_reviewer.vectorstore import InMemoryVectorStore, ScoredChunk

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


def keyword_hits(text: str, keywords: tuple[str, ...]) -> list[str]:
    lowered = text.lower()
    hits = []
    for kw in keywords:
        if kw.lower() in lowered:
            hits.append(kw)
    return hits


def classify_evidence_type(chunk: Chunk, matched_keywords: list[str]) -> str:
    section = chunk.section.lower()
    body = chunk.text.lower()
    if any(marker in body for marker in WEAK_MARKERS):
        return "mentioned"
    if section in MENTION_SECTIONS:
        # A skills-list line is a mention unless it also describes an activity.
        if any(verb in body for verb in ACTION_VERBS):
            return "demonstrated"
        return "mentioned"
    if any(verb in body for verb in ACTION_VERBS):
        return "demonstrated"
    # Experience/project sections that only name a technology still count as weak demonstration
    # only if they describe a role; otherwise they remain mentions.
    if section in {"experience", "work experience", "employment", "professional experience", "projects", "selected projects", "research", "publications"}:
        if matched_keywords and len(chunk.text) > 80:
            return "demonstrated"
    return "mentioned"


def retrieve_for_area(store: InMemoryVectorStore, area: CompetencyArea, top_k: int = 4) -> list[ScoredChunk]:
    seen: set[int] = set()
    merged: list[ScoredChunk] = []
    for query in area.queries:
        for scored in store.query(query, top_k=top_k):
            if scored.chunk.index in seen:
                continue
            seen.add(scored.chunk.index)
            merged.append(scored)
    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[:top_k]


def relevant_chunks(scored: list[ScoredChunk], area: CompetencyArea) -> list[ScoredChunk]:
    kept: list[ScoredChunk] = []
    for item in scored:
        hits = keyword_hits(item.chunk.text, area.keywords + area.related_tech)
        if hits:
            kept.append(item)
    return kept


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
