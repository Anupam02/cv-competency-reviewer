from __future__ import annotations

import os
import re
from pathlib import Path

from cv_reviewer.chunking import chunk_cv
from cv_reviewer.embeddings import build_embedder
from cv_reviewer.evidence import find_additional_technologies, retrieve_for_area
from cv_reviewer.heuristics import heuristic_review
from cv_reviewer.ingest import ingest_bytes, ingest_path
from cv_reviewer.llm import llm_enabled, refine_with_llm
from cv_reviewer.schema import CompetencyReview
from cv_reviewer.taxonomy import REQUIRED_AREAS
from cv_reviewer.vectorstore import InMemoryVectorStore


def review_cv_file(path: str | Path, *, use_llm: bool | None = None) -> CompetencyReview:
    document = ingest_path(path)
    return review_cv_text(
        document.text,
        filename=document.filename,
        use_llm=use_llm,
    )


def review_cv_bytes(
    data: bytes,
    filename: str,
    *,
    use_llm: bool | None = None,
) -> CompetencyReview:
    document = ingest_bytes(data, filename=filename)
    return review_cv_text(document.text, filename=document.filename, use_llm=use_llm)


def review_cv_text(
    text: str,
    *,
    filename: str | None = None,
    use_llm: bool | None = None,
) -> CompetencyReview:
    chunks = chunk_cv(text)
    embedder = build_embedder(os.getenv("EMBEDDING_BACKEND", "hashed"))
    store = InMemoryVectorStore(embedder)
    store.add(chunks)

    retrieved = {area.id: retrieve_for_area(store, area) for area in REQUIRED_AREAS}
    extra = find_additional_technologies(chunks)
    name = _guess_name(text)

    review = heuristic_review(
        chunks=chunks,
        store=store,
        retrieved=retrieved,
        extra_tech=extra,
        candidate_name=name,
        filename=filename,
    )

    should_llm = llm_enabled() if use_llm is None else use_llm
    if should_llm:
        pack = _excerpt_pack(retrieved)
        try:
            review = refine_with_llm(review, pack)
        except Exception as exc:  # noqa: BLE001 - LLM is optional enrichment
            review.review_limitations += f" LLM refinement was skipped ({exc.__class__.__name__})."

    _strip_decision_language(review)
    return review


def _excerpt_pack(retrieved: dict) -> str:
    parts: list[str] = []
    for area in REQUIRED_AREAS:
        parts.append(f"## {area.name}")
        items = retrieved.get(area.id, [])
        if not items:
            parts.append("(no retrieved chunks)")
            continue
        for scored in items:
            parts.append(
                f"[section={scored.chunk.section} score={scored.score:.3f}]\n{scored.chunk.text}"
            )
        parts.append("")
    return "\n".join(parts)


def _guess_name(text: str) -> str | None:
    first = text.strip().splitlines()[0].strip() if text.strip() else ""
    if 2 <= len(first.split()) <= 5 and len(first) <= 60 and not first.lower().endswith(":"):
        if re.match(r"^[A-Za-z][A-Za-z .'-]+$", first):
            return first
    return None


_BANNED = re.compile(
    r"\b(hire|hiring|hired|reject|rejection|interview|offer|pass\/fail|should be hired|"
    r"do not hire|recommend for(?: the)? role|suitable candidate|not suitable)\b",
    re.IGNORECASE,
)


def _strip_decision_language(review: CompetencyReview) -> None:
    """Replace accidental decision language; the product must never decide employment."""

    def clean(value: str) -> str:
        return _BANNED.sub("[redacted-decision-language]", value)

    for item in review.competencies:
        item.assessment_notes = clean(item.assessment_notes)
        for ev in item.evidence:
            ev.rationale = clean(ev.rationale)
    for item in review.additional_ai_technologies:
        item.assessment_notes = clean(item.assessment_notes)
    review.review_limitations = clean(review.review_limitations)
    review.disclaimer = (
        "This output is an evidence inventory of AI-related technical content found in the CV. "
        "It is not a hiring recommendation, interview decision, pass/fail result, ranking, "
        "or employment determination."
    )
