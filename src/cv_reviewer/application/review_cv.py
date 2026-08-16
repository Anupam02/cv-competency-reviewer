from __future__ import annotations

import re

from cv_reviewer.application.ports import LlmRefinerPort, VectorIndexFactory, retrieve_for_area
from cv_reviewer.domain.assessment import heuristic_review
from cv_reviewer.domain.chunking import chunk_cv
from cv_reviewer.domain.evidence_policy import find_additional_technologies
from cv_reviewer.domain.guardrails import enforce_quote_grounding, strip_decision_language
from cv_reviewer.domain.models import CompetencyReview
from cv_reviewer.domain.taxonomy import REQUIRED_AREAS


class ReviewCvService:
    """Application use case: inventory AI competencies evidenced in one CV."""

    def __init__(
        self,
        *,
        index_factory: VectorIndexFactory,
        llm_refiner: LlmRefinerPort | None = None,
    ) -> None:
        self._index_factory = index_factory
        self._llm_refiner = llm_refiner

    def review_text(
        self,
        text: str,
        *,
        filename: str | None = None,
        use_llm: bool | None = False,
    ) -> CompetencyReview:
        chunks = chunk_cv(text)
        store = self._index_factory()
        store.add(chunks)
        retrieved = {area.id: retrieve_for_area(store, area) for area in REQUIRED_AREAS}
        extra = find_additional_technologies(chunks)
        review = heuristic_review(
            retrieved=retrieved,
            extra_tech=extra,
            candidate_name=guess_name(text),
            filename=filename,
            chunks=chunks,
        )
        if use_llm and self._llm_refiner is not None:
            pack = _excerpt_pack(retrieved)
            try:
                review = self._llm_refiner.refine(review, pack)
            except Exception as exc:  # noqa: BLE001
                review.review_limitations += f" LLM refinement was skipped ({exc.__class__.__name__})."
        review = strip_decision_language(review)
        return enforce_quote_grounding(review, text)


def guess_name(text: str) -> str | None:
    first = text.strip().splitlines()[0].strip() if text.strip() else ""
    if 2 <= len(first.split()) <= 5 and len(first) <= 60 and not first.lower().endswith(":"):
        if re.match(r"^[A-Za-z][A-Za-z .'-]+$", first):
            return first
    return None


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
