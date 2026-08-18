from __future__ import annotations

import logging
import re
import time
import uuid

from cv_reviewer.application.ports import LlmRefinerPort, VectorIndexFactory, retrieve_for_area
from cv_reviewer.domain.assessment import heuristic_review
from cv_reviewer.domain.chunking import chunk_cv
from cv_reviewer.domain.evidence_policy import find_additional_technologies
from cv_reviewer.domain.guardrails import strip_decision_language
from cv_reviewer.domain.models import PipelineTrace, RetrievalEvent, TraceStep, CompetencyReview
from cv_reviewer.domain.taxonomy import REQUIRED_AREAS

logger = logging.getLogger("cv_reviewer.trace")


class ReviewCvService:
    """Application use case: inventory AI competencies evidenced in one CV."""

    def __init__(
        self,
        *,
        index_factory: VectorIndexFactory,
        llm_refiner: LlmRefinerPort | None = None,
        embedding_backend: str = "hashed",
        llm_provider: str = "none",
    ) -> None:
        self._index_factory = index_factory
        self._llm_refiner = llm_refiner
        self._embedding_backend = embedding_backend
        self._llm_provider = llm_provider

    def review_text(
        self,
        text: str,
        *,
        filename: str | None = None,
        use_llm: bool | None = False,
    ) -> CompetencyReview:
        run_id = uuid.uuid4().hex[:12]
        steps: list[TraceStep] = []
        t0 = time.perf_counter()

        chunks = chunk_cv(text)
        steps.append(
            TraceStep(
                name="chunk",
                duration_ms=_ms(t0),
                detail=f"{len(chunks)} section-aware chunks",
            )
        )

        t1 = time.perf_counter()
        store = self._index_factory()
        store.add(chunks)
        retrieved = {area.id: retrieve_for_area(store, area) for area in REQUIRED_AREAS}
        extra = find_additional_technologies(chunks)
        hit_count = sum(len(v) for v in retrieved.values())
        steps.append(
            TraceStep(
                name="retrieve",
                duration_ms=_ms(t1),
                detail=f"{hit_count} retrieved excerpts across {len(REQUIRED_AREAS)} competency areas",
            )
        )

        t2 = time.perf_counter()
        review = heuristic_review(
            retrieved=retrieved,
            extra_tech=extra,
            candidate_name=guess_name(text),
            filename=filename,
            chunks=chunks,
        )
        steps.append(
            TraceStep(
                name="classify",
                duration_ms=_ms(t2),
                detail="heuristic demonstrated / mentioned / ambiguous",
            )
        )

        llm_used = False
        llm_skipped: str | None = None
        t3 = time.perf_counter()
        if use_llm and self._llm_refiner is not None:
            pack = _excerpt_pack(retrieved)
            try:
                review = self._llm_refiner.refine(review, pack)
                llm_used = True
                steps.append(
                    TraceStep(
                        name="llm_refine",
                        duration_ms=_ms(t3),
                        detail=f"provider={self._llm_provider}",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                llm_skipped = exc.__class__.__name__
                review.review_limitations += f" LLM refinement was skipped ({exc.__class__.__name__})."
                steps.append(
                    TraceStep(
                        name="llm_refine",
                        duration_ms=_ms(t3),
                        status="skipped",
                        detail=llm_skipped,
                    )
                )
        else:
            llm_skipped = "disabled"
            steps.append(
                TraceStep(
                    name="llm_refine",
                    duration_ms=_ms(t3),
                    status="skipped",
                    detail="checkbox off, provider none, or no refiner attached",
                )
            )

        t4 = time.perf_counter()
        review = strip_decision_language(review)
        steps.append(
            TraceStep(
                name="guardrail",
                duration_ms=_ms(t4),
                detail="redact employment-decision wording",
            )
        )

        retrieval_events: list[RetrievalEvent] = []
        for area in REQUIRED_AREAS:
            for scored in retrieved.get(area.id, [])[:4]:
                retrieval_events.append(
                    RetrievalEvent(
                        area=area.name,
                        chunk_index=scored.chunk.index,
                        section=scored.chunk.section,
                        score=round(float(scored.score), 4),
                        preview=_preview(scored.chunk.text),
                    )
                )

        review.trace = PipelineTrace(
            run_id=run_id,
            filename=filename,
            embedding_backend=self._embedding_backend,
            llm_provider=self._llm_provider,
            llm_used=llm_used,
            llm_skipped_reason=None if llm_used else llm_skipped,
            chunk_count=len(chunks),
            steps=steps,
            retrieval=retrieval_events,
        )
        logger.info(
            "cv_review_trace run_id=%s chunks=%s llm_used=%s steps=%s",
            run_id,
            len(chunks),
            llm_used,
            [step.name for step in steps],
        )
        return review


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _preview(text: str, limit: int = 140) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


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
