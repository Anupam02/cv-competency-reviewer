from __future__ import annotations

import logging
import re
import time
import uuid

from cv_reviewer.application.ports import LlmRefinerPort, VectorIndexFactory, retrieve_for_area
from cv_reviewer.domain.assessment import heuristic_review
from cv_reviewer.domain.chunking import chunk_cv
from cv_reviewer.domain.evidence_policy import find_additional_technologies
from cv_reviewer.domain.guardrails import enforce_quote_grounding, strip_decision_language
from cv_reviewer.domain.models import (
    ChunkTrace,
    ClassificationEvent,
    CompetencyReview,
    PipelineTrace,
    RetrievalEvent,
    TraceStep,
)
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
        chunk_traces = [
            ChunkTrace(
                index=chunk.index,
                section=chunk.section,
                chars=len(chunk.text),
                preview=_preview(chunk.text, 160),
            )
            for chunk in chunks
        ]
        section_counts: dict[str, int] = {}
        for chunk in chunks:
            section_counts[chunk.section] = section_counts.get(chunk.section, 0) + 1
        chunk_records = [
            f"{section}: {count} chunk(s)" for section, count in section_counts.items()
        ] + [
            f"#{item.index} [{item.section}] {item.chars} chars — {item.preview}"
            for item in chunk_traces
        ]
        steps.append(
            TraceStep(
                name="chunk",
                duration_ms=_ms(t0),
                detail=f"{len(chunks)} chunks across {len(section_counts)} sections",
                records=chunk_records,
            )
        )

        t1 = time.perf_counter()
        store = self._index_factory()
        store.add(chunks)
        retrieved = {area.id: retrieve_for_area(store, area) for area in REQUIRED_AREAS}
        extra = find_additional_technologies(chunks)
        retrieval_events: list[RetrievalEvent] = []
        retrieve_records: list[str] = []
        for area in REQUIRED_AREAS:
            scored_hits = retrieved.get(area.id, [])
            retrieve_records.append(
                f"{area.name}: {len(scored_hits)} hit(s)"
                + (
                    " scores " + ", ".join(f"{float(s.score):.3f}" for s in scored_hits[:4])
                    if scored_hits
                    else " (none)"
                )
            )
            for scored in scored_hits[:4]:
                retrieval_events.append(
                    RetrievalEvent(
                        area=area.name,
                        chunk_index=scored.chunk.index,
                        section=scored.chunk.section,
                        score=round(float(scored.score), 4),
                        preview=_preview(scored.chunk.text, 180),
                    )
                )
                retrieve_records.append(
                    f"  #{scored.chunk.index} [{scored.chunk.section}] "
                    f"score={float(scored.score):.3f} — {_preview(scored.chunk.text, 140)}"
                )
        extra_names = sorted(extra)
        if extra_names:
            retrieve_records.append("additional technologies: " + ", ".join(extra_names))
        hit_count = len(retrieval_events)
        steps.append(
            TraceStep(
                name="retrieve",
                duration_ms=_ms(t1),
                detail=f"{hit_count} retrieved excerpts across {len(REQUIRED_AREAS)} competency areas",
                records=retrieve_records,
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
        classifications: list[ClassificationEvent] = []
        classify_records: list[str] = []
        for assessment in review.competencies:
            counts: dict[str, int] = {}
            for ev in assessment.evidence:
                counts[ev.evidence_type] = counts.get(ev.evidence_type, 0) + 1
            previews = [_preview(ev.quote, 120) for ev in assessment.evidence[:3]]
            classifications.append(
                ClassificationEvent(
                    area=assessment.area,
                    apparent_level=assessment.apparent_level,
                    demonstrated=assessment.demonstrated,
                    evidence_counts=counts,
                    quote_previews=previews,
                )
            )
            count_txt = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no evidence items"
            classify_records.append(
                f"{assessment.area} → {assessment.apparent_level} "
                f"(demonstrated={assessment.demonstrated}; {count_txt})"
            )
            for preview in previews:
                classify_records.append(f"  quote: {preview}")
        if review.skills_not_demonstrated:
            classify_records.append(
                "not demonstrated: " + ", ".join(review.skills_not_demonstrated)
            )
        if review.insufficient_information:
            classify_records.append(
                "thin evidence: " + ", ".join(review.insufficient_information)
            )
        steps.append(
            TraceStep(
                name="classify",
                duration_ms=_ms(t2),
                detail=(
                    f"{sum(1 for c in review.competencies if c.demonstrated)} demonstrated, "
                    f"{len(review.insufficient_information)} thin-evidence, "
                    f"{len(review.skills_not_demonstrated)} not demonstrated"
                ),
                records=classify_records,
            )
        )

        llm_used = False
        llm_skipped: str | None = None
        pack = _excerpt_pack(retrieved)
        excerpt_chars = len(pack)
        t3 = time.perf_counter()
        if use_llm and self._llm_refiner is not None:
            try:
                review = self._llm_refiner.refine(review, pack)
                llm_used = True
                steps.append(
                    TraceStep(
                        name="llm_refine",
                        duration_ms=_ms(t3),
                        detail=f"provider={self._llm_provider}; excerpt pack {excerpt_chars} chars",
                        records=[
                            f"provider={self._llm_provider}",
                            f"excerpt pack={excerpt_chars} chars from retrieved chunks only",
                            "model may rewrite JSON; quotes are grounded afterwards",
                        ],
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
                        records=[
                            f"error={llm_skipped}",
                            f"excerpt pack would have been {excerpt_chars} chars",
                        ],
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
                    records=[
                        f"configured provider={self._llm_provider}",
                        f"excerpt pack prepared anyway ({excerpt_chars} chars) but not sent",
                    ],
                )
            )

        t4 = time.perf_counter()
        quotes_before = sum(len(item.evidence) for item in review.competencies) + sum(
            len(item.evidence) for item in review.additional_ai_technologies
        )
        review = strip_decision_language(review)
        review = enforce_quote_grounding(review, text)
        quotes_after = sum(len(item.evidence) for item in review.competencies) + sum(
            len(item.evidence) for item in review.additional_ai_technologies
        )
        quotes_dropped = max(quotes_before - quotes_after, 0)
        steps.append(
            TraceStep(
                name="guardrail",
                duration_ms=_ms(t4),
                detail=(
                    f"redact employment-decision wording; "
                    f"{quotes_dropped} ungrounded quote(s) removed"
                ),
                records=[
                    f"quotes before grounding={quotes_before}",
                    f"quotes after grounding={quotes_after}",
                    f"removed={quotes_dropped}",
                    "decision-language strip applied to notes and rationales",
                ],
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
            excerpt_chars=excerpt_chars,
            quotes_dropped=quotes_dropped,
            steps=steps,
            chunks=chunk_traces,
            retrieval=retrieval_events,
            classifications=classifications,
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
