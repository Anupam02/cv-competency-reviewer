from __future__ import annotations

import re

from cv_reviewer.domain.models import CompetencyReview

_BANNED = re.compile(
    r"\b(hire|hiring|hired|reject|rejection|interview|offer|pass\/fail|should be hired|"
    r"do not hire|recommend for(?: the)? role|suitable candidate|not suitable)\b",
    re.IGNORECASE,
)


def strip_decision_language(review: CompetencyReview) -> CompetencyReview:
    """Invariant: the bounded context never emits an employment decision."""

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
    return review


def _norm(text: str) -> str:
    cleaned = text.replace("\u2026", " ").replace("...", " ")
    return " ".join(cleaned.split()).lower()


def quote_is_grounded(quote: str, source_text: str) -> bool:
    """True when the quote is a whitespace-normalized substring of the CV."""
    needle = _norm(quote).strip(" .")
    if len(needle) < 8:
        return False
    return needle in _norm(source_text)


def enforce_quote_grounding(review: CompetencyReview, source_text: str) -> CompetencyReview:
    """Drop quotes that do not appear in the CV so the LLM cannot invent evidence."""
    dropped = 0
    for item in review.competencies:
        kept = [ev for ev in item.evidence if quote_is_grounded(ev.quote, source_text)]
        dropped += len(item.evidence) - len(kept)
        item.evidence = kept
        demo_left = any(ev.evidence_type == "demonstrated" for ev in kept)
        mention_left = any(ev.evidence_type == "mentioned" for ev in kept)
        amb_left = any(ev.evidence_type == "ambiguous" for ev in kept)
        if item.demonstrated and not demo_left:
            item.demonstrated = False
            item.mentioned_without_evidence = mention_left
            if mention_left:
                item.apparent_level = "mentioned_only"
            elif amb_left:
                item.apparent_level = "insufficient_information"
            else:
                item.apparent_level = "not_demonstrated"
        elif not kept and item.apparent_level not in {"not_demonstrated", "mentioned_only"}:
            if item.mentioned_without_evidence:
                item.apparent_level = "mentioned_only"
            else:
                item.apparent_level = "not_demonstrated"
                item.demonstrated = False
    for extra in review.additional_ai_technologies:
        kept = [ev for ev in extra.evidence if quote_is_grounded(ev.quote, source_text)]
        dropped += len(extra.evidence) - len(kept)
        extra.evidence = kept
    if dropped:
        review.review_limitations += (
            f" Dropped {dropped} quote(s) that were not found in the CV text."
        )
    return review
