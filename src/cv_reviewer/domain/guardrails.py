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
