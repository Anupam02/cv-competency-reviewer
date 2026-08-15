from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from cv_reviewer.schema import CompetencyReview, DISCLAIMER, EvidenceItem

Coverage = Literal["demonstrated", "mentioned_only", "not_found"]


class RequirementAlignment(BaseModel):
    requirement: str
    coverage: Coverage
    evidence: list[EvidenceItem] = Field(default_factory=list)
    notes: str


class PositionAlignment(BaseModel):
    cv_label: str
    cv_filename: str | None = None
    position_title: str
    position_filename: str | None = None
    demonstrated_requirements: int
    mentioned_requirements: int
    missing_requirements: int
    total_requirements: int
    coverage_ratio: float = Field(
        description="Share of position requirements with demonstrated CV evidence. Not a hiring score."
    )
    requirements: list[RequirementAlignment]
    summary: str


class EvidenceRecommendation(BaseModel):
    subject: str
    subject_kind: Literal["cv", "position"]
    ordered_matches: list[dict]
    method: str
    caveat: str


class AssessmentBundle(BaseModel):
    competency_reviews: list[CompetencyReview]
    alignments: list[PositionAlignment]
    recommendations: list[EvidenceRecommendation]
    disclaimer: str = DISCLAIMER
    note: str = (
        "Recommendations only order the documents you supplied by demonstrated evidence overlap. "
        "They are not hiring, interview, pass/fail, or employment decisions."
    )
