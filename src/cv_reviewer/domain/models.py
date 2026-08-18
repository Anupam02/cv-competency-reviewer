from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EvidenceType = Literal["demonstrated", "mentioned", "ambiguous"]
CompetenceLevel = Literal[
    "advanced",
    "working",
    "foundational",
    "mentioned_only",
    "not_demonstrated",
    "insufficient_information",
]

DISCLAIMER = (
    "This output is an evidence inventory of AI-related technical content found in the CV. "
    "It is not a hiring recommendation, interview decision, pass/fail result, ranking, "
    "or employment determination."
)


class EvidenceItem(BaseModel):
    quote: str = Field(description="Verbatim or near-verbatim excerpt from the CV.")
    source_section: str = Field(description="CV section the excerpt came from, if known.")
    evidence_type: EvidenceType = Field(
        description=(
            "demonstrated = used in work, projects, or outcomes; "
            "mentioned = listed or named without supporting activity; "
            "ambiguous = too thin or incidental to assess reliably."
        )
    )
    rationale: str = Field(description="Why this excerpt is classified this way.")


class CompetencyAssessment(BaseModel):
    area: str
    apparent_level: CompetenceLevel
    demonstrated: bool = Field(
        description="True only when the CV shows the skill used in work or projects."
    )
    mentioned_without_evidence: bool = Field(
        description="True when the skill is named but not backed by activity or outcomes."
    )
    evidence: list[EvidenceItem] = Field(default_factory=list)
    assessment_notes: str


class AdditionalTechnology(BaseModel):
    name: str
    apparent_level: CompetenceLevel
    evidence: list[EvidenceItem] = Field(default_factory=list)
    assessment_notes: str


class TraceStep(BaseModel):
    name: str
    duration_ms: float
    status: str = "ok"
    detail: str = ""


class RetrievalEvent(BaseModel):
    area: str
    chunk_index: int
    section: str
    score: float
    preview: str


class PipelineTrace(BaseModel):
    """Per-review pipeline trace. Local, open-source style observability — not Datadog."""

    run_id: str
    filename: str | None = None
    embedding_backend: str = "hashed"
    llm_provider: str = "none"
    llm_used: bool = False
    llm_skipped_reason: str | None = None
    chunk_count: int = 0
    steps: list[TraceStep] = Field(default_factory=list)
    retrieval: list[RetrievalEvent] = Field(default_factory=list)


class CompetencyReview(BaseModel):
    candidate_name: str | None = None
    source_filename: str | None = None
    competencies: list[CompetencyAssessment]
    additional_ai_technologies: list[AdditionalTechnology] = Field(default_factory=list)
    skills_not_demonstrated: list[str] = Field(default_factory=list)
    insufficient_information: list[str] = Field(default_factory=list)
    review_limitations: str
    disclaimer: str = DISCLAIMER
    retrieval_used: bool = True
    llm_used: bool = False
    trace: PipelineTrace | None = None

    def assert_no_decision_language(self) -> None:
        banned = (
            "hire",
            "hiring",
            "reject",
            "pass",
            "fail",
            "interview",
            "offer",
            "should be hired",
            "do not hire",
            "recommend for the role",
        )
        blob = self.model_dump_json().lower()
        for term in banned:
            if term in blob and term not in DISCLAIMER.lower():
                # Allow the disclaimer itself; block decision language elsewhere.
                remainder = blob.replace(DISCLAIMER.lower(), "")
                if term in remainder:
                    raise ValueError(f"Review contains forbidden decision language: {term}")
