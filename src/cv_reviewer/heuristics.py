from __future__ import annotations

from cv_reviewer.chunking import Chunk
from cv_reviewer.evidence import classify_evidence_type, keyword_hits, relevant_chunks
from cv_reviewer.schema import (
    AdditionalTechnology,
    CompetenceLevel,
    CompetencyAssessment,
    CompetencyReview,
    DISCLAIMER,
    EvidenceItem,
)
from cv_reviewer.taxonomy import REQUIRED_AREAS, CompetencyArea
from cv_reviewer.vectorstore import InMemoryVectorStore, ScoredChunk


def heuristic_review(
    *,
    chunks: list[Chunk],
    store: InMemoryVectorStore,
    retrieved: dict[str, list[ScoredChunk]],
    extra_tech: dict[str, list[Chunk]],
    candidate_name: str | None,
    filename: str | None,
) -> CompetencyReview:
    competencies: list[CompetencyAssessment] = []
    not_demonstrated: list[str] = []
    insufficient: list[str] = []

    for area in REQUIRED_AREAS:
        scored = relevant_chunks(retrieved.get(area.id, []), area)
        assessment = _assess_area(area, scored)
        competencies.append(assessment)
        if assessment.apparent_level == "not_demonstrated":
            not_demonstrated.append(area.name)
        elif assessment.apparent_level in {"insufficient_information", "mentioned_only"}:
            if assessment.apparent_level == "insufficient_information":
                insufficient.append(area.name)
            if not assessment.demonstrated:
                not_demonstrated.append(area.name)

    additional: list[AdditionalTechnology] = []
    for name, tech_chunks in sorted(extra_tech.items()):
        items = [
            EvidenceItem(
                quote=_trim(c.text),
                source_section=c.section,
                evidence_type=classify_evidence_type(c, [name]),
                rationale="Matched additional AI-related technology named in the CV.",
            )
            for c in tech_chunks[:3]
        ]
        demonstrated = any(i.evidence_type == "demonstrated" for i in items)
        additional.append(
            AdditionalTechnology(
                name=name,
                apparent_level="working" if demonstrated else "mentioned_only",
                evidence=items,
                assessment_notes=(
                    "Activity in experience or projects supports use of this technology."
                    if demonstrated
                    else "Named in the CV without a clear description of how it was used."
                ),
            )
        )

    limitations = (
        "Assessments are based only on text extracted from the submitted CV. "
        "Absence of evidence is not evidence of absence. Depth, recency, and quality of work "
        "cannot be verified from a CV alone. Keyword lists are treated as mentions, not demonstrations."
    )
    return CompetencyReview(
        candidate_name=candidate_name,
        source_filename=filename,
        competencies=competencies,
        additional_ai_technologies=additional,
        skills_not_demonstrated=sorted(set(not_demonstrated)),
        insufficient_information=sorted(set(insufficient)),
        review_limitations=limitations,
        disclaimer=DISCLAIMER,
        retrieval_used=True,
        llm_used=False,
    )


def _assess_area(area: CompetencyArea, scored: list[ScoredChunk]) -> CompetencyAssessment:
    evidence: list[EvidenceItem] = []
    demonstrated = False
    mentioned = False
    for item in scored:
        hits = keyword_hits(item.chunk.text, area.keywords + area.related_tech)
        if not hits:
            continue
        etype = classify_evidence_type(item.chunk, hits)
        if etype == "demonstrated":
            demonstrated = True
        else:
            mentioned = True
        evidence.append(
            EvidenceItem(
                quote=_trim(item.chunk.text),
                source_section=item.chunk.section,
                evidence_type=etype,  # type: ignore[arg-type]
                rationale=_rationale(etype, hits, item.chunk.section, item.score),
            )
        )

    level = _level(demonstrated=demonstrated, mentioned=mentioned, evidence=evidence)
    notes = _notes(area, level, evidence)
    return CompetencyAssessment(
        area=area.name,
        apparent_level=level,
        demonstrated=demonstrated,
        mentioned_without_evidence=mentioned and not demonstrated,
        evidence=evidence,
        assessment_notes=notes,
    )


def _level(*, demonstrated: bool, mentioned: bool, evidence: list[EvidenceItem]) -> CompetenceLevel:
    if demonstrated:
        demo = [e for e in evidence if e.evidence_type == "demonstrated"]
        joined = " ".join(e.quote.lower() for e in demo)
        strong_signals = (
            "production",
            "deploy",
            "architect",
            "end-to-end",
            "led",
            "published",
            "evaluation",
            "benchmark",
        )
        if sum(1 for s in strong_signals if s in joined) >= 2 or len(demo) >= 3:
            return "advanced"
        if len(demo) >= 2 or any(s in joined for s in strong_signals):
            return "working"
        return "foundational"
    if mentioned:
        return "mentioned_only"
    if not evidence:
        return "not_demonstrated"
    return "insufficient_information"


def _notes(area: CompetencyArea, level: CompetenceLevel, evidence: list[EvidenceItem]) -> str:
    if level == "not_demonstrated":
        return (
            f"No supporting excerpts were retrieved for {area.name}. "
            "The CV does not demonstrate this competency based on the available text."
        )
    if level == "mentioned_only":
        return (
            f"{area.name} appears in the CV (for example in a skills list) but the retrieved "
            "excerpts do not describe using it in work, projects, or outcomes."
        )
    if level == "insufficient_information":
        return (
            f"Some related wording was retrieved for {area.name}, but it is too thin or ambiguous "
            "to support a reliable competence assessment."
        )
    kinds = {e.evidence_type for e in evidence}
    if "demonstrated" in kinds:
        return (
            f"Retrieved excerpts describe {area.name} in the context of work, projects, or outcomes. "
            "Apparent level reflects how specific and activity-based those excerpts are, not a hiring judgement."
        )
    return f"Limited excerpts were found for {area.name}."


def _rationale(etype: str, hits: list[str], section: str, score: float) -> str:
    hit_text = ", ".join(hits[:5]) if hits else "semantic similarity to the competency queries"
    if etype == "demonstrated":
        return (
            f"Excerpt is from '{section}' and describes activity involving {hit_text} "
            f"(retrieval score {score:.2f})."
        )
    return (
        f"Excerpt names {hit_text} in '{section}' without a clear activity or outcome "
        f"(retrieval score {score:.2f})."
    )


def _trim(text: str, limit: int = 420) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"
