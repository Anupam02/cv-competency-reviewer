from __future__ import annotations

import re
from dataclasses import dataclass

from cv_reviewer.chunking import chunk_cv
from cv_reviewer.evidence import classify_evidence_type
from cv_reviewer.matching_schema import (
    AssessmentBundle,
    EvidenceRecommendation,
    PositionAlignment,
    RequirementAlignment,
)
from cv_reviewer.schema import DISCLAIMER, EvidenceItem
from cv_reviewer.taxonomy import REQUIRED_AREAS
from cv_reviewer.vectorstore import InMemoryVectorStore

WEAK_MATCH_TERMS = {
    "build",
    "built",
    "call",
    "collaborate",
    "communicate",
    "design",
    "designed",
    "develop",
    "developed",
    "document",
    "documented",
    "evaluate",
    "evaluation",
    "follow",
    "implement",
    "implemented",
    "maintain",
    "maintained",
    "partners",
    "production",
    "quality",
    "service",
    "services",
    "support",
    "trade-offs",
    "work",
    "write",
    "approach",
    "behaviour",
    "behavior",
    "weekly",
    "dashboards",
    "incident",
    "operations",
    "operational",
    "reporting",
    "reliable",
    "transactional",
    "generative",
    "required",
    "looking",
    "someone",
}

STOPWORDS = {
    "about",
    "and",
    "for",
    "from",
    "have",
    "that",
    "this",
    "with",
    "your",
    "the",
    "are",
    "was",
    "were",
    "will",
    "not",
    "only",
    "into",
    "over",
    "such",
    "than",
    "then",
    "them",
    "they",
    "those",
    "using",
    "used",
    "role",
    "team",
    "looking",
    "someone",
    "requirements",
    "position",
    "location",
    "fictional",
    "city",
}


@dataclass
class PositionDoc:
    title: str
    filename: str
    text: str
    requirements: list[str]


def parse_position(text: str, filename: str = "position.txt") -> PositionDoc:
    lines = [ln.rstrip() for ln in text.splitlines()]
    title = next((ln.strip() for ln in lines if ln.strip()), filename)
    title = re.sub(r"^position:\s*", "", title, flags=re.IGNORECASE)
    reqs: list[str] = []
    for ln in lines:
        stripped = ln.strip()
        if re.match(r"^[-*•]\s+", stripped):
            reqs.append(re.sub(r"^[-*•]\s+", "", stripped))
    if not reqs:
        reqs = [c.text for c in chunk_cv(text) if len(c.text) > 40][:8]
    if not reqs:
        reqs = [text.strip()[:400]]
    return PositionDoc(title=title, filename=filename, text=text, requirements=reqs)


def requirement_terms(requirement: str) -> tuple[str, ...]:
    terms: list[str] = []
    lowered = requirement.lower()
    for area in REQUIRED_AREAS:
        for kw in area.keywords + area.related_tech + (area.name.lower(),):
            if kw.lower() in lowered and kw.lower() not in terms:
                terms.append(kw.lower())
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9+.#-]{3,}", requirement.lower()):
        if token in STOPWORDS or token in WEAK_MATCH_TERMS or token in terms:
            continue
        terms.append(token)
    return tuple(terms[:24])


def align_cv_to_position(
    *,
    cv_label: str,
    cv_filename: str | None,
    store: InMemoryVectorStore,
    position: PositionDoc,
) -> PositionAlignment:
    items: list[RequirementAlignment] = []
    for req in position.requirements:
        terms = requirement_terms(req)
        hits = store.query(req, top_k=4)
        evidence: list[EvidenceItem] = []
        coverage = "not_found"
        for scored in hits:
            matched = [t for t in terms if t in scored.chunk.text.lower()]
            distinctive = [t for t in matched if t not in WEAK_MATCH_TERMS and t not in STOPWORDS]
            if not distinctive:
                continue
            etype = classify_evidence_type(scored.chunk, distinctive)
            evidence.append(
                EvidenceItem(
                    quote=_trim(scored.chunk.text),
                    source_section=scored.chunk.section,
                    evidence_type=etype,  # type: ignore[arg-type]
                    rationale=(
                        f"Requirement terms ({', '.join(distinctive[:6])}) appear in '{scored.chunk.section}' "
                        f"(retrieval score {scored.score:.2f})."
                    ),
                )
            )
            if etype == "demonstrated":
                coverage = "demonstrated"
            elif coverage != "demonstrated":
                coverage = "mentioned_only"
        if coverage == "demonstrated":
            notes = "The CV describes activity that matches this requirement."
        elif coverage == "mentioned_only":
            notes = "The CV names related terms but does not describe using them in work or projects."
        else:
            notes = "No CV excerpt with overlapping requirement terms was retrieved."
        items.append(
            RequirementAlignment(
                requirement=req,
                coverage=coverage,  # type: ignore[arg-type]
                evidence=evidence[:3],
                notes=notes,
            )
        )

    demo = sum(1 for i in items if i.coverage == "demonstrated")
    mention = sum(1 for i in items if i.coverage == "mentioned_only")
    missing = sum(1 for i in items if i.coverage == "not_found")
    total = len(items) or 1
    ratio = demo / total
    summary = (
        f"{demo} requirement(s) have demonstrated evidence, {mention} are mentioned only, "
        f"and {missing} were not found in the CV text."
    )
    return PositionAlignment(
        cv_label=cv_label,
        cv_filename=cv_filename,
        position_title=position.title,
        position_filename=position.filename,
        demonstrated_requirements=demo,
        mentioned_requirements=mention,
        missing_requirements=missing,
        total_requirements=len(items),
        coverage_ratio=round(ratio, 3),
        requirements=items,
        summary=summary,
    )


def build_recommendations(alignments: list[PositionAlignment]) -> list[EvidenceRecommendation]:
    recs: list[EvidenceRecommendation] = []
    method = (
        "Ordered by the number of position requirements that have demonstrated CV evidence, "
        "then by coverage ratio. Mentions without activity do not increase the order."
    )
    caveat = (
        "This is an evidence-overlap ranking among the documents you uploaded. "
        "It is not a hiring, interview, pass/fail, or employment recommendation."
    )
    cvs = sorted({a.cv_label for a in alignments})
    for cv in cvs:
        rows = [a for a in alignments if a.cv_label == cv]
        rows.sort(key=lambda a: (a.demonstrated_requirements, a.coverage_ratio), reverse=True)
        recs.append(
            EvidenceRecommendation(
                subject=cv,
                subject_kind="cv",
                ordered_matches=[
                    {
                        "position_title": a.position_title,
                        "demonstrated_requirements": a.demonstrated_requirements,
                        "mentioned_requirements": a.mentioned_requirements,
                        "missing_requirements": a.missing_requirements,
                        "coverage_ratio": a.coverage_ratio,
                    }
                    for a in rows
                ],
                method=method,
                caveat=caveat,
            )
        )
    positions = sorted({a.position_title for a in alignments})
    for title in positions:
        rows = [a for a in alignments if a.position_title == title]
        rows.sort(key=lambda a: (a.demonstrated_requirements, a.coverage_ratio), reverse=True)
        recs.append(
            EvidenceRecommendation(
                subject=title,
                subject_kind="position",
                ordered_matches=[
                    {
                        "cv_label": a.cv_label,
                        "demonstrated_requirements": a.demonstrated_requirements,
                        "mentioned_requirements": a.mentioned_requirements,
                        "missing_requirements": a.missing_requirements,
                        "coverage_ratio": a.coverage_ratio,
                    }
                    for a in rows
                ],
                method=method,
                caveat=caveat,
            )
        )
    return recs


def bundle_results(
    reviews,
    alignments: list[PositionAlignment],
) -> AssessmentBundle:
    return AssessmentBundle(
        competency_reviews=reviews,
        alignments=alignments,
        recommendations=build_recommendations(alignments) if alignments else [],
        disclaimer=DISCLAIMER,
    )


def _trim(text: str, limit: int = 420) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"
