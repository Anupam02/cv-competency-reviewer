from pathlib import Path

from cv_reviewer.domain.guardrails import enforce_quote_grounding, quote_is_grounded
from cv_reviewer.domain.models import CompetencyAssessment, CompetencyReview, EvidenceItem, DISCLAIMER
from cv_reviewer.evaluation.matrix import GOLD_MATRIX, accuracy, compare_review, format_matrix
from cv_reviewer.reviewer import review_cv_file, review_cv_text

SAMPLES = Path(__file__).resolve().parents[1] / "sample_cvs"


def _empty_review(evidence: list[EvidenceItem]) -> CompetencyReview:
    return CompetencyReview(
        competencies=[
            CompetencyAssessment(
                area="Python",
                apparent_level="working",
                demonstrated=True,
                mentioned_without_evidence=False,
                evidence=evidence,
                assessment_notes="notes",
            )
        ],
        review_limitations="limit",
        disclaimer=DISCLAIMER,
    )


def test_quote_must_appear_in_cv() -> None:
    source = "Experience\n- Built a RAG pipeline in Python."
    assert quote_is_grounded("Built a RAG pipeline in Python.", source)
    assert not quote_is_grounded("Deployed a secret nuclear model in production.", source)


def test_ungrounded_quotes_are_dropped_and_level_falls_back() -> None:
    review = _empty_review(
        [
            EvidenceItem(
                quote="Invented experience that is not on the CV at all.",
                source_section="experience",
                evidence_type="demonstrated",
                rationale="llm",
            )
        ]
    )
    cleaned = enforce_quote_grounding(review, "Experience\n- Wrote SQL reports.")
    assert cleaned.competencies[0].evidence == []
    assert cleaned.competencies[0].demonstrated is False
    assert cleaned.competencies[0].apparent_level == "not_demonstrated"
    assert "Dropped 1 quote" in cleaned.review_limitations


def test_evaluation_matrix_matches_gold_labels() -> None:
    cells = []
    for filename in GOLD_MATRIX:
        review = review_cv_file(SAMPLES / filename, use_llm=False)
        cells.extend(compare_review(filename, review))
        review.assert_no_decision_language()
    assert accuracy(cells) == 1.0, format_matrix(cells)


def test_evaluation_matrix_covers_all_required_areas() -> None:
    first = next(iter(GOLD_MATRIX.values()))
    assert len(first) == 9
    review = review_cv_text(
        (SAMPLES / "sparse.txt").read_text(encoding="utf-8"),
        filename="sparse.txt",
        use_llm=False,
    )
    assert {c.area for c in review.competencies} == set(first)
