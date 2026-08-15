from pathlib import Path

from cv_reviewer.reviewer import review_cv_file
from cv_reviewer.taxonomy import REQUIRED_AREAS

SAMPLES = Path(__file__).resolve().parents[1] / "sample_cvs"
REQUIRED_NAMES = {area.name for area in REQUIRED_AREAS}


def test_strong_cv_demonstrates_core_ai_areas() -> None:
    review = review_cv_file(SAMPLES / "strong_ai_engineer.txt", use_llm=False)
    by_area = {c.area: c for c in review.competencies}
    assert set(by_area) == REQUIRED_NAMES
    assert by_area["Python"].demonstrated
    assert by_area["Retrieval-Augmented Generation (RAG)"].demonstrated
    assert by_area["Vector databases"].demonstrated
    assert by_area["Large Language Models (LLMs)"].demonstrated
    assert by_area["Embeddings"].demonstrated
    assert review.llm_used is False
    assert "hiring recommendation" in review.disclaimer.lower()


def test_keyword_only_cv_is_mentioned_not_demonstrated() -> None:
    review = review_cv_file(SAMPLES / "keyword_only.txt", use_llm=False)
    by_area = {c.area: c for c in review.competencies}
    rag = by_area["Retrieval-Augmented Generation (RAG)"]
    assert rag.demonstrated is False
    assert rag.mentioned_without_evidence or rag.apparent_level == "mentioned_only"
    python = by_area["Python"]
    # Python is only in the skills list on this sample.
    assert python.demonstrated is False


def test_sparse_cv_marks_gaps() -> None:
    review = review_cv_file(SAMPLES / "sparse.txt", use_llm=False)
    assert "Python" in review.skills_not_demonstrated
    assert all(not c.demonstrated for c in review.competencies)


def test_review_does_not_make_employment_decisions() -> None:
    review = review_cv_file(SAMPLES / "strong_ai_engineer.txt", use_llm=False)
    payload = review.model_dump()
    payload.pop("disclaimer")
    blob = str(payload).lower()
    for term in ("hire", "reject", "interview", "pass/fail", "offer"):
        assert term not in blob
