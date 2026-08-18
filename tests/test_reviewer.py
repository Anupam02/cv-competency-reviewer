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
    apis = by_area["Model integration and APIs"]
    assert apis.demonstrated is False


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


def test_review_includes_pipeline_trace() -> None:
    review = review_cv_file(SAMPLES / "strong_ai_engineer.txt", use_llm=False)
    assert review.trace is not None
    assert review.trace.chunk_count >= 3
    names = [step.name for step in review.trace.steps]
    assert names == ["chunk", "retrieve", "classify", "llm_refine", "guardrail"]
    assert review.trace.llm_used is False
    assert any(event.area == "Python" for event in review.trace.retrieval)
    chunk_step = next(step for step in review.trace.steps if step.name == "chunk")
    assert len(chunk_step.records) >= review.trace.chunk_count
    assert review.trace.chunks
    assert any(item.section == "experience" for item in review.trace.chunks)
    classify_step = next(step for step in review.trace.steps if step.name == "classify")
    assert any("Python" in line for line in classify_step.records)
    assert len(review.trace.classifications) == 9
    python = next(c for c in review.trace.classifications if c.area == "Python")
    assert python.demonstrated is True
