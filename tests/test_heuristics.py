from cv_reviewer.chunking import Chunk
from cv_reviewer.domain.assessment import heuristic_review
from cv_reviewer.domain.taxonomy import REQUIRED_AREAS
from cv_reviewer.evidence import classify_evidence_type
from cv_reviewer.reviewer import review_cv_text


def test_skills_list_is_mentioned() -> None:
    chunk = Chunk(text="Python, LLMs, RAG, embeddings, FAISS", section="skills", index=0)
    assert classify_evidence_type(chunk, ["python", "rag"]) == "mentioned"


def test_experience_with_action_is_demonstrated() -> None:
    chunk = Chunk(
        text="Built a RAG pipeline with FAISS and sentence-transformers in production.",
        section="experience",
        index=1,
    )
    assert classify_evidence_type(chunk, ["rag", "faiss"]) == "demonstrated"


def test_familiar_with_is_mentioned() -> None:
    chunk = Chunk(
        text="Familiar with PyTorch and TensorFlow from a weekend course.",
        section="experience",
        index=2,
    )
    assert classify_evidence_type(chunk, ["pytorch"]) == "mentioned"


def test_incidental_job_reference_is_ambiguous() -> None:
    chunk = Chunk(
        text="Briefly touched TensorFlow while pairing on a bug fix with a teammate.",
        section="experience",
        index=3,
    )
    assert classify_evidence_type(chunk, ["tensorflow"]) == "ambiguous"


def test_insufficient_information_is_reachable_without_llm() -> None:
    cv = """Priya Shah

Experience
- Maintained a Java billing service and SQL reports.
- Briefly touched TensorFlow while pairing on a bug fix with a teammate.
"""
    review = review_cv_text(cv, filename="priya.txt", use_llm=False)
    by_area = {c.area: c for c in review.competencies}
    frameworks = by_area["AI frameworks and libraries"]
    assert frameworks.apparent_level == "insufficient_information"
    assert frameworks.demonstrated is False
    assert "AI frameworks and libraries" in review.insufficient_information
    assert "AI frameworks and libraries" not in review.skills_not_demonstrated


def test_pytorch_tensorflow_count_as_ml_dl_signal() -> None:
    cv = """Casey Ng

Experience
- Trained and fine-tuned a model using PyTorch.
- Briefly touched TensorFlow while pairing on a bug fix with a teammate.
"""
    review = review_cv_text(cv, filename="casey-ml.txt", use_llm=False)
    by_area = {c.area: c for c in review.competencies}
    ml = by_area["Machine Learning / Deep Learning"]
    assert ml.evidence, "PyTorch/TensorFlow-only ML work must retrieve ML/DL evidence"
    assert ml.apparent_level != "not_demonstrated"
    assert ml.demonstrated is True
    assert any(item.evidence_type == "demonstrated" for item in ml.evidence)

    incidental = """Priya Shah

Experience
- Maintained a Java billing service and SQL reports.
- Briefly touched TensorFlow while pairing on a bug fix with a teammate.
"""
    thin = review_cv_text(incidental, filename="priya-ml.txt", use_llm=False)
    ml_thin = {c.area: c for c in thin.competencies}["Machine Learning / Deep Learning"]
    assert ml_thin.evidence
    assert ml_thin.demonstrated is False
    assert ml_thin.apparent_level == "insufficient_information"


def test_empty_area_is_not_demonstrated_not_insufficient() -> None:
    retrieved = {area.id: [] for area in REQUIRED_AREAS}
    review = heuristic_review(
        retrieved=retrieved,
        extra_tech={},
        candidate_name="Sam",
        filename="empty.txt",
        chunks=[],
    )
    assert all(c.apparent_level == "not_demonstrated" for c in review.competencies)
    assert review.insufficient_information == []



def test_skills_list_is_mentioned() -> None:
    chunk = Chunk(text="Python, LLMs, RAG, embeddings, FAISS", section="skills", index=0)
    assert classify_evidence_type(chunk, ["python", "rag"]) == "mentioned"


def test_experience_with_action_is_demonstrated() -> None:
    chunk = Chunk(
        text="Built a RAG pipeline with FAISS and sentence-transformers in production.",
        section="experience",
        index=1,
    )
    assert classify_evidence_type(chunk, ["rag", "faiss"]) == "demonstrated"


def test_familiar_with_is_mentioned() -> None:
    chunk = Chunk(
        text="Familiar with PyTorch and TensorFlow from a weekend course.",
        section="experience",
        index=2,
    )
    assert classify_evidence_type(chunk, ["pytorch"]) == "mentioned"
