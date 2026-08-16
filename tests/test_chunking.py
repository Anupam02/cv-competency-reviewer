from cv_reviewer.chunking import chunk_cv
from cv_reviewer.reviewer import review_cv_text


def test_each_experience_bullet_is_its_own_chunk() -> None:
    text = """Casey Ng

Experience
- Shipped and deployed a production RAG pipeline
- Briefly touched TensorFlow while pairing on a bug fix
"""
    chunks = chunk_cv(text)
    experience = [c.text for c in chunks if c.section == "experience"]
    assert experience == [
        "Shipped and deployed a production RAG pipeline",
        "Briefly touched TensorFlow while pairing on a bug fix",
    ]


def test_bullet_prefixes_are_stripped() -> None:
    text = """Skills
* Python
• RAG
1. FAISS
"""
    chunks = chunk_cv(text)
    skills = [c.text for c in chunks if c.section == "skills"]
    assert skills == ["Python", "RAG", "FAISS"]


def test_oversized_line_splits_by_sentence_then_windows() -> None:
    first = "Shipped a RAG service."
    second = "Deployed embeddings in production."
    line = f"{first} {second}"
    max_chars = max(len(first), len(second))
    chunks = chunk_cv(f"Experience\n{line}", max_chars=max_chars, overlap=0)
    experience = [c.text for c in chunks if c.section == "experience"]
    assert experience == [first, second]

    huge = "x" * 50
    chunks = chunk_cv(f"Experience\n{huge}", max_chars=20, overlap=0)
    experience = [c.text for c in chunks if c.section == "experience"]
    assert all(len(piece) <= 20 for piece in experience)
    assert "".join(experience) == huge


def test_strong_bullet_stays_demonstrated_next_to_ambiguous_bullet() -> None:
    cv = """Casey Ng

Experience
- Shipped and deployed a production RAG pipeline
- Briefly touched TensorFlow while pairing on a bug fix
"""
    review = review_cv_text(cv, filename="casey.txt", use_llm=False)
    by_area = {c.area: c for c in review.competencies}
    rag = by_area["Retrieval-Augmented Generation (RAG)"]
    assert rag.demonstrated is True
    assert rag.apparent_level != "insufficient_information"
    assert any(item.evidence_type == "demonstrated" for item in rag.evidence)

    frameworks = by_area["AI frameworks and libraries"]
    assert frameworks.demonstrated is False
    assert frameworks.apparent_level == "insufficient_information"
