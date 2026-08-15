from cv_reviewer.chunking import Chunk
from cv_reviewer.evidence import classify_evidence_type


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
