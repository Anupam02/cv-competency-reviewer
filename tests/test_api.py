from fastapi.testclient import TestClient

from cv_reviewer.api import app

client = TestClient(app)

CV = """Casey Ng

Experience
- Implemented embeddings with sentence-transformers and stored vectors in FAISS.
- Built a RAG chatbot in Python using the OpenAI API.

Skills
PyTorch
"""

POSITION = """Position: Retrieval Engineer

Requirements
- Build RAG pipelines
- Use a vector database
- Call an LLM API from Python
"""


def test_health() -> None:
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "llm_provider" in body


def test_samples_endpoint() -> None:
    res = client.get("/samples")
    assert res.status_code == 200
    body = res.json()
    assert len(body["cvs"]) >= 3
    assert len(body["positions"]) >= 3


def test_review_text_endpoint() -> None:
    res = client.post("/review-text", json={"cv_text": CV, "use_llm": False})
    assert res.status_code == 200
    body = res.json()
    assert len(body["competencies"]) == 9
    assert "employment determination" in body["disclaimer"]


def test_run_bundle() -> None:
    res = client.post(
        "/run",
        json={
            "cvs": [{"filename": "casey.txt", "text": CV}],
            "positions": [{"filename": "role.txt", "text": POSITION}],
            "use_llm": False,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["competency_reviews"]
    assert body["alignments"]
    assert body["recommendations"]
    alignment = body["alignments"][0]
    assert alignment["demonstrated_requirements"] >= 1
