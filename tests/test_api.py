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


def test_health() -> None:
    res = client.get("/health")
    assert res.status_code == 200


def test_review_text_endpoint() -> None:
    res = client.post("/review-text", json={"cv_text": CV, "use_llm": False})
    assert res.status_code == 200
    body = res.json()
    assert len(body["competencies"]) == 9
    assert "employment determination" in body["disclaimer"]


def test_ask_rejects_hiring_questions() -> None:
    res = client.post("/ask", json={"cv_text": CV, "question": "Should we hire this person?"})
    assert res.status_code == 400


def test_ask_returns_excerpts() -> None:
    res = client.post("/ask", json={"cv_text": CV, "question": "What evidence is there of RAG?"})
    assert res.status_code == 200
    body = res.json()
    assert body["retrieved_excerpts"]
    assert body["answer_mode"] == "retrieved_excerpts_only"
