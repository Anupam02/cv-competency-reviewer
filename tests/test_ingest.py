from __future__ import annotations

from pathlib import Path

from cv_reviewer.ingest import ingest_bytes, ingest_path
from cv_reviewer.chunking import chunk_cv


def test_ingest_txt(tmp_path: Path) -> None:
    path = tmp_path / "cv.txt"
    path.write_text("Ada Lovelace\n\nSkills\nPython", encoding="utf-8")
    doc = ingest_path(path)
    assert "Ada Lovelace" in doc.text
    assert doc.filename == "cv.txt"


def test_ingest_rejects_empty() -> None:
    try:
        ingest_bytes(b"   \n", filename="empty.txt")
    except ValueError as exc:
        assert "extractable" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")


def test_chunk_preserves_skills_section() -> None:
    text = """Name\n\nSkills\nPython, RAG\n\nExperience\nBuilt a FastAPI service.\n"""
    chunks = chunk_cv(text)
    sections = {c.section for c in chunks}
    assert "skills" in sections
    assert "experience" in sections
