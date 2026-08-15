from pathlib import Path

from cv_reviewer.pipeline import TextDocument, run_assessment
from cv_reviewer.matching import parse_position


def test_parse_position_extracts_bullets() -> None:
    text = Path("sample_positions/ai_platform_engineer.txt").read_text(encoding="utf-8")
    position = parse_position(text, "ai_platform_engineer.txt")
    assert "AI Platform Engineer" in position.title
    assert len(position.requirements) >= 5


def test_strong_cv_has_more_ai_evidence_than_sparse() -> None:
    root = Path("sample_cvs")
    pos = Path("sample_positions/ai_platform_engineer.txt")
    bundle = run_assessment(
        [
            TextDocument("strong.txt", (root / "strong_ai_engineer.txt").read_text()),
            TextDocument("sparse.txt", (root / "sparse.txt").read_text()),
        ],
        [TextDocument(pos.name, pos.read_text())],
        use_llm=False,
    )
    by_cv = {a.cv_filename: a for a in bundle.alignments}
    strong = by_cv["strong.txt"]
    sparse = by_cv["sparse.txt"]
    assert strong.demonstrated_requirements > sparse.demonstrated_requirements
    assert bundle.recommendations
    blob = bundle.model_dump_json().lower()
    remainder = blob.replace("hiring", "", 2)
    assert "should be hired" not in remainder


def test_keyword_cv_aligns_better_to_backend_than_ai_role() -> None:
    cv = Path("sample_cvs/keyword_only.txt").read_text()
    backend = Path("sample_positions/backend_services_engineer.txt").read_text()
    ai = Path("sample_positions/ai_platform_engineer.txt").read_text()
    bundle = run_assessment(
        [TextDocument("keyword.txt", cv)],
        [
            TextDocument("backend.txt", backend),
            TextDocument("ai.txt", ai),
        ],
        use_llm=False,
    )
    backend_row = next(a for a in bundle.alignments if "backend" in a.position_title.lower())
    ai_row = next(a for a in bundle.alignments if "platform" in a.position_title.lower())
    assert backend_row.demonstrated_requirements >= ai_row.demonstrated_requirements


def test_strong_cv_covers_ai_role_more_than_backend_role() -> None:
    cv = Path("sample_cvs/strong_ai_engineer.txt").read_text()
    backend = Path("sample_positions/backend_services_engineer.txt").read_text()
    ai = Path("sample_positions/ai_platform_engineer.txt").read_text()
    bundle = run_assessment(
        [TextDocument("strong.txt", cv)],
        [TextDocument("backend.txt", backend), TextDocument("ai.txt", ai)],
        use_llm=False,
    )
    backend_row = next(a for a in bundle.alignments if "backend" in a.position_title.lower())
    ai_row = next(a for a in bundle.alignments if "platform" in a.position_title.lower())
    assert ai_row.demonstrated_requirements > backend_row.demonstrated_requirements
