from __future__ import annotations

from pathlib import Path

from cv_reviewer.composition import build_review_service
from cv_reviewer.domain.models import CompetencyReview
from cv_reviewer.infrastructure.ingest import ingest_bytes, ingest_path
from cv_reviewer.infrastructure.llm_openai import llm_enabled


def review_cv_file(path: str | Path, *, use_llm: bool | None = None) -> CompetencyReview:
    document = ingest_path(path)
    return review_cv_text(document.text, filename=document.filename, use_llm=use_llm)


def review_cv_bytes(
    data: bytes,
    filename: str,
    *,
    use_llm: bool | None = None,
) -> CompetencyReview:
    document = ingest_bytes(data, filename=filename)
    return review_cv_text(document.text, filename=document.filename, use_llm=use_llm)


def review_cv_text(
    text: str,
    *,
    filename: str | None = None,
    use_llm: bool | None = None,
) -> CompetencyReview:
    enabled = llm_enabled() if use_llm is None else use_llm
    service = build_review_service(use_llm=enabled)
    return service.review_text(text, filename=filename, use_llm=enabled)


def guess_name(text: str) -> str | None:
    from cv_reviewer.application.review_cv import guess_name as _guess

    return _guess(text)
