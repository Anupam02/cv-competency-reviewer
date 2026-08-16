from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cv_reviewer.domain.models import CompetencyReview
from cv_reviewer.infrastructure.ingest import SUPPORTED_SUFFIXES, ingest_bytes
from cv_reviewer.infrastructure.samples import load_sample_library
from cv_reviewer.matching_schema import AssessmentBundle
from cv_reviewer.pipeline import TextDocument, run_assessment
from cv_reviewer.reviewer import review_cv_bytes, review_cv_text

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="CV AI Competency Reviewer",
    description=(
        "Inventories AI technical competencies evidenced in CVs and compares them with "
        "position descriptions using retrieved excerpts. "
        "This service does not make hiring, pass/fail, interview, or employment decisions."
    ),
    version="0.2.0",
)


class TextReviewRequest(BaseModel):
    cv_text: str = Field(min_length=40)
    filename: str | None = "pasted.txt"
    use_llm: bool | None = None


class NamedText(BaseModel):
    filename: str = "document.txt"
    text: str = Field(min_length=20)


class RunRequest(BaseModel):
    cvs: list[NamedText]
    positions: list[NamedText] = Field(default_factory=list)
    use_llm: bool | None = False


def _parse_llm_flag(use_llm: str | None) -> bool | None:
    if use_llm is None or use_llm == "":
        return None
    return use_llm.strip().lower() in {"1", "true", "yes", "on"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/samples")
def samples() -> dict:
    return load_sample_library()


@app.post("/review", response_model=CompetencyReview)
async def review_upload(
    file: UploadFile = File(...),
    use_llm: str | None = Form(default=None),
) -> CompetencyReview:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Upload a PDF, DOCX, TXT, or MD file.")
    data = await file.read()
    try:
        return review_cv_bytes(data, filename=file.filename or "cv.txt", use_llm=_parse_llm_flag(use_llm))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/review-text", response_model=CompetencyReview)
def review_text(payload: TextReviewRequest) -> CompetencyReview:
    return review_cv_text(payload.cv_text, filename=payload.filename, use_llm=payload.use_llm)


@app.post("/run", response_model=AssessmentBundle)
def run(payload: RunRequest) -> AssessmentBundle:
    if not payload.cvs:
        raise HTTPException(status_code=400, detail="Provide at least one CV.")
    cvs = [TextDocument(filename=item.filename, text=item.text) for item in payload.cvs]
    positions = [TextDocument(filename=item.filename, text=item.text) for item in payload.positions]
    try:
        return run_assessment(cvs, positions, use_llm=payload.use_llm)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/run-files", response_model=AssessmentBundle)
async def run_files(
    cvs: list[UploadFile] = File(default=[]),
    positions: list[UploadFile] = File(default=[]),
    use_llm: str | None = Form(default=None),
) -> AssessmentBundle:
    if not cvs:
        raise HTTPException(status_code=400, detail="Upload at least one CV.")
    cv_docs: list[TextDocument] = []
    pos_docs: list[TextDocument] = []
    try:
        for file in cvs:
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                raise HTTPException(status_code=400, detail=f"Unsupported CV type: {file.filename}")
            ingested = ingest_bytes(await file.read(), filename=file.filename or "cv.txt")
            cv_docs.append(TextDocument(filename=ingested.filename, text=ingested.text))
        for file in positions:
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                raise HTTPException(status_code=400, detail=f"Unsupported position type: {file.filename}")
            ingested = ingest_bytes(await file.read(), filename=file.filename or "position.txt")
            pos_docs.append(TextDocument(filename=ingested.filename, text=ingested.text))
        return run_assessment(cv_docs, pos_docs, use_llm=_parse_llm_flag(use_llm) or False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
