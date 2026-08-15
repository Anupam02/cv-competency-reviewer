from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cv_reviewer.chunking import chunk_cv
from cv_reviewer.embeddings import build_embedder
from cv_reviewer.ingest import SUPPORTED_SUFFIXES
from cv_reviewer.reviewer import review_cv_bytes, review_cv_text
from cv_reviewer.schema import CompetencyReview
from cv_reviewer.vectorstore import InMemoryVectorStore

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="CV AI Competency Reviewer",
    description=(
        "Inventories AI technical competencies evidenced in a CV. "
        "This service does not make hiring, pass/fail, interview, or employment decisions."
    ),
    version="0.1.0",
)


class TextReviewRequest(BaseModel):
    cv_text: str = Field(min_length=40)
    filename: str | None = "pasted.txt"
    use_llm: bool | None = None


class AskRequest(BaseModel):
    cv_text: str = Field(min_length=40)
    question: str = Field(min_length=5)
    top_k: int = Field(default=4, ge=1, le=10)


class AskResponse(BaseModel):
    question: str
    answer_mode: str
    retrieved_excerpts: list[dict]
    note: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/review", response_model=CompetencyReview)
async def review_upload(
    file: UploadFile = File(...),
    use_llm: str | None = Form(default=None),
) -> CompetencyReview:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Upload a PDF, DOCX, TXT, or MD file.")
    data = await file.read()
    parsed_llm: bool | None
    if use_llm is None or use_llm == "":
        parsed_llm = None
    else:
        parsed_llm = use_llm.strip().lower() in {"1", "true", "yes", "on"}
    try:
        return review_cv_bytes(data, filename=file.filename or "cv.txt", use_llm=parsed_llm)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/review-text", response_model=CompetencyReview)
def review_text(payload: TextReviewRequest) -> CompetencyReview:
    return review_cv_text(payload.cv_text, filename=payload.filename, use_llm=payload.use_llm)


@app.post("/ask", response_model=AskResponse)
def ask_cv(payload: AskRequest) -> AskResponse:
    """Retrieve CV excerpts relevant to a reviewer question.

    Returns evidence only. It does not answer whether to hire or interview anyone.
    """
    lowered = payload.question.lower()
    blocked = ("hire", "hiring", "interview", "reject", "offer", "pass", "fail")
    if any(term in lowered for term in blocked):
        raise HTTPException(
            status_code=400,
            detail="Questions about hiring, interviews, or pass/fail outcomes are not supported.",
        )
    chunks = chunk_cv(payload.cv_text)
    store = InMemoryVectorStore(build_embedder())
    store.add(chunks)
    hits = store.query(payload.question, top_k=payload.top_k)
    excerpts = [
        {
            "section": item.chunk.section,
            "score": round(item.score, 4),
            "text": item.chunk.text,
        }
        for item in hits
    ]
    return AskResponse(
        question=payload.question,
        answer_mode="retrieved_excerpts_only",
        retrieved_excerpts=excerpts,
        note=(
            "These excerpts were retrieved by embedding similarity. "
            "They are evidence from the CV, not a competency judgement or employment decision."
        ),
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
