from __future__ import annotations

import os

from cv_reviewer.application.review_cv import ReviewCvService
from cv_reviewer.infrastructure.embeddings import build_embedder
from cv_reviewer.infrastructure.llm_openai import OpenAiRefiner, llm_enabled
from cv_reviewer.infrastructure.vectorstore import InMemoryVectorStore


def build_review_service(*, use_llm: bool | None = None) -> ReviewCvService:
    embedder = build_embedder(os.getenv("EMBEDDING_BACKEND", "hashed"))

    def index_factory() -> InMemoryVectorStore:
        return InMemoryVectorStore(embedder)

    llm = None
    enabled = llm_enabled() if use_llm is None else use_llm
    if enabled:
        llm = OpenAiRefiner()
    return ReviewCvService(index_factory=index_factory, llm_refiner=llm)
