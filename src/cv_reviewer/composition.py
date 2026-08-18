from __future__ import annotations

import os

from cv_reviewer.application.review_cv import ReviewCvService
from cv_reviewer.infrastructure.embeddings import build_embedder
from cv_reviewer.infrastructure.llm_openai import OpenAiRefiner, llm_provider, should_attach_llm
from cv_reviewer.infrastructure.vectorstore import InMemoryVectorStore


def build_review_service(*, use_llm: bool | None = None) -> ReviewCvService:
    backend = os.getenv("EMBEDDING_BACKEND", "hashed")
    embedder = build_embedder(backend)

    def index_factory() -> InMemoryVectorStore:
        return InMemoryVectorStore(embedder)

    llm = OpenAiRefiner() if should_attach_llm(use_llm) else None
    return ReviewCvService(
        index_factory=index_factory,
        llm_refiner=llm,
        embedding_backend=backend,
        llm_provider=llm_provider(),
    )
