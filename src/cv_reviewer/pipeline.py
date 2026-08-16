from __future__ import annotations

import os
from dataclasses import dataclass

from cv_reviewer.application.review_cv import guess_name
from cv_reviewer.composition import build_review_service
from cv_reviewer.domain.chunking import chunk_cv
from cv_reviewer.infrastructure.embeddings import build_embedder
from cv_reviewer.infrastructure.llm_openai import llm_enabled
from cv_reviewer.infrastructure.vectorstore import InMemoryVectorStore
from cv_reviewer.matching import align_cv_to_position, bundle_results, parse_position
from cv_reviewer.matching_schema import AssessmentBundle


@dataclass
class TextDocument:
    filename: str
    text: str


def run_assessment(
    cvs: list[TextDocument],
    positions: list[TextDocument],
    *,
    use_llm: bool | None = False,
) -> AssessmentBundle:
    if not cvs:
        raise ValueError("Provide at least one CV.")
    enabled = llm_enabled() if use_llm is None else use_llm
    service = build_review_service(use_llm=enabled)
    reviews = [
        service.review_text(doc.text, filename=doc.filename, use_llm=enabled) for doc in cvs
    ]
    parsed_positions = [parse_position(doc.text, doc.filename) for doc in positions]
    alignments = []
    embedder = build_embedder(os.getenv("EMBEDDING_BACKEND", "hashed"))
    for cv in cvs:
        store = InMemoryVectorStore(embedder)
        store.add(chunk_cv(cv.text))
        label = guess_name(cv.text) or cv.filename
        for position in parsed_positions:
            alignments.append(
                align_cv_to_position(
                    cv_label=label,
                    cv_filename=cv.filename,
                    store=store,
                    position=position,
                )
            )
    return bundle_results(reviews, alignments)
