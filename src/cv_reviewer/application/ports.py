from __future__ import annotations

from typing import Protocol

from cv_reviewer.domain.chunking import Chunk
from cv_reviewer.domain.models import CompetencyReview
from cv_reviewer.domain.taxonomy import CompetencyArea


class ScoredChunk(Protocol):
    chunk: Chunk
    score: float


class VectorIndexPort(Protocol):
    """Persistence-agnostic similarity index over CV passages."""

    def add(self, chunks: list[Chunk]) -> None: ...

    def query(self, text: str, top_k: int = 4) -> list[ScoredChunk]: ...


class VectorIndexFactory(Protocol):
    def __call__(self) -> VectorIndexPort: ...


class LlmRefinerPort(Protocol):
    def refine(self, seed: CompetencyReview, excerpt_pack: str) -> CompetencyReview: ...


class DocumentLoaderPort(Protocol):
    def load_path(self, path: str) -> tuple[str, str]:
        """Return (text, filename)."""
        ...


def retrieve_for_area(store: VectorIndexPort, area: CompetencyArea, top_k: int = 4) -> list:
    seen: set[int] = set()
    merged = []
    for query in area.queries:
        for scored in store.query(query, top_k=top_k):
            if scored.chunk.index in seen:
                continue
            seen.add(scored.chunk.index)
            merged.append(scored)
    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[:top_k]
