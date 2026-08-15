from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cv_reviewer.chunking import Chunk
from cv_reviewer.embeddings import EmbeddingBackend


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class InMemoryVectorStore:
    """Cosine-similarity vector index over CV chunks.

    This is a small in-process vector database. The interface (add + query)
    is intentionally close to Chroma / FAISS / pgvector so it can be swapped.
    """

    def __init__(self, embedder: EmbeddingBackend) -> None:
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("No chunks to index.")
        self.chunks = list(chunks)
        self._matrix = self.embedder.embed([c.text for c in chunks])

    def query(self, text: str, top_k: int = 4) -> list[ScoredChunk]:
        if self._matrix is None:
            raise RuntimeError("Vector store is empty. Call add() first.")
        query_vec = self.embedder.embed([text])[0]
        scores = self._matrix @ query_vec
        k = min(top_k, len(self.chunks))
        # Prefer a score floor so unrelated chunks are not treated as evidence.
        ranked = np.argsort(-scores)
        results: list[ScoredChunk] = []
        for idx in ranked[:k]:
            results.append(ScoredChunk(chunk=self.chunks[int(idx)], score=float(scores[int(idx)])))
        return results
