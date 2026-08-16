from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np


class EmbeddingBackend(ABC):
    name: str

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised vectors of shape (n, d)."""


class HashedNgramEmbeddings(EmbeddingBackend):
    """Deterministic character n-gram hashing encoder.

    Used as the default so the application and tests run without downloading a model.
    Swap in SentenceTransformerEmbeddings for denser semantic retrieval.
    """

    name = "hashed-ngram"

    def __init__(self, dim: int = 384, ngram_min: int = 3, ngram_max: int = 5) -> None:
        self.dim = dim
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            matrix[i] = self._vector(text)
        return _l2_normalise(matrix)

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        blob = f" {text.lower()} "
        for n in range(self.ngram_min, self.ngram_max + 1):
            if len(blob) < n:
                continue
            for j in range(len(blob) - n + 1):
                gram = blob[j : j + n]
                digest = hashlib.md5(gram.encode("utf-8")).digest()
                h = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[h] += sign
        return vec


class SentenceTransformerEmbeddings(EmbeddingBackend):
    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers:{model_name}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)
        return vectors


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def build_embedder(backend: str = "hashed") -> EmbeddingBackend:
    backend = (backend or "hashed").lower()
    if backend in {"hashed", "hash", "ngram"}:
        return HashedNgramEmbeddings()
    if backend in {"sentence-transformers", "st", "semantic"}:
        return SentenceTransformerEmbeddings()
    raise ValueError(f"Unknown embedding backend: {backend}")
