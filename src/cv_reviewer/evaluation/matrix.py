from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from cv_reviewer.domain.models import CompetenceLevel, CompetencyReview

# Coarse labels used for the evaluation matrix (exact apparent_level is too brittle).
EvidenceClass = str

DEMONSTRATED_LEVELS = {"advanced", "working", "foundational"}


def evidence_class(level: CompetenceLevel | str) -> EvidenceClass:
    if level in DEMONSTRATED_LEVELS:
        return "demonstrated"
    return str(level)


# Gold labels for the fictional sample CVs. These are the intended assessments,
# not hiring outcomes.
GOLD_MATRIX: dict[str, dict[str, EvidenceClass]] = {
    "strong_ai_engineer.txt": {
        "Python": "demonstrated",
        "Large Language Models (LLMs)": "demonstrated",
        "Embeddings": "demonstrated",
        "Vector databases": "demonstrated",
        "Retrieval-Augmented Generation (RAG)": "demonstrated",
        "Machine Learning / Deep Learning": "demonstrated",
        "AI frameworks and libraries": "demonstrated",
        "Model integration and APIs": "demonstrated",
        "AI solution architecture": "demonstrated",
    },
    "keyword_only.txt": {
        "Python": "mentioned_only",
        "Large Language Models (LLMs)": "mentioned_only",
        "Embeddings": "mentioned_only",
        "Vector databases": "mentioned_only",
        "Retrieval-Augmented Generation (RAG)": "mentioned_only",
        "Machine Learning / Deep Learning": "mentioned_only",
        "AI frameworks and libraries": "mentioned_only",
        "Model integration and APIs": "mentioned_only",
        "AI solution architecture": "mentioned_only",
    },
    "sparse.txt": {
        "Python": "not_demonstrated",
        "Large Language Models (LLMs)": "not_demonstrated",
        "Embeddings": "not_demonstrated",
        "Vector databases": "not_demonstrated",
        "Retrieval-Augmented Generation (RAG)": "not_demonstrated",
        "Machine Learning / Deep Learning": "not_demonstrated",
        "AI frameworks and libraries": "not_demonstrated",
        "Model integration and APIs": "not_demonstrated",
        "AI solution architecture": "not_demonstrated",
    },
}


@dataclass
class Cell:
    filename: str
    area: str
    gold: EvidenceClass
    predicted: EvidenceClass

    @property
    def ok(self) -> bool:
        return self.gold == self.predicted


def predicted_class(review: CompetencyReview, area: str) -> EvidenceClass:
    by_area = {item.area: item for item in review.competencies}
    return evidence_class(by_area[area].apparent_level)


def compare_review(filename: str, review: CompetencyReview) -> list[Cell]:
    gold = GOLD_MATRIX[filename]
    return [
        Cell(filename=filename, area=area, gold=expected, predicted=predicted_class(review, area))
        for area, expected in gold.items()
    ]


def confusion(cells: list[Cell]) -> dict[str, Counter[str]]:
    table: dict[str, Counter[str]] = {}
    for cell in cells:
        table.setdefault(cell.gold, Counter())[cell.predicted] += 1
    return table


def accuracy(cells: list[Cell]) -> float:
    if not cells:
        return 0.0
    return sum(1 for cell in cells if cell.ok) / len(cells)


def format_matrix(cells: list[Cell]) -> str:
    labels = [
        "demonstrated",
        "mentioned_only",
        "insufficient_information",
        "not_demonstrated",
    ]
    table = confusion(cells)
    header = ["gold \\ pred", *labels]
    widths = [max(len(h), 12) for h in header]
    lines = [" | ".join(h.ljust(w) for h, w in zip(header, widths, strict=True))]
    lines.append("-+-".join("-" * w for w in widths))
    for gold in labels:
        row = [gold.ljust(widths[0])]
        counts = table.get(gold, Counter())
        for i, pred in enumerate(labels, start=1):
            row.append(str(counts.get(pred, 0)).ljust(widths[i]))
        lines.append(" | ".join(row))
    lines.append("")
    lines.append(f"cells={len(cells)}  accuracy={accuracy(cells):.3f}")
    misses = [c for c in cells if not c.ok]
    if misses:
        lines.append("mismatches:")
        for cell in misses:
            lines.append(f"  {cell.filename} / {cell.area}: gold={cell.gold} pred={cell.predicted}")
    return "\n".join(lines)
