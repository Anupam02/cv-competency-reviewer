"""Run the labelled evaluation matrix on the fictional sample CVs (heuristic path)."""

from __future__ import annotations

from pathlib import Path

from cv_reviewer.evaluation.matrix import GOLD_MATRIX, compare_review, format_matrix
from cv_reviewer.reviewer import review_cv_file


def sample_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "sample_cvs"


def collect_cells():
    cells = []
    root = sample_dir()
    for filename in GOLD_MATRIX:
        review = review_cv_file(root / filename, use_llm=False)
        cells.extend(compare_review(filename, review))
    return cells


def main() -> int:
    print(format_matrix(collect_cells()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
