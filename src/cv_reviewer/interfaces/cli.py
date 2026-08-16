from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cv_reviewer.infrastructure.ingest import ingest_path
from cv_reviewer.pipeline import TextDocument, run_assessment
from cv_reviewer.reviewer import review_cv_file, review_cv_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Review AI competencies in CVs and compare them with position descriptions. "
            "Does not make hiring or interview decisions."
        )
    )
    parser.add_argument("cv_path", nargs="?", help="Path to a single CV (PDF, DOCX, TXT, MD)")
    parser.add_argument("--cvs", nargs="+", help="One or more CV files")
    parser.add_argument("--positions", nargs="+", help="One or more position description files")
    parser.add_argument("--text", help="Raw CV text instead of a file")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM refinement even if an API key is set")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args(argv)

    use_llm = False if args.no_llm else None
    indent = 2 if args.pretty else None

    if args.cvs or args.positions:
        cv_paths = list(args.cvs or [])
        if args.cv_path:
            cv_paths.insert(0, args.cv_path)
        if not cv_paths:
            parser.error("Provide --cvs or a CV path")
        cvs = []
        for path in cv_paths:
            doc = ingest_path(path)
            cvs.append(TextDocument(filename=doc.filename, text=doc.text))
        positions = []
        for path in args.positions or []:
            doc = ingest_path(path)
            positions.append(TextDocument(filename=doc.filename, text=doc.text))
        result = run_assessment(cvs, positions, use_llm=use_llm)
        json.dump(result.model_dump(), sys.stdout, indent=indent, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0

    if not args.cv_path and not args.text:
        parser.error("Provide a CV path, --text, or --cvs/--positions")

    if args.text:
        review = review_cv_text(args.text, filename="stdin.txt", use_llm=use_llm)
    else:
        review = review_cv_file(Path(args.cv_path), use_llm=use_llm)

    json.dump(review.model_dump(), sys.stdout, indent=indent, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
