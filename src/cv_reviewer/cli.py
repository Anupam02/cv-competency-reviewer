from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cv_reviewer.reviewer import review_cv_file, review_cv_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Review AI technical competencies evidenced in a CV. "
            "Does not make hiring or interview decisions."
        )
    )
    parser.add_argument("cv_path", nargs="?", help="Path to a CV (PDF, DOCX, TXT, MD)")
    parser.add_argument("--text", help="Raw CV text instead of a file")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM refinement even if an API key is set")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args(argv)

    if not args.cv_path and not args.text:
        parser.error("Provide a CV path or --text")

    use_llm = False if args.no_llm else None
    if args.text:
        review = review_cv_text(args.text, filename="stdin.txt", use_llm=use_llm)
    else:
        review = review_cv_file(Path(args.cv_path), use_llm=use_llm)

    dump = review.model_dump()
    indent = 2 if args.pretty else None
    json.dump(dump, sys.stdout, indent=indent, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
