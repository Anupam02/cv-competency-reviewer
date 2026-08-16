from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from cv_reviewer.infrastructure.ingest import ingest_path


class SampleDoc(BaseModel):
    filename: str
    label: str
    text: str
    kind: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_sample_library() -> dict[str, list[SampleDoc]]:
    root = _repo_root()
    cvs = _load_dir(root / "sample_cvs", kind="cv")
    positions = _load_dir(root / "sample_positions", kind="position")
    return {"cvs": cvs, "positions": positions}


def _load_dir(path: Path, kind: str) -> list[SampleDoc]:
    if not path.exists():
        return []
    docs: list[SampleDoc] = []
    for file in sorted(path.glob("*")):
        if file.suffix.lower() not in {".txt", ".md", ".pdf", ".docx"}:
            continue
        ingested = ingest_path(file)
        docs.append(
            SampleDoc(
                filename=file.name,
                label=file.stem.replace("_", " "),
                text=ingested.text,
                kind=kind,
            )
        )
    return docs
