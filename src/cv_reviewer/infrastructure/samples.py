from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from cv_reviewer.infrastructure.ingest import ingest_path

_PKG_DATA = Path(__file__).resolve().parents[1] / "data"


class SampleDoc(BaseModel):
    filename: str
    label: str
    text: str
    kind: str
    path: str = ""


class SampleLibrary(BaseModel):
    cvs: list[SampleDoc]
    positions: list[SampleDoc]
    cv_source: str
    position_source: str


def _repo_root() -> Path | None:
    here = Path(__file__).resolve()
    if len(here.parents) >= 4:
        root = here.parents[3]
        if (root / "sample_cvs").is_dir():
            return root
    cwd = Path.cwd()
    if (cwd / "sample_cvs").is_dir():
        return cwd
    return None


def sample_directories() -> tuple[Path, Path]:
    """Prefer the repo folders; fall back to files shipped inside the package."""
    root = _repo_root()
    if root is not None:
        return root / "sample_cvs", root / "sample_positions"
    return _PKG_DATA / "sample_cvs", _PKG_DATA / "sample_positions"


def load_sample_library() -> SampleLibrary:
    cv_dir, pos_dir = sample_directories()
    return SampleLibrary(
        cvs=_load_dir(cv_dir, kind="cv"),
        positions=_load_dir(pos_dir, kind="position"),
        cv_source=str(cv_dir.resolve()) if cv_dir.exists() else "",
        position_source=str(pos_dir.resolve()) if pos_dir.exists() else "",
    )


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
                path=str(file.resolve()),
            )
        )
    return docs
