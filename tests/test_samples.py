from pathlib import Path

from cv_reviewer.infrastructure.samples import load_sample_library, sample_directories


def test_sample_library_reads_repo_files() -> None:
    library = load_sample_library()
    cv_names = {doc.filename for doc in library.cvs}
    pos_names = {doc.filename for doc in library.positions}
    assert "strong_ai_engineer.txt" in cv_names
    assert "keyword_only.txt" in cv_names
    assert "sparse.txt" in cv_names
    assert "bmc_project_architect_consulting_india.txt" in pos_names
    assert all(doc.text.strip() for doc in library.cvs)
    assert Path(library.cv_source).name == "sample_cvs"


def test_sample_directories_survive_other_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cv_dir, pos_dir = sample_directories()
    assert (cv_dir / "strong_ai_engineer.txt").exists()
    assert (pos_dir / "ai_platform_engineer.txt").exists()
