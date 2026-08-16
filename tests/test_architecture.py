from __future__ import annotations

import ast
from pathlib import Path

DOMAIN = Path(__file__).resolve().parents[1] / "src" / "cv_reviewer" / "domain"


def test_domain_layer_has_no_infrastructure_or_web_imports() -> None:
    forbidden_prefixes = (
        "cv_reviewer.infrastructure",
        "cv_reviewer.interfaces",
        "cv_reviewer.application",
        "fastapi",
        "openai",
        "pypdf",
        "docx",
        "uvicorn",
    )
    for path in DOMAIN.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                assert all(not name.startswith(prefix) for prefix in forbidden_prefixes), (
                    f"{path.name} imports {name}"
                )
