from cv_reviewer.infrastructure.llm_openai import (
    llm_enabled,
    llm_model,
    parse_json_object,
    should_attach_llm,
)


def test_parse_json_object_from_fenced_block() -> None:
    raw = """Here you go
```json
{"candidate_name": "Alex", "competencies": []}
```
"""
    parsed = parse_json_object(raw)
    assert parsed["candidate_name"] == "Alex"


def test_llm_enabled_ollama_without_openai_key(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    assert llm_enabled() is True
    assert should_attach_llm(None) is True
    assert should_attach_llm(False) is False
    assert llm_model() == "llama3.2"


def test_llm_disabled_when_provider_none(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "none")
    assert llm_enabled() is False
    assert should_attach_llm(True) is False


def test_openai_requires_key(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert llm_enabled() is False
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert llm_enabled() is True
