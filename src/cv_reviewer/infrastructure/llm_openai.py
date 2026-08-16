from __future__ import annotations

import json
import os
import re

from cv_reviewer.domain.models import CompetencyReview
from cv_reviewer.infrastructure.prompts import SYSTEM_PROMPT, user_prompt

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def llm_provider() -> str:
    return (os.getenv("LLM_PROVIDER") or "ollama").strip().lower()


def llm_enabled() -> bool:
    provider = llm_provider()
    if provider in {"none", "off", "heuristic", "disabled"}:
        return False
    if provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    if provider == "ollama":
        return True
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL"))


def llm_model() -> str:
    if os.getenv("OPENAI_MODEL"):
        return os.getenv("OPENAI_MODEL")
    if llm_provider() == "ollama":
        return os.getenv("OLLAMA_MODEL", "llama3.2")
    return "gpt-4o-mini"


def llm_base_url() -> str | None:
    if os.getenv("OPENAI_BASE_URL"):
        return os.getenv("OPENAI_BASE_URL")
    if llm_provider() == "ollama":
        host = (os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        if host.endswith("/v1"):
            return host
        return host + "/v1"
    return None


def should_attach_llm(use_llm: bool | None) -> bool:
    if use_llm is False:
        return False
    if llm_provider() in {"none", "off", "heuristic", "disabled"}:
        return False
    if use_llm is True:
        return True
    return llm_enabled()


def parse_json_object(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object")
    return parsed


class OpenAiRefiner:
    """OpenAI-compatible chat adapter (Ollama /v1 or OpenAI)."""

    def refine(self, seed: CompetencyReview, excerpt_pack: str) -> CompetencyReview:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY") or "ollama",
            base_url=llm_base_url(),
        )
        schema = CompetencyReview.model_json_schema()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\nJSON schema:\n" + json.dumps(schema)},
            {
                "role": "user",
                "content": user_prompt(excerpt_pack, seed.candidate_name)
                + "\n\nHeuristic draft JSON (correct using excerpts only):\n"
                + seed.model_dump_json()
                + "\n\nRespond with a single JSON object only.",
            },
        ]
        kwargs: dict = {"model": llm_model(), "temperature": 0, "messages": messages}
        try:
            response = client.chat.completions.create(
                **kwargs,
                response_format={"type": "json_object"},
            )
        except Exception as first:
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception:
                raise first
        content = response.choices[0].message.content or ""
        refined = CompetencyReview.model_validate(parse_json_object(content))
        refined.llm_used = True
        refined.retrieval_used = True
        refined.disclaimer = seed.disclaimer
        if not refined.source_filename:
            refined.source_filename = seed.source_filename
        return refined


def refine_with_llm(seed: CompetencyReview, excerpt_pack: str) -> CompetencyReview:
    return OpenAiRefiner().refine(seed, excerpt_pack)
