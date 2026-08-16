from __future__ import annotations

import json
import os

from cv_reviewer.domain.models import CompetencyReview
from cv_reviewer.infrastructure.prompts import SYSTEM_PROMPT, user_prompt


def llm_enabled() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


class OpenAiRefiner:
    """Infrastructure adapter: optional LLM polish of an already-retrieved review."""

    def refine(self, seed: CompetencyReview, excerpt_pack: str) -> CompetencyReview:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        schema = CompetencyReview.model_json_schema()
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + "\nJSON schema:\n" + json.dumps(schema)},
                {
                    "role": "user",
                    "content": user_prompt(excerpt_pack, seed.candidate_name)
                    + "\n\nHeuristic draft JSON (correct using excerpts only):\n"
                    + seed.model_dump_json(),
                },
            ],
        )
        content = response.choices[0].message.content or ""
        refined = CompetencyReview.model_validate_json(content)
        refined.llm_used = True
        refined.retrieval_used = True
        refined.disclaimer = seed.disclaimer
        if not refined.source_filename:
            refined.source_filename = seed.source_filename
        return refined


def refine_with_llm(seed: CompetencyReview, excerpt_pack: str) -> CompetencyReview:
    return OpenAiRefiner().refine(seed, excerpt_pack)
