# CV AI Competency Reviewer

Python application that reviews a candidate CV for **demonstrated AI technical competence**.

It inventories evidence. It does **not** make a hiring, pass/fail, interview, ranking, or employment decision.

## Use Case 1 — AI Technical Competency Review

Given a CV (PDF, DOCX, TXT, or Markdown), the application reviews these areas:

- Python
- Large Language Models (LLMs)
- Embeddings
- Vector databases
- Retrieval-Augmented Generation (RAG)
- Machine Learning / Deep Learning
- AI frameworks and libraries
- Model integration and APIs
- AI solution architecture
- Other relevant AI technologies identified from the CV

The structured review explains:

- which AI competencies are demonstrated
- the apparent level in each area
- the CV evidence supporting each assessment
- skills that are not demonstrated
- areas with insufficient information

**Mentioned vs demonstrated.** A technology listed under Skills is treated as a mention. The same technology described in Experience or Projects with activity (built, deployed, trained, evaluated, integrated) is treated as demonstrated.

## How it works

1. **Ingest** CV text from PDF / DOCX / plain text.
2. **Chunk** by detected sections (Experience, Skills, Projects, …).
3. **Embed** chunks (hashed n-gram vectors by default; optional `sentence-transformers`).
4. **Retrieve** with an in-memory cosine vector index — one query set per competency area (RAG).
5. **Assess** each area from retrieved excerpts, classifying evidence type and level.
6. **Optional LLM refinement** if `OPENAI_API_KEY` is set. The model may only use retrieved excerpts and must not produce an employment decision.

Default mode needs no API key so the pipeline is testable and reproducible.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip>=24.2"
python -m pip install -r requirements.txt
```

If you already have packages in the environment (for example `uvicorn[standard]==0.34.3`), keep that version instead of letting pip backtrack:

```bash
python -m pip install -e ".[dev]" --upgrade-strategy only-if-needed
```

Do not install `uvicorn` and `uvicorn[standard]` as separate requirements. This project depends on plain `uvicorn>=0.32,<1`, which is compatible with 0.34.3.

Optional semantic embeddings:

```bash
pip install -e ".[dev,semantic]"
export EMBEDDING_BACKEND=sentence-transformers
```

Optional LLM refinement (OpenAI-compatible):

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
# export OPENAI_BASE_URL=https://api.openai.com/v1
```

## CLI

```bash
python -m cv_reviewer sample_cvs/strong_ai_engineer.txt --pretty --no-llm
python -m cv_reviewer sample_cvs/keyword_only.txt --pretty --no-llm
```

## HTTP API and UI

```bash
uvicorn cv_reviewer.api:app --reload --port 8000
```

Open http://127.0.0.1:8000

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Upload UI |
| GET | `/health` | Liveness |
| POST | `/review` | File upload review |
| POST | `/review-text` | JSON `{ "cv_text": "..." }` |
| POST | `/ask` | Retrieve CV excerpts for a factual question (no hiring questions) |

## Tests

```bash
pytest
```

Sample CVs:

- `sample_cvs/strong_ai_engineer.txt` — activity-based evidence across the required areas
- `sample_cvs/keyword_only.txt` — AI terms listed, backend work described in Java
- `sample_cvs/sparse.txt` — almost no AI content

## Design choices

- **No employment decision** is encoded in prompts, API validation, and output sanitisation.
- **Retrieval before generation** so assessments are grounded in CV passages.
- **Heuristic review is first-class**, not a degraded mode: it distinguishes skills lists from project evidence even without an LLM.
- **Hashable embeddings** keep CI deterministic; sentence-transformers can be swapped in via `EMBEDDING_BACKEND`.

Only Use Case 1 was specified in the request. The `/ask` endpoint is supporting evidence retrieval over the same index, still without employment decisions.
