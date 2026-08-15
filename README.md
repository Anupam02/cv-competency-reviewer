# CV competency and position evidence

A small Python web application that:

1. Reviews **AI technical competencies** evidenced in one or more CVs.
2. Compares those CVs with **position descriptions** and ranks evidence overlap.

It does **not** make a hiring, pass/fail, interview, ranking-for-employment, or offer decision. “Recommendation” here means: among the documents you uploaded, which pairs have more **demonstrated** requirement evidence.

The sample CVs and positions are fictional and small on purpose.

## User flow

1. Open the UI.
2. Upload one or more CVs, or click **Load fictional samples**.
3. Provide one or more position descriptions the same way.
4. Run **assessment and recommendation**.
5. Inspect:
   - a CV × position table (demonstrated / mentioned only / not found)
   - quotes from the CV that support each requirement
   - the nine AI competency areas per CV

```bash
python -m pip install --upgrade "pip>=24.2"
python -m pip install -r requirements.txt
python -m uvicorn cv_reviewer.api:app --port 8000
```

Open http://127.0.0.1:8000

## What is assessed

For each CV, the nine required AI areas plus any extra AI technologies found in the text:

- Python
- Large Language Models (LLMs)
- Embeddings
- Vector databases
- Retrieval-Augmented Generation (RAG)
- Machine Learning / Deep Learning
- AI frameworks and libraries
- Model integration and APIs
- AI solution architecture

**Mentioned vs demonstrated.** A skills-list bullet is a mention. The same technology described in Experience or Projects with activity (built, deployed, trained, evaluated, integrated) is demonstrated.

## Architecture

```
CV / JD files
    → ingest (PDF, DOCX, TXT)
    → section-aware chunking
    → embeddings + in-memory cosine index
    → retrieve excerpts (RAG)
    → classify evidence (demonstrated | mentioned | not found)
    → structured JSON + HTML UI
```

| Choice | Why |
| --- | --- |
| Python + FastAPI + HTML | One language, a real UI, no notebook-only demo |
| Hashed n-gram embeddings by default | Deterministic, no model download, tests run offline |
| Optional sentence-transformers | Swap denser semantics with `EMBEDDING_BACKEND=sentence-transformers` |
| In-memory vector index | Enough for a handful of CVs; same add/query shape as FAISS/Chroma |
| Heuristic review as the primary path | Mention vs demonstration is a rule we can explain and test |
| Optional LLM refinement | Only rewrites the structured review from retrieved excerpts |
| Coverage ratio, not a “fit score” | Avoids pretending the model can decide employment |

## CLI

```bash
python -m cv_reviewer --cvs sample_cvs/*.txt --positions sample_positions/*.txt --pretty --no-llm
python -m cv_reviewer sample_cvs/strong_ai_engineer.txt --pretty --no-llm
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | UI |
| GET | `/health` | Liveness |
| GET | `/samples` | Fictional CVs and positions |
| POST | `/run` | JSON: `{ "cvs": [...], "positions": [...] }` |
| POST | `/run-files` | Multipart upload of CV and position files |
| POST | `/review` | Single CV file, competency review only |
| POST | `/review-text` | Single CV as JSON |

## Tests

```bash
python -m pytest
```

## Limitations (honest)

- A CV is not verified work. Missing text is not proof the person cannot do the work.
- Hashed embeddings are lexical; similar phrasing without shared tokens can be missed.
- Requirement parsing expects bullet lists. Free-prose JDs are chunked more coarsely.
- Coverage ratio can be gamed by short JDs or keyword stuffing.
- Optional LLM output can still drift; the UI always shows the underlying quotes.

## If this were production

- Persistent vector DB, access control, and audit logs
- Human review required before any downstream HR use
- Evaluation set of labelled mention vs demonstration examples
- Bias and completeness checks (gaps in CVs that reflect formatting, not skill)
- Stronger PDF layout extraction
- Separate indexing from query, plus tracing of every retrieval

## Presentation notes

Be ready to demo **Load fictional samples → Run**, open one CV × AI Platform Engineer row, and show a demonstrated quote versus the keyword-only CV (AI terms listed, Java work described). Then explain why the backend position picks up the Java REST API evidence and why that still is not a hiring decision.
