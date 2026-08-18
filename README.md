# AI Technical Competency Review (Use Case 1)

This repository implements **Use Case 1 only** from the AI Engineer Technical Exercise: review a candidate CV for **demonstrated AI technical competence**.

**Use Case 2** (architecture diagram generation from technical notes) is out of scope here and will live in a separate repository.

The system inventories evidence. It does **not** make a hiring, pass/fail, interview, ranking-for-employment, or offer decision.

---

## Quick start (Use Case 1 only)

Python 3.11+. From this repo (not the architecture-diagram app):

```bash
git fetch origin
git checkout cursor/uc1-traceability-16a6
git pull origin cursor/uc1-traceability-16a6

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

python -m pip install --upgrade "pip>=24.2"
python -m pip install -r requirements.txt
```

That install already includes FastAPI, uvicorn, PDF/DOCX ingest, numpy, the OpenAI-compatible client (used for **local Ollama**), and pytest. You do **not** need Datadog or an extra tracing package.

Optional: local LLM (separate install, not pip):

```bash
# other terminal
ollama serve
ollama pull llama3.2
```

Copy `.env.example` to `.env` if your model tag is not `llama3.2`.

Run tests, then the UI:

```bash
python -m pytest
python -m uvicorn cv_reviewer.api:app --port 8000
```

Open http://127.0.0.1:8000 → **Load sample CVs and positions** → upload your resume if you want → **Run assessment** → open **Trace**.

CLI:

```bash
python -m cv_reviewer --pretty --no-llm \
  --cvs sample_cvs/strong_ai_engineer.txt \
  --positions sample_positions/bmc_project_architect_consulting_india.txt
```

Omit `--no-llm` when `ollama serve` is running.

---

## What has been done

### Use Case 1 requirements

| Exercise requirement | How this app meets it |
| --- | --- |
| Review Python, LLMs, embeddings, vector DBs, RAG, ML/DL, AI frameworks, model APIs, AI architecture | Each area is assessed on every CV |
| Other AI technologies found on the CV | Extra terms (LoRA, LangSmith, Docker, …) are listed when present |
| Which competencies are demonstrated | `demonstrated: true` only when experience/projects describe activity |
| Apparent level in each area | `advanced` / `working` / `foundational` / `mentioned_only` / `not_demonstrated` / `insufficient_information` |
| Evidence from the CV | Quotes plus section name and rationale |
| Skills not demonstrated | Listed explicitly |
| Insufficient information | Listed when the text is too thin to judge |
| Mentioned vs demonstrated | Skills-list / “familiar with” = mentioned; built/deployed/trained/evaluated in a job or project = demonstrated |
| No hiring decision | Prompts, API, and output copy forbid hire/reject/interview/pass-fail language |

### User experience (from the exercise)

The app is a small web UI plus API, not a notebook or a single prompt.

| Expected interaction | In this app |
| --- | --- |
| Provide or upload one or more CVs | File picker (PDF, DOCX, TXT, MD), paste, or **Load fictional samples** |
| Provide position descriptions | Same, optional; used to show evidence overlap with a role |
| Run assessment or recommendation | **Run assessment and recommendation** |
| Review results clearly | Table + competency cards |
| See supporting evidence | Expandable quotes per requirement and per AI area |

Position descriptions are **not** a second exercise use case. They are only there so a reviewer can compare CV evidence with a fictional role. Ranking is “how many role requirements have demonstrated CV evidence among the files you uploaded”, not “hire this person”.

### Technical choices

| Choice | Reason |
| --- | --- |
| Python, FastAPI, HTML UI | Primary language from the brief; usable app, not a script-only demo |
| PDF / DOCX / text ingest | Matches “upload a CV” |
| Section-aware chunking | Keeps Skills vs Experience distinct |
| Embeddings + in-memory cosine index | RAG: retrieve CV passages per competency / requirement |
| Hashed n-gram embeddings by default | Deterministic, no model download, tests run offline |
| Optional `sentence-transformers` | Denser semantics if you set `EMBEDDING_BACKEND=sentence-transformers` |
| Heuristic review as the main path | Mention vs demonstration is testable without an API key |
| Local Ollama by default | `LLM_PROVIDER=ollama` talks to `ollama serve` on this machine; no OpenAI key |
| Small fictional samples | The brief says dataset size/realism is not scored |

### Architecture (Clean Architecture + DDD-lite)

The first cut of this repo is **CV competency review**. Domain rules do not depend on FastAPI, PDFs, or OpenAI.

```
interfaces/          HTTP + CLI
application/         ReviewCvService use case + ports (DIP)
domain/              evidence policy, levels, taxonomy, no I/O
infrastructure/      PDF ingest, embeddings, vector index, optional LLM
composition.py       wires adapters into the use case
```

| Principle | How it shows up |
| --- | --- |
| DDD ubiquitous language | `demonstrated`, `mentioned`, `CompetencyArea`, `CompetencyReview` |
| Bounded context | Evidence inventory only — no hiring decision in the domain |
| Dependency inversion | `VectorIndexPort` / `LlmRefinerPort`; numpy/Ollama/OpenAI stay in infrastructure |
| Single responsibility | Policy ≠ retrieval ≠ HTTP |
| Open/closed | New embedder or vector DB implements the port without changing the use case |

Position overlap stays in `matching.py` so it is not inside the competency domain model.

### Sample data (fictional on purpose)

CVs in `sample_cvs/`:

- `strong_ai_engineer.txt` — activity-based AI work (RAG, vectors, APIs, …)
- `keyword_only.txt` — AI words in Skills; job history is Java backend
- `sparse.txt` — almost no AI content

Positions in `sample_positions/`:

- AI Platform Engineer (fictional)
- Backend Services Engineer (fictional)
- Applied ML Research Scientist (fictional)
- `bmc_project_architect_consulting_india.txt` — public BMC **Project Architect / Lead AI Architect** JD ([job 47275](https://jobs.bmc.com/Careers/JobDetail/Project-Architect-Consulting-Services-India/47275)), kept as bullets so you can upload your own CV and compare evidence

To test against your resume: put the file in `sample_cvs/` (or upload it in the UI), click **Load sample CVs and positions**, then **Run**.

---

## How to install

Python 3.11+. Prefer a virtualenv.

```bash
python -m pip install --upgrade "pip>=24.2"
python -m pip install -r requirements.txt
```

If the environment already has packages (for example `uvicorn[standard]==0.34.3`):

```bash
python -m pip install -e ".[dev]" --upgrade-strategy only-if-needed
```

Do not also install `uvicorn[standard]` as a second requirement. This project depends on plain `uvicorn>=0.32,<1`.

Optional semantic embeddings:

```bash
python -m pip install -e ".[dev,semantic]"
export EMBEDDING_BACKEND=sentence-transformers
```

Local LLM refinement uses **Ollama on the same machine as the app** (this is the default):

```bash
# in another terminal
ollama serve
ollama pull llama3.2
```

Copy `.env.example` to `.env` if you want to pin the model:

```bash
LLM_PROVIDER=ollama
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2
```

The UI checkbox is on by default. If Ollama is not running, refinement is skipped and the heuristic review is still returned.

Optional cloud OpenAI instead:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
```

Heuristic-only (no LLM process):

```bash
export LLM_PROVIDER=none
```

---

## How to test

### 1. Automated tests (required check)

```bash
python -m pytest
```

Expect all tests to pass. They cover ingest, mention vs demonstration, full CV reviews, API `/run`, CV×position overlap, and domain-layer import boundaries.

### 2. UI demo (Use Case 1)

```bash
python -m uvicorn cv_reviewer.api:app --port 8000
```

Open http://127.0.0.1:8000

1. Click **Load sample CVs and positions** (includes the BMC Project Architect JD).
2. Optionally upload **your** CV in the CV file picker (PDF/DOCX/TXT). If the picker has a file, that run uses it instead of the fictional CVs.
3. Click **Run assessment and recommendation**.
4. **Evidence table** — CV × position counts (demonstrated / mentioned only / not found). Open the BMC row for your CV.
5. Expand a row — quotes from the CV.
6. **Competency reviews** — the nine AI areas with levels and evidence.
7. **Recommendations** — order of uploaded documents by demonstrated overlap only.

What you should see on the samples:

- **Alex Rivera** (strong CV) vs **AI Platform Engineer**: most requirements **demonstrated**, with project/job quotes.
- **Jordan Lee** (keyword CV) vs AI role: AI terms **mentioned only**; vs **Backend Services Engineer**: Java REST/SQL work can show **demonstrated** evidence.
- **Sam Patel** (sparse CV): AI areas **not demonstrated**; position requirements mostly **not found**.

You can also upload your own PDF/DOCX/TXT CVs and position files instead of samples. For the BMC JD, drop your resume in `sample_cvs/` or use the CV file picker after loading samples.

### 3. CLI

Single CV competency review:

```bash
python -m cv_reviewer sample_cvs/strong_ai_engineer.txt --pretty
python -m cv_reviewer sample_cvs/strong_ai_engineer.txt --pretty --no-llm
python -m cv_reviewer sample_cvs/keyword_only.txt --pretty --no-llm
python -m cv_reviewer sample_cvs/sparse.txt --pretty --no-llm
```

CVs plus positions:

```bash
python -m cv_reviewer --no-llm --pretty \
  --cvs sample_cvs/strong_ai_engineer.txt sample_cvs/keyword_only.txt sample_cvs/sparse.txt \
  --positions sample_positions/ai_platform_engineer.txt sample_positions/backend_services_engineer.txt sample_positions/ml_research_scientist.txt
```

Omit `--no-llm` to refine with local Ollama. `--no-llm` keeps the run offline and deterministic.

### 4. API smoke checks

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/samples | python -m json.tool | head
```

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | UI |
| GET | `/health` | Liveness |
| GET | `/samples` | Fictional CVs and positions |
| POST | `/run` | JSON CVs + positions → full assessment |
| POST | `/run-files` | Multipart file upload |
| POST | `/review` | One CV file → competency review |
| POST | `/review-text` | One CV as JSON |

---

## How assessment works

```
CV / position files
  → extract text
  → chunk by CV section
  → embed + store in an in-memory vector index
  → retrieve passages per AI area and per position requirement
  → classify each excerpt as demonstrated or mentioned
  → structured JSON shown in the UI
```

**Demonstrated** means the retrieved excerpt is from experience/projects (or similar) and describes doing the work. **Mentioned** means the term appears (often in a skills list) without that activity.

**Coverage ratio** = demonstrated requirements ÷ total requirements for that position. It is an evidence count, not a hiring score.

---

## Traceability, Ollama, guardrails, and evaluation

There is **no Datadog agent**. Datadog is commercial SaaS. Every review has an in-app **Trace** tab: chunk → retrieve → classify → optional LLM → guardrail (decision-language strip + quote grounding).

Ollama is **optional**. Default `LLM_PROVIDER=ollama` talks to `ollama serve` on this machine. The heuristic review always runs first. If Ollama is down, refinement is skipped.

```bash
python -m cv_reviewer.evaluation
```

Gold labels: `src/cv_reviewer/evaluation/matrix.py`. Do not gold-label a hiring outcome.

If you later want vendor-style APM: OpenTelemetry → Jaeger/Grafana Tempo. For LLM spans: Langfuse or Arize Phoenix.


---

## Limitations

- A CV is not verified employment history. Missing text is not proof of missing skill.
- Default embeddings are lexical; paraphrases without shared tokens can be missed.
- Position parsing works best with bullet requirements.
- Optional LLM output can drift; the UI still shows the underlying quotes. Ollama must run on the **same host** as uvicorn — a remote agent cannot reach `ollama serve` on your laptop.

## If this were production

- Persistent vector store, auth, and audit logs
- A larger labelled mention-vs-demonstration set than the three fictional samples
- Human review before any HR use
- Better PDF layout extraction; optional OpenTelemetry export of the in-app traces

## Presentation demo script

1. `python -m pytest`
2. Start `ollama serve` and the UI, load samples, run with the Ollama checkbox on.
3. Open Alex Rivera × AI Platform Engineer and show a **demonstrated** quote.
4. Open the **Trace** tab and walk the chunk → retrieve → classify steps.
5. Open Jordan Lee’s competency review and show AI skills as **mentioned only**.
6. State clearly: the app finds evidence; it does not decide hiring.
