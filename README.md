# CareerForge

**An agentic, fact-grounded CV generator.** Instead of filling a template with a single prompt, CareerForge builds a personal knowledge base from your own material and uses a team of agents to write, critique and fix a CV tailored to each job offer, without inventing experience.

## Architecture

```
 CVs / reports / GitHub ──► Ingestion ──► Vector store (Chroma + multilingual embeddings)
                                                   │
 Job offer ──► Job Analyst ──► Retriever ◄─────────┘
                 (JobProfile)     (evidence per requirement)
                                      │
                                      ▼
                                   Writer ◄──────────── feedback ──────────┐
                                      │                                    │
                                      ▼                                    │
                                   Critic ── faithfulness / ATS / quality ─┤ (max 3 loops)
                                      │ passed
                                      ▼
                      LaTeX → PDF  +  cover letter  +  interview questions
```

Orchestrated with **LangGraph** (conditional loop Writer ↔ Critic), built on **LangChain**.

| Agent | Job |
|---|---|
| Ingestion | Loads PDF/DOCX/MD/TXT and public GitHub repos, chunks and embeds them |
| Job Analyst | Offer → structured `JobProfile` (requirements, importance, ATS keywords) via Pydantic structured output |
| Retriever | RAG: for every requirement, fetches the candidate's strongest evidence chunks |
| Writer | Writes impact-style bullets using **only** retrieved evidence; every bullet cites evidence IDs |
| Critic | Scores the CV and sends actionable feedback back to the Writer |

### Critic metrics
- **Faithfulness**: share of bullets and skills grounded in the evidence. A bullet counts only if it cites valid evidence IDs **and** an LLM judge confirms the evidence supports every claim. Skills must literally appear in the evidence.
- **ATS score**: 50% job keyword coverage + 50% embedding cosine similarity between CV and offer.
- **Quality**: LLM-as-judge rubric (clarity, impact, action verbs, concision).
- **Overall** = 0.4 faithfulness + 0.3 ATS + 0.3 quality. The loop stops when faithfulness and overall pass their thresholds, or after `MAX_ITERATIONS`.

### Design choices
- **Why not RNN/LSTM?** Pretrained transformer embeddings (`paraphrase-multilingual-MiniLM`) capture semantics across French/English CVs without any training; an LSTM would need labeled data and would perform worse. LLMs handle the generation.
- **Contact details are never generated** by the LLM; they come from the user.
- **No invented experience**: grounding + citation check + judge, with an ablation to prove it works (below).

## Quick start

```bash
git clone <your-repo-url> && cd careerforge
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # add your API key (Groq has a free tier)
streamlit run app/streamlit_app.py                     # UI at http://localhost:8501
# or the API:
uvicorn src.api.main:app --reload                      # docs at http://localhost:8000/docs
```

Docker (builds the React app, includes LaTeX for PDF export): `docker compose up --build` then open http://localhost:8000

No API key? Use a local model: `LLM_PROVIDER=ollama`, `LLM_MODEL=llama3.1`.

## Web app (React + FastAPI, one process)

The "Job Application Copilot" UI is a React/TypeScript SPA in `frontend/`, served by FastAPI.

```bash
cd frontend && npm install && npm run build && cd ..   # builds frontend/dist
uvicorn src.api.main:app                               # app + API on http://localhost:8000
# dev mode with hot reload: terminal 1 uvicorn src.api.main:app --reload ; terminal 2 cd frontend && npm run dev
```

Features: live progress (Server-Sent Events), Critic loop chart with rejected claims per iteration, editable CV with
LaTeX re-export (3 templates), cover letter, interview questions, per-step timings, and **replay of a recorded run
without any LLM call** (plan B for the demo).

Record a demo: generate once in the app, then `python scripts/save_demo.py <run_id>` and commit `demo/`.

API: `POST /api/jobs`, `GET /api/jobs/{id}/events` (SSE), `POST /api/runs/{run_id}/render`, `GET /api/replays`,
`POST /api/replay/{run_id}`, `GET /api/config`, `GET /download/{run_id}/{file}`.

Speed knobs (`.env`): `PLATEAU_DELTA` (stop when the score gain is below it), `REJUDGE_AFTER_PRUNE=false` (skip the
second judge call), `PARALLEL_EXTRAS` (letter and questions in parallel), `KB_CACHE_SIZE`, `GITHUB_CACHE_SECONDS`.
Keep `REJUDGE_AFTER_PRUNE=true` for the ablation so the reported scores are re-measured.

### Retrieval quality knobs

Dense retrieval (multilingual MiniLM) picks a pool of candidate passages (`RERANK_POOL`, default 12), which can then be
re-ordered by `HYBRID_SEARCH=true` (BM25 fused with the dense ranking by reciprocal rank fusion) and/or
`RERANKER_MODEL=<cross-encoder>` (a Transformer that attends jointly over requirement + passage).
Both are off by default so the baseline stays comparable; compare with `python -m eval.ablation ... --out eval/<name>`.

## Evaluation (ablation)

```bash
python -m eval.ablation --cv data/sample/sample_cv.md --jobs data/sample/jobs
```

Compares the full system against a single-pass Writer (no Critic) on the sample jobs, scored by the same evaluator. Outputs `eval/results.csv`, `eval/results.md`, `eval/results.png`.

**Results:** _(paste the table from `eval/results.md` here after running)_

## Tests

```bash
pytest -q
```

Unit tests cover the metrics, the LaTeX renderer and the LangGraph loop logic (with stub nodes, no API key needed).

## Project layout

```
src/
  ingestion/   loaders (PDF, DOCX, GitHub) + Chroma knowledge base
  agents/      job_analyst, retriever, writer, critic, extras
  graph.py     LangGraph workflow
  pipeline.py  wiring + PipelineResult
  render/      LaTeX / Markdown rendering, PDF compilation
  api/         FastAPI backend
app/           Streamlit UI
eval/          ablation script
data/sample/   fictional, anonymized sample CV and job offers
tests/
```

## Limits and ethics
- The LLM judge can itself be wrong; faithfulness is a strong signal, not a proof. Always review the CV before sending.
- Personal data stays local except for the text sent to the chosen LLM provider. Use `ollama` for fully local processing.
- Do not commit real CVs; `data/private/` is git-ignored.
