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

Docker (includes LaTeX for PDF export): `docker compose up --build`

No API key? Use a local model: `LLM_PROVIDER=ollama`, `LLM_MODEL=llama3.1`.

## Evaluation

### Dataset: 5 real CV / offer pairs
Real pairs live in `data/private/pairs/` (git-ignored: they contain personal data), one folder per pair:

```
data/private/pairs/01_ml_intern/
  cv.pdf            one or more candidate files (pdf, docx, md, txt, tex)
  job.txt           the offer text, or job.url with its URL (fetched once, cached)
  candidate.json    optional: {"name": "...", "github": "username"}
```

Without real pairs, every command falls back to `--sample`: the fictional sample CV against the 5 sample offers (one in French).

### Proof run
```bash
python scripts/run_proof.py            # full system on every pair
```
For each pair, writes the CV (md/tex/pdf), cover letter, interview questions, match report and a full trace (evidence, every Critic pass) to `outputs/proof/<pair>/`, plus a summary table in `outputs/proof/index.md`.

### Ablation
```bash
python -m eval.ablation --repeats 2     # add --variants ... low_recall for the retrieval ablation
python -m eval.analysis                 # rebuild the report from eval/results.jsonl
```

| Variant | What runs |
|---|---|
| `single_pass` | Writer only, no Critic (baseline) |
| `critic_once` | one Critic pass: unsupported claims are pruned, no rewrite |
| `full` | full Writer ↔ Critic loop + pruning |
| `low_recall` | full system with 1 evidence chunk per requirement instead of 2 |

All variants are scored by the same evaluator on the returned CV. Each run is appended to `eval/results.jsonl` as soon as it ends, so a run interrupted by a free-tier quota resumes where it stopped (`--fresh` starts over). The analysis writes `eval/results.md` (mean ± std, paired comparison against the baseline with wins/ties/losses and an exact sign test, per-pair tables, score per Critic iteration), `eval/results.csv`, `eval/results.png` and `eval/trajectory.png`.

**Results:** _(paste the summary table from `eval/results.md` here after running)_

### Match and gap report (`gap_analyst`)
`src/agents/gap_analyst.py` classifies every requirement of the offer as **matched** (evidence and in the CV), **in profile, not in CV** (evidence the CV left out), **partial** (only related evidence) or **missing** (no evidence). The classification is deterministic (retrieval similarity plus literal checks), and an optional LLM step adds advice for each gap without ever suggesting a skill the candidate lacks. It reports an importance-weighted **profile match** (what the candidate's material supports) and **CV coverage** (what the generated CV shows).

### Job offer import by URL
`src/ingestion/job_url.py` reads the schema.org `JobPosting` most job boards embed. Otherwise it falls back to the page's main text. Only public http(s) addresses are fetched (redirects are re-checked), with a size cap. Sites that need a login or JavaScript (LinkedIn, for example) are refused with a clear message, so the user pastes the text instead.
```bash
python -m src.ingestion.job_url https://example.com/jobs/123 > job.txt
```

### Evaluation limits
- The Critic is also the evaluator, so the scores measure agreement with the judge rather than ground truth. Use a separate `JUDGE_MODEL` and spot-check `trace.json` by hand.
- With 5 pairs the sign test cannot go below p = 0.0625. Read the results as a consistent trend, not as statistical proof.
- LLM outputs vary between runs: use `--repeats` and report the std.

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
eval/          dataset loader, ablation runner, analysis
scripts/       run_proof.py (proof run on the CV/offer pairs)
data/sample/   fictional, anonymized sample CV and job offers
tests/
```

## Limits and ethics
- The LLM judge can itself be wrong; faithfulness is a strong signal, not a proof. Always review the CV before sending.
- Personal data stays local except for the text sent to the chosen LLM provider. Use `ollama` for fully local processing.
- Do not commit real CVs; `data/private/` is git-ignored.
