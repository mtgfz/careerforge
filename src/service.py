"""Application service shared by the FastAPI backend and the Streamlit UI."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from . import config
from .ingestion.loaders import RawDoc, load_file, load_github, load_text
from .ingestion.store import KnowledgeBase
from .llm import quota_error
from .pipeline import PipelineResult, run_pipeline
from .render.latex import compile_pdf, render_latex, render_markdown
from .render.latex import TEMPLATES
from .schemas import Candidate, GeneratedCV

log = logging.getLogger(__name__)

DEMO_DIR = Path("demo")  # a recorded run committed to the repo (see scripts/save_demo.py)
_KB_CACHE: "OrderedDict[str, KnowledgeBase]" = OrderedDict()  # embedded documents, LRU
_GH_CACHE: dict[str, tuple[float, list[RawDoc]]] = {}  # github user -> (fetched at, docs)
_CACHE_LOCK = threading.Lock()


def run_dir(run_id: str) -> Path:
    """Folder holding the files of a run ('demo' = the recorded demo shipped with the repo)."""
    return DEMO_DIR / "run" if run_id == "demo" else Path(config.OUTPUT_DIR) / run_id


def _github_docs(user: str) -> list[RawDoc]:
    """GitHub profile text, cached for a few minutes (repos change rarely, the API is slow)."""
    now = time.time()
    with _CACHE_LOCK:
        hit = _GH_CACHE.get(user.lower())
        if hit and now - hit[0] < config.GITHUB_CACHE_SECONDS:
            return hit[1]
    docs = load_github(user)
    with _CACHE_LOCK:
        _GH_CACHE[user.lower()] = (now, docs)
    return docs


_HIDDEN = {  # internal fields, not for the UI tables
    "feedback", "unsupported_bullets", "unsupported_skills", "unsupported_titles", "summary_supported",
}


def build_kb(
    files: list[Path] | None = None,
    github_user: str = "",
    extra_text: str = "",
) -> KnowledgeBase:
    return build_kb_cached(files, github_user, extra_text)[0]


def build_kb_cached(
    files: list[Path] | None = None,
    github_user: str = "",
    extra_text: str = "",
) -> tuple[KnowledgeBase, bool]:
    """Returns (knowledge base, from_cache). Same documents => no re-chunking, no re-embedding."""
    docs: list[RawDoc] = [load_file(f) for f in (files or [])]
    if github_user.strip():
        docs += _github_docs(github_user.strip())
    if extra_text.strip():
        docs.append(load_text(extra_text, "pasted_text"))
    if not docs:
        raise ValueError("Provide at least one CV/document, a GitHub user, or pasted text.")

    h = hashlib.sha256()
    for d in sorted(docs, key=lambda d: (d.source, d.text)):
        h.update(d.source.encode())
        h.update(d.text.encode())
    key = h.hexdigest()
    with _CACHE_LOCK:
        if key in _KB_CACHE:
            _KB_CACHE.move_to_end(key)
            return _KB_CACHE[key], True
    kb = KnowledgeBase()
    kb.add(docs)
    with _CACHE_LOCK:
        _KB_CACHE[key] = kb
        while len(_KB_CACHE) > config.KB_CACHE_SIZE:
            _KB_CACHE.popitem(last=False)
    return kb, False


def generate_application(
    candidate: Candidate,
    job_text: str,
    files: list[Path] | None = None,
    github_user: str = "",
    extra_text: str = "",
    use_critic: bool = True,
    out_root: str | None = None,
    template: str = "classic",
    on_event=None,
) -> dict:
    """Runs the pipeline, writes outputs to disk, returns a JSON-friendly summary."""

    def emit(event: dict) -> None:
        if on_event:
            on_event(event)

    emit({"type": "start", "node": "ingest", "iteration": 0})
    t0 = time.perf_counter()
    kb, kb_cached = build_kb_cached(files, github_user, extra_text)
    ingest_s = round(time.perf_counter() - t0, 2)
    emit({"type": "done", "node": "ingest", "iteration": 0, "seconds": round(ingest_s, 1),
          "cached": kb_cached})
    try:
        result: PipelineResult = run_pipeline(
            kb, job_text, candidate, use_critic=use_critic, on_event=on_event
        )
    except Exception as e:
        quota = quota_error(e)  # also catches 429s from the unretried free-text calls
        if quota:
            raise quota from e
        raise

    run_id = uuid.uuid4().hex[:10]
    out_dir = Path(out_root or config.OUTPUT_DIR) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    emit({"type": "start", "node": "render", "iteration": 0})
    t0 = time.perf_counter()
    tex = render_latex(result.cv, candidate, template)
    (out_dir / "cv.tex").write_text(tex, encoding="utf-8")
    (out_dir / "cv.json").write_text(result.cv.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / "cv.md").write_text(render_markdown(result.cv, candidate), encoding="utf-8")
    (out_dir / "cover_letter.txt").write_text(result.cover_letter, encoding="utf-8")
    (out_dir / "interview_questions.txt").write_text(result.questions, encoding="utf-8")
    pdf = compile_pdf(out_dir / "cv.tex")
    timings = {"ingest": ingest_s, **result.timings, "render": round(time.perf_counter() - t0, 2)}
    emit({"type": "done", "node": "render", "iteration": 0, "seconds": round(timings["render"], 1)})

    summary = {
        "run_id": run_id,
        "job_title": result.job.title,
        "template": template,
        "best_iteration": result.best_iteration,
        "timings": timings,
        "kb_chunks": kb.n_chunks,
        "kb_cached": kb_cached,
        "cv": result.cv.model_dump(),
        "candidate": candidate.model_dump(),
        "iterations": result.iterations,
        "scores": result.critique.model_dump(exclude=_HIDDEN),
        "pruned": result.pruned,
        "keywords": result.job.keywords,
        "evidence": {
            cid: {"text": c.text, "source": c.source} for cid, c in result.evidence.chunks.items()
        },
        "history": [
            {"iteration": h["iteration"], **h["critique"].model_dump(exclude=_HIDDEN)}
            for h in result.history
        ],
        "files": sorted(p.name for p in out_dir.iterdir()),
        "pdf": pdf is not None,
    }
    (out_dir / "run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        **summary,
        "out_dir": str(out_dir),
        "markdown": (out_dir / "cv.md").read_text(encoding="utf-8"),
        "cover_letter": result.cover_letter,
        "questions": result.questions,
    }


def rerender(run_id: str, cv: dict, candidate: dict, template: str = "classic") -> dict:
    """Re-export a CV the user edited by hand (cv_edited.tex / .md / .pdf next to the original)."""
    out_dir = run_dir(run_id)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"Unknown run '{run_id}'")
    if template not in TEMPLATES:
        template = "classic"
    cv_obj = GeneratedCV.model_validate(cv)
    cand = Candidate.model_validate(candidate)
    (out_dir / "cv_edited.tex").write_text(render_latex(cv_obj, cand, template), encoding="utf-8")
    (out_dir / "cv_edited.md").write_text(render_markdown(cv_obj, cand), encoding="utf-8")
    (out_dir / "cv_edited.json").write_text(cv_obj.model_dump_json(indent=2), encoding="utf-8")
    pdf = compile_pdf(out_dir / "cv_edited.tex")
    return {
        "run_id": run_id,
        "template": template,
        "pdf": pdf is not None,
        "files": sorted(p.name for p in out_dir.iterdir() if p.name.startswith("cv_edited")),
        "markdown": (out_dir / "cv_edited.md").read_text(encoding="utf-8"),
    }
