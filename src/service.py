"""Application service shared by the FastAPI backend and the Streamlit UI."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from . import config
from .ingestion.loaders import RawDoc, load_file, load_github, load_text
from .ingestion.store import KnowledgeBase
from .pipeline import PipelineResult, run_pipeline
from .render.latex import compile_pdf, render_latex, render_markdown
from .schemas import Candidate


def build_kb(
    files: list[Path] | None = None,
    github_user: str = "",
    extra_text: str = "",
) -> KnowledgeBase:
    docs: list[RawDoc] = [load_file(f) for f in (files or [])]
    if github_user.strip():
        docs += load_github(github_user.strip())
    if extra_text.strip():
        docs.append(load_text(extra_text, "pasted_text"))
    if not docs:
        raise ValueError("Provide at least one CV/document, a GitHub user, or pasted text.")
    kb = KnowledgeBase()
    kb.add(docs)
    return kb


def generate_application(
    candidate: Candidate,
    job_text: str,
    files: list[Path] | None = None,
    github_user: str = "",
    extra_text: str = "",
    use_critic: bool = True,
    out_root: str | None = None,
) -> dict:
    """Runs the pipeline, writes outputs to disk, returns a JSON-friendly summary."""
    kb = build_kb(files, github_user, extra_text)
    result: PipelineResult = run_pipeline(kb, job_text, candidate, use_critic=use_critic)

    run_id = uuid.uuid4().hex[:10]
    out_dir = Path(out_root or config.OUTPUT_DIR) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tex = render_latex(result.cv, candidate)
    (out_dir / "cv.tex").write_text(tex, encoding="utf-8")
    (out_dir / "cv.md").write_text(render_markdown(result.cv, candidate), encoding="utf-8")
    (out_dir / "cover_letter.txt").write_text(result.cover_letter, encoding="utf-8")
    (out_dir / "interview_questions.txt").write_text(result.questions, encoding="utf-8")
    pdf = compile_pdf(out_dir / "cv.tex")

    summary = {
        "run_id": run_id,
        "job_title": result.job.title,
        "iterations": result.iterations,
        "scores": result.critique.model_dump(exclude={"feedback"}),
        "history": [
            {"iteration": h["iteration"], **h["critique"].model_dump(exclude={"feedback"})}
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
