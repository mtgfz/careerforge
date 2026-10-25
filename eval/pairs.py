"""Evaluation dataset: CV / job-offer pairs.

Real pairs live in data/private/pairs/ (git-ignored, they contain personal data):

    data/private/pairs/
      01_ml_intern/
        cv.pdf               one or more candidate files (pdf, docx, md, txt, tex)
        job.txt              the offer text ...
        job.url              ... or its URL (fetched once, cached in job.fetched.txt)
        candidate.json       optional: {"name": "...", "github": "username"}

Without real pairs, `sample_pairs()` pairs the fictional sample CV with every sample offer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PAIRS_DIR = Path("data/private/pairs")
SAMPLE_CV = Path("data/sample/sample_cv.md")
SAMPLE_JOBS = Path("data/sample/jobs")

CV_SUFFIXES = {".pdf", ".docx", ".md", ".txt", ".tex"}
JOB_FILES = {"job.txt", "job.url", "job.fetched.txt", "candidate.json"}


@dataclass
class Pair:
    id: str
    cv_files: list[Path]
    job_text: str
    job_source: str  # file path or URL
    candidate_name: str = "Candidate"
    github_user: str = ""
    extra: dict = field(default_factory=dict)


def _job_from_dir(d: Path) -> tuple[str, str]:
    txt, url_file, cache = d / "job.txt", d / "job.url", d / "job.fetched.txt"
    if txt.exists():
        return txt.read_text(encoding="utf-8"), str(txt)
    if url_file.exists():
        url = url_file.read_text(encoding="utf-8").strip()
        if not cache.exists():
            from src.ingestion.job_url import import_job_from_url

            cache.write_text(import_job_from_url(url).text, encoding="utf-8")
        return cache.read_text(encoding="utf-8"), url
    raise FileNotFoundError(f"{d}: needs job.txt or job.url")


def load_pair(d: Path) -> Pair:
    d = Path(d)
    meta = json.loads((d / "candidate.json").read_text("utf-8")) if (d / "candidate.json").exists() else {}
    cv_files = sorted(
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in CV_SUFFIXES and p.name not in JOB_FILES
    )
    if not cv_files and not meta.get("github"):
        raise FileNotFoundError(f"{d}: no CV file (pdf, docx, md, txt, tex)")
    job_text, source = _job_from_dir(d)
    return Pair(
        id=d.name,
        cv_files=cv_files,
        job_text=job_text,
        job_source=source,
        candidate_name=meta.get("name", "Candidate"),
        github_user=meta.get("github", ""),
        extra={k: v for k, v in meta.items() if k not in ("name", "github")},
    )


def load_pairs(root: Path | str = DEFAULT_PAIRS_DIR, only: list[str] | None = None) -> list[Pair]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Pairs directory not found: {root}")
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if only:
        dirs = [d for d in dirs if d.name in only]
    return [load_pair(d) for d in dirs]


def sample_pairs(cv: Path = SAMPLE_CV, jobs_dir: Path = SAMPLE_JOBS) -> list[Pair]:
    return [
        Pair(
            id=job.stem,
            cv_files=[Path(cv)],
            job_text=job.read_text(encoding="utf-8"),
            job_source=str(job),
            candidate_name="Sample Candidate",
        )
        for job in sorted(Path(jobs_dir).glob("*.txt"))
    ]


def resolve_pairs(pairs_dir: str | None = None, sample: bool = False) -> list[Pair]:
    """--sample, an explicit --pairs dir, or data/private/pairs if it exists, else the samples."""
    if sample:
        return sample_pairs()
    if pairs_dir:
        return load_pairs(pairs_dir)
    if DEFAULT_PAIRS_DIR.is_dir() and any(DEFAULT_PAIRS_DIR.iterdir()):
        return load_pairs(DEFAULT_PAIRS_DIR)
    print(f"[eval] {DEFAULT_PAIRS_DIR} is empty: using the fictional sample pairs")
    return sample_pairs()


def build_kb(pair: Pair, embeddings=None):
    from src.ingestion.loaders import load_file, load_github
    from src.ingestion.store import KnowledgeBase

    docs = [load_file(p) for p in pair.cv_files]
    if pair.github_user:
        docs += load_github(pair.github_user)
    kb = KnowledgeBase(embeddings=embeddings)
    kb.add(docs)
    return kb
