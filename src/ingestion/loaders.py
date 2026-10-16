"""Turn files and GitHub profiles into RawDoc objects."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests

from .. import config


@dataclass
class RawDoc:
    text: str
    source: str


def load_text(text: str, source: str) -> RawDoc:
    return RawDoc(text=text, source=source)


def load_file(path: str | Path) -> RawDoc:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix == ".docx":
        from docx import Document

        doc = Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs)
    elif suffix in {".md", ".txt", ".tex"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {path.name}")
    return RawDoc(text=text, source=path.name)


def load_github(username: str, max_repos: int = 8) -> list[RawDoc]:
    """Pull public non-fork repos (description, languages, README) for a user."""
    headers = {"Accept": "application/vnd.github+json"}
    if config.GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"

    resp = requests.get(
        f"https://api.github.com/users/{username}/repos",
        params={"sort": "updated", "per_page": 30},
        headers=headers,
        timeout=20,
    )
    resp.raise_for_status()
    repos = [r for r in resp.json() if not r.get("fork")][:max_repos]

    docs: list[RawDoc] = []
    for r in repos:
        name = r["name"]
        readme = ""
        try:
            rr = requests.get(
                f"https://raw.githubusercontent.com/{username}/{name}/HEAD/README.md",
                timeout=15,
            )
            if rr.ok:
                readme = rr.text[:4000]
        except requests.RequestException:
            pass
        topics = ", ".join(r.get("topics") or [])
        text = (
            f"GitHub project: {name}\n"
            f"Description: {r.get('description') or 'n/a'}\n"
            f"Main language: {r.get('language') or 'n/a'}\n"
            f"Topics: {topics or 'n/a'}\n"
            f"README:\n{readme}"
        )
        docs.append(RawDoc(text=text, source=f"github:{username}/{name}"))
    return docs
