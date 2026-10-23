"""Turn files and GitHub profiles into RawDoc objects."""
from __future__ import annotations

import html
import math
import re
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


_ICON_GLYPHS = re.compile(r"[-]")  # icon-font characters (phone, mail...) used in CV templates


def _pdf_text(path: Path) -> str:
    """Extract PDF text. Many CV templates have no space glyphs, so words come out glued
    together with pypdf; pdfplumber rebuilds spaces from the gaps between characters."""
    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            text = "\n".join(page.extract_text(x_tolerance=1.5) or "" for page in pdf.pages)
    except ImportError:
        text = ""
    if not text.strip():
        from pypdf import PdfReader

        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return _ICON_GLYPHS.sub("", text)


def load_file(path: str | Path) -> RawDoc:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _pdf_text(path)
    elif suffix == ".docx":
        from docx import Document

        doc = Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs)
    elif suffix in {".md", ".txt", ".tex"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {path.name}")
    return RawDoc(text=text, source=path.name)


README_MAX_CHARS = 5000
MIN_README_CHARS = 150  # below this (and without a description) a repo carries no information
MAX_CANDIDATES = 30  # most recently updated repos considered before ranking

_README_NAMES = ("README.md", "readme.md", "README.rst")


def _clean_readme(md: str, max_chars: int = README_MAX_CHARS) -> str:
    """Strip HTML, badges, images, link targets and rules so only readable text is embedded."""
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    md = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", md, flags=re.S | re.I)
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)  # images and badges
    md = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", md)  # [text](url) -> text
    md = re.sub(r"<br\s*/?>", "\n", md, flags=re.I)
    md = re.sub(r"</?[A-Za-z][^>]*>", "", md)  # remaining HTML tags
    md = html.unescape(md)
    md = re.sub(r"^[ \t]*([-*_][ \t]*){3,}$", "", md, flags=re.M)  # horizontal rules
    text = "\n".join(line.rstrip() for line in md.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        cut = text[:max_chars]
        text = cut[: cut.rfind("\n")] if "\n" in cut else cut
    return text


def _fetch_readme(username: str, repo: str) -> str:
    for fname in _README_NAMES:
        try:
            rr = requests.get(
                f"https://raw.githubusercontent.com/{username}/{repo}/HEAD/{fname}", timeout=15
            )
        except requests.RequestException:
            continue
        if rr.ok:
            return rr.text
    return ""


def _repo_score(repo: dict, readme: str) -> float:
    """Heuristic: a documented, described, starred project is worth more than a stub."""
    return (
        min(len(readme), 3000) / 3000 * 2
        + (1.0 if (repo.get("description") or "").strip() else 0.0)
        + (0.5 if repo.get("language") else 0.0)
        + 0.3 * math.log1p(repo.get("stargazers_count") or 0)
    )


def github_username(value: str) -> str:
    """Accept a bare username, '@name', or a profile/repo URL (github.com/name/...)."""
    value = value.strip()
    match = re.search(r"github\.com/([^/\s?#]+)", value, flags=re.I)
    return (match.group(1) if match else value).lstrip("@").strip("/")


def load_github(username: str, max_repos: int = 8, include_forks: bool = False) -> list[RawDoc]:
    """Pull the user's best public repos (description, language, cleaned README).

    Repos with no description and (almost) no README are ignored; the rest are ranked and the
    `max_repos` best are kept. Forks are skipped unless `include_forks` is set.
    """
    username = github_username(username)
    headers = {"Accept": "application/vnd.github+json"}
    if config.GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"

    resp = requests.get(
        f"https://api.github.com/users/{username}/repos",
        params={"sort": "updated", "per_page": 100},
        headers=headers,
        timeout=20,
    )
    if resp.status_code == 404:
        raise ValueError(f"GitHub user '{username}' not found")
    if resp.status_code in (403, 429) and resp.headers.get("X-RateLimit-Remaining") == "0":
        raise ValueError("GitHub API rate limit reached: set GITHUB_TOKEN in .env or retry later")
    resp.raise_for_status()
    repos = [r for r in resp.json() if include_forks or not r.get("fork")][:MAX_CANDIDATES]

    scored: list[tuple[float, RawDoc]] = []
    for r in repos:
        name = r["name"]
        readme = _clean_readme(_fetch_readme(username, name))
        description = (r.get("description") or "").strip()
        if len(readme) < MIN_README_CHARS and not description:
            continue  # empty repo
        topics = ", ".join(r.get("topics") or [])
        text = (
            f"GitHub project: {name}\n"
            f"Description: {description or 'n/a'}\n"
            f"Main language: {r.get('language') or 'n/a'}\n"
            f"Topics: {topics or 'n/a'}\n"
            f"README:\n{readme}"
        )
        scored.append((_repo_score(r, readme), RawDoc(text=text, source=f"github:{username}/{name}")))
    scored.sort(key=lambda item: -item[0])  # stable: ties keep the most recently updated first
    return [doc for _, doc in scored[:max_repos]]
