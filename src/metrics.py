"""Pure-Python metrics (no LLM, no heavy deps) so they are fast to test."""
from __future__ import annotations

import math
import re

from .schemas import GeneratedCV


def _pattern(term: str) -> re.Pattern:
    # Word-ish boundaries that keep C++, C#, .NET, Node.js intact
    return re.compile(r"(?<![\w+#])" + re.escape(term.lower().strip()) + r"(?![\w+#])")


def contains_term(text: str, term: str) -> bool:
    if not term.strip():
        return False
    return _pattern(term).search(text.lower()) is not None


def keyword_coverage(text: str, keywords: list[str]) -> tuple[float, list[str]]:
    """Fraction of job keywords present in the text, plus the missing ones."""
    keywords = [k for k in keywords if k.strip()]
    if not keywords:
        return 1.0, []
    missing = [k for k in keywords if not contains_term(text, k)]
    return 1 - len(missing) / len(keywords), missing


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def all_bullets(cv: GeneratedCV):
    for entry in [*cv.experiences, *cv.projects]:
        yield from entry.bullets


def cv_to_text(cv: GeneratedCV) -> str:
    parts = [cv.headline, cv.summary, ", ".join(cv.skills)]
    for entry in [*cv.experiences, *cv.projects]:
        parts.append(f"{entry.title} {entry.organization}")
        parts.extend(b.text for b in entry.bullets)
    parts.extend(cv.education)
    parts.extend(cv.languages)
    return "\n".join(p for p in parts if p)


def valid_citation_flags(cv: GeneratedCV, valid_ids: set[str]) -> list[bool]:
    """One flag per bullet: it cites at least one id and all cited ids exist."""
    return [
        bool(b.source_ids) and all(i in valid_ids for i in b.source_ids)
        for b in all_bullets(cv)
    ]


def skill_grounded_flags(skills: list[str], evidence_text: str) -> list[bool]:
    return [contains_term(evidence_text, s) for s in skills]
