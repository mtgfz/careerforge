"""Pure-Python metrics (no LLM, no heavy deps) so they are fast to test."""
from __future__ import annotations

import math
import re

from .schemas import GeneratedCV


# "(E3)", "[E4]", "(source: E7)", "(E1, E2)": evidence citations the LLM sometimes leaks into text
_EVIDENCE_MARKER = re.compile(
    r"\s*[(\[]\s*(?:sources?\s*:?\s*)?E\d+(?:\s*[,;]\s*E\d+)*\s*[)\]]", re.IGNORECASE
)


def _clean(value):
    if isinstance(value, str):
        return _EVIDENCE_MARKER.sub("", value).strip()
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {k: (v if k == "source_ids" else _clean(v)) for k, v in value.items()}
    return value


def strip_evidence_markers(cv: GeneratedCV) -> GeneratedCV:
    """Remove leaked citation markers from every text field; source_ids are kept."""
    return GeneratedCV(**_clean(cv.model_dump()))


def prune_unsupported(
    cv: GeneratedCV,
    bullets: list[str],
    skills: list[str],
    titles: list[str] = (),
    drop_summary: bool = False,
) -> GeneratedCV:
    """Drop what the Critic flagged: bullets, skills and (optionally) the summary. An entry
    whose bullets were all flagged is dropped too; entries that never had bullets are kept.
    A flagged entry title is replaced by a neutral one instead of losing the entry."""
    bad_bullets = {t.strip() for t in bullets}
    bad_skills = {s.strip().lower() for s in skills}
    bad_titles = {t.strip().lower() for t in titles}

    def prune_entries(entries, fallback_title):
        kept = []
        for e in entries:
            remaining = [b for b in e.bullets if b.text.strip() not in bad_bullets]
            if e.bullets and not remaining:
                continue
            title = e.title
            if title.strip().lower() in bad_titles:
                title = e.organization or fallback_title
            kept.append(e.model_copy(update={"bullets": remaining, "title": title}))
        return kept

    return cv.model_copy(
        update={
            "summary": "" if drop_summary else cv.summary,
            "skills": [s for s in cv.skills if s.strip().lower() not in bad_skills],
            "experiences": prune_entries(cv.experiences, "Experience"),
            "projects": prune_entries(cv.projects, "Project"),
        }
    )


_TOKEN = re.compile(r"[a-z0-9+#]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def dedupe_bullets(cv: GeneratedCV, threshold: float = 0.55) -> GeneratedCV:
    """Drop bullets that repeat an earlier one (word overlap >= threshold), experiences first.
    An entry left without bullets is dropped; entries that never had bullets are kept."""
    seen: list[set[str]] = []

    def dedupe_entries(entries):
        kept = []
        for e in entries:
            remaining = []
            for b in e.bullets:
                toks = _tokens(b.text)
                if toks and any(len(toks & s) / len(toks | s) >= threshold for s in seen):
                    continue
                seen.append(toks)
                remaining.append(b)
            if e.bullets and not remaining:
                continue
            kept.append(e.model_copy(update={"bullets": remaining}))
        return kept

    def merge_same_source(entries):
        """Entries whose bullets cite exactly the same evidence describe the same thing."""
        merged: list = []
        by_sources: dict[frozenset, int] = {}
        for e in entries:
            ids = frozenset(i for b in e.bullets for i in b.source_ids)
            if ids and ids in by_sources:
                first = merged[by_sources[ids]]
                merged[by_sources[ids]] = first.model_copy(
                    update={"bullets": [*first.bullets, *e.bullets]}
                )
                continue
            if ids:
                by_sources[ids] = len(merged)
            merged.append(e)
        return merged

    return cv.model_copy(
        update={
            "experiences": merge_same_source(dedupe_entries(cv.experiences)),
            "projects": merge_same_source(dedupe_entries(cv.projects)),
        }
    )


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
