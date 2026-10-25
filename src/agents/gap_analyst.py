"""Gap analyst agent: requirement-by-requirement match between the candidate, the CV and the job.

For every requirement of the JobProfile it decides, from the retrieved evidence and the
generated CV, whether the requirement is:

    matched    supported by the evidence AND visible in the CV
    not_shown  supported by the evidence but absent from the CV (a tailoring opportunity)
    partial    only weakly related evidence (adjacent skill, low similarity)
    missing    no evidence at all (a real gap: never claim it, learn it or address it)

The classification is deterministic (retrieval similarity + literal term checks), so the scores
are reproducible and testable. An optional LLM step adds honest advice for each gap.
"""
from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..metrics import contains_term, cv_to_text, keyword_coverage
from ..schemas import EvidencePack, GeneratedCV, JobProfile, Requirement

# Cosine relevance of the best evidence chunk (same scale as config.MIN_RELEVANCE).
STRONG_RELEVANCE = 0.5
WEAK_RELEVANCE = 0.3

Status = Literal["matched", "not_shown", "partial", "missing"]
PROFILE_CREDIT = {"matched": 1.0, "not_shown": 1.0, "partial": 0.5, "missing": 0.0}
STATUS_LABEL = {
    "matched": "Matched",
    "not_shown": "In profile, not in CV",
    "partial": "Partial",
    "missing": "Missing",
}


class GapItem(BaseModel):
    requirement: str
    kind: str = "other"
    importance: int = 3
    status: Status
    evidence_score: float = 0.0
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_excerpt: str = ""
    in_cv: bool = False
    advice: str = ""


class RequirementAdvice(BaseModel):
    requirement: str = Field(description="Requirement name, copied exactly")
    advice: str = Field(description="One or two concrete, honest sentences")


class GapAdvice(BaseModel):
    """Raw LLM output."""

    advice: list[RequirementAdvice]
    summary: str = Field(description="2-3 sentences: overall fit and the most important next step")


class GapReport(BaseModel):
    job_title: str
    company: str = ""
    profile_match: float = Field(description="Importance-weighted fit of the candidate's material")
    cv_coverage: float = Field(description="Importance-weighted share of requirements the CV shows")
    keyword_coverage: float = 1.0
    missing_keywords: list[str] = Field(default_factory=list)
    items: list[GapItem] = Field(default_factory=list)
    summary: str = ""

    def by_status(self, status: str) -> list[GapItem]:
        return [i for i in self.items if i.status == status]

    def counts(self) -> dict[str, int]:
        return {s: len(self.by_status(s)) for s in STATUS_LABEL}

    def to_markdown(self) -> str:
        head = f"# Match report: {self.job_title}" + (f" at {self.company}" if self.company else "")
        lines = [
            head,
            "",
            f"- **Profile match**: {self.profile_match:.0%} (what your material supports)",
            f"- **CV coverage**: {self.cv_coverage:.0%} (what the generated CV shows)",
            f"- **ATS keyword coverage**: {self.keyword_coverage:.0%}",
            "",
        ]
        if self.summary:
            lines += [self.summary, ""]
        lines += [
            "| Requirement | Importance | Status | Evidence | Advice |",
            "|---|---|---|---|---|",
        ]
        for i in self.items:
            ev = f"{', '.join(i.evidence_ids)} ({i.evidence_score:.2f})" if i.evidence_ids else "none"
            lines.append(
                f"| {_cell(i.requirement)} | {i.importance} | {STATUS_LABEL[i.status]} | {ev} "
                f"| {_cell(i.advice)} |"
            )
        if self.missing_keywords:
            lines += ["", "**Job keywords absent from the CV:** " + ", ".join(self.missing_keywords)]
        return "\n".join(lines) + "\n"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _related_keywords(req: Requirement, keywords: list[str]) -> list[str]:
    """Job keywords mentioned inside the requirement ('Python (pandas)' -> Python, pandas)."""
    return [k for k in keywords if k.strip() and contains_term(req.name, k)]


def _mentions(text: str, req: Requirement, related: list[str]) -> bool:
    return contains_term(text, req.name) or any(contains_term(text, k) for k in related)


def classify(
    job: JobProfile,
    evidence: EvidencePack,
    cv: GeneratedCV | None = None,
    kb=None,
) -> list[GapItem]:
    """Deterministic requirement-by-requirement classification (no LLM)."""
    evidence_text = evidence.text()
    cv_text = cv_to_text(cv) if cv else ""
    cited = {i for e in ([*cv.experiences, *cv.projects] if cv else []) for b in e.bullets
             for i in b.source_ids}
    items: list[GapItem] = []
    for req in sorted(job.requirements, key=lambda r: -r.importance):
        related = _related_keywords(req, job.keywords)
        ids = list(evidence.by_requirement.get(req.name, []))
        chunks = [evidence.chunks[i] for i in ids if i in evidence.chunks]
        score = max((c.score for c in chunks), default=0.0)
        excerpt = chunks[0].text if chunks else ""
        if not chunks and kb is not None:
            # Requirement beyond MAX_REQUIREMENTS: it was never retrieved, search it now.
            hits = kb.search(req.name, k=1)
            if hits:
                excerpt, _, score = hits[0]
                evidence_text += "\n" + excerpt

        literal = _mentions(evidence_text, req, related)
        supported = literal or score >= STRONG_RELEVANCE
        in_cv = bool(cv) and (
            _mentions(cv_text, req, related) or (score >= STRONG_RELEVANCE and bool(cited & set(ids)))
        )
        if supported:
            status: Status = "matched" if in_cv else "not_shown"
        elif score >= WEAK_RELEVANCE:
            status = "partial"
        else:
            status = "missing"
        items.append(
            GapItem(
                requirement=req.name,
                kind=req.kind,
                importance=req.importance,
                status=status,
                evidence_score=round(score, 3),
                evidence_ids=ids,
                evidence_excerpt=excerpt[:240],
                in_cv=in_cv,
            )
        )
    return items


def _weighted(items: list[GapItem], credit) -> float:
    total = sum(i.importance for i in items)
    return round(sum(i.importance * credit(i) for i in items) / total, 3) if total else 1.0


ADVICE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a career coach reviewing how a candidate fits a job offer. For each "
            "requirement listed, give one or two concrete sentences of advice, in {language}:\n"
            "- not_shown: the candidate HAS this (see evidence); say which fact to put forward.\n"
            "- partial: name the adjacent experience to highlight and what is still lacking.\n"
            "- missing: never suggest claiming it; suggest a short project, course or how to "
            "address it honestly in an interview or cover letter.\n"
            "Then write a 2-3 sentence summary of the overall fit. Be honest and specific.",
        ),
        (
            "human",
            "JOB: {title}\n\nREQUIREMENTS TO ADVISE ON\n{gaps}\n\nMATCHED: {matched}",
        ),
    ]
)


def _advise(job: JobProfile, items: list[GapItem], llm, fallback_llm=None) -> GapAdvice | None:
    from ..llm import invoke_structured

    gaps = [i for i in items if i.status != "matched"]
    if not gaps:
        return None
    lines = "\n".join(
        f"- {i.requirement} [status={i.status}, importance={i.importance}] "
        f"evidence: {i.evidence_excerpt[:160] or 'none'}"
        for i in gaps
    )
    return invoke_structured(
        ADVICE_PROMPT,
        GapAdvice,
        {
            "language": job.language,
            "title": job.title,
            "gaps": lines,
            "matched": ", ".join(i.requirement for i in items if i.status == "matched") or "none",
        },
        llm,
        fallback=fallback_llm,
    )


def analyze_gaps(
    job: JobProfile,
    evidence: EvidencePack,
    cv: GeneratedCV | None = None,
    kb=None,
    advise: bool = True,
    llm=None,
    fallback_llm=None,
) -> GapReport:
    items = classify(job, evidence, cv, kb)
    coverage, missing_kw = keyword_coverage(cv_to_text(cv), job.keywords) if cv else (0.0, job.keywords)
    report = GapReport(
        job_title=job.title,
        company=job.company,
        profile_match=_weighted(items, lambda i: PROFILE_CREDIT[i.status]),
        cv_coverage=_weighted(
            items, lambda i: 1.0 if i.status == "matched" else (0.5 if i.in_cv else 0.0)
        ),
        keyword_coverage=round(coverage, 3),
        missing_keywords=missing_kw,
        items=items,
    )
    if advise:
        if llm is None:
            from ..llm import get_light_llm

            llm = get_light_llm(0.2)
        out = _advise(job, items, llm, fallback_llm)
        if out:
            by_req = {a.requirement.strip().lower(): a.advice for a in out.advice}
            for i in report.items:
                i.advice = by_req.get(i.requirement.strip().lower(), "")
            report.summary = out.summary
    return report

