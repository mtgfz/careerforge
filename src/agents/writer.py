"""Writer agent: evidence-grounded CV generation, with optional critic feedback."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from ..llm import get_llm
from ..schemas import EvidencePack, GeneratedCV, JobProfile

SYSTEM = """You are an expert CV writer. You tailor a CV to ONE job offer.

HARD RULES
1. Use ONLY facts found in the EVIDENCE chunks. Never invent employers, dates, degrees, \
tools, metrics or achievements. If a requirement has no evidence, simply do not claim it.
2. Every bullet must list in source_ids the evidence IDs (e.g. "E3") that support it.
3. Every skill in `skills` must literally appear in the evidence.
4. Write bullets in impact style: strong action verb + what you did + how/with which tools \
+ result (only if the result is in the evidence). One line each, max ~25 words.
5. Order sections and bullets so the most relevant to the job come first; weave in the job \
keywords ONLY where the evidence supports them.
6. Write the CV in this language: {language}.
7. Do not output name or contact details; they are added separately."""

HUMAN = """JOB PROFILE
Title: {title} ({seniority}) at {company}
Requirements: {requirements}
Keywords: {keywords}

EVIDENCE
{evidence}

{revision}
Produce the tailored CV."""

REVISION_TEMPLATE = """REVISION REQUEST
Your previous draft was reviewed. Fix every problem below while keeping what was good.
Remove or rewrite unsupported claims; add missing keywords only if the evidence supports them.

Previous draft:
{previous}

Reviewer feedback:
{feedback}
"""


def format_evidence(evidence: EvidencePack) -> str:
    return "\n\n".join(
        f"[{c.id}] (source: {c.source})\n{c.text}" for c in evidence.chunks.values()
    )


def write_cv(
    job: JobProfile,
    evidence: EvidencePack,
    feedback: str | None = None,
    previous: GeneratedCV | None = None,
    llm=None,
) -> GeneratedCV:
    llm = llm or get_llm(temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])
    revision = ""
    if feedback and previous:
        revision = REVISION_TEMPLATE.format(
            previous=previous.model_dump_json(indent=1), feedback=feedback
        )
    chain = prompt | llm.with_structured_output(GeneratedCV)
    return chain.invoke(
        {
            "language": job.language,
            "title": job.title,
            "seniority": job.seniority or "n/a",
            "company": job.company or "n/a",
            "requirements": "; ".join(
                f"{r.name} (importance {r.importance})" for r in job.requirements
            ),
            "keywords": ", ".join(job.keywords),
            "evidence": format_evidence(evidence),
            "revision": revision,
        }
    )
