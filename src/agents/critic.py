"""Critic agent: faithfulness + ATS + quality scoring, with actionable feedback.

Faithfulness  = share of claims (bullets + skills) that are grounded in the evidence.
                A bullet counts only if (a) it cites valid evidence IDs AND (b) the LLM judge
                confirms the evidence supports it. A skill counts if it appears in the evidence.
ATS score     = 50% keyword coverage + 50% embedding similarity between CV and job offer.
Quality       = LLM-as-judge rubric (1-10) on clarity, impact, action verbs, concision.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from .. import config
from ..llm import get_llm
from ..metrics import (
    all_bullets,
    cosine,
    cv_to_text,
    keyword_coverage,
    skill_grounded_flags,
    valid_citation_flags,
)
from ..schemas import Critique, EvidencePack, GeneratedCV, JobProfile, JudgeOutput
from .writer import format_evidence

JUDGE_SYSTEM = """You are a strict CV reviewer and fact-checker.

TASK 1, fact-check: for EVERY bullet, decide if the cited evidence clearly supports ALL claims \
in it (tools, numbers, outcomes, responsibilities). If any detail is missing from the \
evidence, mark supported=false and say what is unsupported.

TASK 2, quality: rate the CV from 1 to 10 for clarity, impact (action verb + result), \
relevance to the job, and concision. Give 2-4 concrete improvement suggestions."""

JUDGE_HUMAN = """JOB: {title}

EVIDENCE
{evidence}

CV BULLETS TO CHECK (with cited evidence IDs)
{bullets}

FULL CV TEXT (for quality rating)
{cv_text}"""


def _judge(job: JobProfile, cv: GeneratedCV, evidence: EvidencePack, llm) -> JudgeOutput:
    bullets = "\n".join(
        f"- {b.text}  [cites: {', '.join(b.source_ids) or 'NONE'}]" for b in all_bullets(cv)
    )
    prompt = ChatPromptTemplate.from_messages([("system", JUDGE_SYSTEM), ("human", JUDGE_HUMAN)])
    chain = prompt | llm.with_structured_output(JudgeOutput)
    return chain.invoke(
        {
            "title": job.title,
            "evidence": format_evidence(evidence),
            "bullets": bullets or "(no bullets)",
            "cv_text": cv_to_text(cv),
        }
    )


def evaluate_cv(
    job: JobProfile,
    cv: GeneratedCV,
    evidence: EvidencePack,
    job_text: str = "",
    llm=None,
    embeddings=None,
) -> Critique:
    llm = llm or get_llm(temperature=0)
    bullets = list(all_bullets(cv))

    # --- faithfulness (programmatic citation check AND LLM judge) ---
    cite_ok = valid_citation_flags(cv, set(evidence.chunks))
    judge = _judge(job, cv, evidence, llm)
    verdict_by_text = {v.bullet.strip(): v for v in judge.verdicts}

    unsupported: list[str] = []
    ok_items = 0
    for b, cited in zip(bullets, cite_ok):
        v = verdict_by_text.get(b.text.strip())
        judged_ok = v.supported if v else cited  # judge may paraphrase; fall back to citation
        if cited and judged_ok:
            ok_items += 1
        else:
            why = (v.reason if v and v.reason else "no valid evidence citation")
            unsupported.append(f"BULLET: {b.text}  -> {why}")

    skill_flags = skill_grounded_flags(cv.skills, evidence.text())
    for skill, grounded in zip(cv.skills, skill_flags):
        if grounded:
            ok_items += 1
        else:
            unsupported.append(f"SKILL not in evidence: {skill}")

    total_items = len(bullets) + len(cv.skills)
    faithfulness = ok_items / total_items if total_items else 0.0

    # --- ATS score ---
    cv_text = cv_to_text(cv)
    coverage, missing = keyword_coverage(cv_text, job.keywords)
    if embeddings is not None and job_text:
        sim = max(0.0, cosine(embeddings.embed_query(cv_text), embeddings.embed_query(job_text)))
        ats = 0.5 * coverage + 0.5 * sim
    else:
        ats = coverage

    quality = judge.quality_score / 10
    overall = 0.4 * faithfulness + 0.3 * ats + 0.3 * quality
    passed = faithfulness >= config.FAITHFULNESS_MIN and overall >= config.OVERALL_MIN

    feedback_parts = []
    if unsupported:
        feedback_parts.append("UNSUPPORTED CLAIMS (remove or rewrite):\n" + "\n".join(unsupported))
    if missing:
        feedback_parts.append(
            "MISSING JOB KEYWORDS (add ONLY if the evidence supports them): " + ", ".join(missing)
        )
    if judge.quality_feedback:
        feedback_parts.append("QUALITY FEEDBACK: " + judge.quality_feedback)

    return Critique(
        faithfulness=round(faithfulness, 3),
        ats=round(ats, 3),
        quality=round(quality, 3),
        overall=round(overall, 3),
        unsupported=unsupported,
        missing_keywords=missing,
        feedback="\n\n".join(feedback_parts),
        passed=passed,
    )
