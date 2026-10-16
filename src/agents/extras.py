"""Bonus outputs: cover letter and likely interview questions (grounded in the CV)."""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..llm import get_llm
from ..metrics import cv_to_text
from ..schemas import Candidate, GeneratedCV, JobProfile

COVER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write a concise cover letter (max 220 words) in {language}. Use ONLY facts from the "
            "CV below; never invent experience. Plain text, no placeholders other than the "
            "candidate name at the end.",
        ),
        (
            "human",
            "Candidate: {name}\nJob: {title} at {company}\n\nCV:\n{cv_text}",
        ),
    ]
)

QUESTIONS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "List the 5 interview questions this candidate is most likely to face for this job, "
            "in {language}. For each, add one line starting with 'Tip:' on which project or fact "
            "from the CV to use in the answer. Numbered list.",
        ),
        ("human", "Job: {title}\nRequirements: {requirements}\n\nCV:\n{cv_text}"),
    ]
)


def cover_letter(job: JobProfile, cv: GeneratedCV, candidate: Candidate, llm=None) -> str:
    llm = llm or get_llm(0.4)
    return (COVER_PROMPT | llm | StrOutputParser()).invoke(
        {
            "language": job.language,
            "name": candidate.name,
            "title": job.title,
            "company": job.company or "your company",
            "cv_text": cv_to_text(cv),
        }
    )


def interview_questions(job: JobProfile, cv: GeneratedCV, llm=None) -> str:
    llm = llm or get_llm(0.4)
    return (QUESTIONS_PROMPT | llm | StrOutputParser()).invoke(
        {
            "language": job.language,
            "title": job.title,
            "requirements": "; ".join(r.name for r in job.requirements),
            "cv_text": cv_to_text(cv),
        }
    )
