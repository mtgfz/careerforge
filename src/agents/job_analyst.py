"""Job analyst agent: job offer text -> structured JobProfile."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from ..llm import get_llm, invoke_structured
from ..schemas import JobProfile

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You analyse job offers for a CV-tailoring system. Extract: job title, company, "
            "seniority, the language the offer is written in (ISO code), the concrete "
            "requirements (each with a kind and an importance from 1 to 5), and 8-20 ATS "
            "keywords (tools, technologies, methods, domain terms) exactly as written in the "
            "offer. Do not invent anything that is not in the offer. Always answer by calling the "
            "tool, even if the offer is very short or vague: fill only what the text states, use "
            "an empty string or empty list for the rest, and never ask for more information.",
        ),
        ("human", "{job_text}"),
    ]
)


def analyze_job(job_text: str, llm=None, fallback_llm=None) -> JobProfile:
    llm = llm or get_llm(temperature=0)
    return invoke_structured(PROMPT, JobProfile, {"job_text": job_text}, llm, fallback=fallback_llm)
