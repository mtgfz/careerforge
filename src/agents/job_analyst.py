"""Job analyst agent: job offer text -> structured JobProfile."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from ..llm import get_llm
from ..schemas import JobProfile

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You analyse job offers for a CV-tailoring system. Extract: job title, company, "
            "seniority, the language the offer is written in (ISO code), the concrete "
            "requirements (each with a kind and an importance from 1 to 5), and 8-20 ATS "
            "keywords (tools, technologies, methods, domain terms) exactly as written in the "
            "offer. Do not invent anything that is not in the offer.",
        ),
        ("human", "{job_text}"),
    ]
)


def analyze_job(job_text: str, llm=None) -> JobProfile:
    llm = llm or get_llm(temperature=0)
    chain = PROMPT | llm.with_structured_output(JobProfile)
    return chain.invoke({"job_text": job_text})
