"""High-level entry point wiring the agents into the LangGraph workflow."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .agents.critic import evaluate_cv
from .agents.extras import cover_letter, interview_questions
from .agents.job_analyst import analyze_job
from .agents.retriever import retrieve_evidence
from .agents.writer import write_cv
from .graph import build_graph
from .ingestion.store import KnowledgeBase
from .llm import get_embeddings, get_llm
from .schemas import Candidate, Critique, EvidencePack, GeneratedCV, JobProfile


@dataclass
class PipelineResult:
    job: JobProfile
    cv: GeneratedCV
    evidence: EvidencePack
    critique: Critique
    iterations: int
    history: list = field(default_factory=list)
    cover_letter: str = ""
    questions: str = ""


def run_pipeline(
    kb: KnowledgeBase,
    job_text: str,
    candidate: Candidate,
    use_critic: bool = True,
    max_iters: int | None = None,
    with_extras: bool = True,
    llm=None,
    embeddings=None,
) -> PipelineResult:
    max_iters = max_iters or config.MAX_ITERATIONS
    llm = llm or get_llm(0.2)
    embeddings = embeddings or get_embeddings()

    def n_analyze(state):
        return {"job": analyze_job(state["job_text"], llm=llm)}

    def n_retrieve(state):
        return {"evidence": retrieve_evidence(state["job"], kb)}

    def n_write(state):
        crit = state.get("critique")
        cv = write_cv(
            state["job"],
            state["evidence"],
            feedback=crit.feedback if crit else None,
            previous=state.get("cv"),
            llm=llm,
        )
        return {"cv": cv, "iteration": state.get("iteration", 0) + 1}

    def n_critique(state):
        crit = evaluate_cv(
            state["job"], state["cv"], state["evidence"], state["job_text"], llm=llm,
            embeddings=embeddings,
        )
        hist = [*state.get("history", []), {"iteration": state["iteration"], "critique": crit}]
        return {"critique": crit, "history": hist}

    graph = build_graph(n_analyze, n_retrieve, n_write, n_critique, use_critic, max_iters)
    final = graph.invoke({"job_text": job_text, "iteration": 0, "history": []})

    # Always score the final CV (also in no-critic mode) so ablations are comparable.
    critique = final.get("critique") or evaluate_cv(
        final["job"], final["cv"], final["evidence"], job_text, llm=llm, embeddings=embeddings
    )

    result = PipelineResult(
        job=final["job"],
        cv=final["cv"],
        evidence=final["evidence"],
        critique=critique,
        iterations=final["iteration"],
        history=final.get("history", []),
    )
    if with_extras:
        result.cover_letter = cover_letter(result.job, result.cv, candidate, llm=llm)
        result.questions = interview_questions(result.job, result.cv, llm=llm)
    return result
