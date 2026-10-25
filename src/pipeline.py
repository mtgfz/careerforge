"""High-level entry point wiring the agents into the LangGraph workflow."""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import config
from .agents.critic import evaluate_cv
from .agents.extras import cover_letter, interview_questions
from .agents.job_analyst import analyze_job
from .agents.retriever import retrieve_evidence
from .agents.writer import write_cv
from .graph import build_graph
from .ingestion.store import KnowledgeBase
from .llm import (
    get_embeddings,
    get_judge_fallback_llm,
    get_judge_llm,
    get_light_llm,
    get_llm,
    light_is_distinct,
    quota_error,
)
from .metrics import all_bullets, prune_unsupported
from .schemas import Candidate, Critique, EvidencePack, GeneratedCV, JobProfile


log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    job: JobProfile
    cv: GeneratedCV
    evidence: EvidencePack
    critique: Critique
    iterations: int
    history: list = field(default_factory=list)
    best_iteration: int = 0  # iteration whose CV is returned (0 = no critic)
    pruned: int = 0  # bullets + skills removed from the returned CV
    cover_letter: str = ""
    questions: str = ""
    timings: dict = field(default_factory=dict)  # seconds spent per step


def run_pipeline(
    kb: KnowledgeBase,
    job_text: str,
    candidate: Candidate,
    use_critic: bool = True,
    max_iters: int | None = None,
    with_extras: bool = True,
    llm=None,
    embeddings=None,
    judge_llm=None,
    on_event=None,
) -> PipelineResult:
    """`on_event(dict)` receives progress events: start/done per step, with timings and scores."""
    max_iters = max_iters or config.MAX_ITERATIONS
    timings: dict[str, float] = {}

    def emit(event: dict) -> None:
        if on_event:
            on_event(event)

    def instrument(name, fn):
        """Wrap a graph node: emit start/done events and accumulate its wall-clock time."""

        def wrapped(state):
            it = state.get("iteration", 0) + (1 if name == "write" else 0)
            emit({"type": "start", "node": name, "iteration": it})
            t0 = time.perf_counter()
            out = fn(state)
            dt = time.perf_counter() - t0
            timings[name] = round(timings.get(name, 0.0) + dt, 2)
            ev = {"type": "done", "node": name, "iteration": it, "seconds": round(dt, 1)}
            crit = out.get("critique") if isinstance(out, dict) else None
            if crit is not None:
                ev["scores"] = {
                    k: getattr(crit, k) for k in ("faithfulness", "ats", "quality", "overall", "passed")
                }
            emit(ev)
            return out

        return wrapped

    injected_llm = llm is not None
    # An injected llm (tests) also judges; otherwise the Critic uses the JUDGE_* model.
    judge_fallback = None if (llm is not None or judge_llm is not None) else get_judge_fallback_llm()
    judge_llm = judge_llm or (llm if llm is not None else get_judge_llm())
    # Analysis and extras run on the light model (separate free quota) and fall back to the
    # main one; an injected llm (tests) does everything.
    if llm is None:
        llm = get_llm(0.2)
        light_llm = get_light_llm(0.2)
        light_backup = llm if light_is_distinct() else None
    else:
        light_llm, light_backup = llm, None
    embeddings = embeddings or get_embeddings()

    def n_analyze(state):
        return {"job": analyze_job(state["job_text"], llm=light_llm, fallback_llm=light_backup)}

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
            state["job"], state["cv"], state["evidence"], state["job_text"], llm=judge_llm,
            embeddings=embeddings, fallback_llm=judge_fallback,
        )
        hist = [
            *state.get("history", []),
            {"iteration": state["iteration"], "critique": crit, "cv": state["cv"]},
        ]
        return {"critique": crit, "history": hist}

    def run_best_of_n() -> dict:
        """Alternative to the sequential loop: write N candidates in parallel, score each, keep the best.

        Same evidence for every candidate; diversity comes from the sampling temperature. Trades
        extra parallel LLM calls for fewer sequential rounds (lower latency, no feedback loop).
        """
        state = {"job_text": job_text, "iteration": 0, "history": []}
        state.update(instrument("analyze", n_analyze)(state))
        state.update(instrument("retrieve", n_retrieve)(state))
        n = max(1, config.BEST_OF_N)
        temps = [0.2, 0.5, 0.8, 0.35, 0.65][:n] + [0.5] * max(0, n - 5)

        def candidate(i: int):
            model = llm if injected_llm else get_llm(temps[i])
            cv = write_cv(state["job"], state["evidence"], feedback=None, previous=None, llm=model)
            crit = evaluate_cv(
                state["job"], cv, state["evidence"], job_text, llm=judge_llm,
                embeddings=embeddings, fallback_llm=judge_fallback,
            )
            return {"iteration": i + 1, "critique": crit, "cv": cv}

        emit({"type": "start", "node": "write", "iteration": 1})
        t0 = time.perf_counter()
        with ThreadPoolExecutor(n) as ex:
            hist = list(ex.map(candidate, range(n)))
        dt = time.perf_counter() - t0
        timings["write"] = round(dt, 2)
        best = max(hist, key=lambda h: (h["critique"].passed, h["critique"].faithfulness, h["critique"].overall))
        emit({"type": "done", "node": "write", "iteration": 1, "seconds": round(dt, 1)})
        emit({"type": "done", "node": "critique", "iteration": 1, "seconds": 0.0, "scores": {
            k: getattr(best["critique"], k) for k in ("faithfulness", "ats", "quality", "overall", "passed")}})
        return {**state, "cv": best["cv"], "critique": best["critique"], "iteration": n, "history": hist}

    if use_critic and config.STRATEGY == "best_of_n":
        final = run_best_of_n()
    else:
        graph = build_graph(
            instrument("analyze", n_analyze),
            instrument("retrieve", n_retrieve),
            instrument("write", n_write),
            instrument("critique", n_critique),
            use_critic,
            max_iters,
            plateau_delta=config.PLATEAU_DELTA,
        )
        final = graph.invoke({"job_text": job_text, "iteration": 0, "history": []})

    history = final.get("history", [])
    pruned = 0
    if history:
        # The last draft is not always the best one: return the highest-scoring iteration.
        best = max(
            history,
            key=lambda h: (h["critique"].passed, h["critique"].faithfulness, h["critique"].overall),
        )
        cv, critique, best_iteration = best["cv"], best["critique"], best["iteration"]
        drop_summary = not critique.summary_supported
        if (
            critique.unsupported_bullets
            or critique.unsupported_skills
            or critique.unsupported_titles
            or drop_summary
        ):
            # Remove what the Critic could not ground, then score the cleaned CV once more.
            n_before = len(list(all_bullets(cv))) + len(cv.skills)
            cv = prune_unsupported(
                cv,
                critique.unsupported_bullets,
                critique.unsupported_skills,
                critique.unsupported_titles,
                drop_summary,
            )
            pruned = (
                n_before - len(list(all_bullets(cv))) - len(cv.skills)
                + len(critique.unsupported_titles) + int(drop_summary)
            )
            if config.REJUDGE_AFTER_PRUNE:
                emit({"type": "start", "node": "rejudge", "iteration": best_iteration})
                t0 = time.perf_counter()
                critique = evaluate_cv(
                    final["job"], cv, final["evidence"], job_text, llm=judge_llm,
                    embeddings=embeddings, fallback_llm=judge_fallback,
                )
                timings["rejudge"] = round(time.perf_counter() - t0, 2)
                emit({"type": "done", "node": "rejudge", "iteration": best_iteration,
                      "seconds": round(timings["rejudge"], 1)})
            else:
                # Everything the judge flagged was removed: faithfulness is 1.0 by construction.
                overall = round(0.4 * 1.0 + 0.3 * critique.ats + 0.3 * critique.quality, 3)
                critique = critique.model_copy(update={
                    "faithfulness": 1.0, "overall": overall, "unsupported": [],
                    "unsupported_bullets": [], "unsupported_skills": [], "unsupported_titles": [],
                    "summary_supported": True,
                    "passed": overall >= config.OVERALL_MIN,
                })
    else:
        # No-critic mode: still score the final CV so ablations are comparable.
        cv, best_iteration = final["cv"], 0
        critique = evaluate_cv(
            final["job"], cv, final["evidence"], job_text, llm=judge_llm, embeddings=embeddings,
            fallback_llm=judge_fallback,
        )

    result = PipelineResult(
        job=final["job"],
        cv=cv,
        evidence=final["evidence"],
        critique=critique,
        iterations=final["iteration"],
        history=history,
        best_iteration=best_iteration,
        pruned=pruned,
        timings=timings,
    )
    if with_extras:
        emit({"type": "start", "node": "extras", "iteration": 0})
        t0 = time.perf_counter()
        models = [light_llm] + ([light_backup] if light_backup is not None else [])
        if config.PARALLEL_EXTRAS:
            with ThreadPoolExecutor(2) as ex:
                f_letter = ex.submit(
                    _free_text, cover_letter, models, result.job, result.cv, candidate
                )
                f_questions = ex.submit(
                    _free_text, interview_questions, models, result.job, result.cv
                )
                result.cover_letter, result.questions = f_letter.result(), f_questions.result()
        else:
            result.cover_letter = _free_text(cover_letter, models, result.job, result.cv, candidate)
            result.questions = _free_text(interview_questions, models, result.job, result.cv)
        timings["extras"] = round(time.perf_counter() - t0, 2)
        emit({"type": "done", "node": "extras", "iteration": 0, "seconds": round(timings["extras"], 1)})
    return result


def _free_text(fn, models: list, *args) -> str:
    """Run a free-text agent on each model in turn. A failed bonus output must not throw away
    a finished CV, so after the last failure the reason is returned in place of the text."""
    reason = ""
    for model in models:
        try:
            return fn(*args, llm=model)
        except Exception as e:
            reason = str(quota_error(e, model) or e)
            log.warning("%s failed: %s", fn.__name__, reason[:200])
    return f"(Not generated: {reason[:200]})"
