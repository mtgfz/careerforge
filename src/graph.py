"""LangGraph workflow: analyze -> retrieve -> write -> (critique -> write)* -> end.

The node functions are injected so the control flow can be unit-tested with stubs and
so the same graph runs both with the Critic loop (full system) and without it (ablation).
"""
from __future__ import annotations

from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph


class State(TypedDict, total=False):
    job_text: str
    job: Any            # JobProfile
    evidence: Any       # EvidencePack
    cv: Any             # GeneratedCV
    critique: Any       # Critique (latest)
    iteration: int
    history: list       # [{"iteration": n, "critique": Critique}]


def build_graph(
    analyze: Callable[[State], dict],
    retrieve: Callable[[State], dict],
    write: Callable[[State], dict],
    critique: Callable[[State], dict],
    use_critic: bool = True,
    max_iters: int = 3,
    plateau_delta: float = 0.0,
):
    g = StateGraph(State)
    g.add_node("analyze", analyze)
    g.add_node("retrieve", retrieve)
    g.add_node("write", write)
    g.add_node("critique", critique)

    g.set_entry_point("analyze")
    g.add_edge("analyze", "retrieve")
    g.add_edge("retrieve", "write")

    def after_write(state: State) -> str:
        return "critique" if use_critic else END

    def after_critique(state: State) -> str:
        if state["critique"].passed or state.get("iteration", 0) >= max_iters:
            return END
        hist = state.get("history", [])
        if plateau_delta > 0 and len(hist) >= 2:
            # Another rewrite is not worth an LLM round trip when the score stopped improving.
            gain = hist[-1]["critique"].overall - hist[-2]["critique"].overall
            if gain < plateau_delta:
                return END
        return "write"

    g.add_conditional_edges("write", after_write, {"critique": "critique", END: END})
    g.add_conditional_edges("critique", after_critique, {"write": "write", END: END})
    return g.compile()
