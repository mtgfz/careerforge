"""Control-flow tests for the LangGraph workflow, using stub nodes (no LLM needed)."""
from types import SimpleNamespace

from src.graph import build_graph


def run(scores, use_critic=True, max_iters=3):
    calls = {"write": 0, "critique": 0}

    def analyze(s):
        return {"job": "job"}

    def retrieve(s):
        return {"evidence": "ev"}

    def write(s):
        calls["write"] += 1
        return {"cv": f"cv{calls['write']}", "iteration": s.get("iteration", 0) + 1}

    def critique(s):
        i = calls["critique"]
        calls["critique"] += 1
        return {"critique": SimpleNamespace(passed=scores[min(i, len(scores) - 1)])}

    g = build_graph(analyze, retrieve, write, critique, use_critic, max_iters)
    final = g.invoke({"job_text": "x", "iteration": 0})
    return final, calls


def test_stops_as_soon_as_critic_passes():
    final, calls = run([False, True])
    assert calls == {"write": 2, "critique": 2}
    assert final["iteration"] == 2


def test_respects_max_iterations():
    final, calls = run([False], max_iters=3)
    assert calls["write"] == 3 and calls["critique"] == 3


def test_no_critic_mode_writes_once():
    final, calls = run([False], use_critic=False)
    assert calls == {"write": 1, "critique": 0}
    assert "critique" not in final or final.get("critique") is None
