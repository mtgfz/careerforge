"""Tests for the React app backend: progress events, speed knobs, jobs API, replay, re-export."""
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_chroma")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src import config, service
from src.api import jobs as jobs_mod
from src.api.main import app
from src.graph import build_graph
from src.ingestion.loaders import load_text
from src.ingestion.store import KnowledgeBase
from src.pipeline import run_pipeline
from src.schemas import (
    Candidate,
    BulletVerdict,
    CVBullet,
    CVEntry,
    GeneratedCV,
    JobProfile,
    JudgeOutput,
    Requirement,
)
from tests.test_pipeline_e2e import FakeEmbeddings

client = TestClient(app)

GOOD = "Built FastAPI tools in Python with Docker"
BAD = "Led a team of 50 engineers"


class StubbornLLM(RunnableLambda):
    """Always writes one grounded and one invented bullet; counts judge calls."""

    def __init__(self):
        super().__init__(lambda _: AIMessage(content="text"))
        self.judge_calls = 0

    def with_structured_output(self, schema, **kw):
        if schema is JobProfile:
            return RunnableLambda(lambda _: JobProfile(
                title="ML Intern", language="en",
                requirements=[Requirement(name="Python", importance=5)], keywords=["Python"],
            ))
        if schema is GeneratedCV:
            return RunnableLambda(lambda _: GeneratedCV(
                headline="ML Intern", summary="Student.", skills=["Python"],
                projects=[CVEntry(title="Tool", bullets=[
                    CVBullet(text=GOOD, source_ids=["E1"]), CVBullet(text=BAD, source_ids=[])])],
            ))

        def judge(_):
            self.judge_calls += 1
            return JudgeOutput(
                verdicts=[BulletVerdict(bullet=GOOD, supported=True),
                          BulletVerdict(bullet=BAD, supported=False, reason="not in evidence")],
                quality_score=8, quality_feedback="ok",
            )

        return RunnableLambda(judge)


def _kb():
    emb = FakeEmbeddings()
    kb = KnowledgeBase(embeddings=emb)
    kb.add([load_text("I built tools in Python with FastAPI and Docker.", "cv.md")])
    return kb, emb


# ---------- graph: plateau early stop ----------
def _plateau_run(overalls, delta):
    calls = {"write": 0}

    def write(s):
        calls["write"] += 1
        return {"cv": "cv", "iteration": s.get("iteration", 0) + 1}

    def critique(s):
        c = SimpleNamespace(passed=False, overall=overalls[min(len(s.get("history", [])), len(overalls) - 1)])
        return {"critique": c, "history": [*s.get("history", []), {"critique": c}]}

    g = build_graph(lambda s: {"job": 1}, lambda s: {"evidence": 1}, write, critique,
                    True, 5, plateau_delta=delta)
    g.invoke({"job_text": "x", "iteration": 0, "history": []})
    return calls["write"]


def test_plateau_stops_when_score_stops_improving():
    assert _plateau_run([0.5, 0.52, 0.521, 0.9], delta=0.05) == 2  # +0.02 < 0.05 after 2nd draft


def test_plateau_keeps_going_while_improving_and_is_off_by_default():
    assert _plateau_run([0.5, 0.6, 0.7, 0.8, 0.9], delta=0.05) == 5
    assert _plateau_run([0.5, 0.5, 0.5, 0.5, 0.5], delta=0.0) == 5


# ---------- pipeline: events, timings, re-judge flag ----------
def test_pipeline_emits_progress_events_and_timings():
    kb, emb = _kb()
    events = []
    res = run_pipeline(kb, "ML intern", Candidate(name="T"), llm=StubbornLLM(),
                       embeddings=emb, on_event=events.append, max_iters=1)
    kinds = [(e["type"], e["node"]) for e in events]
    assert kinds[0] == ("start", "analyze")
    assert ("done", "critique") in kinds and ("done", "extras") in kinds
    crit_done = next(e for e in events if e["type"] == "done" and e["node"] == "critique")
    assert set(crit_done["scores"]) == {"faithfulness", "ats", "quality", "overall", "passed"}
    assert {"analyze", "retrieve", "write", "critique", "extras"} <= set(res.timings)


@pytest.mark.parametrize("rejudge", [True, False])
def test_rejudge_flag_saves_one_judge_call(monkeypatch, rejudge):
    monkeypatch.setattr(config, "REJUDGE_AFTER_PRUNE", rejudge)
    kb, emb = _kb()
    llm = StubbornLLM()
    res = run_pipeline(kb, "ML intern", Candidate(name="T"), llm=llm, embeddings=emb,
                       max_iters=1, with_extras=False)
    assert res.pruned >= 1  # the invented bullet is removed
    assert all(BAD not in b.text for e in res.cv.projects for b in e.bullets)
    assert llm.judge_calls == (2 if rejudge else 1)
    assert res.critique.faithfulness == 1.0 and not res.critique.unsupported


# ---------- service: knowledge base cache ----------
def test_kb_is_cached_for_identical_documents(monkeypatch):
    emb = FakeEmbeddings()
    monkeypatch.setattr(service, "KnowledgeBase", lambda: KnowledgeBase(embeddings=emb))
    service._KB_CACHE.clear()
    kb1, cached1 = service.build_kb_cached(extra_text="Python and Docker projects")
    kb2, cached2 = service.build_kb_cached(extra_text="Python and Docker projects")
    kb3, cached3 = service.build_kb_cached(extra_text="Something else entirely")
    assert (cached1, cached2, cached3) == (False, True, False)
    assert kb1 is kb2 and kb1 is not kb3


# ---------- API ----------
def _read_events(job_id):
    out = []
    with client.stream("GET", f"/api/jobs/{job_id}/events") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("data: "):
                out.append(json.loads(line[6:]))
    return out


def test_job_flow_streams_events_then_result(monkeypatch, tmp_path):
    def fake_generate(candidate, job_text, files, github, extra, use_critic, template="classic",
                      on_event=None, **kw):
        on_event({"type": "start", "node": "analyze", "iteration": 0})
        on_event({"type": "done", "node": "analyze", "iteration": 0, "seconds": 0.1})
        return {"run_id": "abc123", "job_title": "ML Intern", "out_dir": "/secret/path",
                "template": template, "files": []}

    monkeypatch.setattr(jobs_mod, "generate_application", fake_generate)
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    r = client.post("/api/jobs", data={"job_text": "offer", "name": "Test", "extra_text": "I know Python",
                                       "template": "modern"})
    assert r.status_code == 200
    events = _read_events(r.json()["job_id"])
    assert [e["type"] for e in events] == ["start", "done", "result"]
    result = events[-1]["result"]
    assert result["template"] == "modern" and "out_dir" not in result  # server path is not leaked


def test_job_validation_errors():
    assert client.post("/api/jobs", data={"job_text": "  ", "name": "T", "extra_text": "x"}).status_code == 400
    assert client.post("/api/jobs", data={"job_text": "", "name": "T", "extra_text": "x"}).status_code == 422
    assert client.post("/api/jobs", data={"job_text": "offer", "name": "T"}).status_code == 400
    assert client.get("/api/jobs/unknown/events").status_code == 404


def test_error_inside_a_job_is_reported(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(jobs_mod, "generate_application", boom)
    r = client.post("/api/jobs", data={"job_text": "offer", "name": "T", "extra_text": "x"})
    events = _read_events(r.json()["job_id"])
    assert events[-1]["type"] == "error" and "quota exceeded" in events[-1]["message"]


def test_rerender_edited_cv_and_download(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    (tmp_path / "run1").mkdir()
    cv = GeneratedCV(headline="Edited & ready", summary="S", skills=["Python"]).model_dump()
    body = {"cv": cv, "candidate": {"name": "T"}, "template": "compact"}
    r = client.post("/api/runs/run1/render", json=body)
    assert r.status_code == 200
    assert "cv_edited.tex" in r.json()["files"] and "cv_edited.md" in r.json()["files"]
    assert "Edited \\& ready" in (tmp_path / "run1" / "cv_edited.tex").read_text()
    assert client.get("/download/run1/cv_edited.md").status_code == 200
    assert client.get("/download/run1/../../etc/passwd").status_code == 404
    assert client.get("/download/run1/secret.txt").status_code == 404
    assert client.post("/api/runs/nope/render", json=body).status_code == 404
    bad = client.post("/api/runs/run1/render", json={**body, "cv": {"headline": 3}})
    assert bad.status_code == 422


def test_replay_a_recorded_run(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    (tmp_path / "run2").mkdir()
    rec = {"events": [{"type": "start", "node": "analyze", "iteration": 0, "t": 0.0},
                      {"type": "done", "node": "analyze", "iteration": 0, "seconds": 1.0, "t": 1.0}],
           "result": {"run_id": "run2", "job_title": "ML Intern", "iterations": 2}}
    (tmp_path / "run2" / "replay.json").write_text(json.dumps(rec))
    assert client.get("/api/replays").json()[0]["run_id"] == "run2"
    jid = client.post("/api/replay/run2").json()["job_id"]
    events = _read_events(jid)
    assert [e["type"] for e in events] == ["start", "done", "result"]
    assert events[-1]["result"]["replay"] is True
    assert client.post("/api/replay/missing").status_code == 404
