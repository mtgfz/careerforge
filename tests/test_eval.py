"""Evaluation tooling: pair loading, ablation runner (resume), analysis and proof run.

Everything runs offline with the fake LLM / embeddings of the e2e test.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("langchain_chroma")

from langchain_core.runnables import RunnableLambda

from eval import analysis, pairs as pairs_module
from eval.ablation import run_ablation
from eval.pairs import load_pairs, sample_pairs
from src.agents.gap_analyst import GapAdvice, RequirementAdvice
from tests.test_pipeline_e2e import FakeEmbeddings, FakeLLM

CV_TEXT = "I built internal tools in Python with FastAPI and Docker at an agency."


class FakeLLMWithAdvice(FakeLLM):
    def with_structured_output(self, schema, **kw):
        if schema is GapAdvice:
            return RunnableLambda(lambda _: GapAdvice(
                advice=[RequirementAdvice(requirement="Docker", advice="Show it.")], summary="Fine."
            ))
        return super().with_structured_output(schema, **kw)


@pytest.fixture
def pairs_dir(tmp_path, monkeypatch):
    root = tmp_path / "pairs"
    a = root / "01_ml"
    a.mkdir(parents=True)
    (a / "cv.md").write_text(CV_TEXT)
    (a / "job.txt").write_text("ML intern, Python and Docker")
    (a / "candidate.json").write_text(json.dumps({"name": "Amina", "note": "x"}))
    b = root / "02_web"
    b.mkdir()
    (b / "resume.txt").write_text(CV_TEXT)
    (b / "job.url").write_text("https://jobs.example.com/42\n")

    fetched = []

    def fake_import(url):
        fetched.append(url)
        return type("J", (), {"text": "Web intern, Python and Docker"})()

    monkeypatch.setattr("src.ingestion.job_url.import_job_from_url", fake_import)
    return root, fetched


def test_load_pairs(pairs_dir):
    root, fetched = pairs_dir
    a, b = load_pairs(root)
    assert (a.id, a.candidate_name, a.extra) == ("01_ml", "Amina", {"note": "x"})
    assert [p.name for p in a.cv_files] == ["cv.md"] and a.job_text.startswith("ML intern")
    assert [p.name for p in b.cv_files] == ["resume.txt"]
    assert b.job_source == "https://jobs.example.com/42" and b.job_text.startswith("Web intern")
    # the fetched offer is cached: a second load does not hit the network
    assert (root / "02_web" / "job.fetched.txt").exists()
    load_pairs(root)
    assert fetched == ["https://jobs.example.com/42"]
    assert [p.id for p in load_pairs(root, only=["02_web"])] == ["02_web"]


def test_pair_without_cv_is_rejected(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "job.txt").write_text("job")
    with pytest.raises(FileNotFoundError, match="no CV"):
        load_pairs(tmp_path)


def test_sample_pairs_cover_five_offers():
    pairs = sample_pairs()
    assert len(pairs) == 5
    assert all(p.cv_files[0].name == "sample_cv.md" and p.job_text for p in pairs)


def test_sign_test():
    assert analysis.sign_test(5, 0) == pytest.approx(0.0625)
    assert analysis.sign_test(3, 2) == 1.0
    assert analysis.sign_test(0, 0) == 1.0


def synthetic_rows():
    rows = []
    for i, pair in enumerate(["a", "b", "c"]):
        for variant, f in (("single_pass", 0.7), ("full", 0.95)):
            rows.append({"pair": pair, "variant": variant, "repeat": 0, "faithfulness": f + i * 0.01,
                         "ats": 0.6, "quality": 0.8, "overall": 0.7 + (variant == "full") * 0.1,
                         "passed": variant == "full", "iterations": 1 + (variant == "full"),
                         "history": [{"iteration": 1, "faithfulness": 0.8, "ats": 0.6, "quality": 0.8, "overall": 0.75},
                                     {"iteration": 2, "faithfulness": 0.95, "ats": 0.6, "quality": 0.8, "overall": 0.8}]
                         if variant == "full" else []})
    rows.append({"pair": "c", "variant": "full", "repeat": 1, "error": "QuotaExceeded: daily"})
    return rows


def test_summary_and_paired_comparison():
    df = pd.DataFrame(synthetic_rows())
    s = analysis.summarize(df)
    assert list(s.index) == ["single_pass", "full"]
    assert s.loc["full", "runs"] == 3  # the failed run is ignored
    assert s.loc["full", "pass_rate"] == 1.0
    p = analysis.paired(df, "faithfulness")
    assert p.loc["full", "wins"] == 3 and p.loc["full", "mean_delta"] == pytest.approx(0.25)
    t = analysis.trajectory(df)
    assert list(t.index) == [1, 2] and t.loc[2, "faithfulness"] == 0.95


def test_write_report(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in synthetic_rows()))
    text = analysis.write_report(path, tmp_path)
    for f in ("results.md", "results.csv", "results.png", "trajectory.png"):
        assert (tmp_path / f).exists(), f
    assert "3 CV/offer pairs, 6 successful runs, 1 failed" in text
    assert "paired comparison" in text and "QuotaExceeded" in text


def test_run_ablation_end_to_end_and_resume(pairs_dir, tmp_path, monkeypatch):
    root, _ = pairs_dir
    pairs = load_pairs(root)
    emb = FakeEmbeddings()
    out = tmp_path / "eval"

    path = run_ablation(pairs, ["single_pass", "full"], out=out, llm=FakeLLM(), embeddings=emb)
    rows = [json.loads(ln) for ln in path.read_text().splitlines()]
    assert len(rows) == 4 and not any(r.get("error") for r in rows)
    full = [r for r in rows if r["variant"] == "full"]
    single = [r for r in rows if r["variant"] == "single_pass"]
    assert all(r["faithfulness"] == 1.0 and r["passed"] and r["history"] for r in full)
    assert all(r["history"] == [] and r["iterations"] == 1 for r in single)
    assert {"profile_match", "cv_coverage", "n_bullets", "seconds"} <= set(rows[0])

    # Resume: nothing left to do, and a failed row is retried on the next call.
    calls = []
    monkeypatch.setattr("eval.ablation.run_one", lambda *a, **k: calls.append(a[1]) or {"error": "stub"})
    run_ablation(pairs, ["single_pass", "full"], out=out, embeddings=emb)
    assert calls == []
    with path.open("a") as f:
        f.write(json.dumps({"pair": "01_ml", "variant": "low_recall", "repeat": 0, "error": "boom"}) + "\n")
    run_ablation(pairs, ["single_pass", "full", "low_recall"], out=out, embeddings=emb)
    assert calls == ["low_recall", "low_recall"]
    assert "boom" not in path.read_text()


def test_low_recall_variant_restores_the_config(pairs_dir):
    from src import config
    from eval.ablation import run_one

    root, _ = pairs_dir
    pair = load_pairs(root)[0]
    emb = FakeEmbeddings()
    before = config.K_PER_REQUIREMENT
    row = run_one(pair, "low_recall", pairs_module.build_kb(pair, emb), llm=FakeLLM(), embeddings=emb)
    assert not row.get("error") and config.K_PER_REQUIREMENT == before


def test_run_proof(pairs_dir, tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_proof", Path("scripts/run_proof.py"))
    run_proof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_proof)

    root, _ = pairs_dir
    out = tmp_path / "proof"
    rows = run_proof.main(["--pairs", str(root), "--out", str(out), "--no-pdf", "--only", "01_ml"],
                          llm=FakeLLMWithAdvice(), embeddings=FakeEmbeddings())
    assert len(rows) == 1 and not rows[0].get("error"), rows
    d = out / "01_ml"
    for f in ("cv.md", "cv.tex", "cover_letter.txt", "interview_questions.txt", "gap_report.md",
              "trace.json", "run.json"):
        assert (d / f).exists(), f
    assert "Amina" in (d / "cv.md").read_text()
    trace = json.loads((d / "trace.json").read_text())
    assert len(trace["history"]) == 2 and trace["gap_report"]["items"]
    assert "| [01_ml](01_ml/cv.md) | ML Intern |" in (out / "index.md").read_text()


def test_ablation_forces_the_rejudge_after_pruning(pairs_dir, monkeypatch):
    """Even with REJUDGE_AFTER_PRUNE=false in .env (web app speed knob), the ablation re-judges."""
    from src import config
    from eval import ablation

    monkeypatch.setattr(config, "REJUDGE_AFTER_PRUNE", False)
    seen = []

    def fake_pipeline(*a, **k):
        seen.append(config.REJUDGE_AFTER_PRUNE)
        raise RuntimeError("stop")

    monkeypatch.setattr("src.pipeline.run_pipeline", fake_pipeline)
    root, _ = pairs_dir
    pair = load_pairs(root)[0]
    row = ablation.run_one(pair, "full", kb=None)
    assert seen == [True] and "stop" in row["error"]
    assert config.REJUDGE_AFTER_PRUNE is False  # restored afterwards


def test_ablation_stops_at_the_daily_quota_and_runs_repeat_major(pairs_dir, tmp_path, monkeypatch):
    root, _ = pairs_dir
    pairs = load_pairs(root)
    calls = []

    def fake_run_one(pair, variant, kb, rep, **kw):
        calls.append((pair.id, variant, rep))
        if len(calls) == 3:
            return {"pair": pair.id, "variant": variant, "repeat": rep, "error": "QuotaExceeded: daily",
                    "daily_quota": True}
        return {"pair": pair.id, "variant": variant, "repeat": rep,
                **{m: 0.5 for m in ("faithfulness", "ats", "quality", "overall")}}

    monkeypatch.setattr("eval.ablation.run_one", fake_run_one)
    monkeypatch.setattr("eval.ablation.build_kb", lambda *a, **k: None)
    run_ablation(pairs, ["single_pass", "full"], repeats=2, out=tmp_path)
    # repeat 0 of every pair comes before any repeat 1, and the loop stops at the daily quota
    assert calls == [("01_ml", "single_pass", 0), ("01_ml", "full", 0), ("02_web", "single_pass", 0)]


def test_wait_quota_sleeps_and_retries_the_same_run(pairs_dir, tmp_path, monkeypatch):
    from eval import ablation

    root, _ = pairs_dir
    pairs = load_pairs(root)[:1]
    calls, slept = [], []

    def fake_run_one(pair, variant, kb, rep, **kw):
        calls.append(variant)
        if len(calls) == 2:  # the 2nd call hits the daily quota once
            return {"pair": pair.id, "variant": variant, "repeat": rep, "error": "QuotaExceeded",
                    "daily_quota": True, "quota_wait": 30 * 60}
        return {"pair": pair.id, "variant": variant, "repeat": rep,
                **{m: 0.5 for m in ("faithfulness", "ats", "quality", "overall")}}

    monkeypatch.setattr("eval.ablation.run_one", fake_run_one)
    monkeypatch.setattr("eval.ablation.build_kb", lambda *a, **k: None)
    monkeypatch.setattr(ablation, "_sleep", slept.append)
    path = run_ablation(pairs, ["single_pass", "full"], out=tmp_path, wait_quota=True)
    assert calls == ["single_pass", "full", "full"]  # the quota-hit run is retried, not skipped
    assert slept == [20 * 60 + 5]  # capped at max_wait
    rows = [json.loads(ln) for ln in path.read_text().splitlines()]
    assert len(rows) == 2 and not any(r.get("error") for r in rows)  # the failed try is not saved
