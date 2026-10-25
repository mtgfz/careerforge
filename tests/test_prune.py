import pytest

pytest.importorskip("langchain_chroma")

from src.ingestion.loaders import load_text
from src.ingestion.store import KnowledgeBase
from src.metrics import dedupe_bullets, prune_unsupported
from src.pipeline import run_pipeline
from src.schemas import Candidate, CVBullet, CVEntry, GeneratedCV
from tests.test_pipeline_e2e import FakeEmbeddings, FakeLLM


def _cv():
    return GeneratedCV(
        headline="Dev",
        summary="s",
        skills=["Python", "Cassandra"],
        experiences=[
            CVEntry(title="Only bad", bullets=[CVBullet(text="Led 50 engineers", source_ids=[])]),
            CVEntry(title="Mixed", bullets=[
                CVBullet(text="Built an API", source_ids=["E1"]),
                CVBullet(text="Won an award", source_ids=[]),
            ]),
            CVEntry(title="No bullets"),
        ],
    )


def test_prune_removes_flagged_bullets_skills_and_empty_entries():
    out = prune_unsupported(_cv(), ["Led 50 engineers", "Won an award"], ["cassandra"])
    assert out.skills == ["Python"]
    assert [e.title for e in out.experiences] == ["Mixed", "No bullets"]
    assert [b.text for b in out.experiences[0].bullets] == ["Built an API"]


def test_prune_neutralizes_flagged_titles_and_drops_flagged_summary():
    cv = GeneratedCV(
        headline="h",
        summary="Expert in everything.",
        skills=[],
        experiences=[CVEntry(title="Senior Architect", organization="ACME",
                             bullets=[CVBullet(text="Built X", source_ids=["E1"])])],
        projects=[CVEntry(title="Lead Researcher",
                          bullets=[CVBullet(text="Built Y", source_ids=["E2"])])],
    )
    out = prune_unsupported(cv, [], [], ["senior architect", "Lead Researcher"], drop_summary=True)
    assert out.summary == ""
    assert out.experiences[0].title == "ACME"
    assert out.projects[0].title == "Project"
    assert out.experiences[0].bullets == cv.experiences[0].bullets


def test_dedupe_removes_repeated_achievements_across_entries():
    a = "Developed FastAPI backend integrating three NLP models (RoBERTa, DistilBERT, BART) for real-time fact-checking scores"
    b = "Implemented FastAPI backend serving three NLP models (RoBERTa, DistilBERT, BART) for fact-checking"
    other = "Compared Docker Compose and Kubernetes deployments with k6 load testing"
    cv = GeneratedCV(
        headline="h", summary="s", skills=[],
        experiences=[CVEntry(title="Role", bullets=[CVBullet(text=a, source_ids=["E3"])])],
        projects=[
            CVEntry(title="Truthnet again", bullets=[CVBullet(text=b, source_ids=["E3"])]),
            CVEntry(title="Cloud", bullets=[CVBullet(text=other, source_ids=["E5"])]),
            CVEntry(title="No bullets"),
        ],
    )
    out = dedupe_bullets(cv)
    assert [b.text for b in out.experiences[0].bullets] == [a]
    assert [e.title for e in out.projects] == ["Cloud", "No bullets"]


def test_entries_citing_the_same_evidence_are_merged():
    cv = GeneratedCV(
        headline="h", summary="s", skills=[],
        projects=[
            CVEntry(title="TRUTHNET", bullets=[CVBullet(text="Built a FastAPI backend", source_ids=["E3"])]),
            CVEntry(title="Cloud", bullets=[CVBullet(text="Ran k6 load tests", source_ids=["E5"])]),
            CVEntry(title="TRUTHNET mobile app", bullets=[CVBullet(text="Added a Flutter client", source_ids=["E3"])]),
        ],
    )
    out = dedupe_bullets(cv)
    assert [e.title for e in out.projects] == ["TRUTHNET", "Cloud"]
    assert [b.text for b in out.projects[0].bullets] == ["Built a FastAPI backend", "Added a Flutter client"]


def test_prune_without_flags_changes_nothing():
    assert prune_unsupported(_cv(), [], []) == _cv()


def test_pipeline_prunes_when_the_loop_runs_out_of_iterations():
    emb = FakeEmbeddings()
    kb = KnowledgeBase(embeddings=emb)
    kb.add([load_text("I built internal tools in Python with FastAPI and Docker.", "cv.md")])

    # Only one iteration allowed: the first (hallucinating) draft is the last one.
    res = run_pipeline(
        kb, "ML intern, Python and Docker", Candidate(name="T"),
        llm=FakeLLM(), embeddings=emb, max_iters=1, with_extras=False,
    )

    assert res.iterations == 1
    assert res.pruned == 1
    assert all("50 engineers" not in b.text for e in res.cv.projects for b in e.bullets)
    assert res.critique.faithfulness == 1.0
