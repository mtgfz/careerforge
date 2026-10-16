"""End-to-end smoke test with a fake LLM and fake embeddings (no network, no API key).

Verifies the wiring: ingestion -> job analysis -> retrieval -> writing -> critic loop ->
rendering. The fake Writer first hallucinates, then fixes it after the Critic's feedback.
"""
import hashlib
import math

import pytest

pytest.importorskip("langchain_chroma")

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.ingestion.loaders import load_text
from src.ingestion.store import KnowledgeBase
from src.pipeline import run_pipeline
from src.render.latex import render_latex
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


class FakeEmbeddings:
    """Deterministic bag-of-words hashing embeddings, L2-normalized."""

    dim = 64

    def _vec(self, text):
        v = [0.0] * self.dim
        for tok in text.lower().split():
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        n = math.sqrt(sum(x * x for x in v)) or 1
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM(RunnableLambda):
    def __init__(self):
        super().__init__(lambda _: AIMessage(content="Fake text output."))
        self.writer_calls = 0

    def with_structured_output(self, schema, **kw):
        if schema is JobProfile:
            obj = lambda _: JobProfile(
                title="ML Intern",
                language="en",
                requirements=[Requirement(name="Python", importance=5), Requirement(name="Docker")],
                keywords=["Python", "Docker"],
            )
        elif schema is GeneratedCV:
            def obj(_):
                self.writer_calls += 1
                bullets = [CVBullet(text="Built FastAPI tools in Python with Docker", source_ids=["E1"])]
                if self.writer_calls == 1:  # first draft hallucinates
                    bullets.append(CVBullet(text="Led a team of 50 engineers", source_ids=[]))
                return GeneratedCV(
                    headline="ML Intern",
                    summary="Student.",
                    skills=["Python", "Docker"],
                    projects=[CVEntry(title="Tool", bullets=bullets)],
                )
        elif schema is JudgeOutput:
            def obj(_):
                return JudgeOutput(
                    verdicts=[
                        BulletVerdict(bullet="Built FastAPI tools in Python with Docker", supported=True),
                        BulletVerdict(bullet="Led a team of 50 engineers", supported=False, reason="not in evidence"),
                    ],
                    quality_score=8,
                    quality_feedback="Good.",
                )
        else:
            raise AssertionError(schema)
        return RunnableLambda(obj)


def test_critic_loop_removes_hallucination():
    emb = FakeEmbeddings()
    kb = KnowledgeBase(embeddings=emb)
    kb.add([load_text("I built internal tools in Python with FastAPI and Docker at an agency.", "cv.md")])

    res = run_pipeline(
        kb, "ML intern, Python and Docker", Candidate(name="T"),
        llm=FakeLLM(), embeddings=emb, max_iters=3,
    )

    assert res.iterations == 2  # draft 1 rejected, draft 2 accepted
    assert res.history[0]["critique"].faithfulness < 1
    assert res.history[0]["critique"].unsupported
    assert res.critique.faithfulness == 1.0 and res.critique.passed
    assert all("50 engineers" not in b.text for e in res.cv.projects for b in e.bullets)
    assert "\\begin{document}" in render_latex(res.cv, Candidate(name="T"))
    assert res.cover_letter and res.questions
