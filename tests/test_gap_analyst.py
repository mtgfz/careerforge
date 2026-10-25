"""Gap analyst: deterministic requirement classification, scores, report and advice step."""
from langchain_core.runnables import RunnableLambda

from src.agents.gap_analyst import GapAdvice, RequirementAdvice, analyze_gaps, classify
from src.schemas import (
    CVBullet,
    CVEntry,
    EvidenceChunk,
    EvidencePack,
    GeneratedCV,
    JobProfile,
    Requirement,
)

JOB = JobProfile(
    title="Data Engineer Intern",
    company="Example Bank",
    requirements=[
        Requirement(name="Python and SQL", kind="hard_skill", importance=5),
        Requirement(name="Docker", kind="hard_skill", importance=4),
        Requirement(name="Data pipelines", kind="experience", importance=3),
        Requirement(name="Spark", kind="hard_skill", importance=2),
    ],
    keywords=["Python", "SQL", "Docker", "Spark"],
)

EVIDENCE = EvidencePack(
    chunks={
        "E1": EvidenceChunk(id="E1", text="Wrote ETL scripts in Python and SQL.", source="cv", score=0.62),
        "E2": EvidenceChunk(id="E2", text="Deployed the app with Docker.", source="cv", score=0.55),
        "E3": EvidenceChunk(id="E3", text="Automated a weekly report export.", source="cv", score=0.36),
    },
    by_requirement={
        "Python and SQL": ["E1"],
        "Docker": ["E2"],
        "Data pipelines": ["E3"],
        "Spark": [],
    },
)

CV = GeneratedCV(
    headline="Data student",
    summary="Student.",
    skills=["Python", "SQL"],
    projects=[CVEntry(title="ETL", bullets=[CVBullet(text="Wrote ETL scripts in Python and SQL", source_ids=["E1"])])],
)


def statuses(items):
    return {i.requirement: i.status for i in items}


def test_classification():
    items = classify(JOB, EVIDENCE, CV)
    assert statuses(items) == {
        "Python and SQL": "matched",  # evidence + in the CV
        "Docker": "not_shown",  # evidence, but the CV forgot it
        "Data pipelines": "partial",  # only weakly related evidence
        "Spark": "missing",
    }
    assert [i.importance for i in items] == [5, 4, 3, 2]  # most important first


def test_without_a_cv_nothing_is_matched():
    assert "matched" not in statuses(classify(JOB, EVIDENCE, None)).values()


def test_unretrieved_requirement_is_searched_in_the_kb():
    class KB:
        def search(self, query, k=1):
            return [("Built Spark jobs on a cluster.", "cv", 0.7)] if query == "Spark" else []

    pack = EVIDENCE.model_copy(update={"by_requirement": {k: v for k, v in EVIDENCE.by_requirement.items() if k != "Spark"}})
    assert statuses(classify(JOB, pack, CV, kb=KB()))["Spark"] == "not_shown"


def test_scores_and_markdown():
    report = analyze_gaps(JOB, EVIDENCE, CV, advise=False)
    # profile: 5*1 + 4*1 + 3*0.5 + 2*0 = 10.5 / 14 ; cv: only Python and SQL -> 5 / 14
    assert report.profile_match == round(10.5 / 14, 3)
    assert report.cv_coverage == round(5 / 14, 3)
    assert report.missing_keywords == ["Docker", "Spark"]
    assert report.counts() == {"matched": 1, "not_shown": 1, "partial": 1, "missing": 1}
    md = report.to_markdown()
    assert "# Match report: Data Engineer Intern at Example Bank" in md
    assert "| Docker | 4 | In profile, not in CV | E2 (0.55) |" in md


def test_advice_is_attached_to_each_gap():
    seen = {}

    class FakeLLM:
        def with_structured_output(self, schema, **kw):
            assert schema is GapAdvice

            def run(prompt_value):
                seen["prompt"] = prompt_value.to_string()
                return GapAdvice(
                    advice=[
                        RequirementAdvice(requirement="docker", advice="Put the Docker deployment forward."),
                        RequirementAdvice(requirement="Spark", advice="Do a small PySpark project."),
                    ],
                    summary="Good fit, Spark is the main gap.",
                )

            return RunnableLambda(run)

    report = analyze_gaps(JOB, EVIDENCE, CV, llm=FakeLLM())
    advice = {i.requirement: i.advice for i in report.items}
    assert advice["Docker"] == "Put the Docker deployment forward."  # case-insensitive match
    assert advice["Spark"] == "Do a small PySpark project."
    assert advice["Python and SQL"] == ""
    assert report.summary.startswith("Good fit")
    assert "Python and SQL [status" not in seen["prompt"]  # matched ones are not sent as gaps
