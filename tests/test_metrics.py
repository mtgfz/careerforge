from src.metrics import (
    contains_term,
    cosine,
    cv_to_text,
    keyword_coverage,
    skill_grounded_flags,
    valid_citation_flags,
)
from src.schemas import CVBullet, CVEntry, GeneratedCV


def make_cv():
    return GeneratedCV(
        headline="ML intern",
        summary="Student in data science.",
        skills=["Python", "Docker", "Kubernetes"],
        projects=[
            CVEntry(
                title="Forecasting",
                bullets=[
                    CVBullet(text="Trained LSTM models with PyTorch", source_ids=["E1"]),
                    CVBullet(text="Led a team of 50", source_ids=[]),
                    CVBullet(text="Deployed on GCP", source_ids=["E9"]),
                ],
            )
        ],
    )


def test_contains_term_handles_special_names():
    assert contains_term("I know C++ and C# and Node.js", "C++")
    assert contains_term("I know C++ and C# and Node.js", "C#")
    assert contains_term("I know C++ and C# and Node.js", "node.js")
    assert not contains_term("I know Javascript", "Java")
    assert not contains_term("anything", "")


def test_keyword_coverage():
    cov, missing = keyword_coverage("Python and Docker", ["Python", "Docker", "Kubernetes", "MLflow"])
    assert cov == 0.5
    assert missing == ["Kubernetes", "MLflow"]
    assert keyword_coverage("x", []) == (1.0, [])


def test_cosine():
    assert abs(cosine([1, 0], [1, 0]) - 1) < 1e-9
    assert abs(cosine([1, 0], [0, 1])) < 1e-9
    assert cosine([0, 0], [1, 1]) == 0.0


def test_valid_citation_flags():
    cv = make_cv()
    # E1 valid, empty citation invalid, E9 unknown id invalid
    assert valid_citation_flags(cv, {"E1", "E2"}) == [True, False, False]


def test_skill_grounding():
    flags = skill_grounded_flags(["Python", "Kubernetes"], "worked with python and docker")
    assert flags == [True, False]


def test_cv_to_text_contains_everything():
    text = cv_to_text(make_cv())
    assert "ML intern" in text and "LSTM" in text and "Kubernetes" in text
