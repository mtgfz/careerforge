from src.metrics import strip_evidence_markers
from src.schemas import CVBullet, CVEntry, GeneratedCV


def _cv():
    return GeneratedCV(
        headline="Dev (E1)",
        summary="Builds APIs [E2, E3] with FastAPI.",
        skills=["Python", "Docker (E4)"],
        projects=[
            CVEntry(
                title="Tool",
                bullets=[CVBullet(text="Built a FastAPI tool (E3)", source_ids=["E3"])],
            )
        ],
        education=["2024-Present: Cycle ingenieur, ENSIAS (source: E7)"],
    )


def test_markers_removed_from_text_fields():
    out = strip_evidence_markers(_cv())
    assert out.headline == "Dev"
    assert out.summary == "Builds APIs with FastAPI."
    assert out.skills == ["Python", "Docker"]
    assert out.projects[0].bullets[0].text == "Built a FastAPI tool"
    assert out.education == ["2024-Present: Cycle ingenieur, ENSIAS"]


def test_source_ids_are_kept():
    out = strip_evidence_markers(_cv())
    assert out.projects[0].bullets[0].source_ids == ["E3"]


def test_text_without_markers_is_untouched():
    cv = GeneratedCV(headline="E2E testing (Selenium)", summary="Uses Python 3.12.", skills=["C++"])
    assert strip_evidence_markers(cv) == cv
