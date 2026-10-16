from src.render.latex import latex_escape, render_latex, render_markdown
from src.schemas import Candidate, CVBullet, CVEntry, GeneratedCV


def sample():
    cv = GeneratedCV(
        headline="Data & ML Intern",
        summary="Built tools at 100% focus_",
        skills=["C#", "Python"],
        experiences=[
            CVEntry(
                title="Intern",
                organization="Agency",
                period="2026",
                bullets=[CVBullet(text="Cut sorting time by 40% using FastAPI & Docker", source_ids=["E1"])],
            )
        ],
        education=["Engineering degree"],
        languages=["French"],
    )
    cand = Candidate(name="Test User", email="t@example.com", links=["https://github.com/test_user"])
    return cv, cand


def test_escape_special_characters():
    assert latex_escape("R&D 100% #1 a_b") == r"R\&D 100\% \#1 a\_b"


def test_latex_is_well_formed_and_escaped():
    cv, cand = sample()
    tex = render_latex(cv, cand)
    assert tex.startswith("\\documentclass") and tex.strip().endswith("\\end{document}")
    assert "\\begin{itemize}" in tex and "\\end{itemize}" in tex
    assert "Data \\& ML Intern" in tex
    assert "40\\% using FastAPI \\& Docker" in tex
    assert "focus\\_" in tex
    assert tex.count("\\begin{itemize}") == tex.count("\\end{itemize}")


def test_markdown_render():
    cv, cand = sample()
    md = render_markdown(cv, cand)
    assert md.startswith("# Test User")
    assert "- Cut sorting time" in md
