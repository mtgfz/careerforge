"""Render a GeneratedCV to LaTeX (+ PDF when a LaTeX engine is installed) and Markdown."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..schemas import Candidate, CVEntry, GeneratedCV

_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=1.6cm]{geometry}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{xcolor}
\usepackage{titlesec}
\definecolor{accent}{HTML}{1F4E79}
\titleformat{\section}{\large\bfseries\color{accent}}{}{0em}{}[\titlerule]
\titlespacing*{\section}{0pt}{10pt}{5pt}
\setlist[itemize]{leftmargin=1.2em,itemsep=1pt,topsep=2pt}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\begin{document}
"""


def latex_escape(text: str) -> str:
    return "".join(_ESCAPES.get(ch, ch) for ch in text)


def _href(url: str) -> str:
    safe = url.replace("%", r"\%").replace("#", r"\#")
    label = latex_escape(url.replace("https://", "").replace("http://", "").rstrip("/"))
    return "\\href{" + safe + "}{" + label + "}"


def _entry(e: CVEntry) -> str:
    head = "\\textbf{" + latex_escape(e.title) + "}"
    if e.period:
        head += " \\hfill " + latex_escape(e.period)
    out = [head + "\\\\"]
    if e.organization:
        out.append("\\textit{" + latex_escape(e.organization) + "}")
    if e.bullets:
        out.append("\\begin{itemize}")
        out.extend("\\item " + latex_escape(b.text) for b in e.bullets)
        out.append("\\end{itemize}")
    return "\n".join(out) + "\n"


def _section(title: str, body: str) -> str:
    return "\\section*{" + latex_escape(title) + "}\n" + body + "\n"


def render_latex(cv: GeneratedCV, candidate: Candidate) -> str:
    parts = [PREAMBLE]
    parts.append("\\begin{center}\n{\\Huge\\bfseries " + latex_escape(candidate.name) + "}\\\\[3pt]")
    parts.append("{\\large " + latex_escape(cv.headline) + "}\\\\[3pt]")
    contact = [latex_escape(x) for x in (candidate.email, candidate.phone, candidate.location) if x]
    contact += [_href(u) for u in candidate.links]
    parts.append(" \\textbar{} ".join(contact) + "\n\\end{center}\n")

    if cv.summary:
        parts.append(_section("Profile", latex_escape(cv.summary)))
    if cv.skills:
        parts.append(_section("Skills", latex_escape(" \u00b7 ".join(cv.skills))))
    if cv.experiences:
        parts.append(_section("Experience", "\n".join(_entry(e) for e in cv.experiences)))
    if cv.projects:
        parts.append(_section("Projects", "\n".join(_entry(e) for e in cv.projects)))
    if cv.education:
        edu = "\\begin{itemize}\n" + "\n".join("\\item " + latex_escape(x) for x in cv.education)
        parts.append(_section("Education", edu + "\n\\end{itemize}"))
    if cv.languages:
        parts.append(_section("Languages", latex_escape(", ".join(cv.languages))))
    parts.append("\\end{document}\n")
    return "\n".join(parts)


def render_markdown(cv: GeneratedCV, candidate: Candidate) -> str:
    contact = " | ".join(
        x for x in [candidate.email, candidate.phone, candidate.location, *candidate.links] if x
    )
    lines = [f"# {candidate.name}", f"**{cv.headline}**", "", contact]
    if cv.summary:
        lines += ["", "## Profile", cv.summary]
    if cv.skills:
        lines += ["", "## Skills", " · ".join(cv.skills)]
    for title, entries in (("Experience", cv.experiences), ("Projects", cv.projects)):
        if entries:
            lines += ["", f"## {title}"]
            for e in entries:
                period = f" ({e.period})" if e.period else ""
                org = f" - {e.organization}" if e.organization else ""
                lines.append(f"**{e.title}**{org}{period}")
                lines += [f"- {b.text}" for b in e.bullets]
                lines.append("")
    if cv.education:
        lines += ["## Education", *[f"- {x}" for x in cv.education]]
    if cv.languages:
        lines += ["", "## Languages", ", ".join(cv.languages)]
    return "\n".join(lines).strip() + "\n"


def compile_pdf(tex_path: Path) -> Path | None:
    """Compile with pdflatex or tectonic if installed. Returns the PDF path or None."""
    tex_path = Path(tex_path)
    engine = shutil.which("pdflatex") or shutil.which("tectonic")
    if not engine:
        return None
    if Path(engine).name == "tectonic":
        cmd = [engine, tex_path.name]
    else:
        cmd = [engine, "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    try:
        subprocess.run(cmd, cwd=tex_path.parent, check=True, capture_output=True, timeout=90)
    except (subprocess.SubprocessError, OSError):
        return None
    pdf = tex_path.with_suffix(".pdf")
    return pdf if pdf.exists() else None
