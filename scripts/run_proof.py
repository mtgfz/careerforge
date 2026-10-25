"""Proof run: the full system on every CV/offer pair, with every artefact saved for review.

For each pair it runs the complete pipeline (Critic loop + cover letter + interview questions),
the gap analyst, and writes to <out>/<pair>/:

    cv.md, cv.tex (+ cv.pdf if LaTeX is installed)   the tailored CV
    cover_letter.txt, interview_questions.txt
    gap_report.md                                     requirement-by-requirement match + advice
    trace.json                                        job profile, evidence, every Critic pass
    run.json                                          scores and timings

and <out>/index.md with one line per pair (the table for the README / slides).

Usage:
    python scripts/run_proof.py                     # data/private/pairs, else the samples
    python scripts/run_proof.py --sample --no-pdf
    python scripts/run_proof.py --pairs data/private/pairs --only 01_ml_intern

The default output folder (outputs/proof) is git-ignored: real CVs are personal data.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.ablation import EVAL_CONFIG, overridden  # noqa: E402
from eval.pairs import Pair, build_kb, resolve_pairs  # noqa: E402

DEFAULT_OUT = Path("outputs/proof")


def _critique_dict(c) -> dict:
    return c.model_dump(exclude={"feedback"})


def prove_pair(pair: Pair, out: Path, pdf: bool = True, llm=None, embeddings=None) -> dict:
    from src.agents.gap_analyst import analyze_gaps
    from src.pipeline import run_pipeline
    from src.render.latex import compile_pdf, render_latex, render_markdown
    from src.schemas import Candidate

    out.mkdir(parents=True, exist_ok=True)
    candidate = Candidate(name=pair.candidate_name)
    t0 = time.perf_counter()
    kb = build_kb(pair, embeddings=embeddings)
    with overridden(EVAL_CONFIG):  # scores in index.md must be measured, not assumed
        res = run_pipeline(kb, pair.job_text, candidate, llm=llm, embeddings=embeddings)
    t_pipeline = time.perf_counter() - t0
    gaps = analyze_gaps(res.job, res.evidence, res.cv, kb=kb, llm=llm)

    (out / "cv.md").write_text(render_markdown(res.cv, candidate), encoding="utf-8")
    tex = out / "cv.tex"
    tex.write_text(render_latex(res.cv, candidate), encoding="utf-8")
    has_pdf = bool(pdf and compile_pdf(tex))
    (out / "cover_letter.txt").write_text(res.cover_letter, encoding="utf-8")
    (out / "interview_questions.txt").write_text(res.questions, encoding="utf-8")
    (out / "gap_report.md").write_text(gaps.to_markdown(), encoding="utf-8")
    (out / "trace.json").write_text(
        json.dumps(
            {
                "job_source": pair.job_source,
                "job": res.job.model_dump(),
                "evidence": res.evidence.model_dump(),
                "history": [
                    {"iteration": h["iteration"], "critique": h["critique"].model_dump(),
                     "cv": h["cv"].model_dump() if "cv" in h else None}
                    for h in res.history
                ],
                "final_critique": res.critique.model_dump(),
                "gap_report": gaps.model_dump(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    summary = {
        "pair": pair.id,
        "job_title": res.job.title,
        "language": res.job.language,
        "iterations": res.iterations,
        "best_iteration": getattr(res, "best_iteration", res.iterations),
        "pruned": getattr(res, "pruned", 0),
        **_critique_dict(res.critique),
        "profile_match": gaps.profile_match,
        "cv_coverage": gaps.cv_coverage,
        "gaps": gaps.counts(),
        "pdf": has_pdf,
        "seconds": round(time.perf_counter() - t0, 1),
        "pipeline_seconds": round(t_pipeline, 1),
    }
    (out / "run.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def write_index(rows: list[dict], out: Path) -> str:
    lines = [
        "# CareerForge proof run",
        "",
        "| Pair | Job | Lang | Iter. | Faithfulness | ATS | Quality | Overall | Passed "
        "| Profile match | CV coverage | Missing reqs | Time (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("error"):
            lines.append(f"| {r['pair']} | **failed**: {r['error'][:120]} |" + " |" * 11)
            continue
        lines.append(
            f"| [{r['pair']}]({r['pair']}/cv.md) | {r['job_title']} | {r['language']} "
            f"| {r['iterations']} | {r['faithfulness']:.2f} | {r['ats']:.2f} | {r['quality']:.2f} "
            f"| {r['overall']:.2f} | {'yes' if r['passed'] else 'no'} | {r['profile_match']:.0%} "
            f"| {r['cv_coverage']:.0%} | {r['gaps']['missing']} | {r['seconds']} |"
        )
    ok = [r for r in rows if not r.get("error")]
    if ok:
        mean = {k: sum(r[k] for r in ok) / len(ok) for k in ("faithfulness", "ats", "quality", "overall")}
        lines += [
            "",
            f"**Mean over {len(ok)} pairs**: faithfulness {mean['faithfulness']:.3f}, "
            f"ATS {mean['ats']:.3f}, quality {mean['quality']:.3f}, overall {mean['overall']:.3f}; "
            f"{sum(r['passed'] for r in ok)}/{len(ok)} passed the Critic thresholds.",
        ]
    text = "\n".join(lines) + "\n"
    (out / "index.md").write_text(text, encoding="utf-8")
    return text


def main(argv=None, llm=None, embeddings=None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", help="directory of pairs (default: data/private/pairs)")
    ap.add_argument("--sample", action="store_true", help="use the fictional sample pairs")
    ap.add_argument("--only", nargs="+", help="pair ids to run")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-pdf", action="store_true", help="skip LaTeX compilation")
    args = ap.parse_args(argv)

    pairs = resolve_pairs(args.pairs, args.sample)
    if args.only:
        pairs = [p for p in pairs if p.id in args.only]
    out = Path(args.out)
    rows = []
    for n, pair in enumerate(pairs, 1):
        print(f"[{n}/{len(pairs)}] {pair.id} ...", flush=True)
        try:
            row = prove_pair(pair, out / pair.id, pdf=not args.no_pdf, llm=llm, embeddings=embeddings)
            print(f"    faithfulness={row['faithfulness']:.2f} overall={row['overall']:.2f} "
                  f"passed={row['passed']} ({row['seconds']}s)")
        except Exception as e:  # report and continue with the next pair
            traceback.print_exc()
            row = {"pair": pair.id, "error": f"{type(e).__name__}: {e}"}
        rows.append(row)
    print("\n" + write_index(rows, out))
    print(f"Artefacts in {out}/")
    return rows


if __name__ == "__main__":
    main()
