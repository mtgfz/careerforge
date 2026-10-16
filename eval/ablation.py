"""Ablation: full system (with Critic loop) vs. single-pass Writer (no Critic).

Usage:
    python -m eval.ablation --cv data/sample/sample_cv.md --jobs data/sample/jobs

Outputs eval/results.csv, eval/results.md and eval/results.png (ready for the slides).
Both modes are scored by the SAME independent evaluator (evaluate_cv) on the final CV.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.ingestion.loaders import load_file
from src.ingestion.store import KnowledgeBase
from src.pipeline import run_pipeline
from src.schemas import Candidate

METRICS = ["faithfulness", "ats", "quality", "overall"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", nargs="+", default=["data/sample/sample_cv.md"])
    ap.add_argument("--jobs", default="data/sample/jobs")
    ap.add_argument("--out", default="eval")
    args = ap.parse_args()

    kb = KnowledgeBase()
    kb.add([load_file(p) for p in args.cv])
    candidate = Candidate(name="Sample Candidate")

    rows = []
    for job_file in sorted(Path(args.jobs).glob("*.txt")):
        job_text = job_file.read_text(encoding="utf-8")
        for mode, use_critic in (("no_critic", False), ("with_critic", True)):
            res = run_pipeline(kb, job_text, candidate, use_critic=use_critic, with_extras=False)
            row = {"job": job_file.stem, "mode": mode, "iterations": res.iterations}
            row.update({m: getattr(res.critique, m) for m in METRICS})
            rows.append(row)
            print(row)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    df.to_csv(out / "results.csv", index=False)

    summary = df.groupby("mode")[METRICS + ["iterations"]].mean().round(3)
    (out / "results.md").write_text(summary.to_markdown(), encoding="utf-8")
    print("\n", summary.to_markdown())

    import matplotlib

    matplotlib.use("Agg")
    ax = summary[METRICS].T.plot(kind="bar", figsize=(7, 4), rot=0)
    ax.set_ylim(0, 1)
    ax.set_title("CareerForge ablation: Critic loop vs single pass")
    ax.set_ylabel("score (higher is better)")
    ax.figure.tight_layout()
    ax.figure.savefig(out / "results.png", dpi=160)


if __name__ == "__main__":
    main()
