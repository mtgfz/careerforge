"""Speed benchmark: baseline vs optimised pipeline, same inputs, real LLM.

Usage: python scripts/benchmark.py            (3 sample jobs, ~10-15 LLM runs)
       python scripts/benchmark.py --jobs 1   (quick: 1 job)
Writes eval/results/benchmark.csv and benchmark.md (numbers measured on YOUR run only).
"""
import argparse
import csv
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.schemas import Candidate
from src.service import generate_application

MODES = {
    "baseline": dict(PLATEAU_DELTA=0.0, REJUDGE_AFTER_PRUNE=True, PARALLEL_EXTRAS=False),
    "optimise": dict(PLATEAU_DELTA=0.02, REJUDGE_AFTER_PRUNE=False, PARALLEL_EXTRAS=True),
}

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", type=int, default=3)
args = ap.parse_args()

jobs = sorted(Path("data/sample/jobs").glob("*.txt"))[: args.jobs]
cv_text = Path("data/sample/sample_cv.md").read_text(encoding="utf-8")
rows = []
for job in jobs:
    for mode, knobs in MODES.items():
        for k, v in knobs.items():
            setattr(config, k, v)
        t0 = time.perf_counter()
        res = generate_application(
            Candidate(name="Benchmark", email="bench@example.com"),
            job.read_text(encoding="utf-8"),
            extra_text=cv_text,
            out_root="outputs/benchmark",
        )
        total = time.perf_counter() - t0
        last = res["scores"]
        rows.append({"job": job.stem, "mode": mode, "seconds": round(total, 1),
                     "iterations": res["iterations"], **{k: last.get(k, "") for k in ("faithfulness", "ats", "quality", "overall")}})
        print(rows[-1], flush=True)

out = Path("eval/results"); out.mkdir(parents=True, exist_ok=True)
with open(out / "benchmark.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
lines = ["| Mode | Temps moyen (s) | Itérations moy. | Overall moy. |", "|---|---|---|---|"]
for mode in MODES:
    r = [x for x in rows if x["mode"] == mode]
    ov = [x["overall"] for x in r if isinstance(x["overall"], (int, float))]
    lines.append(f"| {mode} | {st.mean(x['seconds'] for x in r):.1f} | {st.mean(x['iterations'] for x in r):.1f} | "
                 f"{st.mean(ov):.2f} |" if ov else f"| {mode} | {st.mean(x['seconds'] for x in r):.1f} | - | - |")
(out / "benchmark.md").write_text("\n".join(lines) + f"\n\n{len(jobs)} offres, 1 run par mode.\n", encoding="utf-8")
print("\n".join(lines))
