"""Ablation: what does each part of the Critic loop bring?

Variants (all scored by the SAME evaluator, evaluate_cv, on the CV that is returned):
    single_pass   Writer only, no Critic                       (baseline)
    critic_once   one Critic pass: unsupported claims are pruned, no rewrite
    full          full system: Critic <-> Writer loop (MAX_ITERATIONS) + pruning
    low_recall    full system with 1 evidence chunk per requirement (retrieval ablation)

Usage:
    python -m eval.ablation                         # data/private/pairs, else the sample pairs
    python -m eval.ablation --sample --repeats 2
    python -m eval.ablation --pairs data/private/pairs --variants single_pass full
    python -m eval.ablation --cv data/sample/sample_cv.md --jobs data/sample/jobs   # legacy

Every run re-judges the CV after pruning (REJUDGE_AFTER_PRUNE is forced on), so faithfulness
is measured and never fixed at 1.0 by construction.
Every run is appended to <out>/results.jsonl as soon as it finishes, so an interrupted run
(free-tier quota...) resumes where it stopped. Use --fresh to start over.
Then eval.analysis writes results.md, results.csv and the charts.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from src import config

from .pairs import Pair, build_kb, resolve_pairs, sample_pairs

METRICS = ["faithfulness", "ats", "quality", "overall"]

VARIANTS: dict[str, dict] = {
    "single_pass": {"use_critic": False},
    "critic_once": {"use_critic": True, "max_iters": 1},
    "full": {"use_critic": True},
    "low_recall": {"use_critic": True, "config": {"K_PER_REQUIREMENT": 1}},
}
DEFAULT_VARIANTS = ["single_pass", "critic_once", "full"]

# Applied to every run whatever .env says: the CV returned after pruning must be judged again,
# otherwise its faithfulness is 1.0 by construction and the ablation measures nothing.
EVAL_CONFIG = {"REJUDGE_AFTER_PRUNE": True}

MAX_QUOTA_WAITS = 48  # with --wait-quota: at most ~16 h of waiting in one session
_sleep = time.sleep  # replaced in tests


@contextmanager
def overridden(values: dict):
    old = {k: getattr(config, k) for k in values}
    try:
        for k, v in values.items():
            setattr(config, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(config, k, v)


def run_one(pair: Pair, variant: str, kb, repeat: int = 0, llm=None, embeddings=None) -> dict:
    """Runs the pipeline once and flattens everything the analysis needs into one row."""
    from src.agents.gap_analyst import analyze_gaps
    from src.metrics import all_bullets
    from src.pipeline import run_pipeline
    from src.schemas import Candidate

    spec = VARIANTS[variant]
    row = {"pair": pair.id, "variant": variant, "repeat": repeat}
    t0 = time.perf_counter()
    try:
        with overridden({**EVAL_CONFIG, **spec.get("config", {})}):
            res = run_pipeline(
                kb,
                pair.job_text,
                Candidate(name=pair.candidate_name),
                use_critic=spec["use_critic"],
                max_iters=spec.get("max_iters"),
                with_extras=False,
                llm=llm,
                embeddings=embeddings,
            )
    except Exception as e:  # keep going: one failed run must not lose the others
        from src.llm import QuotaExceeded

        row.update(error=f"{type(e).__name__}: {e}"[:500], seconds=round(time.perf_counter() - t0, 1))
        if isinstance(e, QuotaExceeded) and e.daily:
            row["daily_quota"] = True
            row["quota_wait"] = e.wait
        traceback.print_exc()
        return row

    crit = res.critique
    gaps = analyze_gaps(res.job, res.evidence, res.cv, advise=False)
    row.update({m: getattr(crit, m) for m in METRICS})
    row.update(
        passed=crit.passed,
        iterations=res.iterations,
        best_iteration=getattr(res, "best_iteration", res.iterations),
        pruned=getattr(res, "pruned", 0),
        n_unsupported=len(crit.unsupported),
        n_bullets=len(list(all_bullets(res.cv))),
        n_skills=len(res.cv.skills),
        n_evidence=len(res.evidence.chunks),
        missing_keywords=len(crit.missing_keywords),
        keyword_coverage=gaps.keyword_coverage,
        profile_match=gaps.profile_match,
        cv_coverage=gaps.cv_coverage,
        job_title=res.job.title,
        job_language=res.job.language,
        plateau_delta=config.PLATEAU_DELTA,
        seconds=round(time.perf_counter() - t0, 1),
        history=[
            {"iteration": h["iteration"], **{m: getattr(h["critique"], m) for m in METRICS}}
            for h in res.history
        ],
    )
    return row


def _done(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("error"):
                keys.add((r["pair"], r["variant"], r["repeat"]))
    return keys


def run_ablation(
    pairs: list[Pair],
    variants: list[str] = DEFAULT_VARIANTS,
    repeats: int = 1,
    out: Path | str = "eval",
    fresh: bool = False,
    llm=None,
    embeddings=None,
    wait_quota: bool = False,
    max_wait: float = 20 * 60,
) -> Path:
    """wait_quota: when the daily quota is used up, sleep (Groq's window is rolling, tokens come
    back within minutes) and retry the same run instead of stopping, at most MAX_QUOTA_WAITS times."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.jsonl"
    if fresh and path.exists():
        path.unlink()
    if path.exists():  # failed runs are retried: drop their rows
        kept = [ln for ln in path.read_text("utf-8").splitlines() if ln.strip() and not json.loads(ln).get("error")]
        path.write_text("".join(ln + "\n" for ln in kept), encoding="utf-8")
    done = _done(path)

    # Repeat-major order: if the quota stops the run, every pair has the same number of repeats.
    todo = [(p, v, r) for r in range(repeats) for p in pairs for v in variants if (p.id, v, r) not in done]
    print(f"[ablation] {len(pairs)} pairs x {len(variants)} variants x {repeats} repeats: "
          f"{len(todo)} runs to do, {len(done)} already done")
    kbs: dict[str, object] = {}
    waits, n = 0, 0
    queue = list(todo)
    while queue:
        pair, variant, rep = queue[0]
        if pair.id not in kbs:
            kbs[pair.id] = build_kb(pair, embeddings=embeddings)
        row = run_one(pair, variant, kbs[pair.id], rep, llm=llm, embeddings=embeddings)
        if row.get("daily_quota") and wait_quota and waits < MAX_QUOTA_WAITS:
            waits += 1
            delay = min(max(row.get("quota_wait") or 0, 60), max_wait) + 5
            print(f"[ablation] daily quota used up: waiting {delay / 60:.0f} min, then retrying "
                  f"{pair.id} / {variant} / r{rep} (wait {waits}/{MAX_QUOTA_WAITS})", flush=True)
            _sleep(delay)
            continue
        queue.pop(0)
        n += 1
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        brief = row.get("error") or "  ".join(f"{m}={row[m]:.2f}" for m in METRICS)
        print(f"[{n}/{len(todo)}] {pair.id} / {variant} / r{rep}: {brief}", flush=True)
        if row.get("daily_quota"):
            print(f"[ablation] daily quota used up: stopping after {n - 1} new runs. "
                  "Run the same command again later (without --fresh) to resume, "
                  "or use --wait-quota.")
            break
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", help="directory of pairs (default: data/private/pairs)")
    ap.add_argument("--sample", action="store_true", help="use the fictional sample pairs")
    ap.add_argument("--cv", nargs="+", help="legacy: one CV for every offer in --jobs")
    ap.add_argument("--jobs", default="data/sample/jobs", help="legacy: directory of *.txt offers")
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(VARIANTS))
    ap.add_argument("--repeats", type=int, default=1, help="runs per pair and variant")
    ap.add_argument("--out", default="eval")
    ap.add_argument("--fresh", action="store_true", help="ignore previous results")
    ap.add_argument("--wait-quota", action="store_true",
                    help="when the daily quota is used up, wait and continue instead of stopping")
    args = ap.parse_args(argv)

    if args.cv:
        pairs = sample_pairs(Path(args.cv[0]), Path(args.jobs))
        for p in pairs:
            p.cv_files = [Path(c) for c in args.cv]
    else:
        pairs = resolve_pairs(args.pairs, args.sample)

    path = run_ablation(pairs, args.variants, args.repeats, args.out, args.fresh,
                        wait_quota=args.wait_quota)

    from .analysis import write_report

    write_report(path, args.out)


if __name__ == "__main__":
    main()
