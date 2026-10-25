"""Analysis of the ablation results: tables, paired comparison and charts for the slides.

Usage:
    python -m eval.analysis eval/results.jsonl --out eval

Writes results.csv (one row per run), results.md (summary + per-pair + paired test) and
results.png / trajectory.png.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

METRICS = ["faithfulness", "ats", "quality", "overall"]
EXTRA = ["n_unsupported", "pruned", "iterations", "cv_coverage", "profile_match", "seconds"]
BASELINE = "single_pass"
VARIANT_ORDER = ["single_pass", "critic_once", "full", "low_recall"]

# Reference categorical palette (light mode), fixed order: one slot per variant.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load_rows(path: Path | str) -> pd.DataFrame:
    rows = [json.loads(ln) for ln in Path(path).read_text("utf-8").splitlines() if ln.strip()]
    df = pd.DataFrame(rows)
    if "error" not in df:
        df["error"] = None
    return df


def _order(variants) -> list[str]:
    return [v for v in VARIANT_ORDER if v in set(variants)] + sorted(set(variants) - set(VARIANT_ORDER))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std per variant over successful runs."""
    ok = df[df["error"].isna()]
    cols = [c for c in METRICS + EXTRA if c in ok]
    g = ok.groupby("variant")[cols]
    mean, std = g.mean(), g.std(ddof=1).fillna(0.0)
    out = pd.DataFrame(index=mean.index)
    for c in cols:
        out[c] = mean[c]
        if c in METRICS:
            out[c + "_std"] = std[c]
    if "passed" in ok:
        out["pass_rate"] = ok.groupby("variant")["passed"].mean()
    out["runs"] = g.size()
    return out.loc[_order(out.index)]


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def paired(df: pd.DataFrame, metric: str, baseline: str = BASELINE, eps: float = 0.005) -> pd.DataFrame:
    """Per-pair mean of each variant vs the baseline: mean delta, wins/ties/losses, sign test."""
    ok = df[df["error"].isna()]
    per_pair = ok.groupby(["pair", "variant"])[metric].mean().unstack("variant")
    if baseline not in per_pair:
        return pd.DataFrame()
    rows = []
    for v in _order(per_pair.columns):
        if v == baseline:
            continue
        d = (per_pair[v] - per_pair[baseline]).dropna()
        wins, losses = int((d > eps).sum()), int((d < -eps).sum())
        rows.append({
            "variant": v,
            "pairs": len(d),
            "mean_delta": round(d.mean(), 3) if len(d) else float("nan"),
            "wins": wins,
            "ties": len(d) - wins - losses,
            "losses": losses,
            "sign_test_p": round(sign_test(wins, losses), 3),
        })
    return pd.DataFrame(rows).set_index("variant") if rows else pd.DataFrame()


def per_pair_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    ok = df[df["error"].isna()]
    t = ok.groupby(["pair", "variant"])[metric].mean().unstack("variant")
    return t[_order(t.columns)].round(3)


def trajectory(df: pd.DataFrame, variant: str = "full") -> pd.DataFrame:
    """Mean score per Critic iteration (only runs that reached that iteration)."""
    ok = df[(df["error"].isna()) & (df["variant"] == variant)]
    rows = [h for hist in ok.get("history", []) if isinstance(hist, list) for h in hist]
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows).groupby("iteration")[METRICS].mean()
    t["runs"] = pd.DataFrame(rows).groupby("iteration").size()
    return t.round(3)


# ---------- charts ----------
def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    variants = list(summary.index)
    n = len(variants)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _style(ax)
    for j, v in enumerate(variants):
        color = SERIES[j % len(SERIES)]
        for i, m in enumerate(METRICS):
            x = i - 0.4 + j * width + 0.02
            h = float(summary.loc[v, m])
            ax.add_patch(FancyBboxPatch(
                (x, 0), width - 0.04, h, boxstyle="round,pad=0,rounding_size=0.015",
                facecolor=color, edgecolor=SURFACE, linewidth=2,
            ))
            err = float(summary.loc[v].get(m + "_std", 0) or 0)
            if err:
                ax.plot([x + (width - 0.04) / 2] * 2, [max(0, h - err), min(1, h + err)],
                        color=INK_2, linewidth=1)
            ax.text(x + (width - 0.04) / 2, min(h + err, 1) + 0.015, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7, color=INK_2)
        ax.bar(0, 0, color=color, label=v.replace("_", " "))  # legend handle
    ax.set_xlim(-0.5, len(METRICS) - 0.5)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(range(len(METRICS)), [m.upper() if m == "ats" else m.capitalize() for m in METRICS], color=INK)
    ax.set_ylabel("score (mean ± std, higher is better)", color=INK_2)
    ax.set_title("CareerForge ablation: what the Critic loop adds", color=INK, loc="left")
    ax.legend(frameon=False, ncol=n, loc="upper left", bbox_to_anchor=(0, -0.08), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_trajectory(traj: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.6))
    _style(ax)
    for k, m in enumerate(METRICS):
        ax.plot(traj.index, traj[m], marker="o", markersize=6, linewidth=2,
                color=SERIES[k], markeredgecolor=SURFACE, markeredgewidth=2, label=m)
        ax.annotate(m, (traj.index[-1], traj[m].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.set_xticks(list(traj.index))
    ax.set_xlabel("Critic iteration", color=INK_2)
    ax.set_ylim(0, 1.05)
    ax.set_title("Scores across Writer ↔ Critic iterations (full system)", color=INK, loc="left")
    ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.18), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ---------- report ----------
def write_report(results: Path | str, out: Path | str = "eval") -> str:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_rows(results)
    flat = df.drop(columns=["history"], errors="ignore")
    flat.to_csv(out / "results.csv", index=False)

    ok = df[df["error"].isna()]
    failed = df[df["error"].notna()]
    lines = ["# CareerForge evaluation results", ""]
    lines.append(
        f"{ok['pair'].nunique()} CV/offer pairs, {len(ok)} successful runs"
        + (f", {len(failed)} failed" if len(failed) else "") + "."
    )
    if ok.empty:
        text = "\n".join(lines + ["", "No successful run yet."]) + "\n"
        (out / "results.md").write_text(text, encoding="utf-8")
        print(text)
        return text

    summary = summarize(df)
    show = [c for c in METRICS + ["pass_rate", "n_unsupported", "pruned", "iterations", "cv_coverage", "seconds", "runs"] if c in summary]
    fmt = summary[show].copy()
    for m in METRICS:
        fmt[m] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(summary[m], summary[m + "_std"])]
    lines += ["", "## Summary (mean ± std per variant)", "", fmt.round(3).to_markdown()]

    for metric in ("faithfulness", "overall"):
        p = paired(df, metric)
        if not p.empty:
            lines += ["", f"## {metric.capitalize()}: paired comparison vs `{BASELINE}`", "",
                      p.to_markdown(),
                      "", "_Wins/losses count pairs where the variant beats/trails the baseline by "
                      "more than 0.005; p is a two-sided exact sign test (few pairs: read it as a "
                      "trend, not a proof)._"]

    for metric in ("faithfulness", "overall"):
        lines += ["", f"## {metric.capitalize()} per pair", "", per_pair_table(df, metric).to_markdown()]

    traj = trajectory(df)
    if not traj.empty:
        lines += ["", "## Critic loop trajectory (`full`)", "", traj.to_markdown()]
        plot_trajectory(traj, out / "trajectory.png")

    if len(failed):
        lines += ["", "## Failed runs", ""]
        lines += [f"- {r.pair} / {r.variant}: {r.error}" for r in failed.itertuples()]

    plot_summary(summary, out / "results.png")
    text = "\n".join(lines) + "\n"
    (out / "results.md").write_text(text, encoding="utf-8")
    print(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="eval/results.jsonl")
    ap.add_argument("--out", default="eval")
    args = ap.parse_args(argv)
    write_report(args.results, args.out)


if __name__ == "__main__":
    main()
