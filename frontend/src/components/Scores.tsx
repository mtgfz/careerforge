import type { RunResult } from "../types";

const pct = (v: number) => Math.round(v * 100);

function tone(v: number): "good" | "mid" | "low" {
  if (v >= 0.85) return "good";
  if (v >= 0.6) return "mid";
  return "low";
}

const WORD = { good: "bon", mid: "moyen", low: "faible" } as const;

function Card({ label, value, hint }: { label: string; value: number; hint: string }) {
  const t = tone(value);
  return (
    <div className={`score ${t}`}>
      <p className="score-label">{label}</p>
      <p className="score-value">
        {pct(value)}
        <span> %</span>
      </p>
      <div className="bar" aria-hidden="true">
        <div style={{ width: `${pct(value)}%` }} />
      </div>
      <p className="score-hint">
        {WORD[t]} · {hint}
      </p>
    </div>
  );
}

export default function Scores({ result }: { result: RunResult }) {
  const s = result.scores;
  return (
    <section aria-label="Scores">
      <div className="scores">
        <Card label="Fidélité" value={s.faithfulness} hint="affirmations prouvées" />
        <Card label="Score ATS" value={s.ats} hint="mots-clés et sens" />
        <Card label="Qualité" value={s.quality} hint="clarté et impact" />
        <Card label="Global" value={s.overall} hint={s.passed ? "seuil atteint" : "sous le seuil"} />
      </div>
      <p className="summary-line">
        {result.iterations} itération{result.iterations > 1 ? "s" : ""}
        {result.best_iteration > 0 ? ` · version retenue : itération ${result.best_iteration}` : ""}
        {result.pruned > 0
          ? ` · ${result.pruned} affirmation${result.pruned > 1 ? "s" : ""} non prouvée${result.pruned > 1 ? "s" : ""} retirée${result.pruned > 1 ? "s" : ""}`
          : ""}
        {result.replay ? " · rediffusion d’une exécution enregistrée" : ""}
      </p>
    </section>
  );
}
