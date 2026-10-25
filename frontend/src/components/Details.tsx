import type { RunResult } from "../types";

const LABELS: Record<string, string> = {
  ingest: "Indexation des documents",
  analyze: "Analyse de l’offre",
  retrieve: "Recherche des preuves",
  write: "Rédaction (total)",
  critique: "Critique (total)",
  rejudge: "Revérification",
  extras: "Lettre et questions",
  render: "Export",
};

export default function Details({ result, llm }: { result: RunResult; llm: string }) {
  const entries = Object.entries(result.timings).filter(([, v]) => v > 0);
  const max = Math.max(...entries.map(([, v]) => v), 0.01);
  const total = entries.reduce((a, [, v]) => a + v, 0);
  return (
    <div className="stack">
      <div className="panel">
        <h3>Temps par étape</h3>
        <ul className="timings">
          {entries.map(([k, v]) => (
            <li key={k}>
              <span>{LABELS[k] ?? k}</span>
              <div className="bar" aria-hidden="true">
                <div style={{ width: `${(v / max) * 100}%` }} />
              </div>
              <span className="num">{v.toFixed(1)} s</span>
            </li>
          ))}
        </ul>
        <p className="hint">Total : {total.toFixed(1)} s</p>
      </div>
      <div className="panel">
        <h3>Détails du run</h3>
        <dl className="facts">
          <dt>Offre</dt>
          <dd>{result.job_title}</dd>
          <dt>Modèle</dt>
          <dd>{llm || "n/a"}</dd>
          <dt>Documents indexés</dt>
          <dd>
            {result.kb_chunks} passages{result.kb_cached ? " (cache)" : ""}
          </dd>
          <dt>Itérations</dt>
          <dd>{result.iterations}</dd>
          <dt>Identifiant</dt>
          <dd>{result.run_id}</dd>
        </dl>
      </div>
    </div>
  );
}
