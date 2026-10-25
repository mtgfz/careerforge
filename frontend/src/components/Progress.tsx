import type { NodeId, ProgressEvent, Scores } from "../types";

interface Step {
  key: string;
  node: NodeId;
  iteration: number;
  status: "running" | "done";
  seconds?: number;
  scores?: Scores;
  cached?: boolean;
}

const LABELS: Record<NodeId, string> = {
  ingest: "Lecture et indexation de vos documents",
  analyze: "Analyse de l’offre",
  retrieve: "Recherche des preuves dans vos documents",
  write: "Rédaction du CV",
  critique: "Critique : fidélité, ATS, qualité",
  rejudge: "Revérification après nettoyage",
  extras: "Lettre de motivation et questions d’entretien",
  render: "Mise en page et export PDF",
};

export function buildSteps(events: ProgressEvent[]): Step[] {
  const steps: Step[] = [];
  for (const ev of events) {
    if (!ev.node) continue;
    const key = `${ev.node}-${ev.iteration ?? 0}`;
    if (ev.type === "start") {
      steps.push({ key, node: ev.node, iteration: ev.iteration ?? 0, status: "running" });
    } else if (ev.type === "done") {
      const step = [...steps].reverse().find((s) => s.key === key);
      if (step) {
        step.status = "done";
        step.seconds = ev.seconds;
        step.scores = ev.scores;
        step.cached = ev.cached;
      }
    }
  }
  return steps;
}

const pct = (v: number) => `${Math.round(v * 100)} %`;

export default function Progress({ events, failed }: { events: ProgressEvent[]; failed: boolean }) {
  const steps = buildSteps(events);
  const queued = events.some((e) => e.type === "queued") && steps.length === 0;
  return (
    <div className="panel">
      <h2>Les agents travaillent</h2>
      {queued && <p className="hint">En attente d’une place (2 générations en parallèle au maximum)…</p>}
      <ol className="timeline">
        {steps.map((s) => {
          const state = s.status === "done" ? "done" : failed ? "failed" : "running";
          return (
            <li key={s.key} className={state}>
              <span className="dot" aria-hidden="true" />
              <div>
                <p className="step-title">
                  {LABELS[s.node]}
                  {s.iteration > 0 && (s.node === "write" || s.node === "critique") ? (
                    <span className="iter"> · itération {s.iteration}</span>
                  ) : null}
                </p>
                <p className="step-meta">
                  {s.status === "done"
                    ? `${s.seconds?.toFixed(1) ?? "0.0"} s${s.cached ? " · déjà indexé (cache)" : ""}`
                    : failed
                      ? "interrompu"
                      : "en cours…"}
                </p>
                {s.scores && (
                  <p className="chips">
                    <span className="chip">Fidélité {pct(s.scores.faithfulness)}</span>
                    <span className="chip">ATS {pct(s.scores.ats)}</span>
                    <span className="chip">Qualité {pct(s.scores.quality)}</span>
                    <span className={`chip ${s.scores.passed ? "ok" : "warn"}`}>
                      {s.scores.passed ? "Seuil atteint" : "Seuil non atteint, correction"}
                    </span>
                  </p>
                )}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
