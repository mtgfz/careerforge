import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { HistoryItem } from "../types";

const pct = (v: number) => Math.round(v * 100);

export default function CriticLoop({ history }: { history: HistoryItem[] }) {
  if (history.length === 0) {
    return (
      <div className="panel">
        <p>La boucle du Critic était désactivée pour cette exécution : le CV est une version unique.</p>
      </div>
    );
  }
  const data = history.map((h) => ({
    name: `Itér. ${h.iteration}`,
    Fidélité: pct(h.faithfulness),
    ATS: pct(h.ats),
    Qualité: pct(h.quality),
    Global: pct(h.overall),
  }));
  return (
    <div className="stack">
      <div className="panel">
        <h3>Évolution des scores à chaque itération</h3>
        <div className="chart" role="img" aria-label="Évolution des scores par itération">
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: -12 }}>
              <CartesianGrid stroke="#D9DEDD" strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fill: "#4A5A68", fontSize: 13 }} />
              <YAxis domain={[0, 100]} unit=" %" tick={{ fill: "#4A5A68", fontSize: 13 }} />
              <Tooltip formatter={(v: number) => `${v} %`} />
              <Legend />
              <Line type="monotone" dataKey="Fidélité" stroke="#0B6E70" strokeWidth={3} dot={{ r: 5 }} />
              <Line type="monotone" dataKey="ATS" stroke="#B77F00" strokeWidth={2.5} strokeDasharray="6 4" dot={{ r: 4 }} />
              <Line type="monotone" dataKey="Qualité" stroke="#5B6B7A" strokeWidth={2.5} strokeDasharray="2 4" dot={{ r: 4 }} />
              <Line type="monotone" dataKey="Global" stroke="#12263A" strokeWidth={4} dot={{ r: 5 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="panel">
        <h3>Ce que le Critic a rejeté</h3>
        <p className="hint">
          Chaque affirmation sans preuve dans vos documents est signalée, puis corrigée ou retirée.
        </p>
        {history.map((h) => (
          <details key={h.iteration} open={h.iteration === 1}>
            <summary>
              Itération {h.iteration} · fidélité {pct(h.faithfulness)} % ·{" "}
              {h.unsupported?.length ?? 0} point{(h.unsupported?.length ?? 0) > 1 ? "s" : ""} rejeté
              {(h.unsupported?.length ?? 0) > 1 ? "s" : ""}
            </summary>
            {h.unsupported && h.unsupported.length > 0 ? (
              <ul className="rejected">
                {h.unsupported.map((u, i) => (
                  <li key={i}>{u}</li>
                ))}
              </ul>
            ) : (
              <p className="hint">Aucune affirmation rejetée.</p>
            )}
            {h.missing_keywords && h.missing_keywords.length > 0 && (
              <p className="chips">
                <span className="hint">Mots-clés de l’offre absents :</span>
                {h.missing_keywords.map((k) => (
                  <span className="chip warn" key={k}>
                    {k}
                  </span>
                ))}
              </p>
            )}
          </details>
        ))}
      </div>
    </div>
  );
}
