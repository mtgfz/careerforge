import { useEffect, useState } from "react";
import { getConfig, listReplays } from "./api";
import CVEditor from "./components/CVEditor";
import CriticLoop from "./components/CriticLoop";
import Details from "./components/Details";
import ProfileForm from "./components/ProfileForm";
import Progress from "./components/Progress";
import Scores from "./components/Scores";
import TextPanel from "./components/TextPanel";
import type { AppConfig, ReplayInfo } from "./types";
import { useJob } from "./useJob";

type Tab = "cv" | "critic" | "letter" | "questions" | "details";
const TABS: [Tab, string][] = [
  ["cv", "CV"],
  ["critic", "Boucle du Critic"],
  ["letter", "Lettre de motivation"],
  ["questions", "Questions d’entretien"],
  ["details", "Détails"],
];

const STAGES = [
  ["Ingestion", "Découpe et indexe vos documents"],
  ["Analyste d’offre", "Extrait exigences et mots-clés"],
  ["Retriever", "Trouve une preuve par exigence"],
  ["Writer", "Rédige avec les seules preuves citées"],
  ["Critic", "Note le CV et demande des corrections"],
];

export default function App() {
  const { phase, events, result, error, start, replay } = useJob();
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [replays, setReplays] = useState<ReplayInfo[]>([]);
  const [pick, setPick] = useState("");
  const [tab, setTab] = useState<Tab>("cv");
  const busy = phase === "running";

  useEffect(() => {
    getConfig().then(setConfig);
    listReplays().then((r) => {
      setReplays(r);
      if (r.length > 0) setPick(r[0].run_id);
    });
  }, []);

  useEffect(() => {
    if (phase === "done") {
      setTab("cv");
      listReplays().then((r) => setReplays(r));
    }
  }, [phase]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden="true">
            CF
          </span>
          <div>
            <h1>Job Application Copilot</h1>
            <p>CareerForge · un CV adapté à l’offre, sans affirmation inventée</p>
          </div>
        </div>
        <div className="top-actions">
          {replays.length > 0 && (
            <div className="replay">
              <select value={pick} onChange={(e) => setPick(e.target.value)} aria-label="Exécution enregistrée">
                {replays.map((r) => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.job_title || (r.run_id === "demo" ? "Démo enregistrée" : "Exécution enregistrée")}
                  </option>
                ))}
              </select>
              <button className="ghost small" disabled={busy || !pick} onClick={() => replay(pick)}>
                Rejouer sans LLM
              </button>
            </div>
          )}
        </div>
      </header>

      <main className="layout">
        <aside>
          <ProfileForm
            busy={busy}
            templates={config?.templates ?? ["classic", "modern", "compact"]}
            maxFiles={config?.max_files ?? 10}
            onSubmit={start}
          />
        </aside>

        <section className="results" aria-live="polite">
          {phase === "idle" && (
            <div className="panel intro">
              <h2>Comment ça marche</h2>
              <p>
                Vos documents deviennent une base de preuves. Chaque ligne du CV doit citer une preuve, et un
                Critic vérifie le résultat avant de vous le rendre.
              </p>
              <ol className="stages">
                {STAGES.map(([name, text], i) => (
                  <li key={name}>
                    <span className="step-no">{i + 1}</span>
                    <div>
                      <b>{name}</b>
                      <p>{text}</p>
                    </div>
                  </li>
                ))}
              </ol>
            </div>
          )}

          {(phase === "running" || phase === "error") && <Progress events={events} failed={phase === "error"} />}

          {phase === "error" && (
            <div className="panel error" role="alert">
              <h3>La génération s’est arrêtée</h3>
              <p>{error}</p>
              <p className="hint">
                Si c’est une limite de quota gratuit, attendez un moment ou changez de modèle dans le fichier .env.
              </p>
            </div>
          )}

          {phase === "done" && result && (
            <>
              <Scores result={result} />
              <div className="tabs" role="tablist">
                {TABS.map(([id, label]) => (
                  <button key={id} role="tab" aria-selected={tab === id} onClick={() => setTab(id)}>
                    {label}
                  </button>
                ))}
              </div>
              {tab === "cv" && <CVEditor key={result.run_id} result={result} templates={config?.templates ?? ["classic"]} />}
              {tab === "critic" && <CriticLoop history={result.history} />}
              {tab === "letter" && <TextPanel title="Lettre de motivation" text={result.cover_letter} />}
              {tab === "questions" && <TextPanel title="Questions d’entretien probables" text={result.questions} />}
              {tab === "details" && <Details result={result} llm={config?.llm ?? ""} />}
            </>
          )}
        </section>
      </main>
    </div>
  );
}
