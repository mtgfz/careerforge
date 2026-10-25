import { useState } from "react";
import { downloadUrl, renderEdited } from "../api";
import type { CV, Candidate, Entry, RenderResponse, RunResult } from "../types";

const TEMPLATE_LABELS: Record<string, string> = {
  classic: "Classique",
  modern: "Moderne",
  compact: "Compact",
};

type Section = "experiences" | "projects";

export function CVPaper({ cv, candidate }: { cv: CV; candidate: Candidate }) {
  const contact = [candidate.email, candidate.phone, candidate.location, ...candidate.links].filter(Boolean);
  const entries = (title: string, list: Entry[]) =>
    list.length > 0 && (
      <section>
        <h4>{title}</h4>
        {list.map((e, i) => (
          <div className="paper-entry" key={i}>
            <p className="paper-entry-head">
              <b>{e.title}</b>
              {e.organization && <i> · {e.organization}</i>}
              {e.period && <span className="paper-period">{e.period}</span>}
            </p>
            <ul>
              {e.bullets.map((b, j) => (
                <li key={j}>{b.text}</li>
              ))}
            </ul>
          </div>
        ))}
      </section>
    );
  return (
    <article className="paper">
      <header>
        <h3>{candidate.name}</h3>
        <p className="paper-headline">{cv.headline}</p>
        <p className="paper-contact">{contact.join("  ·  ")}</p>
      </header>
      {cv.summary && (
        <section>
          <h4>Profil</h4>
          <p>{cv.summary}</p>
        </section>
      )}
      {cv.skills.length > 0 && (
        <section>
          <h4>Compétences</h4>
          <p className="chips">
            {cv.skills.map((s) => (
              <span className="chip" key={s}>
                {s}
              </span>
            ))}
          </p>
        </section>
      )}
      {entries("Expérience", cv.experiences)}
      {entries("Projets", cv.projects)}
      {cv.education.length > 0 && (
        <section>
          <h4>Formation</h4>
          <ul>
            {cv.education.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </section>
      )}
      {cv.languages.length > 0 && (
        <section>
          <h4>Langues</h4>
          <p>{cv.languages.join(", ")}</p>
        </section>
      )}
    </article>
  );
}

function EntryEditor({
  entry,
  onChange,
  onRemove,
}: {
  entry: Entry;
  onChange: (e: Entry) => void;
  onRemove: () => void;
}) {
  const setBullet = (i: number, text: string) =>
    onChange({
      ...entry,
      bullets: entry.bullets.map((b, j) => (j === i ? { ...b, text } : b)),
    });
  return (
    <div className="entry-edit">
      <div className="grid3">
        <label>
          Titre
          <input value={entry.title} onChange={(e) => onChange({ ...entry, title: e.target.value })} />
        </label>
        <label>
          Organisation
          <input value={entry.organization} onChange={(e) => onChange({ ...entry, organization: e.target.value })} />
        </label>
        <label>
          Période
          <input value={entry.period} onChange={(e) => onChange({ ...entry, period: e.target.value })} />
        </label>
      </div>
      {entry.bullets.map((b, i) => (
        <div className="bullet-edit" key={i}>
          <textarea
            rows={2}
            aria-label={`Puce ${i + 1}`}
            value={b.text}
            onChange={(e) => setBullet(i, e.target.value)}
          />
          <button
            type="button"
            className="icon-btn"
            aria-label={`Supprimer la puce ${i + 1}`}
            onClick={() => onChange({ ...entry, bullets: entry.bullets.filter((_, j) => j !== i) })}
          >
            ×
          </button>
        </div>
      ))}
      <div className="row">
        <button
          type="button"
          className="ghost"
          onClick={() => onChange({ ...entry, bullets: [...entry.bullets, { text: "", source_ids: [] }] })}
        >
          + Ajouter une puce
        </button>
        <button type="button" className="ghost danger" onClick={onRemove}>
          Supprimer cette entrée
        </button>
      </div>
    </div>
  );
}

export default function CVEditor({ result, templates }: { result: RunResult; templates: string[] }) {
  const [cv, setCv] = useState<CV>(result.cv);
  const [mode, setMode] = useState<"preview" | "edit">("preview");
  const [template, setTemplate] = useState(result.template);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [exported, setExported] = useState<RenderResponse | null>(null);
  const dirty = JSON.stringify(cv) !== JSON.stringify(result.cv);

  const setEntry = (section: Section, i: number, entry: Entry) =>
    setCv((c) => ({ ...c, [section]: c[section].map((e, j) => (j === i ? entry : e)) }));
  const removeEntry = (section: Section, i: number) =>
    setCv((c) => ({ ...c, [section]: c[section].filter((_, j) => j !== i) }));

  const exportEdited = async () => {
    setBusy(true);
    setError("");
    try {
      setExported(await renderEdited(result.run_id, cv, result.candidate, template));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const useEdited = exported !== null;
  const base = useEdited ? "cv_edited" : "cv";
  const pdfAvailable = useEdited ? exported.pdf : result.pdf;

  return (
    <div className="stack">
      <div className="panel toolbar">
        <div className="seg" role="tablist" aria-label="Mode d’affichage">
          <button role="tab" aria-selected={mode === "preview"} onClick={() => setMode("preview")}>
            Aperçu
          </button>
          <button role="tab" aria-selected={mode === "edit"} onClick={() => setMode("edit")}>
            Modifier
          </button>
        </div>
        <label className="inline">
          Modèle
          <select value={template} onChange={(e) => setTemplate(e.target.value)}>
            {templates.map((t) => (
              <option key={t} value={t}>
                {TEMPLATE_LABELS[t] ?? t}
              </option>
            ))}
          </select>
        </label>
        <button className="primary small" onClick={exportEdited} disabled={busy}>
          {busy ? "Export…" : dirty || template !== result.template ? "Réexporter avec mes modifications" : "Réexporter"}
        </button>
      </div>
      {error && <p className="error">{error}</p>}

      {mode === "preview" ? (
        <CVPaper cv={cv} candidate={result.candidate} />
      ) : (
        <div className="panel edit">
          <label>
            Titre du CV
            <input value={cv.headline} onChange={(e) => setCv({ ...cv, headline: e.target.value })} />
          </label>
          <label>
            Profil
            <textarea rows={4} value={cv.summary} onChange={(e) => setCv({ ...cv, summary: e.target.value })} />
          </label>
          <label>
            Compétences (séparées par des virgules)
            <input
              value={cv.skills.join(", ")}
              onChange={(e) =>
                setCv({ ...cv, skills: e.target.value.split(",").map((s) => s.trim()).filter(Boolean) })
              }
            />
          </label>
          {(["experiences", "projects"] as Section[]).map((section) => (
            <div key={section}>
              <h4>{section === "experiences" ? "Expérience" : "Projets"}</h4>
              {cv[section].map((entry, i) => (
                <EntryEditor
                  key={i}
                  entry={entry}
                  onChange={(e) => setEntry(section, i, e)}
                  onRemove={() => removeEntry(section, i)}
                />
              ))}
              {cv[section].length === 0 && <p className="hint">Aucune entrée.</p>}
            </div>
          ))}
          <label>
            Formation (une ligne par entrée)
            <textarea
              rows={3}
              value={cv.education.join("\n")}
              onChange={(e) => setCv({ ...cv, education: e.target.value.split("\n").filter((l) => l.trim()) })}
            />
          </label>
          <label>
            Langues (séparées par des virgules)
            <input
              value={cv.languages.join(", ")}
              onChange={(e) =>
                setCv({ ...cv, languages: e.target.value.split(",").map((s) => s.trim()).filter(Boolean) })
              }
            />
          </label>
          <p className="hint">
            Les puces ajoutées à la main n’ont pas de source dans vos documents : le Critic ne les a pas vérifiées.
          </p>
        </div>
      )}

      <div className="panel downloads">
        <h3>Télécharger {useEdited ? "(version modifiée)" : ""}</h3>
        <div className="row wrap">
          {pdfAvailable ? (
            <a className="btn" href={downloadUrl(result.run_id, `${base}.pdf`)} download>
              CV en PDF
            </a>
          ) : (
            <span className="hint">PDF indisponible : installez pdflatex ou utilisez Docker. Le .tex reste disponible.</span>
          )}
          <a className="btn ghost" href={downloadUrl(result.run_id, `${base}.tex`)} download>
            LaTeX
          </a>
          <a className="btn ghost" href={downloadUrl(result.run_id, `${base}.md`)} download>
            Markdown
          </a>
          <a className="btn ghost" href={downloadUrl(result.run_id, "cover_letter.txt")} download>
            Lettre
          </a>
          <a className="btn ghost" href={downloadUrl(result.run_id, "interview_questions.txt")} download>
            Questions
          </a>
          <a className="btn ghost" href={downloadUrl(result.run_id, "run.json")} download>
            Rapport du run
          </a>
        </div>
      </div>
    </div>
  );
}
