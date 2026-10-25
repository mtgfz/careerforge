import { useEffect, useState, type DragEvent } from "react";

const STORAGE_KEY = "careerforge.profile";
const TEMPLATE_LABELS: Record<string, string> = {
  classic: "Classique",
  modern: "Moderne",
  compact: "Compact",
};

interface Saved {
  name: string;
  email: string;
  phone: string;
  location: string;
  links: string;
  github: string;
}
const EMPTY: Saved = { name: "", email: "", phone: "", location: "", links: "", github: "" };

function loadSaved(): Saved {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return { ...EMPTY, ...JSON.parse(raw) };
  } catch {
    /* storage unavailable: start empty */
  }
  return EMPTY;
}

interface Props {
  busy: boolean;
  templates: string[];
  maxFiles: number;
  onSubmit: (form: FormData) => void;
}

export default function ProfileForm({ busy, templates, maxFiles, onSubmit }: Props) {
  const [saved, setSaved] = useState<Saved>(loadSaved);
  const [files, setFiles] = useState<File[]>([]);
  const [extra, setExtra] = useState("");
  const [job, setJob] = useState("");
  const [useCritic, setUseCritic] = useState(true);
  const [template, setTemplate] = useState("classic");
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(saved));
    } catch {
      /* ignore */
    }
  }, [saved]);

  const set = (k: keyof Saved) => (v: string) => setSaved((s) => ({ ...s, [k]: v }));
  const hasMaterial = files.length > 0 || saved.github.trim() !== "" || extra.trim() !== "";
  const valid = saved.name.trim() !== "" && job.trim().length >= 20 && hasMaterial;

  const addFiles = (list: FileList | null) => {
    if (!list) return;
    setFiles((prev) => {
      const names = new Set(prev.map((f) => f.name + f.size));
      const next = [...prev, ...Array.from(list).filter((f) => !names.has(f.name + f.size))];
      return next.slice(0, maxFiles);
    });
  };
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    addFiles(e.dataTransfer.files);
  };

  const submit = () => {
    const fd = new FormData();
    fd.append("job_text", job);
    fd.append("name", saved.name.trim());
    fd.append("email", saved.email.trim());
    fd.append("phone", saved.phone.trim());
    fd.append("location", saved.location.trim());
    fd.append("links", saved.links);
    fd.append("github_user", saved.github.trim());
    fd.append("extra_text", extra);
    fd.append("use_critic", String(useCritic));
    fd.append("template", template);
    files.forEach((f) => fd.append("files", f));
    onSubmit(fd);
  };

  return (
    <form
      className="panel form"
      onSubmit={(e) => {
        e.preventDefault();
        if (valid && !busy) submit();
      }}
    >
      <section>
        <h2>
          <span className="step-no">1</span> Votre profil
        </h2>
        <div className="grid2">
          <label>
            Nom complet *
            <input value={saved.name} onChange={(e) => set("name")(e.target.value)} autoComplete="name" />
          </label>
          <label>
            E-mail
            <input value={saved.email} onChange={(e) => set("email")(e.target.value)} autoComplete="email" />
          </label>
          <label>
            Téléphone
            <input value={saved.phone} onChange={(e) => set("phone")(e.target.value)} autoComplete="tel" />
          </label>
          <label>
            Ville
            <input value={saved.location} onChange={(e) => set("location")(e.target.value)} />
          </label>
        </div>
        <label>
          Liens (séparés par des virgules)
          <input
            value={saved.links}
            onChange={(e) => set("links")(e.target.value)}
            placeholder="https://github.com/vous, https://linkedin.com/in/vous"
          />
        </label>
      </section>

      <section>
        <h2>
          <span className="step-no">2</span> Vos documents
        </h2>
        <div
          className={`drop ${dragging ? "drag" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <p>
            Déposez vos CV, rapports ou notes ici, ou{" "}
            <label className="link-btn">
              parcourir
              <input
                type="file"
                multiple
                hidden
                accept=".pdf,.docx,.md,.txt,.tex"
                onChange={(e) => {
                  addFiles(e.target.files);
                  e.target.value = "";
                }}
              />
            </label>
          </p>
          <p className="hint">PDF, DOCX, MD, TXT, TEX · {maxFiles} fichiers maximum</p>
        </div>
        {files.length > 0 && (
          <ul className="files">
            {files.map((f) => (
              <li key={f.name + f.size}>
                <span>{f.name}</span>
                <button
                  type="button"
                  className="icon-btn"
                  aria-label={`Retirer ${f.name}`}
                  onClick={() => setFiles((p) => p.filter((x) => x !== f))}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        )}
        <label>
          Compte GitHub (optionnel)
          <input
            value={saved.github}
            onChange={(e) => set("github")(e.target.value)}
            placeholder="nom d’utilisateur"
            autoComplete="off"
          />
        </label>
        <label>
          Ou collez du texte sur votre parcours
          <textarea rows={3} value={extra} onChange={(e) => setExtra(e.target.value)} />
        </label>
      </section>

      <section>
        <h2>
          <span className="step-no">3</span> L’offre d’emploi
        </h2>
        <label>
          Collez l’offre *
          <textarea rows={8} value={job} onChange={(e) => setJob(e.target.value)} />
        </label>
      </section>

      <section className="options">
        <label className="check">
          <input type="checkbox" checked={useCritic} onChange={(e) => setUseCritic(e.target.checked)} />
          Boucle du Critic (vérifie chaque affirmation)
        </label>
        <label>
          Modèle de CV
          <select value={template} onChange={(e) => setTemplate(e.target.value)}>
            {templates.map((t) => (
              <option key={t} value={t}>
                {TEMPLATE_LABELS[t] ?? t}
              </option>
            ))}
          </select>
        </label>
      </section>

      <button className="primary" type="submit" disabled={!valid || busy}>
        {busy ? "Génération en cours…" : "Générer mon CV"}
      </button>
      {!valid && !busy && (
        <p className="hint">
          Il faut un nom, au moins un document (fichier, GitHub ou texte) et une offre.
        </p>
      )}
    </form>
  );
}
