import { useState } from "react";

export default function TextPanel({ title, text }: { title: string; text: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      /* clipboard blocked: the text stays selectable */
    }
  };
  return (
    <div className="panel">
      <div className="row between">
        <h3>{title}</h3>
        <button className="ghost small" onClick={copy}>
          {copied ? "Copié" : "Copier"}
        </button>
      </div>
      <pre className="plain">{text || "(vide)"}</pre>
    </div>
  );
}
