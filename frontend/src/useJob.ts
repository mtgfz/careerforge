import { useCallback, useEffect, useRef, useState } from "react";
import { createJob, startReplay } from "./api";
import type { ProgressEvent, RunResult } from "./types";

export type Phase = "idle" | "running" | "done" | "error";

/** Starts a run (or a replay) and follows its Server-Sent Events until the result arrives. */
export function useJob() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const [result, setResult] = useState<RunResult | null>(null);
  const [error, setError] = useState("");
  const source = useRef<EventSource | null>(null);

  const follow = useCallback((jobId: string) => {
    source.current?.close();
    const es = new EventSource(`/api/jobs/${jobId}/events`);
    source.current = es;
    es.onmessage = (m) => {
      const ev = JSON.parse(m.data) as ProgressEvent;
      setEvents((prev) => [...prev, ev]);
      if (ev.type === "result" && ev.result) {
        setResult(ev.result);
        setPhase("done");
        es.close();
      } else if (ev.type === "error") {
        setError(ev.message || "Erreur inconnue");
        setPhase("error");
        es.close();
      }
    };
    es.onerror = () => {
      es.close();
      setPhase((p) => (p === "running" ? "error" : p));
      setError((e) => e || "Connexion au serveur perdue.");
    };
  }, []);

  const launch = useCallback(
    async (starter: () => Promise<string>) => {
      setPhase("running");
      setEvents([]);
      setResult(null);
      setError("");
      try {
        follow(await starter());
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setPhase("error");
      }
    },
    [follow],
  );

  const start = useCallback((form: FormData) => launch(() => createJob(form)), [launch]);
  const replay = useCallback((runId: string) => launch(() => startReplay(runId)), [launch]);

  useEffect(() => () => source.current?.close(), []);

  return { phase, events, result, error, start, replay };
}
