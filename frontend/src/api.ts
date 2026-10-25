import type { AppConfig, CV, Candidate, RenderResponse, ReplayInfo } from "./types";

async function errorMessage(r: Response): Promise<string> {
  try {
    const body = await r.json();
    const d = body.detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) return d.map((x) => x.msg ?? JSON.stringify(x)).join(" ; ");
  } catch {
    /* not JSON */
  }
  return `Erreur ${r.status}`;
}

export async function createJob(form: FormData): Promise<string> {
  const r = await fetch("/api/jobs", { method: "POST", body: form });
  if (!r.ok) throw new Error(await errorMessage(r));
  return (await r.json()).job_id as string;
}

export async function startReplay(runId: string): Promise<string> {
  const r = await fetch(`/api/replay/${runId}`, { method: "POST" });
  if (!r.ok) throw new Error(await errorMessage(r));
  return (await r.json()).job_id as string;
}

export async function getConfig(): Promise<AppConfig | null> {
  try {
    const r = await fetch("/api/config");
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}

export async function listReplays(): Promise<ReplayInfo[]> {
  try {
    const r = await fetch("/api/replays");
    return r.ok ? await r.json() : [];
  } catch {
    return [];
  }
}

export async function renderEdited(
  runId: string,
  cv: CV,
  candidate: Candidate,
  template: string,
): Promise<RenderResponse> {
  const r = await fetch(`/api/runs/${runId}/render`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cv, candidate, template }),
  });
  if (!r.ok) throw new Error(await errorMessage(r));
  return await r.json();
}

export const downloadUrl = (runId: string, file: string) => `/download/${runId}/${file}`;
