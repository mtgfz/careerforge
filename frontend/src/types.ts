export interface Scores {
  faithfulness: number;
  ats: number;
  quality: number;
  overall: number;
  passed: boolean;
}

export interface HistoryItem extends Scores {
  iteration: number;
  unsupported?: string[];
  missing_keywords?: string[];
}

export interface Bullet {
  text: string;
  source_ids: string[];
}

export interface Entry {
  title: string;
  organization: string;
  period: string;
  bullets: Bullet[];
}

export interface CV {
  headline: string;
  summary: string;
  skills: string[];
  experiences: Entry[];
  projects: Entry[];
  education: string[];
  languages: string[];
}

export interface Candidate {
  name: string;
  email: string;
  phone: string;
  location: string;
  links: string[];
}

export interface RunResult {
  run_id: string;
  job_title: string;
  template: string;
  best_iteration: number;
  timings: Record<string, number>;
  kb_chunks: number;
  kb_cached: boolean;
  cv: CV;
  candidate: Candidate;
  iterations: number;
  scores: Scores & { missing_keywords?: string[] };
  pruned: number;
  history: HistoryItem[];
  files: string[];
  pdf: boolean;
  markdown: string;
  cover_letter: string;
  questions: string;
  replay?: boolean;
}

export type NodeId =
  | "ingest"
  | "analyze"
  | "retrieve"
  | "write"
  | "critique"
  | "rejudge"
  | "extras"
  | "render";

export interface ProgressEvent {
  type: "queued" | "start" | "done" | "result" | "error";
  node?: NodeId;
  iteration?: number;
  seconds?: number;
  cached?: boolean;
  scores?: Scores;
  result?: RunResult;
  message?: string;
  t?: number;
}

export interface ReplayInfo {
  run_id: string;
  job_title: string;
  iterations: number;
}

export interface AppConfig {
  templates: string[];
  llm: string;
  max_files: number;
  max_iterations: number;
}

export interface RenderResponse {
  run_id: string;
  template: string;
  pdf: boolean;
  files: string[];
  markdown: string;
}
