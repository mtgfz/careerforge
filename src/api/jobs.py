"""Job API for the React app: asynchronous runs with live progress (SSE), replay, re-export.

POST /api/jobs                 start a run (multipart form), returns {"job_id"}
GET  /api/jobs/{id}/events     Server-Sent Events: queued | start | done | result | error
POST /api/runs/{run}/render    re-export a CV the user edited by hand
GET  /api/replays, POST /api/replay/{run}   replay a recorded run (plan B for the demo)
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import config
from ..render.latex import TEMPLATES
from ..schemas import Candidate
from ..service import DEMO_DIR, generate_application, rerender, run_dir

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

MAX_FILES = 10
MAX_JOBS = 50
_RUNS = threading.Semaphore(2)  # at most 2 pipelines at once: free LLM quotas are small


class Job:
    """A run in progress: an append-only event list that any number of clients can follow."""

    def __init__(self) -> None:
        self.id = uuid.uuid4().hex[:10]
        self.events: list[dict] = []
        self.done = False
        self.cond = threading.Condition()
        self.t0 = time.time()

    def emit(self, event: dict) -> None:
        with self.cond:
            self.events.append({**event, "t": round(time.time() - self.t0, 2)})
            self.cond.notify_all()

    def finish(self) -> None:
        with self.cond:
            self.done = True
            self.cond.notify_all()


JOBS: "OrderedDict[str, Job]" = OrderedDict()
_JOBS_LOCK = threading.Lock()


def _new_job() -> Job:
    job = Job()
    with _JOBS_LOCK:
        JOBS[job.id] = job
        while len(JOBS) > MAX_JOBS:
            JOBS.popitem(last=False)
    return job


def _valid_run_id(run_id: str) -> bool:
    return run_id == "demo" or run_id.isalnum()


def _save_replay(job: Job, result: dict) -> None:
    """Keep the events of a finished run so it can be replayed later without any LLM call."""
    try:
        data = {"events": [e for e in job.events if e["type"] != "result"], "result": result}
        (run_dir(result["run_id"]) / "replay.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:  # a failed recording must never break a finished run
        log.exception("could not save the replay")


def _run_job(job, tmp, saved, candidate, job_text, github_user, extra_text, use_critic, template):
    try:
        if not _RUNS.acquire(blocking=False):
            job.emit({"type": "queued"})
            _RUNS.acquire()
        try:
            result = generate_application(
                candidate, job_text, saved, github_user, extra_text, use_critic,
                template=template, on_event=job.emit,
            )
        finally:
            _RUNS.release()
        result.pop("out_dir", None)  # server path: not for the browser
        _save_replay(job, result)
        job.emit({"type": "result", "result": result})
    except Exception as e:
        log.exception("job %s failed", job.id)
        job.emit({"type": "error", "message": str(e)[:600] or e.__class__.__name__})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        job.finish()


@router.get("/config")
def get_config():
    return {
        "templates": list(TEMPLATES),
        "llm": f"{config.LLM_PROVIDER}:{config.LLM_MODEL}",
        "max_files": MAX_FILES,
        "max_iterations": config.MAX_ITERATIONS,
    }


@router.post("/jobs")
def create_job(
    job_text: str = Form(...),
    name: str = Form(...),
    email: str = Form(""),
    phone: str = Form(""),
    location: str = Form(""),
    links: str = Form("", description="comma-separated URLs"),
    github_user: str = Form(""),
    extra_text: str = Form(""),
    use_critic: bool = Form(True),
    template: str = Form("classic"),
    files: list[UploadFile] = File(default=[]),
):
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"At most {MAX_FILES} files.")
    if not job_text.strip():
        raise HTTPException(400, "The job offer is empty.")
    if not (files or github_user.strip() or extra_text.strip()):
        raise HTTPException(400, "Provide a CV/document, a GitHub user or some text about you.")
    candidate = Candidate(
        name=name, email=email, phone=phone, location=location,
        links=[x.strip() for x in links.split(",") if x.strip()],
    )
    tmp = Path(tempfile.mkdtemp(prefix="cf_upload_"))
    saved: list[Path] = []
    for f in files:
        dest = tmp / Path(f.filename or "upload").name  # strip any path components
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(dest)
    job = _new_job()
    threading.Thread(
        target=_run_job,
        args=(job, tmp, saved, candidate, job_text, github_user, extra_text, use_critic,
              template if template in TEMPLATES else "classic"),
        daemon=True,
    ).start()
    return {"job_id": job.id}


@router.get("/jobs/{job_id}/events")
def job_events(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")

    def stream():
        i = 0
        while True:
            with job.cond:
                if i >= len(job.events) and not job.done:
                    job.cond.wait(timeout=15)
                batch, done = job.events[i:], job.done
            if batch:
                for ev in batch:
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                i += len(batch)
            elif done:
                return
            else:
                yield ": keepalive\n\n"

    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class RenderRequest(BaseModel):
    cv: dict
    candidate: dict
    template: str = "classic"


@router.post("/runs/{run_id}/render")
def render_edited(run_id: str, body: RenderRequest):
    if not _valid_run_id(run_id):
        raise HTTPException(404)
    try:
        return rerender(run_id, body.cv, body.candidate, body.template)
    except FileNotFoundError:
        raise HTTPException(404, "Unknown run")
    except ValueError as e:  # pydantic validation errors
        raise HTTPException(422, str(e)[:400])


def _replay_path(run_id: str) -> Path:
    return run_dir(run_id) / "replay.json"


@router.get("/replays")
def list_replays():
    found = []
    candidates = [DEMO_DIR / "run" / "replay.json"]
    root = Path(config.OUTPUT_DIR)
    if root.is_dir():
        candidates += sorted(root.glob("*/replay.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:8]
    for p in candidates:
        if not p.exists():
            continue
        try:
            res = json.loads(p.read_text(encoding="utf-8"))["result"]
        except Exception:
            continue
        run_id = "demo" if p.parent == DEMO_DIR / "run" else p.parent.name
        found.append({"run_id": run_id, "job_title": res.get("job_title", ""),
                      "iterations": res.get("iterations", 0)})
    return found


def _play(job: Job, data: dict) -> None:
    prev = 0.0
    for ev in data["events"]:
        t = ev.get("t", 0.0)
        time.sleep(min(max(0.0, t - prev), 4.0) * 0.25)  # same rhythm, 4x faster
        prev = t
        job.emit({k: v for k, v in ev.items() if k != "t"})
    job.emit({"type": "result", "result": {**data["result"], "replay": True}})
    job.finish()


@router.post("/replay/{run_id}")
def replay(run_id: str):
    if not _valid_run_id(run_id) or not _replay_path(run_id).exists():
        raise HTTPException(404, "No recording for this run")
    data = json.loads(_replay_path(run_id).read_text(encoding="utf-8"))
    job = _new_job()
    threading.Thread(target=_play, args=(job, data), daemon=True).start()
    return {"job_id": job.id}
