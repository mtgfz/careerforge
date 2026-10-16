"""FastAPI backend. Run: uvicorn src.api.main:app --reload"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .. import config
from ..schemas import Candidate
from ..service import generate_application

app = FastAPI(title="CareerForge", version="0.1.0")

ALLOWED_DOWNLOADS = {"cv.pdf", "cv.tex", "cv.md", "cover_letter.txt", "interview_questions.txt", "run.json"}


@app.get("/health")
def health():
    return {"status": "ok", "llm": f"{config.LLM_PROVIDER}:{config.LLM_MODEL}"}


@app.post("/generate")
async def generate(
    job_text: str = Form(...),
    name: str = Form(...),
    email: str = Form(""),
    phone: str = Form(""),
    location: str = Form(""),
    links: str = Form("", description="comma-separated URLs"),
    github_user: str = Form(""),
    extra_text: str = Form(""),
    use_critic: bool = Form(True),
    files: list[UploadFile] = File(default=[]),
):
    candidate = Candidate(
        name=name,
        email=email,
        phone=phone,
        location=location,
        links=[x.strip() for x in links.split(",") if x.strip()],
    )
    tmp = Path(tempfile.mkdtemp(prefix="cf_upload_"))
    try:
        saved: list[Path] = []
        for f in files:
            dest = tmp / Path(f.filename or "upload").name  # strip any path components
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            saved.append(dest)
        try:
            return generate_application(
                candidate, job_text, saved, github_user, extra_text, use_critic
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.get("/download/{run_id}/{filename}")
def download(run_id: str, filename: str):
    if filename not in ALLOWED_DOWNLOADS or not run_id.isalnum():
        raise HTTPException(status_code=404)
    path = Path(config.OUTPUT_DIR) / run_id / filename
    if not path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(path)
