"""FastAPI study server — stub (implemented in Step 7).

Tables: participant, assignment, trial, response, event (SQLite).
Serves pre-rendered stimuli as full-file static responses (no Accept-Ranges).
Generator (sap_stimulus) is NEVER imported here.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Acoustic Persuasion Lab — Study")

_STATIC = Path(__file__).parent / "static"
_STATIC.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/")
async def index():
    from fastapi.responses import FileResponse
    index_html = _STATIC / "index.html"
    if index_html.exists():
        return FileResponse(str(index_html))
    return {"status": "study app stub — static/index.html not yet created"}


@app.get("/health")
async def health():
    return {"status": "ok"}


# Admin export — token gated (Step 7)
@app.get("/admin/export.csv")
async def export_csv(request_token: str = ""):
    admin_token = os.environ.get("ADMIN_TOKEN", "")
    from fastapi import HTTPException
    from fastapi.responses import Response
    if not admin_token or request_token != admin_token:
        raise HTTPException(status_code=403, detail="Forbidden")
    # Stub: return empty CSV header
    csv = "participant_id,arm_code,trial_id,audio_offset_ts,rt_ms,response,rt_invalid\n"
    return Response(content=csv, media_type="text/csv")
