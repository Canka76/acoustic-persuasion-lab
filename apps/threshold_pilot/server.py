"""FastAPI threshold pilot server — stub (implemented in Step 4).

3-down-1-up adaptive staircase, 2-interval forced choice.
Two interleaved tracks: sub-harmonic layer, AM layer.
n=12 listeners; threshold = mean of last 6 of 8 reversals.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Acoustic Persuasion Lab — Threshold Pilot")

_STATIC = Path(__file__).parent / "static"
_STATIC.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/")
async def index():
    from fastapi.responses import FileResponse
    index_html = _STATIC / "index.html"
    if index_html.exists():
        return FileResponse(str(index_html))
    return {"status": "threshold pilot stub — static/index.html not yet created"}


@app.get("/health")
async def health():
    return {"status": "ok"}
