"""FastAPI study server — Step 7 implementation.

Tables: participant, trial, response, event (SQLite at repo root study.db).
Serves pre-rendered stimuli as full-file responses (Accept-Ranges disabled).
Generator (sap_stimulus) is NEVER imported here.
"""
from __future__ import annotations

import csv
import io
import itertools
import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]       # apps/study/server.py → apps/study → apps → repo root
_DB_PATH = _REPO_ROOT / "study.db"
_STATIC = _HERE.parent / "static"
_STATIC.mkdir(exist_ok=True)
_STIMULI_ROOT = _REPO_ROOT / "corpus" / "stimuli"
_UNBLIND_KEY = _REPO_ROOT / "analysis" / "unblind_key.json"

# ---------------------------------------------------------------------------
# Arm assignment
# ---------------------------------------------------------------------------
_ARM_CYCLE = itertools.cycle(["A", "B", "C"])

def _next_arm() -> str:
    return next(_ARM_CYCLE)

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS participant (
                id TEXT PRIMARY KEY,
                arm_code TEXT,
                created_at REAL,
                ua TEXT,
                age_band TEXT,
                native_lang TEXT,
                hearing_impaired INTEGER,
                hp_score INTEGER,
                hp_attempts INTEGER
            );
            CREATE TABLE IF NOT EXISTS trial (
                id TEXT PRIMARY KEY,
                participant_id TEXT,
                item_id TEXT,
                block TEXT,
                position INTEGER,
                audio_path TEXT,
                presented_at REAL
            );
            CREATE TABLE IF NOT EXISTS response (
                trial_id TEXT PRIMARY KEY,
                rt_ms REAL,
                rt_invalid INTEGER,
                has_replay INTEGER DEFAULT 0,
                answer TEXT,
                confidence INTEGER,
                allocation INTEGER,
                audio_offset_ts REAL
            );
            CREATE TABLE IF NOT EXISTS event (
                id TEXT PRIMARY KEY,
                participant_id TEXT,
                type TEXT,
                data TEXT,
                ts REAL
            );
        """)
        # Migration: add has_replay if DB predates this column (idempotent)
        try:
            conn.execute("ALTER TABLE response ADD COLUMN has_replay INTEGER DEFAULT 0")
        except Exception:
            pass
        # Migration: add hp_score / hp_attempts if DB predates these columns (idempotent)
        for col, default in [("hp_score", "NULL"), ("hp_attempts", "NULL")]:
            try:
                conn.execute(f"ALTER TABLE participant ADD COLUMN {col} INTEGER DEFAULT {default}")
            except Exception:
                pass


# Run at import time
_init_db()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Acoustic Persuasion Lab — Study")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ParticipantIn(BaseModel):
    ua: str = ""
    age_band: str = ""
    native_lang: str = ""
    hearing_impaired: bool = False


class TrialStartIn(BaseModel):
    participant_id: str
    item_id: str
    block: str
    position: int


class TrialResponseIn(BaseModel):
    trial_id: str
    rt_ms: Optional[float] = None
    rt_invalid: Optional[int] = None
    has_replay: Optional[int] = 0
    answer: Optional[str] = None
    confidence: Optional[int] = None
    allocation: Optional[int] = None
    audio_offset_ts: Optional[float] = None


class HeadphoneIn(BaseModel):
    participant_id: str
    hp_score: int
    hp_attempts: int


class EventIn(BaseModel):
    participant_id: str
    type: str
    data: Any = None

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/participant")
async def create_participant(body: ParticipantIn) -> dict:
    import time
    pid = str(uuid.uuid4())
    arm = _next_arm()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO participant (id, arm_code, created_at, ua, age_band, native_lang, hearing_impaired) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (pid, arm, time.time(), body.ua, body.age_band, body.native_lang, int(body.hearing_impaired)),
        )
    return {"participant_id": pid, "arm_code": arm}

@app.post("/headphone")
async def record_headphone(body: HeadphoneIn) -> dict:
    """Persist final headphone screening result (called once: on pass or terminal fail)."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE participant SET hp_score=?, hp_attempts=? WHERE id=?",
            (body.hp_score, body.hp_attempts, body.participant_id),
        )
    return {"ok": True}


@app.get("/stimuli/{arm}/{item_id}")
async def get_stimulus(arm: str, item_id: str):
    wav = _STIMULI_ROOT / arm / f"{item_id}.wav"
    if not wav.exists():
        raise HTTPException(status_code=404, detail=f"Stimulus not found: {arm}/{item_id}")
    return FileResponse(
        str(wav),
        media_type="audio/wav",
        headers={"Accept-Ranges": "none", "Cache-Control": "no-store"},
    )


@app.post("/trial/start")
async def trial_start(body: TrialStartIn) -> dict:
    import time
    tid = str(uuid.uuid4())
    arm_row = None
    with _get_conn() as conn:
        row = conn.execute("SELECT arm_code FROM participant WHERE id=?", (body.participant_id,)).fetchone()
        if row:
            arm_row = row["arm_code"]
        audio_path = f"corpus/stimuli/{arm_row}/{body.item_id}.wav" if arm_row else ""
        conn.execute(
            "INSERT INTO trial (id, participant_id, item_id, block, position, audio_path, presented_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (tid, body.participant_id, body.item_id, body.block, body.position, audio_path, time.time()),
        )
    return {"trial_id": tid}


@app.post("/trial/response")
async def trial_response(body: TrialResponseIn) -> dict:
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO response
               (trial_id, rt_ms, rt_invalid, has_replay, answer, confidence, allocation, audio_offset_ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(trial_id) DO UPDATE SET
                 rt_ms=excluded.rt_ms,
                 rt_invalid=excluded.rt_invalid,
                 has_replay=excluded.has_replay,
                 answer=excluded.answer,
                 confidence=excluded.confidence,
                 allocation=excluded.allocation,
                 audio_offset_ts=excluded.audio_offset_ts""",
            (body.trial_id, body.rt_ms, body.rt_invalid, body.has_replay or 0,
             body.answer, body.confidence, body.allocation, body.audio_offset_ts),
        )
    return {"ok": True}


@app.post("/event")
async def log_event(body: EventIn) -> dict:
    import time
    eid = str(uuid.uuid4())
    data_str = json.dumps(body.data) if body.data is not None else ""
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO event (id, participant_id, type, data, ts) VALUES (?, ?, ?, ?, ?)",
            (eid, body.participant_id, body.type, data_str, time.time()),
        )
    return {"ok": True}


@app.get("/admin/export.csv")
async def export_csv(x_admin_token: Optional[str] = Header(default=None)):
    expected = os.environ.get("ADMIN_TOKEN", "dev-token")
    if x_admin_token != expected:
        raise HTTPException(status_code=403, detail="Forbidden")

    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT
                p.id AS participant_id,
                p.arm_code,
                p.age_band,
                p.native_lang,
                p.hearing_impaired,
                p.hp_score,
                p.hp_attempts,
                t.id AS trial_id,
                t.item_id,
                t.block,
                t.position,
                t.audio_path,
                t.presented_at,
                r.rt_ms,
                r.rt_invalid,
                COALESCE(r.has_replay, 0) AS has_replay,
                r.answer,
                r.confidence,
                r.allocation,
                r.audio_offset_ts
            FROM participant p
            LEFT JOIN trial t ON t.participant_id = p.id
            LEFT JOIN response r ON r.trial_id = t.id
            ORDER BY p.created_at, t.position
        """).fetchall()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "participant_id", "arm_code", "age_band", "native_lang", "hearing_impaired",
        "hp_score", "hp_attempts",
        "trial_id", "item_id", "block", "position", "audio_path", "presented_at",
        "rt_ms", "rt_invalid", "has_replay", "answer", "confidence", "allocation", "audio_offset_ts",
    ])
    for row in rows:
        writer.writerow(list(row))

    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=export.csv"},
    )


@app.get("/admin/participants")
async def admin_participants(x_admin_token: Optional[str] = Header(default=None)):
    expected = os.environ.get("ADMIN_TOKEN", "dev-token")
    if x_admin_token != expected:
        raise HTTPException(status_code=403, detail="Forbidden")
    with _get_conn() as conn:
        rows = conn.execute("SELECT * FROM participant ORDER BY created_at DESC LIMIT 100").fetchall()
    return [dict(r) for r in rows]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    index_html = _STATIC / "index.html"
    if index_html.exists():
        return FileResponse(str(index_html))
    return Response(content="<h1>Study app — static/index.html not yet created</h1>",
                    media_type="text/html")


# Mount static AFTER route definitions so / is not shadowed
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
