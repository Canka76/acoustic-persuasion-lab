"""FastAPI threshold pilot server.

3-down-1-up adaptive staircase, 2-interval forced choice.
Two interleaved tracks: sub-harmonic layer (track 0), AM layer (track 1).
n=12 listeners run sequentially; threshold = mean of last 6 of 8 reversals.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent
_STATIC = _HERE / "static"
_STATIC.mkdir(exist_ok=True)

_STIMULI_TREATED = _REPO / "corpus" / "stimuli" / "B" / "arg_03.wav"
_STIMULI_REFERENCE = _REPO / "corpus" / "stimuli" / "A" / "arg_03.wav"
_RESULTS_JSON = _REPO / "analysis" / "threshold_pilot.json"

# ---------------------------------------------------------------------------
# Staircase constants
# ---------------------------------------------------------------------------
_START_LEVEL_DB = -20.0
_MIN_LEVEL_DB = -50.0
_MAX_LEVEL_DB = -20.0
_STEP_LARGE_DB = 3.0
_STEP_SMALL_DB = 1.0
_REVERSALS_FOR_SMALL_STEP = 2
_TOTAL_REVERSALS = 8
_THRESHOLD_REVERSALS = 6  # mean of last 6

TRACK_NAMES = {0: "subharmonic", 1: "am"}


# ---------------------------------------------------------------------------
# Staircase state
# ---------------------------------------------------------------------------
@dataclass
class StaircaseTrack:
    name: str
    level_db: float = _START_LEVEL_DB
    # 3-down-1-up: need 3 consecutive correct to go down
    correct_streak: int = 0
    # direction of last level change: +1 = went up (harder), -1 = went down (easier), 0 = start
    last_direction: int = 0
    reversals: list[float] = field(default_factory=list)
    done: bool = False
    threshold_db: Optional[float] = None
    trial_num: int = 0  # trials completed on this track

    @property
    def step_db(self) -> float:
        return _STEP_SMALL_DB if len(self.reversals) >= _REVERSALS_FOR_SMALL_STEP else _STEP_LARGE_DB

    def update(self, correct: bool) -> None:
        """Apply one 2AFC response and update level."""
        if self.done:
            return
        self.trial_num += 1

        if correct:
            self.correct_streak += 1
            if self.correct_streak >= 3:
                # 3-down: go harder (more negative SNR)
                self.correct_streak = 0
                new_direction = -1
                new_level = max(self.level_db - self.step_db, _MIN_LEVEL_DB)
                if self.last_direction == 1:
                    self.reversals.append(self.level_db)
                self.last_direction = new_direction
                self.level_db = new_level
        else:
            # 1-up: go easier (less negative SNR)
            self.correct_streak = 0
            new_direction = 1
            new_level = min(self.level_db + self.step_db, _MAX_LEVEL_DB)
            if self.last_direction == -1:
                self.reversals.append(self.level_db)
            self.last_direction = new_direction
            self.level_db = new_level

        if len(self.reversals) >= _TOTAL_REVERSALS:
            last = self.reversals[-_THRESHOLD_REVERSALS:]
            self.threshold_db = sum(last) / len(last)
            self.done = True

    @property
    def estimated_threshold(self) -> Optional[float]:
        """Running estimate from available reversals (last up to 6)."""
        if not self.reversals:
            return None
        usable = self.reversals[-_THRESHOLD_REVERSALS:]
        return sum(usable) / len(usable)


# ---------------------------------------------------------------------------
# Session state (single-process, sequential listeners)
# ---------------------------------------------------------------------------
_tracks: list[StaircaseTrack] = [
    StaircaseTrack(name="subharmonic"),
    StaircaseTrack(name="am"),
]
# Which track goes next (interleave)
_next_track: int = 0
# Has result already been saved?
_result_saved: bool = False


def _reset_session() -> None:
    global _tracks, _next_track, _result_saved
    _tracks = [StaircaseTrack(name="subharmonic"), StaircaseTrack(name="am")]
    _next_track = 0
    _result_saved = False


def _advance_next_track() -> None:
    global _next_track
    # Pick whichever non-done track comes next in round-robin
    for _ in range(2):
        _next_track = (_next_track + 1) % 2
        if not _tracks[_next_track].done:
            return
    # Both done — keep at 0
    _next_track = 0


def _build_trial_response(track_idx: int) -> dict:
    """Build the JSON payload for the next trial on track_idx."""
    track = _tracks[track_idx]
    # Randomise which interval is treated
    treated_interval = random.choice([1, 2])
    total_trials = sum(t.trial_num for t in _tracks) + 1
    return {
        "track": track_idx,
        "track_name": track.name,
        "treated_interval": treated_interval,
        "interval_1_path": "/stimuli/treated" if treated_interval == 1 else "/stimuli/reference",
        "interval_2_path": "/stimuli/reference" if treated_interval == 1 else "/stimuli/treated",
        "current_level_db": track.level_db,
        "trial_num": total_trials,
        "reversals_done": len(track.reversals),
        "total_reversals": _TOTAL_REVERSALS,
        "done": False,
        "threshold": track.estimated_threshold,
    }


def _save_results() -> None:
    global _result_saved
    if _result_saved:
        return
    out = {}
    for t in _tracks:
        out[t.name] = {
            "threshold_snr_db": t.threshold_db,
            "reversals": t.reversals,
            "trials_completed": t.trial_num,
        }
    _RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS_JSON.write_text(json.dumps(out, indent=2))
    _result_saved = True


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Acoustic Persuasion Lab — Threshold Pilot")

app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/")
async def index():
    index_html = _STATIC / "index.html"
    if index_html.exists():
        return FileResponse(str(index_html))
    raise HTTPException(status_code=404, detail="index.html not found in static/")


@app.get("/stimuli/treated")
async def stimuli_treated():
    if not _STIMULI_TREATED.exists():
        raise HTTPException(status_code=404, detail=f"Treated stimulus not found: {_STIMULI_TREATED}")
    return FileResponse(str(_STIMULI_TREATED), media_type="audio/wav")


@app.get("/stimuli/reference")
async def stimuli_reference():
    if not _STIMULI_REFERENCE.exists():
        raise HTTPException(status_code=404, detail=f"Reference stimulus not found: {_STIMULI_REFERENCE}")
    return FileResponse(str(_STIMULI_REFERENCE), media_type="audio/wav")


class TrialRequest(BaseModel):
    track: int          # 0 or 1
    response: int       # 1 or 2 (which interval the listener chose); -1 = bootstrap (first call)
    interval: int       # which interval actually contained the layer (treated_interval); -1 = bootstrap


@app.post("/trial")
async def post_trial(body: TrialRequest):
    global _next_track

    # Bootstrap call: return first trial without updating any state
    if body.response == -1 and body.interval == -1:
        # Find first non-done track
        for idx in range(2):
            if not _tracks[idx].done:
                _next_track = idx
                return _build_trial_response(idx)
        # Both already done
        return _all_done_response()

    # Validate track index
    if body.track not in (0, 1):
        raise HTTPException(status_code=400, detail="track must be 0 or 1")

    track = _tracks[body.track]
    if track.done:
        raise HTTPException(status_code=400, detail=f"Track {body.track} is already complete")

    # Evaluate correctness
    correct = (body.response == body.interval)
    track.update(correct)

    # Check if all done
    all_done = all(t.done for t in _tracks)
    if all_done:
        _save_results()
        return _all_done_response()

    # Advance to next non-done track
    _advance_next_track()
    return _build_trial_response(_next_track)


def _all_done_response() -> dict:
    return {
        "done": True,
        "threshold": None,
        "track": -1,
        "track_name": None,
        "treated_interval": None,
        "interval_1_path": None,
        "interval_2_path": None,
        "current_level_db": None,
        "trial_num": sum(t.trial_num for t in _tracks),
        "reversals_done": None,
        "total_reversals": _TOTAL_REVERSALS,
        "thresholds": {t.name: t.threshold_db for t in _tracks},
    }


@app.get("/results")
async def get_results():
    if not all(t.done for t in _tracks):
        incomplete = [t.name for t in _tracks if not t.done]
        raise HTTPException(status_code=400, detail=f"Tracks not yet complete: {incomplete}")
    _save_results()
    return JSONResponse({
        "subharmonic": {
            "threshold_snr_db": _tracks[0].threshold_db,
            "reversals": _tracks[0].reversals,
            "trials_completed": _tracks[0].trial_num,
        },
        "am": {
            "threshold_snr_db": _tracks[1].threshold_db,
            "reversals": _tracks[1].reversals,
            "trials_completed": _tracks[1].trial_num,
        },
    })


@app.post("/reset")
async def reset_session():
    """Start a new listener session (sequential n=12 use)."""
    _reset_session()
    return {"status": "reset", "message": "Session reset. Ready for next listener."}
