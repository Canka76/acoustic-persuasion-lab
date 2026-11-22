# Deviations from Pre-Registration

Per §9 of `preregistration.md`: any deviation must be documented here with date,
rationale, and direction of expected bias. Undocumented deviations make
confirmatory claims invalid; affected analyses become exploratory.

---

## DEV-001 — One-shot audio replay per trial

**Date:** 2026-11-21  
**Type:** Deviation from single-presentation design  
**Status:** Pre-data (no participants run in main study yet)

### What changed
The registered design specified single audio presentation per trial (implicit in
§4 "listen to each clip"). The implemented study app (commit `e5c1e3b`, amended
`3F24`) adds a one-shot replay button that participants may use after the first
play completes. Replay is logged as an `event` row with `first_play_offset` and
`replay_offset` timestamps.

### Rationale
Internal dry-run tester (n=1, arm A) reported audio was "hard to understand"
and felt "too fast" with no way to re-listen. Without replay, unintelligible
trials produce floor-performance noise that inflates within-arm variance and
reduces power. A single permitted replay was added to address this. No citation
is provided — this is a pragmatic decision based on observed usability failure,
not a precedent from the literature.

### Direction of expected bias
Replay increases acoustic exposure for self-selecting participants. If arm
assignment is unconfounded with replay rate (expected under randomization), the
net effect is reduced measurement error (less noise) rather than systematic bias
on the B–A contrast. However:

1. **H5 (log-RT)**: Replayed trials have `has_replay = 1` and are **excluded
   from H5 by design**. `rt_invalid` remains binary (0/1) as registered.
   The `rt_ms` for a replayed trial is anchored to first-play offset but
   includes the replay duration (~12–16 s), making it incomparable to
   single-presentation trials. `analyze.py` filters `has_replay == 0` for H5.
   Nonrandom missingness risk: if replay rate differs by arm (e.g. arm B
   participants replay more due to manipulation distraction), H5 power is
   asymmetrically reduced. Report replay rate per arm as a check.

2. **H1/H2 (d-prime)**: Replay may selectively help participants detect fallacies
   (ceiling pressure on accuracy). Pre-specified sensitivity analysis: re-run
   primary H1/H2 test excluding trials with replay (`has_replay == 1`).
   If effect direction reverses or significance changes, the result is exploratory.

3. **H3 (sub-perceptual / awareness)**: Additional exposure increases detection
   opportunity for the acoustic manipulation. The 2AFC manipulation check
   (single presentation, no replay) is unaffected. Report replay rate per arm
   as a secondary table.

### Analysis additions (not registered, exploratory)
- Table: replay rate by arm and block
- Sensitivity: H1/H2 excluding replayed trials (`has_replay == 1`)
- Covariate: `has_replay` (0/1) added to mixed model as nuisance regressor
- Check: replay rate per arm (test for arm × replay interaction)

---

## DEV-002 — Headphone screen retry (capped at 2 attempts)

**Date:** 2026-11-21  
**Type:** Relaxation of hard gate from prereg §5 exclusion 1  
**Status:** Pre-data

### What changed
Prereg §5 excludes participants who fail the 6-trial Huggins-pitch headphone
screen (< 5/6 correct) with no retry. The implemented app allows one retry
(maximum 2 attempts total). Each attempt is logged as a `headphone_screen`
event with `attempt` index (1 or 2) and `score`.

### Rationale
Internal dry-run tester passed the headphone check repeatedly while using
in-ear buds but failed twice on the first attempt due to unclear instructions
(did not know what "tone in the centre" meant). The instructions were improved
(see commit `2bfa410`) and a single retry was added so legitimate headphone
users who fail on first attempt due to instruction ambiguity are not excluded.

### Why 2 attempts (not unlimited)
P(passing 6-trial 2AFC by guessing) ≈ 7/64 ≈ 11% per attempt. With unlimited
retries a speaker user passes within ~10 attempts. With a 2-attempt cap,
P(speaker user passing) ≥ 1 − (1 − 7/64)² ≈ 20% — this is a **lower bound**:
the cap is enforced client-side only (JavaScript state). A participant who
reloads the page starts a fresh session with `hpAttempt = 1`, circumventing
the cap. The terminal-fail message does **not** suggest reloading (commit
`96ff5ae` removed that instruction). Motivated speaker users can still reload,
so this gate is best treated as a soft screen, not a hard exclude.
The gate remains materially tighter than no screen at all.

### Direction of expected bias
Participants who pass on attempt 2 may have lower headphone compliance than
attempt-1 passers. If arm assignment is balanced across attempt index (expected
under randomization), there is no systematic directional bias on H1/H2.

### Analysis additions (exploratory)
- Sensitivity: re-run H1/H2 restricting to attempt-1 passers only
  (`event.data->>attempt == 1` in the headphone_screen event log)


---

## DEV-003 — WORLD vocoder inherent amplitude jitter; unremovable by parameter tuning

**Date:** 2026-11-22  
**Type:** Instrumentation limitation; no code change made  
**Status:** Pre-data (dry run only)

### Finding
Internal dry-run tester (1 completed arm A session, `392b1726`) reported
"very noticeable jitter" in the stimulus audio. Systematic probe confirmed
the WORLD analysis/synthesis round-trip introduces a **+17.7% increase in
short-time amplitude envelope jitter** relative to the Piper TTS source
(source: 0.0121, arm A delivered: 0.0142, 5ms RMS envelope frames).

### Root cause investigation (probe `scratch_jitter_probe.py`, deleted)
All targeted fixes were tested and falsified:

| Intervention | Result |
|---|---|
| Harvest octave-error correction (`_clean_f0`) | F0 synthesis has only 2–5 real oct-jumps; `_clean_f0` worsened V/UV flips on some items |
| Spectral envelope temporal smoothing (3-pt weighted) | −0.1% amplitude jitter change |
| Aperiodicity (D4C) temporal smoothing | −0.1% |
| Finer frame period (2 ms vs 5 ms) | 0% change, slightly worse |
| `d4c(threshold=0)` (documented HQ mode) | +0.6% (worse) |
| Hard-clip limiter engagement | 0 samples clipped (peak at −5.32 dBFS); limiter not a factor |
| Existing rendered files vs fresh synthesis | corr=0.9990 (files are correct) |

The +17.7% amplitude jitter is a property of WORLD's harmonic pulse-train synthesis
model and cannot be reduced with WORLD parameter changes.

### Design consequence
The WORLD round-trip is applied **identically to all three arms** (arm A: strength=0
passthrough; arms B/C: same analysis path, then F0 pull / placebo added). The
vocoder quality is therefore a constant nuisance variable present in all arms
and does **not** create directional bias in the B−A contrast (H1/H2).

Participants may notice the vocoded quality and report manipulation "awareness"
in the manipulation check even when in arm A. This is expected and noted in the
study instructions ("audio has been processed"). It does not invalidate H1/H2.

### No code change
No code change was made as a result of this investigation. The WORLD pipeline
remains unchanged. This entry exists for transparency per prereg §9.
