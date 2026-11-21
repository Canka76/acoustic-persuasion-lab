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
reduces power. A single permitted replay is standard in self-paced psycholinguistics
paradigms (e.g. Clifton et al. 2003).

### Direction of expected bias
Replay increases acoustic exposure for self-selecting participants. If arm
assignment is unconfounded with replay rate (expected under randomization), the
net effect is reduced measurement error (less noise) rather than systematic bias
on the B–A contrast. However:

1. **H5 (log-RT)**: RT is anchored to the **first play offset** (not replay
   offset), preserving the registered timing discipline. Trials with replay will
   have longer wall-clock RT but the logged `rt_ms` reflects post-first-play
   decision time, same as non-replay trials.

2. **H1/H2 (d-prime)**: Replay may selectively help participants detect fallacies
   (ceiling pressure on accuracy). Pre-specified sensitivity analysis: re-run
   primary H1/H2 test excluding trials with replay events (`n_replay > 0`).
   If effect direction reverses or significance changes, the result is exploratory.

3. **H3 (sub-perceptual / awareness)**: Additional exposure increases detection
   opportunity for the acoustic manipulation. The 2AFC manipulation check
   (single presentation, no replay) is unaffected. Report replay rate per arm
   as a secondary table.

### Analysis additions (not registered, exploratory)
- Table: replay rate by arm and block
- Sensitivity: H1/H2 excluding replayed trials
- Covariate: `has_replay` (0/1) added to mixed model as nuisance regressor
