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

1. **H5 (log-RT)**: Replayed trials receive `rt_invalid = 2` and are **excluded
   from H5 by design**. The `rt_ms` for a replayed trial is measured from the
   first-play offset but includes the replay duration (~12–16 s), making it
   incomparable to non-replay trials and routinely exceeding the 20 s cap. Code 2
   marks this as an explicit, designed exclusion rather than a silent dropout.
   Nonrandom missingness risk: if replay rate differs by arm (e.g. arm B
   participants replay more due to manipulation distraction), H5 power is
   asymmetrically reduced. Report replay rate per arm as a check.

2. **H1/H2 (d-prime)**: Replay may selectively help participants detect fallacies
   (ceiling pressure on accuracy). Pre-specified sensitivity analysis: re-run
   primary H1/H2 test excluding trials with replay events (`rt_invalid == 2`).
   If effect direction reverses or significance changes, the result is exploratory.

3. **H3 (sub-perceptual / awareness)**: Additional exposure increases detection
   opportunity for the acoustic manipulation. The 2AFC manipulation check
   (single presentation, no replay) is unaffected. Report replay rate per arm
   as a secondary table.

### Analysis additions (not registered, exploratory)
- Table: replay rate by arm and block
- Sensitivity: H1/H2 excluding replayed trials (`rt_invalid == 2`)
- Covariate: `has_replay` (0/1) added to mixed model as nuisance regressor
- Check: replay rate per arm (test for arm × replay interaction)

