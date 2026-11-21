# Pre-Registration: Sub-Perceptual Acoustic Persuasion Study

**Study title:** Does macro-harmonic scaffolding with sub-perceptual amplitude modulation under neural TTS affect critical reasoning?  
**Principal investigator:** [PI NAME]  
**Date registered:** 2026-11-21 (before any data collection)  
**Version:** 1.0 — LOCKED. This document must not be altered after participant data collection begins.

---

## 1. Hypotheses

**H1 (Primary — fallacy detection):** Participants in Arm B (treatment: F0-pull + subharmonic drone + coherent 10 Hz AM) will show lower per-subject d′ on the 8-item fallacy detection task than participants in Arm A (control: unprocessed speech), i.e. `d′_B < d′_A`. Planned contrast B−A.

**H2 (Primary — placebo control):** The effect of treatment exceeds that of the placebo, i.e. `d′_B < d′_C`. Planned contrast B−C. This isolates the specific harmonic/entrainment structure from incidental added-energy effects.

**H3 (Sub-perceptual claim):** Participants are unaware of the acoustic manipulation. Criterion: Bayes factor BF₀₁ ≥ 3 on the 2AFC manipulation-check trial (against chance = 0.5), computed within each arm, AND a significant H1/H2 effect must survive when the analysis is restricted to participants whose 2AFC trial was at chance.

**H4 (Secondary — trust):** Trust allocation toward AI advisor recommendations (units allocated to the advisor-endorsed option) will be higher in Arm B than Arm A.

**H5 (Secondary — RT):** Median log-RT per participant will differ across arms (exploratory direction).

---

## 2. Design

- **Type:** 3-arm parallel-group RCT  
- **Arms:** A = control (unprocessed speech), B = treatment (macro-harmonic + AM), C = active placebo (matched energy/spectrum, jittered AM, no F0 pull)  
- **Assignment:** Server-side round-robin, masked from participants and analyst  
- **Blinding:** Analyst reads `analysis/unblind_key.json` only after primary analysis code is locked (after `analysis/preregistration.md` is committed and primary results function is run)  
- **N:** 204 (68/arm), + 10 % over-recruitment buffer for exclusions  
- **Power basis:** d = 0.49, α = .05 two-tailed, 80 % power (simulation-verified in `tests/test_analysis_power.py` before data collection)

---

## 3. Participants

- Recruited via Prolific (or university convenience sample if Prolific unavailable)  
- Inclusion: age 18+, self-reported English as native language, no self-reported hearing impairment  
- Headphone requirement: verified via 6-trial antiphase screen (Milne et al. 2021 protocol); ≥ 5/6 required  
- Monetary compensation stated in consent

---

## 4. Measures

### Primary outcome
Per-subject d′ on the 8-item fallacy detection block.  
- Signal = fallacious items (4), Noise = sound items (4)  
- Response: binary "sounds valid" / "contains a reasoning flaw"  
- d′ computed with log-linear correction: add 0.5 to hits and false alarms, add 1 to signal and noise counts  
- One d′ value per participant

### Secondary outcomes
- Trust allocation: mean units allocated to AI-endorsed option across 4 trust trials (0–100 slider)  
- Median log-RT per participant on the fallacy block (trials with RT < 200 ms or > 20 s excluded pre-analysis; RT logged from audio-offset timestamp)  
- Confidence calibration slope: regression of confidence (1–7) on accuracy (0/1) per participant, slope averaged per arm

### Manipulation check
- Q1: Forced-choice yes/no "Did you hear background music, hum, or tonal modulation?"  
- Q2: 2AFC discrimination trial: "Which of these two clips was processed?" (arm's arg_03.wav vs Arm A's arg_03.wav)  
- Per-arm binomial test of Q2 against p=0.5, and BF₀₁ (Jeffreys prior, `pingouin.bayesfactor_binom`)

---

## 5. Exclusion criteria (pre-specified, applied before unblinding)

1. Failed headphone screen (< 5/6 correct) — excluded at session gate, no data stored  
2. > 50 % of fallacy-block trials flagged `rt_invalid` (RT < 200 ms or > 20 s)  
3. Self-reported hearing impairment (collected at demographics, applied to all analyses)  
4. AudioContext statechange to `suspended`/`interrupted` during a trial block (logged via /event)  
5. Duplicate Prolific ID (keep first submission only)

---

## 6. Statistical analysis

All analyses run via `analysis/analyze.py`. Analyst unblinding sequence:
1. Run exclusion filter on blinded data  
2. Lock analysis code (git commit)  
3. Run `analyze.py --key analysis/unblind_key.json` — this reads the key and decodes arm labels  
4. Output `results/report.html` — no further analysis after this point

### Primary test
One-way **permutation ANOVA** (10,000 permutations, seed=42) on per-subject d′ across 3 arms. Report F-statistic and permutation p-value.

**Planned contrasts** (run regardless of ANOVA p-value, Holm-corrected):
- Contrast 1 (H1): B − A (Welch two-sample t-test + permutation p)  
- Contrast 2 (H2): B − C (Welch two-sample t-test + permutation p)

**Effect sizes:** Cohen's d with BCa bootstrap 95 % CI (10,000 resamples, seed=42)

### Secondary tests
- H4 (trust): One-way Welch ANOVA + rank-based Kruskal-Wallis; planned B−A contrast  
- H5 (log-RT): Per-subject median log-RT; Welch ANOVA across arms  
- Confidence calibration: per-subject slope from logistic regression (confidence → accuracy); arm means compared with Welch ANOVA

### H3 test
1. Per-arm binomial test of 2AFC Q2 against 0.5  
2. BF₀₁ via `pingouin.bayesfactor_binom` (Jeffreys default prior) — report per arm  
3. H3 supported only if BF₀₁ ≥ 3 (evidence for null) in arm B AND  
4. H1 contrast (B−A) remains significant when restricted to arm B participants whose 2AFC was incorrect (d′ ≈ 0 on the detection trial)

---

## 7. Pilot screens (Step 9)

Before main run:
- Item difficulty screen on pilot n=20 (Arm A only): items with accuracy < 0.25 or > 0.85 in Arm A are replaced before the main run with a new item of the same fallacy type. Pilot data are discarded.
- Replacement must preserve: same fallacy type, 35–55 s target duration, non-political non-emotive topic.

---

## 8. Unblinding key

`analysis/unblind_key.json` maps internal arm codes (1/2/3) to experimental arms (A/B/C). This file is written at stimulus render time and is **not** read by `analyze.py` during the exclusion or analysis-code-lock phase. The key is revealed only when `analyze.py --key` is called.

---

## 9. Deviations policy

Any deviation from this pre-registration must be documented in `analysis/deviations.md` with date, rationale, and direction of expected bias. Undocumented deviations make the confirmatory claims invalid; affected analyses become exploratory.

---
## 10. Dose-finding audit (pre-pilot, ground-truth measurements)

The following physical measurements were made before any participants were run,
using the actual rendered corpus (strength=0.350, SNR=−31 dB).
All |Δ¢| figures use `pull_f0`'s internal pre/post F0 arrays (zero estimator noise).

### F0 manipulation magnitude (strength = 0.350)

| Statistic | Value | Reference |
|---|---|---|
| Median \|Δ¢\| across all voiced frames | **27.4 ¢** | Naive-listener JND ~25–50 ¢ |
| p90 \|Δ¢\| | **58.2 ¢** | Trained-listener JND ~5–10 ¢ |
| Max \|Δ¢\| | 60.0 ¢ (clamp ceiling) | — |

**Assessment:** The median shift (27.4 ¢) is at the lower boundary of the
naive-listener JND range. The p90 (58.2 ¢) is audible to trained listeners.
The clamp bites on ~90% of frames. The sub-perceptual claim for F0 is
**not established** prior to the threshold pilot — it is a hypothesis, not a
confirmed fact. H3 must be treated as open until threshold-pilot data are in hand.

### AM manipulation (SNR = −31 dB, 10 Hz, depth = 3%)

Physical dose-response sweep (B−A margin in 55–75 Hz subharmonic band):

| SNR | Mean B−A margin | Items ≥ 3 dB gate |
|---:|---:|---:|
| −31 dB (operating) | +0.14 dB | 1/10 |
| −25 dB | +0.36 dB | 1/10 |
| −20 dB | +2.24 dB | 3/10 |
| −15 dB | +4.74 dB | 6/10 |
| −10 dB | +6.48 dB | 7/10 |

**Assessment:** At the operating SNR (−31 dB), the 10 Hz modulation is
indistinguishable from baseline noise in a narrowband analysis. The
threshold pilot must establish the actual psychoacoustic modulation detection
threshold before the AM dose can be confirmed as sub-perceptual-yet-present.
If the detection threshold is better (more sensitive) than −15 dB SNR,
"sub-perceptual" and "physically detectable in the delivered mix" are
mutually exclusive in this pipeline — a finding that narrows or refutes H1/H2.


*End of pre-registration. SHA-256 of this file at lock time: [computed at commit]*
