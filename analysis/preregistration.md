# Pre-registration: Sub-perceptual Acoustic Persuasion RCT

**Status: STUB — to be locked before any human data collection.**

## Hypotheses

- **H1**: Arm B (treatment) produces lower d′ on the fallacy detection task than Arm A (control).
- **H2**: Arm B produces higher trust allocation than Arm A.
- **H3**: Participants in Arm B cannot reliably detect the acoustic manipulation (BF₀₁ ≥ 3 on 2AFC; per-subject d′ ≈ 0).
- **H4**: The `acoustic_firewall` detector achieves ROC-AUC ≥ 0.95 on clean held-out data.

## Sample Size

N = 204 (68/arm), powering d = 0.49 at 80%, α = .05 two-tailed.
Over-recruitment: +10% for exclusions → target enrolment 225.

## Exclusion Rules (pre-registered)

1. Failed headphone screen (< 5/6 Huggins-pitch trials).
2. RT < 200 ms or > 20 s on > 50% of trials.
3. Self-reported hearing impairment.
4. Incomplete session (< 14 trial responses).

## Primary Outcome

Per-subject d′ on 8-item fallacy block (loglinear correction +0.5/+1).
One-way permutation ANOVA (10,000 permutations) across arms.
Planned contrasts B−A and B−C, Holm-corrected.
Effect sizes with BCa bootstrap 95% CIs.

## Secondary Outcomes

- Trust allocation (Welch + rank-based).
- Log-RT (per-subject median log-RT, Welch ANOVA).
- Confidence-accuracy calibration slope.

## H3 Decision Rule

H3 supported only if BF₀₁ ≥ 3 AND significant B−A effect survives when restricted
to participants with 2AFC at chance.

## Blinding

Arm stored as `arm_code` mapped via `analysis/unblind_key.json`, written at render time.
`analyze.py` must not read `unblind_key.json` until primary analysis is locked.

## Analysis Functions

See `analysis/analyze.py`. All test functions fixed in this document before unblinding.
