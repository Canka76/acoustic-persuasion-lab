# Acoustic Persuasion Lab

A research monorepo for empirically testing and neutralizing sub-perceptual acoustic persuasion techniques.

## Structure

```
packages/
  sap_core/          # Shared audio I/O, loudness, alignment, metrics
  sap_stimulus/      # LAB-ONLY stimulus generator (not a service)
  acoustic_firewall/ # PRODUCT: detector + neutralizer
apps/
  study/             # Web RCT platform (FastAPI + plain ES modules)
  threshold_pilot/   # 2AFC adaptive-staircase perceptual threshold app
  firewall_worklet/  # AudioWorklet real-time demo
corpus/
  scripts/           # Stimulus text corpus (JSON)
analysis/            # Pre-registered analysis pipeline
tests/               # Boundary + ethics gate tests
```

## Experimental Arms

| Arm | Label     | Description                                                            |
|-----|-----------|------------------------------------------------------------------------|
| A   | Control   | Unmodified base TTS render; byte-identical source across all arms      |
| B   | Treatment | Macro-harmonic scaffolding + sub-perceptual AM entrainment applied     |
| C   | Placebo   | Energy- and spectral-envelope-matched noise layer; AM frequency-jittered; F0 unmodified |

All three arms share bit-identical source audio; they differ only in post-processing.

## Execution Priority

1. **Effect first**: Demonstrate (or falsify) the persuasion effect empirically via the RCT (H1–H3).
2. **Firewall second**: Train, evaluate, and ship `acoustic_firewall` whose claims are bounded by measured ROC-AUC, not by the proposal's rhetoric.

The stimulus generator (`sap_stimulus`) exposes no HTTP surface and is never imported by the study app. The watermark is mandatory on every generated file.

## Setup

```bash
uv sync --all-extras
uv run pytest tests/test_dependency_boundary.py tests/test_no_generator_endpoint.py -q
```

## License

- `acoustic_firewall`: Apache-2.0
- `sap_stimulus`: PolyForm Noncommercial 1.0.0
