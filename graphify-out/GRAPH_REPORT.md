# Graph Report - acoustic-persuasion-lab  (2026-11-21)

## Corpus Check
- Corpus is ~20,094 words - fits in a single context window. You may not need a graph.

## Summary
- 418 nodes · 819 edges · 21 communities (13 shown, 8 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 12 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Audio DSP Core
- Pre-registration & Hypotheses
- Test Harness & Fixtures
- Statistical Analysis
- Analysis Pipeline
- Threshold Report
- Loudness Normalization
- Threshold Pilot App
- Study FastAPI App
- Dependency & Boundary Tests
- Acoustic Firewall Detector
- Speech Quality Metrics
- Workspace Packages
- Analyst Blinding Protocol

## God Nodes (most connected - your core abstractions)
1. `render_all()` - 17 edges
2. `load()` - 14 edges
3. `_post_process()` - 13 edges
4. `_build_arm_b()` - 12 edges
5. `pull_f0()` - 11 edges
6. `_build_arm_c()` - 11 edges
7. `planned_contrast()` - 10 edges
8. `_get_conn()` - 10 edges
9. `extract()` - 10 edges
10. `save()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Threshold Detection Task (2AFC adaptive staircase)` --conceptually_related_to--> `threshold_pilot app`  [INFERRED]
  apps/threshold_pilot/static/index.html → README.md
- `Study Admin Panel` --conceptually_related_to--> `study app (FastAPI RCT platform)`  [INFERRED]
  apps/study/static/admin.html → README.md
- `Study participant flow (8 sections)` --conceptually_related_to--> `study app (FastAPI RCT platform)`  [INFERRED]
  apps/study/static/index.html → README.md
- `Threshold Detection Task (2AFC adaptive staircase)` --conceptually_related_to--> `H3: Sub-perceptual claim — participants unaware (BF01 >= 3 on 2AFC)`  [INFERRED]
  apps/threshold_pilot/static/index.html → analysis/preregistration.md
- `test_am_coherence_exceeds_clean()` --calls--> `extract()`  [EXTRACTED]
  tests/test_firewall_features.py → packages/acoustic_firewall/features.py

## Import Cycles
- None detected.

## Communities (21 total, 8 thin omitted)

### Community 0 - "Audio DSP Core"
Cohesion: 0.08
Nodes (18): load(), rms(), save(), sha256(), Act, segment(), duck_to_snr(), _make_act_lookup() (+10 more)

### Community 1 - "Pre-registration & Hypotheses"
Cohesion: 0.05
Nodes (44): d-prime primary outcome (log-linear correction, per-subject), Exclusion criteria (5 pre-specified, applied before unblinding), H1: Arm B shows lower d-prime on fallacy detection than Arm A, H2: Treatment effect exceeds placebo (d-prime B < C), H3: Sub-perceptual claim — participants unaware (BF01 >= 3 on 2AFC), H4: Trust allocation higher in Arm B than Arm A, H5: Median log-RT differs across arms (exploratory), Headphone screen: 6-trial antiphase (Milne et al. 2021), >=5/6 required (+36 more)

### Community 2 - "Test Harness & Fixtures"
Cohesion: 0.09
Nodes (21): _base_render_path(), _f0_pitch_class_histogram(), _load_corpus(), _load_wav(), _local_median_prominence(), _manifest_path(), _modulation_spectrum(), _rms() (+13 more)

### Community 3 - "Statistical Analysis"
Cohesion: 0.08
Nodes (12): bayes_factor_binom(), decode(), _pn_sequence(), db(), log_spectral_distance(), segmental_snr(), make_placebo_layer(), _shape_noise_to_layer() (+4 more)

### Community 4 - "Analysis Pipeline"
Cohesion: 0.10
Nodes (16): apply_exclusions(), _bca_ci(), dprime_loglinear(), holm_correct(), load_data(), main(), per_subject_dprime(), permutation_anova() (+8 more)

### Community 5 - "Threshold Report"
Cohesion: 0.09
Nodes (15): main(), cmd_clean(), cmd_scan(), cmd_verify_watermark(), _load_wav(), main(), extract(), load_model() (+7 more)

### Community 6 - "Loudness Normalization"
Cohesion: 0.13
Nodes (11): measure_lufs(), normalize_lufs(), peak_limit(), true_peak_db(), EdgeTtsBackend, PiperBackend, _post_process(), render_corpus() (+3 more)

### Community 7 - "Threshold Pilot App"
Cohesion: 0.10
Nodes (13): _advance_next_track(), _all_done_response(), _build_trial_response(), get_results(), index(), post_trial(), reset_session(), _reset_session() (+5 more)

### Community 8 - "Study FastAPI App"
Cohesion: 0.15
Nodes (16): admin_participants(), create_participant(), EventIn, export_csv(), _get_conn(), get_stimulus(), health(), index() (+8 more)

### Community 9 - "Dependency & Boundary Tests"
Cohesion: 0.10
Nodes (8): _load_or_run(), sim_results(), test_power_geq_080(), test_type_i_in_range(), _imports_in_file(), test_acoustic_firewall_does_not_import_sap_stimulus(), _imports_in_file(), test_apps_do_not_import_sap_stimulus()

### Community 10 - "Acoustic Firewall Detector"
Cohesion: 0.12
Nodes (6): DetectReport, neutralize(), _energy_boundaries(), sentence_boundaries(), SentenceBoundary, _split_sentences()

### Community 11 - "Speech Quality Metrics"
Cohesion: 0.31
Nodes (4): estoi(), log_spectral_distance(), main(), segmental_snr()

### Community 12 - "Workspace Packages"
Cohesion: 0.83
Nodes (4): acoustic-firewall, acoustic-persuasion-lab, sap-core, sap-stimulus

## Knowledge Gaps
- **20 isolated node(s):** `sap_core package`, `firewall_worklet app (AudioWorklet real-time demo)`, `acoustic_firewall: Apache-2.0 license`, `sap_stimulus: PolyForm Noncommercial 1.0.0 license`, `H5: Median log-RT differs across arms (exploratory)` (+15 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 162 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `render_all()` connect `Audio DSP Core` to `Statistical Analysis`, `Threshold Report`, `Loudness Normalization`?**
  _High betweenness centrality (0.019) - this node is a cross-community bridge._
- **What connects `sap_core package`, `firewall_worklet app (AudioWorklet real-time demo)`, `acoustic_firewall: Apache-2.0 license` to the rest of the system?**
  _20 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Audio DSP Core` be split into smaller, more focused modules?**
  _Cohesion score 0.07510204081632653 - nodes in this community are weakly interconnected._
- **Why does `load()` connect `Audio DSP Core` to `Speech Quality Metrics`, `Threshold Report`, `Loudness Normalization`?**
  _High betweenness centrality (0.018) - this node is a cross-community bridge._
- **Should `Pre-registration & Hypotheses` be split into smaller, more focused modules?**
  _Cohesion score 0.05272895467160037 - nodes in this community are weakly interconnected._
- **Should `Test Harness & Fixtures` be split into smaller, more focused modules?**
  _Cohesion score 0.08562367864693446 - nodes in this community are weakly interconnected._
- **Should `Statistical Analysis` be split into smaller, more focused modules?**
  _Cohesion score 0.07564102564102564 - nodes in this community are weakly interconnected._