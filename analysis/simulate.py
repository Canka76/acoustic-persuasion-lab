"""Power simulation for the pre-registered analysis.

Runs 1000 datasets under two conditions:
  (a) effect:  d′_B = d′_A + 0.45  (H1 effect size)
  (b) null:    all arms equal

For each dataset, runs the permutation ANOVA and planned B-A contrast.
Saves analysis/simulation_results.json with power and type-I estimates.

Usage:
    python analysis/simulate.py
    python analysis/simulate.py --n-sims 2000 --n-per-arm 68 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent))
from analyze import permutation_anova, planned_contrast


def simulate_dataset(
    n_per_arm: int,
    delta_d: float,
    dprime_mean_a: float,
    dprime_sd: float,
    rng: np.random.Generator,
) -> dict:
    """Simulate one dataset. Returns p-values for ANOVA and B-A contrast."""
    d_a = rng.normal(dprime_mean_a,          dprime_sd, n_per_arm)
    d_b = rng.normal(dprime_mean_a + delta_d, dprime_sd, n_per_arm)
    d_c = rng.normal(dprime_mean_a,           dprime_sd, n_per_arm)

    anova  = permutation_anova([d_a, d_b, d_c], n_perm=1000, seed=int(rng.integers(1 << 31)))
    contra = planned_contrast(d_b, d_a, label="B-A")
    return {
        "anova_p":    anova["p_perm"],
        "contrast_p": contra["p_welch"],
        "cohens_d":   contra["cohens_d"],
    }


def run(
    n_sims: int = 1000,
    n_per_arm: int = 68,
    effect_delta: float = 0.45,
    dprime_mean_a: float = 0.8,
    dprime_sd: float = 0.9,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)

    effect_results: list[dict] = []
    null_results:   list[dict] = []

    print(f"Running {n_sims} simulations under effect (Δd′={effect_delta}) …", flush=True)
    for i in range(n_sims):
        effect_results.append(simulate_dataset(n_per_arm, effect_delta, dprime_mean_a, dprime_sd, rng))
        if (i + 1) % 100 == 0:
            print(f"  effect {i+1}/{n_sims}", flush=True)

    print(f"Running {n_sims} simulations under null (Δd′=0) …", flush=True)
    for i in range(n_sims):
        null_results.append(simulate_dataset(n_per_arm, 0.0, dprime_mean_a, dprime_sd, rng))
        if (i + 1) % 100 == 0:
            print(f"  null   {i+1}/{n_sims}", flush=True)

    power_anova    = float(np.mean([r["anova_p"]    < alpha for r in effect_results]))
    power_contrast = float(np.mean([r["contrast_p"] < alpha for r in effect_results]))
    type1_anova    = float(np.mean([r["anova_p"]    < alpha for r in null_results]))
    type1_contrast = float(np.mean([r["contrast_p"] < alpha for r in null_results]))

    results = {
        "parameters": {
            "n_sims": n_sims, "n_per_arm": n_per_arm,
            "effect_delta_dprime": effect_delta,
            "dprime_mean_a": dprime_mean_a, "dprime_sd": dprime_sd,
            "alpha": alpha, "seed": seed,
        },
        "power": {
            "anova":           power_anova,
            "contrast_BA":     power_contrast,
        },
        "type1_error": {
            "anova":           type1_anova,
            "contrast_BA":     type1_contrast,
        },
        "gate_pass": {
            "power_ge_0.80":   power_contrast >= 0.80,
            "type1_in_range":  0.03 <= type1_contrast <= 0.07,
        },
    }

    out = Path(__file__).parent / "simulation_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nPower (B-A contrast): {power_contrast:.3f}")
    print(f"Type-I (B-A contrast): {type1_contrast:.3f}")
    print(f"Gate: power≥0.80 → {'PASS' if results['gate_pass']['power_ge_0.80'] else 'FAIL'}, "
          f"type-I in [0.03,0.07] → {'PASS' if results['gate_pass']['type1_in_range'] else 'FAIL'}")
    print(f"Results written to {out}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Power simulation for SAP RCT")
    ap.add_argument("--n-sims",     type=int,   default=1000)
    ap.add_argument("--n-per-arm",  type=int,   default=68)
    ap.add_argument("--effect",     type=float, default=0.45,
                    help="Expected d′ difference B−A under H1")
    ap.add_argument("--dprime-sd",  type=float, default=0.9)
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    run(
        n_sims=args.n_sims, n_per_arm=args.n_per_arm,
        effect_delta=args.effect, dprime_sd=args.dprime_sd, seed=args.seed,
    )


if __name__ == "__main__":
    main()
