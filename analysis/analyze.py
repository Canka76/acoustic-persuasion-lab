"""Pre-registered analysis pipeline.

Usage:
    python analysis/analyze.py --data study.db --key analysis/unblind_key.json --out results/

The --key flag is what triggers unblinding. Do NOT pass it until analysis code
is committed and primary exclusions are finalized (see preregistration.md §8).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(db_path: Path, key_path: Path | None = None) -> pd.DataFrame:
    """Load trial+response+participant data. Decode arm only if key_path given."""
    con = sqlite3.connect(str(db_path))
    df = pd.read_sql_query("""
        SELECT
            p.id          AS participant_id,
            p.arm_code,
            p.created_at,
            t.id          AS trial_id,
            t.item_id,
            t.block,
            t.position,
            r.rt_ms,
            r.rt_invalid,
            r.answer,
            r.confidence,
            r.allocation,
            r.audio_offset_ts
        FROM participant p
        JOIN trial t ON t.participant_id = p.id
        LEFT JOIN response r ON r.trial_id = t.id
    """, con)
    con.close()

    if key_path is not None and key_path.exists():
        key = json.loads(key_path.read_text())
        df["arm"] = df["arm_code"].map(key)
    else:
        df["arm"] = df["arm_code"]  # use raw code (blinded)

    return df


def apply_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply pre-registered exclusions (§5 of preregistration)."""
    # Mark rt_invalid trials
    df = df.copy()
    df["rt_invalid"] = df["rt_invalid"].fillna(0).astype(bool)

    # Exclude participants with >50% rt_invalid on fallacy block
    fallacy = df[df["block"] == "fallacy"]
    invalid_rate = fallacy.groupby("participant_id")["rt_invalid"].mean()
    keep = invalid_rate[invalid_rate <= 0.5].index
    df = df[df["participant_id"].isin(keep)]

    # Exclude self-reported hearing impairment (would need to join demographics table)
    # Skipped here — apply manually from demographics if needed

    return df


# ── Signal detection ──────────────────────────────────────────────────────────

def dprime_loglinear(hits: int, misses: int, fas: int, crs: int) -> float:
    """Log-linear corrected d′. Adds 0.5 to H/FA counts, 1 to signal/noise totals."""
    h_rate = (hits + 0.5) / (hits + misses + 1.0)
    fa_rate = (fas + 0.5) / (fas + crs + 1.0)
    return float(stats.norm.ppf(h_rate) - stats.norm.ppf(fa_rate))


def per_subject_dprime(df: pd.DataFrame) -> pd.DataFrame:
    """Return one d′ per participant from the fallacy block."""
    fallacy = df[(df["block"] == "fallacy") & (~df["rt_invalid"].fillna(False))].copy()
    # Determine item correctness from corpus (sound=True → correct answer is "valid")
    # We encode: answer=="flaw" on a fallacious item = hit; answer=="flaw" on sound item = FA
    # Items are identified by item_id: arg_05/06/07/08 are fallacious, arg_01-04 are sound
    fallacious_ids = {"arg_05", "arg_06", "arg_07", "arg_08"}

    records = []
    for pid, grp in fallacy.groupby("participant_id"):
        signal_items = grp[grp["item_id"].isin(fallacious_ids)]
        noise_items  = grp[~grp["item_id"].isin(fallacious_ids)]
        hits   = int((signal_items["answer"] == "flaw").sum())
        misses = int((signal_items["answer"] == "valid").sum())
        fas    = int((noise_items["answer"] == "flaw").sum())
        crs    = int((noise_items["answer"] == "valid").sum())
        d = dprime_loglinear(hits, misses, fas, crs)
        arm = grp["arm"].iloc[0]
        records.append({"participant_id": pid, "arm": arm, "dprime": d,
                         "hits": hits, "misses": misses, "fas": fas, "crs": crs})
    return pd.DataFrame(records)


# ── Permutation ANOVA ─────────────────────────────────────────────────────────

def permutation_anova(groups: list[np.ndarray], n_perm: int = 10_000, seed: int = 42) -> dict:
    """One-way permutation ANOVA. Fully vectorised — all permutations in one batch.

    Returns observed F, p-value, n_perm.
    """
    rng = np.random.default_rng(seed)
    all_vals = np.asarray(np.concatenate(groups), dtype=float)
    sizes = [len(g) for g in groups]
    n_total = len(all_vals)
    k = len(groups)
    df_between = k - 1
    df_within  = n_total - k

    def _f_batch(data: np.ndarray) -> np.ndarray:
        """data: (m, n_total) — returns F-stat for each of m rows."""
        grand = data.mean(axis=1, keepdims=True)          # (m, 1)
        ss_b = np.zeros(len(data))
        ss_w = np.zeros(len(data))
        col = 0
        for s in sizes:
            g = data[:, col:col + s]                       # (m, s)
            gm = g.mean(axis=1)                            # (m,)
            ss_b += s * (gm - grand[:, 0]) ** 2
            ss_w += ((g - gm[:, None]) ** 2).sum(axis=1)
            col += s
        denom = (ss_w / df_within) + 1e-15
        return (ss_b / df_between) / denom

    # Observed F (single row)
    observed_f = float(_f_batch(all_vals[None, :])[0])

    # All permutations in one shot: (n_perm, n_total)
    batch = np.tile(all_vals, (n_perm, 1))
    rng.permuted(batch, axis=1, out=batch)
    f_perm = _f_batch(batch)

    count = int((f_perm >= observed_f).sum())
    return {"F": observed_f, "p_perm": (count + 1) / (n_perm + 1), "n_perm": n_perm}



# ── Planned contrasts ─────────────────────────────────────────────────────────

def _bca_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 42,
             alpha: float = 0.05) -> tuple[float, float]:
    """BCa bootstrap 95 % CI on Cohen's d (a vs b). Fully vectorised."""
    rng = np.random.default_rng(seed)
    n1, n2 = len(a), len(b)

    def _pooled_d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """x: (..., n1), y: (..., n2) → Cohen's d along last axis."""
        s = np.sqrt(
            ((n1 - 1) * x.var(axis=-1, ddof=1) + (n2 - 1) * y.var(axis=-1, ddof=1))
            / (n1 + n2 - 2)
        )
        return (x.mean(axis=-1) - y.mean(axis=-1)) / (s + 1e-12)

    obs_d = float(_pooled_d(a, b))

    # Vectorised bootstrap: draw all samples at once
    idx_a = rng.integers(0, n1, size=(n_boot, n1))
    idx_b = rng.integers(0, n2, size=(n_boot, n2))
    boot_ds = _pooled_d(a[idx_a], b[idx_b])              # (n_boot,)

    # Bias-correction z0
    z0 = stats.norm.ppf(np.clip((boot_ds < obs_d).mean(), 1e-9, 1 - 1e-9))

    # Acceleration via jackknife (vectorised over combined sample)
    combined = np.concatenate([a, b])
    n_comb = len(combined)
    # Leave-one-out d values
    mask = np.ones((n_comb, n_comb), dtype=bool)
    np.fill_diagonal(mask, False)
    left_a = combined[np.where(mask)[1].reshape(n_comb, n_comb - 1)][:, :n1]
    left_b = combined[np.where(mask)[1].reshape(n_comb, n_comb - 1)][:, n1:]
    jack = _pooled_d(left_a, left_b)
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack) ** 3)
    den = 6 * (np.sum((jack_mean - jack) ** 2) ** 1.5) + 1e-12
    acc = num / den

    z_lo = stats.norm.ppf(alpha / 2)
    z_hi = stats.norm.ppf(1 - alpha / 2)
    p_lo = stats.norm.cdf(z0 + (z0 + z_lo) / (1 - acc * (z0 + z_lo)))
    p_hi = stats.norm.cdf(z0 + (z0 + z_hi) / (1 - acc * (z0 + z_hi)))
    p_lo = float(np.clip(p_lo, 0.001, 0.999))
    p_hi = float(np.clip(p_hi, 0.001, 0.999))
    return (float(np.quantile(boot_ds, p_lo)), float(np.quantile(boot_ds, p_hi)))


def planned_contrast(a: np.ndarray, b: np.ndarray, label: str = "") -> dict:
    """Welch t-test + Cohen's d + BCa CI on contrast a − b."""
    t, p = stats.ttest_ind(a, b, equal_var=False)
    n1, n2 = len(a), len(b)
    s_pool = np.sqrt(((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2) / (n1 + n2 - 2))
    d = float((a.mean() - b.mean()) / (s_pool + 1e-12))
    ci_lo, ci_hi = _bca_ci(a, b)
    return {
        "label": label, "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "t": float(t), "p_welch": float(p), "cohens_d": d,
        "bca_ci_95": [ci_lo, ci_hi], "n_a": n1, "n_b": n2,
    }


def holm_correct(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction. Returns adjusted p-values in original order."""
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.array(p_values, dtype=float)
    for rank, idx in enumerate(order):
        adjusted[idx] = min(1.0, p_values[idx] * (n - rank))
    # Enforce monotonicity
    for i in range(1, n):
        adjusted[order[i]] = max(adjusted[order[i]], adjusted[order[i - 1]])
    return adjusted.tolist()


# ── Bayes factor ──────────────────────────────────────────────────────────────

def bayes_factor_binom(k: int, n: int, p0: float = 0.5) -> float:
    """BF₀₁ for binomial test against p0 (Jeffreys prior Beta(0.5, 0.5)).
    BF₀₁ > 1 = evidence for null."""
    from scipy.special import betaln
    # Marginal likelihood under H1 (Beta-Binomial with Jeffrey's prior)
    log_ml_h1 = betaln(k + 0.5, n - k + 0.5) - betaln(0.5, 0.5)
    # Likelihood under H0
    log_ml_h0 = k * np.log(p0) + (n - k) * np.log(1 - p0)
    return float(np.exp(log_ml_h0 - log_ml_h1))


# ── Primary analysis ──────────────────────────────────────────────────────────

def run_primary(df: pd.DataFrame) -> dict:
    dp = per_subject_dprime(df)
    arms = sorted(dp["arm"].unique())
    groups = [dp[dp["arm"] == a]["dprime"].values for a in arms]

    anova = permutation_anova(groups)

    # Planned contrasts: B−A and B−C
    arm_map = {a: g for a, g in zip(arms, groups)}
    contrasts_raw = []
    if "B" in arm_map and "A" in arm_map:
        contrasts_raw.append(planned_contrast(arm_map["B"], arm_map["A"], "B-A"))
    if "B" in arm_map and "C" in arm_map:
        contrasts_raw.append(planned_contrast(arm_map["B"], arm_map["C"], "B-C"))

    # Holm correction
    p_vals = [c["p_welch"] for c in contrasts_raw]
    p_adj = holm_correct(p_vals)
    for c, pa in zip(contrasts_raw, p_adj):
        c["p_holm"] = pa

    return {"dprime_per_subject": dp.to_dict(orient="records"),
            "anova": anova, "contrasts": contrasts_raw,
            "arm_means": {a: float(g.mean()) for a, g in zip(arms, groups)},
            "arm_sds":   {a: float(g.std(ddof=1)) for a, g in zip(arms, groups)}}


# ── Secondary ─────────────────────────────────────────────────────────────────

def run_secondary(df: pd.DataFrame) -> dict:
    results = {}

    # Trust allocation
    trust = df[(df["block"] == "trust") & (~df["rt_invalid"].fillna(False))]
    if not trust.empty:
        trust_per_sub = trust.groupby(["participant_id", "arm"])["allocation"].mean().reset_index()
        arms = sorted(trust_per_sub["arm"].unique())
        groups = [trust_per_sub[trust_per_sub["arm"] == a]["allocation"].values for a in arms]
        results["trust_anova"] = permutation_anova(groups)
        if "B" in trust_per_sub["arm"].values and "A" in trust_per_sub["arm"].values:
            g = {a: trust_per_sub[trust_per_sub["arm"] == a]["allocation"].values for a in arms}
            results["trust_contrast_BA"] = planned_contrast(g.get("B", np.array([])),
                                                             g.get("A", np.array([])), "trust B-A")

    # Log RT
    fallacy = df[(df["block"] == "fallacy") & (~df["rt_invalid"].fillna(False)) & (df["rt_ms"].notna())]
    if not fallacy.empty:
        log_rt = fallacy.groupby(["participant_id", "arm"])["rt_ms"].median().reset_index()
        log_rt["log_rt"] = np.log(log_rt["rt_ms"] + 1)
        arms = sorted(log_rt["arm"].unique())
        groups = [log_rt[log_rt["arm"] == a]["log_rt"].values for a in arms]
        results["logrt_anova"] = permutation_anova(groups)

    return results


# ── H3: sub-perceptual check ─────────────────────────────────────────────────

def run_h3(df: pd.DataFrame, manip_check_col: str = "manip_2afc") -> dict:
    """BF₀₁ on 2AFC + restricted B−A contrast."""
    # This requires manipulation check data stored in a separate event or response row
    # Placeholder: return structure, populated when manip check data is in DB
    return {"note": "Requires manipulation-check responses joined from event table",
            "bf01_arm_b": None, "restricted_contrast": None}


# ── Report ────────────────────────────────────────────────────────────────────

def save_report(primary: dict, secondary: dict, h3: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    # Write JSON
    all_results = {"primary": primary, "secondary": secondary, "h3": h3}
    (out_dir / "results.json").write_text(json.dumps(all_results, indent=2, default=str))

    # d′ bar chart
    try:
        import matplotlib.pyplot as plt
        dp = pd.DataFrame(primary["dprime_per_subject"])
        if not dp.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            order = sorted(dp["arm"].unique())
            means = [dp[dp["arm"] == a]["dprime"].mean() for a in order]
            sems  = [dp[dp["arm"] == a]["dprime"].sem()  for a in order]
            ax.bar(order, means, yerr=sems, capsize=5, color=["#4c72b0", "#dd8452", "#55a868"])
            ax.set_ylabel("d′  (mean ± SEM)")
            ax.set_title("Fallacy detection d′ by arm")
            ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
            fig.tight_layout()
            fig.savefig(out_dir / "figures" / "dprime_by_arm.png", dpi=150)
            plt.close(fig)
    except Exception as e:
        print(f"[warn] Could not plot: {e}")

    # Simple HTML report
    contrasts_html = ""
    for c in primary.get("contrasts", []):
        sig = "✓" if c["p_holm"] < 0.05 else "✗"
        contrasts_html += (
            f"<tr><td>{c['label']}</td><td>{c['cohens_d']:.3f}</td>"
            f"<td>[{c['bca_ci_95'][0]:.3f}, {c['bca_ci_95'][1]:.3f}]</td>"
            f"<td>{c['p_welch']:.4f}</td><td>{c['p_holm']:.4f}</td><td>{sig}</td></tr>"
        )
    anova = primary.get("anova", {})
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>SAP Study Results</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:2em auto}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:6px 10px}}
th{{background:#f0f0f0}}.sig{{color:green}}.ns{{color:#999}}</style></head><body>
<h1>Sub-Perceptual Acoustic Persuasion — Results</h1>
<h2>Primary: Fallacy detection d′</h2>
<p>Permutation ANOVA: F = {anova.get('F', 'n/a'):.3f}, p = {anova.get('p_perm', 'n/a'):.4f} ({anova.get('n_perm', '?')} permutations)</p>
<table><tr><th>Contrast</th><th>Cohen's d</th><th>BCa 95% CI</th>
<th>p (Welch)</th><th>p (Holm)</th><th>Sig.</th></tr>
{contrasts_html}
</table>
<h2>Arm means (d′)</h2>
<ul>{''.join(f"<li>{a}: M={v:.3f}, SD={primary['arm_sds'].get(a, 0):.3f}</li>" for a, v in primary.get('arm_means', {}).items())}</ul>
<img src="figures/dprime_by_arm.png" alt="d-prime by arm" style="max-width:500px">
<h2>H3 — Sub-perceptual claim</h2>
<pre>{json.dumps(h3, indent=2)}</pre>
</body></html>"""
    (out_dir / "report.html").write_text(html)
    print(f"Report written to {out_dir / 'report.html'}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-registered analysis pipeline")
    ap.add_argument("--data", type=Path, default=Path("study.db"))
    ap.add_argument("--key",  type=Path, default=None,
                    help="Path to unblind_key.json. Pass ONLY after analysis code is locked.")
    ap.add_argument("--out",  type=Path, default=Path("results"))
    args = ap.parse_args()

    if not args.data.exists():
        print(f"Database not found: {args.data}")
        return

    df = load_data(args.data, args.key)
    df = apply_exclusions(df)
    print(f"Loaded {len(df['participant_id'].unique())} participants after exclusions")

    primary   = run_primary(df)
    secondary = run_secondary(df)
    h3        = run_h3(df)
    save_report(primary, secondary, h3, args.out)


if __name__ == "__main__":
    main()
