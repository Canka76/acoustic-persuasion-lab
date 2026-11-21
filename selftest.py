"""
selftest.py — blinded self-test of the persuasion manipulation.

Plays each argument stimulus from a randomly-assigned arm (A or B),
asks you to rate it, then reveals which arm it was and shows your
accuracy split by arm.

This is a single-session approximation of the full RCT — statistically
underpowered (8 items, ~4 per arm), but enough to feel the difference.

Usage
-----
    uv run python selftest.py              # arms A vs B (default)
    uv run python selftest.py --arms A C   # compare A vs placebo
    uv run python selftest.py --arms B C   # compare treatment vs placebo
    uv run python selftest.py --no-shuffle # keep corpus order (less blind)

Requirements
------------
    corpus/stimuli/{A,B,C}/ must exist.
    Run sap_stimulus.render first if they don't:
        uv run python -m sap_stimulus.render \\
            --corpus corpus/scripts/corpus_v1.json --arms A B C --out corpus/stimuli/
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).parent
_CORPUS_PATH = _ROOT / "corpus" / "scripts" / "corpus_v1.json"
_STIMULI_ROOT = _ROOT / "corpus" / "stimuli"

# Ground-truth: True = sound argument, False = contains a fallacy
_GROUND_TRUTH: dict[str, bool | None] = {}

# Fallacy type labels for the reveal
_FALLACY_LABELS: dict[str, str] = {
    "false_dilemma":             "False dilemma (only two options presented)",
    "hasty_generalization":      "Hasty generalisation (too few cases)",
    "affirming_the_consequent":  "Affirming the consequent (invalid deductive form)",
    "circular_justification":    "Circular reasoning (conclusion restates premise)",
}


def _load_corpus() -> list[dict]:
    with _CORPUS_PATH.open() as f:
        data = json.load(f)
    items = [it for it in data["items"] if it["block"] in ("fallacy", "practice")]
    for it in items:
        _GROUND_TRUTH[it["id"]] = it["sound"]  # True / False / None
    return items


def _play(path: Path) -> None:
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(str(path.resolve()))
            # Give the media player a moment to open before the prompt appears
            time.sleep(1.5)
        elif system == "Darwin":
            subprocess.Popen(["afplay", str(path)])
        else:
            subprocess.Popen(["aplay", "-q", str(path)])
    except (OSError, FileNotFoundError) as e:
        print(f"  [play failed: {e}]")
        print(f"  Open manually: {path.resolve()}")


@dataclass
class Trial:
    item_id: str
    arm: str
    text: str
    ground_truth: bool | None   # True=sound, False=fallacy, None=trust
    fallacy_type: str | None
    response: str = ""          # "valid" or "flaw"
    correct: bool | None = None


def _run(arms: list[str], shuffle: bool) -> None:
    items = _load_corpus()

    # --- build trial list: each item assigned to one arm randomly ---
    rng = random.Random()   # seeded from system entropy — genuinely random each run
    trials: list[Trial] = []
    for item in items:
        arm = rng.choice(arms)
        wav_path = _STIMULI_ROOT / arm / f"{item['id']}.wav"
        if not wav_path.exists():
            print(f"  [skip] {arm}/{item['id']}.wav not found — render stimuli first")
            continue
        trials.append(Trial(
            item_id=item["id"],
            arm=arm,
            text=item["text"],
            ground_truth=item["sound"],
            fallacy_type=item.get("fallacy_type"),
        ))

    if not trials:
        print("No stimuli found. Run sap_stimulus.render first.")
        sys.exit(1)

    if shuffle:
        rng.shuffle(trials)

    n = len(trials)
    arms_str = " vs ".join(arms)

    print()
    print("┌─────────────────────────────────────────────────────────────┐")
    print(f"│  Persuasion self-test  ({arms_str})                        ")
    print(f"│  {n} arguments  •  arm assignment is hidden until the end  ")
    print("│                                                             │")
    print("│  For each argument:                                         │")
    print("│    Listen fully, then rate:                                 │")
    print("│      v  — argument seems Valid / sound                      │")
    print("│      f  — argument contains a Flaw / fallacy                │")
    print("│      ?  — unsure (counts as incorrect for scoring)          │")
    print("│      r  — replay the audio                                  │")
    print("└─────────────────────────────────────────────────────────────┘")
    print()
    input("  Press Enter when you have headphones on and volume set …")
    print()

    for i, trial in enumerate(trials, 1):
        wav_path = _STIMULI_ROOT / trial.arm / f"{trial.item_id}.wav"

        print(f"  ─── Argument {i}/{n} ──────────────────────────────────────")
        print()
        _play(wav_path)

        while True:
            raw = input("  Your rating  [v]alid / [f]law / [?]unsure / [r]eplay: ").strip().lower()
            if raw == "r":
                _play(wav_path)
                continue
            if raw in ("v", "f", "?"):
                break
            print("  Enter v, f, ?, or r.")

        trial.response = {"v": "valid", "f": "flaw", "?": "unsure"}[raw]
        if trial.ground_truth is None:
            trial.correct = None   # trust items — not scored
        elif raw == "?":
            trial.correct = False
        else:
            truth_response = "valid" if trial.ground_truth else "flaw"
            trial.correct = trial.response == truth_response

        print()

    # --- results ---
    print()
    print("══════════════════════════════════════════════════════════════")
    print("  RESULTS — arm revealed")
    print("══════════════════════════════════════════════════════════════")
    print()

    scoreable = [t for t in trials if t.ground_truth is not None]

    for t in trials:
        gt_label = "sound" if t.ground_truth else ("FALLACY" if t.ground_truth is False else "trust")
        correct_marker = "✓" if t.correct else ("✗" if t.correct is False else "—")
        fallacy_note = ""
        if not t.ground_truth and t.fallacy_type:
            fallacy_note = f"\n       └─ {_FALLACY_LABELS.get(t.fallacy_type, t.fallacy_type)}"
        print(f"  {correct_marker}  [{t.arm}]  {t.item_id:8s}  truth={gt_label:7s}  "
              f"you said={t.response}{fallacy_note}")

    print()
    print("── Accuracy by arm ──────────────────────────────────────────")
    print()

    arm_stats: dict[str, dict] = {a: {"correct": 0, "total": 0, "hits": 0, "fas": 0,
                                       "misses": 0, "crs": 0} for a in arms}
    for t in scoreable:
        s = arm_stats[t.arm]
        s["total"] += 1
        if t.correct:
            s["correct"] += 1
        # SDT counts
        is_flaw = not t.ground_truth
        said_flaw = t.response == "flaw"
        if is_flaw and said_flaw:
            s["hits"] += 1
        elif is_flaw and not said_flaw:
            s["misses"] += 1
        elif not is_flaw and said_flaw:
            s["fas"] += 1
        elif not is_flaw and not said_flaw:
            s["crs"] += 1

    for arm, s in arm_stats.items():
        if s["total"] == 0:
            continue
        pct = 100 * s["correct"] / s["total"]
        n_flaw = s["hits"] + s["misses"]
        n_valid = s["fas"] + s["crs"]
        hr  = s["hits"]  / n_flaw  if n_flaw  > 0 else float("nan")
        far = s["fas"]   / n_valid if n_valid > 0 else float("nan")
        print(f"  Arm {arm}:  {s['correct']}/{s['total']} correct  ({pct:.0f} %)")
        print(f"         hit rate = {s['hits']}/{n_flaw}   "
              f"false-alarm rate = {s['fas']}/{n_valid}")
        if not any(map(__import__("math").isnan, [hr, far])):
            import math
            # log-linear d-prime correction (avoids 0/1 rates)
            hr_c  = (s["hits"]  + 0.5) / (n_flaw  + 1)
            far_c = (s["fas"]   + 0.5) / (n_valid + 1)
            from scipy.stats import norm
            dprime = norm.ppf(hr_c) - norm.ppf(far_c)
            print(f"         d′ = {dprime:.2f}  "
                  f"(log-linear corrected; ≥8 items/arm needed for reliability)")
        print()

    print("── Interpretation ───────────────────────────────────────────")
    print()
    if len(arms) == 2 and all(arm_stats[a]["total"] > 0 for a in arms):
        a0, a1 = arms
        d0 = arm_stats[a0]
        d1 = arm_stats[a1]
        if d0["total"] > 0 and d1["total"] > 0:
            acc0 = d0["correct"] / d0["total"]
            acc1 = d1["correct"] / d1["total"]
            diff = acc0 - acc1
            if abs(diff) < 0.1:
                print(f"  No clear accuracy difference between arm {a0} and arm {a1}.")
                print("  With only ~4 items per arm this is expected — the RCT needs")
                print("  40–60 participants per arm for 80 % power.")
            elif acc1 < acc0:
                print(f"  You were less accurate on arm {a1} arguments ({acc1:.0%}) than")
                print(f"  arm {a0} ({acc0:.0%}). This is the direction the manipulation predicts.")
                print("  Single-session, single-participant — treat as anecdotal.")
            else:
                print(f"  You were more accurate on arm {a1} ({acc1:.0%}) than arm {a0} ({acc0:.0%}).")
                print("  Opposite of the predicted direction — or noise with n≈4 items.")
    print()
    print("  To replay any argument:  open corpus/stimuli/<arm>/<id>.wav")
    print("══════════════════════════════════════════════════════════════")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs=2, default=["A", "B"],
                   choices=["A", "B", "C", "a", "b", "c"],
                   metavar="ARM",
                   help="Two arms to compare (default: A B)")
    p.add_argument("--no-shuffle", dest="shuffle", action="store_false",
                   help="Keep corpus order instead of randomising")
    args = p.parse_args()
    arms = [a.upper() for a in args.arms]
    if arms[0] == arms[1]:
        print("Error: pick two different arms.")
        sys.exit(1)
    _run(arms, args.shuffle)


if __name__ == "__main__":
    main()
