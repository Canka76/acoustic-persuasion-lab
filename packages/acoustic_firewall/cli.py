"""CLI entry point for acoustic-firewall.

Commands:
  acoustic-firewall scan <file>            — print JSON detection report
  acoustic-firewall clean <in> <out>       — neutralize and write cleaned file
  acoustic-firewall verify-watermark <file> — decode and print DSSS watermark payload
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _load_wav(path: Path):
    from sap_core.audio import load
    return load(path)


def cmd_scan(args: list[str]) -> int:
    if not args:
        print("Usage: acoustic-firewall scan <file>", file=sys.stderr)
        return 1
    path = Path(args[0])
    wav, sr = _load_wav(path)
    from acoustic_firewall.features import extract
    features = extract(wav, sr)
    try:
        from acoustic_firewall.model import load_model, predict
        model = load_model()
        report = predict(features, model)
        print(json.dumps({
            "verdict": report.verdict,
            "score": round(report.score, 4),
            "f_mod_hz": report.f_mod_hz,
            "threshold_used": report.threshold_used,
        }, indent=2))
    except FileNotFoundError:
        print(json.dumps({"error": "Model not trained yet — run: uv run python -m acoustic_firewall.train"}))
        return 1
    return 0


def cmd_clean(args: list[str]) -> int:
    if len(args) < 2:
        print("Usage: acoustic-firewall clean <in> <out>", file=sys.stderr)
        return 1
    in_path, out_path = Path(args[0]), Path(args[1])
    wav, sr = _load_wav(in_path)
    from acoustic_firewall.features import extract
    from acoustic_firewall.model import load_model, predict
    from acoustic_firewall.neutralize import neutralize
    model = load_model()
    features = extract(wav, sr)
    report = predict(features, model)
    cleaned = neutralize(wav, sr, report)
    from sap_core.audio import save
    save(out_path, cleaned, sr)
    print(f"Cleaned → {out_path}")
    return 0


def cmd_verify_watermark(args: list[str]) -> int:
    if not args:
        print("Usage: acoustic-firewall verify-watermark <file>", file=sys.stderr)
        return 1
    path = Path(args[0])
    wav, sr = _load_wav(path)
    from acoustic_firewall.watermark_verify import PAYLOAD_HEADER, decode
    payload = decode(wav, sr)
    if payload is None:
        print(json.dumps({"status": "not_found"}))
        return 1
    import uuid
    stimulus_uuid = uuid.UUID(bytes=payload[8:24])
    print(json.dumps({
        "status": "ok",
        "header": payload[:8].decode("ascii", errors="replace"),
        "stimulus_uuid": str(stimulus_uuid),
    }))
    return 0


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        sys.exit(0)
    cmd, *rest = argv
    dispatch = {
        "scan": cmd_scan,
        "clean": cmd_clean,
        "verify-watermark": cmd_verify_watermark,
    }
    if cmd not in dispatch:
        print(f"Unknown command: {cmd}\n{__doc__}", file=sys.stderr)
        sys.exit(1)
    sys.exit(dispatch[cmd](rest))
