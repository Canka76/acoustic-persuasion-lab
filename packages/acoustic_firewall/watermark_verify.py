"""DSSS watermark decoder for acoustic_firewall.

This module mirrors only the DECODE side of sap_stimulus.watermark so that
acoustic_firewall never imports sap_stimulus (dependency boundary enforced by
tests/test_dependency_boundary.py).

The embed side stays exclusively in sap_stimulus.watermark.
"""
from __future__ import annotations

import numpy as np

PAYLOAD_HEADER = b"SPAP-LAB"
_CARRIER_HZ = 8000.0   # midpoint of 7–9 kHz carrier band
_CHIPS = 127


def _pn_sequence(n_chips: int = _CHIPS) -> np.ndarray:
    """Maximal-length 7-bit LFSR sequence (±1), identical to sap_stimulus.watermark."""
    state = 0b1111111
    seq = []
    for _ in range(n_chips):
        bit = ((state >> 6) ^ (state >> 5)) & 1
        seq.append(1 if (state & 1) else -1)
        state = ((state >> 1) | (bit << 6)) & 0x7F
    return np.array(seq, dtype=np.float32)


def decode(wav: np.ndarray, sr: int) -> bytes | None:
    """Attempt to decode the DSSS watermark payload. Returns bytes or None."""
    pn = _pn_sequence()
    n = len(wav)
    t = np.arange(n, dtype=np.float64) / sr
    carrier = np.cos(2 * np.pi * _CARRIER_HZ * t).astype(np.float32)

    demod = wav * carrier
    payload_len_bytes = len(PAYLOAD_HEADER) + 16  # header + UUID
    n_bits = payload_len_bytes * 8
    chip_block = _CHIPS
    total_chips_needed = n_bits * chip_block

    if n < total_chips_needed:
        return None

    demod_trimmed = demod[:total_chips_needed]
    bits = []
    for i in range(n_bits):
        chunk = demod_trimmed[i * chip_block : (i + 1) * chip_block]
        corr = float(np.dot(chunk[: len(pn)], pn))
        bits.append(1 if corr > 0 else 0)

    bit_array = np.array(bits, dtype=np.uint8)
    payload = np.packbits(bit_array).tobytes()
    if payload[:8] != PAYLOAD_HEADER:
        return None
    return payload
