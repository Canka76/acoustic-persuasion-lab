"""DSSS watermarking: 127-chip PN sequence, BPSK, carrier 7–9 kHz, −50 dBFS."""
from __future__ import annotations

import struct
import uuid

import numpy as np

PAYLOAD_HEADER = b"SPAP-LAB"
_CARRIER_LOW_HZ = 7000.0
_CARRIER_HIGH_HZ = 9000.0
_LEVEL_DBFS = -50.0
_CHIPS = 127  # PN sequence length (maximal-length 7-bit LFSR)


def _pn_sequence(n_chips: int = _CHIPS) -> np.ndarray:
    """Generate a maximal-length binary PN sequence (±1) of length n_chips."""
    # 7-bit LFSR with feedback polynomial x^7 + x^6 + 1
    state = 0b1111111
    seq = []
    for _ in range(n_chips):
        bit = ((state >> 6) ^ (state >> 5)) & 1
        seq.append(1 if (state & 1) else -1)
        state = ((state >> 1) | (bit << 6)) & 0x7F
    return np.array(seq, dtype=np.float32)


def _build_payload(stimulus_uuid: uuid.UUID) -> bytes:
    return PAYLOAD_HEADER + stimulus_uuid.bytes  # 8 + 16 = 24 bytes


def embed(
    wav: np.ndarray,
    sr: int,
    stimulus_uuid: uuid.UUID | None = None,
) -> tuple[np.ndarray, uuid.UUID]:
    """Embed DSSS watermark into *wav*. Returns (watermarked_wav, stimulus_uuid)."""
    if stimulus_uuid is None:
        stimulus_uuid = uuid.uuid4()

    payload = _build_payload(stimulus_uuid)
    pn = _pn_sequence()
    carrier_hz = (_CARRIER_LOW_HZ + _CARRIER_HIGH_HZ) / 2.0
    level_linear = 10.0 ** (_LEVEL_DBFS / 20.0)

    # Repeat payload bits as BPSK chips across the entire signal
    payload_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    bpsk = np.where(payload_bits, 1.0, -1.0).astype(np.float32)

    n = len(wav)
    chip_duration = max(1, n // (len(bpsk) * _CHIPS))
    t = np.arange(n, dtype=np.float64) / sr
    carrier = np.cos(2 * np.pi * carrier_hz * t).astype(np.float32)

    # Spread: tile (bpsk ⊗ pn) repeated to fill n samples
    spread = np.tile(np.outer(bpsk, pn).ravel(), n).ravel()[:n].astype(np.float32)
    wm_signal = spread * carrier * level_linear

    return (wav + wm_signal).astype(np.float32), stimulus_uuid


def decode(wav: np.ndarray, sr: int) -> bytes | None:
    """Attempt to decode the DSSS watermark payload. Returns bytes or None."""
    # Simplified correlator — despreads and demodulates
    pn = _pn_sequence()
    carrier_hz = (_CARRIER_LOW_HZ + _CARRIER_HIGH_HZ) / 2.0
    n = len(wav)
    t = np.arange(n, dtype=np.float64) / sr
    carrier = np.cos(2 * np.pi * carrier_hz * t).astype(np.float32)

    demod = wav * carrier
    payload_len_bytes = len(PAYLOAD_HEADER) + 16
    n_bits = payload_len_bytes * 8
    chip_block = _CHIPS
    total_chips_needed = n_bits * chip_block

    if n < total_chips_needed:
        return None

    demod_trimmed = demod[:total_chips_needed]
    bits = []
    for i in range(n_bits):
        chunk = demod_trimmed[i * chip_block : (i + 1) * chip_block]
        corr = float(np.dot(chunk[:len(pn)], pn))
        bits.append(1 if corr > 0 else 0)

    bit_array = np.array(bits, dtype=np.uint8)
    payload = np.packbits(bit_array).tobytes()
    if payload[:8] != PAYLOAD_HEADER:
        return None
    return payload
