"""DSSS (Direct-Sequence Spread-Spectrum) watermark.

Payload: b"SPAP-LAB" + 8-byte stimulus UUID (16 bytes = 128 bits).
Carrier: 7 500 Hz single-frequency. BPSK with 127-chip m-sequence. Level: −50 dBFS.

Phase-coherence constraint: the carrier must complete an integer number of
cycles per rep_len = 128 × 127 = 16 256 samples so coherent averaging across
repetitions amplifies rather than cancels the DSSS signal.

    cycles_per_rep = rep_len × fc / sr  must be integer
    = 16 256 × fc / 48 000
    7 500 Hz → 16256 × 7500 / 48000 = 2540  ✓  (integer)
    8 000 Hz → 16256 × 8000 / 48000 = 2709.333  ✗  (caused cancellation)
"""
from __future__ import annotations

import uuid

import numpy as np

PAYLOAD_PREFIX = b"SPAP-LAB"
PN_LENGTH = 127
CARRIER_LOW_HZ  = 7500.0  # fc = (low+high)/2 = 7500 — phase-coherent at 48 kHz
CARRIER_HIGH_HZ = 7500.0  # kept equal; band label is now a point frequency
LEVEL_DBFS = -50.0


def _m_sequence(length: int = PN_LENGTH, seed: int = 0b111_1111) -> np.ndarray:
    """Generate a ±1 m-sequence of given length using a degree-7 LFSR (taps 7,6)."""
    reg = seed & 0x7F
    seq: list[int] = []
    for _ in range(length):
        bit = ((reg >> 6) ^ (reg >> 5)) & 1
        seq.append(1 if bit else -1)
        reg = ((reg << 1) | bit) & 0x7F
    return np.array(seq, dtype=np.float32)


_PN = _m_sequence()


def _bytes_to_bpsk(payload: bytes) -> np.ndarray:
    bits = []
    for byte in payload:
        for i in range(7, -1, -1):
            bits.append(1 if (byte >> i) & 1 else -1)
    return np.array(bits, dtype=np.float32)


def embed(wav: np.ndarray, sr: int, stimulus_id: str | None = None) -> tuple[np.ndarray, str]:
    """Embed watermark. Returns (watermarked_wav, stimulus_id)."""
    if stimulus_id is None:
        stimulus_id = str(uuid.uuid4())

    uid_bytes = uuid.UUID(stimulus_id).bytes[:8]
    payload = PAYLOAD_PREFIX + uid_bytes  # 16 bytes = 128 bits
    bpsk = _bytes_to_bpsk(payload)        # 128 symbols

    # Spread: each symbol × full PN chip sequence
    spread = np.repeat(bpsk, PN_LENGTH) * np.tile(_PN, len(bpsk))

    # Modulate onto carrier
    fc = (CARRIER_LOW_HZ + CARRIER_HIGH_HZ) / 2.0
    t = np.arange(len(spread)) / sr
    carrier = np.cos(2.0 * np.pi * fc * t).astype(np.float32)
    wm = (spread * carrier).astype(np.float32)

    # Tile to cover full wav duration
    if len(wm) < len(wav):
        reps = int(np.ceil(len(wav) / len(wm)))
        wm = np.tile(wm, reps)
    wm = wm[:len(wav)]

    # Scale to target level
    wm_rms = float(np.sqrt(np.mean(wm ** 2) + 1e-12))
    target_rms = 10.0 ** (LEVEL_DBFS / 20.0)
    wm *= target_rms / wm_rms

    return (wav + wm).astype(np.float32), stimulus_id


def decode(wav: np.ndarray, sr: int) -> bytes | None:
    """Attempt to decode the DSSS watermark. Returns payload bytes or None.

    Coherently averages all complete repetitions of the watermark across the
    file before correlating — necessary because the watermark sits at −50 dBFS
    and a single-repetition SNR (−27 dB before DSSS gain) is too low for
    reliable per-repetition decoding.

    After coherent averaging, a narrow bandpass around fc is applied before
    demodulation to remove systematic interference from speech harmonics that
    would otherwise create a coherent (non-averaging-out) bias on individual
    bit correlations.  The carrier at 7 500 Hz has fc±1 500 Hz pass-band.
    """
    from scipy.signal import butter, sosfiltfilt

    n_bits = 128
    chips_per_bit = PN_LENGTH
    rep_len = n_bits * chips_per_bit  # samples per one watermark cycle (16 256 @ 48 kHz)

    if len(wav) < rep_len:
        return None

    # Coherent average of all complete repetitions
    n_reps = len(wav) // rep_len
    stacked = np.zeros(rep_len, dtype=np.float64)
    for i in range(n_reps):
        stacked += wav[i * rep_len:(i + 1) * rep_len].astype(np.float64)
    stacked /= n_reps  # averaging gives +10*log10(n_reps) dB SNR gain

    # Bandpass around carrier to suppress speech harmonics that survive averaging
    fc = (CARRIER_LOW_HZ + CARRIER_HIGH_HZ) / 2.0
    bw = max(1500.0, fc * 0.2)  # ±bw Hz around fc
    low = max(fc - bw, 100.0)
    high = min(fc + bw, sr / 2.0 - 100.0)
    sos = butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
    stacked = sosfiltfilt(sos, stacked)

    t = np.arange(rep_len) / sr
    carrier = np.cos(2.0 * np.pi * fc * t)
    demod = stacked * carrier

    bits: list[int] = []
    for i in range(n_bits):
        chunk = demod[i * chips_per_bit:(i + 1) * chips_per_bit]
        corr = float(np.dot(chunk, _PN.astype(np.float64)))
        bits.append(1 if corr > 0 else 0)

    payload = bytes(
        sum(bits[j * 8 + k] << (7 - k) for k in range(8))
        for j in range(len(bits) // 8)
    )
    return payload if payload[:8] == PAYLOAD_PREFIX else None
