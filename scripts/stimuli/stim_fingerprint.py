#!/usr/bin/env python3
"""Perceptual-hash fingerprints for verifying regenerated Hyperface stimuli.

Shared by ``build_fingerprints.py`` (authors, needs the original stimuli) and
``verify_stimuli.py`` (anyone, needs only their regenerated clips).

Why a perceptual hash?
----------------------
The original stimulus clips cannot be redistributed, so a public user has nothing
to diff their regenerated clips against. Instead we publish, per clip, a compact
sequence of **perceptual hashes** (one 64-bit pHash per frame) computed from the
original stored clip -- a one-way fingerprint that cannot reconstruct the footage.

To check a regenerated clip you recompute the same per-frame hashes and score

    similarity = median_i [ max_{|j-i| <= WIN} (1 - Hamming(ref_i, regen_j) / 64) ]

This mirrors the ``robust_ncc`` metric used during recovery: the per-frame pHash is
invariant to the brightness/contrast/desaturation drift of re-encoded YouTube
sources, and the small temporal window absorbs a 25<->30 fps frame-phase mismatch.

Calibration (measured on all 674 regenerable clips): a correctly regenerated clip
scores >= 0.93 (median 1.00); the same clip compared against a *different* clip's
fingerprint tops out at 0.72. The verdict thresholds below sit in that gap.

Only ``ffmpeg`` (for decoding), ``numpy`` and ``scipy`` are required -- the same
tools ``generate_stimuli.py`` already needs, plus SciPy's DCT.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from scipy.fft import dct

# pHash parameters (do NOT change without rebuilding the published fingerprints).
HASH_SIDE = 32      # decode each frame to HASH_SIDE x HASH_SIDE grayscale
DCT_KEEP = 8        # keep top-left DCT_KEEP x DCT_KEEP low-frequency block -> 64 bits
HASH_BITS = DCT_KEEP * DCT_KEEP
HEX_LEN = HASH_BITS // 4    # 16 hex chars per 64-bit hash
WIN = 6             # temporal search half-window (frames), absorbs fps-phase drift

# Verdict thresholds on the similarity score (see calibration above).
PASS_MIN = 0.85     # >= PASS_MIN: reproduces the reference content
WARN_MIN = 0.70     # [WARN_MIN, PASS_MIN): ambiguous, inspect manually
#                     < WARN_MIN: does not match the reference clip


def _dctn(a: np.ndarray) -> np.ndarray:
    return dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")


def clip_phashes(path: str | Path) -> list[int]:
    """Return one 64-bit perceptual hash per frame of the video at ``path``.

    Frames are decoded straight from ffmpeg as HASH_SIDE x HASH_SIDE grayscale; each
    frame's low-frequency DCT block is thresholded at the median of its AC terms to
    give a 64-bit hash (the DC bit is zeroed, so 63 bits carry information). Returns
    [] if the clip cannot be decoded.
    """
    cmd = ["ffmpeg", "-loglevel", "error", "-i", str(path),
           "-vf", f"scale={HASH_SIDE}:{HASH_SIDE},format=gray",
           "-f", "rawvideo", "-"]
    buf = np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, dtype=np.uint8)
    px = HASH_SIDE * HASH_SIDE
    n = buf.size // px
    if n == 0:
        return []
    frames = buf[: n * px].reshape(n, HASH_SIDE, HASH_SIDE).astype(np.float32)
    hashes = []
    for f in frames:
        block = _dctn(f)[:DCT_KEEP, :DCT_KEEP].flatten()
        med = np.median(block[1:])          # median of the AC terms only
        bits = block > med
        # The DC term (block[0]) encodes overall brightness, not structure, and is
        # almost always above the AC median -- as a hash bit it is near-constant and,
        # being the largest coefficient, the most sensitive to cross-machine rounding.
        # Zero it deterministically so all 63 informative bits come from the AC terms.
        bits[0] = False
        h = 0
        for b in bits:
            h = (h << 1) | int(b)
        hashes.append(h)
    return hashes


def similarity(ref: list[int], test: list[int], win: int = WIN) -> float:
    """Temporal-windowed pHash similarity of ``test`` against reference ``ref``.

    For each reference frame, take the best (max) bit-agreement over a +-``win``
    window of test frames (mapped by relative position), then the median over
    reference frames. Returns a value in [0, 1], or -1.0 if either sequence is empty.
    """
    if not ref or not test:
        return -1.0
    n_test = len(test)
    vals = []
    for i, rh in enumerate(ref):
        c = int(i * n_test / len(ref))
        j0, j1 = max(0, c - win), min(n_test, c + win + 1)
        best = max(HASH_BITS - bin(rh ^ test[j]).count("1") for j in range(j0, j1))
        vals.append(best / HASH_BITS)
    return float(np.median(vals))


def verdict(sim: float) -> str:
    if sim < 0:
        return "NO_DATA"
    if sim >= PASS_MIN:
        return "PASS"
    if sim >= WARN_MIN:
        return "WARN"
    return "FAIL"


def encode(hashes: list[int]) -> str:
    """Comma-separated fixed-width hex, one token per frame hash."""
    return ",".join(format(h, f"0{HEX_LEN}x") for h in hashes)


def decode(field: str) -> list[int]:
    return [int(tok, 16) for tok in field.strip().split(",") if tok]


def load_fingerprints(path: str | Path) -> dict[str, list[int]]:
    """Read a ``stimulus_fingerprints.tsv`` -> {stimulus: [frame hashes]}."""
    import csv
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            out[row["stimulus"]] = decode(row["frame_phash"])
    return out
