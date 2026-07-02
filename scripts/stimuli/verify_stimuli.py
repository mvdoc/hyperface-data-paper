#!/usr/bin/env python3
"""Verify regenerated Hyperface stimuli against the published fingerprints.

After running ``generate_stimuli.py`` you can check that each clip you re-cut is
the RIGHT content -- same footage, timing and crop as the original stimulus -- with
no access to the (non-redistributable) originals. This recomputes the per-frame
perceptual hashes of each regenerated clip and compares them to the published
``stimulus_fingerprints.tsv`` using the same temporal-windowed, photometric-robust
metric that was used to recover the clips (see ``stim_fingerprint.py``).

    python verify_stimuli.py                          # verify ./regenerated_stimuli
    python verify_stimuli.py --regen-dir /tmp/stimuli
    python verify_stimuli.py --only face023.mp4 face300.mp4
    python verify_stimuli.py --out verification.tsv   # also write a per-clip TSV

Verdicts (similarity score in [0, 1]):
    PASS        >= 0.85   reproduces the reference clip (correct clips score >= 0.93)
    WARN        0.70-0.85 ambiguous -- inspect (e.g. a slightly wrong crop or start)
    FAIL        < 0.70    does not match the reference, or the file cannot be decoded
    MISSING     source still downloadable, but you have not produced this clip
    UNAVAILABLE source video was deleted from YouTube; the clip cannot be regenerated
    SKIP        clip is flagged non-regenerable in the manifest (no reference)

Exits non-zero if any clip is WARN or FAIL, or if nothing was verified at all.
Requires ``ffmpeg`` on PATH plus ``numpy`` and ``scipy``.
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

from stim_fingerprint import (
    PASS_MIN,
    WARN_MIN,
    clip_phashes,
    load_fingerprints,
    similarity,
    verdict,
)

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "stimulus_sources.tsv"
DEFAULT_FINGERPRINTS = HERE / "stimulus_fingerprints.tsv"
DEFAULT_REGEN = HERE / "regenerated_stimuli"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regen-dir", type=Path, default=DEFAULT_REGEN,
                    help="regenerated clips dir (default ./regenerated_stimuli)")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--fingerprints", type=Path, default=DEFAULT_FINGERPRINTS)
    ap.add_argument("--only", nargs="+", help="verify only these stimulus filenames")
    ap.add_argument("--out", type=Path, help="write a per-clip TSV report here")
    ap.add_argument("--quiet", action="store_true", help="only print the summary")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("error: 'ffmpeg' not found on PATH", file=sys.stderr)
        return 2
    if not args.manifest.exists():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    if not args.fingerprints.exists():
        print(f"error: fingerprints file not found: {args.fingerprints}\n"
              "(authors build it with build_fingerprints.py; ships with the manifest)",
              file=sys.stderr)
        return 2

    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    fps = load_fingerprints(args.fingerprints)
    if args.only:
        want = set(args.only)
        rows = [r for r in rows if r["stimulus"] in want]

    report = []
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0,
              "MISSING": 0, "UNAVAILABLE": 0, "SKIP": 0}
    for r in rows:
        stim = r["stimulus"]
        regenerable = (r.get("regenerable", "").strip().lower() != "no"
                       and bool(r.get("source_url")))
        available = r.get("youtube_available", "").strip().lower() == "yes"
        clip = args.regen_dir / stim
        ref = fps.get(stim, [])
        if not regenerable or not ref:
            v, sim = "SKIP", None
        elif not clip.exists():
            # distinguish a source that no longer exists from one you simply
            # have not regenerated yet (matches report_missing_per_run.py)
            v, sim = ("MISSING" if available else "UNAVAILABLE"), None
        else:
            hashes = clip_phashes(clip)
            if not hashes:
                v, sim = "FAIL", None        # file present but cannot be decoded
            else:
                sim = similarity(ref, hashes)
                v = verdict(sim)
        counts[v] += 1
        report.append((stim, r.get("verified", ""), sim, v))
        if not args.quiet and v != "SKIP":
            s = "  --  " if sim is None else f"{sim:.3f}"
            print(f"{stim:24s} sim={s}  {v}")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["stimulus", "manifest_verified", "similarity", "verdict"])
            for stim, mv, sim, v in report:
                w.writerow([stim, mv, "" if sim is None else f"{sim:.4f}", v])

    checked = counts["PASS"] + counts["WARN"] + counts["FAIL"]
    print(f"\n{'='*58}")
    print(f"verified {checked} regenerated clips "
          f"(thresholds: PASS>={PASS_MIN:.2f}, WARN>={WARN_MIN:.2f})")
    print(f"  PASS        {counts['PASS']:4d}   reproduces the reference clip")
    print(f"  WARN        {counts['WARN']:4d}   ambiguous -- inspect")
    print(f"  FAIL        {counts['FAIL']:4d}   no match, or undecodable")
    print(f"  MISSING     {counts['MISSING']:4d}   downloadable, not produced")
    print(f"  UNAVAILABLE {counts['UNAVAILABLE']:4d}   source deleted from YouTube")
    print(f"  SKIP        {counts['SKIP']:4d}   non-regenerable (no source / dropped)")
    if args.out:
        print(f"per-clip report -> {args.out}")
    # fail if any produced clip looks wrong, or if nothing was verified at all
    if counts["FAIL"] or counts["WARN"]:
        return 1
    if checked == 0:
        print("warning: no regenerated clips were verified", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
