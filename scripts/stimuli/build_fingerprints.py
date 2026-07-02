#!/usr/bin/env python3
"""Build the published perceptual-hash fingerprints from the ORIGINAL stimuli.

Authors-only: this needs the original stored stimulus clips (which are NOT
redistributed). It computes, for every clip, a per-frame perceptual-hash sequence
(see ``stim_fingerprint.py``) and writes ``stimulus_fingerprints.tsv`` -- the
compact, one-way reference that ``verify_stimuli.py`` checks regenerated clips
against. Re-run this only if the pHash parameters or the stored clips change.

    python build_fingerprints.py --stored-dir /path/to/original/stimuli

The output is a two-column TSV: ``stimulus`` and ``frame_phash`` (comma-separated
64-bit hashes, one per frame). It contains no pixel data and cannot reconstruct the
footage, so it is safe to publish.
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

from stim_fingerprint import DCT_KEEP, HASH_SIDE, WIN, clip_phashes, encode

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "stimulus_sources.tsv"
DEFAULT_OUT = HERE / "stimulus_fingerprints.tsv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stored-dir", type=Path, required=True,
                    help="directory of the original stored stimulus clips")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", nargs="+", help="fingerprint only these stimuli")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("error: 'ffmpeg' not found on PATH", file=sys.stderr)
        return 1
    if not args.manifest.exists():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if args.only:
        want = set(args.only)
        rows = [r for r in rows if r["stimulus"] in want]

    out_rows, missing, empty = [], 0, 0
    for r in rows:
        stim = r["stimulus"]
        clip = args.stored_dir / stim
        if not clip.exists():
            missing += 1
            print(f"[missing] {stim} (no stored clip in {args.stored_dir})")
            continue
        hashes = clip_phashes(clip)
        if not hashes:
            empty += 1
            print(f"[decode-fail] {stim}")
            continue
        out_rows.append((stim, encode(hashes)))
        print(f"[ok] {stim}  {len(hashes)} frames")

    if not out_rows:
        print(f"error: no fingerprints computed ({missing} missing, {empty} "
              f"decode-fail); refusing to overwrite {args.out}", file=sys.stderr)
        return 1

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["stimulus", "frame_phash"])
        w.writerows(out_rows)

    size_kb = args.out.stat().st_size / 1024
    print(f"\nwrote {len(out_rows)} fingerprints -> {args.out} ({size_kb:.0f} KB) | "
          f"missing {missing} | decode-fail {empty}")
    print(f"params: {HASH_SIDE}x{HASH_SIDE} gray, {DCT_KEEP}x{DCT_KEEP} DCT "
          f"(64-bit), window +-{WIN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
