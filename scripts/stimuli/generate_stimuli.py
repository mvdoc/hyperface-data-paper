#!/usr/bin/env python3
"""Regenerate the Hyperface video stimuli from their public YouTube sources.

The stimulus clips are short cuts of publicly available YouTube videos and cannot be
redistributed. Instead we publish, for every stimulus, the source URL + in-video
timestamp + crop rectangle (``stimulus_sources.tsv``); this script downloads each
source video and re-cuts the exact clip, so the stimulus set can be reproduced from
scratch.

For each stimulus it runs, in effect:

    ffmpeg -ss <start> -i <video> -t <duration> \
           -vf "crop=<box>,scale=<W>:<H>,fps=30" -an ... <stimulus>

How the source info was recovered (URL / timestamp / crop) is documented in the
acquisition repository under ``stimuli/source_recovery/``.

Requirements: ``ffmpeg`` and ``yt-dlp`` on PATH (``pip install yt-dlp``). No other deps.

Examples
--------
    python generate_stimuli.py                              # regenerate everything
    python generate_stimuli.py --output-dir /tmp/stimuli    # custom output
    python generate_stimuli.py --only face023.mp4 face300.mp4   # a few
    python generate_stimuli.py --limit 3                    # quick smoke test
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "stimulus_sources.tsv"
DEFAULT_OUTPUT = HERE / "regenerated_stimuli"
DEFAULT_VIDEO_CACHE = HERE / "source_videos"


def video_id(url: str) -> str:
    return url.rsplit("v=", 1)[-1]


def download_source(url: str, cache: Path) -> Path | None:
    """Download the source video once (cached). Returns path or None if unavailable."""
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / f"{video_id(url)}.mp4"
    if dst.exists() and dst.stat().st_size > 100_000:
        return dst
    try:
        subprocess.run(
            ["yt-dlp", "-q", "--no-warnings",
             "-f", "mp4[height<=720]/best[height<=720]/best",
             "-o", str(dst), url],
            check=True, timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return dst if dst.exists() else None


def crop_filter(row: dict) -> str | None:
    """ffmpeg crop expression from the normalized box, or None for a full frame."""
    try:
        x0, y0, x1, y1 = (float(row[k]) for k in ("crop_x0", "crop_y0", "crop_x1", "crop_y1"))
    except (KeyError, ValueError):
        return None
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(1.0, x1), min(1.0, y1)
    if (x0, y0, x1, y1) == (0.0, 0.0, 1.0, 1.0) or x1 <= x0 or y1 <= y0:
        return None
    w, h, x, y = x1 - x0, y1 - y0, x0, y0
    return f"crop={w:.4f}*iw:{h:.4f}*ih:{x:.4f}*iw:{y:.4f}*ih"


def generate_one(row: dict, video: Path, out_path: Path) -> bool:
    vf = [f for f in (crop_filter(row),
                      f"scale={row['out_width']}:{row['out_height']}",
                      "fps=30") if f]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(row["start"]), "-i", str(video),
        "-t", str(row["duration_s"]),
        "-vf", ",".join(vf),
        "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p", str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True)
        return out_path.exists() and out_path.stat().st_size > 1000
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--video-cache", type=Path, default=DEFAULT_VIDEO_CACHE)
    ap.add_argument("--only", nargs="+", help="regenerate only these stimulus filenames")
    ap.add_argument("--limit", type=int, help="stop after N stimuli (smoke test)")
    ap.add_argument("--overwrite", action="store_true", help="re-cut clips that already exist")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest), delimiter="\t"))
    if args.only:
        want = set(args.only)
        rows = [r for r in rows if r["stimulus"] in want]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    made = skipped = no_source = failed = 0
    for i, row in enumerate(rows):
        if args.limit and made + failed >= args.limit:
            break
        out_path = args.output_dir / row["stimulus"]
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        if row.get("regenerable") == "no" or not row.get("source_url"):
            no_source += 1
            note = "source video deleted from YouTube" if row.get("source_url") \
                else "no source recovered"
            print(f"[no-source] {row['stimulus']} ({note})")
            continue
        video = download_source(row["source_url"], args.video_cache)
        if video is None:
            no_source += 1
            print(f"[unavailable] {row['stimulus']} <- {row['source_url']}")
            continue
        ok = generate_one(row, video, out_path)
        if ok:
            made += 1
            print(f"[ok] {row['stimulus']}  ({row['source_url']} @ {row['start']})")
        else:
            failed += 1
            print(f"[FAIL] {row['stimulus']}")

    print(f"\nregenerated {made} | skipped(existing) {skipped} | "
          f"no-source {no_source} | failed {failed} -> {args.output_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
