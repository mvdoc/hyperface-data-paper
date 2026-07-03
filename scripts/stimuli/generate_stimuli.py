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
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "stimulus_sources.tsv"
DEFAULT_OUTPUT = HERE / "regenerated_stimuli"
DEFAULT_VIDEO_CACHE = HERE / "source_videos"


def video_id(url: str) -> str:
    """Extract the YouTube video id from a watch or youtu.be URL."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").removeprefix("www.")
    if host == "youtu.be":
        return parsed.path.lstrip("/")
    if host in ("youtube.com", "m.youtube.com"):
        vals = parse_qs(parsed.query).get("v")
        if vals:
            return vals[0]
    return url.rsplit("v=", 1)[-1]


# yt-dlp download strategies, tried in order. YouTube gates its default web client
# behind proof-of-origin (PO) tokens and answers anonymous requests with "Sign in to
# confirm you're not a bot"; the android_vr client is currently exempt and serves up
# to 720p. It often only offers video-only streams, which is fine: the stimuli are
# silent (the clips are cut with -an). See the yt-dlp PO Token Guide:
# https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide
_YTDLP_STRATEGIES = [
    ["-f", "mp4[height<=720]/best[height<=720]/best"],
    ["--extractor-args", "youtube:player_client=android_vr",
     "-f", "bestvideo[height<=720][ext=mp4]/best[height<=720][ext=mp4]"
           "/best[height<=720]/best"],
]


def download_source(url: str, cache: Path, extra_args: list[str]) -> Path | str:
    """Download the source video once (cached). Returns the path, or an error label:
    'blocked' (bot check / sign-in required -- the video may still exist) or
    'unavailable' (deleted, private, or otherwise gone)."""
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / f"{video_id(url)}.mp4"
    if dst.exists() and dst.stat().st_size > 100_000:
        return dst
    blocked = False
    for strategy in _YTDLP_STRATEGIES:
        try:
            proc = subprocess.run(
                ["yt-dlp", "-q", "--no-warnings", *strategy, *extra_args,
                 "-o", str(dst), url],
                capture_output=True, text=True, timeout=600,
            )
        except subprocess.TimeoutExpired:
            continue
        if proc.returncode == 0 and dst.exists():
            return dst
        err = proc.stderr.lower()
        # label reflects the LAST attempt: a bot check is worth retrying with the
        # next client, but a clear "unavailable" from any client is authoritative
        blocked = "sign in" in err or "not a bot" in err or "captcha" in err
        if not blocked:
            break
    return "blocked" if blocked else "unavailable"


def crop_filter(row: dict) -> str | None:
    """ffmpeg crop expression from the normalized box, or None for a full frame."""
    try:
        keys = ("crop_x0", "crop_y0", "crop_x1", "crop_y1")
        x0, y0, x1, y1 = (float(row[k]) for k in keys)
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
    ap.add_argument("--only", nargs="+", help="regenerate only these stimuli")
    ap.add_argument("--limit", type=int, help="stop after N stimuli (smoke test)")
    ap.add_argument("--overwrite", action="store_true", help="re-cut existing clips")
    ap.add_argument("--ytdlp-args", default="",
                    help="extra arguments passed through to yt-dlp, e.g. "
                         '"--cookies-from-browser firefox" if YouTube blocks '
                         "the anonymous download")
    args = ap.parse_args()
    extra_args = args.ytdlp_args.split()

    for tool in ("ffmpeg", "yt-dlp"):
        if not shutil.which(tool):
            print(f"error: '{tool}' not found on PATH", file=sys.stderr)
            return 1
    if not args.manifest.exists():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if args.only:
        want = set(args.only)
        rows = [r for r in rows if r["stimulus"] in want]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    made = skipped = no_source = failed = 0
    for row in rows:
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
        video = download_source(row["source_url"], args.video_cache, extra_args)
        if isinstance(video, str):
            no_source += 1
            if video == "blocked":
                print(f"[blocked] {row['stimulus']} <- {row['source_url']} "
                      "(bot check; see --ytdlp-args and the README)")
            else:
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
