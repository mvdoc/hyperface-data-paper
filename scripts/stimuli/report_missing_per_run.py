#!/usr/bin/env python3
"""How many stimulus clips to expect MISSING per fMRI run if you regenerate the stimuli
by downloading the source videos from YouTube *today*.

This is reader-facing: a clip is REPRODUCIBLE only if its source video is still
on YouTube (`youtube_available=yes`) AND it reproduces the original (`verified` =
parity or close). Otherwise it is MISSING, for one of:
  - youtube_deleted : the source YouTube video has been removed (you cannot download it)
  - no_source       : the clip's source was never recovered (no URL)
  - drift_corrupt   : the video is still up but its content drifted / can't reproduce

Note: many clips whose source video is now deleted were still verified as reproducing
(the authors regenerated them from an archived copy), but a NEW download cannot obtain
them -- so they count as missing here. The fraction missing grows over time as more
source videos are taken down.

  python report_missing_per_run.py [--manifest stimulus_sources.tsv]
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ('n', 'parity', 'close', 'deleted', 'no_source', 'drift')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=os.path.join(HERE, 'stimulus_sources.tsv'))
    a = ap.parse_args()

    if not os.path.exists(a.manifest):
        print(f"error: manifest not found: {a.manifest}", file=sys.stderr)
        return

    by_run = defaultdict(lambda: dict.fromkeys(COLS, 0))
    with open(a.manifest, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh, delimiter='\t'))
    for r in rows:
        d = by_run[r.get('run', '')]
        d['n'] += 1
        up = r.get('youtube_available') == 'yes'
        v = r.get('verified')
        has_url = bool((r.get('source_url') or '').strip())
        if up and v == 'parity':
            d['parity'] += 1
        elif up and v == 'close':
            d['close'] += 1
        elif up:            # on YouTube but can't reproduce (drift / corrupt)
            d['drift'] += 1
        elif has_url:       # had a URL, video now deleted
            d['deleted'] += 1
        else:               # source never recovered
            d['no_source'] += 1

    runs = sorted(by_run, key=lambda x: int(x) if str(x).isdigit() else 99)
    tot = dict.fromkeys(COLS, 0)
    print('Per-run clips to expect MISSING if you regenerate from YouTube today.')
    print('  reproducible = source still on YouTube AND matches the original')
    print('  (parity = frame-for-frame, close = same clip minor residual).\n')
    print(f"  {'run':>4} {'n':>4} {'parity':>7} {'close':>6} {'MISSING':>8} "
          f"{'deleted':>8} {'no_src':>7} {'drift':>6} {'%miss':>6}")
    for run in runs:
        d = by_run[run]
        miss = d['deleted'] + d['no_source'] + d['drift']
        for k in tot:
            tot[k] += d[k]
        print(f"  {str(run):>4} {d['n']:>4} {d['parity']:>7} {d['close']:>6} "
              f"{miss:>8} {d['deleted']:>8} {d['no_source']:>7} {d['drift']:>6} "
              f"{100.0 * miss / d['n']:>6.1f}")
    n = tot['n']
    miss = tot['deleted'] + tot['no_source'] + tot['drift']
    ok = tot['parity'] + tot['close']
    print(f"  {'ALL':>4} {n:>4} {tot['parity']:>7} {tot['close']:>6} "
          f"{miss:>8} {tot['deleted']:>8} {tot['no_source']:>7} {tot['drift']:>6} "
          f"{100.0 * miss / n:>6.1f}")
    print(f"\n  Downloading from YouTube today you can reproduce {ok}/{n} clips "
          f"({100.0 * ok / n:.1f}%): {tot['parity']} frame-exact + "
          f"{tot['close']} near-exact.")
    print(f"  {miss}/{n} ({100.0 * miss / n:.1f}%) are MISSING: "
          f"{tot['deleted']} source video deleted, {tot['no_source']} no source "
          f"on record, {tot['drift']} drifted/corrupt.")
    print(f"  (The {tot['deleted']} deleted-source clips were verified by the "
          f"authors from an archived copy, but a fresh download can no longer "
          f"obtain them.)")


if __name__ == '__main__':
    main()
