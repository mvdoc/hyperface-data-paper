# Regenerating the video stimuli

The Hyperface stimuli are short clips cut from publicly available YouTube videos and
cannot be redistributed for copyright reasons. This directory provides everything needed
to reproduce the stimulus set from the original sources: a manifest of source URLs,
timestamps, and crop boxes, a script to re-cut the clips, and a script to verify the
result.

| file | contents |
|------|----------|
| `stimulus_sources.tsv` | one row per stimulus: source URL, in-video `start`, `duration_s`, normalized crop box (`crop_x0`–`crop_y1`), output size, fMRI `run`, `verified` status, `robust_ncc` score, and `youtube_available` |
| `stimulus_source_videos.tsv` | one row per source video: URL, title, channel, upload date, number of clips, and availability |
| `generate_stimuli.py` | downloads each source video and re-cuts the clips |
| `verify_stimuli.py` | checks regenerated clips against the published fingerprints |
| `stimulus_fingerprints.tsv` | per-clip perceptual-hash fingerprints of the originals, used by `verify_stimuli.py` |
| `report_missing_per_run.py` | reports, per fMRI run, how many clips cannot be regenerated |
| `stim_fingerprint.py`, `build_fingerprints.py` | perceptual-hash implementation and the authors-only fingerprint builder |

## Usage

```bash
pip install yt-dlp          # ffmpeg must also be on PATH
python generate_stimuli.py                     # regenerate all clips -> ./regenerated_stimuli/
python generate_stimuli.py --only face023.mp4  # a single clip
python verify_stimuli.py                       # verify the regenerated clips
python report_missing_per_run.py               # expected missing clips, per run
```

For each stimulus, `generate_stimuli.py` runs:

```
ffmpeg -ss <start> -i <source.mp4> -t <duration_s> \
       -vf "crop=<crop_box>,scale=<W>:<H>,fps=30" -an  <stimulus>
```

Output clips are 30 fps and silent. Most are 854×480; `face227`–`face230` are 854×468.
Each clip's size is given by the manifest's `out_width` and `out_height`. Clip length
varies: most clips are 4 s (120 frames) and about 340 are longer (commonly 5 s /
150 frames), so `duration_s` gives each clip's stored length. The `start` column may be
written as `H:MM:SS`, `M:SS`, or decimal seconds, all of which `ffmpeg -ss` accepts.

## Coverage

Source videos are removed from YouTube over time. As of July 2026, 58 of the 177 source
videos (195 clips) have been deleted, so a regeneration today yields:

- 486 / 707 clips reproducible (366 frame-for-frame, 120 from a re-encoded source)
- 221 / 707 missing (source deleted, no source recorded, or source altered)

The `youtube_available` column marks which clips are still downloadable, and
`report_missing_per_run.py` gives the per-run breakdown (22–42 % missing per run).

Each clip carries a `verified` status assigned by comparing our regeneration against the
original stored clip with two metrics:

- **SSIM** (Structural Similarity Index): per-frame agreement in local luminance,
  contrast, and structure, scored in `[-1, 1]`, after aligning for a small frame offset.
- **robust NCC**: the median over frames of the best zero-mean normalized
  cross-correlation within a ±6-frame window. NCC is invariant to brightness and contrast,
  and the window tolerates a 25-vs-30 fps offset, so this score remains high when the only
  surviving source is a re-encode. Correctly located clips score ≥ 0.94; a wrong time or
  crop scores ≤ 0.46.

The counts below reflect every reproduction the authors confirmed against their original
stored clips (674 as `parity` or `close`), independent of whether the source is still
online. Since verification, 188 of those source videos have been removed from YouTube, so
a download today reproduces only the 486 clips reported above; the other 188 remain
verified but no longer obtainable.

| `verified` | count | meaning |
|------------|------:|---------|
| `parity` | 517 | reproduces the original frame for frame (≥ 90 % of aligned frames at SSIM ≥ 0.90; `robust_ncc` ≥ 0.94) |
| `close` | 157 | same clip, but the current source is a re-encode (lower frame rate or desaturated) that lowers SSIM; confirmed at `robust_ncc` ≥ 0.70 |
| `unverified` | 4 | `face050`, `face054`, `face056`, `face058`: the original full-length source was missing from the authors' archive, and the only surviving upload is a different capture of the same event, with different framing and timing, so the clip cannot be reproduced or verified. Flagged `regenerable=no`; `source_url` kept as provenance |
| `source_unavailable` | 29 | source video deleted from YouTube and not held in the authors' archive, so the clip could be neither regenerated nor verified. Flagged `regenerable=no` |

## Verifying a regeneration

Because the originals cannot be redistributed, `verify_stimuli.py` compares regenerated
clips against `stimulus_fingerprints.tsv` instead. That file stores, for each clip, one
64-bit perceptual hash per frame, computed from the original. A perceptual hash encodes a
frame's low-frequency (DCT) structure and cannot be used to reconstruct the frame, so
releasing it does not redistribute the clips.

```bash
python verify_stimuli.py                        # verify ./regenerated_stimuli
python verify_stimuli.py --only face023.mp4
python verify_stimuli.py --out verification.tsv # also write a per-clip report
```

The script recomputes the per-frame hashes and reports a similarity in `[0, 1]`: the
median over reference frames of the best hash agreement within a ±6-frame window (the same
windowing as robust NCC, so it tolerates frame-rate and re-encoding differences).

| verdict | similarity | meaning |
|---------|-----------|---------|
| PASS | ≥ 0.85 | matches the reference clip (correct clips score ≥ 0.93) |
| WARN | 0.70–0.85 | ambiguous; inspect |
| FAIL | < 0.70 | does not match the reference clip, or the file cannot be decoded |
| MISSING | — | source still downloadable, but this clip was not produced |
| UNAVAILABLE | — | source video was deleted from YouTube; the clip cannot be regenerated |
| SKIP | — | clip is non-regenerable in the manifest |

The thresholds fall in a gap measured across all 674 regenerable clips: a correct
regeneration scores ≥ 0.93, while a clip compared against a different clip's fingerprint
never exceeds 0.72. The script exits non-zero if any clip is WARN or FAIL, or if nothing
was verified, and requires only `ffmpeg`, `numpy`, and `scipy`.
