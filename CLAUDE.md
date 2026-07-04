# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Essential Commands

### Environment Setup
```bash
uv venv
source .venv/bin/activate
uv sync
```

Always activate the virtual environment before running scripts:
```bash
source .venv/bin/activate && python scripts/qa/...
```

### QA Pipeline

```bash
# tSNR analysis
python scripts/qa/qa-save-tsnr-volume.py               # Compute volume tSNR maps
python scripts/qa/qa-save-tsnr-surface.py              # Compute surface (fsaverage6) tSNR
python scripts/qa/qa-plot-tsnr.py                      # Generate plots
python scripts/qa/qa-generate-html-reports-tsnr.py     # HTML reports
python scripts/qa/print-tsnr-summary.py                # Summary stats for paper

# Motion analysis
python scripts/qa/qa-plot-motion.py                    # Generate plots
python scripts/qa/qa-generate-html-reports-motion.py   # HTML reports
python scripts/qa/print-motion-summary.py              # Summary stats for paper

# ISC analysis
python scripts/qa/qa-save-isc.py                       # Compute ISC
python scripts/qa/qa-plot-isc.py                       # Surface plots

# Behavior / participants
python scripts/qa/qa-plot-accuracy.py                  # Task accuracy per run
python scripts/qa/qa-plot-participant-datasets.py      # Participant x dataset matrix
python scripts/qa/qa-print-participant-demographics.py # Demographic stats

# Process specific subjects
python scripts/qa/qa-plot-tsnr.py --subjects sub-sid000005
```

### Stimulus Regeneration

The video stimuli cannot be redistributed (copyright), so `scripts/stimuli/` regenerates
them from the original YouTube sources and verifies the result against released perceptual
fingerprints. See `scripts/stimuli/README.md` for full details.

```bash
pip install yt-dlp                                     # ffmpeg must also be on PATH
python scripts/stimuli/generate_stimuli.py             # re-cut all clips -> ./regenerated_stimuli/
python scripts/stimuli/verify_stimuli.py               # verify against stimulus_fingerprints.tsv
python scripts/stimuli/report_missing_per_run.py       # expected missing clips, per run
```

### Code Quality
```bash
ruff check src/ scripts/qa/
ruff format src/ scripts/qa/
```

## Code Architecture

### Package Structure (`src/hyperface/`)
- `utils.py` - fMRI processing (`compute_tsnr()`, `clean_data()`)
- `viz.py` - Visualization (mosaic plots, surface plots)
- `io.py` - Data loading and BIDS I/O
- `isc.py` - Inter-subject correlation
- `qa/` - QA pipeline (`config.py`, `bids.py`, `motion.py`, `tsnr.py`, `plotting.py`)
- `assets/` - Packaged config/data files (`qa_config.yaml` default config, `visualmemory_run_order.yaml`)

### Data Organization (BIDS)
`data/` is a git submodule sourced from the OpenNeuro dataset **ds007329**
(`https://github.com/OpenNeuroDatasets/ds007329.git`, `main` branch); its derivatives
are themselves nested OpenNeuro submodules (fmriprep → ds007384, freesurfer → ds007378).
Keep the submodule on the OpenNeuro remote — do not check it out from private/backup
remotes. Annexed content is fetched via git-annex/datalad.

- `data/derivatives/fmriprep/` - Preprocessed fMRI data
- `data/derivatives/freesurfer/` - FreeSurfer outputs
- `data/derivatives/qa/` - QA outputs (generated)

### Scripts
- `scripts/qa/` - Main QA pipeline scripts
- `scripts/stimuli/` - Regenerate and verify the video stimuli from YouTube sources
- `scripts/presentation/` - Legacy stimulus presentation (do not modify)

### Writing New Scripts

Use the centralized configuration:

```python
from hyperface.qa import create_qa_argument_parser, get_config

def main():
    parser = create_qa_argument_parser(description="Script description.")
    args = parser.parse_args()
    config = get_config(config_path=args.config, data_dir=args.data_dir)

    # Use config.paths for all paths
    output_dir = config.paths.qa_base_dir / "output"
```

Config paths available: `data_dir`, `derivatives_dir`, `fmriprep_dir`, `qa_base_dir`, `tsnr_dir`, `motion_dir`, `isc_dir`, `stimuli_dir`

## Key Details

### fMRI Processing
`compute_tsnr()` uses confound regression:
- 6 motion parameters + derivatives
- Global signal, framewise displacement
- 6 aCompCor components
- Polynomial regressors (2nd order)

### Path Conventions
- Inputs: `data/derivatives/fmriprep/{subject}/func/`
- Outputs: `data/derivatives/qa/{analysis_type}/`
- Figures: `data/derivatives/qa/{analysis_type}/figures/`

### Expected File Naming (BIDS)
- Volume: `*_space-T1w_desc-preproc_bold.nii.gz`
- Surface: `*_space-fsaverage_hemi-{L,R}_bold.func.gii`
- Confounds: `*_desc-confounds_timeseries.tsv`
