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
python scripts/qa/qa-save-tsnr-volume.py              # Compute tSNR maps
python scripts/qa/qa-plot-tsnr.py                      # Generate plots
python scripts/qa/qa-generate-html-reports-tsnr.py    # HTML reports

# Motion analysis
python scripts/qa/qa-plot-motion.py                    # Generate plots
python scripts/qa/qa-generate-html-reports-motion.py  # HTML reports

# ISC analysis
python scripts/qa/qa-save-isc.py                       # Compute ISC
python scripts/qa/qa-plot-isc.py                       # Surface plots

# Process specific subjects
python scripts/qa/qa-plot-tsnr.py --subjects sub-sid000005
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
- `qa/` - QA pipeline (config, BIDS utilities, plotting)

### Data Organization (BIDS)
- `data/derivatives/fmriprep/` - Preprocessed fMRI data
- `data/derivatives/freesurfer/` - FreeSurfer outputs
- `data/derivatives/qa/` - QA outputs (generated)

### Scripts
- `scripts/qa/` - Main QA pipeline scripts
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
