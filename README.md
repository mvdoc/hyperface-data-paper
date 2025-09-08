# Hyperface fMRI Dataset: Quality Assurance & Analysis

This repository contains preprocessing scripts, quality assurance analyses, and data visualization tools for the Hyperface functional MRI dataset. The Hyperface dataset consists of fMRI data collected while participants viewed dynamic facial stimuli.

## Repository Structure

```
hyperface-data-paper/
├── data/                    # BIDS dataset (git submodule/datalad)
│   ├── [raw BIDS data]     # Raw functional and structural data
│   └── derivatives/        # BIDS derivatives
│       ├── fmriprep/      # fMRIprep preprocessed data
│       ├── freesurfer/    # FreeSurfer outputs
│       └── qa/            # Quality assurance metrics & figures
├── src/hyperface/         # Python analysis package
├── notebooks/             # Jupyter notebooks for QA analyses
├── scripts/               # Processing scripts
├── docs/                  # Jupyter Book source
└── pyproject.toml         # UV dependency management
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management.

### Prerequisites
- Python 3.8+
- uv (install with: `pip install uv`)

### Setup
```bash
# Clone the repository
git clone https://github.com/mvdoc/hyperface-data-paper
cd hyperface-data-paper

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### Data Setup
The dataset will be managed as a git submodule with datalad:
```bash
# Add data submodule (when ready)
git submodule add <datalad-dataset-url> data
cd data && datalad get .
```

## Quality Assurance Pipeline

The QA pipeline includes comprehensive analyses:

1. **Motion Assessment**: Head motion parameters during scanning
2. **Temporal SNR**: Signal quality after confound regression  
3. **Inter-Subject Correlation**: Stimulus-driven response consistency
4. **Surface Visualization**: Cortical surface mapping with pycortex
5. **Outlier Detection**: Identification of problematic scans

### Running QA Scripts

Scripts are organized by analysis type:
```bash
# Compute tSNR for volume data
python scripts/quality-assurance/compute-tsnr-volume.py sub-001

# Compute tSNR for surface data  
python scripts/quality-assurance/compute-tsnr-fsaverage.py sub-001

# Compute inter-subject correlation
python scripts/quality-assurance/compute-isc-fsaverage.py
```

All outputs are saved to `data/derivatives/qa/` following BIDS conventions.

## Documentation Website

This repository generates a Jupyter Book website with QA results:

```bash
# Install docs dependencies
uv sync --extra docs

# Build the book
jupyter-book build docs/

# Serve locally
jupyter-book build docs/ --builder=html
```

## Key Features

- **BIDS Compliance**: Follows neuroimaging data standards
- **Modern Dependency Management**: Uses uv for fast, reproducible environments  
- **Comprehensive QA**: Motion, tSNR, ISC, outlier detection
- **Web Documentation**: Interactive Jupyter Book with all analyses
- **Surface Visualization**: High-quality cortical surface plots
- **Automated Pipeline**: Scripts for reproducible QA workflow

## Contributing

See `TODO.md` for current development tasks and contribution opportunities.
