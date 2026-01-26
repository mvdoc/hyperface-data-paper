# Hyperface fMRI Dataset

This repository contains analysis code and quality assurance tools for the Hyperface fMRI dataset. The dataset consists of fMRI data collected while participants viewed dynamic face stimuli in a visual memory task.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone https://github.com/mvdoc/hyperface-data-paper
cd hyperface-data-paper

# Create virtual environment and install
uv venv
source .venv/bin/activate
uv sync
```

## Data Access

The dataset is managed with [datalad](https://www.datalad.org/) and follows the [BIDS](https://bids.neuroimaging.io/) standard:

```bash
# Initialize the data submodule
git submodule update --init
cd data && datalad get .
```

## Quality Assurance

The repository includes a comprehensive QA pipeline for assessing data quality.

### Motion Analysis

```bash
# Generate motion parameter plots
python scripts/qa/qa-plot-motion.py

# Generate interactive HTML reports
python scripts/qa/qa-generate-html-reports-motion.py
```

### Temporal SNR Analysis

```bash
# Compute tSNR maps
python scripts/qa/qa-save-tsnr-volume.py

# Generate visualizations
python scripts/qa/qa-plot-tsnr.py

# Generate HTML reports
python scripts/qa/qa-generate-html-reports-tsnr.py
```

### Inter-Subject Correlation

```bash
# Compute ISC
python scripts/qa/qa-save-isc.py

# Generate surface visualizations
python scripts/qa/qa-plot-isc.py
```

All outputs are saved to `data/derivatives/qa/` following BIDS conventions.

## Repository Structure

```
hyperface-data-paper/
├── data/                    # BIDS dataset (datalad)
│   └── derivatives/
│       ├── fmriprep/       # Preprocessed fMRI data
│       ├── freesurfer/     # FreeSurfer outputs
│       └── qa/             # QA outputs
├── src/hyperface/          # Python analysis package
├── scripts/qa/             # QA pipeline scripts
├── notebooks/              # Analysis notebooks
└── docs/                   # Documentation (Jupyter Book)
```

## Documentation

Build the documentation website:

```bash
uv sync --extra docs
jupyter-book build docs/
```

## Citation

If you use this dataset, please cite:

> [Citation information to be added]

## License

BSD 2-Clause License. See [LICENSE](LICENSE) for details.
