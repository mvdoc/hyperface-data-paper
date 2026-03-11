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

The dataset is available on [OpenNeuro](https://openneuro.org/) and follows the [BIDS](https://bids.neuroimaging.io/) standard:

- **Raw data**: [ds007329](https://openneuro.org/datasets/ds007329)
- **fMRIPrep derivatives**: [ds007384](https://openneuro.org/datasets/ds007384)
- **FreeSurfer derivatives**: [ds007378](https://openneuro.org/datasets/ds007378)

To download the data using [datalad](https://www.datalad.org/):

```bash
# Install the raw data (included as a git submodule)
datalad install -r .

# Install the derivative datasets
datalad install -s https://github.com/OpenNeuroDatasets/ds007384.git data/derivatives/fmriprep
datalad install -s https://github.com/OpenNeuroDatasets/ds007378.git data/derivatives/freesurfer

# Download specific files as needed
datalad get data/sub-sid000005/
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
└── scripts/qa/             # QA pipeline scripts
```

## Citation

If you use this dataset, please cite:

> [Citation information to be added]

## License

BSD 2-Clause License. See [LICENSE](LICENSE) for details.
