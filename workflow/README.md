# QA Pipeline Workflow

This Snakemake workflow orchestrates the QA analysis pipelines for motion, tSNR, and ISC plots.

## Installation

Install snakemake as an optional dependency:

```bash
uv sync --extra workflow
```

Or install snakemake directly:

```bash
uv pip install snakemake>=8.0.0
```

## Usage

All commands should be run from the project root directory.

### Run all QA pipelines

```bash
snakemake --snakefile workflow/Snakefile --cores 1
```

### Run specific pipelines

```bash
# Motion plots only
snakemake --snakefile workflow/Snakefile motion_plots --cores 1

# tSNR plots (includes tSNR computation)
snakemake --snakefile workflow/Snakefile tsnr_plots --cores 1

# ISC plots (includes ISC computation)
snakemake --snakefile workflow/Snakefile isc_plots --cores 1
```

### Dry run (preview what would be executed)

```bash
snakemake --snakefile workflow/Snakefile -n
```

### Process specific subjects

```bash
snakemake --snakefile workflow/Snakefile motion_plots --cores 1 --config subjects="sub-001 sub-002"
```

### Force regeneration

```bash
snakemake --snakefile workflow/Snakefile tsnr_plots --cores 1 --forcerun
```

### Generate HTML reports

```bash
# Motion HTML reports
snakemake --snakefile workflow/Snakefile motion_report --cores 1

# tSNR HTML reports
snakemake --snakefile workflow/Snakefile tsnr_report --cores 1
```

### Summary statistics

```bash
snakemake --snakefile workflow/Snakefile tsnr_summary motion_summary --cores 1
```

## Pipeline Dependencies

```
fMRIprep outputs
├── motion_plots ─────────────────────→ motion figures
│   └── motion_report ────────────────→ HTML reports
│   └── motion_summary ───────────────→ summary stats
│
├── compute_tsnr ─────→ tsnr volumes
│   └── tsnr_plots ───────────────────→ tsnr figures
│       └── tsnr_report ──────────────→ HTML reports
│   └── tsnr_summary ─────────────────→ summary stats
│
└── compute_isc ──────→ ISC data
    └── isc_plots ────────────────────→ ISC figures
```

## Configuration

Edit `workflow/config/config.yaml` to:
- Enable/disable specific pipelines
- Filter subjects to process
- Force regeneration of outputs

The actual data paths are read from `src/hyperface/assets/qa_config.yaml`.

## Outputs

All outputs are saved under `data/derivatives/qa/`:
- `motion/` - Motion traces, FD traces, violin plots
- `tsnr/` - tSNR mosaics, violin plots, surface plots
- `isc/` - ISC violin plots, surface plots

## Cleanup

```bash
# Remove specific outputs
snakemake --snakefile workflow/Snakefile clean_motion
snakemake --snakefile workflow/Snakefile clean_tsnr
snakemake --snakefile workflow/Snakefile clean_isc

# Remove all QA outputs
snakemake --snakefile workflow/Snakefile clean_all
```
