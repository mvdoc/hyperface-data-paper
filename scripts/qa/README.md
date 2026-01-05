# Quality Assurance Scripts

## tSNR Analysis
- `qa-save-tsnr-volume.py` - Compute temporal signal-to-noise ratio (tSNR) for fMRI data in volume space
- `qa-plot-tsnr.py` - Generate quality assurance visualizations from pre-computed tSNR data
- `qa-generate-html-reports-tsnr.py` - Generate interactive HTML reports for tSNR

## Motion Analysis
- `qa-plot-motion.py` - Generate motion QA plots (motion traces, FD, violin plots)
- `qa-generate-html-reports-motion.py` - Generate interactive HTML reports for motion

## Usage

```bash
# Step 1: Compute tSNR for all subjects
python scripts/qa/qa-save-tsnr-volume.py

# Step 2: Generate QA plots for all subjects
python scripts/qa/qa-plot-tsnr.py

# Or process specific subjects
python scripts/qa/qa-save-tsnr-volume.py --subjects sub-001 sub-002
python scripts/qa/qa-plot-tsnr.py --subjects sub-001 sub-002

# Generate motion QA plots
python scripts/qa/qa-plot-motion.py
```

### Output Structure

All outputs follow BIDS derivatives conventions and are saved to `data/derivatives/qa/tsnr/`:

```
data/derivatives/qa/tsnr/
├── sub-001/
│   ├── figures/                           # QA visualizations
│   │   ├── *_tsnr_mosaic.png             # Brain slice mosaics per run
│   │   ├── *_mediantsnr_mosaic.png       # Median tSNR across runs  
│   │   ├── *_brainmask_mosaic.png        # Conjunction brain mask
│   │   └── *_tsnr_violinplot.png         # tSNR distributions per session
│   ├── ses-1/                            # tSNR volumes (session 1)
│   │   └── *_desc-tsnr.nii.gz           
│   └── ses-2/                            # tSNR volumes (session 2)
│       └── *_desc-tsnr.nii.gz           
```
