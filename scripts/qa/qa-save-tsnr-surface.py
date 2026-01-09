#!/usr/bin/env python
"""Compute and save tSNR for surface (fsaverage6) fMRI data.

This script computes temporal signal-to-noise ratio (tSNR) for preprocessed
fMRI data in fsaverage6 surface space, using confound regression to clean
the data before computing the metric.

Examples:
    # Process all subjects, sessions, and tasks
    python scripts/qa/qa-save-tsnr-surface.py

    # Process specific subjects
    python scripts/qa/qa-save-tsnr-surface.py --subjects sub-001 sub-002

    # Process specific sessions and tasks
    python scripts/qa/qa-save-tsnr-surface.py --sessions ses-1 ses-2 --tasks localizer

    # Preview what would be processed
    python scripts/qa/qa-save-tsnr-surface.py --dry-run
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from bids import BIDSLayout
from tqdm import tqdm

from hyperface.io import save_gifti
from hyperface.qa import create_qa_argument_parser, get_config
from hyperface.utils import compute_tsnr_surface


def load_gifti_timeseries(gifti_path: str) -> np.ndarray:
    """Load GIFTI time series and stack into 2D array.

    Parameters
    ----------
    gifti_path : str
        Path to GIFTI func file with multiple darrays (one per timepoint)

    Returns
    -------
    data : np.ndarray
        Array of shape (n_timepoints, n_vertices)
    """
    gii = nib.load(gifti_path)
    return np.vstack([d.data for d in gii.darrays])


def main():
    parser = create_qa_argument_parser(
        description="Compute tSNR for surface (fsaverage6) fMRI data",
        include_subjects=True,
        include_sessions=True,
        include_tasks=True,
        include_config=True,
        include_dry_run=True,
    )
    args = parser.parse_args()

    config = get_config(
        config_path=args.config,
        data_dir=args.data_dir,
    )

    fmriprep_dir = config.paths.fmriprep_dir
    if not fmriprep_dir.exists():
        print(f"Error: fMRIprep directory not found: {fmriprep_dir}")
        return 1

    print(f"Loading BIDS layout from {fmriprep_dir}...")
    layout = BIDSLayout(fmriprep_dir, validate=False, derivatives=True)

    # Build query filters for fsaverage6 surface data (left hemisphere)
    # Use .func.gii extension for GIFTI functional files
    query = {
        "suffix": "bold",
        "space": "fsaverage6",
        "hemi": "L",
        "extension": [".gii", ".func.gii"],
    }

    if args.subjects:
        query["subject"] = [s.replace("sub-", "") for s in args.subjects]
    if args.sessions:
        query["session"] = [s.replace("ses-", "") for s in args.sessions]
    if args.tasks:
        query["task"] = [t.replace("task-", "") for t in args.tasks]

    func_files_L = layout.get(**query)

    if not func_files_L:
        print("No files found matching query:")
        print(f"  Query parameters: {query}")
        all_gii = layout.get(suffix="bold", extension=".gii")
        if all_gii:
            n_files = len(all_gii)
            print(f"\n  Found {n_files} BOLD GIFTI files, but none matched.")
            print(f"  Example available file: {all_gii[0].path}")
        else:
            print(f"\n  No BOLD GIFTI files found in {fmriprep_dir}")
        return 1

    print(f"Found {len(func_files_L)} file pairs to process")

    if args.dry_run:
        print("\nDry run - files that would be processed:")
        for bf in func_files_L:
            entities = bf.get_entities()
            sub = entities.get("subject", "?")
            ses = entities.get("session")
            task = entities.get("task", "?")
            run = entities.get("run")
            ses_str = f"/ses-{ses}" if ses else ""
            run_str = f"_run-{run}" if run else ""
            print(f"  sub-{sub}{ses_str}: task-{task}{run_str} (L+R hemispheres)")
        print(f"\nTotal: {len(func_files_L)} file pairs")
        return 0

    outdir_base = config.paths.tsnr_dir

    processed = 0
    skipped = 0
    errors = 0

    for bf_L in tqdm(func_files_L, desc="Processing files", unit="file"):
        entities = bf_L.get_entities()
        subject = f"sub-{entities['subject']}"
        session = f"ses-{entities['session']}" if entities.get("session") else None
        task = entities.get("task")
        run = entities.get("run")

        # Find corresponding right hemisphere file
        R_query = {
            "subject": entities["subject"],
            "task": task,
            "suffix": "bold",
            "space": "fsaverage6",
            "hemi": "R",
            "extension": [".gii", ".func.gii"],
        }
        if entities.get("session"):
            R_query["session"] = entities["session"]
        if run:
            R_query["run"] = run

        func_files_R = layout.get(**R_query)
        if not func_files_R:
            tqdm.write(f"  Error: No right hemisphere file found for {bf_L.path}")
            errors += 1
            continue
        bf_R = func_files_R[0]

        # Create output directory
        if session:
            outdir = outdir_base / subject / session
        else:
            outdir = outdir_base / subject
        outdir.mkdir(parents=True, exist_ok=True)

        # Generate output filenames
        func_basename_L = Path(bf_L.path).name
        func_basename_R = Path(bf_R.path).name

        old_suffix = "_bold.func.gii"
        new_suffix = "_desc-tsnr.func.gii"
        output_basename_L = func_basename_L.replace(old_suffix, new_suffix)
        output_basename_R = func_basename_R.replace(old_suffix, new_suffix)

        output_path_L = outdir / output_basename_L
        output_path_R = outdir / output_basename_R

        # Skip if both outputs already exist
        if output_path_L.exists() and output_path_R.exists():
            tqdm.write(f"  Skipping {output_basename_L} (already exists)")
            skipped += 1
            continue

        # Find corresponding confounds file
        conf_query = {
            "subject": entities["subject"],
            "task": task,
            "suffix": "timeseries",
            "desc": "confounds",
            "extension": ".tsv",
        }
        if entities.get("session"):
            conf_query["session"] = entities["session"]
        if run:
            conf_query["run"] = run

        conf_files = layout.get(**conf_query)
        if not conf_files:
            tqdm.write(f"  Error: No confounds file found for {func_basename_L}")
            errors += 1
            continue

        conf_file = conf_files[0].path

        try:
            session_str = f"/{session}" if session else ""
            tqdm.write(
                f"  Computing tSNR for {subject}{session_str}: {func_basename_L}"
            )

            # Load surface data and confounds
            data_L = load_gifti_timeseries(bf_L.path)
            data_R = load_gifti_timeseries(bf_R.path)
            conf = pd.read_csv(conf_file, sep="\t")

            # Compute and save tSNR for each hemisphere
            save_gifti(compute_tsnr_surface(data_L, conf), output_path_L)
            save_gifti(compute_tsnr_surface(data_R, conf), output_path_R)

            tqdm.write(f"  Saved: {output_basename_L}, {output_basename_R}")
            processed += 1

        except (
            FileNotFoundError,
            nib.filebasedimages.ImageFileError,
            ValueError,
            pd.errors.ParserError,
        ) as e:
            tqdm.write(f"  Error processing {func_basename_L}: {e}")
            errors += 1

    print("\nCompleted processing:")
    print(f"  Processed: {processed} file pairs")
    print(f"  Skipped: {skipped} file pairs (already existed)")
    print(f"  Errors: {errors} file pairs")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
