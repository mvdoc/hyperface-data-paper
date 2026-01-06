#!/usr/bin/env python
"""Compute and save tSNR volumes for BIDS fMRI data.

This script computes temporal signal-to-noise ratio (tSNR) for preprocessed
fMRI data from fMRIprep, using confound regression to clean the data before
computing the metric.

Examples:
    # Process all subjects, sessions, and tasks
    python scripts/qa/qa-save-tsnr-volume.py

    # Process specific subjects
    python scripts/qa/qa-save-tsnr-volume.py --subjects sub-001 sub-002

    # Process specific sessions and tasks
    python scripts/qa/qa-save-tsnr-volume.py --sessions ses-1 ses-2 --tasks localizer

    # Preview what would be processed
    python scripts/qa/qa-save-tsnr-volume.py --dry-run
"""

import os

import nibabel as nib
import nilearn.image as nimage
import pandas as pd
from bids import BIDSLayout
from tqdm import tqdm

from hyperface.qa import create_qa_argument_parser, get_config
from hyperface.utils import compute_tsnr


def main():
    parser = create_qa_argument_parser(
        description="Compute tSNR for BIDS fMRI data",
        include_subjects=True,
        include_sessions=True,
        include_tasks=True,
        include_config=True,
        include_dry_run=True,
    )
    args = parser.parse_args()

    # Load configuration
    config = get_config(
        config_path=args.config,
        data_dir=args.data_dir,
    )

    # Set up BIDSLayout for input data
    fmriprep_dir = config.paths.fmriprep_dir
    if not fmriprep_dir.exists():
        print(f"Error: fMRIprep directory not found: {fmriprep_dir}")
        return 1

    print(f"Loading BIDS layout from {fmriprep_dir}...")
    layout = BIDSLayout(fmriprep_dir, validate=False, derivatives=True)

    # Build query filters
    query = {
        "suffix": "bold",
        "space": "T1w",
        "desc": "preproc",
        "extension": ".nii.gz",
    }

    if args.subjects:
        # Normalize subject IDs (remove 'sub-' prefix if present)
        query["subject"] = [s.replace("sub-", "") for s in args.subjects]
    if args.sessions:
        query["session"] = [s.replace("ses-", "") for s in args.sessions]
    if args.tasks:
        query["task"] = [t.replace("task-", "") for t in args.tasks]

    # Find all functional files
    func_files = layout.get(**query)

    if not func_files:
        print("No files found matching query:")
        print(f"  Query parameters: {query}")
        # Show what IS available
        all_bold = layout.get(suffix="bold", extension=".nii.gz")
        if all_bold:
            n_files = len(all_bold)
            print(f"\n  Found {n_files} BOLD files total, but none matched filters.")
            print(f"  Example available file: {all_bold[0].path}")
            available_subjects = layout.get_subjects()[:5]
            available_tasks = layout.get_tasks()
            print(f"  Available subjects (first 5): {available_subjects}")
            print(f"  Available tasks: {available_tasks}")
        else:
            print(f"\n  No BOLD files found in {fmriprep_dir}")
            print("  Is this a valid fMRIprep derivatives directory?")
        return 1

    print(f"Found {len(func_files)} files to process")

    if args.dry_run:
        print("\nDry run - files that would be processed:")
        for bf in func_files:
            entities = bf.get_entities()
            sub = entities.get("subject", "?")
            ses = entities.get("session")
            task = entities.get("task", "?")
            run = entities.get("run")
            ses_str = f"/ses-{ses}" if ses else ""
            run_str = f"_run-{run}" if run else ""
            print(f"  sub-{sub}{ses_str}: task-{task}{run_str}")
        print(f"\nTotal: {len(func_files)} files")
        return 0

    # Output directory
    outdir_base = config.paths.tsnr_dir

    # Process files
    processed = 0
    skipped = 0
    errors = 0

    for bf in tqdm(func_files, desc="Processing files", unit="file"):
        entities = bf.get_entities()
        subject = f"sub-{entities['subject']}"
        session = f"ses-{entities['session']}" if entities.get("session") else None
        task = entities.get("task")
        run = entities.get("run")

        # Create output directory
        if session:
            outdir = outdir_base / subject / session
        else:
            outdir = outdir_base / subject
        outdir.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        func_basename = os.path.basename(bf.path)
        output_basename = func_basename.replace("desc-preproc_bold", "desc-tsnr")
        output_path = outdir / output_basename

        # Skip if output already exists
        if output_path.exists():
            tqdm.write(f"  Skipping {output_basename} (already exists)")
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
            tqdm.write(f"  Error: No confounds file found for {func_basename}")
            errors += 1
            continue

        conf_file = conf_files[0].path

        try:
            # Load data and compute tSNR
            session_str = f"/{session}" if session else ""
            tqdm.write(f"  Computing tSNR for {subject}{session_str}: {func_basename}")

            data = nib.load(bf.path).get_fdata()
            conf = pd.read_csv(conf_file, sep="\t")

            tsnr_data = compute_tsnr(data, conf)

            # Save tSNR volume
            tsnr_img = nimage.new_img_like(bf.path, tsnr_data)
            tsnr_img.to_filename(str(output_path))

            tqdm.write(f"  Saved: {output_basename}")
            processed += 1

        except FileNotFoundError as e:
            tqdm.write(f"  Error: File not found for {func_basename}: {e}")
            errors += 1
            continue
        except (nib.filebasedimages.ImageFileError, ValueError) as e:
            tqdm.write(f"  Error loading/processing {func_basename}: {e}")
            tqdm.write("    The NIfTI file may be corrupted.")
            errors += 1
            continue
        except pd.errors.ParserError as e:
            tqdm.write(f"  Error reading confounds for {func_basename}: {e}")
            tqdm.write(f"    Confounds file may be malformed: {conf_file}")
            errors += 1
            continue

    print("\nCompleted processing:")
    print(f"  Processed: {processed} files")
    print(f"  Skipped: {skipped} files (already existed)")
    print(f"  Errors: {errors} files")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
