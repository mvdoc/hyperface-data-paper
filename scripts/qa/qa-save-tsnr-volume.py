#!/usr/bin/env python
import argparse
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import nibabel as nib
import nilearn.image as nimage
import pandas as pd
import yaml
from tqdm import tqdm

from hyperface.utils import compute_tsnr


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required keys
    required_keys = ["directories"]
    for key in required_keys:
        if key not in config:
            msg = f"Missing required key in config: {key}"
            raise ValueError(msg)

    dirs = config["directories"]
    required_dir_keys = [
        "data_dir",
        "derivatives_dir",
        "input_derivative",
        "output_derivative",
        "output_subdir",
    ]
    for key in required_dir_keys:
        if key not in dirs:
            msg = f"Missing required directory key in config: {key}"
            raise ValueError(msg)

    return config


def get_directory_paths(config: dict, script_dir: str) -> Tuple[str, str]:
    """Get input and output directory paths from config."""
    dirs = config["directories"]

    # Base paths
    data_dir = os.path.abspath(os.path.join(script_dir, "..", dirs["data_dir"]))
    indir = os.path.join(data_dir, dirs["derivatives_dir"], dirs["input_derivative"])
    outdir_base = os.path.join(
        data_dir,
        dirs["derivatives_dir"],
        dirs["output_derivative"],
        dirs["output_subdir"],
    )

    return indir, outdir_base


def discover_subjects(indir: str, subjects: Optional[List[str]] = None) -> List[str]:
    """Discover subjects in the input directory."""
    if subjects:
        # Validate specified subjects
        found_subjects = []
        for subj in subjects:
            if not subj.startswith("sub-"):
                subj = f"sub-{subj}"

            subj_dir = os.path.join(indir, subj)
            if os.path.exists(subj_dir):
                found_subjects.append(subj)
            else:
                pass
        return found_subjects
    else:
        # Auto-discover all subjects
        indir_path = Path(indir)
        if not indir_path.exists():
            msg = f"Input directory not found: {indir}"
            raise FileNotFoundError(msg)

        subjects = sorted(
            [
                d.name
                for d in indir_path.iterdir()
                if d.is_dir() and d.name.startswith("sub-")
            ]
        )
        return subjects


def discover_sessions(
    subject_dir: Path, sessions: Optional[List[str]] = None
) -> List[str]:
    """Discover sessions for a subject."""
    if sessions:
        # Validate specified sessions
        found_sessions = []
        for sess in sessions:
            if not sess.startswith("ses-"):
                sess = f"ses-{sess}"

            sess_dir = subject_dir / sess
            if sess_dir.exists():
                found_sessions.append(sess)
            else:
                pass
        return found_sessions
    else:
        # Auto-discover all sessions
        sessions = sorted(
            [
                d.name
                for d in subject_dir.iterdir()
                if d.is_dir() and d.name.startswith("ses-")
            ]
        )

        # If no sessions found, check if there's a func directory directly under subject
        if not sessions and (subject_dir / "func").exists():
            return []  # No sessions, files are directly under subject/func

        return sessions


def discover_tasks_in_directory(
    func_dir: Path, tasks: Optional[List[str]] = None
) -> List[str]:
    """Discover tasks in a functional directory."""
    if not func_dir.exists():
        return []

    # Find all functional files and extract unique tasks
    pattern = "*_task-*_*space-T1w_desc-preproc_bold.nii.gz"
    func_files = list(func_dir.glob(pattern))

    if tasks:
        # Filter to specified tasks
        available_tasks = set()
        for f in func_files:
            parts = f.name.split("_")
            task_part = [p for p in parts if p.startswith("task-")]
            if task_part:
                available_tasks.add(task_part[0])

        found_tasks = []
        for task in tasks:
            if not task.startswith("task-"):
                task = f"task-{task}"

            if task in available_tasks:
                found_tasks.append(task)
            else:
                pass
        return found_tasks
    else:
        # Auto-discover all tasks
        tasks_set = set()
        for f in func_files:
            parts = f.name.split("_")
            task_part = [p for p in parts if p.startswith("task-")]
            if task_part:
                tasks_set.add(task_part[0])

        return sorted(tasks_set)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute tSNR for BIDS fMRI data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subjects, sessions, and tasks
  python scripts/qa/qa-save-tsnr-volume.py

  # Process specific subjects
  python scripts/qa/qa-save-tsnr-volume.py --subjects sub-001 sub-002

  # Process specific sessions and tasks
  python scripts/qa/qa-save-tsnr-volume.py --sessions ses-1 ses-2 --tasks localizer

  # Use custom config file
  python scripts/qa/qa-save-tsnr-volume.py --config /path/to/custom-config.yaml

  # Preview what would be processed
  python scripts/qa/qa-save-tsnr-volume.py --dry-run
""",
    )

    parser.add_argument(
        "--subjects", nargs="+", help="Subject IDs to process (default: all subjects)"
    )
    parser.add_argument(
        "--sessions", nargs="+", help="Session IDs to process (default: all sessions)"
    )
    parser.add_argument(
        "--tasks", nargs="+", help="Task names to process (default: all tasks)"
    )
    parser.add_argument(
        "--config",
        default="config/qa-config.yaml",
        help="Path to configuration file (default: config/qa-config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually running",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Get script directory and load config
    script_dir = os.path.dirname(__file__)
    if not os.path.isabs(args.config):
        config_path = os.path.join(script_dir, "..", args.config)
    else:
        config_path = args.config

    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Get directory paths
    indir, outdir_base = get_directory_paths(config, script_dir)

    # Discover subjects
    subjects = discover_subjects(indir, args.subjects)
    if not subjects:
        print("No subjects found to process")
        sys.exit(1)

    print(f"Found {len(subjects)} subjects: {', '.join(subjects)}")

    total_files = 0
    processing_plan = []

    # Build processing plan
    for subject in tqdm(subjects, desc="Building processing plan", unit="subject"):
        subject_dir = Path(indir) / subject
        sessions = discover_sessions(subject_dir, args.sessions)

        if not sessions:
            # No sessions, check for direct func directory
            func_dir = subject_dir / "func"
            if func_dir.exists():
                tasks = discover_tasks_in_directory(func_dir, args.tasks)
                for task in tasks:
                    func_pattern = (
                        f"{func_dir}/{subject}_{task}_*"
                        f"space-T1w_desc-preproc_bold.nii.gz"
                    )
                    conf_pattern = (
                        f"{func_dir}/{subject}_{task}_*desc-confounds_timeseries.tsv"
                    )

                    func_files = sorted(glob(func_pattern))
                    conf_files = sorted(glob(conf_pattern))

                    if func_files and len(func_files) == len(conf_files):
                        for func_file, conf_file in zip(func_files, conf_files):
                            processing_plan.append(
                                (subject, None, task, func_file, conf_file)
                            )
                            total_files += 1
        else:
            for session in sessions:
                session_func_dir = subject_dir / session / "func"
                if not session_func_dir.exists():
                    continue

                tasks = discover_tasks_in_directory(session_func_dir, args.tasks)
                for task in tasks:
                    func_pattern = (
                        f"{session_func_dir}/{subject}_{session}_{task}_*"
                        f"space-T1w_desc-preproc_bold.nii.gz"
                    )
                    conf_pattern = (
                        f"{session_func_dir}/{subject}_{session}_{task}_*"
                        f"desc-confounds_timeseries.tsv"
                    )

                    func_files = sorted(glob(func_pattern))
                    conf_files = sorted(glob(conf_pattern))

                    if func_files and len(func_files) == len(conf_files):
                        for func_file, conf_file in zip(func_files, conf_files):
                            processing_plan.append(
                                (subject, session, task, func_file, conf_file)
                            )
                            total_files += 1

    if not processing_plan:
        print("No files found to process")
        sys.exit(1)

    print(f"Processing plan: {total_files} files across {len(subjects)} subjects")

    if args.dry_run:
        print("\nDry run - files that would be processed:")
        for subject, session, _task, func_file, _conf_file in processing_plan:
            session_str = f"/{session}" if session else ""
            print(f"  {subject}{session_str}: {os.path.basename(func_file)}")
        print(f"\nTotal: {total_files} files")
        return

    # Process files
    processed = 0
    skipped = 0
    errors = 0

    for subject, session, _task, func_file, conf_file in tqdm(
        processing_plan, desc="Processing files", unit="file"
    ):
        # Create output directory
        if session:
            outdir = os.path.join(outdir_base, subject, session)
        else:
            outdir = os.path.join(outdir_base, subject)
        os.makedirs(outdir, exist_ok=True)

        # Generate output filename
        func_basename = os.path.basename(func_file)
        output_basename = func_basename.replace("desc-preproc_bold", "desc-tsnr")
        output_path = os.path.join(outdir, output_basename)

        # Skip if output already exists
        if os.path.exists(output_path):
            tqdm.write(f"  Skipping {output_basename} (already exists)")
            skipped += 1
            continue

        try:
            # Load data and compute tSNR
            session_str = f"/{session}" if session else ""
            tqdm.write(f"  Computing tSNR for {subject}{session_str}: {func_basename}")

            data = nib.load(func_file).get_fdata()
            conf = pd.read_csv(conf_file, sep="\t")

            tsnr_data = compute_tsnr(data, conf)

            # Save tSNR volume
            tsnr_img = nimage.new_img_like(func_file, tsnr_data)
            tsnr_img.to_filename(output_path)

            tqdm.write(f"  Saved: {output_basename}")
            processed += 1

        except Exception as e:
            tqdm.write(f"  Error processing {func_basename}: {str(e)}")
            errors += 1
            continue

    print("\nCompleted processing:")
    print(f"  Processed: {processed} files")
    print(f"  Skipped: {skipped} files (already existed)")
    print(f"  Errors: {errors} files")


if __name__ == "__main__":
    main()
