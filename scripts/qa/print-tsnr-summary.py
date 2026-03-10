#!/usr/bin/env python
"""Print tSNR summary statistics for paper.

Outputs statistics in a format suitable for copy-pasting into a paper,
including mean tSNR per subject and group statistics.
Statistics are computed separately for each task.

Example:
    python scripts/qa/print-tsnr-summary.py
    python scripts/qa/print-tsnr-summary.py --subjects sub-001 sub-002
"""

from pathlib import Path

import nibabel as nib
import numpy as np

from hyperface.qa import (
    collect_tsnr_files_by_task,
    create_qa_argument_parser,
    discover_subjects,
    get_config,
    load_subject_brainmask,
)


def compute_subject_tsnr_stats(
    subject: str,
    tsnr_files: list[Path],
    fmriprep_dir: Path,
) -> dict | None:
    """Compute mean tSNR for a subject across runs."""
    if not tsnr_files:
        return None

    brainmask = load_subject_brainmask(subject, tsnr_files, fmriprep_dir)
    if brainmask is None:
        return None

    mask_bool = brainmask.astype(bool)

    # Compute mean tSNR within mask for each run
    run_means = []
    for tsnr_file in tsnr_files:
        data = nib.load(tsnr_file).get_fdata()
        masked_data = data[mask_bool]
        run_means.append(np.mean(masked_data))

    return {
        "subject": subject,
        "mean_tsnr": np.mean(run_means),
        "n_runs": len(tsnr_files),
    }


def compute_task_stats(
    task: str,
    subject_files: dict[str, list[Path]],
    fmriprep_dir: Path,
) -> dict:
    """Compute tSNR statistics for all subjects in a task."""
    subject_stats = []

    for subject, tsnr_files in subject_files.items():
        stats = compute_subject_tsnr_stats(subject, tsnr_files, fmriprep_dir)
        if stats:
            subject_stats.append(stats)

    return {"task": task, "subjects": subject_stats}


def format_task_summary(task_stats: dict) -> str:
    """Format tSNR summary for a single task."""
    task = task_stats["task"]
    subject_stats = task_stats["subjects"]

    if not subject_stats:
        return f"Task: {task}\n  No valid subject data found.\n"

    mean_tsnrs = [s["mean_tsnr"] for s in subject_stats]
    n_subjects = len(subject_stats)

    lines = [
        f"Task: {task}",
        f"  Number of subjects: {n_subjects}",
        "",
        "  tSNR (temporal signal-to-noise ratio):",
        f"    Mean across subjects: {np.mean(mean_tsnrs):.1f}"
        f" ± {np.std(mean_tsnrs):.1f}",
        f"    Median across subjects: {np.median(mean_tsnrs):.1f}",
        f"    Min: {np.min(mean_tsnrs):.1f}",
        f"    Max: {np.max(mean_tsnrs):.1f}",
        "",
        "  Paper-ready text:",
        f"    The mean tSNR across subjects was "
        f"{np.mean(mean_tsnrs):.1f} ± {np.std(mean_tsnrs):.1f} "
        f"(median {np.median(mean_tsnrs):.1f}, "
        f"min {np.min(mean_tsnrs):.1f}, "
        f"max {np.max(mean_tsnrs):.1f}).",
        "",
    ]

    return "\n".join(lines)


def main():
    parser = create_qa_argument_parser(
        description="Print tSNR summary statistics for paper",
        include_subjects=True,
    )
    args = parser.parse_args()

    config = get_config(config_path=args.config, data_dir=args.data_dir)
    tsnr_dir = config.paths.tsnr_dir
    fmriprep_dir = config.paths.fmriprep_dir
    subjects = discover_subjects(tsnr_dir, args.subjects)

    print(f"Processing {len(subjects)} subjects...")

    # Collect tSNR files organized by task
    task_files = collect_tsnr_files_by_task(tsnr_dir, subjects)

    if not task_files:
        print("No tSNR files found.")
        return 1

    # Compute stats per task
    all_task_stats = []
    for task, subject_files in sorted(task_files.items()):
        task_stats = compute_task_stats(task, subject_files, fmriprep_dir)
        all_task_stats.append(task_stats)

    # Build output
    output_lines = [
        "=" * 60,
        "tSNR SUMMARY STATISTICS",
        "=" * 60,
        "",
    ]

    for task_stats in all_task_stats:
        output_lines.append(format_task_summary(task_stats))
        output_lines.append("-" * 60)

    output_lines.append("")
    output_text = "\n".join(output_lines)

    # Print to console
    print(output_text)

    # Save to file
    tsnr_dir.mkdir(parents=True, exist_ok=True)
    output_path = tsnr_dir / "tsnr_summary.txt"
    output_path.write_text(output_text)
    print(f"Saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
