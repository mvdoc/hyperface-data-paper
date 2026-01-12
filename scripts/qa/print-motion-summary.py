#!/usr/bin/env python
"""Print motion summary statistics for paper.

Outputs statistics in a format suitable for copy-pasting into a paper,
including median FD, outlier percentages, and subject counts.
Statistics are computed separately for each task.

Example:
    python scripts/qa/print-motion-summary.py
    python scripts/qa/print-motion-summary.py --subjects sub-001 sub-002
"""

from pathlib import Path

import numpy as np
import pandas as pd

from hyperface.qa import (
    collect_confounds_by_task,
    create_qa_argument_parser,
    discover_subjects,
    get_config,
    get_motion_outlier_counts,
)


def compute_task_stats(
    task: str, subject_files: dict[str, list[Path]]
) -> dict[str, list[dict]]:
    """Compute motion statistics for all subjects in a task.

    Parameters
    ----------
    task : str
        Task name.
    subject_files : dict[str, list[Path]]
        Dictionary mapping subject IDs to lists of confounds files.

    Returns
    -------
    dict
        Dictionary with task name and list of per-subject stats.
    """
    subject_stats = []

    for subject, confounds_files in subject_files.items():
        all_fd_values = []
        total_outliers = 0
        total_timepoints = 0

        for confounds_file in confounds_files:
            df = pd.read_csv(confounds_file, sep="\t")
            if "framewise_displacement" in df.columns:
                fd = df["framewise_displacement"].fillna(0).values
                all_fd_values.extend(fd)

            n_outliers, n_timepoints = get_motion_outlier_counts(str(confounds_file))
            total_outliers += n_outliers
            total_timepoints += n_timepoints

        if all_fd_values and total_timepoints > 0:
            subject_stats.append(
                {
                    "subject": subject,
                    "median_fd": np.median(all_fd_values),
                    "outlier_pct": (total_outliers / total_timepoints) * 100,
                    "n_outliers": total_outliers,
                    "n_timepoints": total_timepoints,
                }
            )

    return {"task": task, "subjects": subject_stats}


def format_task_summary(task_stats: dict) -> str:
    """Format motion summary for a single task.

    Parameters
    ----------
    task_stats : dict
        Dictionary with task name and list of per-subject stats.

    Returns
    -------
    str
        Formatted summary string.
    """
    task = task_stats["task"]
    subject_stats = task_stats["subjects"]

    if not subject_stats:
        return f"Task: {task}\n  No valid subject data found.\n"

    median_fds = [s["median_fd"] for s in subject_stats]
    outlier_pcts = [s["outlier_pct"] for s in subject_stats]

    n_subjects = len(subject_stats)
    n_low_outliers = sum(1 for pct in outlier_pcts if pct < 5.0)

    lines = [
        f"Task: {task}",
        f"  Number of subjects: {n_subjects}",
        "",
        "  Framewise Displacement (FD):",
        f"    Median across subjects: {np.median(median_fds):.2f} mm",
        f"    Min median: {np.min(median_fds):.2f} mm",
        f"    Max median: {np.max(median_fds):.2f} mm",
        "",
        "  fMRIprep Motion Outliers (FD > 0.5mm OR DVARS > 1.5):",
        f"    Median across subjects: {np.median(outlier_pcts):.2f}%",
        f"    Min: {np.min(outlier_pcts):.2f}%",
        f"    Max: {np.max(outlier_pcts):.2f}%",
        f"    Subjects with <5% outliers: {n_low_outliers}/{n_subjects}",
        "",
        "  Paper-ready text:",
        f"    The median framewise displacement across subjects was "
        f"{np.median(median_fds):.2f} mm (minimum median across subjects of "
        f"{np.min(median_fds):.2f} mm, max {np.max(median_fds):.2f} mm). "
        f"Across subjects, the median percentage of volumes marked as motion "
        f"outliers by fMRIprep was {np.median(outlier_pcts):.2f}% "
        f"(min {np.min(outlier_pcts):.2f}%, max {np.max(outlier_pcts):.2f}%), "
        f"with {n_low_outliers} subjects out of {n_subjects} having less than "
        f"5% volumes marked as outliers.",
        "",
    ]

    return "\n".join(lines)


def main():
    parser = create_qa_argument_parser(
        description="Print motion summary statistics for paper",
        include_subjects=True,
    )
    args = parser.parse_args()

    config = get_config(config_path=args.config, data_dir=args.data_dir)
    fmriprep_dir = config.paths.fmriprep_dir
    motion_dir = config.paths.motion_dir
    subjects = discover_subjects(fmriprep_dir, args.subjects)

    print(f"Processing {len(subjects)} subjects...")

    # Collect confounds files organized by task
    task_files = collect_confounds_by_task(fmriprep_dir, subjects)

    if not task_files:
        print("No confounds files found.")
        return 1

    # Compute stats per task
    all_task_stats = []
    for task, subject_files in sorted(task_files.items()):
        task_stats = compute_task_stats(task, subject_files)
        all_task_stats.append(task_stats)

    # Build output
    output_lines = [
        "=" * 60,
        "MOTION SUMMARY STATISTICS",
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
    motion_dir.mkdir(parents=True, exist_ok=True)
    output_path = motion_dir / "motion_summary.txt"
    output_path.write_text(output_text)
    print(f"Saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
