#!/usr/bin/env python
"""Generate quality assurance plots for motion data.

This script creates motion trace plots, FD plots, and violin plots from fMRIprep
confounds files. It follows BIDS derivatives naming conventions and outputs to
data/derivatives/qa/motion/.

Examples:
    # Process all subjects
    python scripts/qa/qa-plot-motion.py

    # Process specific subjects
    python scripts/qa/qa-plot-motion.py --subjects sub-001 sub-002
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from hyperface.qa import (
    collect_confounds_by_task,
    create_qa_argument_parser,
    discover_sessions,
    discover_subjects,
    get_config,
    get_fd_outlier_counts,
    get_motion_outlier_counts,
    parse_bids_filename,
    style_violin_plot,
)


def load_motion_data(confounds_file: str) -> pd.DataFrame:
    """Load motion data from fMRIprep confounds file.

    Parameters
    ----------
    confounds_file : str
        Path to fMRIprep confounds TSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with motion columns (trans_x/y/z, rot_x/y/z,
        framewise_displacement) and a timepoint column.

    Raises
    ------
    FileNotFoundError
        If the confounds file doesn't exist.
    ValueError
        If the file is malformed or contains no motion columns.
    """
    df = pd.read_csv(confounds_file, sep="\t")
    motion_cols = [
        "trans_x",
        "trans_y",
        "trans_z",
        "rot_x",
        "rot_y",
        "rot_z",
        "framewise_displacement",
    ]
    available_cols = [col for col in motion_cols if col in df.columns]
    if not available_cols:
        raise ValueError(
            f"No motion columns found in {confounds_file}. "
            f"Expected columns like: {motion_cols[:3]}. "
            f"Found: {list(df.columns[:10])}..."
        )

    motion_df = df[available_cols].copy()
    motion_df["timepoint"] = range(len(motion_df))
    return motion_df


def create_motion_trace_plots(
    subject: str,
    session: str | None,
    confounds_files: list[str],
    figures_dir: Path,
) -> None:
    """Create motion trace plots for all 6 motion parameters."""
    print("  Creating motion trace plots...")

    for confounds_file in confounds_files:
        try:
            motion_data = load_motion_data(confounds_file)
        except (FileNotFoundError, ValueError) as e:
            print(f"    Warning: Skipping {confounds_file}: {e}")
            continue

        parts = parse_bids_filename(confounds_file)

        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        fig.suptitle(f"{subject} - Motion Parameters", fontsize=16)

        trans_params = ["trans_x", "trans_y", "trans_z"]
        for i, param in enumerate(trans_params):
            if param in motion_data.columns:
                x = motion_data["timepoint"]
                y = motion_data[param]
                axes[i, 0].plot(x, y, "b-", linewidth=1)
                axes[i, 0].set_ylabel(f"{param} (mm)", fontsize=10)
                axes[i, 0].grid(True, alpha=0.3)
                axis_label = param.split("_")[1].upper()
                axes[i, 0].set_title(f"Translation {axis_label}", fontsize=12)

        rot_params = ["rot_x", "rot_y", "rot_z"]
        for i, param in enumerate(rot_params):
            if param in motion_data.columns:
                rot_degrees = motion_data[param] * 180 / np.pi
                x = motion_data["timepoint"]
                axes[i, 1].plot(x, rot_degrees, "r-", linewidth=1)
                axes[i, 1].set_ylabel(f"{param} (degrees)", fontsize=10)
                axes[i, 1].grid(True, alpha=0.3)
                axis_label = param.split("_")[1].upper()
                axes[i, 1].set_title(f"Rotation {axis_label}", fontsize=12)

        for i in range(3):
            for j in range(2):
                if i == 2:
                    axes[i, j].set_xlabel("Timepoint", fontsize=10)

        plt.tight_layout()

        filename_parts = [subject]
        if session:
            filename_parts.append(session)
        if parts.task:
            filename_parts.append(f"task-{parts.task}")
        if parts.run:
            filename_parts.append(f"run-{parts.run}")
        filename_parts.extend(["desc-motion", "traces.png"])

        output_path = figures_dir / "_".join(filename_parts)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {output_path.name}")


def create_fd_trace_plots(
    subject: str,
    session: str | None,
    confounds_files: list[str],
    figures_dir: Path,
) -> None:
    """Create framewise displacement trace plots."""
    print("  Creating FD trace plots...")

    for confounds_file in confounds_files:
        try:
            motion_data = load_motion_data(confounds_file)
        except (FileNotFoundError, ValueError) as e:
            print(f"    Warning: Skipping {confounds_file}: {e}")
            continue
        if "framewise_displacement" not in motion_data.columns:
            continue

        parts = parse_bids_filename(confounds_file)

        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

        fd_data = motion_data["framewise_displacement"].fillna(0)
        ax.plot(motion_data["timepoint"], fd_data, "b-", linewidth=1, alpha=0.8)
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="0.5mm threshold")

        ax.set_xlabel("Timepoint", fontsize=12)
        ax.set_ylabel("Framewise Displacement (mm)", fontsize=12)
        ax.set_title(f"{subject} - Framewise Displacement", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

        fd_mean = fd_data.mean()
        fd_max = fd_data.max()
        fd_above_thresh = (fd_data > 0.5).sum()
        stats_text = (
            f"Mean: {fd_mean:.3f}mm | Max: {fd_max:.3f}mm | "
            f">0.5mm: {fd_above_thresh} TRs"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
            verticalalignment="top",
            fontsize=10,
        )

        plt.tight_layout()

        filename_parts = [subject]
        if session:
            filename_parts.append(session)
        if parts.task:
            filename_parts.append(f"task-{parts.task}")
        if parts.run:
            filename_parts.append(f"run-{parts.run}")
        filename_parts.extend(["desc-fd", "trace.png"])

        output_path = figures_dir / "_".join(filename_parts)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {output_path.name}")


def create_fd_violin_plots_by_session(
    subject: str,
    confounds_files: list[str],
    sessions: list[str | None],
    figures_dir: Path,
) -> None:
    """Create violin plots for FD data, one plot per session."""
    print("  Creating FD violin plots by session...")

    session_data: dict[str | None, list[str]] = {}
    for confounds_file, session in zip(confounds_files, sessions, strict=False):
        session_data.setdefault(session, []).append(confounds_file)

    for session, session_files in session_data.items():
        print(f"    Creating FD violin plot for session: {session}")

        run_info = []
        for confounds_file in session_files:
            try:
                motion_data = load_motion_data(confounds_file)
            except (FileNotFoundError, ValueError):
                continue
            if "framewise_displacement" not in motion_data.columns:
                continue

            fd_data = motion_data["framewise_displacement"].fillna(0).values
            parts = parse_bids_filename(confounds_file)
            run_num = int(parts.run.lstrip("0") or "0") if parts.run else 0

            run_info.append(
                {
                    "data": fd_data,
                    "run_num": run_num,
                    "task": parts.task or "unknown",
                    "run_str": parts.run or "01",
                }
            )

        if not run_info:
            print(f"    Warning: No valid FD data for session {session}")
            continue

        run_info.sort(key=lambda x: (x["task"], x["run_num"]))

        fd_data_list = [info["data"] for info in run_info]
        run_labels = [
            f"task-{info['task']}\nrun-{info['run_str']}" for info in run_info
        ]

        if len(fd_data_list) > 1:
            min_length = min(len(data) for data in fd_data_list)
            trimmed_data = [data[:min_length] for data in fd_data_list]
            median_fd = np.median(trimmed_data, axis=0)
            fd_data_list.append(median_fd)
            run_labels.append("Median\nFD")

        fig, ax = plt.subplots(1, 1, figsize=(max(8, len(fd_data_list) * 1.2), 6))
        positions = list(range(len(fd_data_list)))

        if len(fd_data_list) > 1 and "Median" in run_labels[-1]:
            positions[-1] = positions[-2] + 1.5

        violin_parts = ax.violinplot(
            fd_data_list, positions=positions, showmedians=True
        )
        style_violin_plot(violin_parts, style="fd")

        ax.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="0.5mm threshold")

        ax.set_xticks(positions)
        ax.set_xticklabels(run_labels, fontsize=14, rotation=45, ha="right")
        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylabel("Framewise Displacement (mm)", fontsize=16)

        title = subject
        if session:
            title += f" - {session}"
        title += " - FD Distribution"
        ax.set_title(title, fontsize=18, pad=20)

        ax.grid(True, axis="y", alpha=0.5)
        ax.legend()

        sns.despine()
        plt.tight_layout()

        filename_parts = [subject]
        if session:
            filename_parts.append(session)
        filename_parts.extend(["desc-fd", "violinplot.png"])

        output_path = figures_dir / "_".join(filename_parts)
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"      Saved: {output_path.name}")


def process_subject(subject: str, fmriprep_dir: Path, motion_qa_dir: Path) -> None:
    """Process a single subject to create all motion plots."""
    print(f"Processing {subject}...")

    subject_dir = fmriprep_dir / subject
    sessions = discover_sessions(subject_dir)

    figures_dir = motion_qa_dir / subject / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_confounds_files: list[str] = []
    all_sessions: list[str | None] = []

    if not sessions:
        func_dir = subject_dir / "func"
        confounds_files = sorted(func_dir.glob("*_desc-confounds_timeseries.tsv"))

        if not confounds_files:
            print(f"  No confounds files found for {subject}")
            return

        all_confounds_files = [str(f) for f in confounds_files]
        all_sessions = [None] * len(confounds_files)

        create_motion_trace_plots(subject, None, all_confounds_files, figures_dir)
        create_fd_trace_plots(subject, None, all_confounds_files, figures_dir)
    else:
        for session in sessions:
            session_dir = subject_dir / session / "func"
            pattern = "*_desc-confounds_timeseries.tsv"
            confounds_files = sorted(session_dir.glob(pattern))

            if not confounds_files:
                print(f"  No confounds files found for {subject} {session}")
                continue

            session_files = [str(f) for f in confounds_files]
            all_confounds_files.extend(session_files)
            all_sessions.extend([session] * len(confounds_files))

            create_motion_trace_plots(subject, session, session_files, figures_dir)
            create_fd_trace_plots(subject, session, session_files, figures_dir)

    if all_confounds_files:
        create_fd_violin_plots_by_session(
            subject, all_confounds_files, all_sessions, figures_dir
        )


def create_group_fd_violin_plots_by_task(
    fmriprep_dir: Path, subjects: list[str], figures_dir: Path
) -> None:
    """Create group-level FD violin plots split by task."""
    print("\nCreating group-level FD violin plots by task...")

    task_files = collect_confounds_by_task(fmriprep_dir, subjects)

    for task, subject_files in task_files.items():
        subject_labels = []
        subject_fd_data = []
        outlier_percentages = []

        for subject, confounds_files in subject_files.items():
            fd_values: list[float] = []
            total_outliers = 0
            total_timepoints = 0

            for confounds_file in confounds_files:
                try:
                    motion_data = load_motion_data(str(confounds_file))
                    n_outliers, n_timepoints = get_fd_outlier_counts(
                        str(confounds_file)
                    )
                    total_outliers += n_outliers
                    total_timepoints += n_timepoints
                except (FileNotFoundError, ValueError):
                    continue
                if "framewise_displacement" in motion_data.columns:
                    fd_values.extend(
                        motion_data["framewise_displacement"].fillna(0).values
                    )

            if fd_values:
                subject_labels.append(subject)
                subject_fd_data.append(fd_values)
                pct = (
                    (total_outliers / total_timepoints * 100)
                    if total_timepoints > 0
                    else 0
                )
                outlier_percentages.append(pct)

        if not subject_fd_data:
            continue

        # Separate main distribution (<=0.5mm) from high-motion points (>0.5mm)
        outlier_threshold = 0.5  # mm, matches motion censoring threshold
        main_data = []
        outlier_values = []
        outlier_positions = []

        for i, fd in enumerate(subject_fd_data):
            fd_array = np.array(fd)
            mask = fd_array <= outlier_threshold
            main_data.append(fd_array[mask])
            outliers = fd_array[~mask]
            if len(outliers) > 0:
                outlier_values.extend(outliers)
                outlier_positions.extend([i] * len(outliers))

        fig, ax = plt.subplots(1, 1, figsize=(max(12, len(main_data) * 0.8), 6))
        positions = list(range(len(main_data)))

        violin_parts = ax.violinplot(main_data, positions=positions, showmedians=True)
        style_violin_plot(violin_parts, style="fd")

        # Add high-motion points as individual scatter markers
        if outlier_values:
            rng = np.random.default_rng(seed=42)
            jittered_positions = [
                p + rng.uniform(-0.15, 0.15) for p in outlier_positions
            ]
            ax.scatter(
                jittered_positions,
                outlier_values,
                c="red",
                alpha=0.4,
                s=15,
                marker="o",
                label="High motion (>0.5mm)",
            )

        ax.axhline(0.5, color="red", linestyle="--", alpha=0.7)

        # Set y-axis to accommodate all data and labels
        if outlier_values:
            y_max = max(outlier_values) * 1.15
        else:
            y_max = 1.0
        ax.set_ylim(0, y_max)

        # Add FD outlier percentage labels above each violin (same y-line)
        # y_label_pos = y_max * 0.92
        # for i, pct in enumerate(outlier_percentages):
        #     ax.text(
        #         positions[i],
        #         y_label_pos,
        #         f"{pct:.1f}%",
        #         ha="center",
        #         va="top",
        #         fontsize=16,
        #     )

        ax.set_xticks(positions)
        ax.set_xticklabels(subject_labels, fontsize=14, rotation=45, ha="right")
        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylabel("Framewise Displacement (mm)", fontsize=16)
        ax.set_title(f"Group FD - task-{task}", fontsize=18)
        ax.grid(True, axis="y", alpha=0.5)
        if outlier_values:
            ax.legend(loc="upper right")

        sns.despine()
        plt.tight_layout()

        output_path = figures_dir / f"group_task-{task}_desc-fd_violinplot.png"
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_path.name}")


def create_group_motion_outlier_plots_by_task(
    fmriprep_dir: Path, subjects: list[str], figures_dir: Path
) -> None:
    """Create group-level fMRIprep outlier percentage bar plots split by task.

    Note: fMRIprep flags outliers using a joint criterion: FD > 0.5mm OR DVARS > 1.5.
    """
    print("\nCreating group-level fMRIprep outlier plots by task...")

    task_files = collect_confounds_by_task(fmriprep_dir, subjects)

    for task, subject_files in task_files.items():
        subject_labels = []
        percentages = []

        for subject, confounds_files in subject_files.items():
            total_outliers = 0
            total_timepoints = 0

            for confounds_file in confounds_files:
                try:
                    n_outliers, n_timepoints = get_motion_outlier_counts(
                        str(confounds_file)
                    )
                    total_outliers += n_outliers
                    total_timepoints += n_timepoints
                except (FileNotFoundError, ValueError):
                    continue

            if total_timepoints > 0:
                subject_labels.append(subject)
                percentages.append((total_outliers / total_timepoints) * 100)

        if not percentages:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(max(12, len(percentages) * 0.8), 6))
        positions = list(range(len(percentages)))

        ax.bar(positions, percentages, color="steelblue", edgecolor="darkslategray")
        ax.set_xticks(positions)
        ax.set_xticklabels(subject_labels, fontsize=14, rotation=45, ha="right")
        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylabel("fMRIprep Outliers (%, FD|DVARS)", fontsize=16)
        ax.set_title(f"Group fMRIprep Outliers - task-{task}", fontsize=18)
        ax.grid(True, axis="y", alpha=0.5)

        sns.despine()
        plt.tight_layout()

        output_path = figures_dir / f"group_task-{task}_desc-motionoutliers_barplot.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_path.name}")


def create_group_fd_outlier_plots_by_task(
    fmriprep_dir: Path,
    subjects: list[str],
    figures_dir: Path,
    fd_threshold: float = 0.5,
) -> None:
    """Create group-level FD outlier percentage bar plots split by task.

    This counts timepoints where FD exceeds the threshold, providing a
    motion-only metric (unlike fMRIprep outliers which also include DVARS).

    Parameters
    ----------
    fmriprep_dir : Path
        Path to fMRIprep derivatives directory.
    subjects : list[str]
        List of subject IDs to process.
    figures_dir : Path
        Output directory for figures.
    fd_threshold : float
        FD threshold in mm (default 0.5mm).
    """
    print("\nCreating group-level FD outlier plots by task...")

    task_files = collect_confounds_by_task(fmriprep_dir, subjects)

    for task, subject_files in task_files.items():
        subject_labels = []
        percentages = []

        for subject, confounds_files in subject_files.items():
            total_outliers = 0
            total_timepoints = 0

            for confounds_file in confounds_files:
                try:
                    n_outliers, n_timepoints = get_fd_outlier_counts(
                        str(confounds_file), fd_threshold=fd_threshold
                    )
                    total_outliers += n_outliers
                    total_timepoints += n_timepoints
                except (FileNotFoundError, ValueError):
                    continue

            if total_timepoints > 0:
                subject_labels.append(subject)
                percentages.append((total_outliers / total_timepoints) * 100)

        if not percentages:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(max(12, len(percentages) * 0.8), 6))
        positions = list(range(len(percentages)))

        ax.bar(positions, percentages, color="coral", edgecolor="darkred")
        ax.set_xticks(positions)
        ax.set_xticklabels(subject_labels, fontsize=14, rotation=45, ha="right")
        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylabel("FD Outliers (%)", fontsize=16)
        ax.set_title(
            f"Group FD Outliers (>{fd_threshold}mm) - task-{task}", fontsize=18
        )
        ax.grid(True, axis="y", alpha=0.5)

        sns.despine()
        plt.tight_layout()

        output_path = figures_dir / f"group_task-{task}_desc-fdoutliers_barplot.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_path.name}")


def main():
    parser = create_qa_argument_parser(
        description="Generate QA plots for motion data",
        include_subjects=True,
    )
    parser.add_argument(
        "--group-only",
        action="store_true",
        help="Only generate group-level plots, skip per-subject plots",
    )
    args = parser.parse_args()

    config = get_config(config_path=args.config, data_dir=args.data_dir)

    fmriprep_dir = config.paths.fmriprep_dir
    motion_qa_dir = config.paths.motion_dir

    if not fmriprep_dir.exists():
        print(f"Error: fMRIprep directory not found: {fmriprep_dir}")
        print("Please ensure fMRIprep preprocessing has been completed.")
        return 1

    motion_qa_dir.mkdir(parents=True, exist_ok=True)

    subjects = discover_subjects(fmriprep_dir, args.subjects)
    if not subjects:
        print("No subjects found to process")
        return 1

    print(f"Found {len(subjects)} subjects: {', '.join(subjects)}")

    if not args.group_only:
        for subject in tqdm(subjects, desc="Processing subjects", unit="subject"):
            try:
                process_subject(subject, fmriprep_dir, motion_qa_dir)
            except FileNotFoundError as e:
                print(f"Missing data for {subject}: {e}")
                continue
            except ValueError as e:
                print(f"Data error for {subject}: {e}")
                continue

    # Create group-level figures directory
    figures_dir = motion_qa_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        create_group_fd_violin_plots_by_task(fmriprep_dir, subjects, figures_dir)
        create_group_motion_outlier_plots_by_task(fmriprep_dir, subjects, figures_dir)
        create_group_fd_outlier_plots_by_task(fmriprep_dir, subjects, figures_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Warning: Could not create group plots: {e}")
        if not args.group_only:
            print("Individual subject plots were still generated successfully.")

    if args.group_only:
        print(f"\nGenerated group-level plots for {len(subjects)} subjects")
        print(f"Group figures saved to: {figures_dir}/")
    else:
        print(f"\nCompleted processing {len(subjects)} subjects")
        print(f"Subject figures saved to: {motion_qa_dir}/*/figures/")
        print(f"Group figures saved to: {figures_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
