#!/usr/bin/env python
"""Plot task accuracy per run for each participant.

Extracts accuracy from events.tsv files (visualmemory task only) and generates:
- A bar chart showing mean accuracy per subject with individual run values as scatter
- A text summary file with accuracy statistics

Outputs:
    - desc-accuracy_barplot.png: Bar chart of accuracy per subject
    - accuracy_summary.txt: Text summary of accuracy statistics

Usage:
    python scripts/qa/qa-plot-accuracy.py
    python scripts/qa/qa-plot-accuracy.py --subjects sub-sid000005 sub-sid000009
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from hyperface.qa import create_qa_argument_parser, discover_subjects, get_config

# Plot settings
DPI = 300
PRIMARY_COLOR = "steelblue"
EDGE_COLOR = "darkslategray"
SCATTER_COLOR = "darkred"


def extract_accuracy_from_events(events_file: Path) -> int | None:
    """Extract accuracy percentage from an events.tsv file.

    Parameters
    ----------
    events_file : Path
        Path to the events.tsv file.

    Returns
    -------
    int or None
        Accuracy percentage (0-100), or None if not found.
    """
    df = pd.read_csv(events_file, sep="\t")

    # Look for accuracy_XX pattern in trial_type column
    for trial_type in df["trial_type"].values:
        if isinstance(trial_type, str) and trial_type.startswith("accuracy_"):
            match = re.match(r"accuracy_(\d+)", trial_type)
            if match:
                return int(match.group(1))

    return None


def collect_accuracy_data(
    data_dir: Path, subjects: list[str]
) -> dict[str, dict[str, int]]:
    """Collect accuracy data from events.tsv files for all subjects.

    Parameters
    ----------
    data_dir : Path
        Root BIDS data directory.
    subjects : list[str]
        List of subject IDs to process.

    Returns
    -------
    dict
        Dictionary mapping subject IDs to dict of run -> accuracy.
        Example: {"sub-001": {"run-01": 100, "run-02": 75}}
    """
    accuracy_data = {}

    for subject in sorted(subjects):
        subject_dir = data_dir / subject
        if not subject_dir.exists():
            continue

        # Find all visualmemory events files
        events_files = list(
            subject_dir.glob("ses-*/func/*_task-visualmemory_run-*_events.tsv")
        )

        if not events_files:
            continue

        subject_accuracy = {}
        for events_file in sorted(events_files):
            # Extract session and run number from filename
            ses_match = re.search(r"ses-(\d+)", events_file.name)
            run_match = re.search(r"run-(\d+)", events_file.name)
            if ses_match and run_match:
                run_id = f"ses-{ses_match.group(1)}_run-{run_match.group(1)}"
                accuracy = extract_accuracy_from_events(events_file)
                if accuracy is not None:
                    subject_accuracy[run_id] = accuracy

        if subject_accuracy:
            accuracy_data[subject] = subject_accuracy

    return accuracy_data


def plot_accuracy_figure(
    accuracy_data: dict[str, dict[str, int]], output_path: Path
) -> None:
    """Create a bar chart with scatter overlay showing accuracy per subject.

    Parameters
    ----------
    accuracy_data : dict
        Dictionary mapping subject IDs to dict of run -> accuracy.
    output_path : Path
        Path to save the figure.
    """
    # Prepare data for plotting
    subjects = sorted(accuracy_data.keys())
    mean_accuracies = []
    all_run_values = []

    for subject in subjects:
        runs = accuracy_data[subject]
        values = list(runs.values())
        mean_accuracies.append(np.mean(values))
        all_run_values.append(values)

    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(subjects) * 0.5), 6))

    # Bar plot for mean accuracy
    x_positions = np.arange(len(subjects))
    ax.bar(
        x_positions,
        mean_accuracies,
        color=PRIMARY_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=1,
        alpha=0.7,
        label="Mean accuracy",
    )

    # Scatter plot for individual run values
    for x_pos, values in zip(x_positions, all_run_values, strict=False):
        jitter = np.random.uniform(-0.15, 0.15, len(values))
        ax.scatter(
            [x_pos + j for j in jitter],
            values,
            color=SCATTER_COLOR,
            s=50,
            zorder=5,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
        )

    # Add a single scatter point to legend
    ax.scatter([], [], color=SCATTER_COLOR, s=50, label="Individual runs")

    # Styling
    ax.set_xticks(x_positions)
    # Shorten subject labels for readability
    short_labels = [s.replace("sub-sid", "s") for s in subjects]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_xlabel("Subject", fontsize=13)
    ax.set_title(
        "Task Accuracy - Visual Memory", fontsize=15, fontweight="bold", pad=12
    )
    ax.set_ylim(0, 105)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def format_accuracy_summary(accuracy_data: dict[str, dict[str, int]]) -> str:
    """Format accuracy summary text.

    Parameters
    ----------
    accuracy_data : dict
        Dictionary mapping subject IDs to dict of run -> accuracy.

    Returns
    -------
    str
        Formatted summary string.
    """
    if not accuracy_data:
        return "No accuracy data found."

    # Compute statistics
    all_values = []
    subject_means = []
    perfect_subjects = 0

    for _subject, runs in accuracy_data.items():
        values = list(runs.values())
        all_values.extend(values)
        subject_means.append(np.mean(values))
        if all(v == 100 for v in values):
            perfect_subjects += 1

    n_subjects = len(accuracy_data)
    n_runs_per_subject = len(next(iter(accuracy_data.values())))

    lines = [
        "=" * 60,
        "ACCURACY SUMMARY - Visual Memory Task",
        "=" * 60,
        "",
        f"Number of subjects: {n_subjects}",
        f"Number of runs per subject: {n_runs_per_subject}",
        "",
        "Accuracy Statistics (per-subject averages):",
        f"  Mean: {np.mean(subject_means):.1f}%",
        f"  Median: {np.median(subject_means):.1f}%",
        f"  Min: {np.min(subject_means):.1f}%",
        f"  Max: {np.max(subject_means):.1f}%",
        f"  Subjects with 100% accuracy (all runs): {perfect_subjects}/{n_subjects}",
        "",
        "Per-subject breakdown:",
    ]

    for subject in sorted(accuracy_data.keys()):
        runs = accuracy_data[subject]
        run_str = ", ".join(
            [f"{run}: {acc}%" for run, acc in sorted(runs.items())]
        )
        lines.append(f"  {subject}: {run_str}")

    lines.extend(
        [
            "",
            "-" * 60,
            "",
            "Paper-ready text:",
            f"  Participants achieved a mean accuracy of {np.mean(subject_means):.1f}% "
            f"(median {np.median(subject_means):.1f}%, "
            f"min {np.min(subject_means):.1f}%, "
            f"max {np.max(subject_means):.1f}%) on the visual memory task. "
            f"{perfect_subjects} out of {n_subjects} participants achieved "
            f"100% accuracy across all runs.",
            "",
        ]
    )

    return "\n".join(lines)


def main():
    parser = create_qa_argument_parser(
        description="Plot task accuracy per run for each participant.",
        include_subjects=True,
    )
    args = parser.parse_args()

    # Load configuration
    config = get_config(config_path=args.config, data_dir=args.data_dir)
    data_dir = config.paths.data_dir
    accuracy_dir = config.paths.accuracy_dir

    # Discover subjects from raw data directory
    subjects = discover_subjects(data_dir, args.subjects)
    print(f"Processing {len(subjects)} subjects...")

    # Collect accuracy data
    accuracy_data = collect_accuracy_data(data_dir, subjects)

    if not accuracy_data:
        print("No accuracy data found in events.tsv files.")
        return 1

    print(f"Found accuracy data for {len(accuracy_data)} subjects")

    # Create output directories
    figures_dir = accuracy_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate figure
    plot_accuracy_figure(accuracy_data, figures_dir / "desc-accuracy_barplot.png")

    # Generate and save text summary
    summary_text = format_accuracy_summary(accuracy_data)
    print(summary_text)

    summary_path = accuracy_dir / "accuracy_summary.txt"
    summary_path.write_text(summary_text)
    print(f"Saved: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
