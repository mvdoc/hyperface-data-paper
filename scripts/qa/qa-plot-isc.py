#!/usr/bin/env python
"""Generate ISC visualization plots.

This script loads pre-computed ISC GIFTI files and creates:
1. Violin plot showing ISC distribution across vertices for each subject
2. Pycortex surface visualizations (flatmap or inflated views)

Requires qa-save-isc.py to be run first.

Examples:
    # Generate plots for all subjects
    python scripts/qa/qa-plot-isc.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns

from hyperface.qa import create_qa_argument_parser, get_config, style_violin_plot
from hyperface.viz import create_fsaverage6_plot, has_display, start_webgl_viewer


def load_isc_data(isc_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    """Load ISC data for all subjects."""
    subject_dirs = sorted(isc_dir.glob("sub-*"))
    subject_dirs = [d for d in subject_dirs if d.is_dir()]

    if not subject_dirs:
        raise FileNotFoundError(f"No subject directories found in {isc_dir}")

    isc_data = []
    subject_ids = []

    for subj_dir in subject_dirs:
        subject_id = subj_dir.name

        fn_L = subj_dir / f"{subject_id}_hemi-L_desc-isc.func.gii"
        fn_R = subj_dir / f"{subject_id}_hemi-R_desc-isc.func.gii"

        if not fn_L.exists() or not fn_R.exists():
            print(f"Warning: Missing ISC files for {subject_id}, skipping")
            continue

        gii_L = nib.load(fn_L)
        gii_R = nib.load(fn_R)

        data_L = gii_L.darrays[0].data
        data_R = gii_R.darrays[0].data
        isc_data.append(np.concatenate([data_L, data_R]))
        subject_ids.append(subject_id)

    return isc_data, subject_ids


def create_isc_violin_plot(
    isc_data: list[np.ndarray],
    subject_ids: list[str],
    output_path: Path,
) -> None:
    """Create violin plot of ISC distribution across subjects."""
    n_subjects = len(isc_data)

    fig_width = max(12, n_subjects * 0.6)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 6))

    group_mean_isc = np.concatenate(isc_data)
    plot_data = list(isc_data) + [group_mean_isc]

    positions = list(range(n_subjects)) + [n_subjects + 0.5]
    labels = [s.replace("sub-", "") for s in subject_ids] + ["Group\nMean"]

    parts = ax.violinplot(plot_data, positions=positions, showmedians=True)
    style_violin_plot(parts, style="default")

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("ISC (correlation)", fontsize=12)
    ax.set_title("Inter-Subject Correlation Distribution", fontsize=14)
    ax.set_ylim(-0.2, 1.0)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    sns.despine()
    plt.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved violin plot to: {output_path}")


def print_summary_stats(isc_data: list[np.ndarray], subject_ids: list[str]) -> None:
    """Print summary statistics for ISC data."""
    print("\n=== ISC Summary Statistics ===\n")
    cols = ["Subject", "Mean", "Median", "Std", "Min", "Max"]
    header = f"{cols[0]:<15} " + " ".join(f"{c:>8}" for c in cols[1:])
    print(header)
    print("-" * 60)

    for subject_id, isc in zip(subject_ids, isc_data, strict=False):
        valid = isc[np.isfinite(isc)]
        vals = [valid.mean(), np.median(valid), valid.std(), valid.min(), valid.max()]
        row = f"{subject_id:<15} " + " ".join(f"{v:>8.4f}" for v in vals)
        print(row)

    all_isc = np.concatenate(isc_data)
    valid_all = all_isc[np.isfinite(all_isc)]
    print("-" * 60)
    vals = [
        valid_all.mean(),
        np.median(valid_all),
        valid_all.std(),
        valid_all.min(),
        valid_all.max(),
    ]
    row = f"{'Group':<15} " + " ".join(f"{v:>8.4f}" for v in vals)
    print(row)


def main():
    parser = create_qa_argument_parser(
        description="Generate ISC visualization plots",
        include_subjects=False,
    )
    parser.add_argument(
        "--webgl",
        action="store_true",
        help="Start interactive webgl viewer instead of generating plots",
    )
    args = parser.parse_args()

    config = get_config(config_path=args.config, data_dir=args.data_dir)
    isc_qa_dir = config.paths.isc_dir

    if not isc_qa_dir.exists():
        print(f"Error: ISC directory not found: {isc_qa_dir}")
        print("Please run qa-save-isc.py first to compute ISC.")
        return 1

    print("Loading ISC data...")
    isc_data, subject_ids = load_isc_data(isc_qa_dir)
    print(f"Loaded ISC data for {len(subject_ids)} subjects")

    group_median_isc = np.median(isc_data, axis=0)

    # If --webgl, start interactive viewer and exit
    if args.webgl:
        start_webgl_viewer(
            group_median_isc,
            freesurfer_subjects_dir=config.paths.freesurfer_dir,
        )
        return 0

    print_summary_stats(isc_data, subject_ids)

    # Create figures directory
    figures_dir = isc_qa_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    print("\nCreating violin plot...")
    violin_path = figures_dir / "group_desc-isc_violinplot.png"
    create_isc_violin_plot(isc_data, subject_ids, violin_path)

    # Determine plot type based on display availability
    display_available = has_display()
    plot_type = "inflated" if display_available else "flatmap"
    display_status = "available" if display_available else "not available"
    print(f"\nUsing {plot_type} visualization (DISPLAY={display_status})")

    # Create individual subject surface plots
    print("\nCreating individual subject surface plots...")
    for subject_id, isc in zip(subject_ids, isc_data, strict=False):
        subject_path = figures_dir / f"{subject_id}_desc-isc_{plot_type}.png"
        create_fsaverage6_plot(
            isc,
            subject_path,
            freesurfer_subjects_dir=config.paths.freesurfer_dir,
            title=subject_id.replace("sub-", ""),
        )

    # Create group median surface plot
    print("\nCreating group median surface plot...")
    surface_path = figures_dir / f"group_desc-isc_{plot_type}.png"
    create_fsaverage6_plot(
        group_median_isc,
        surface_path,
        freesurfer_subjects_dir=config.paths.freesurfer_dir,
        title="Group Median",
    )

    print(f"\nFigures saved to: {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
