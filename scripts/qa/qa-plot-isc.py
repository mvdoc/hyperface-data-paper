#!/usr/bin/env python
"""
Generate ISC visualization plots.

This script loads pre-computed ISC GIFTI files and creates:
1. Violin plot showing ISC distribution across vertices for each subject
2. (Future) pycortex surface visualizations

Requires qa-save-isc.py to be run first.
"""

import argparse
import os
import sys
from glob import glob
from typing import List

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns


def load_isc_data(isc_dir: str) -> tuple[List[np.ndarray], List[str]]:
    """Load ISC data for all subjects.

    Parameters
    ----------
    isc_dir : str
        Path to ISC QA directory

    Returns
    -------
    isc_data : list of np.ndarray
        ISC values for each subject (bilateral, concatenated L+R)
    subject_ids : list of str
        Subject IDs in order
    """
    # Find all subject directories
    subject_dirs = sorted(glob(os.path.join(isc_dir, "sub-*")))
    subject_dirs = [d for d in subject_dirs if os.path.isdir(d)]

    if not subject_dirs:
        raise FileNotFoundError(f"No subject directories found in {isc_dir}")

    isc_data = []
    subject_ids = []

    for subj_dir in subject_dirs:
        subject_id = os.path.basename(subj_dir)

        # Load left and right hemisphere GIFTI files
        fn_L = os.path.join(subj_dir, f"{subject_id}_hemi-L_desc-isc.func.gii")
        fn_R = os.path.join(subj_dir, f"{subject_id}_hemi-R_desc-isc.func.gii")

        if not os.path.exists(fn_L) or not os.path.exists(fn_R):
            print(f"Warning: Missing ISC files for {subject_id}, skipping")
            continue

        gii_L = nib.load(fn_L)
        gii_R = nib.load(fn_R)

        # Extract data from GIFTI and concatenate hemispheres
        data_L = gii_L.darrays[0].data
        data_R = gii_R.darrays[0].data
        isc_data.append(np.concatenate([data_L, data_R]))
        subject_ids.append(subject_id)

    return isc_data, subject_ids


def create_isc_violin_plot(
    isc_data: List[np.ndarray],
    subject_ids: List[str],
    output_path: str,
) -> None:
    """Create violin plot of ISC distribution across subjects.

    Parameters
    ----------
    isc_data : list of np.ndarray
        ISC values for each subject
    subject_ids : list of str
        Subject IDs
    output_path : str
        Path to save the figure
    """
    n_subjects = len(isc_data)

    # Create figure with adaptive width
    fig_width = max(12, n_subjects * 0.6)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 6))

    # Prepare data with group mean appended
    group_mean_isc = np.concatenate(isc_data)
    plot_data = isc_data + [group_mean_isc]

    # Positions with gap before group mean
    positions = list(range(n_subjects)) + [n_subjects + 0.5]
    labels = [s.replace("sub-", "") for s in subject_ids] + ["Group\nMean"]

    # Create violin plot
    parts = ax.violinplot(plot_data, positions=positions, showmedians=True)

    # Style the violins (consistent with other QA scripts)
    for pc in parts["bodies"]:
        pc.set_facecolor("lightblue")
        pc.set_edgecolor("navy")
        pc.set_alpha(0.7)

    for part_name in ["cbars", "cmins", "cmaxes", "cmedians"]:
        if part_name in parts:
            parts[part_name].set_edgecolor("navy")
            parts[part_name].set_linewidth(1.5)

    # Add horizontal line at zero
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    # Configure axes
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("ISC (correlation)", fontsize=12)
    ax.set_title("Inter-Subject Correlation Distribution", fontsize=14)
    ax.set_ylim(-0.2, 1.0)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    sns.despine()
    plt.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved violin plot to: {output_path}")


def print_summary_stats(isc_data: List[np.ndarray], subject_ids: List[str]) -> None:
    """Print summary statistics for ISC data."""
    print("\n=== ISC Summary Statistics ===\n")
    header = f"{'Subject':<20} {'Mean':>10} {'Median':>10} "
    header += f"{'Std':>10} {'Min':>10} {'Max':>10}"
    print(header)
    print("-" * 72)

    for subject_id, isc in zip(subject_ids, isc_data):
        valid = isc[np.isfinite(isc)]
        print(
            f"{subject_id:<20} {valid.mean():>10.4f} {np.median(valid):>10.4f} "
            f"{valid.std():>10.4f} {valid.min():>10.4f} {valid.max():>10.4f}"
        )

    # Group statistics
    all_isc = np.concatenate(isc_data)
    valid_all = all_isc[np.isfinite(all_isc)]
    print("-" * 72)
    print(
        f"{'Group':<20} {valid_all.mean():>10.4f} {np.median(valid_all):>10.4f} "
        f"{valid_all.std():>10.4f} {valid_all.min():>10.4f} {valid_all.max():>10.4f}"
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate ISC visualization plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate plots for all subjects
  python scripts/qa/qa-plot-isc.py

  # Use custom data directory
  python scripts/qa/qa-plot-isc.py --data-dir /path/to/data
""",
    )

    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to BIDS dataset directory (default: data)",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Set up paths - go up two levels from scripts/qa/ to project root
    script_dir = os.path.dirname(__file__)
    if not os.path.isabs(args.data_dir):
        data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", args.data_dir))
    else:
        data_dir = args.data_dir

    isc_qa_dir = os.path.join(data_dir, "derivatives", "qa", "isc")

    # Check if ISC directory exists
    if not os.path.exists(isc_qa_dir):
        print(f"Error: ISC directory not found: {isc_qa_dir}")
        print("Please run qa-save-isc.py first to compute ISC.")
        sys.exit(1)

    # Load ISC data
    print("Loading ISC data...")
    isc_data, subject_ids = load_isc_data(isc_qa_dir)
    print(f"Loaded ISC data for {len(subject_ids)} subjects")

    # Print summary statistics
    print_summary_stats(isc_data, subject_ids)

    # Create violin plot (saved directly in isc folder, like motion and tsnr)
    print("\nCreating violin plot...")
    violin_path = os.path.join(isc_qa_dir, "group_desc-isc_violinplot.png")
    create_isc_violin_plot(isc_data, subject_ids, violin_path)

    print(f"\nFigures saved to: {isc_qa_dir}")


if __name__ == "__main__":
    main()
