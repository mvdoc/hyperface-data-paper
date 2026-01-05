#!/usr/bin/env python
"""
Generate quality assurance plots for tSNR data.

This script creates mosaic plots and violin plots from pre-computed tSNR volumes
stored in data/derivatives/qa/tsnr/. It follows BIDS derivatives naming conventions.
"""

import argparse
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns
from tqdm import tqdm

from hyperface.viz import make_mosaic, plot_mosaic


def discover_subjects(tsnr_dir: str, subjects: Optional[List[str]] = None) -> List[str]:
    """Discover subjects in the tSNR directory."""
    if subjects:
        # Validate specified subjects
        found_subjects = []
        for subj in subjects:
            if not subj.startswith("sub-"):
                subj = f"sub-{subj}"

            subj_dir = os.path.join(tsnr_dir, subj)
            if os.path.exists(subj_dir):
                found_subjects.append(subj)
            else:
                print(f"Warning: Subject directory not found: {subj}")
        return found_subjects
    else:
        # Auto-discover all subjects
        tsnr_path = Path(tsnr_dir)
        if not tsnr_path.exists():
            raise FileNotFoundError(f"tSNR directory not found: {tsnr_dir}")

        subjects = sorted([
            d.name for d in tsnr_path.iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        ])
        return subjects


def discover_sessions(subject_dir: Path) -> List[str]:
    """Discover sessions for a subject."""
    sessions = sorted([
        d.name for d in subject_dir.iterdir()
        if d.is_dir() and d.name.startswith("ses-")
    ])

    # If no sessions found, check if there are tSNR files directly under subject
    if not sessions and list(subject_dir.glob("*_desc-tsnr.nii.gz")):
        return []  # No sessions, files are directly under subject

    return sessions


def get_bids_filename_parts(filepath: str) -> dict:
    """Parse BIDS filename to extract components."""
    basename = os.path.basename(filepath)
    parts = {}

    # Split by underscores and parse key-value pairs
    elements = basename.split("_")
    for element in elements:
        if "-" in element and not element.startswith("desc-"):
            key, value = element.split("-", 1)
            parts[key] = value
        elif element.startswith("desc-"):
            parts["desc"] = element.split("-", 1)[1]
        elif element.endswith(".nii.gz"):
            parts["extension"] = element

    return parts


def create_mosaic_plots(subject: str, session: Optional[str], tsnr_files: List[str],
                       figures_dir: str, fmriprep_dir: str) -> None:
    """Create mosaic plots for tSNR data."""
    print("  Creating mosaic plots...")

    # Load tSNR data
    tsnr_data = []
    for tsnr_file in tsnr_files:
        data = nib.load(tsnr_file).get_fdata()
        tsnr_data.append(data)

    # Create mosaics for individual runs
    for tsnr_file, data in zip(tsnr_files, tsnr_data):
        parts = get_bids_filename_parts(tsnr_file)

        # Generate BIDS-compliant output filename
        filename_parts = [subject]
        if session:
            filename_parts.append(session)
        if "task" in parts:
            filename_parts.append(f"task-{parts['task']}")
        if "run" in parts:
            filename_parts.append(f"run-{parts['run']}")
        filename_parts.extend(["space-T1w", "desc-tsnr", "mosaic.png"])

        output_filename = "_".join(filename_parts)
        output_path = os.path.join(figures_dir, output_filename)

        # Create and save mosaic
        mosaic = make_mosaic(data)
        title = f"{subject}"
        if session:
            title += f" {session}"
        if "task" in parts and "run" in parts:
            title += f" task-{parts['task']} run-{parts['run']}"

        fig = plot_mosaic(mosaic, vmin=0, vmax=150, title=title)
        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {output_filename}")

    # Create median tSNR mosaic if multiple runs
    if len(tsnr_data) > 1:
        median_tsnr = np.median(tsnr_data, axis=0)
        mosaic_median = make_mosaic(median_tsnr)

        # Get task from first file for naming
        first_parts = get_bids_filename_parts(tsnr_files[0])
        filename_parts = [subject]
        if session:
            filename_parts.append(session)
        if "task" in first_parts:
            filename_parts.append(f"task-{first_parts['task']}")
        filename_parts.extend(["space-T1w", "desc-mediantsnr", "mosaic.png"])

        output_filename = "_".join(filename_parts)
        output_path = os.path.join(figures_dir, output_filename)

        title = f"{subject}"
        if session:
            title += f" {session}"
        title += " median tSNR"

        fig = plot_mosaic(mosaic_median, vmin=0, vmax=150, title=title)
        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {output_filename}")


def create_brainmask_conjunction(subject: str, session: Optional[str],
                               tsnr_files: List[str], figures_dir: str,
                               fmriprep_dir: str) -> Optional[np.ndarray]:
    """Create conjunction brain mask and visualization."""
    print("  Creating conjunction brain mask...")

    # Find corresponding brain masks in fMRIprep directory
    mask_files = []
    for tsnr_file in tsnr_files:
        # Convert tSNR filename to brain mask filename
        tsnr_basename = os.path.basename(tsnr_file)
        mask_basename = tsnr_basename.replace("desc-tsnr", "desc-brain_mask")

        # Look for mask in fMRIprep directory structure
        if session:
            mask_pattern = os.path.join(
                fmriprep_dir, subject, session, "func", mask_basename
            )
        else:
            mask_pattern = os.path.join(fmriprep_dir, subject, "func", mask_basename)

        mask_matches = glob(mask_pattern)
        if mask_matches:
            mask_files.append(mask_matches[0])
        else:
            print(f"    Warning: Brain mask not found for {tsnr_basename}")

    if not mask_files:
        print("    Error: No brain masks found, skipping conjunction mask")
        return None

    # Load first tSNR file to get shape
    reference_shape = nib.load(tsnr_files[0]).get_fdata().shape
    brainmask = np.ones(reference_shape)

    # Create conjunction mask
    for mask_file in mask_files:
        mask_data = nib.load(mask_file).get_fdata()
        brainmask *= mask_data

    # Create mosaic visualization
    mosaic_brainmask = make_mosaic(brainmask)

    # Generate BIDS-compliant filename
    filename_parts = [subject]
    if session:
        filename_parts.append(session)
    filename_parts.extend(["space-T1w", "desc-brainmask", "mosaic.png"])

    output_filename = "_".join(filename_parts)
    output_path = os.path.join(figures_dir, output_filename)

    title = f"{subject}"
    if session:
        title += f" {session}"
    title += " conjunction brainmask"

    fig = plot_mosaic(mosaic_brainmask, vmin=0, vmax=1, title=title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {output_filename}")

    return brainmask


def create_violin_plots_by_session(
    subject: str,
    tsnr_files: List[str],
    sessions: List[Optional[str]],
    brainmask: Optional[np.ndarray],
    figures_dir: str,
) -> None:
    """Create violin plots for tSNR files, one plot per session."""
    print("  Creating violin plots by session...")

    if brainmask is None:
        print("    Warning: No brain mask available, skipping violin plots")
        return

    # Group files by session
    session_data = {}
    for tsnr_file, session in zip(tsnr_files, sessions):
        if session not in session_data:
            session_data[session] = []
        session_data[session].append(tsnr_file)

    # Create a violin plot for each session
    for session, session_files in session_data.items():
        print(f"    Creating violin plot for session: {session}")

        # Load tSNR data and create labels for this session
        run_info = []

        for tsnr_file in session_files:
            data = nib.load(tsnr_file).get_fdata()
            masked_data = data[brainmask.astype(bool)]

            # Extract run and task info for labeling and sorting
            parts = get_bids_filename_parts(tsnr_file)
            run_num = int(parts.get('run', '0').lstrip('0') or '0')
            task = parts.get('task', 'unknown')
            run_str = parts.get('run', '01')

            run_info.append({
                'data': masked_data,
                'run_num': run_num,
                'task': task,
                'session': session,
                'label': f"task-{task}\nrun-{run_str}"
            })

        # Sort by task, then by run number
        run_info.sort(key=lambda x: (x['task'], x['run_num']))

        # Extract sorted data and labels
        tsnr_masked = [info['data'] for info in run_info]
        run_labels = [info['label'] for info in run_info]

        # Add median if multiple runs
        if len(tsnr_masked) > 1:
            median_tsnr = np.median(tsnr_masked, axis=0)
            tsnr_masked.append(median_tsnr)
            run_labels.append("Median\ntSNR")

        # Create violin plot
        fig, ax = plt.subplots(1, 1, figsize=(max(8, len(tsnr_masked) * 1.2), 6))
        positions = list(range(len(tsnr_masked)))

        # Add gap before median if present
        if len(tsnr_masked) > 1 and "Median" in run_labels[-1]:
            positions[-1] = positions[-2] + 1.5

        parts = ax.violinplot(tsnr_masked, positions=positions, showmedians=True)

        # Style the violins
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_edgecolor('navy')
            pc.set_alpha(0.7)

        for part_name in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
            if part_name in parts:
                parts[part_name].set_edgecolor('navy')
                parts[part_name].set_linewidth(1.5)

        ax.set_xticks(positions)
        ax.set_xticklabels(run_labels, fontsize=10, rotation=45, ha='right')
        ax.set_ylabel('tSNR', fontsize=12)

        # Create title
        title = f"{subject}"
        if session:
            title += f" - {session}"
        title += " - tSNR Distribution"
        ax.set_title(title, fontsize=14, pad=20)

        # Add y-grid
        ax.grid(True, axis='y', alpha=0.3)

        sns.despine()
        plt.tight_layout()

        # Generate BIDS-compliant filename
        filename_parts = [subject]
        if session:
            filename_parts.append(session)
        filename_parts.extend(["space-T1w", "desc-tsnr", "violinplot.png"])
        output_filename = "_".join(filename_parts)
        output_path = os.path.join(figures_dir, output_filename)

        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"      Saved: {output_filename}")


def create_group_tsnr_violin_plot(
    tsnr_dir: str, fmriprep_dir: str, subjects: List[str]
) -> None:
    """Create group-level tSNR violin plot across all subjects."""
    print("\nCreating group-level tSNR violin plot...")

    subject_tsnr_data = []
    subject_labels = []

    for subject in subjects:
        subject_dir = Path(tsnr_dir) / subject
        if not subject_dir.exists():
            continue

        # Find all tSNR files for this subject (handles sessions and no-sessions)
        tsnr_files = list(subject_dir.glob("**/*_desc-tsnr.nii.gz"))
        if not tsnr_files:
            continue

        # Find corresponding brain masks from fMRIprep
        mask_files = []
        for tsnr_file in tsnr_files:
            tsnr_basename = os.path.basename(str(tsnr_file))
            mask_basename = tsnr_basename.replace("desc-tsnr", "desc-brain_mask")

            # Search in fMRIprep directory
            fmriprep_masks = list(
                Path(fmriprep_dir).glob(f"{subject}/**/func/{mask_basename}")
            )
            if fmriprep_masks:
                mask_files.append(fmriprep_masks[0])

        if not mask_files:
            print(f"  Warning: No brain masks found for {subject}, skipping")
            continue

        # Create conjunction brain mask
        reference_shape = nib.load(tsnr_files[0]).get_fdata().shape
        brainmask = np.ones(reference_shape)
        for mask_file in mask_files:
            mask_data = nib.load(mask_file).get_fdata()
            brainmask *= mask_data

        # Load all tSNR volumes
        tsnr_volumes = []
        for tsnr_file in tsnr_files:
            data = nib.load(tsnr_file).get_fdata()
            tsnr_volumes.append(data)

        # Compute median tSNR across runs
        if len(tsnr_volumes) > 1:
            median_tsnr = np.median(tsnr_volumes, axis=0)
        else:
            median_tsnr = tsnr_volumes[0]

        # Extract masked voxels
        masked_data = median_tsnr[brainmask.astype(bool)]

        if len(masked_data) > 0:
            subject_tsnr_data.append(masked_data)
            subject_labels.append(subject)

    if not subject_tsnr_data:
        print("  No tSNR data found for group plot")
        return

    # Create violin plot
    fig, ax = plt.subplots(1, 1, figsize=(max(12, len(subject_tsnr_data) * 0.8), 6))
    positions = list(range(len(subject_tsnr_data)))

    parts = ax.violinplot(subject_tsnr_data, positions=positions, showmedians=True)

    # Style the violins
    for pc in parts["bodies"]:
        pc.set_facecolor("lightblue")
        pc.set_edgecolor("navy")
        pc.set_alpha(0.7)

    for part_name in ["cbars", "cmins", "cmaxes", "cmedians"]:
        if part_name in parts:
            parts[part_name].set_edgecolor("navy")
            parts[part_name].set_linewidth(1.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(subject_labels, fontsize=10, rotation=45, ha="right")
    ax.set_ylabel("tSNR", fontsize=12)
    ax.set_title("Group tSNR Distribution Across All Subjects", fontsize=14, pad=20)

    # Add y-grid
    ax.grid(True, axis="y", alpha=0.3)

    sns.despine()
    plt.tight_layout()

    # Save plot
    output_path = os.path.join(tsnr_dir, "group_desc-tsnr_violinplot.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: group_desc-tsnr_violinplot.png")


def process_subject(subject: str, tsnr_base_dir: str, fmriprep_dir: str) -> None:
    """Process a single subject to create all plots."""
    print(f"Processing {subject}...")

    subject_dir = Path(tsnr_base_dir) / subject
    sessions = discover_sessions(subject_dir)

    # Create single figures directory at subject level
    figures_dir = subject_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Collect all tSNR files across sessions for combined violin plot
    all_tsnr_files = []
    all_sessions = []

    if not sessions:
        # No sessions - process files directly under subject
        tsnr_pattern = str(subject_dir / "*_desc-tsnr.nii.gz")
        tsnr_files = sorted(glob(tsnr_pattern))

        if not tsnr_files:
            print(f"  No tSNR files found for {subject}")
            return

        all_tsnr_files = tsnr_files
        all_sessions = [None] * len(tsnr_files)

        # Create plots
        create_mosaic_plots(subject, None, tsnr_files, str(figures_dir), fmriprep_dir)
        brainmask = create_brainmask_conjunction(subject, None, tsnr_files,
                                               str(figures_dir), fmriprep_dir)

    else:
        # Process each session and collect files
        brainmask = None
        for session in sessions:
            session_dir = subject_dir / session
            tsnr_pattern = str(session_dir / "*_desc-tsnr.nii.gz")
            tsnr_files = sorted(glob(tsnr_pattern))

            if not tsnr_files:
                print(f"  No tSNR files found for {subject} {session}")
                continue

            # Add to collection for combined violin plot
            all_tsnr_files.extend(tsnr_files)
            all_sessions.extend([session] * len(tsnr_files))

            # Create mosaic plots for this session
            create_mosaic_plots(
                subject, session, tsnr_files, str(figures_dir), fmriprep_dir
            )

            # Create brainmask from first session (or update if None)
            if brainmask is None:
                brainmask = create_brainmask_conjunction(subject, session, tsnr_files,
                                                       str(figures_dir), fmriprep_dir)

    # Create violin plots by session
    if all_tsnr_files:
        create_violin_plots_by_session(
            subject, all_tsnr_files, all_sessions, brainmask, str(figures_dir)
        )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate QA plots for tSNR data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subjects
  python scripts/qa/qa-plot-tsnr.py

  # Process specific subjects
  python scripts/qa/qa-plot-tsnr.py --subjects sub-001 sub-002

  # Use custom data directory
  python scripts/qa/qa-plot-tsnr.py --data-dir /path/to/data
""",
    )

    parser.add_argument(
        "--subjects", nargs="+",
        help="Subject IDs to process (default: all subjects)"
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Path to BIDS dataset directory (default: data)"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Set up paths
    script_dir = os.path.dirname(__file__)
    if not os.path.isabs(args.data_dir):
        data_dir = os.path.abspath(os.path.join(script_dir, "..", args.data_dir))
    else:
        data_dir = args.data_dir

    tsnr_dir = os.path.join(data_dir, "derivatives", "qa", "tsnr")
    fmriprep_dir = os.path.join(data_dir, "derivatives", "fmriprep")

    # Check if tSNR directory exists
    if not os.path.exists(tsnr_dir):
        print(f"Error: tSNR directory not found: {tsnr_dir}")
        print("Please run qa-save-tsnr-volume.py first to compute tSNR data.")
        sys.exit(1)

    # Discover subjects
    subjects = discover_subjects(tsnr_dir, args.subjects)
    if not subjects:
        print("No subjects found to process")
        sys.exit(1)

    print(f"Found {len(subjects)} subjects: {', '.join(subjects)}")

    # Process each subject
    for subject in tqdm(subjects, desc="Processing subjects", unit="subject"):
        try:
            process_subject(subject, tsnr_dir, fmriprep_dir)
        except Exception as e:
            print(f"Error processing {subject}: {str(e)}")
            continue

    # Create group-level violin plot
    create_group_tsnr_violin_plot(tsnr_dir, fmriprep_dir, subjects)

    print(f"\nCompleted processing {len(subjects)} subjects")
    print(f"Figures saved to: {tsnr_dir}/*/figures/")
    print(f"Group plot saved to: {tsnr_dir}/group_desc-tsnr_violinplot.png")


if __name__ == "__main__":
    main()

