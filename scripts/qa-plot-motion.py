#!/usr/bin/env python
"""
Generate quality assurance plots for motion data.

This script creates motion trace plots, FD plots, and violin plots from fMRIprep
confounds files. It follows BIDS derivatives naming conventions and outputs to
data/derivatives/qa/motion/.
"""

import argparse
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def discover_subjects(fmriprep_dir: str, subjects: Optional[List[str]] = None) -> List[str]:
    """Discover subjects in the fMRIprep directory."""
    if subjects:
        # Validate specified subjects
        found_subjects = []
        for subj in subjects:
            if not subj.startswith("sub-"):
                subj = f"sub-{subj}"

            subj_dir = os.path.join(fmriprep_dir, subj)
            if os.path.exists(subj_dir):
                found_subjects.append(subj)
            else:
                print(f"Warning: Subject directory not found: {subj}")
        return found_subjects
    else:
        # Auto-discover all subjects
        fmriprep_path = Path(fmriprep_dir)
        if not fmriprep_path.exists():
            raise FileNotFoundError(f"fMRIprep directory not found: {fmriprep_dir}")

        subjects = sorted([
            d.name for d in fmriprep_path.iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        ])
        return subjects


def discover_sessions(subject_dir: Path) -> List[str]:
    """Discover sessions for a subject."""
    sessions = sorted([
        d.name for d in subject_dir.iterdir()
        if d.is_dir() and d.name.startswith("ses-")
    ])

    # If no sessions found, check if there are func files directly under subject
    if not sessions:
        func_dir = subject_dir / "func"
        if func_dir.exists() and list(func_dir.glob("*_desc-confounds_timeseries.tsv")):
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
        elif element.endswith(".tsv"):
            parts["extension"] = element

    return parts


def load_motion_data(confounds_file: str) -> pd.DataFrame:
    """Load motion data from fMRIprep confounds file."""
    try:
        df = pd.read_csv(confounds_file, sep='\t')

        # Extract motion columns
        motion_cols = [
            'trans_x', 'trans_y', 'trans_z',
            'rot_x', 'rot_y', 'rot_z',
            'framewise_displacement'
        ]

        # Check which columns exist
        available_cols = [col for col in motion_cols if col in df.columns]
        if not available_cols:
            print(f"Warning: No motion columns found in {confounds_file}")
            return pd.DataFrame()

        motion_df = df[available_cols].copy()

        # Add time index (assuming TR is consistent)
        motion_df['timepoint'] = range(len(motion_df))

        return motion_df

    except Exception as e:
        print(f"Error loading {confounds_file}: {str(e)}")
        return pd.DataFrame()


def create_motion_trace_plots(subject: str, session: Optional[str], confounds_files: List[str],
                             figures_dir: str) -> None:
    """Create motion trace plots for all 6 motion parameters."""
    print("  Creating motion trace plots...")

    for confounds_file in confounds_files:
        motion_data = load_motion_data(confounds_file)
        if motion_data.empty:
            continue

        parts = get_bids_filename_parts(confounds_file)

        # Create figure with 6 subplots (3x2 grid)
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        fig.suptitle(f"{subject} - Motion Parameters", fontsize=16)

        # Translation parameters (left column)
        trans_params = ['trans_x', 'trans_y', 'trans_z']
        for i, param in enumerate(trans_params):
            if param in motion_data.columns:
                axes[i, 0].plot(motion_data['timepoint'], motion_data[param], 'b-', linewidth=1)
                axes[i, 0].set_ylabel(f'{param} (mm)', fontsize=10)
                axes[i, 0].grid(True, alpha=0.3)
                axes[i, 0].set_title(f'Translation {param.split("_")[1].upper()}', fontsize=12)

        # Rotation parameters (right column)
        rot_params = ['rot_x', 'rot_y', 'rot_z']
        for i, param in enumerate(rot_params):
            if param in motion_data.columns:
                # Convert radians to degrees for better interpretation
                rot_degrees = motion_data[param] * 180 / np.pi
                axes[i, 1].plot(motion_data['timepoint'], rot_degrees, 'r-', linewidth=1)
                axes[i, 1].set_ylabel(f'{param} (degrees)', fontsize=10)
                axes[i, 1].grid(True, alpha=0.3)
                axes[i, 1].set_title(f'Rotation {param.split("_")[1].upper()}', fontsize=12)

        # Set x-axis labels
        for i in range(3):
            for j in range(2):
                if i == 2:  # Bottom row
                    axes[i, j].set_xlabel('Timepoint', fontsize=10)

        plt.tight_layout()

        # Generate BIDS-compliant output filename
        filename_parts = [subject]
        if session:
            filename_parts.append(session)
        if "task" in parts:
            filename_parts.append(f"task-{parts['task']}")
        if "run" in parts:
            filename_parts.append(f"run-{parts['run']}")
        filename_parts.extend(["desc-motion", "traces.png"])

        output_filename = "_".join(filename_parts)
        output_path = os.path.join(figures_dir, output_filename)

        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {output_filename}")


def create_fd_trace_plots(subject: str, session: Optional[str], confounds_files: List[str],
                         figures_dir: str) -> None:
    """Create framewise displacement trace plots."""
    print("  Creating FD trace plots...")

    for confounds_file in confounds_files:
        motion_data = load_motion_data(confounds_file)
        if motion_data.empty or 'framewise_displacement' not in motion_data.columns:
            continue

        parts = get_bids_filename_parts(confounds_file)

        # Create FD plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

        fd_data = motion_data['framewise_displacement'].fillna(0)  # Fill NaN with 0
        ax.plot(motion_data['timepoint'], fd_data, 'b-', linewidth=1, alpha=0.8)

        # Add 0.5mm threshold line
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.7,
                  label='0.5mm threshold')

        ax.set_xlabel('Timepoint', fontsize=12)
        ax.set_ylabel('Framewise Displacement (mm)', fontsize=12)
        ax.set_title(f"{subject} - Framewise Displacement", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add summary statistics as text
        fd_mean = fd_data.mean()
        fd_max = fd_data.max()
        fd_above_thresh = (fd_data > 0.5).sum()
        stats_text = f'Mean: {fd_mean:.3f}mm | Max: {fd_max:.3f}mm | >0.5mm: {fd_above_thresh} TRs'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top', fontsize=10)

        plt.tight_layout()

        # Generate BIDS-compliant output filename
        filename_parts = [subject]
        if session:
            filename_parts.append(session)
        if "task" in parts:
            filename_parts.append(f"task-{parts['task']}")
        if "run" in parts:
            filename_parts.append(f"run-{parts['run']}")
        filename_parts.extend(["desc-fd", "trace.png"])

        output_filename = "_".join(filename_parts)
        output_path = os.path.join(figures_dir, output_filename)

        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {output_filename}")


def create_fd_violin_plots_by_session(
    subject: str,
    confounds_files: List[str],
    sessions: List[Optional[str]],
    figures_dir: str,
) -> None:
    """Create violin plots for FD data, one plot per session."""
    print("  Creating FD violin plots by session...")

    # Group files by session
    session_data = {}
    for confounds_file, session in zip(confounds_files, sessions):
        if session not in session_data:
            session_data[session] = []
        session_data[session].append(confounds_file)

    # Create a violin plot for each session
    for session, session_files in session_data.items():
        print(f"    Creating FD violin plot for session: {session}")

        # Load FD data and create labels for this session
        run_info = []

        for confounds_file in session_files:
            motion_data = load_motion_data(confounds_file)
            if motion_data.empty or 'framewise_displacement' not in motion_data.columns:
                continue

            fd_data = motion_data['framewise_displacement'].fillna(0).values

            # Extract run and task info for labeling and sorting
            parts = get_bids_filename_parts(confounds_file)
            run_num = int(parts.get('run', '0').lstrip('0') or '0')
            task = parts.get('task', 'unknown')
            run_str = parts.get('run', '01')

            run_info.append({
                'data': fd_data,
                'run_num': run_num,
                'task': task,
                'session': session,
                'label': f"task-{task}\nrun-{run_str}"
            })

        if not run_info:
            print(f"    Warning: No valid FD data for session {session}")
            continue

        # Sort by task, then by run number
        run_info.sort(key=lambda x: (x['task'], x['run_num']))

        # Extract sorted data and labels
        fd_data_list = [info['data'] for info in run_info]
        run_labels = [info['label'] for info in run_info]

        # Add median if multiple runs
        if len(fd_data_list) > 1:
            # Calculate median across runs (per timepoint, then take all timepoints)
            min_length = min(len(data) for data in fd_data_list)
            trimmed_data = [data[:min_length] for data in fd_data_list]
            median_fd = np.median(trimmed_data, axis=0)
            fd_data_list.append(median_fd)
            run_labels.append("Median\nFD")

        # Create violin plot
        fig, ax = plt.subplots(1, 1, figsize=(max(8, len(fd_data_list) * 1.2), 6))
        positions = list(range(len(fd_data_list)))

        # Add gap before median if present
        if len(fd_data_list) > 1 and "Median" in run_labels[-1]:
            positions[-1] = positions[-2] + 1.5

        parts = ax.violinplot(fd_data_list, positions=positions, showmedians=True)

        # Style the violins
        for pc in parts['bodies']:
            pc.set_facecolor('lightcoral')
            pc.set_edgecolor('darkred')
            pc.set_alpha(0.7)

        for part_name in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
            if part_name in parts:
                parts[part_name].set_edgecolor('darkred')
                parts[part_name].set_linewidth(1.5)

        # Add threshold line
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.7,
                  label='0.5mm threshold')

        ax.set_xticks(positions)
        ax.set_xticklabels(run_labels, fontsize=10, rotation=45, ha='right')
        ax.set_ylabel('Framewise Displacement (mm)', fontsize=12)

        # Create title
        title = f"{subject}"
        if session:
            title += f" - {session}"
        title += " - FD Distribution"
        ax.set_title(title, fontsize=14, pad=20)

        # Add y-grid and legend
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend()

        sns.despine()
        plt.tight_layout()

        # Generate BIDS-compliant filename
        filename_parts = [subject]
        if session:
            filename_parts.append(session)
        filename_parts.extend(["desc-fd", "violinplot.png"])
        output_filename = "_".join(filename_parts)
        output_path = os.path.join(figures_dir, output_filename)

        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"      Saved: {output_filename}")


def process_subject(subject: str, fmriprep_dir: str, motion_qa_dir: str) -> None:
    """Process a single subject to create all motion plots."""
    print(f"Processing {subject}...")

    subject_dir = Path(fmriprep_dir) / subject
    sessions = discover_sessions(subject_dir)

    # Create output directory structure
    output_subject_dir = Path(motion_qa_dir) / subject
    figures_dir = output_subject_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Collect all confounds files across sessions
    all_confounds_files = []
    all_sessions = []

    if not sessions:
        # No sessions - process files directly under subject
        func_dir = subject_dir / "func"
        confounds_pattern = str(func_dir / "*_desc-confounds_timeseries.tsv")
        confounds_files = sorted(glob(confounds_pattern))

        if not confounds_files:
            print(f"  No confounds files found for {subject}")
            return

        all_confounds_files = confounds_files
        all_sessions = [None] * len(confounds_files)

        # Create plots
        create_motion_trace_plots(subject, None, confounds_files, str(figures_dir))
        create_fd_trace_plots(subject, None, confounds_files, str(figures_dir))

    else:
        # Process each session
        for session in sessions:
            session_dir = subject_dir / session / "func"
            confounds_pattern = str(session_dir / "*_desc-confounds_timeseries.tsv")
            confounds_files = sorted(glob(confounds_pattern))

            if not confounds_files:
                print(f"  No confounds files found for {subject} {session}")
                continue

            # Add to collection for combined violin plot
            all_confounds_files.extend(confounds_files)
            all_sessions.extend([session] * len(confounds_files))

            # Create plots for this session
            create_motion_trace_plots(subject, session, confounds_files, str(figures_dir))
            create_fd_trace_plots(subject, session, confounds_files, str(figures_dir))

    # Create violin plots by session
    if all_confounds_files:
        create_fd_violin_plots_by_session(
            subject, all_confounds_files, all_sessions, str(figures_dir)
        )


def create_group_fd_violin_plot(motion_qa_dir: str, subjects: List[str]) -> None:
    """Create group-level FD violin plot across all subjects."""
    print("Creating group-level FD violin plot...")

    subject_fd_data = []
    subject_labels = []

    for subject in subjects:
        subject_dir = Path(motion_qa_dir) / subject
        if not subject_dir.exists():
            continue

        # Find all confounds files for this subject
        confounds_files = list(subject_dir.glob("**/*_desc-confounds_timeseries.tsv"))
        if not confounds_files:
            # Look in fMRIprep directory structure
            fmriprep_dir = Path(motion_qa_dir).parent.parent / "fmriprep" / subject
            confounds_files = list(fmriprep_dir.glob("**/func/*_desc-confounds_timeseries.tsv"))

        if not confounds_files:
            continue

        # Collect FD data across all runs for this subject
        all_fd_data = []
        for confounds_file in confounds_files:
            motion_data = load_motion_data(str(confounds_file))
            if not motion_data.empty and 'framewise_displacement' in motion_data.columns:
                fd_data = motion_data['framewise_displacement'].fillna(0).values
                all_fd_data.extend(fd_data)

        if all_fd_data:
            subject_fd_data.append(all_fd_data)
            subject_labels.append(subject)

    if not subject_fd_data:
        print("  No FD data found for group plot")
        return

    # Create violin plot
    fig, ax = plt.subplots(1, 1, figsize=(max(12, len(subject_fd_data) * 0.8), 6))
    positions = list(range(len(subject_fd_data)))

    parts = ax.violinplot(subject_fd_data, positions=positions, showmedians=True)

    # Style the violins
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_edgecolor('navy')
        pc.set_alpha(0.7)

    for part_name in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
        if part_name in parts:
            parts[part_name].set_edgecolor('navy')
            parts[part_name].set_linewidth(1.5)

    # Add threshold line
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.7,
              label='0.5mm threshold')

    ax.set_xticks(positions)
    ax.set_xticklabels(subject_labels, fontsize=10, rotation=45, ha='right')
    ax.set_ylabel('Framewise Displacement (mm)', fontsize=12)
    ax.set_title('Group FD Distribution Across All Subjects', fontsize=14, pad=20)

    # Add y-grid and legend
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()

    sns.despine()
    plt.tight_layout()

    # Save plot
    output_path = os.path.join(motion_qa_dir, "group_desc-fd_violinplot.png")
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: group_desc-fd_violinplot.png")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate QA plots for motion data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subjects
  python scripts/qa-plot-motion.py

  # Process specific subjects
  python scripts/qa-plot-motion.py --subjects sub-001 sub-002

  # Use custom data directory
  python scripts/qa-plot-motion.py --data-dir /path/to/data
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

    fmriprep_dir = os.path.join(data_dir, "derivatives", "fmriprep")
    motion_qa_dir = os.path.join(data_dir, "derivatives", "qa", "motion")

    # Check if fMRIprep directory exists
    if not os.path.exists(fmriprep_dir):
        print(f"Error: fMRIprep directory not found: {fmriprep_dir}")
        print("Please ensure fMRIprep preprocessing has been completed.")
        sys.exit(1)

    # Create output directory
    os.makedirs(motion_qa_dir, exist_ok=True)

    # Discover subjects
    subjects = discover_subjects(fmriprep_dir, args.subjects)
    if not subjects:
        print("No subjects found to process")
        sys.exit(1)

    print(f"Found {len(subjects)} subjects: {', '.join(subjects)}")

    # Process each subject
    for subject in tqdm(subjects, desc="Processing subjects", unit="subject"):
        try:
            process_subject(subject, fmriprep_dir, motion_qa_dir)
        except Exception as e:
            print(f"Error processing {subject}: {str(e)}")
            continue

    # Create group-level plot
    try:
        create_group_fd_violin_plot(motion_qa_dir, subjects)
    except Exception as e:
        print(f"Error creating group plot: {str(e)}")

    print(f"\nCompleted processing {len(subjects)} subjects")
    print(f"Figures saved to: {motion_qa_dir}/*/figures/")


if __name__ == "__main__":
    main()