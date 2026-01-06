#!/usr/bin/env python
"""Generate quality assurance plots for tSNR data.

This script creates mosaic plots and violin plots from pre-computed tSNR volumes
stored in data/derivatives/qa/tsnr/. It follows BIDS derivatives naming conventions.

Examples:
    # Process all subjects
    python scripts/qa/qa-plot-tsnr.py

    # Process specific subjects
    python scripts/qa/qa-plot-tsnr.py --subjects sub-001 sub-002
"""

import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns
from tqdm import tqdm

from hyperface.qa import (
    create_qa_argument_parser,
    discover_sessions,
    discover_subjects,
    get_config,
    parse_bids_filename,
    style_violin_plot,
)
from hyperface.viz import make_mosaic, plot_mosaic


def create_mosaic_plots(
    subject: str,
    session: str | None,
    tsnr_files: list[str],
    figures_dir: Path,
) -> None:
    """Create mosaic plots for tSNR data."""
    print("  Creating mosaic plots...")

    tsnr_data = [nib.load(f).get_fdata() for f in tsnr_files]

    for tsnr_file, data in zip(tsnr_files, tsnr_data):
        parts = parse_bids_filename(tsnr_file)

        filename_parts = [subject]
        if session:
            filename_parts.append(session)
        if parts.task:
            filename_parts.append(f"task-{parts.task}")
        if parts.run:
            filename_parts.append(f"run-{parts.run}")
        filename_parts.extend(["space-T1w", "desc-tsnr", "mosaic.png"])

        output_path = figures_dir / "_".join(filename_parts)

        mosaic = make_mosaic(data)
        title = subject
        if session:
            title += f" {session}"
        if parts.task and parts.run:
            title += f" task-{parts.task} run-{parts.run}"

        fig = plot_mosaic(mosaic, vmin=0, vmax=150, title=title)
        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {output_path.name}")

    if len(tsnr_data) > 1:
        median_tsnr = np.median(tsnr_data, axis=0)
        mosaic_median = make_mosaic(median_tsnr)

        first_parts = parse_bids_filename(tsnr_files[0])
        filename_parts = [subject]
        if session:
            filename_parts.append(session)
        if first_parts.task:
            filename_parts.append(f"task-{first_parts.task}")
        filename_parts.extend(["space-T1w", "desc-mediantsnr", "mosaic.png"])

        output_path = figures_dir / "_".join(filename_parts)
        title = f"{subject} {session or ''} median tSNR".strip()

        fig = plot_mosaic(mosaic_median, vmin=0, vmax=150, title=title)
        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {output_path.name}")


def create_brainmask_conjunction(
    subject: str,
    session: str | None,
    tsnr_files: list[str],
    figures_dir: Path,
    fmriprep_dir: Path,
) -> np.ndarray | None:
    """Create conjunction brain mask and visualization."""
    print("  Creating conjunction brain mask...")

    mask_files = []
    for tsnr_file in tsnr_files:
        tsnr_basename = os.path.basename(tsnr_file)
        mask_basename = tsnr_basename.replace("desc-tsnr", "desc-brain_mask")

        if session:
            mask_pattern = fmriprep_dir / subject / session / "func" / mask_basename
        else:
            mask_pattern = fmriprep_dir / subject / "func" / mask_basename

        matches = glob(str(mask_pattern))
        if matches:
            mask_files.append(matches[0])
        else:
            print(f"    Warning: Brain mask not found for {tsnr_basename}")

    if not mask_files:
        print("    Error: No brain masks found, skipping conjunction mask")
        return None

    reference_shape = nib.load(tsnr_files[0]).get_fdata().shape
    brainmask = np.ones(reference_shape)
    for mask_file in mask_files:
        brainmask *= nib.load(mask_file).get_fdata()

    mosaic_brainmask = make_mosaic(brainmask)

    filename_parts = [subject]
    if session:
        filename_parts.append(session)
    filename_parts.extend(["space-T1w", "desc-brainmask", "mosaic.png"])

    output_path = figures_dir / "_".join(filename_parts)
    title = f"{subject} {session or ''} conjunction brainmask".strip()

    fig = plot_mosaic(mosaic_brainmask, vmin=0, vmax=1, title=title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {output_path.name}")

    return brainmask


def create_violin_plots_by_session(
    subject: str,
    tsnr_files: list[str],
    sessions: list[str | None],
    brainmask: np.ndarray | None,
    figures_dir: Path,
) -> None:
    """Create violin plots for tSNR files, one plot per session."""
    print("  Creating violin plots by session...")

    if brainmask is None:
        print("    Warning: No brain mask available, skipping violin plots")
        return

    session_data = {}
    for tsnr_file, session in zip(tsnr_files, sessions):
        session_data.setdefault(session, []).append(tsnr_file)

    for session, session_files in session_data.items():
        print(f"    Creating violin plot for session: {session}")

        run_info = []
        for tsnr_file in session_files:
            data = nib.load(tsnr_file).get_fdata()
            masked_data = data[brainmask.astype(bool)]

            parts = parse_bids_filename(tsnr_file)
            run_num = int(parts.run.lstrip("0") or "0") if parts.run else 0

            run_info.append({
                "data": masked_data,
                "run_num": run_num,
                "task": parts.task or "unknown",
                "run_str": parts.run or "01",
            })

        run_info.sort(key=lambda x: (x["task"], x["run_num"]))

        tsnr_masked = [info["data"] for info in run_info]
        run_labels = [
            f"task-{info['task']}\nrun-{info['run_str']}" for info in run_info
        ]

        if len(tsnr_masked) > 1:
            median_tsnr = np.median(tsnr_masked, axis=0)
            tsnr_masked.append(median_tsnr)
            run_labels.append("Median\ntSNR")

        fig, ax = plt.subplots(1, 1, figsize=(max(8, len(tsnr_masked) * 1.2), 6))
        positions = list(range(len(tsnr_masked)))

        if len(tsnr_masked) > 1 and "Median" in run_labels[-1]:
            positions[-1] = positions[-2] + 1.5

        parts = ax.violinplot(tsnr_masked, positions=positions, showmedians=True)
        style_violin_plot(parts, style="default")

        ax.set_xticks(positions)
        ax.set_xticklabels(run_labels, fontsize=10, rotation=45, ha="right")
        ax.set_ylabel("tSNR", fontsize=12)

        title = subject
        if session:
            title += f" - {session}"
        title += " - tSNR Distribution"
        ax.set_title(title, fontsize=14, pad=20)
        ax.grid(True, axis="y", alpha=0.3)

        sns.despine()
        plt.tight_layout()

        filename_parts = [subject]
        if session:
            filename_parts.append(session)
        filename_parts.extend(["space-T1w", "desc-tsnr", "violinplot.png"])

        output_path = figures_dir / "_".join(filename_parts)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"      Saved: {output_path.name}")


def create_group_tsnr_violin_plot(
    tsnr_dir: Path, fmriprep_dir: Path, subjects: list[str]
) -> None:
    """Create group-level tSNR violin plot across all subjects."""
    print("\nCreating group-level tSNR violin plot...")

    subject_tsnr_data = []
    subject_labels = []

    for subject in subjects:
        subject_dir = tsnr_dir / subject
        if not subject_dir.exists():
            continue

        tsnr_files = list(subject_dir.glob("**/*_desc-tsnr.nii.gz"))
        if not tsnr_files:
            continue

        mask_files = []
        for tsnr_file in tsnr_files:
            tsnr_basename = tsnr_file.name
            mask_basename = tsnr_basename.replace("desc-tsnr", "desc-brain_mask")
            pattern = f"{subject}/**/func/{mask_basename}"
            fmriprep_masks = list(fmriprep_dir.glob(pattern))
            if fmriprep_masks:
                mask_files.append(fmriprep_masks[0])

        if not mask_files:
            print(f"  Warning: No brain masks found for {subject}, skipping")
            continue

        reference_shape = nib.load(tsnr_files[0]).get_fdata().shape
        brainmask = np.ones(reference_shape)
        for mask_file in mask_files:
            brainmask *= nib.load(mask_file).get_fdata()

        tsnr_volumes = [nib.load(f).get_fdata() for f in tsnr_files]
        if len(tsnr_volumes) > 1:
            median_tsnr = np.median(tsnr_volumes, axis=0)
        else:
            median_tsnr = tsnr_volumes[0]

        masked_data = median_tsnr[brainmask.astype(bool)]
        if len(masked_data) > 0:
            subject_tsnr_data.append(masked_data)
            subject_labels.append(subject)

    if not subject_tsnr_data:
        print("  No tSNR data found for group plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(max(12, len(subject_tsnr_data) * 0.8), 6))
    positions = list(range(len(subject_tsnr_data)))

    parts = ax.violinplot(subject_tsnr_data, positions=positions, showmedians=True)
    style_violin_plot(parts, style="default")

    ax.set_xticks(positions)
    ax.set_xticklabels(subject_labels, fontsize=10, rotation=45, ha="right")
    ax.set_ylabel("tSNR", fontsize=12)
    ax.set_title("Group tSNR Distribution Across All Subjects", fontsize=14, pad=20)
    ax.grid(True, axis="y", alpha=0.3)

    sns.despine()
    plt.tight_layout()

    output_path = tsnr_dir / "group_desc-tsnr_violinplot.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: group_desc-tsnr_violinplot.png")


def process_subject(subject: str, tsnr_dir: Path, fmriprep_dir: Path) -> None:
    """Process a single subject to create all plots."""
    print(f"Processing {subject}...")

    subject_dir = tsnr_dir / subject
    sessions = discover_sessions(subject_dir)

    figures_dir = subject_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    all_tsnr_files = []
    all_sessions: list[str | None] = []

    if not sessions:
        tsnr_files = sorted(subject_dir.glob("*_desc-tsnr.nii.gz"))
        if not tsnr_files:
            print(f"  No tSNR files found for {subject}")
            return

        all_tsnr_files = [str(f) for f in tsnr_files]
        all_sessions = [None] * len(tsnr_files)

        create_mosaic_plots(subject, None, all_tsnr_files, figures_dir)
        brainmask = create_brainmask_conjunction(
            subject, None, all_tsnr_files, figures_dir, fmriprep_dir
        )
    else:
        brainmask = None
        for session in sessions:
            session_dir = subject_dir / session
            tsnr_files = sorted(session_dir.glob("*_desc-tsnr.nii.gz"))

            if not tsnr_files:
                print(f"  No tSNR files found for {subject} {session}")
                continue

            session_files = [str(f) for f in tsnr_files]
            all_tsnr_files.extend(session_files)
            all_sessions.extend([session] * len(tsnr_files))

            create_mosaic_plots(subject, session, session_files, figures_dir)

            if brainmask is None:
                brainmask = create_brainmask_conjunction(
                    subject, session, session_files, figures_dir, fmriprep_dir
                )

    if all_tsnr_files:
        create_violin_plots_by_session(
            subject, all_tsnr_files, all_sessions, brainmask, figures_dir
        )


def main():
    parser = create_qa_argument_parser(
        description="Generate QA plots for tSNR data",
        include_subjects=True,
    )
    args = parser.parse_args()

    config = get_config(config_path=args.config, data_dir=args.data_dir)

    tsnr_dir = config.paths.tsnr_dir
    fmriprep_dir = config.paths.fmriprep_dir

    if not tsnr_dir.exists():
        print(f"Error: tSNR directory not found: {tsnr_dir}")
        print("Please run qa-save-tsnr-volume.py first to compute tSNR data.")
        return 1

    subjects = discover_subjects(tsnr_dir, args.subjects)
    if not subjects:
        print("No subjects found to process")
        return 1

    print(f"Found {len(subjects)} subjects: {', '.join(subjects)}")

    for subject in tqdm(subjects, desc="Processing subjects", unit="subject"):
        try:
            process_subject(subject, tsnr_dir, fmriprep_dir)
        except FileNotFoundError as e:
            print(f"Missing data for {subject}: {e}")
            continue
        except (nib.filebasedimages.ImageFileError, ValueError) as e:
            print(f"Data loading error for {subject}: {e}")
            print("  NIfTI files may be corrupted. Try re-running tSNR computation.")
            continue

    create_group_tsnr_violin_plot(tsnr_dir, fmriprep_dir, subjects)

    print(f"\nCompleted processing {len(subjects)} subjects")
    print(f"Figures saved to: {tsnr_dir}/*/figures/")
    print(f"Group plot saved to: {tsnr_dir}/group_desc-tsnr_violinplot.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
