#!/usr/bin/env python
"""Generate quality assurance plots for tSNR data.

This script creates mosaic plots, violin plots, and surface plots from
pre-computed tSNR data. It also saves median tSNR maps per task.

Outputs are split by task (visualmemory, localizer).

The script automatically skips outputs that already exist:
- Volume: skips subjects with existing median tSNR NIfTI files
- Group volume: skips if group violin plots exist
- Surface: skips individual subject/group plots that exist for current
  display mode (inflated with display, flatmap without)

Examples:
    # Process all subjects (auto-skips completed work)
    python scripts/qa/qa-plot-tsnr.py

    # Process specific subjects
    python scripts/qa/qa-plot-tsnr.py --subjects sub-001 sub-002

    # Force regenerate all outputs
    python scripts/qa/qa-plot-tsnr.py --force
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import nilearn.image as nimage
import numpy as np
import seaborn as sns
from tqdm import tqdm

from hyperface.io import save_gifti
from hyperface.qa import (
    build_bids_filename,
    compute_conjunction_brainmask,
    create_qa_argument_parser,
    discover_sessions,
    discover_subjects,
    get_config,
    group_files_by_task,
    load_subject_brainmask,
    parse_bids_filename,
    style_violin_plot,
)
from hyperface.viz import (
    create_fsaverage6_plot,
    has_display,
    make_mosaic,
    plot_mosaic,
)

# Consistent colormap and range for all tSNR plots (volume and surface)
TSNR_CMAP = "inferno"
TSNR_VMIN = 0
TSNR_VMAX = 200


def save_figure(fig: plt.Figure, output_path: Path, dpi: int = 150) -> None:
    """Save matplotlib figure and close it."""
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {output_path.name}")


def strip_bids_prefix(value: str, prefix: str) -> str:
    """Strip BIDS prefix from a value (e.g., 'sub-001' -> '001')."""
    return value.replace(f"{prefix}-", "")


# =============================================================================
# Volume plotting functions
# =============================================================================


def create_mosaic_plots_by_task(
    subject: str,
    session: str | None,
    tsnr_files_by_task: dict[str, list[str]],
    figures_dir: Path,
) -> None:
    """Create mosaic plots for each run and task-level median."""
    print("  Creating mosaic plots by task...")

    subj_id = strip_bids_prefix(subject, "sub")
    sess_id = strip_bids_prefix(session, "ses") if session else None

    for task, task_files in tsnr_files_by_task.items():
        for tsnr_file in task_files:
            parts = parse_bids_filename(tsnr_file)
            data = nib.load(tsnr_file).get_fdata()

            filename = build_bids_filename(
                subject=subj_id,
                session=sess_id,
                task=task,
                run=parts.run,
                space="T1w",
                desc="tsnr",
                suffix="mosaic",
                extension=".png",
            )
            output_path = figures_dir / filename

            title = f"{subject} task-{task}"
            if parts.run:
                title += f" run-{parts.run}"

            mosaic = make_mosaic(data)
            fig = plot_mosaic(mosaic, vmin=TSNR_VMIN, vmax=TSNR_VMAX, title=title)
            save_figure(fig, output_path)


def compute_and_save_volume_median_tsnr(
    subject: str,
    session: str | None,
    tsnr_files: list[str],
    task: str,
    subject_dir: Path,
    figures_dir: Path,
    reference_img_path: str,
) -> np.ndarray:
    """Compute median tSNR across runs for a task and save as NIfTI."""
    tsnr_data = [nib.load(f).get_fdata() for f in tsnr_files]
    median_tsnr = np.median(tsnr_data, axis=0)

    subj_id = strip_bids_prefix(subject, "sub")
    sess_id = strip_bids_prefix(session, "ses") if session else None

    # Save NIfTI in subject/session directory
    nifti_dir = subject_dir / session if session else subject_dir
    nifti_filename = build_bids_filename(
        subject=subj_id,
        session=sess_id,
        task=task,
        space="T1w",
        desc="mediantsnr",
        extension=".nii.gz",
    )
    nifti_path = nifti_dir / nifti_filename
    median_img = nimage.new_img_like(reference_img_path, median_tsnr)
    median_img.to_filename(str(nifti_path))
    print(f"    Saved NIfTI: {nifti_path.name}")

    # Create mosaic plot in figures dir
    mosaic_filename = build_bids_filename(
        subject=subj_id,
        session=sess_id,
        task=task,
        space="T1w",
        desc="mediantsnr",
        suffix="mosaic",
        extension=".png",
    )
    mosaic_path = figures_dir / mosaic_filename

    title = f"{subject} task-{task} median tSNR"
    mosaic = make_mosaic(median_tsnr)
    fig = plot_mosaic(mosaic, vmin=TSNR_VMIN, vmax=TSNR_VMAX, title=title)
    save_figure(fig, mosaic_path)

    return median_tsnr


def create_brainmask_conjunction(
    subject: str,
    session: str | None,
    tsnr_files: list[str],
    figures_dir: Path,
    fmriprep_dir: Path,
) -> np.ndarray | None:
    """Create conjunction brain mask and visualization."""
    print("  Creating conjunction brain mask...")

    func_dir = fmriprep_dir / subject
    if session:
        func_dir = func_dir / session
    func_dir = func_dir / "func"

    mask_files = []
    for tsnr_file in tsnr_files:
        tsnr_basename = Path(tsnr_file).name
        mask_basename = tsnr_basename.replace("desc-tsnr", "desc-brain_mask")
        mask_path = func_dir / mask_basename

        if mask_path.exists():
            mask_files.append(mask_path)
        else:
            print(f"    Warning: Brain mask not found for {tsnr_basename}")

    if not mask_files:
        print("    Error: No brain masks found, skipping conjunction mask")
        return None

    reference_shape = nib.load(tsnr_files[0]).get_fdata().shape
    brainmask = compute_conjunction_brainmask(mask_files, reference_shape)

    subj_id = strip_bids_prefix(subject, "sub")
    sess_id = strip_bids_prefix(session, "ses") if session else None

    filename = build_bids_filename(
        subject=subj_id,
        session=sess_id,
        space="T1w",
        desc="brainmask",
        suffix="mosaic",
        extension=".png",
    )
    output_path = figures_dir / filename

    title = f"{subject} {session or ''} conjunction brainmask".strip()
    fig = plot_mosaic(make_mosaic(brainmask), vmin=0, vmax=1, title=title)
    save_figure(fig, output_path)

    return brainmask


def parse_run_number(run_str: str | None) -> int:
    """Parse run string to integer for sorting."""
    if not run_str:
        return 0
    return int(run_str.lstrip("0") or "0")


def create_violin_plots_by_task(
    subject: str,
    session: str | None,
    tsnr_files_by_task: dict[str, list[str]],
    brainmask: np.ndarray | None,
    figures_dir: Path,
) -> None:
    """Create violin plots for tSNR files, one plot per task."""
    print("  Creating violin plots by task...")

    if brainmask is None:
        print("    Warning: No brain mask available, skipping violin plots")
        return

    mask_bool = brainmask.astype(bool)

    for task, task_files in tsnr_files_by_task.items():
        # Load and sort data by run number
        run_info = []
        for tsnr_file in task_files:
            parts = parse_bids_filename(tsnr_file)
            data = nib.load(tsnr_file).get_fdata()[mask_bool]
            run_info.append((parse_run_number(parts.run), parts.run or "01", data))

        run_info.sort(key=lambda x: x[0])

        tsnr_masked = [info[2] for info in run_info]
        run_labels = [f"run-{info[1]}" for info in run_info]

        # Add median if multiple runs
        if len(tsnr_masked) > 1:
            tsnr_masked.append(np.median(tsnr_masked, axis=0))
            run_labels.append("Median")

        # Create violin plot
        fig, ax = plt.subplots(1, 1, figsize=(max(8, len(tsnr_masked) * 1.2), 6))
        positions = list(range(len(tsnr_masked)))

        # Offset median position for visual separation
        if len(tsnr_masked) > 1 and run_labels[-1] == "Median":
            positions[-1] = positions[-2] + 1.5

        violin_parts = ax.violinplot(tsnr_masked, positions=positions, showmedians=True)
        style_violin_plot(violin_parts, style="default")

        ax.set_xticks(positions)
        ax.set_xticklabels(run_labels, fontsize=14, rotation=45, ha="right")
        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylabel("tSNR", fontsize=16)
        title = f"{subject} - task-{task} - tSNR Distribution"
        ax.set_title(title, fontsize=18, pad=20)
        ax.grid(True, axis="y", alpha=0.5)
        sns.despine()

        subj_id = strip_bids_prefix(subject, "sub")
        sess_id = strip_bids_prefix(session, "ses") if session else None

        filename = build_bids_filename(
            subject=subj_id,
            session=sess_id,
            task=task,
            desc="tsnr",
            suffix="violinplot",
            extension=".png",
        )
        save_figure(fig, figures_dir / filename)


def create_group_volume_plots_by_task(
    tsnr_dir: Path,
    fmriprep_dir: Path,
    subjects: list[str],
    figures_dir: Path,
) -> None:
    """Create group-level volume tSNR plots split by task."""
    print("\nCreating group-level volume tSNR plots by task...")

    # Collect data by task across all subjects
    task_data: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)

    for subject in subjects:
        subject_dir = tsnr_dir / subject
        if not subject_dir.exists():
            continue

        tsnr_files = list(subject_dir.glob("**/*_desc-tsnr.nii.gz"))
        if not tsnr_files:
            continue

        brainmask = load_subject_brainmask(subject, tsnr_files, fmriprep_dir)
        if brainmask is None:
            continue

        mask_bool = brainmask.astype(bool)
        files_by_task = group_files_by_task([str(f) for f in tsnr_files])

        for task, task_files in files_by_task.items():
            tsnr_volumes = [nib.load(f).get_fdata() for f in task_files]
            median_tsnr = np.median(tsnr_volumes, axis=0)
            masked_data = median_tsnr[mask_bool]
            if len(masked_data) > 0:
                task_data[task].append((subject, masked_data))

    # Create violin plot for each task
    for task, subject_data_list in task_data.items():
        if not subject_data_list:
            continue

        subject_labels = [s[0] for s in subject_data_list]
        subject_tsnr_data = [s[1] for s in subject_data_list]

        fig, ax = plt.subplots(1, 1, figsize=(max(12, len(subject_tsnr_data) * 0.8), 6))
        positions = list(range(len(subject_tsnr_data)))

        violin_parts = ax.violinplot(
            subject_tsnr_data, positions=positions, showmedians=True
        )
        style_violin_plot(violin_parts, style="default")

        ax.set_xticks(positions)
        ax.set_xticklabels(subject_labels, fontsize=14, rotation=45, ha="right")
        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylabel("tSNR", fontsize=16)
        ax.set_title(f"Group tSNR - task-{task}", fontsize=18, pad=20)
        ax.grid(True, axis="y", alpha=0.5)
        sns.despine()

        output_path = figures_dir / f"group_task-{task}_desc-tsnr_violinplot.png"
        save_figure(fig, output_path)


# =============================================================================
# Surface plotting functions
# =============================================================================


def load_surface_hemispheres(fn_L: Path) -> np.ndarray | None:
    """Load and concatenate left and right hemisphere surface data."""
    fn_R = Path(str(fn_L).replace("hemi-L", "hemi-R"))
    if not fn_R.exists():
        return None

    data_L = nib.load(fn_L).darrays[0].data
    data_R = nib.load(fn_R).darrays[0].data
    return np.concatenate([data_L, data_R])


def split_hemispheres(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split combined hemisphere data into left and right."""
    n_vertices_hemi = len(data) // 2
    return data[:n_vertices_hemi], data[n_vertices_hemi:]


def load_surface_tsnr_data_by_task(
    tsnr_dir: Path,
) -> dict[str, dict[str, list[np.ndarray]]]:
    """Load surface tSNR data organized by task and subject.

    Returns dict: {task: {subject_id: [run_data, ...]}}
    """
    result: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )

    subject_dirs = [d for d in sorted(tsnr_dir.glob("sub-*")) if d.is_dir()]

    for subj_dir in subject_dirs:
        subject_id = subj_dir.name
        pattern = "**/*_hemi-L_space-fsaverage6_desc-tsnr.func.gii"
        gii_files_L = sorted(subj_dir.glob(pattern))

        for fn_L in gii_files_L:
            parts = parse_bids_filename(str(fn_L))
            task = parts.task or "unknown"

            combined = load_surface_hemispheres(fn_L)
            if combined is not None:
                result[task][subject_id].append(combined)

    return dict(result)


def save_surface_gifti_pair(
    data: np.ndarray,
    output_dir: Path,
    filename_base: str,
) -> None:
    """Save combined hemisphere data as L/R GIFTI pair."""
    tsnr_L, tsnr_R = split_hemispheres(data)
    suffix = "_space-fsaverage6_desc-mediantsnr.func.gii"
    fn_L = output_dir / f"{filename_base}_hemi-L{suffix}"
    fn_R = output_dir / f"{filename_base}_hemi-R{suffix}"
    save_gifti(tsnr_L, fn_L)
    save_gifti(tsnr_R, fn_R)
    print(f"    Saved surface GIFTI: {fn_L.name}, {fn_R.name}")


def save_surface_median_tsnr(
    tsnr_runs: list[np.ndarray],
    subject_id: str,
    session: str | None,
    task: str,
    subject_dir: Path,
) -> np.ndarray:
    """Save subject median surface tSNR as GIFTI (L+R hemispheres)."""
    median_tsnr = np.median(tsnr_runs, axis=0)

    out_dir = subject_dir / session if session else subject_dir
    subj_id = strip_bids_prefix(subject_id, "sub")
    sess_id = strip_bids_prefix(session, "ses") if session else None

    # Build base filename (hemi/space/desc added by save_surface_gifti_pair)
    parts = [f"sub-{subj_id}"]
    if sess_id:
        parts.append(f"ses-{sess_id}")
    parts.append(f"task-{task}")
    filename_base = "_".join(parts)

    save_surface_gifti_pair(median_tsnr, out_dir, filename_base)
    return median_tsnr


def save_group_surface_median_tsnr(
    median_tsnr_list: list[np.ndarray],
    task: str,
    tsnr_dir: Path,
) -> np.ndarray:
    """Save group median surface tSNR as GIFTI."""
    group_median = np.median(median_tsnr_list, axis=0)
    save_surface_gifti_pair(group_median, tsnr_dir, f"group_task-{task}")
    return group_median


def create_surface_tsnr_plots_by_task(
    tsnr_data_by_task: dict[str, dict[str, list[np.ndarray]]],
    tsnr_dir: Path,
    figures_dir: Path,
    freesurfer_subjects_dir: Path | None,
    force: bool = False,
) -> None:
    """Create surface plots for each task separately.

    Skips individual subject plots and group plots that already exist
    unless force=True.
    """
    plot_type = "inflated" if has_display() else "flatmap"

    for task, subject_data in tsnr_data_by_task.items():
        if not subject_data:
            continue

        print(f"\nCreating surface tSNR plots for task-{task} ({plot_type})...")

        subject_medians = []

        for subject_id, run_list in sorted(subject_data.items()):
            subject_dir = tsnr_dir / subject_id

            # Determine session from file structure
            sessions = list(subject_dir.glob("ses-*"))
            session = sessions[0].name if sessions else None

            median_tsnr = save_surface_median_tsnr(
                run_list, subject_id, session, task, subject_dir
            )
            subject_medians.append(median_tsnr)

            # Create individual subject surface plot (skip if exists)
            if force or not subject_surface_output_exists(
                subject_id, task, figures_dir, plot_type
            ):
                fname = f"{subject_id}_task-{task}_desc-tsnr_{plot_type}.png"
                output_path = figures_dir / fname
                create_fsaverage6_plot(
                    median_tsnr,
                    output_path,
                    cmap=TSNR_CMAP,
                    vmin=TSNR_VMIN,
                    vmax=TSNR_VMAX,
                    freesurfer_subjects_dir=freesurfer_subjects_dir,
                    title=f"{subject_id.replace('sub-', '')} task-{task}",
                )
            else:
                print(f"    Skipping {subject_id}: surface plot already exists")

        # Save and plot group median (skip if exists)
        if subject_medians:
            group_median = save_group_surface_median_tsnr(
                subject_medians, task, tsnr_dir
            )

            if force or not group_surface_output_exists(task, figures_dir, plot_type):
                group_path = (
                    figures_dir / f"group_task-{task}_desc-tsnr_{plot_type}.png"
                )
                create_fsaverage6_plot(
                    group_median,
                    group_path,
                    cmap=TSNR_CMAP,
                    vmin=TSNR_VMIN,
                    vmax=TSNR_VMAX,
                    freesurfer_subjects_dir=freesurfer_subjects_dir,
                    title=f"Group Median task-{task}",
                )
            else:
                print(f"    Skipping group plot for task-{task}: already exists")


# =============================================================================
# Subject processing
# =============================================================================


def process_session_data(
    subject: str,
    session: str | None,
    tsnr_files: list[str],
    subject_dir: Path,
    figures_dir: Path,
    fmriprep_dir: Path,
    brainmask: np.ndarray | None,
) -> np.ndarray | None:
    """Process volume data for a single session (or no-session case).

    Returns the brainmask (created if not provided).
    """
    files_by_task = group_files_by_task(tsnr_files)

    create_mosaic_plots_by_task(subject, session, files_by_task, figures_dir)

    if brainmask is None:
        brainmask = create_brainmask_conjunction(
            subject, session, tsnr_files, figures_dir, fmriprep_dir
        )

    for task, task_files in files_by_task.items():
        compute_and_save_volume_median_tsnr(
            subject, session, task_files, task, subject_dir, figures_dir, task_files[0]
        )

    create_violin_plots_by_task(subject, session, files_by_task, brainmask, figures_dir)

    return brainmask


def process_subject(
    subject: str,
    tsnr_dir: Path,
    fmriprep_dir: Path,
) -> None:
    """Process a single subject to create all volume plots and save medians."""
    print(f"Processing {subject}...")

    subject_dir = tsnr_dir / subject
    sessions = discover_sessions(subject_dir)

    figures_dir = subject_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    if not sessions:
        # Single-session dataset (no ses- directories)
        tsnr_files = sorted(subject_dir.glob("*_desc-tsnr.nii.gz"))
        if not tsnr_files:
            print(f"  No tSNR files found for {subject}")
            return

        process_session_data(
            subject,
            None,
            [str(f) for f in tsnr_files],
            subject_dir,
            figures_dir,
            fmriprep_dir,
            brainmask=None,
        )
    else:
        # Multi-session dataset
        brainmask = None
        for session in sessions:
            session_dir = subject_dir / session
            tsnr_files = sorted(session_dir.glob("*_desc-tsnr.nii.gz"))

            if not tsnr_files:
                print(f"  No tSNR files found for {subject} {session}")
                continue

            brainmask = process_session_data(
                subject,
                session,
                [str(f) for f in tsnr_files],
                subject_dir,
                figures_dir,
                fmriprep_dir,
                brainmask,
            )


# =============================================================================
# Output existence checks (for skipping completed work)
# =============================================================================


def subject_volume_outputs_exist(subject: str, tsnr_dir: Path) -> bool:
    """Check if volume outputs already exist for a subject."""
    subject_dir = tsnr_dir / subject
    median_files = list(subject_dir.glob("**/*_desc-mediantsnr.nii.gz"))
    return len(median_files) > 0


def group_volume_outputs_exist(figures_dir: Path) -> bool:
    """Check if group volume violin plots exist."""
    group_plots = list(figures_dir.glob("group_task-*_violinplot.png"))
    return len(group_plots) > 0


def subject_surface_output_exists(
    subject: str, task: str, figures_dir: Path, plot_type: str
) -> bool:
    """Check if surface plot exists for a specific subject and task."""
    fname = f"{subject}_task-{task}_desc-tsnr_{plot_type}.png"
    return (figures_dir / fname).exists()


def group_surface_output_exists(task: str, figures_dir: Path, plot_type: str) -> bool:
    """Check if group surface plot exists for a task."""
    fname = f"group_task-{task}_desc-tsnr_{plot_type}.png"
    return (figures_dir / fname).exists()


def main():
    parser = create_qa_argument_parser(
        description="Generate QA plots for tSNR data",
        include_subjects=True,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of all outputs, even if they exist.",
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

    # Create global figures directory
    figures_dir = tsnr_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Process volume data for each subject (skip if outputs exist)
    subjects_to_process = []
    for subject in subjects:
        if not args.force and subject_volume_outputs_exist(subject, tsnr_dir):
            print(f"Skipping {subject}: volume outputs already exist")
        else:
            subjects_to_process.append(subject)

    if subjects_to_process:
        for subject in tqdm(
            subjects_to_process, desc="Processing subjects", unit="subject"
        ):
            try:
                process_subject(subject, tsnr_dir, fmriprep_dir)
            except FileNotFoundError as e:
                print(f"Missing data for {subject}: {e}")
                continue
            except (nib.filebasedimages.ImageFileError, ValueError) as e:
                print(f"Data loading error for {subject}: {e}")
                print(
                    "  NIfTI files may be corrupted. Try re-running tSNR computation."
                )
                continue

    # Create group-level volume plots (skip if outputs exist)
    if args.force or not group_volume_outputs_exist(figures_dir):
        create_group_volume_plots_by_task(tsnr_dir, fmriprep_dir, subjects, figures_dir)
    else:
        print("\nSkipping group volume plots: already exist")

    # Surface plotting (skips individual plots that already exist)
    print("\nChecking for surface tSNR data...")
    surface_data_by_task = load_surface_tsnr_data_by_task(tsnr_dir)

    if surface_data_by_task:
        total_subjects = sum(len(v) for v in surface_data_by_task.values())
        tasks = list(surface_data_by_task.keys())
        print(f"Found surface tSNR data: {total_subjects} subjects, tasks: {tasks}")

        create_surface_tsnr_plots_by_task(
            surface_data_by_task,
            tsnr_dir,
            figures_dir,
            config.paths.freesurfer_dir,
            force=args.force,
        )
        print(f"\nSurface plots saved to: {figures_dir}/")
    else:
        print("No surface tSNR data found.")
        print("Run qa-save-tsnr-surface.py first to compute surface tSNR.")

    print(f"\nCompleted processing {len(subjects)} subjects")
    print(f"Volume figures saved to: {tsnr_dir}/*/figures/")
    print(f"Group figures saved to: {figures_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
