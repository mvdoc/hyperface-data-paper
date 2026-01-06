#!/usr/bin/env python
"""
Compute and save ISC (Inter-Subject Correlation) for the visualmemory task.

This script loads fMRI data for all subjects, cleans and z-scores it,
extracts only the TRs corresponding to the main stimulus clips (excluding
buffer and catch trials), and computes leave-one-out ISC at each vertex.

Results are saved as GIFTI files in data/derivatives/qa/isc/.
"""

import argparse
import os
import sys

# Limit threads for numerical libraries to avoid oversubscription with joblib
# Must be set before importing numpy
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "3")

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from tqdm import tqdm  # noqa: E402

from hyperface import (  # noqa: E402
    compute_isc,
    get_clip_tr_mask,
    load_events,
    load_responses,
    load_run_order_config,
    zscore,
)  # noqa: E402


def save_gifti(data: np.ndarray, output_path: str) -> None:
    """Save 1D array as GIFTI func file.

    Parameters
    ----------
    data : np.ndarray
        1D array of values (e.g., ISC per vertex)
    output_path : str
        Path to save the GIFTI file
    """
    darray = nib.gifti.GiftiDataArray(
        data.astype(np.float32),
        intent="NIFTI_INTENT_NONE",
        datatype="NIFTI_TYPE_FLOAT32",
    )
    gii = nib.gifti.GiftiImage(darrays=[darray])
    nib.save(gii, output_path)


def load_and_process_subject(
    subject: str | int,
    data_dir: str,
    tr: float = 1.0,
) -> np.ndarray:
    """Load, clean, and concatenate data for one subject.

    Parameters
    ----------
    subject : str or int
        Subject identifier
    data_dir : str
        Path to BIDS dataset root
    tr : float, default=1.0
        Repetition time in seconds

    Returns
    -------
    np.ndarray
        Concatenated, cleaned, z-scored data for all runs
        Shape: (n_total_clip_trs, n_vertices)
    """
    # Load data (already cleaned by default) and events
    responses = load_responses(
        subject, task="visualmemory", space="fsaverage6", data_dir=data_dir, clean=True
    )
    events = load_events(subject, task="visualmemory", data_dir=data_dir)

    processed_runs = []

    for data, evt in zip(responses, events):
        # Create TR mask for main clips only
        n_trs = data.shape[0]
        clip_mask = get_clip_tr_mask(evt, n_trs, tr=tr)

        # Extract only clip TRs, then z-score
        data_clips = data[clip_mask]
        data_zscored = zscore(data_clips)
        processed_runs.append(data_zscored)

    # Concatenate all runs
    concatenated = np.vstack(processed_runs)
    return concatenated


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute and save ISC for visualmemory task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subjects
  python scripts/qa/qa-save-isc.py

  # Use custom data directory
  python scripts/qa/qa-save-isc.py --data-dir /path/to/data
""",
    )

    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to BIDS dataset directory (default: data)",
    )
    parser.add_argument(
        "--tr",
        type=float,
        default=1.0,
        help="Repetition time in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for loading subjects (default: -1, use all CPUs)",
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

    fmriprep_dir = os.path.join(data_dir, "derivatives", "fmriprep")
    isc_qa_dir = os.path.join(data_dir, "derivatives", "qa", "isc")

    # Check if data directory exists
    if not os.path.exists(fmriprep_dir):
        print(f"Error: fMRIprep directory not found: {fmriprep_dir}")
        print("Please ensure fMRIprep preprocessing has been completed.")
        sys.exit(1)

    # Get all subjects from config
    config = load_run_order_config()
    subject_ids = [f"sub-{sid}" for sid in config["run_orders"].keys()]
    print(f"Found {len(subject_ids)} subjects in config")

    # Create output directory
    os.makedirs(isc_qa_dir, exist_ok=True)

    # Load and process all subjects in parallel
    print(f"\nLoading and processing data for all subjects (n_jobs={args.n_jobs})...")

    subjects_data = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(load_and_process_subject)(subject_id, data_dir, tr=args.tr)
        for subject_id in subject_ids
    )

    # Verify all subjects have same number of timepoints
    n_trs_list = [d.shape[0] for d in subjects_data]
    unique_tr_counts = set(n_trs_list)
    if len(unique_tr_counts) > 1:
        print(f"\nERROR: Subjects have inconsistent TR counts: {unique_tr_counts}")
        print("TR counts per subject:")
        for subj_id, n_trs in zip(subject_ids, n_trs_list):
            print(f"  {subj_id}: {n_trs}")
        print("\nThis may indicate preprocessing failures or missing data.")
        sys.exit(1)

    print(f"\nData shape per subject: {subjects_data[0].shape}")
    print(f"Total subjects: {len(subjects_data)}")

    # Compute ISC
    print("\nComputing ISC (leave-one-out)...")
    isc_all = compute_isc(subjects_data)
    print(f"ISC shape: {isc_all.shape}")

    # Get number of vertices per hemisphere
    n_vertices_total = isc_all.shape[1]
    n_vertices_hemi = n_vertices_total // 2

    # Save per-subject ISC
    print("\nSaving ISC results...")
    subject_iter = tqdm(subject_ids, desc="Saving subjects", unit="subject")
    for i, subject_id in enumerate(subject_iter):
        subject_dir = os.path.join(isc_qa_dir, subject_id)
        os.makedirs(subject_dir, exist_ok=True)

        # Split into hemispheres
        isc_L = isc_all[i, :n_vertices_hemi]
        isc_R = isc_all[i, n_vertices_hemi:]

        # Save GIFTI files
        fn_L = os.path.join(subject_dir, f"{subject_id}_hemi-L_desc-isc.func.gii")
        fn_R = os.path.join(subject_dir, f"{subject_id}_hemi-R_desc-isc.func.gii")
        save_gifti(isc_L, fn_L)
        save_gifti(isc_R, fn_R)

    # Compute and save group mean ISC
    print("\nSaving group mean ISC...")
    isc_mean = isc_all.mean(axis=0)
    isc_mean_L = isc_mean[:n_vertices_hemi]
    isc_mean_R = isc_mean[n_vertices_hemi:]

    save_gifti(isc_mean_L, os.path.join(isc_qa_dir, "group_hemi-L_desc-isc.func.gii"))
    save_gifti(isc_mean_R, os.path.join(isc_qa_dir, "group_hemi-R_desc-isc.func.gii"))

    # Print summary statistics
    print("\n=== ISC Summary ===")
    print(f"Mean ISC across subjects and vertices: {isc_all.mean():.4f}")
    print(f"Median ISC: {np.median(isc_all):.4f}")
    print(f"Min ISC: {isc_all.min():.4f}")
    print(f"Max ISC: {isc_all.max():.4f}")
    print(f"Std ISC: {isc_all.std():.4f}")

    print(f"\nISC results saved to: {isc_qa_dir}")


if __name__ == "__main__":
    main()
