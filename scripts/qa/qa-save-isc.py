#!/usr/bin/env python
"""Compute and save ISC (Inter-Subject Correlation) for the visualmemory task.

This script loads fMRI data for all subjects, cleans and z-scores it,
extracts only the TRs corresponding to the main stimulus clips (excluding
buffer and catch trials), and computes leave-one-out ISC at each vertex.

Results are saved as GIFTI files in data/derivatives/qa/isc/.

Examples:
    # Process all subjects
    python scripts/qa/qa-save-isc.py

    # Use custom data directory
    python scripts/qa/qa-save-isc.py --data-dir /path/to/data
"""

import argparse
import os

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
)
from hyperface.qa import get_config  # noqa: E402


def save_gifti(data: np.ndarray, output_path: str) -> None:
    """Save 1D array as GIFTI func file."""
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
    """Load, clean, and concatenate data for one subject."""
    responses = load_responses(
        subject, task="visualmemory", space="fsaverage6",
        data_dir=data_dir, clean=True,
    )
    events = load_events(subject, task="visualmemory", data_dir=data_dir)

    processed_runs = []
    for data, evt in zip(responses, events):
        n_trs = data.shape[0]
        clip_mask = get_clip_tr_mask(evt, n_trs, tr=tr)
        data_clips = data[clip_mask]
        data_zscored = zscore(data_clips)
        processed_runs.append(data_zscored)

    return np.vstack(processed_runs)


def main():
    parser = argparse.ArgumentParser(
        description="Compute and save ISC for visualmemory task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--data-dir", help="Override data directory")
    parser.add_argument("--tr", type=float, default=1.0, help="TR in seconds")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs")
    args = parser.parse_args()

    config = get_config(config_path=args.config, data_dir=args.data_dir)
    data_dir = config.paths.data_dir
    isc_qa_dir = config.paths.isc_dir

    if not config.paths.fmriprep_dir.exists():
        print(f"Error: fMRIprep directory not found: {config.paths.fmriprep_dir}")
        return 1

    # Get all subjects from config
    run_config = load_run_order_config()
    subject_ids = [f"sub-{sid}" for sid in run_config["run_orders"].keys()]
    print(f"Found {len(subject_ids)} subjects in config")

    isc_qa_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading and processing data for all subjects (n_jobs={args.n_jobs})...")
    subjects_data = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(load_and_process_subject)(subject_id, str(data_dir), tr=args.tr)
        for subject_id in subject_ids
    )

    # Verify consistent TR counts
    n_trs_list = [d.shape[0] for d in subjects_data]
    if len(set(n_trs_list)) > 1:
        print(f"\nERROR: Inconsistent TR counts: {set(n_trs_list)}")
        for subj_id, n_trs in zip(subject_ids, n_trs_list):
            print(f"  {subj_id}: {n_trs}")
        return 1

    print(f"\nData shape per subject: {subjects_data[0].shape}")
    print(f"Total subjects: {len(subjects_data)}")

    print("\nComputing ISC (leave-one-out)...")
    isc_all = compute_isc(subjects_data)
    print(f"ISC shape: {isc_all.shape}")

    n_vertices_hemi = isc_all.shape[1] // 2

    print("\nSaving ISC results...")
    for i, subject_id in enumerate(tqdm(subject_ids, desc="Saving subjects")):
        subject_dir = isc_qa_dir / subject_id
        subject_dir.mkdir(exist_ok=True)

        isc_L = isc_all[i, :n_vertices_hemi]
        isc_R = isc_all[i, n_vertices_hemi:]

        fn_L = str(subject_dir / f"{subject_id}_hemi-L_desc-isc.func.gii")
        fn_R = str(subject_dir / f"{subject_id}_hemi-R_desc-isc.func.gii")
        save_gifti(isc_L, fn_L)
        save_gifti(isc_R, fn_R)

    print("\nSaving group mean ISC...")
    isc_mean = isc_all.mean(axis=0)
    fn_L = str(isc_qa_dir / "group_hemi-L_desc-isc.func.gii")
    fn_R = str(isc_qa_dir / "group_hemi-R_desc-isc.func.gii")
    save_gifti(isc_mean[:n_vertices_hemi], fn_L)
    save_gifti(isc_mean[n_vertices_hemi:], fn_R)

    print("\n=== ISC Summary ===")
    print(f"Mean ISC: {isc_all.mean():.4f}")
    print(f"Median ISC: {np.median(isc_all):.4f}")
    print(f"Min ISC: {isc_all.min():.4f}")
    print(f"Max ISC: {isc_all.max():.4f}")
    print(f"\nResults saved to: {isc_qa_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
