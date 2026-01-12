"""tSNR utilities for the Hyperface QA pipeline.

This module provides functions for working with tSNR data, including
file organization by task and brain mask computation.
"""

from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

from hyperface.qa.bids import parse_bids_filename


def group_files_by_task(tsnr_files: list[str]) -> dict[str, list[str]]:
    """Group tSNR files by task.

    Parameters
    ----------
    tsnr_files : list[str]
        List of tSNR file paths.

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping task names to lists of file paths.
    """
    by_task: dict[str, list[str]] = defaultdict(list)
    for f in tsnr_files:
        parts = parse_bids_filename(f)
        task = parts.task or "unknown"
        by_task[task].append(f)
    return by_task


def compute_conjunction_brainmask(
    mask_files: list[Path],
    reference_shape: tuple,
) -> np.ndarray:
    """Compute conjunction brain mask from multiple mask files.

    Parameters
    ----------
    mask_files : list[Path]
        List of brain mask file paths.
    reference_shape : tuple
        Shape of the reference volume.

    Returns
    -------
    np.ndarray
        Conjunction (AND) of all brain masks.
    """
    brainmask = np.ones(reference_shape)
    for mask_file in mask_files:
        brainmask *= nib.load(mask_file).get_fdata()
    return brainmask


def load_subject_brainmask(
    subject: str,
    tsnr_files: list[Path],
    fmriprep_dir: Path,
) -> np.ndarray | None:
    """Load conjunction brain mask for a subject from fMRIPrep outputs.

    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-001').
    tsnr_files : list[Path]
        List of tSNR file paths for the subject.
    fmriprep_dir : Path
        Path to fMRIPrep derivatives directory.

    Returns
    -------
    np.ndarray | None
        Conjunction brain mask, or None if no masks found.
    """
    mask_files = []
    for tsnr_file in tsnr_files:
        mask_basename = tsnr_file.name.replace("desc-tsnr", "desc-brain_mask")
        pattern = f"{subject}/**/func/{mask_basename}"
        matches = list(fmriprep_dir.glob(pattern))
        if matches:
            mask_files.append(matches[0])

    if not mask_files:
        return None

    reference_shape = nib.load(tsnr_files[0]).get_fdata().shape
    return compute_conjunction_brainmask(mask_files, reference_shape)


def collect_tsnr_files_by_task(
    tsnr_dir: Path, subjects: list[str]
) -> dict[str, dict[str, list[Path]]]:
    """Collect tSNR files organized by task and subject.

    Parameters
    ----------
    tsnr_dir : Path
        Path to tSNR output directory.
    subjects : list[str]
        List of subject IDs to include.

    Returns
    -------
    dict[str, dict[str, list[Path]]]
        Nested dictionary: {task: {subject: [tsnr_file_paths]}}
    """
    result: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))

    for subject in subjects:
        subject_dir = tsnr_dir / subject
        if not subject_dir.exists():
            continue

        tsnr_files = list(subject_dir.glob("**/*_desc-tsnr.nii.gz"))
        for tsnr_file in tsnr_files:
            parts = parse_bids_filename(str(tsnr_file))
            task = parts.task or "unknown"
            result[task][subject].append(tsnr_file)

    return dict(result)
