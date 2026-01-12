"""Motion QA utilities for analyzing fMRIprep confounds data."""

from collections import defaultdict
from pathlib import Path

import pandas as pd

from hyperface.qa.bids import parse_bids_filename


def get_motion_outlier_counts(confounds_file: str) -> tuple[int, int]:
    """Get motion outlier count and total timepoints from confounds file.

    Note: fMRIprep flags motion outliers using a joint criterion:
    FD > 0.5mm OR standardized DVARS > 1.5.

    Parameters
    ----------
    confounds_file : str
        Path to fMRIprep confounds TSV file.

    Returns
    -------
    tuple[int, int]
        Number of motion outliers and total timepoints.
    """
    df = pd.read_csv(confounds_file, sep="\t")
    outlier_cols = [c for c in df.columns if c.startswith("motion_outlier")]
    return len(outlier_cols), len(df)


def get_fd_outlier_counts(
    confounds_file: str, fd_threshold: float = 0.5
) -> tuple[int, int]:
    """Count timepoints where FD exceeds threshold.

    Parameters
    ----------
    confounds_file : str
        Path to fMRIprep confounds TSV file.
    fd_threshold : float
        FD threshold in mm (default 0.5mm).

    Returns
    -------
    tuple[int, int]
        Number of FD outliers and total timepoints.
    """
    df = pd.read_csv(confounds_file, sep="\t")
    fd = df["framewise_displacement"].fillna(0)
    n_outliers = int((fd > fd_threshold).sum())
    return n_outliers, len(df)


def collect_confounds_by_task(
    fmriprep_dir: Path, subjects: list[str]
) -> dict[str, dict[str, list[Path]]]:
    """Collect confounds files organized by task and subject.

    Parameters
    ----------
    fmriprep_dir : Path
        Path to fMRIprep derivatives directory.
    subjects : list[str]
        List of subject IDs to process.

    Returns
    -------
    dict[str, dict[str, list[Path]]]
        Nested dict: task -> subject -> list of confounds file paths.
    """
    task_subject_files: dict[str, dict[str, list[Path]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for subject in subjects:
        subject_fmriprep = fmriprep_dir / subject
        pattern = "**/func/*_desc-confounds_timeseries.tsv"

        for confounds_file in subject_fmriprep.glob(pattern):
            parts = parse_bids_filename(str(confounds_file))
            task = parts.task or "unknown"
            task_subject_files[task][subject].append(confounds_file)

    return dict(task_subject_files)
