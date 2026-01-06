"""ISC (Inter-Subject Correlation) computation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import trange


def get_clip_tr_mask(
    events_df: pd.DataFrame, n_trs: int, tr: float = 1.0
) -> np.ndarray:
    """Create boolean mask for TRs corresponding to main stimulus clips.

    Includes only video clips (trial_type ends with .mp4) that are not
    catch trials (trial_type does not start with catch_).

    Parameters
    ----------
    events_df : pd.DataFrame
        Events dataframe with columns 'trial_type', 'onset', 'duration'
    n_trs : int
        Total number of TRs in the run
    tr : float, default=1.0
        Repetition time in seconds

    Returns
    -------
    np.ndarray
        Boolean mask of shape (n_trs,) where True indicates TRs to include
    """
    mask = np.zeros(n_trs, dtype=bool)

    # Filter to main clips only (not catch trials)
    is_clip = events_df["trial_type"].str.endswith(".mp4") & ~events_df[
        "trial_type"
    ].str.startswith("catch_")
    clips = events_df[is_clip]

    for _, row in clips.iterrows():
        start_tr = int(row["onset"] / tr)
        end_tr = int((row["onset"] + row["duration"]) / tr)
        # Clip to valid range
        start_tr = max(0, start_tr)
        end_tr = min(n_trs, end_tr)
        mask[start_tr:end_tr] = True

    return mask


def zscore(data: np.ndarray, copy: bool = True) -> np.ndarray:
    """Z-score data along axis 0 (time).

    Parameters
    ----------
    data : np.ndarray
        Data array of shape (n_timepoints, n_features)
    copy : bool, default=True
        If True, return a new array. If False, modify data in place.

    Returns
    -------
    np.ndarray
        Z-scored data with same shape. Vertices with zero std are left as zero.
    """
    if copy:
        data = data.copy()
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data -= mean
    nonzero_std = std != 0
    data[:, nonzero_std] /= std[nonzero_std]
    return data


def compute_isc(subjects_data: list[np.ndarray], copy: bool = False) -> np.ndarray:
    """Compute leave-one-out ISC for all subjects.

    For each subject, correlates their timeseries with the mean of all
    other subjects at each vertex/voxel.

    Parameters
    ----------
    subjects_data : list of np.ndarray
        List of N subjects, each array shape (n_timepoints, n_vertices).
        Data should already be cleaned and have matching timepoints.
    copy : bool, default=False
        If True, preserve input data. If False, z-score in place to save memory.

    Returns
    -------
    np.ndarray
        ISC values of shape (n_subjects, n_vertices)

    Notes
    -----
    Uses an efficient implementation that pre-computes the sum of all subjects
    to avoid redundant computation of group means.
    """
    n_subjects = len(subjects_data)
    n_samples, n_vertices = subjects_data[0].shape

    # Stack data and pre-compute sum for efficient leave-one-out mean
    data_stack = np.stack(subjects_data)  # (n_subjects, n_samples, n_vertices)
    data_sum = data_stack.sum(axis=0)

    correlations = np.zeros((n_subjects, n_vertices))

    for i in trange(n_subjects, desc="Computing ISC", unit="subject"):
        # Leave-one-out mean (excluding subject i)
        group_mean = (data_sum - subjects_data[i]) / (n_subjects - 1)

        # Column-wise correlation via z-score and dot product
        subj_z = zscore(subjects_data[i], copy=copy)
        group_z = zscore(group_mean, copy=False)  # group_mean is temporary
        correlations[i] = (subj_z * group_z).sum(axis=0) / n_samples

    return correlations
