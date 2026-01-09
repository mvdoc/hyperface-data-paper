"""Data loading utilities for the Hyperface dataset."""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

from hyperface.utils import clean_data

DEFAULT_DATA_DIR = Path("data")


def _normalize_data_dir(data_dir: Path | str | None) -> Path:
    """Normalize data_dir to Path, using default if None."""
    if data_dir is None:
        return DEFAULT_DATA_DIR
    return Path(data_dir)


def save_gifti(data: np.ndarray, output_path: str | Path) -> None:
    """Save 1D array as GIFTI func file.

    Parameters
    ----------
    data : np.ndarray
        1D array of vertex data to save
    output_path : str or Path
        Output path for the GIFTI file
    """
    darray = nib.gifti.GiftiDataArray(
        data.astype(np.float32),
        intent="NIFTI_INTENT_NONE",
        datatype="NIFTI_TYPE_FLOAT32",
    )
    gii = nib.gifti.GiftiImage(darrays=[darray])
    nib.save(gii, str(output_path))


def normalize_subject_id(subject: str | int) -> str:
    """Normalize subject identifier to full BIDS format (sub-sidXXXXXX).

    Parameters
    ----------
    subject : str or int
        Subject identifier in any of these formats:
        - int: 24 -> "sub-sid000024"
        - str without prefix: "sid000024" -> "sub-sid000024"
        - str with sub- prefix: "sub-sid000024" -> "sub-sid000024" (passthrough)

    Returns
    -------
    str
        Full BIDS subject ID in format "sub-sid000XXX"

    Raises
    ------
    ValueError
        If the subject ID format is not recognized

    Examples
    --------
    >>> normalize_subject_id(24)
    'sub-sid000024'
    >>> normalize_subject_id('sid000024')
    'sub-sid000024'
    >>> normalize_subject_id('sub-sid000024')
    'sub-sid000024'
    """
    if isinstance(subject, int):
        return f"sub-sid{subject:06d}"

    subject = str(subject)

    # Already in full format
    if subject.startswith("sub-sid"):
        return subject

    # Has sid prefix but not sub-
    if subject.startswith("sid"):
        return f"sub-{subject}"

    # Try to extract number from string
    match = re.search(r"(\d+)", subject)
    if match:
        num = int(match.group(1))
        return f"sub-sid{num:06d}"

    raise ValueError(
        f"Could not parse subject ID: {subject}. "
        "Expected int, 'sidXXXXXX', or 'sub-sidXXXXXX'"
    )


def load_run_order_config() -> dict:
    """Load ISC run order configuration from package data.

    Returns
    -------
    dict
        Configuration dictionary containing:
        - subjects: mapping of subject_nr -> subject_id
        - run_mapping: mapping of task -> presentation_run -> fmri_run
        - run_orders: per-subject session/run -> order_orig mapping

    Raises
    ------
    FileNotFoundError
        If config file not found in package
    """
    try:
        # Python 3.9+
        config_file = resources.files("hyperface.assets").joinpath(
            "visualmemory_run_order.yaml"
        )
        config_text = config_file.read_text()
    except AttributeError:
        # Python 3.8 fallback
        config_filename = "visualmemory_run_order.yaml"
        with resources.open_text("hyperface.assets", config_filename) as f:
            config_text = f.read()

    return yaml.safe_load(config_text)


def get_run_order(subject: str | int) -> dict:
    """Get the run order mapping for a subject.

    Parameters
    ----------
    subject : str or int
        Subject identifier (see normalize_subject_id for accepted formats)

    Returns
    -------
    dict
        Nested dict: session -> presentation_run -> order_orig
        e.g., {"ses-1": {1: 8, 2: 10, ...}, "ses-2": {1: 0, 2: 1, ...}}

    Raises
    ------
    KeyError
        If subject not found in configuration
    """
    config = load_run_order_config()
    subject_id = normalize_subject_id(subject)

    # Extract just the sid part (without sub-)
    sid = subject_id.replace("sub-", "")

    if sid not in config["run_orders"]:
        raise KeyError(f"Subject {sid} not found in run order configuration")

    return config["run_orders"][sid]


def _get_run_files(
    subject: str | int,
    task: str = "visualmemory",
) -> list[dict]:
    """Get list of run metadata dicts, sorted by order_orig.

    Returns list of info dicts with keys: subject, session, pres_run,
    fmri_run, order_orig, task.
    """
    config = load_run_order_config()
    subject_id = normalize_subject_id(subject)
    sid = subject_id.replace("sub-", "")

    run_mapping = config["run_mapping"].get(task, {})
    run_orders = config["run_orders"].get(sid, {})

    runs_info = []

    for session in ["ses-1", "ses-2"]:
        if session not in run_orders:
            continue

        for pres_run, order_orig in run_orders[session].items():
            fmri_run = run_mapping.get(pres_run)
            if fmri_run is None:
                continue

            info = {
                "subject": subject_id,
                "session": session,
                "pres_run": pres_run,
                "fmri_run": fmri_run,
                "order_orig": order_orig,
                "task": task,
            }
            runs_info.append(info)

    # Sort by order_orig
    runs_info.sort(key=lambda x: x["order_orig"])

    return runs_info


def _get_fmriprep_func_dir(data_dir: Path, subject_id: str, session: str) -> Path:
    """Build path to fMRIPrep func directory."""
    return data_dir / "derivatives" / "fmriprep" / subject_id / session / "func"


def _load_surface_data(func_dir: Path, fn_base: str) -> np.ndarray:
    """Load and concatenate bilateral surface data from GIFTI files."""
    fn_L = func_dir / f"{fn_base}_hemi-L_space-fsaverage6_bold.func.gii"
    fn_R = func_dir / f"{fn_base}_hemi-R_space-fsaverage6_bold.func.gii"

    gii_L = nib.load(fn_L)
    gii_R = nib.load(fn_R)

    data_L = np.vstack([d.data for d in gii_L.darrays])
    data_R = np.vstack([d.data for d in gii_R.darrays])

    return np.hstack([data_L, data_R])


def load_responses(
    subject: str | int,
    task: str = "visualmemory",
    space: str = "fsaverage6",
    data_dir: Path | str | None = None,
    clean: bool = True,
) -> list[np.ndarray]:
    """Load fMRI response data for one subject, aligned to stimulus order.

    Parameters
    ----------
    subject : str or int
        Subject identifier (see normalize_subject_id for accepted formats)
    task : str, default="visualmemory"
        Task to load. One of "visualmemory" or "localizer"
    space : str, default="fsaverage6"
        Data space to load:
        - "fsaverage6": Surface data (GIFTI)
        - "T1w": Volume data (NIfTI)
    data_dir : str or Path, optional
        Path to BIDS dataset root. If None, uses "data" relative to cwd.
    clean : bool, default=True
        If True, regress out confounds from each run before returning.
        This saves memory by not storing both raw and cleaned data.

    Returns
    -------
    list of np.ndarray
        List of 12 arrays (for visualmemory) or 4 arrays (for localizer),
        ordered by order_orig 0-11 (or 0-3 for localizer).
        For surface data: each array is (n_timepoints, n_vertices*2)
        For volume data: each array is (x, y, z, n_timepoints)

    Raises
    ------
    FileNotFoundError
        If data files not found
    ValueError
        If invalid task or space specified
    """
    data_dir = _normalize_data_dir(data_dir)
    runs_info = _get_run_files(subject, task)
    subject_id = normalize_subject_id(subject)

    # Load confounds upfront if cleaning
    confounds_list = None
    if clean:
        confounds_list = _load_tsv_files(
            subject=subject,
            task=task,
            data_dir=data_dir,
            filename_suffix="_desc-confounds_timeseries.tsv",
            use_derivatives=True,
        )

    data_list = []

    for i, info in enumerate(runs_info):
        session = info["session"]
        fmri_run = info["fmri_run"]
        func_dir = _get_fmriprep_func_dir(data_dir, subject_id, session)
        fn_base = f"{subject_id}_{session}_task-{task}_run-{fmri_run}"

        if space == "fsaverage6":
            data = _load_surface_data(func_dir, fn_base)
        elif space == "T1w":
            fn = func_dir / f"{fn_base}_space-T1w_desc-preproc_bold.nii.gz"
            img = nib.load(fn)
            data = img.get_fdata()
        else:
            raise ValueError(f"Unknown space: {space}. Expected 'fsaverage6' or 'T1w'")

        # Clean data immediately to save memory
        if clean:
            if space == "T1w":
                # Volume data: reshape (x,y,z,t) -> (t, n_voxels) for cleaning
                orig_shape = data.shape
                data = data.reshape(-1, orig_shape[-1]).T
                data = clean_data(data, confounds_list[i])
                data = data.T.reshape(orig_shape)
            else:
                data = clean_data(data, confounds_list[i])

        data_list.append(data)

    return data_list


def _load_tsv_files(
    subject: str | int,
    task: str,
    data_dir: Path,
    filename_suffix: str,
    use_derivatives: bool,
) -> list[pd.DataFrame]:
    """Load TSV files for all runs of a subject, sorted by order_orig.

    Parameters
    ----------
    subject : str or int
        Subject identifier
    task : str
        Task name
    data_dir : Path
        BIDS dataset root
    filename_suffix : str
        Suffix to append to filename base (e.g., "_desc-confounds_timeseries.tsv")
    use_derivatives : bool
        If True, load from derivatives/fmriprep; if False, load from raw BIDS

    Returns
    -------
    list of pd.DataFrame
        DataFrames for each run, ordered by order_orig
    """
    runs_info = _get_run_files(subject, task)
    subject_id = normalize_subject_id(subject)

    result = []
    for info in runs_info:
        session = info["session"]
        fmri_run = info["fmri_run"]
        fn_base = f"{subject_id}_{session}_task-{task}_run-{fmri_run}"

        if use_derivatives:
            func_dir = _get_fmriprep_func_dir(data_dir, subject_id, session)
        else:
            func_dir = data_dir / subject_id / session / "func"

        fn = func_dir / f"{fn_base}{filename_suffix}"
        result.append(pd.read_csv(fn, sep="\t"))

    return result


def load_confounds(
    subject: str | int,
    task: str = "visualmemory",
    data_dir: Path | str | None = None,
) -> list[pd.DataFrame]:
    """Load confound regressors for one subject in same order as load_responses.

    Parameters
    ----------
    subject : str or int
        Subject identifier (see normalize_subject_id for accepted formats)
    task : str, default="visualmemory"
        Task to load. One of "visualmemory" or "localizer"
    data_dir : str or Path, optional
        Path to BIDS dataset root. If None, uses "data" relative to cwd.

    Returns
    -------
    list of pd.DataFrame
        List of 12 DataFrames (for visualmemory) or 4 DataFrames (for localizer),
        ordered by order_orig 0-11 (or 0-3 for localizer).

    Raises
    ------
    FileNotFoundError
        If confound files not found
    """
    return _load_tsv_files(
        subject=subject,
        task=task,
        data_dir=_normalize_data_dir(data_dir),
        filename_suffix="_desc-confounds_timeseries.tsv",
        use_derivatives=True,
    )


def load_events(
    subject: str | int,
    task: str = "visualmemory",
    data_dir: Path | str | None = None,
) -> list[pd.DataFrame]:
    """Load events.tsv files for one subject in same order as load_responses.

    Parameters
    ----------
    subject : str or int
        Subject identifier (see normalize_subject_id for accepted formats)
    task : str, default="visualmemory"
        Task to load. One of "visualmemory" or "localizer"
    data_dir : str or Path, optional
        Path to BIDS dataset root. If None, uses "data" relative to cwd.

    Returns
    -------
    list of pd.DataFrame
        List of 12 DataFrames (for visualmemory) or 4 DataFrames (for localizer),
        ordered by order_orig 0-11 (or 0-3 for localizer).

    Raises
    ------
    FileNotFoundError
        If events files not found
    """
    return _load_tsv_files(
        subject=subject,
        task=task,
        data_dir=_normalize_data_dir(data_dir),
        filename_suffix="_events.tsv",
        use_derivatives=False,
    )
