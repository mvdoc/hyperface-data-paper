"""BIDS filename parsing and construction utilities."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BIDSComponents:
    """Parsed components of a BIDS filename."""

    subject: str | None = None
    session: str | None = None
    task: str | None = None
    run: str | None = None
    space: str | None = None
    desc: str | None = None
    suffix: str | None = None
    extension: str | None = None
    extra: dict = field(default_factory=dict)

    def to_filename(self, include: list[str] | None = None) -> str:
        """Reconstruct filename from components.

        Parameters
        ----------
        include : list of str, optional
            Keys to include. If None, includes all non-None components.

        Returns
        -------
        str
            Reconstructed filename from components. BIDS compliance depends on
            which components are present and their values.
        """
        parts = []
        ordered_keys = ["subject", "session", "task", "run", "space", "desc"]

        for key in ordered_keys:
            if include is not None and key not in include:
                continue
            value = getattr(self, key, None)
            if value is not None:
                # Use standard BIDS key prefixes
                prefix = (
                    "sub" if key == "subject" else ("ses" if key == "session" else key)
                )
                parts.append(f"{prefix}-{value}")

        # Add extra components
        for key, value in self.extra.items():
            if include is None or key in include:
                parts.append(f"{key}-{value}")

        # Add suffix and extension
        if self.suffix:
            parts.append(self.suffix)

        filename = "_".join(parts)

        if self.extension:
            filename += self.extension

        return filename


def parse_bids_filename(filepath: str | Path) -> BIDSComponents:
    """Parse BIDS filename into components.

    Parameters
    ----------
    filepath : str or Path
        Path to file or just the filename.

    Returns
    -------
    BIDSComponents
        Parsed filename components.

    Examples
    --------
    >>> fname = "sub-001_ses-1_task-rest_run-01_space-T1w_desc-preproc_bold.nii.gz"
    >>> parse_bids_filename(fname)  # doctest: +NORMALIZE_WHITESPACE
    BIDSComponents(subject='001', session='1', task='rest', run='01',
                   space='T1w', desc='preproc', suffix='bold',
                   extension='.nii.gz', extra={})
    """
    basename = os.path.basename(str(filepath))

    # Handle extensions (including compound like .nii.gz)
    extension = ""
    known_extensions = [
        ".nii.gz",
        ".func.gii",
        ".surf.gii",
        ".tsv",
        ".json",
        ".nii",
        ".gii",
        ".png",
        ".html",
    ]
    for ext in known_extensions:
        if basename.endswith(ext):
            extension = ext
            basename = basename[: -len(ext)]
            break

    components = BIDSComponents(extension=extension)

    # Split by underscores
    elements = basename.split("_")

    # Map BIDS keys to component attributes
    key_mapping = {
        "sub": "subject",
        "ses": "session",
        "task": "task",
        "run": "run",
        "space": "space",
        "desc": "desc",
    }

    for element in elements:
        if "-" in element:
            key, value = element.split("-", 1)
            if key in key_mapping:
                setattr(components, key_mapping[key], value)
            else:
                components.extra[key] = value
        else:
            # Last element without dash is the suffix
            if element:
                components.suffix = element

    return components


def build_bids_filename(
    subject: str,
    session: str | None = None,
    task: str | None = None,
    run: str | None = None,
    space: str | None = None,
    desc: str | None = None,
    suffix: str = "",
    extension: str = "",
) -> str:
    """Build a BIDS-compliant filename from components.

    Parameters
    ----------
    subject : str
        Subject ID (without 'sub-' prefix).
    session : str, optional
        Session ID (without 'ses-' prefix).
    task : str, optional
        Task name.
    run : str, optional
        Run number.
    space : str, optional
        Space name.
    desc : str, optional
        Description.
    suffix : str
        File suffix (e.g., 'bold', 'tsnr').
    extension : str
        File extension (e.g., '.nii.gz').

    Returns
    -------
    str
        BIDS-compliant filename.

    Examples
    --------
    >>> build_bids_filename(
    ...     "001", session="1", task="rest", suffix="bold", extension=".nii.gz"
    ... )
    'sub-001_ses-1_task-rest_bold.nii.gz'
    """
    parts = [f"sub-{subject}"]

    if session:
        parts.append(f"ses-{session}")
    if task:
        parts.append(f"task-{task}")
    if run:
        parts.append(f"run-{run}")
    if space:
        parts.append(f"space-{space}")
    if desc:
        parts.append(f"desc-{desc}")
    if suffix:
        parts.append(suffix)

    filename = "_".join(parts)

    if extension:
        if not extension.startswith("."):
            extension = "." + extension
        filename += extension

    return filename


def discover_subjects(
    base_dir: Path,
    subjects: list[str] | None = None,
    prefix: str = "sub-",
) -> list[str]:
    """Discover subject directories in a BIDS-style directory.

    Parameters
    ----------
    base_dir : Path
        Directory containing subject folders.
    subjects : list of str, optional
        Specific subjects to look for. If None, discovers all.
    prefix : str, default="sub-"
        Prefix for subject directories.

    Returns
    -------
    list of str
        Sorted list of subject directory names (e.g., ['sub-001', 'sub-002']).

    Raises
    ------
    FileNotFoundError
        If base_dir doesn't exist, or if specific subjects were requested
        but none were found.
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    if subjects:
        found = []
        missing = []
        for subj in subjects:
            # Normalize subject ID
            if not subj.startswith(prefix):
                subj = f"{prefix}{subj}"
            if (base_dir / subj).exists():
                found.append(subj)
            else:
                missing.append(subj)

        if missing:
            print(f"Warning: {len(missing)} subject(s) not found: {', '.join(missing)}")
            print(f"  Searched in: {base_dir}")

        if not found and subjects:
            available = sorted(
                d.name
                for d in base_dir.iterdir()
                if d.is_dir() and d.name.startswith(prefix)
            )[:5]
            raise FileNotFoundError(
                f"None of the specified subjects were found.\n"
                f"  Requested: {subjects}\n"
                f"  Directory: {base_dir}\n"
                f"  Available (first 5): {available}"
            )
        return found

    return sorted(
        d.name for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)
    )


def discover_sessions(
    subject_dir: Path,
    sessions: list[str] | None = None,
    prefix: str = "ses-",
) -> list[str]:
    """Discover session directories within a subject directory.

    Parameters
    ----------
    subject_dir : Path
        Subject directory containing session folders.
    sessions : list of str, optional
        Specific sessions to look for. If None, discovers all.
    prefix : str, default="ses-"
        Prefix for session directories.

    Returns
    -------
    list of str
        Sorted list of session directory names (e.g., ['ses-01', 'ses-02']).
        Returns empty list if no session directories exist (single-session dataset).
    """
    if not subject_dir.exists():
        return []

    if sessions:
        found = []
        for sess in sessions:
            # Normalize session ID
            if not sess.startswith(prefix):
                sess = f"{prefix}{sess}"
            if (subject_dir / sess).exists():
                found.append(sess)
            else:
                print(f"Warning: Session not found: {sess} in {subject_dir}")
        return found

    return sorted(
        d.name
        for d in subject_dir.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    )
