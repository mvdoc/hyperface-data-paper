"""QA utilities for the Hyperface dataset.

This module provides centralized configuration, BIDS filename parsing,
and plotting utilities for the quality assurance pipeline.

Example
-------
>>> from hyperface.qa import get_config, parse_bids_filename, discover_subjects
>>> config = get_config()
>>> print(config.paths.tsnr_dir)
>>> parts = parse_bids_filename("sub-001_task-rest_bold.nii.gz")
>>> print(parts.subject, parts.task)
>>> subjects = discover_subjects(config.paths.tsnr_dir)
"""

from hyperface.qa.bids import (
    BIDSComponents,
    build_bids_filename,
    discover_sessions,
    discover_subjects,
    parse_bids_filename,
)
from hyperface.qa.config import (
    QAConfig,
    QAPaths,
    create_qa_argument_parser,
    get_config,
)
from hyperface.qa.motion import (
    collect_confounds_by_task,
    get_fd_outlier_counts,
    get_motion_outlier_counts,
)
from hyperface.qa.plotting import style_violin_plot
from hyperface.qa.tsnr import (
    collect_tsnr_files_by_task,
    compute_conjunction_brainmask,
    group_files_by_task,
    load_subject_brainmask,
)

__all__ = [
    "QAConfig",
    "QAPaths",
    "get_config",
    "create_qa_argument_parser",
    "BIDSComponents",
    "parse_bids_filename",
    "build_bids_filename",
    "discover_subjects",
    "discover_sessions",
    "style_violin_plot",
    "get_motion_outlier_counts",
    "get_fd_outlier_counts",
    "collect_confounds_by_task",
    "collect_tsnr_files_by_task",
    "compute_conjunction_brainmask",
    "group_files_by_task",
    "load_subject_brainmask",
]
