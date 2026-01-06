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
from hyperface.qa.plotting import style_violin_plot

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
]
