"""Centralized configuration for QA scripts."""

import argparse
import os
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import yaml


@dataclass
class QAPaths:
    """Resolved absolute paths for QA pipeline directories.

    Attributes
    ----------
    data_dir : Path
        Root BIDS dataset directory.
    derivatives_dir : Path
        BIDS derivatives directory (data_dir/derivatives).
    fmriprep_dir : Path
        fMRIprep output directory.
    freesurfer_dir : Path
        FreeSurfer output directory.
    qa_base_dir : Path
        Base QA output directory.
    tsnr_dir : Path
        tSNR analysis output directory.
    motion_dir : Path
        Motion analysis output directory.
    isc_dir : Path
        Inter-subject correlation output directory.
    """

    data_dir: Path
    derivatives_dir: Path
    fmriprep_dir: Path
    freesurfer_dir: Path
    qa_base_dir: Path
    tsnr_dir: Path
    motion_dir: Path
    isc_dir: Path

    @classmethod
    def from_config(
        cls, config: dict, base_dir: Path | None = None
    ) -> "QAPaths":
        """Create QAPaths from config dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary with 'directories' key.
        base_dir : Path, optional
            Base directory for resolving relative paths.
            Defaults to current working directory.

        Returns
        -------
        QAPaths
            Resolved paths object.

        Raises
        ------
        ValueError
            If config is missing required 'directories' section or has
            invalid structure.
        """
        if base_dir is None:
            # Try to find project root by looking for pyproject.toml
            # Start from cwd and walk up
            check_dir = Path.cwd()
            while check_dir != check_dir.parent:
                if (check_dir / "pyproject.toml").exists():
                    base_dir = check_dir
                    break
                check_dir = check_dir.parent
            else:
                base_dir = Path.cwd()

        # Validate config structure
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")

        if "directories" not in config:
            raise ValueError(
                "Config missing required 'directories' section. "
                "See hyperface/assets/qa_config.yaml for the expected structure."
            )

        dirs = config["directories"]
        if not isinstance(dirs, dict):
            raise ValueError(
                f"Config 'directories' must be a dictionary, not {type(dirs).__name__}"
            )

        # Resolve data_dir (can be absolute or relative to base_dir)
        data_dir_str = dirs.get("data_dir", "data")
        if os.path.isabs(data_dir_str):
            data_dir = Path(data_dir_str)
        else:
            data_dir = base_dir / data_dir_str

        # Build derivative paths
        derivatives_dir = data_dir / dirs.get("derivatives_dir", "derivatives")
        fmriprep_dir = derivatives_dir / dirs.get("fmriprep", "fmriprep")
        freesurfer_dir = derivatives_dir / dirs.get("freesurfer", "freesurfer")

        # QA output directories
        qa_config = dirs.get("qa", {})
        qa_base = qa_config.get("base", "qa") if isinstance(qa_config, dict) else "qa"
        qa_base_dir = derivatives_dir / qa_base

        if isinstance(qa_config, dict):
            tsnr_subdir = qa_config.get("tsnr", "tsnr")
            motion_subdir = qa_config.get("motion", "motion")
            isc_subdir = qa_config.get("isc", "isc")
        else:
            tsnr_subdir = "tsnr"
            motion_subdir = "motion"
            isc_subdir = "isc"

        return cls(
            data_dir=data_dir.resolve(),
            derivatives_dir=derivatives_dir.resolve(),
            fmriprep_dir=fmriprep_dir.resolve(),
            freesurfer_dir=freesurfer_dir.resolve(),
            qa_base_dir=qa_base_dir.resolve(),
            tsnr_dir=(qa_base_dir / tsnr_subdir).resolve(),
            motion_dir=(qa_base_dir / motion_subdir).resolve(),
            isc_dir=(qa_base_dir / isc_subdir).resolve(),
        )


@dataclass
class QAConfig:
    """Configuration for QA pipeline.

    Attributes
    ----------
    paths : QAPaths
        Resolved filesystem paths for all QA directories.
    defaults : dict
        Default processing parameters (e.g., tsnr_vmin, fd_threshold).
    plot_style : dict
        Preset styling options for visualization.
    raw_config : dict
        The original loaded YAML configuration dictionary.
    """

    paths: QAPaths
    defaults: dict = field(default_factory=dict)
    plot_style: dict = field(default_factory=dict)
    raw_config: dict = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        config_path: str | None = None,
        data_dir: str | None = None,
        base_dir: Path | None = None,
    ) -> "QAConfig":
        """Load configuration from file or package defaults.

        Priority for config file:
        1. Explicit config_path argument
        2. HYPERFACE_QA_CONFIG environment variable
        3. Package default (hyperface/assets/qa_config.yaml)

        Parameters
        ----------
        config_path : str, optional
            Path to custom config file.
        data_dir : str, optional
            Override data directory from config.
        base_dir : Path, optional
            Base directory for resolving relative paths.

        Returns
        -------
        QAConfig
            Loaded configuration.

        Raises
        ------
        FileNotFoundError
            If specified config file doesn't exist.
        ValueError
            If config file has invalid YAML syntax or is empty.
        """
        # Determine config file to load
        # Also set config_file_dir for resolving relative paths
        config_file_dir = None

        if config_path is not None:
            cfg_path = Path(config_path).resolve()
            if not cfg_path.exists():
                msg = f"Config file not found: {config_path}"
                raise FileNotFoundError(msg)
            config_file_dir = cfg_path.parent
            try:
                with open(cfg_path) as f:
                    config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(
                    f"Invalid YAML syntax in config file: {config_path}\n"
                    f"Error: {e}\n"
                    f"See hyperface/assets/qa_config.yaml for an example."
                ) from e
        elif os.environ.get("HYPERFACE_QA_CONFIG"):
            env_path = Path(os.environ["HYPERFACE_QA_CONFIG"]).resolve()
            if not env_path.exists():
                msg = f"Config file from HYPERFACE_QA_CONFIG not found: {env_path}"
                raise FileNotFoundError(msg)
            config_file_dir = env_path.parent
            try:
                with open(env_path) as f:
                    config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(
                    f"Invalid YAML syntax in config file: {env_path}\n"
                    f"Error: {e}\n"
                    f"See hyperface/assets/qa_config.yaml for an example."
                ) from e
        else:
            # Load package default
            print("Using default configuration from hyperface package")
            try:
                config_file = files("hyperface.assets").joinpath("qa_config.yaml")
                config_text = config_file.read_text()
            except (ModuleNotFoundError, FileNotFoundError) as e:
                raise FileNotFoundError(
                    "Default config not found. The hyperface package may not "
                    "be properly installed. Try: uv pip install -e . OR "
                    "provide --config path/to/config.yaml"
                ) from e
            try:
                config = yaml.safe_load(config_text)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in package default config: {e}") from e

        # Validate config is not empty
        if config is None:
            raise ValueError(
                "Config file is empty. "
                "See hyperface/assets/qa_config.yaml for the expected structure."
            )

        # Override data_dir if provided
        if data_dir is not None:
            config.setdefault("directories", {})["data_dir"] = data_dir

        # Determine base_dir for path resolution:
        # 1. Explicit base_dir argument takes priority
        # 2. Config file's parent directory (for custom configs)
        # 3. QAPaths.from_config will find pyproject.toml or use cwd
        effective_base_dir = base_dir or config_file_dir

        # Build paths
        paths = QAPaths.from_config(config, base_dir=effective_base_dir)

        return cls(
            paths=paths,
            defaults=config.get("defaults", {}),
            plot_style=config.get("plot_style", {}),
            raw_config=config,
        )


def get_config(
    config_path: str | None = None,
    data_dir: str | None = None,
    base_dir: Path | None = None,
) -> QAConfig:
    """Convenience function to get QA configuration.

    Parameters
    ----------
    config_path : str, optional
        Path to custom config file.
    data_dir : str, optional
        Override data directory.
    base_dir : Path, optional
        Base directory for resolving relative paths.

    Returns
    -------
    QAConfig
        Loaded configuration.
    """
    return QAConfig.load(
        config_path=config_path,
        data_dir=data_dir,
        base_dir=base_dir,
    )


def create_qa_argument_parser(
    description: str,
    include_subjects: bool = True,
    include_sessions: bool = False,
    include_tasks: bool = False,
    include_config: bool = True,
    include_dry_run: bool = False,
) -> argparse.ArgumentParser:
    """Create a standard argument parser for QA scripts.

    Parameters
    ----------
    description : str
        Script description for help text.
    include_subjects : bool, default=True
        Include --subjects argument.
    include_sessions : bool, default=False
        Include --sessions argument.
    include_tasks : bool, default=False
        Include --tasks argument.
    include_config : bool, default=True
        Include --config and --data-dir arguments.
    include_dry_run : bool, default=False
        Include --dry-run argument.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    if include_subjects:
        parser.add_argument(
            "--subjects",
            nargs="+",
            help="Subject IDs to process (default: all discovered subjects)",
        )

    if include_sessions:
        parser.add_argument(
            "--sessions",
            nargs="+",
            help="Session IDs to process (default: all sessions)",
        )

    if include_tasks:
        parser.add_argument(
            "--tasks",
            nargs="+",
            help="Task names to process (default: all tasks)",
        )

    if include_config:
        parser.add_argument(
            "--config",
            help="Path to config file (default: package default)",
        )
        parser.add_argument(
            "--data-dir",
            help="Override data directory from config",
        )

    if include_dry_run:
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be processed without running",
        )

    return parser
