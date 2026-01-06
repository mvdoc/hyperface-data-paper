#!/usr/bin/env python
"""Generate HTML reports for motion quality assessment data.

This script creates individual HTML reports for each subject displaying
motion traces, FD plots, and violin plots. The reports follow the same structure
as the tSNR reports and are saved as {subject}.html in the motion QA directory.

Examples:
    # Generate reports for all subjects
    python scripts/qa/qa-generate-html-reports-motion.py

    # Generate reports for specific subjects
    python scripts/qa/qa-generate-html-reports-motion.py --subjects sub-001 sub-002
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from jinja2 import Environment, PackageLoader, select_autoescape
except ImportError:
    print("Error: jinja2 is required. Install with: uv pip install jinja2")
    sys.exit(1)

from hyperface.qa import (
    create_qa_argument_parser,
    discover_sessions,
    discover_subjects,
    get_config,
    parse_bids_filename,
)


def collect_figures(
    figures_dir: Path, subject: str, session: str | None = None
) -> dict:
    """Collect all relevant figures for a subject/session."""
    figures: dict = {
        "motion_traces": [],
        "fd_traces": [],
        "fd_violin_plot": None,
    }

    if not figures_dir.exists():
        return figures

    prefix = subject
    if session:
        prefix += f"_{session}"

    for png_file in figures_dir.glob("*.png"):
        filename = png_file.name

        if session and not filename.startswith(prefix):
            continue
        elif not session:
            # Skip session-specific files when processing no-session data
            if any(f"_ses-{i}_" in filename for i in ["1", "2", "3", "4", "5"]):
                continue

        if "desc-motion_traces" in filename and "run-" in filename:
            parts = parse_bids_filename(filename)
            figures["motion_traces"].append({
                "filename": filename,
                "task": parts.task or "unknown",
                "run": parts.run or "00",
                "title": f"Task {parts.task or 'unknown'} - Run {parts.run or '00'}",
            })
        elif "desc-fd_trace" in filename and "run-" in filename:
            parts = parse_bids_filename(filename)
            figures["fd_traces"].append({
                "filename": filename,
                "task": parts.task or "unknown",
                "run": parts.run or "00",
                "title": f"Task {parts.task or 'unknown'} - Run {parts.run or '00'}",
            })
        elif "desc-fd_violinplot" in filename:
            figures["fd_violin_plot"] = filename

    figures["motion_traces"].sort(key=lambda x: (x["task"], int(x["run"])))
    figures["fd_traces"].sort(key=lambda x: (x["task"], int(x["run"])))

    return figures


def combine_run_figures(motion_traces: list[dict], fd_traces: list[dict]) -> list[dict]:
    """Combine motion traces and FD traces for the same runs."""
    motion_map = {(t["task"], t["run"]): t for t in motion_traces}
    fd_map = {(t["task"], t["run"]): t for t in fd_traces}

    all_runs = set(motion_map.keys()) | set(fd_map.keys())

    runs = []
    for task, run in sorted(all_runs):
        runs.append({
            "task": task,
            "run": run,
            "title": f"Task {task} - Run {run}",
            "motion_traces": motion_map.get((task, run), {}).get("filename"),
            "fd_trace": fd_map.get((task, run), {}).get("filename"),
        })

    return runs


def calculate_quality_metrics(
    fmriprep_dir: Path, subject: str, session: str | None = None
) -> list[str]:
    """Calculate quality metrics from actual confounds data."""
    subject_fmriprep = fmriprep_dir / subject

    confounds_files = []
    if session:
        session_dir = subject_fmriprep / session / "func"
        if session_dir.exists():
            confounds_files = list(session_dir.glob("*_desc-confounds_timeseries.tsv"))
    else:
        for session_path in subject_fmriprep.glob("ses-*/func/"):
            confounds_files.extend(list(session_path.glob("*_desc-confounds_timeseries.tsv")))
        func_dir = subject_fmriprep / "func"
        if func_dir.exists():
            confounds_files.extend(list(func_dir.glob("*_desc-confounds_timeseries.tsv")))

    if not confounds_files:
        return ["No motion data available"]

    all_fd_data = []
    failed_files = 0
    for confounds_file in confounds_files:
        try:
            df = pd.read_csv(confounds_file, sep="\t")
            if "framewise_displacement" in df.columns:
                fd_data = df["framewise_displacement"].fillna(0).values
                all_fd_data.extend(fd_data)
        except (FileNotFoundError, pd.errors.ParserError, ValueError):
            failed_files += 1
            continue

    if not all_fd_data:
        return ["No FD data available"]

    all_fd_array = np.array(all_fd_data)
    mean_fd = np.mean(all_fd_array)
    max_fd = np.max(all_fd_array)
    median_fd = np.median(all_fd_array)
    trs_above_thresh = np.sum(all_fd_array > 0.5)
    percent_above_thresh = (trs_above_thresh / len(all_fd_array)) * 100

    n_trs = len(all_fd_array)
    metrics = [
        f"Mean FD: {mean_fd:.3f}mm",
        f"Median FD: {median_fd:.3f}mm",
        f"Max FD: {max_fd:.3f}mm",
        f"TRs > 0.5mm: {trs_above_thresh}/{n_trs} ({percent_above_thresh:.1f}%)",
        f"Total runs: {len(confounds_files)}",
    ]
    if failed_files > 0:
        metrics.append(f"Note: {failed_files} confounds file(s) could not be loaded")
    return metrics


def prepare_template_data(subject: str, motion_dir: Path, fmriprep_dir: Path) -> dict:
    """Prepare data for template rendering."""
    subject_dir = motion_dir / subject
    figures_dir = subject_dir / "figures"

    sessions = discover_sessions(subject_dir)

    sessions_data = []

    if not sessions:
        figures = collect_figures(figures_dir, subject)
        runs = combine_run_figures(figures["motion_traces"], figures["fd_traces"])

        sessions_data.append({
            "session": None,
            "runs": runs,
            "fd_violin_plot": figures["fd_violin_plot"],
            "quality_metrics": calculate_quality_metrics(fmriprep_dir, subject),
        })
    else:
        for session in sessions:
            figures = collect_figures(figures_dir, subject, session)
            runs = combine_run_figures(figures["motion_traces"], figures["fd_traces"])

            metrics = calculate_quality_metrics(fmriprep_dir, subject, session)
            sessions_data.append({
                "session": session,
                "runs": runs,
                "fd_violin_plot": figures["fd_violin_plot"],
                "quality_metrics": metrics,
            })

    group_fd_plot = None
    group_plot_path = motion_dir / "group_desc-fd_violinplot.png"
    if group_plot_path.exists():
        group_fd_plot = "group_desc-fd_violinplot.png"

    return {
        "subject": subject,
        "sessions": sessions if sessions else ["main"],
        "sessions_data": sessions_data,
        "group_fd_plot": group_fd_plot,
        "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def generate_html_report(
    subject: str, motion_dir: Path, fmriprep_dir: Path, template_env: Environment
) -> bool:
    """Generate HTML report for a single subject."""
    print(f"Generating HTML report for {subject}...")

    template_data = prepare_template_data(subject, motion_dir, fmriprep_dir)

    has_data = any(
        sd["runs"] or sd["fd_violin_plot"]
        for sd in template_data["sessions_data"]
    )

    if not has_data:
        print(f"  Warning: No figures found for {subject}")
        return False

    template = template_env.get_template("motion_report.html")
    html_content = template.render(**template_data)

    output_path = motion_dir / f"{subject}.html"
    output_path.write_text(html_content, encoding="utf-8")

    print(f"  Report saved: {output_path}")
    return True


def setup_template_environment() -> Environment:
    """Set up Jinja2 template environment using package templates."""
    return Environment(
        loader=PackageLoader("hyperface", "templates"),
        autoescape=select_autoescape(["html", "xml"]),
    )


def main():
    parser = create_qa_argument_parser(
        description="Generate HTML reports for motion QA data",
        include_subjects=True,
    )
    args = parser.parse_args()

    config = get_config(config_path=args.config, data_dir=args.data_dir)
    motion_dir = config.paths.motion_dir
    fmriprep_dir = config.paths.fmriprep_dir

    if not motion_dir.exists():
        print(f"Error: Motion QA directory not found: {motion_dir}")
        print("Please run qa-plot-motion.py first to generate figures.")
        return 1

    try:
        template_env = setup_template_environment()
    except (ModuleNotFoundError, ValueError) as e:
        print(f"Error setting up templates: {e}")
        print("Ensure the hyperface package is installed: uv pip install -e .")
        return 1

    subjects = discover_subjects(motion_dir, args.subjects)
    if not subjects:
        print("No subjects found to process")
        return 1

    print(f"Found {len(subjects)} subjects: {', '.join(subjects)}")

    successful = 0
    failed = 0

    for subject in subjects:
        try:
            if generate_html_report(subject, motion_dir, fmriprep_dir, template_env):
                successful += 1
            else:
                failed += 1
        except FileNotFoundError as e:
            print(f"  Error: Missing files for {subject}: {e}")
            failed += 1
        except (ValueError, OSError) as e:
            print(f"  Error generating report for {subject}: {e}")
            failed += 1

    print(f"\nCompleted: {successful} successful, {failed} failed")
    if successful > 0:
        print(f"HTML reports saved to: {motion_dir}/{{subject}}.html")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
