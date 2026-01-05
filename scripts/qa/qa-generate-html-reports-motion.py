#!/usr/bin/env python
"""
Generate HTML reports for motion quality assessment data.

This script creates individual HTML reports for each subject displaying
motion traces, FD plots, and violin plots. The reports follow the same structure
as the tSNR reports and are saved as {subject}.html in the motion QA directory.
"""

import argparse
import os
import sys
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
except ImportError:
    print("Error: jinja2 is required. Install with: uv pip install jinja2")
    sys.exit(1)


def discover_subjects(motion_dir: str, subjects: Optional[List[str]] = None) -> List[str]:
    """Discover subjects in the motion QA directory."""
    if subjects:
        # Validate specified subjects
        found_subjects = []
        for subj in subjects:
            if not subj.startswith("sub-"):
                subj = f"sub-{subj}"

            subj_dir = os.path.join(motion_dir, subj)
            if os.path.exists(subj_dir):
                found_subjects.append(subj)
            else:
                print(f"Warning: Subject directory not found: {subj}")
        return found_subjects
    else:
        # Auto-discover all subjects
        motion_path = Path(motion_dir)
        if not motion_path.exists():
            raise FileNotFoundError(f"Motion directory not found: {motion_dir}")

        subjects = sorted([
            d.name for d in motion_path.iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        ])
        return subjects


def discover_sessions(subject_dir: Path) -> List[str]:
    """Discover sessions for a subject by looking at figure filenames."""
    figures_dir = subject_dir / "figures"
    if not figures_dir.exists():
        return []

    # Look at figure filenames to determine sessions
    sessions = set()
    for png_file in figures_dir.glob("*.png"):
        filename = png_file.name
        # Parse for session info in filename
        parts = filename.split("_")
        for part in parts:
            if part.startswith("ses-"):
                sessions.add(part)

    if sessions:
        return sorted(list(sessions))
    else:
        return []  # No sessions found


def parse_figure_filename(filename: str) -> Dict[str, str]:
    """Parse BIDS filename to extract components."""
    basename = Path(filename).stem
    parts = {}

    # Split by underscores and parse key-value pairs
    elements = basename.split("_")
    for element in elements:
        if "-" in element:
            key, value = element.split("-", 1)
            parts[key] = value

    return parts


def collect_figures(figures_dir: Path, subject: str, session: Optional[str] = None) -> Dict:
    """Collect all relevant figures for a subject/session."""
    figures = {
        'motion_traces': [],
        'fd_traces': [],
        'fd_violin_plot': None
    }

    if not figures_dir.exists():
        return figures

    # Build filename pattern prefix
    prefix = subject
    if session:
        prefix += f"_{session}"

    # Find all PNG files
    png_files = list(figures_dir.glob("*.png"))

    for png_file in png_files:
        filename = png_file.name

        # Skip files that don't match the subject/session pattern
        if session and not filename.startswith(prefix):
            continue
        elif not session and any(f"_ses-{i}_" in filename for i in ["1", "2", "3", "4", "5"]):
            continue  # Skip session-specific files when looking for non-session files

        if "desc-motion_traces" in filename and "run-" in filename:
            # Individual run motion traces
            parts = parse_figure_filename(filename)
            task = parts.get('task', 'unknown')
            run = parts.get('run', '00')

            figures['motion_traces'].append({
                'filename': filename,
                'task': task,
                'run': run,
                'title': f"Task {task} - Run {run}"
            })

        elif "desc-fd_trace" in filename and "run-" in filename:
            # Individual run FD traces
            parts = parse_figure_filename(filename)
            task = parts.get('task', 'unknown')
            run = parts.get('run', '00')

            figures['fd_traces'].append({
                'filename': filename,
                'task': task,
                'run': run,
                'title': f"Task {task} - Run {run}"
            })

        elif "desc-fd_violinplot" in filename:
            figures['fd_violin_plot'] = filename

    # Sort traces by task and run
    figures['motion_traces'].sort(key=lambda x: (x['task'], int(x['run'])))
    figures['fd_traces'].sort(key=lambda x: (x['task'], int(x['run'])))

    return figures


def combine_run_figures(motion_traces: List[Dict], fd_traces: List[Dict]) -> List[Dict]:
    """Combine motion traces and FD traces for the same runs."""
    runs = []

    # Create a mapping from (task, run) to figures
    motion_map = {(trace['task'], trace['run']): trace for trace in motion_traces}
    fd_map = {(trace['task'], trace['run']): trace for trace in fd_traces}

    # Get all unique (task, run) combinations
    all_runs = set(motion_map.keys()) | set(fd_map.keys())

    for task, run in sorted(all_runs):
        run_data = {
            'task': task,
            'run': run,
            'title': f"Task {task} - Run {run}",
            'motion_traces': motion_map.get((task, run), {}).get('filename'),
            'fd_trace': fd_map.get((task, run), {}).get('filename')
        }
        runs.append(run_data)

    return runs


def calculate_quality_metrics(motion_dir: str, subject: str, session: Optional[str] = None) -> List[str]:
    """Calculate quality metrics from actual confounds data."""
    import pandas as pd
    import numpy as np

    # Find confounds files for this subject/session
    fmriprep_dir = Path(motion_dir).parent.parent / "fmriprep" / subject

    confounds_files = []
    if session:
        session_dir = fmriprep_dir / session / "func"
        if session_dir.exists():
            confounds_files = list(session_dir.glob("*_desc-confounds_timeseries.tsv"))
    else:
        # Look in all sessions or directly under subject
        for session_path in fmriprep_dir.glob("ses-*/func/"):
            confounds_files.extend(list(session_path.glob("*_desc-confounds_timeseries.tsv")))

        # Also check directly under subject/func if no sessions
        func_dir = fmriprep_dir / "func"
        if func_dir.exists():
            confounds_files.extend(list(func_dir.glob("*_desc-confounds_timeseries.tsv")))

    if not confounds_files:
        return ["No motion data available"]

    # Collect all FD data across runs
    all_fd_data = []
    total_trs = 0

    for confounds_file in confounds_files:
        try:
            df = pd.read_csv(confounds_file, sep='\t')
            if 'framewise_displacement' in df.columns:
                fd_data = df['framewise_displacement'].fillna(0).values
                all_fd_data.extend(fd_data)
                total_trs += len(fd_data)
        except Exception as e:
            continue

    if not all_fd_data:
        return ["No FD data available"]

    # Calculate statistics
    all_fd_array = np.array(all_fd_data)
    mean_fd = np.mean(all_fd_array)
    max_fd = np.max(all_fd_array)
    median_fd = np.median(all_fd_array)
    trs_above_thresh = np.sum(all_fd_array > 0.5)
    percent_above_thresh = (trs_above_thresh / len(all_fd_array)) * 100

    # Format metrics
    metrics = [
        f"Mean FD: {mean_fd:.3f}mm",
        f"Median FD: {median_fd:.3f}mm",
        f"Max FD: {max_fd:.3f}mm",
        f"TRs > 0.5mm: {trs_above_thresh}/{len(all_fd_array)} ({percent_above_thresh:.1f}%)",
        f"Total runs: {len(confounds_files)}"
    ]

    return metrics


def prepare_template_data(subject: str, motion_dir: str) -> Dict:
    """Prepare data for template rendering."""
    subject_dir = Path(motion_dir) / subject
    figures_dir = subject_dir / "figures"

    sessions = discover_sessions(subject_dir)

    sessions_data = []

    if not sessions:
        # No sessions - process files directly under subject
        figures = collect_figures(figures_dir, subject)

        # Combine motion traces and FD traces by run
        runs = combine_run_figures(figures['motion_traces'], figures['fd_traces'])

        session_data = {
            'session': None,
            'runs': runs,
            'fd_violin_plot': figures['fd_violin_plot'],
            'quality_metrics': calculate_quality_metrics(motion_dir, subject)
        }
        sessions_data.append(session_data)
    else:
        # Process each session
        for session in sessions:
            figures = collect_figures(figures_dir, subject, session)

            # Combine motion traces and FD traces by run
            runs = combine_run_figures(figures['motion_traces'], figures['fd_traces'])

            session_data = {
                'session': session,
                'runs': runs,
                'fd_violin_plot': figures['fd_violin_plot'],
                'quality_metrics': calculate_quality_metrics(motion_dir, subject, session)
            }
            sessions_data.append(session_data)

    # Check for group-level FD plot
    group_fd_plot = None
    group_plot_path = Path(motion_dir) / "group_desc-fd_violinplot.png"
    if group_plot_path.exists():
        group_fd_plot = "group_desc-fd_violinplot.png"

    template_data = {
        'subject': subject,
        'sessions': sessions if sessions else ['main'],
        'sessions_data': sessions_data,
        'group_fd_plot': group_fd_plot,
        'generation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return template_data


def generate_html_report(subject: str, motion_dir: str, template_env: Environment) -> bool:
    """Generate HTML report for a single subject."""
    print(f"Generating HTML report for {subject}...")

    try:
        # Prepare template data
        template_data = prepare_template_data(subject, motion_dir)

        # Check if we have any data to display
        has_data = any(
            session_data['runs'] or session_data['fd_violin_plot']
            for session_data in template_data['sessions_data']
        )

        if not has_data:
            print(f"  Warning: No figures found for {subject}")
            return False

        # Load and render template
        template = template_env.get_template('motion_report.html')
        html_content = template.render(**template_data)

        # Write HTML file
        output_path = Path(motion_dir) / f"{subject}.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"  Report saved: {output_path}")
        return True

    except Exception as e:
        print(f"  Error generating report for {subject}: {str(e)}")
        return False


def setup_template_environment(script_dir: str) -> Environment:
    """Set up Jinja2 template environment."""
    # Find the templates directory
    templates_dir = os.path.join(script_dir, "..", "src", "hyperface", "templates")
    templates_dir = os.path.abspath(templates_dir)

    if not os.path.exists(templates_dir):
        raise FileNotFoundError(f"Templates directory not found: {templates_dir}")

    env = Environment(
        loader=FileSystemLoader(templates_dir),
        autoescape=select_autoescape(['html', 'xml'])
    )

    return env


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate HTML reports for motion QA data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate reports for all subjects
  python scripts/qa/qa-generate-html-reports-motion.py

  # Generate reports for specific subjects
  python scripts/qa/qa-generate-html-reports-motion.py --subjects sub-001 sub-002

  # Use custom data directory
  python scripts/qa/qa-generate-html-reports-motion.py --data-dir /path/to/data
""",
    )

    parser.add_argument(
        "--subjects", nargs="+",
        help="Subject IDs to process (default: all subjects)"
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Path to BIDS dataset directory (default: data)"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Set up paths
    script_dir = os.path.dirname(__file__)
    if not os.path.isabs(args.data_dir):
        data_dir = os.path.abspath(os.path.join(script_dir, "..", args.data_dir))
    else:
        data_dir = args.data_dir

    motion_dir = os.path.join(data_dir, "derivatives", "qa", "motion")

    # Check if motion directory exists
    if not os.path.exists(motion_dir):
        print(f"Error: Motion QA directory not found: {motion_dir}")
        print("Please run qa-plot-motion.py first to generate figures.")
        sys.exit(1)

    # Set up template environment
    try:
        template_env = setup_template_environment(script_dir)
    except Exception as e:
        print(f"Error setting up templates: {e}")
        sys.exit(1)

    # Discover subjects
    subjects = discover_subjects(motion_dir, args.subjects)
    if not subjects:
        print("No subjects found to process")
        sys.exit(1)

    print(f"Found {len(subjects)} subjects: {', '.join(subjects)}")

    # Generate reports
    successful = 0
    failed = 0

    for subject in subjects:
        if generate_html_report(subject, motion_dir, template_env):
            successful += 1
        else:
            failed += 1

    print(f"\nCompleted: {successful} successful, {failed} failed")
    if successful > 0:
        print(f"HTML reports saved to: {motion_dir}/{{subject}}.html")


if __name__ == "__main__":
    main()