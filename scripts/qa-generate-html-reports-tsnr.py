#!/usr/bin/env python
"""
Generate HTML reports for tSNR quality assessment data.

This script creates individual HTML reports for each subject displaying
tSNR mosaics and violin plots. The reports follow fMRIPrep's HTML structure
and are saved as {subject}.html in the tSNR QA directory.
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


def discover_subjects(tsnr_dir: str, subjects: Optional[List[str]] = None) -> List[str]:
    """Discover subjects in the tSNR directory."""
    if subjects:
        # Validate specified subjects
        found_subjects = []
        for subj in subjects:
            if not subj.startswith("sub-"):
                subj = f"sub-{subj}"

            subj_dir = os.path.join(tsnr_dir, subj)
            if os.path.exists(subj_dir):
                found_subjects.append(subj)
            else:
                print(f"Warning: Subject directory not found: {subj}")
        return found_subjects
    else:
        # Auto-discover all subjects
        tsnr_path = Path(tsnr_dir)
        if not tsnr_path.exists():
            raise FileNotFoundError(f"tSNR directory not found: {tsnr_dir}")

        subjects = sorted([
            d.name for d in tsnr_path.iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        ])
        return subjects


def discover_sessions(subject_dir: Path) -> List[str]:
    """Discover sessions for a subject."""
    sessions = sorted([
        d.name for d in subject_dir.iterdir()
        if d.is_dir() and d.name.startswith("ses-")
    ])

    # If no sessions found, check if there are figure files directly under subject
    if not sessions:
        figures_dir = subject_dir / "figures"
        if figures_dir.exists() and list(figures_dir.glob("*.png")):
            return []  # No sessions, files are directly under subject

    return sessions


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
        'mosaics': [],
        'violin_plot': None,
        'median_mosaic': None,
        'brainmask_mosaic': None
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

        if "desc-tsnr_mosaic" in filename and "run-" in filename:
            # Individual run mosaic
            parts = parse_figure_filename(filename)
            task = parts.get('task', 'unknown')
            run = parts.get('run', '00')

            figures['mosaics'].append({
                'filename': filename,
                'task': task,
                'run': run,
                'title': f"Task {task} - Run {run}"
            })

        elif "desc-mediantsnr_mosaic" in filename:
            figures['median_mosaic'] = filename

        elif "desc-tsnr_violinplot" in filename:
            figures['violin_plot'] = filename

        elif "desc-brainmask_mosaic" in filename:
            figures['brainmask_mosaic'] = filename

    # Sort mosaics by task and run
    figures['mosaics'].sort(key=lambda x: (x['task'], int(x['run'])))

    return figures


def prepare_template_data(subject: str, tsnr_dir: str) -> Dict:
    """Prepare data for template rendering."""
    subject_dir = Path(tsnr_dir) / subject
    figures_dir = subject_dir / "figures"

    sessions = discover_sessions(subject_dir)

    # Collect brainmask (should be at subject level)
    subject_figures = collect_figures(figures_dir, subject)
    brainmask_mosaic = subject_figures['brainmask_mosaic']

    sessions_data = []

    if not sessions:
        # No sessions - process files directly under subject
        figures = collect_figures(figures_dir, subject)

        session_data = {
            'session': None,
            'runs': [
                {
                    'title': mosaic['title'],
                    'mosaic': mosaic['filename']
                }
                for mosaic in figures['mosaics']
            ],
            'violin_plot': figures['violin_plot'],
            'median_mosaic': figures['median_mosaic']
        }
        sessions_data.append(session_data)
    else:
        # Process each session
        for session in sessions:
            figures = collect_figures(figures_dir, subject, session)

            session_data = {
                'session': session,
                'runs': [
                    {
                        'title': mosaic['title'],
                        'mosaic': mosaic['filename']
                    }
                    for mosaic in figures['mosaics']
                ],
                'violin_plot': figures['violin_plot'],
                'median_mosaic': figures['median_mosaic']
            }
            sessions_data.append(session_data)

    template_data = {
        'subject': subject,
        'sessions': sessions if sessions else ['main'],
        'sessions_data': sessions_data,
        'brainmask_mosaic': brainmask_mosaic,
        'generation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return template_data


def generate_html_report(subject: str, tsnr_dir: str, template_env: Environment) -> bool:
    """Generate HTML report for a single subject."""
    print(f"Generating HTML report for {subject}...")

    try:
        # Prepare template data
        template_data = prepare_template_data(subject, tsnr_dir)

        # Check if we have any data to display
        has_data = any(
            session_data['runs'] or session_data['violin_plot'] or session_data['median_mosaic']
            for session_data in template_data['sessions_data']
        )

        if not has_data:
            print(f"  Warning: No figures found for {subject}")
            return False

        # Load and render template
        template = template_env.get_template('tsnr_report.html')
        html_content = template.render(**template_data)

        # Write HTML file
        output_path = Path(tsnr_dir) / f"{subject}.html"
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
        description="Generate HTML reports for tSNR QA data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate reports for all subjects
  python scripts/qa-generate-html-reports.py

  # Generate reports for specific subjects
  python scripts/qa-generate-html-reports.py --subjects sub-001 sub-002

  # Use custom data directory
  python scripts/qa-generate-html-reports.py --data-dir /path/to/data
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

    tsnr_dir = os.path.join(data_dir, "derivatives", "qa", "tsnr")

    # Check if tSNR directory exists
    if not os.path.exists(tsnr_dir):
        print(f"Error: tSNR directory not found: {tsnr_dir}")
        print("Please run qa-plot-tsnr.py first to generate figures.")
        sys.exit(1)

    # Set up template environment
    try:
        template_env = setup_template_environment(script_dir)
    except Exception as e:
        print(f"Error setting up templates: {e}")
        sys.exit(1)

    # Discover subjects
    subjects = discover_subjects(tsnr_dir, args.subjects)
    if not subjects:
        print("No subjects found to process")
        sys.exit(1)

    print(f"Found {len(subjects)} subjects: {', '.join(subjects)}")

    # Generate reports
    successful = 0
    failed = 0

    for subject in subjects:
        if generate_html_report(subject, tsnr_dir, template_env):
            successful += 1
        else:
            failed += 1

    print(f"\nCompleted: {successful} successful, {failed} failed")
    if successful > 0:
        print(f"HTML reports saved to: {tsnr_dir}/{{subject}}.html")


if __name__ == "__main__":
    main()