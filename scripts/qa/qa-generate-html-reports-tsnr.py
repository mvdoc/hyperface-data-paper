#!/usr/bin/env python
"""Generate HTML reports for tSNR quality assessment data.

This script creates individual HTML reports for each subject displaying
tSNR mosaics and violin plots. The reports follow fMRIPrep's HTML structure
and are saved as {subject}.html in the tSNR QA directory.

Examples:
    # Generate reports for all subjects
    python scripts/qa/qa-generate-html-reports-tsnr.py

    # Generate reports for specific subjects
    python scripts/qa/qa-generate-html-reports-tsnr.py --subjects sub-001 sub-002
"""

import sys
from datetime import datetime
from pathlib import Path

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
        "mosaics": [],
        "violin_plot": None,
        "median_mosaic": None,
        "brainmask_mosaic": None,
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

        if "desc-tsnr_mosaic" in filename and "run-" in filename:
            parts = parse_bids_filename(filename)
            figures["mosaics"].append({
                "filename": filename,
                "task": parts.task or "unknown",
                "run": parts.run or "00",
                "title": f"Task {parts.task or 'unknown'} - Run {parts.run or '00'}",
            })
        elif "desc-mediantsnr_mosaic" in filename:
            figures["median_mosaic"] = filename
        elif "desc-tsnr_violinplot" in filename:
            figures["violin_plot"] = filename
        elif "desc-brainmask_mosaic" in filename:
            figures["brainmask_mosaic"] = filename

    figures["mosaics"].sort(key=lambda x: (x["task"], int(x["run"])))
    return figures


def prepare_template_data(subject: str, tsnr_dir: Path) -> dict:
    """Prepare data for template rendering."""
    subject_dir = tsnr_dir / subject
    figures_dir = subject_dir / "figures"

    sessions = discover_sessions(subject_dir)

    subject_figures = collect_figures(figures_dir, subject)
    brainmask_mosaic = subject_figures["brainmask_mosaic"]

    sessions_data = []

    if not sessions:
        figures = collect_figures(figures_dir, subject)
        runs = [
            {"title": m["title"], "mosaic": m["filename"]}
            for m in figures["mosaics"]
        ]
        session_data = {
            "session": None,
            "runs": runs,
            "violin_plot": figures["violin_plot"],
            "median_mosaic": figures["median_mosaic"],
        }
        sessions_data.append(session_data)
    else:
        for session in sessions:
            figures = collect_figures(figures_dir, subject, session)
            runs = [
                {"title": m["title"], "mosaic": m["filename"]}
                for m in figures["mosaics"]
            ]
            session_data = {
                "session": session,
                "runs": runs,
                "violin_plot": figures["violin_plot"],
                "median_mosaic": figures["median_mosaic"],
            }
            sessions_data.append(session_data)

    return {
        "subject": subject,
        "sessions": sessions if sessions else ["main"],
        "sessions_data": sessions_data,
        "brainmask_mosaic": brainmask_mosaic,
        "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def generate_html_report(
    subject: str, tsnr_dir: Path, template_env: Environment
) -> bool:
    """Generate HTML report for a single subject."""
    print(f"Generating HTML report for {subject}...")

    template_data = prepare_template_data(subject, tsnr_dir)

    has_data = any(
        sd["runs"] or sd["violin_plot"] or sd["median_mosaic"]
        for sd in template_data["sessions_data"]
    )

    if not has_data:
        print(f"  Warning: No figures found for {subject}")
        return False

    template = template_env.get_template("tsnr_report.html")
    html_content = template.render(**template_data)

    output_path = tsnr_dir / f"{subject}.html"
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
        description="Generate HTML reports for tSNR QA data",
        include_subjects=True,
    )
    args = parser.parse_args()

    config = get_config(config_path=args.config, data_dir=args.data_dir)
    tsnr_dir = config.paths.tsnr_dir

    if not tsnr_dir.exists():
        print(f"Error: tSNR directory not found: {tsnr_dir}")
        print("Please run qa-plot-tsnr.py first to generate figures.")
        return 1

    try:
        template_env = setup_template_environment()
    except (ModuleNotFoundError, ValueError) as e:
        print(f"Error setting up templates: {e}")
        print("Ensure the hyperface package is installed: uv pip install -e .")
        return 1

    subjects = discover_subjects(tsnr_dir, args.subjects)
    if not subjects:
        print("No subjects found to process")
        return 1

    print(f"Found {len(subjects)} subjects: {', '.join(subjects)}")

    successful = 0
    failed = 0

    for subject in subjects:
        try:
            if generate_html_report(subject, tsnr_dir, template_env):
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
        print(f"HTML reports saved to: {tsnr_dir}/{{subject}}.html")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
