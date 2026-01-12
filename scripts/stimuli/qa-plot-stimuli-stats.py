#!/usr/bin/env python
"""Plot stimuli statistics for behavioral ratings.

This script generates a publication-ready figure showing the distribution
of behavioral ratings for video stimuli used in the study.

Outputs:
    - desc-proportions_barplot.png: Multi-panel bar chart of all factors

Usage:
    python scripts/stimuli/qa-plot-stimuli-stats.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from hyperface.qa import create_qa_argument_parser, get_config

# Plot settings
DPI = 150
PRIMARY_COLOR = "steelblue"
EDGE_COLOR = "darkslategray"

# Factor display names and order
FACTOR_CONFIG = {
    "gender": {
        "title": "Gender",
        "order": ["Male", "Female"],
    },
    "age": {
        "title": "Age",
        "order": [
            "0-10",
            "11-20",
            "21-30",
            "31-40",
            "41-50",
            "51-60",
            "61-70",
            "70+",
        ],
    },
    "ethnicity": {
        "title": "Ethnicity",
        "order": [
            "White",
            "Black or African American",
            "Hispanic or Latino",
            "Asian",
            "Indian",
            "Other",
        ],
    },
    "emotion": {
        "title": "Emotion",
        "order": [
            "Neutral",
            "Happiness",
            "Surprise",
            "Sadness",
            "Anger",
            "Disgust",
            "Fear",
        ],
    },
    "npeople": {
        "title": "Number of People",
        "order": ["1", "2", "more than 2"],
    },
    "headdir": {
        "title": "Head Direction",
        "order": ["Mostly Left", "Mostly Center", "Mostly Right"],
    },
}


def load_data(path):
    """Load behavioral ratings data."""
    df = pd.read_csv(path, sep="\t")
    # Convert npeople to string for consistent handling
    df["npeople"] = df["npeople"].astype(str)
    return df


def plot_factor_proportions(df, output_path):
    """Create a 2x3 grid of bar plots showing proportions for each factor."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    total_n = len(df)

    for ax, (factor, config) in zip(axes, FACTOR_CONFIG.items()):
        # Get counts in specified order
        counts = df[factor].value_counts()
        order = [lvl for lvl in config["order"] if lvl in counts.index]
        counts = counts.reindex(order)

        # Create bar plot
        bars = ax.bar(
            range(len(counts)),
            counts.values,
            color=PRIMARY_COLOR,
            edgecolor=EDGE_COLOR,
            linewidth=1,
        )

        # Add percentage labels above bars
        for i, (count, bar) in enumerate(zip(counts.values, bars)):
            pct = 100 * count / total_n
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total_n * 0.01,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        # Styling
        ax.set_xticks(range(len(counts)))
        # Shorten long ethnicity labels
        labels = counts.index.tolist()
        if factor == "ethnicity":
            labels = [
                lbl.replace("Black or African American", "Black/AA").replace(
                    "Hispanic or Latino", "Hispanic"
                )
                for lbl in labels
            ]
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
        ax.set_ylabel("Count", fontsize=13)
        ax.set_title(config["title"], fontsize=15, fontweight="bold", pad=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(True, axis="y", alpha=0.5)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)

    plt.suptitle(
        f"Behavioral Ratings (N = {total_n} stimuli)", fontsize=17, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = create_qa_argument_parser(
        description="Plot stimuli statistics for behavioral ratings.",
        include_subjects=False,
    )
    args = parser.parse_args()

    # Load configuration
    config = get_config(config_path=args.config, data_dir=args.data_dir)

    # Paths from config
    ratings_path = config.paths.stimuli_labels_dir / "behavioral-ratings.tsv"
    output_dir = config.paths.stimuli_dir / "figures"

    # Load data
    print(f"Loading data from {ratings_path}")
    df = load_data(ratings_path)
    print(f"Loaded {len(df)} stimuli")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate figure
    plot_factor_proportions(df, output_dir / "desc-proportions_barplot.png")

    print(f"\nFigure saved to {output_dir}")


if __name__ == "__main__":
    main()
