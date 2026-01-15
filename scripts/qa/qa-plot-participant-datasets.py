#!/usr/bin/env python
"""Plot a matrix showing which participants have data from each dataset.

Creates a visualization with participants as rows and datasets (hyperface,
budapest, identity-decoding) as columns.

Examples:
    python scripts/qa/qa-plot-participant-datasets.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hyperface.qa import create_qa_argument_parser, get_config


def main():
    parser = create_qa_argument_parser(
        description="Plot participant dataset availability matrix.",
        include_subjects=False,
    )
    args = parser.parse_args()
    config = get_config(config_path=args.config, data_dir=args.data_dir)

    # Load participants file
    participants_file = config.paths.data_dir / "participants.tsv"
    df = pd.read_csv(participants_file, sep="\t")

    # Create binary matrix
    n_participants = len(df)
    datasets = ["hyperface", "budapest", "identity_decoding"]
    matrix = np.zeros((n_participants, 3), dtype=int)

    # All participants have hyperface
    matrix[:, 0] = 1

    # Convert Yes/No to 1/0
    matrix[:, 1] = (df["budapest"] == "Yes").astype(int)
    matrix[:, 2] = (df["identity_decoding"] == "Yes").astype(int)

    # Create figure
    fig, ax = plt.subplots(figsize=(0.8, n_participants * 0.2))

    # Colors for each dataset
    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = ["hyperface", "budapest", "identity decoding"]

    # Plot dots for each dataset
    for j in range(3):
        for i in range(n_participants):
            if matrix[i, j] == 1:
                ax.scatter(j, i, color=colors[j], s=20, marker="o")
            else:
                ax.scatter(j, i, color="lightgray", s=20, marker="o", facecolors="none")

    # Add legend handles
    for j in range(3):
        ax.scatter([], [], color=colors[j], s=20, marker="o", label=labels[j])

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(n_participants - 0.5, -0.5)

    # Labels
    ax.set_xticks([])
    ax.set_yticks(range(n_participants))
    ax.set_yticklabels(df["participant_id"].str.replace("sub-", ""))

    ax.tick_params(axis="both", length=0)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)

    # Legend at bottom, left-aligned with y-axis labels
    ax.legend(loc="upper left", bbox_to_anchor=(-0.6, -0.02), ncol=1, frameon=False)

    # Save figure
    output_dir = config.paths.qa_base_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "participant-dataset-matrix.png"
    fig.savefig(output_file, dpi=600, bbox_inches="tight")
    print(f"Saved figure to {output_file}")

    plt.close(fig)


if __name__ == "__main__":
    main()
