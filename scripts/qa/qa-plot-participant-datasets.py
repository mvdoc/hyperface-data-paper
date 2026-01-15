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

    # Create figure with square cells
    fig, ax = plt.subplots(figsize=(2, n_participants * 0.25))

    # Plot heatmap using pcolormesh for precise cell edges
    cmap = plt.cm.colors.ListedColormap(["white", "tab:blue"])
    ax.pcolormesh(
        matrix,
        cmap=cmap,
        vmin=0,
        vmax=1,
        edgecolors="lightgray",
        linewidth=0.3,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Labels
    ax.set_xticks(np.arange(3) + 0.5)
    ax.set_xticklabels(
        ["hyperface", "budapest", "identity\ndecoding"], rotation=45, ha="left"
    )
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(n_participants) + 0.5)
    ax.set_yticklabels(df["participant_id"].str.replace("sub-", ""))

    ax.set_ylabel("Participant")

    plt.tight_layout()

    # Save figure
    output_dir = config.paths.qa_base_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "participant-dataset-matrix.png"
    fig.savefig(output_file, dpi=600, bbox_inches="tight")
    print(f"Saved figure to {output_file}")

    plt.close(fig)


if __name__ == "__main__":
    main()
