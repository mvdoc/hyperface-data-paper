#!/usr/bin/env python3
"""Print participant demographic statistics."""

import pandas as pd

from hyperface.qa import create_qa_argument_parser, get_config


def main():
    parser = create_qa_argument_parser(
        description="Print participant demographic statistics.",
        include_subjects=False,
    )
    args = parser.parse_args()

    config = get_config(config_path=args.config, data_dir=args.data_dir)
    df = pd.read_csv(config.paths.data_dir / "participants.tsv", sep="\t")

    n = len(df)
    n_females = (df["sex"] == "F").sum()
    n_males = (df["sex"] == "M").sum()

    lines = [
        f"N participants: {n}",
        f"N females: {n_females}",
        f"N males: {n_males}",
        f"Age mean: {df['age'].mean():.1f}",
        f"Age SD: {df['age'].std():.1f}",
        f"Age range: {df['age'].min()}-{df['age'].max()}",
    ]

    for line in lines:
        print(line)

    output_path = config.paths.qa_base_dir / "participant_demographics.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
