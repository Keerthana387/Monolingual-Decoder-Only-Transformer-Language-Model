import argparse
import csv
import json
from pathlib import Path


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input generations JSONL file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # --------------------------------------------------------
    # Load generation records
    # --------------------------------------------------------

    rows = []

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_no, line in enumerate(f, 1):

            if not line.strip():
                continue

            try:

                record = json.loads(line)

            except json.JSONDecodeError as e:

                raise RuntimeError(
                    f"Invalid JSONL at line {line_no}: {e}"
                )

            rows.append(record)

    if not rows:

        raise RuntimeError(
            "No generation records found."
        )

    # --------------------------------------------------------
    # CSV fields
    # --------------------------------------------------------

    fieldnames = [
        "split",
        "sample_id",
        "temperature",
        "prefix",
        "reference",
        "generated",
    ]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(rows)

    print("=" * 80)
    print("GENERATION SAMPLES")
    print("=" * 80)

    print(
        f"Input records : {len(rows):,}"
    )

    print(
        f"Output        : {output_path}"
    )


if __name__ == "__main__":
    main()