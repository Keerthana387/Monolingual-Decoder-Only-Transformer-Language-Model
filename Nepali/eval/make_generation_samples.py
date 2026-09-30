import argparse
import json
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "sample_id",
        "source_document",
        "setting",
        "temperature",
        "prefix",
        "reference",
        "generated",
    ]

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
