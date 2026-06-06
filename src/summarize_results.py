from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from common import write_rows_csv


def read_metric_csv(path: Path) -> dict[str, float | str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    summary: dict[str, float | str] = {"method": path.parent.name, "images": len(rows)}
    for key in rows[0].keys():
        values = []
        for row in rows:
            value = row.get(key, "")
            if value == "" or key == "filename":
                continue
            try:
                values.append(float(value))
            except ValueError:
                pass
        if values:
            summary[f"mean_{key}"] = float(np.mean(values))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-image metric CSV files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Metric CSV files.")
    parser.add_argument("--output", required=True, help="Summary CSV path.")
    args = parser.parse_args()

    rows = []
    for item in args.inputs:
        summary = read_metric_csv(Path(item))
        if summary:
            rows.append(summary)

    write_rows_csv(args.output, rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

