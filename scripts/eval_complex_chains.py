#!/usr/bin/env python3
"""Compute per-step metrics for the mixed-chain (CX1–CX4) records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from compute_metrics import (  # noqa: E402
    METRIC_FIELDS,
    compute_level_metrics,
    load_original_map,
    merge_editor_metrics,
)
from utils.metrics import MetricComputer
from utils.paths import PROJECT_ROOT
from utils.records import read_csv, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=PROJECT_ROOT / "records" / "complex_chain_records.csv")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "records" / "all_metadata.csv")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "records" / "metrics" / "complex_chain_metrics.csv")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--editor", default=None, help="Filter records by editor; merges into --out")
    args = parser.parse_args()

    records = read_csv(args.records)
    if args.editor:
        records = [r for r in records if r.get("editor") == args.editor]
    if not records:
        raise SystemExit("no records to score")

    originals = load_original_map(args.metadata)
    computer = MetricComputer(device=args.device)
    rows = compute_level_metrics(records, "edit_type", originals, computer)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.editor:
        merge_editor_metrics(args.out, rows, args.editor, METRIC_FIELDS)
    else:
        write_csv(args.out, METRIC_FIELDS, rows)
        print(f"Wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
