#!/usr/bin/env python3
"""Score edited images of one watermark method with a user-supplied decoder.

Example:
    python scripts/score_watermark.py \
        --decoder my_wm.decoder:MyDecoder --protection MyWatermark \
        --reference-bits 0101...  \
        --records records/sequential_simple_records.csv \
        --out records/metrics/watermark/MyWatermark_sequential_simple.csv \
        --with-depth0

Output columns: the record's own columns + bit_accuracy + concept_similarity
(= bit_accuracy, used as S(d) by the survival analysis). With --with-depth0,
one depth-0 row per protected (unedited) image is added, giving S(0).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from utils.records import read_csv, write_csv
from utils.transforms import load_rgb
from utils.watermark import bit_accuracy, load_decoder, text_to_bits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--decoder", required=True, help="module:ClassName implementing decode(image) -> bits")
    parser.add_argument("--protection", required=True, help="protection name as used in the records")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reference-bits", help="embedded payload as a 0/1 string")
    group.add_argument("--reference-text", help="embedded payload as text (converted with UTF-8, 8 bits/byte)")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--with-depth0", action="store_true", help="also score the unedited protected images")
    args = parser.parse_args()

    reference = args.reference_bits or text_to_bits(args.reference_text)
    decoder = load_decoder(args.decoder)

    records = [r for r in read_csv(args.records) if r.get("protection") == args.protection]
    if not records:
        raise SystemExit(f"no records with protection={args.protection!r} in {args.records}")

    rows: list[dict] = []
    if args.with_depth0:
        seen: set[str] = set()
        for rec in records:
            src = rec["input_path"] if str(rec.get("depth", "1")) in ("", "1") else None
            if src is None or src in seen:
                continue
            seen.add(src)
            row = {**rec, "depth": 0, "output_path": src, "edit_type": "none", "prompt": ""}
            rows.append(row)
    rows.extend(dict(r) for r in records)

    for row in tqdm(rows, desc=f"decode {args.protection}"):
        acc = bit_accuracy(decoder.decode(load_rgb(Path(row["output_path"]))), reference)
        row["bit_accuracy"] = f"{acc:.6f}"
        row["concept_similarity"] = row["bit_accuracy"]

    fieldnames = list(dict.fromkeys(k for r in rows for k in r))
    write_csv(args.out, fieldnames, rows)
    print(f"Wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
