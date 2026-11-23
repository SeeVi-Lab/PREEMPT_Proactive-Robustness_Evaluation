#!/usr/bin/env python3
"""Build the metadata CSV from the images under PREEMPT_DATA_ROOT/Protection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.paths import (
    CONCEPT_TYPES,
    DATASETS,
    PROMPTS_FILE,
    PROTECTIONS,
    PROJECT_ROOT,
    PROTECTION_ROOT,
    object_concept_label,
    protection_dir,
    resolve_image_path,
)
from utils.records import write_csv


def load_prompts(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def arts_style_label(image_id: str) -> str:
    stem = Path(image_id).stem
    if "-" in stem:
        return stem.split("-", 1)[0].replace("_", " ")
    return stem.replace("_", " ")


def infer_concept_label(dataset: str, image_id: str, prompts: dict) -> str:
    if dataset == "objects":
        return object_concept_label(image_id)
    if dataset == "arts":
        return arts_style_label(image_id)
    stem = Path(image_id).stem
    for key, val in prompts.items():
        if key in stem or stem.endswith(key.replace(".jpg", "")):
            return "identity"
    return "identity"


def collect_rows(prompts: dict) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        concept_type = CONCEPT_TYPES[dataset]
        for protection in PROTECTIONS:
            img_dir = protection_dir(protection) / dataset
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                image_id = img_path.name
                concept_label = infer_concept_label(dataset, image_id, prompts)
                rows.append(
                    {
                        "image_id": image_id,
                        "dataset": dataset,
                        "concept_type": concept_type,
                        "concept_label": concept_label,
                        "protection": protection,
                        "input_path": str(img_path.resolve()),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=PROTECTION_ROOT.parent)
    parser.add_argument("--prompts", type=Path, default=PROMPTS_FILE)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "records" / "all_metadata.csv")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    rows = collect_rows(prompts)
    write_csv(
        args.out,
        ["image_id", "dataset", "concept_type", "concept_label", "protection", "input_path"],
        rows,
    )
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
