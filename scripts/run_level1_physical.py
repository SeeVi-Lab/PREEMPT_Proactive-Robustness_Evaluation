#!/usr/bin/env python3
"""Apply the one-shot image transformations to clean and protected images."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from tqdm import tqdm

from utils.paths import (
    CONCEPT_TYPES,
    LEVEL1_TRANSFORMS,
    PROTECTIONS,
    PROJECT_ROOT,
    level1_out,
    object_concept_label,
    protection_dir,
    resolve_image_path,
)
from utils.records import append_csv, read_csv, write_csv
from utils.transforms import apply_transform, load_rgb, save_rgb


LEVEL1_FIELDS = [
    "image_id",
    "dataset",
    "concept_type",
    "concept_label",
    "protection",
    "level",
    "depth",
    "transform_chain",
    "input_path",
    "output_path",
]


def arts_style_label(image_id: str) -> str:
    stem = Path(image_id).stem
    if "-" in stem:
        return stem.split("-", 1)[0].replace("_", " ")
    return stem.replace("_", " ")


def concept_label(dataset: str, image_id: str) -> str:
    if dataset == "objects":
        return object_concept_label(image_id)
    if dataset == "arts":
        return arts_style_label(image_id)
    return "identity"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "physical_transforms.yaml")
    parser.add_argument(
        "--transforms",
        nargs="+",
        default=list(LEVEL1_TRANSFORMS),
    )
    parser.add_argument("--datasets", nargs="+", default=["faces", "arts", "objects"])
    parser.add_argument("--protections", nargs="+", default=list(PROTECTIONS))
    parser.add_argument("--out_record", type=Path, default=PROJECT_ROOT / "records" / "level1_records.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--append", action="store_true", help="Append to existing level1_records.csv")
    args = parser.parse_args()

    with args.config.open() as f:
        transform_cfg = yaml.safe_load(f)["transforms"]

    records = []
    count = 0

    for protection in args.protections:
        for dataset in args.datasets:
            img_dir = protection_dir(protection) / dataset
            if not img_dir.exists():
                continue
            images = sorted(
                p.name for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            for image_id in tqdm(images, desc=f"{protection}/{dataset}"):
                input_path = resolve_image_path(img_dir, image_id)
                if input_path is None:
                    continue
                image = load_rgb(input_path)
                for transform in args.transforms:
                    out_dir = level1_out(protection, dataset, transform)
                    out_path = out_dir / Path(image_id).name
                    if out_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                        out_path = out_path.with_suffix(input_path.suffix)
                    transformed = apply_transform(image, transform, transform_cfg)
                    save_rgb(transformed, out_path)
                    records.append(
                        {
                            "image_id": image_id,
                            "dataset": dataset,
                            "concept_type": CONCEPT_TYPES[dataset],
                            "concept_label": concept_label(dataset, image_id),
                            "protection": protection,
                            "level": 1,
                            "depth": 1,
                            "transform_chain": transform,
                            "input_path": str(input_path.resolve()),
                            "output_path": str(out_path.resolve()),
                        }
                    )
                    count += 1
                    if args.limit and count >= args.limit:
                        write_csv(args.out_record, LEVEL1_FIELDS, records)
                        print(f"Stopped at limit {args.limit}")
                        return

    if args.append and args.out_record.exists():
        append_csv(args.out_record, LEVEL1_FIELDS, records)
        print(f"Appended {len(records)} records to {args.out_record}")
    else:
        write_csv(args.out_record, LEVEL1_FIELDS, records)
        print(f"Wrote {len(records)} records to {args.out_record}")


if __name__ == "__main__":
    main()
