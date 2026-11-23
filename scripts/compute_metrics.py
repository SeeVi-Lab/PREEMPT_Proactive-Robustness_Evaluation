#!/usr/bin/env python3
"""Compute image-quality and concept-similarity metrics for edit records, and S(0) of protected images."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils.metrics import MetricComputer
from utils.paths import CONCEPT_TYPES, PROJECT_ROOT, normalize_image_id, protection_dir, resolve_image_path
from utils.records import read_csv, write_csv
from utils.transforms import load_rgb

METRIC_FIELDS = [
    "image_id",
    "dataset",
    "protection",
    "concept_type",
    "concept_label",
    "editor",
    "transform",
    "chain_id",
    "depth",
    "psnr",
    "ssim",
    "lpips",
    "clip_img_sim",
    "dino_sim",
    "concept_similarity",
    "clip_concept_sim",
    "dino_concept_sim",
    "face_detected",
]

DEPTH0_FIELDS = [
    "image_id",
    "dataset",
    "protection",
    "concept_type",
    "concept_label",
    "depth",
    "protected_path",
    "concept_similarity",
    "clip_concept_sim",
    "dino_concept_sim",
    "face_detected",
]

def load_original_map(metadata_csv: Path) -> dict[tuple, dict]:
    """Map (dataset, image_stem) -> none-protection metadata row."""
    rows = read_csv(metadata_csv)
    mapping = {}
    for r in rows:
        if r["protection"] == "none":
            mapping[(r["dataset"], normalize_image_id(r["image_id"]))] = r
    return mapping


def lookup_original(originals: dict[tuple, dict], dataset: str, image_id: str) -> dict | None:
    return originals.get((dataset, normalize_image_id(image_id)))


def compute_level_metrics(
    records: list[dict],
    transform_key: str,
    originals: dict[tuple, dict],
    computer: MetricComputer,
) -> list[dict]:
    out = []
    for rec in tqdm(records, desc=f"metrics-{transform_key}"):
        before = load_rgb(Path(rec["input_path"]))
        after = load_rgb(Path(rec["output_path"]))
        transform = rec.get(transform_key) or rec.get("edit_type") or rec.get("transform_chain", "")

        pair = computer.compute_pair_metrics(before, after)
        if rec["dataset"] == "arts":
            pair["dino_sim"] = math.nan

        orig = lookup_original(originals, rec["dataset"], rec["image_id"])
        concept = {
            "concept_similarity": math.nan,
            "clip_concept_sim": math.nan,
            "dino_concept_sim": math.nan,
            "face_detected": "",
        }
        if orig:
            original_img = load_rgb(Path(orig["input_path"]))
            concept = computer.compute_concept_metrics(
                rec["dataset"], original_img, after, rec["concept_label"]
            )
            concept["face_detected"] = str(concept["face_detected"])

        out.append(
            {
                "image_id": rec["image_id"],
                "dataset": rec["dataset"],
                "protection": rec["protection"],
                "concept_type": rec["concept_type"],
                "concept_label": rec["concept_label"],
                "editor": rec.get("editor", ""),
                "transform": transform,
                "chain_id": rec.get("chain_id", ""),
                "depth": rec.get("depth", ""),
                "psnr": pair["psnr"],
                "ssim": pair["ssim"],
                "lpips": pair["lpips"],
                "clip_img_sim": pair["clip_img_sim"],
                "dino_sim": pair["dino_sim"],
                "concept_similarity": concept["concept_similarity"],
                "clip_concept_sim": concept["clip_concept_sim"],
                "dino_concept_sim": concept["dino_concept_sim"],
                "face_detected": concept["face_detected"],
            }
        )
    return out


def collect_depth0_pairs(level3_records: list[dict]) -> list[dict]:
    """Unique (image, protection) with protected image path from depth-1 chain inputs."""
    seen: set[tuple] = set()
    pairs: list[dict] = []
    for rec in level3_records:
        if str(rec.get("depth")) != "1":
            continue
        key = (rec["dataset"], rec["protection"], rec["image_id"])
        if key in seen:
            continue
        seen.add(key)
        pairs.append(
            {
                "image_id": rec["image_id"],
                "dataset": rec["dataset"],
                "protection": rec["protection"],
                "concept_type": rec["concept_type"],
                "concept_label": rec["concept_label"],
                "protected_path": rec["input_path"],
            }
        )
    return pairs


def compute_depth0_metrics(
    pairs: list[dict],
    originals: dict[tuple, dict],
    computer: MetricComputer,
) -> list[dict]:
    out = []
    for pair in tqdm(pairs, desc="metrics-depth0"):
        protected_path = Path(pair["protected_path"])
        if not protected_path.exists():
            img_dir = protection_dir(pair["protection"]) / pair["dataset"]
            alt = resolve_image_path(img_dir, pair["image_id"])
            if alt is None:
                print(f"[Skip depth0] missing protected image for {pair['image_id']} ({pair['protection']})")
                continue
            protected_path = alt

        orig = lookup_original(originals, pair["dataset"], pair["image_id"])
        if not orig:
            print(f"[Skip depth0] missing original metadata for {pair['image_id']}")
            continue

        original_img = load_rgb(Path(orig["input_path"]))
        protected_img = load_rgb(protected_path)
        concept = computer.compute_concept_metrics(
            pair["dataset"], original_img, protected_img, pair["concept_label"]
        )
        out.append(
            {
                "image_id": pair["image_id"],
                "dataset": pair["dataset"],
                "protection": pair["protection"],
                "concept_type": pair["concept_type"],
                "concept_label": pair["concept_label"],
                "depth": 0,
                "protected_path": str(protected_path.resolve()),
                "concept_similarity": concept["concept_similarity"],
                "clip_concept_sim": concept["clip_concept_sim"],
                "dino_concept_sim": concept["dino_concept_sim"],
                "face_detected": str(concept["face_detected"]),
            }
        )
    return out


def filter_records_by_editor(records: list[dict], editor: str | None) -> list[dict]:
    if not editor:
        return records
    return [r for r in records if (r.get("editor") or "") == editor]


def merge_editor_metrics(
    base_path: Path,
    new_rows: list[dict],
    editor: str,
    fields: list[str],
) -> None:
    """Replace rows for one editor in an existing metrics CSV; keep all other editors."""
    new_df = pd.DataFrame(new_rows)
    if base_path.exists():
        base_df = pd.read_csv(base_path)
        if "editor" in base_df.columns:
            kept = base_df[base_df["editor"] != editor]
        else:
            kept = base_df
        merged = pd.concat([kept, new_df], ignore_index=True)
    else:
        merged = new_df
    write_csv(base_path, fields, merged.to_dict("records"))
    print(
        f"Merged {len(new_rows)} {editor} rows into {base_path}: "
        f"{len(merged)} total rows ({len(merged) - len(new_rows)} from other editors)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_metadata", type=Path, default=PROJECT_ROOT / "records" / "all_metadata.csv")
    parser.add_argument("--level1_records", type=Path, default=PROJECT_ROOT / "records" / "one_shot_transformation_records.csv")
    parser.add_argument("--level2_records", type=Path, default=PROJECT_ROOT / "records" / "one_shot_semantic_records.csv")
    parser.add_argument("--level3_records", type=Path, default=PROJECT_ROOT / "records" / "sequential_simple_records.csv")
    parser.add_argument("--out_dir", type=Path, default=PROJECT_ROOT / "records" / "metrics")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip_level1", action="store_true")
    parser.add_argument("--skip_level2", action="store_true")
    parser.add_argument("--skip_level3", action="store_true")
    parser.add_argument(
        "--editor",
        default=None,
        help="Only compute metrics for this editor (e.g. ip2p). Merges into the default "
        "level2/level3 metrics CSVs without overwriting other editors.",
    )
    parser.add_argument(
        "--level1-datasets",
        nargs="+",
        default=None,
        help="Only compute Level-1 metrics for these datasets (e.g. faces).",
    )
    parser.add_argument(
        "--level1-metrics-out",
        type=Path,
        default=None,
        help="Optional output path for Level-1 metrics (default: out_dir/level1_metrics.csv).",
    )
    parser.add_argument(
        "--level2-datasets",
        nargs="+",
        default=None,
        help="Only compute Level-2 metrics for these datasets (e.g. faces).",
    )
    parser.add_argument(
        "--level2-metrics-out",
        type=Path,
        default=None,
        help="Optional output path for Level-2 metrics (default: out_dir/level2_metrics.csv).",
    )
    parser.add_argument(
        "--level3-datasets",
        nargs="+",
        default=None,
        help="Only compute Level-3 metrics for these datasets (e.g. faces arts).",
    )
    parser.add_argument(
        "--level3-metrics-out",
        type=Path,
        default=None,
        help="Optional output path for Level-3 metrics (default: out_dir/level3_metrics.csv).",
    )
    parser.add_argument(
        "--compute-depth0",
        action="store_true",
        help="Compute depth-0 concept_similarity for protected (unedited) images.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    originals = load_original_map(args.original_metadata)
    computer = MetricComputer(device=args.device)
    for skip, path in ((args.skip_level1, args.level1_records), (args.skip_level2, args.level2_records),
                       (args.skip_level3, args.level3_records)):
        if not skip and not path.exists():
            print(f"[skip] {path} not found")

    if not args.skip_level1 and args.level1_records.exists():
        l1_records = read_csv(args.level1_records)
        if args.level1_datasets:
            l1_records = [r for r in l1_records if r["dataset"] in args.level1_datasets]
        if l1_records:
            l1 = compute_level_metrics(l1_records, "transform_chain", originals, computer)
            l1_out = args.level1_metrics_out or (args.out_dir / "level1_metrics.csv")
            if args.level1_datasets and l1_out.name == "level1_metrics.csv":
                shard_out = args.out_dir / f"level1_metrics_{'_'.join(args.level1_datasets)}.csv"
                write_csv(shard_out, METRIC_FIELDS, l1)
                print(f"Wrote {len(l1)} Level-1 metric rows to {shard_out}")
                base_path = args.out_dir / "level1_metrics.csv"
                if base_path.exists():
                    base_df = pd.read_csv(base_path)
                    base_df = base_df[~base_df["dataset"].isin(args.level1_datasets)]
                    merged = pd.concat([base_df, pd.DataFrame(l1)], ignore_index=True)
                    write_csv(base_path, METRIC_FIELDS, merged.to_dict("records"))
                    print(f"Merged into {base_path}: {len(merged)} total rows")
                else:
                    write_csv(l1_out, METRIC_FIELDS, l1)
                    print(f"Wrote {len(l1)} Level-1 metric rows to {l1_out}")
            else:
                write_csv(l1_out, METRIC_FIELDS, l1)
                print(f"Wrote {len(l1)} Level-1 metric rows to {l1_out}")

    if not args.skip_level2 and args.level2_records.exists():
        l2_records = read_csv(args.level2_records)
        l2_records = filter_records_by_editor(l2_records, args.editor)
        if args.level2_datasets:
            l2_records = [r for r in l2_records if r["dataset"] in args.level2_datasets]
        if l2_records:
            l2 = compute_level_metrics(l2_records, "edit_type", originals, computer)
            l2_out = args.level2_metrics_out or (args.out_dir / "level2_metrics.csv")
            if args.editor and l2_out == args.out_dir / "level2_metrics.csv":
                merge_editor_metrics(l2_out, l2, args.editor, METRIC_FIELDS)
            elif args.level2_datasets and l2_out.name == "level2_metrics.csv":
                shard_out = args.out_dir / f"level2_metrics_{'_'.join(args.level2_datasets)}.csv"
                write_csv(shard_out, METRIC_FIELDS, l2)
                print(f"Wrote {len(l2)} Level-2 metric rows to {shard_out}")
                base_path = args.out_dir / "level2_metrics.csv"
                if base_path.exists():
                    base_df = pd.read_csv(base_path)
                    base_df = base_df[~base_df["dataset"].isin(args.level2_datasets)]
                    merged = pd.concat([base_df, pd.DataFrame(l2)], ignore_index=True)
                    write_csv(base_path, METRIC_FIELDS, merged.to_dict("records"))
                    print(f"Merged into {base_path}: {len(merged)} total rows")
                else:
                    write_csv(l2_out, METRIC_FIELDS, l2)
                    print(f"Wrote {len(l2)} Level-2 metric rows to {l2_out}")
            else:
                write_csv(l2_out, METRIC_FIELDS, l2)
                print(f"Wrote {len(l2)} Level-2 metric rows to {l2_out}")

    if not args.skip_level3 and args.level3_records.exists():
        l3_records = read_csv(args.level3_records)
        l3_records = filter_records_by_editor(l3_records, args.editor)
        if args.level3_datasets:
            l3_records = [r for r in l3_records if r["dataset"] in args.level3_datasets]
        if l3_records:
            l3 = compute_level_metrics(l3_records, "edit_type", originals, computer)
            l3_out = args.level3_metrics_out or (args.out_dir / "level3_metrics.csv")
            if args.editor and l3_out == args.out_dir / "level3_metrics.csv":
                merge_editor_metrics(l3_out, l3, args.editor, METRIC_FIELDS)
            else:
                write_csv(l3_out, METRIC_FIELDS, l3)
                print(f"Wrote {len(l3)} Level-3 metric rows to {l3_out}")

    if args.compute_depth0 and args.level3_records.exists():
        l3_for_depth0 = read_csv(args.level3_records)
        l3_for_depth0 = filter_records_by_editor(l3_for_depth0, args.editor)
        pairs = collect_depth0_pairs(l3_for_depth0)
        depth0 = compute_depth0_metrics(pairs, originals, computer)
        depth0_path = args.out_dir / "level3_depth0_metrics.csv"
        write_csv(depth0_path, DEPTH0_FIELDS, depth0)
        print(f"Wrote {len(depth0)} depth-0 rows to {depth0_path}")

    print(f"Metrics written to {args.out_dir}")


if __name__ == "__main__":
    main()
