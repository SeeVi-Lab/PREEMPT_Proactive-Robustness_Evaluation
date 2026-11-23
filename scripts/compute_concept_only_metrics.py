#!/usr/bin/env python3
"""Compute concept_similarity only (faces/arts/objects), without PSNR/SSIM/LPIPS/CLIP pair metrics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from scripts.compute_metrics import (
    METRIC_FIELDS,
    DEPTH0_FIELDS,
    collect_depth0_pairs,
    load_original_map,
    lookup_original,
)
from utils.metrics import MetricComputer
from utils.paths import PROJECT_ROOT
from utils.records import write_csv
from utils.transforms import load_rgb

KEY_COLS = ("image_id", "dataset", "protection", "chain_id", "depth", "editor")


def _empty_metric_row(rec: dict, transform: str, concept: dict) -> dict:
    return {
        "image_id": rec["image_id"],
        "dataset": rec["dataset"],
        "protection": rec["protection"],
        "concept_type": rec["concept_type"],
        "concept_label": rec["concept_label"],
        "editor": rec.get("editor", ""),
        "transform": transform,
        "chain_id": rec.get("chain_id", ""),
        "depth": rec.get("depth", ""),
        "psnr": math.nan,
        "ssim": math.nan,
        "lpips": math.nan,
        "clip_img_sim": math.nan,
        "dino_sim": math.nan,
        "concept_similarity": concept.get("concept_similarity", math.nan),
        "clip_concept_sim": concept.get("clip_concept_sim", math.nan),
        "dino_concept_sim": concept.get("dino_concept_sim", math.nan),
        "face_detected": str(concept.get("face_detected", "")),
    }


def _load_done(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    keys = set()
    for r in df.to_dict(orient="records"):
        keys.add(tuple(str(r.get(c, "")) for c in KEY_COLS))
    return keys


def _append_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        import csv

        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def score_records(
    records: list[dict],
    originals: dict,
    computer: MetricComputer,
    out_path: Path,
    *,
    resume: bool,
    flush_every: int,
) -> int:
    done = _load_done(out_path) if resume else set()
    if not resume and out_path.exists():
        out_path.unlink()

    buf: list[dict] = []
    n_new = 0
    for rec in tqdm(records, desc=f"concept:{out_path.name}"):
        key = tuple(str(rec.get(c, "")) for c in KEY_COLS)
        if key in done:
            continue
        orig = lookup_original(originals, rec["dataset"], rec["image_id"])
        concept = {
            "concept_similarity": math.nan,
            "clip_concept_sim": math.nan,
            "dino_concept_sim": math.nan,
            "face_detected": "",
        }
        if orig:
            original_img = load_rgb(Path(orig["input_path"]))
            after = load_rgb(Path(rec["output_path"]))
            concept = computer.compute_concept_metrics(
                rec["dataset"], original_img, after, rec["concept_label"]
            )
        transform = rec.get("edit_type") or rec.get("step_op") or rec.get("transform", "")
        buf.append(_empty_metric_row(rec, transform, concept))
        n_new += 1
        if len(buf) >= flush_every:
            _append_rows(out_path, buf, METRIC_FIELDS)
            buf.clear()
    if buf:
        _append_rows(out_path, buf, METRIC_FIELDS)
    return n_new


def score_depth0(
    pairs: list[dict],
    originals: dict,
    computer: MetricComputer,
    out_path: Path,
    *,
    resume: bool,
    flush_every: int,
) -> int:
    done = set()
    if resume and out_path.exists():
        df = pd.read_csv(out_path)
        for r in df.to_dict(orient="records"):
            done.add((str(r["image_id"]), str(r["dataset"]), str(r["protection"])))
    if not resume and out_path.exists():
        out_path.unlink()

    buf: list[dict] = []
    n_new = 0
    for pair in tqdm(pairs, desc=f"concept:{out_path.name}"):
        key = (str(pair["image_id"]), str(pair["dataset"]), str(pair["protection"]))
        if key in done:
            continue
        orig = lookup_original(originals, pair["dataset"], pair["image_id"])
        if not orig:
            continue
        original_img = load_rgb(Path(orig["input_path"]))
        protected_img = load_rgb(Path(pair["protected_path"]))
        concept = computer.compute_concept_metrics(
            pair["dataset"], original_img, protected_img, pair["concept_label"]
        )
        buf.append(
            {
                "image_id": pair["image_id"],
                "dataset": pair["dataset"],
                "protection": pair["protection"],
                "concept_type": pair["concept_type"],
                "concept_label": pair["concept_label"],
                "depth": 0,
                "protected_path": pair["protected_path"],
                "concept_similarity": concept["concept_similarity"],
                "clip_concept_sim": concept.get("clip_concept_sim", math.nan),
                "dino_concept_sim": concept.get("dino_concept_sim", math.nan),
                "face_detected": str(concept.get("face_detected", "")),
            }
        )
        n_new += 1
        if len(buf) >= flush_every:
            _append_rows(out_path, buf, DEPTH0_FIELDS)
            buf.clear()
    if buf:
        _append_rows(out_path, buf, DEPTH0_FIELDS)
    return n_new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "records" / "all_metadata.csv",
    )
    parser.add_argument(
        "--level3-records",
        type=Path,
        default=PROJECT_ROOT / "records" / "sequential_simple_records_perturbation.csv",
    )
    parser.add_argument(
        "--complex-records",
        type=Path,
        default=PROJECT_ROOT / "records" / "sequential_complex_records_perturbation.csv",
    )
    parser.add_argument(
        "--level3-out",
        type=Path,
        default=PROJECT_ROOT / "records" / "metrics" / "level3_metrics.csv",
    )
    parser.add_argument(
        "--complex-out",
        type=Path,
        default=PROJECT_ROOT / "records" / "metrics" / "complex_chain_metrics.csv",
    )
    parser.add_argument(
        "--depth0-out",
        type=Path,
        default=PROJECT_ROOT / "records" / "metrics" / "level3_depth0_metrics.csv",
    )
    parser.add_argument("--protections", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--skip-level3", action="store_true")
    parser.add_argument("--skip-complex", action="store_true")
    parser.add_argument("--skip-depth0", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--flush-every", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0, help="debug: max rows per split")
    args = parser.parse_args()

    if not args.metadata.exists():
        raise SystemExit(f"missing metadata: {args.metadata} (run prepare_metadata.py)")

    originals = load_original_map(args.metadata)
    computer = MetricComputer(device=args.device)

    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        if args.protections:
            df = df[df["protection"].isin(args.protections)]
        if args.datasets:
            df = df[df["dataset"].isin(args.datasets)]
        if args.limit and args.limit > 0:
            df = df.head(args.limit)
        return df

    if not args.skip_level3:
        l3 = _filter(pd.read_csv(args.level3_records))
        n = score_records(
            l3.to_dict(orient="records"),
            originals,
            computer,
            args.level3_out,
            resume=args.resume,
            flush_every=args.flush_every,
        )
        print(f"level3 new={n} -> {args.level3_out}")

    if not args.skip_complex:
        cx = _filter(pd.read_csv(args.complex_records))
        n = score_records(
            cx.to_dict(orient="records"),
            originals,
            computer,
            args.complex_out,
            resume=args.resume,
            flush_every=args.flush_every,
        )
        print(f"complex new={n} -> {args.complex_out}")

    if not args.skip_depth0:
        l3_all = _filter(pd.read_csv(args.level3_records))
        pairs = collect_depth0_pairs(l3_all.to_dict(orient="records"))
        if args.limit and args.limit > 0:
            pairs = pairs[: args.limit]
        n = score_depth0(
            pairs,
            originals,
            computer,
            args.depth0_out,
            resume=args.resume,
            flush_every=args.flush_every,
        )
        print(f"depth0 new={n} -> {args.depth0_out}")


if __name__ == "__main__":
    main()
