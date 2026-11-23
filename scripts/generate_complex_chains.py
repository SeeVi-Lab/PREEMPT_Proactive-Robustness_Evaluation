#!/usr/bin/env python3
"""Sample the mixed chains CX1–CX4 (seed 42) and write the chain specification CSV."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import json

from utils.complex_chain import (
    chain_seed,
    sample_complex_chain,
)
from utils.paths import (
    COMPLEX_CHAIN_DEPTH,
    COMPLEX_CHAIN_IDS,
    CONCEPT_TYPES,
    DATASETS,
    PROTECTIONS,
    PROMPTS_FILE,
    PROJECT_ROOT,
    object_concept_label,
    protection_dir,
    resolve_image_path,
)

EDIT_CLASS_FOR_TYPE = {
    "content_edit": "Content",
    "attribute_edit": "Attribute",
    "style_edit": "Style",
    "spatial_edit": "Spatial",
}


def load_prompts(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def find_prompt_entry(image_id: str, dataset: str, prompts: dict) -> dict | None:
    if image_id in prompts:
        return prompts[image_id]
    stem = Path(image_id).stem
    for key, val in prompts.items():
        key_stem = Path(key).stem
        if key_stem == stem:
            return val
        if dataset == "faces" and key_stem in stem:
            return val
        if dataset == "arts" and key_stem == stem:
            return val
    return None


def concept_label(dataset: str, image_id: str) -> str:
    if dataset == "objects":
        return object_concept_label(image_id)
    if dataset == "arts":
        stem = Path(image_id).stem
        return stem.split("-", 1)[0].replace("_", " ") if "-" in stem else stem.replace("_", " ")
    return "identity"


def semantic_prompt(entry: dict, edit_type: str) -> str:
    cls_name = EDIT_CLASS_FOR_TYPE[edit_type]
    prompt_list = entry["prompts"].get(cls_name) or []
    if not prompt_list:
        raise ValueError(f"no prompts for {cls_name}")
    return str(prompt_list[0]).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=PROMPTS_FILE)
    parser.add_argument("--editor", default="ip2p")
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--out-chains", type=Path, default=PROJECT_ROOT / "records" / "complex_chains.csv")
    parser.add_argument("--out-flip", type=Path, default=PROJECT_ROOT / "records" / "complex_chain_flip_faces.csv")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--protections", nargs="+", default=list(PROTECTIONS))
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    chain_rows: list[dict] = []
    flip_rows: list[dict] = []

    for protection in args.protections:
        for dataset in args.datasets:
            img_dir = protection_dir(protection) / dataset
            if not img_dir.exists():
                print(f"[skip] missing {img_dir}")
                continue
            images = sorted(
                p.name for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            for image_id in images:
                if resolve_image_path(img_dir, image_id) is None:
                    continue
                entry = find_prompt_entry(image_id, dataset, prompts)
                if entry is None:
                    print(f"[skip] no prompts for {image_id}")
                    continue

                for chain_id in COMPLEX_CHAIN_IDS:
                    seed = chain_seed(image_id, chain_id, args.global_seed)
                    rng = random.Random(seed)
                    sampled = sample_complex_chain(rng, depth=COMPLEX_CHAIN_DEPTH)
                    steps: list[dict] = []
                    for kind, op in sampled:
                        if kind == "semantic":
                            steps.append(
                                {
                                    "kind": "semantic",
                                    "op": op,
                                    "prompt": semantic_prompt(entry, op),
                                }
                            )
                        else:
                            steps.append({"kind": "physical", "op": op, "prompt": ""})

                    row = {
                        "image_id": image_id,
                        "dataset": dataset,
                        "concept_type": CONCEPT_TYPES[dataset],
                        "concept_label": concept_label(dataset, image_id),
                        "protection": protection,
                        "editor": args.editor,
                        "chain_id": chain_id,
                        "chain_seed": seed,
                        "global_seed": args.global_seed,
                    }
                    for i, step in enumerate(steps, start=1):
                        row[f"step{i}_kind"] = step["kind"]
                        row[f"step{i}_op"] = step["op"]
                        row[f"step{i}_prompt"] = step["prompt"]
                    row["has_horizontal_flip"] = any(
                        s["op"] == "horizontal_flip" for s in steps
                    )
                    chain_rows.append(row)

                    if dataset == "faces" and row["has_horizontal_flip"]:
                        for depth, step in enumerate(steps, start=1):
                            if step["op"] == "horizontal_flip":
                                flip_rows.append(
                                    {
                                        "image_id": image_id,
                                        "dataset": dataset,
                                        "protection": protection,
                                        "chain_id": chain_id,
                                        "chain_seed": seed,
                                        "flip_depth": depth,
                                        "editor": args.editor,
                                    }
                                )

    args.out_chains.parent.mkdir(parents=True, exist_ok=True)
    if not chain_rows:
        raise SystemExit("no chains generated")

    fields = list(chain_rows[0].keys())
    with args.out_chains.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(chain_rows)
    print(f"Wrote {len(chain_rows)} chains -> {args.out_chains}")

    if flip_rows:
        flip_fields = list(flip_rows[0].keys())
        with args.out_flip.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=flip_fields)
            w.writeheader()
            w.writerows(flip_rows)
        print(f"Wrote {len(flip_rows)} flip-face rows -> {args.out_flip}")


if __name__ == "__main__":
    main()
