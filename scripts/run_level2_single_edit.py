#!/usr/bin/env python3
"""Run the one-shot semantic edits (one edit per prompt)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from utils.editor import EDITORS, get_editor
from utils.paths import (
    CONCEPT_TYPES,
    DATASETS,
    EDIT_TYPE_MAP,
    SD21_EDITOR,
    PROMPTS_FILE,
    PROTECTIONS,
    PROJECT_ROOT,
    level2_output_path,
    object_concept_label,
    protection_dir,
    resolve_image_path,
)
from utils.records import append_csv, read_csv, write_csv
from utils.transforms import load_edit_input, save_rgb

LEVEL2_FIELDS = [
    "image_id",
    "dataset",
    "concept_type",
    "concept_label",
    "protection",
    "level",
    "depth",
    "edit_type",
    "prompt_index",
    "prompt",
    "editor",
    "input_path",
    "output_path",
    "seed",
]

PROMPT_CLASS_TO_EDIT = {v.replace("_edit", "").title(): k for k, v in {
    "Content": "content_edit",
    "Attribute": "attribute_edit",
    "Style": "style_edit",
    "Spatial": "spatial_edit",
}.items()}


def load_prompts(path: Path) -> dict:
    with path.open() as f:
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
    return None


def existing_keys(record_path: Path) -> set[tuple]:
    rows = read_csv(record_path)
    return {
        (
            r.get("editor") or SD21_EDITOR,
            r["protection"],
            r["dataset"],
            r["image_id"],
            r["edit_type"],
            r["prompt"],
        )
        for r in rows
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=PROMPTS_FILE)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--protections", nargs="+", default=list(PROTECTIONS))
    parser.add_argument("--editor", default=SD21_EDITOR, choices=sorted(EDITORS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--record", type=Path, default=PROJECT_ROOT / "records" / "level2_records.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output images even when --resume is set",
    )
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    done = existing_keys(args.record) if args.resume and not args.force and args.record.exists() else set()

    editor = get_editor(args.editor, args.device)

    new_records = []
    count = 0

    for protection in args.protections:
        for dataset in args.datasets:
            img_dir = protection_dir(protection) / dataset
            if not img_dir.exists():
                print(f"[Skip] missing {img_dir}")
                continue
            images = sorted(
                p.name for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            for image_id in tqdm(images, desc=f"L2 {protection}/{dataset}"):
                entry = find_prompt_entry(image_id, dataset, prompts)
                if entry is None:
                    print(f"[Missing prompts] {image_id}")
                    continue
                input_path = resolve_image_path(img_dir, image_id)
                if input_path is None:
                    continue
                image = load_edit_input(input_path)
                concept_label = (
                    object_concept_label(image_id)
                    if dataset == "objects"
                    else ("identity" if dataset == "faces" else Path(image_id).stem.split("-")[0].replace("_", " "))
                )

                for cls_name, prompt_list in entry["prompts"].items():
                    edit_type = EDIT_TYPE_MAP.get(cls_name)
                    if edit_type is None or not prompt_list:
                        continue
                    for prompt_index, prompt in enumerate(prompt_list):
                        key = (args.editor, protection, dataset, image_id, edit_type, prompt)
                        if key in done:
                            continue
                        out_path = level2_output_path(
                            protection, dataset, edit_type, image_id, prompt_index, args.editor
                        )
                        if out_path.exists() and args.resume and not args.force:
                            if key not in done:
                                append_csv(
                                    args.record,
                                    LEVEL2_FIELDS,
                                    [{
                                        "image_id": image_id,
                                        "dataset": dataset,
                                        "concept_type": CONCEPT_TYPES[dataset],
                                        "concept_label": concept_label,
                                        "protection": protection,
                                        "level": 2,
                                        "depth": 1,
                                        "edit_type": edit_type,
                                        "prompt_index": prompt_index,
                                        "prompt": prompt,
                                        "editor": args.editor,
                                        "input_path": str(input_path.resolve()),
                                        "output_path": str(out_path.resolve()),
                                        "seed": args.seed,
                                    }],
                                )
                            continue
                        edited = editor.edit(
                            image,
                            prompt,
                            num_inference_steps=args.num_inference_steps,
                            seed=args.seed,
                        )
                        save_rgb(edited, out_path)
                        rec = {
                            "image_id": image_id,
                            "dataset": dataset,
                            "concept_type": CONCEPT_TYPES[dataset],
                            "concept_label": concept_label,
                            "protection": protection,
                            "level": 2,
                            "depth": 1,
                            "edit_type": edit_type,
                            "prompt_index": prompt_index,
                            "prompt": prompt,
                            "editor": args.editor,
                            "input_path": str(input_path.resolve()),
                            "output_path": str(out_path.resolve()),
                            "seed": args.seed,
                        }
                        new_records.append(rec)
                        append_csv(args.record, LEVEL2_FIELDS, [rec])
                        count += 1
                        if args.limit and count >= args.limit:
                            print(f"Stopped at limit {args.limit}, appended {count} records")
                            return

    print(f"Added {len(new_records)} level2 records this run")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
