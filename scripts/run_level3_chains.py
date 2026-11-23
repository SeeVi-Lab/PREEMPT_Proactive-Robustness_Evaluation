#!/usr/bin/env python3
"""Run sequential semantic edit chains (simple chains C1–C4)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from utils.editor import EDITORS, get_editor
from utils.paths import (
    CONCEPT_TYPES,
    DATASETS,
    EVAL_ROOT,
    SD21_EDITOR,
    PROMPTS_FILE,
    PROTECTIONS,
    PROJECT_ROOT,
    level3_out,
    object_concept_label,
    protection_dir,
    resolve_image_path,
)
from utils.records import append_csv, read_csv, write_csv
from utils.transforms import load_edit_input, save_rgb

LEVEL3_FIELDS = [
    "image_id",
    "dataset",
    "concept_type",
    "concept_label",
    "protection",
    "level",
    "chain_id",
    "depth",
    "edit_type",
    "prompt",
    "editor",
    "input_path",
    "output_path",
    "seed",
]

EDIT_TYPE_TO_CLASS = {
    "content_edit": "Content",
    "attribute_edit": "Attribute",
    "style_edit": "Style",
    "spatial_edit": "Spatial",
}


def load_prompts(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_chains(path: Path) -> dict:
    with path.open() as f:
        data = yaml.safe_load(f)
    return data["chains"]


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


def resolve_prompt(entry: dict, edit_type: str, prompt_index: int) -> str | None:
    cls_name = EDIT_TYPE_TO_CLASS.get(edit_type)
    if cls_name is None:
        return None
    prompt_list = entry["prompts"].get(cls_name) or []
    if prompt_index < 0 or prompt_index >= len(prompt_list):
        return None
    prompt = prompt_list[prompt_index]
    if prompt is None or (isinstance(prompt, str) and not prompt.strip()):
        return None
    return prompt


def existing_keys(record_path: Path) -> set[tuple]:
    rows = read_csv(record_path)
    return {
        (
            r.get("editor") or SD21_EDITOR,
            r["protection"],
            r["dataset"],
            r["image_id"],
            r["chain_id"],
            str(r["depth"]),
        )
        for r in rows
    }


def chain_output_path(
    protection: str, dataset: str, chain_id: str, depth: int, image_id: str, editor: str | None = None
) -> Path:
    return level3_out(protection, dataset, chain_id, depth, editor) / f"{Path(image_id).stem}.png"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=PROMPTS_FILE)
    parser.add_argument("--chains", type=Path, default=PROJECT_ROOT / "config" / "chains_simple.yaml")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--protections", nargs="+", default=list(PROTECTIONS))
    parser.add_argument("--chain_ids", nargs="+", default=None, help="Subset of chains, e.g. C1 C2")
    parser.add_argument("--editor", default=SD21_EDITOR, choices=sorted(EDITORS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--out_root", type=Path, default=EVAL_ROOT / "level3")
    parser.add_argument("--record", type=Path, default=PROJECT_ROOT / "records" / "level3_records.csv")
    parser.add_argument("--max_depth", type=int, default=0, help="0 = use all steps in chain config")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output images even when --resume is set",
    )
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    chains = load_chains(args.chains)
    if args.chain_ids:
        chains = {k: v for k, v in chains.items() if k in args.chain_ids}
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
            for chain_id, chain_cfg in chains.items():
                steps = chain_cfg["steps"]
                max_depth = args.max_depth or len(steps)
                steps = steps[:max_depth]

                for image_id in tqdm(images, desc=f"L3 {protection}/{dataset}/{chain_id}"):
                    entry = find_prompt_entry(image_id, dataset, prompts)
                    if entry is None:
                        print(f"[Missing prompts] {image_id}")
                        continue

                    concept_label = (
                        object_concept_label(image_id)
                        if dataset == "objects"
                        else ("identity" if dataset == "faces" else Path(image_id).stem.split("-")[0].replace("_", " "))
                    )

                    protected_path = resolve_image_path(img_dir, image_id)
                    if protected_path is None:
                        continue

                    current_path = protected_path
                    current_image = load_edit_input(current_path)

                    for depth, step in enumerate(steps, start=1):
                        key = (args.editor, protection, dataset, image_id, chain_id, str(depth))
                        out_path = chain_output_path(
                            protection, dataset, chain_id, depth, image_id, args.editor
                        )

                        if key in done:
                            if out_path.exists():
                                current_path = out_path
                                current_image = load_edit_input(current_path)
                                continue
                        elif out_path.exists() and args.resume and not args.force:
                            current_path = out_path
                            current_image = load_edit_input(current_path)
                            continue

                        prompt = resolve_prompt(entry, step["edit_type"], step["prompt_index"])
                        if prompt is None:
                            print(
                                f"[Missing prompt] {image_id} {chain_id} depth={depth} "
                                f"{step['edit_type']} idx={step['prompt_index']}"
                            )
                            break

                        edited = editor.edit(
                            current_image,
                            prompt,
                            num_inference_steps=args.num_inference_steps,
                            seed=args.seed,
                        )
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        save_rgb(edited, out_path)

                        rec = {
                            "image_id": image_id,
                            "dataset": dataset,
                            "concept_type": CONCEPT_TYPES[dataset],
                            "concept_label": concept_label,
                            "protection": protection,
                            "level": 3,
                            "chain_id": chain_id,
                            "depth": depth,
                            "edit_type": step["edit_type"],
                            "prompt": prompt,
                            "editor": args.editor,
                            "input_path": str(current_path.resolve()),
                            "output_path": str(out_path.resolve()),
                            "seed": args.seed,
                        }
                        new_records.append(rec)
                        append_csv(args.record, LEVEL3_FIELDS, [rec])
                        current_path = out_path
                        current_image = edited
                        count += 1

                        if args.limit and count >= args.limit:
                            print(f"Stopped at limit {args.limit}, appended {count} records")
                            return

    print(f"Added {len(new_records)} level3 records this run")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
