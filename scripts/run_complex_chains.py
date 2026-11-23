#!/usr/bin/env python3
"""Run the mixed chains CX1–CX4: semantic edits and image transformations applied in sequence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm

from utils.complex_chain import load_physical_config
from utils.editor import EDITORS, get_editor
from utils.paths import (
    COMPLEX_CHAIN_DEPTH,
    PROJECT_ROOT,
    complex_chain_out,
    protection_dir,
    resolve_image_path,
)
from utils.records import append_csv, read_csv
from utils.transforms import apply_transform, load_edit_input, save_rgb

RECORD_FIELDS = [
    "image_id",
    "dataset",
    "concept_type",
    "concept_label",
    "protection",
    "level",
    "chain_id",
    "depth",
    "step_kind",
    "step_op",
    "edit_type",
    "prompt",
    "editor",
    "chain_seed",
    "input_path",
    "output_path",
    "seed",
]


def load_chains(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def chain_steps_from_row(row: dict) -> list[dict]:
    steps = []
    for i in range(1, COMPLEX_CHAIN_DEPTH + 1):
        steps.append(
            {
                "kind": row[f"step{i}_kind"],
                "op": row[f"step{i}_op"],
                "prompt": row.get(f"step{i}_prompt", ""),
            }
        )
    return steps


def existing_keys(record_path: Path) -> set[tuple]:
    rows = read_csv(record_path)
    return {
        (
            r.get("editor", ""),
            r["protection"],
            r["dataset"],
            r["image_id"],
            r["chain_id"],
            str(r["depth"]),
        )
        for r in rows
    }


def output_path(protection: str, dataset: str, chain_id: str, depth: int, image_id: str, editor: str) -> Path:
    return complex_chain_out(protection, dataset, chain_id, depth, editor) / f"{Path(image_id).stem}.png"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chains", type=Path, default=PROJECT_ROOT / "records" / "complex_chains.csv")
    parser.add_argument("--physical-config", type=Path, default=PROJECT_ROOT / "config" / "physical_transforms.yaml")
    parser.add_argument("--editor", default="ip2p", choices=sorted(EDITORS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--record", type=Path, default=PROJECT_ROOT / "records" / "complex_chain_records.csv")
    parser.add_argument("--protections", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--resume", action="store_true", help="Skip completed steps")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    chains = load_chains(args.chains)
    if args.protections:
        chains = [c for c in chains if c["protection"] in args.protections]
    if args.datasets:
        chains = [c for c in chains if c["dataset"] in args.datasets]
    if not chains:
        raise SystemExit("no chains to run")

    with args.physical_config.open() as f:
        transform_cfg = yaml.safe_load(f)["transforms"]

    done = existing_keys(args.record) if args.resume and not args.force and args.record.exists() else set()
    editor = get_editor(args.editor, args.device)

    added = 0
    for row in tqdm(chains, desc="complex chains"):
        protection = row["protection"]
        dataset = row["dataset"]
        image_id = row["image_id"]
        chain_id = row["chain_id"]
        chain_seed_val = int(row["chain_seed"])

        img_dir = protection_dir(protection) / dataset
        protected_path = resolve_image_path(img_dir, image_id)
        if protected_path is None:
            continue

        current_path = protected_path
        current_image = load_edit_input(current_path)
        steps = chain_steps_from_row(row)

        for depth, step in enumerate(steps, start=1):
            key = (args.editor, protection, dataset, image_id, chain_id, str(depth))
            out_path = output_path(protection, dataset, chain_id, depth, image_id, args.editor)

            if key in done and out_path.exists():
                current_path = out_path
                current_image = load_edit_input(current_path)
                continue
            if out_path.exists() and args.resume and not args.force:
                current_path = out_path
                current_image = load_edit_input(current_path)
                continue

            if step["kind"] == "physical":
                edited = apply_transform(current_image, step["op"], transform_cfg)
                prompt = ""
                edit_type = step["op"]
            else:
                prompt = str(step["prompt"]).strip()
                if not prompt:
                    print(f"[skip] empty prompt {image_id} {chain_id} depth={depth}")
                    break
                edited = editor.edit(
                    current_image,
                    prompt,
                    num_inference_steps=args.num_inference_steps,
                    seed=args.seed,
                )
                edit_type = step["op"]

            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_rgb(edited, out_path)

            rec = {
                "image_id": image_id,
                "dataset": dataset,
                "concept_type": row["concept_type"],
                "concept_label": row["concept_label"],
                "protection": protection,
                "level": "complex",
                "chain_id": chain_id,
                "depth": depth,
                "step_kind": step["kind"],
                "step_op": step["op"],
                "edit_type": edit_type,
                "prompt": prompt,
                "editor": args.editor,
                "chain_seed": chain_seed_val,
                "input_path": str(current_path.resolve()),
                "output_path": str(out_path.resolve()),
                "seed": args.seed,
            }
            append_csv(args.record, RECORD_FIELDS, [rec])
            current_path = out_path
            current_image = load_edit_input(out_path)
            added += 1

            if args.limit and added >= args.limit:
                print(f"Stopped at limit {args.limit}")
                return

    print(f"Added {added} complex chain records this run")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
