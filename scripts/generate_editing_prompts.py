#!/usr/bin/env python3
"""Generate semantic editing prompts for each image with a vision-language model.

For every image, GPT-4.1 mini reads the image and the instruction in
config/vlm_prompt.txt and returns up to four prompts for each category
(Content, Attribute, Style, Spatial). The result is written in the format of
config/editing_prompts.json, keyed by image file name.

The image folder holds one sub-folder per domain; the sub-folder name is saved
as the "domain" of each image:

    <image-root>/
      faces/    *.png | *.jpg
      arts/     ...
      objects/  ...

Images already in the output file are skipped, so the script can be re-run
to add new images. Requires `pip install openai` and OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

from utils.paths import CONFIG_DIR, PROMPTS_FILE, PROTECTION_ROOT

CATEGORIES = ("Content", "Attribute", "Style", "Spatial")
IMAGE_SUFFIXES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def encode_image(path: Path) -> str:
    mime = IMAGE_SUFFIXES[path.suffix.lower()]
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def parse_json_reply(text: str) -> dict | None:
    """Take the first {...} block of the model reply and parse it."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(match.group(0) if match else text)
    except json.JSONDecodeError:
        return None


def generate_prompts(client, image_path: Path, instruction: str, model: str, temperature: float) -> dict | None:
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction},
                    {"type": "input_image", "image_url": encode_image(image_path)},
                ],
            }
        ],
        temperature=temperature,
    )
    return parse_json_reply(response.output_text.strip())


def list_images(image_root: Path) -> list[tuple[str, Path]]:
    images = []
    for domain_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        for path in sorted(domain_dir.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                images.append((domain_dir.name, path))
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-root", type=Path, default=PROTECTION_ROOT / "Clean",
                        help="folder with one sub-folder per domain (default: $PREEMPT_DATA_ROOT/Protection/Clean)")
    parser.add_argument("--out", type=Path, default=PROMPTS_FILE, help="output JSON (default: config/editing_prompts.json)")
    parser.add_argument("--instruction", type=Path, default=CONFIG_DIR / "vlm_prompt.txt", help="instruction given to the VLM")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--overwrite", action="store_true", help="regenerate images already in the output file")
    args = parser.parse_args()

    from openai import OpenAI  # imported here so --help works without the package

    client = OpenAI()  # reads OPENAI_API_KEY
    instruction = args.instruction.read_text()
    prompts = json.loads(args.out.read_text()) if args.out.exists() else {}

    todo = [(d, p) for d, p in list_images(args.image_root) if args.overwrite or p.name not in prompts]
    print(f"{len(todo)} images to process, {len(prompts)} already in {args.out}")

    failed, incomplete = [], []
    for domain, path in tqdm(todo, desc="Images"):
        result = generate_prompts(client, path, instruction, args.model, args.temperature)
        if result is None:
            failed.append(path.name)
            continue
        result = {c: result.get(c) for c in CATEGORIES}
        if any(not result[c] for c in CATEGORIES):
            incomplete.append(path.name)
        prompts[path.name] = {"domain": domain, "prompts": result}
        args.out.write_text(json.dumps(prompts, indent=2))  # save after each image

    print(f"Wrote {len(prompts)} images to {args.out}")
    if incomplete:
        print(f"{len(incomplete)} images have an empty category; edit them by hand or re-run with --overwrite:")
        print("  " + "\n  ".join(incomplete))
    if failed:
        print(f"{len(failed)} replies could not be parsed as JSON and were skipped; re-run to retry:", file=sys.stderr)
        print("  " + "\n  ".join(failed), file=sys.stderr)


if __name__ == "__main__":
    main()
