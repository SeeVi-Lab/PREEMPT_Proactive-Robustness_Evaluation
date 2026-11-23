"""Path constants and helpers for PREEMPT."""

from __future__ import annotations

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT = os.environ.get("PREEMPT_DATA_ROOT", str(PROJECT_ROOT / "data"))
DATA_ROOT = Path(_DATA_ROOT)
PROTECTION_ROOT = DATA_ROOT / "Protection"
EVAL_ROOT = DATA_ROOT / "Eval"
RECORDS_DIR = PROJECT_ROOT / "records"
CONFIG_DIR = PROJECT_ROOT / "config"
PROMPTS_FILE = CONFIG_DIR / "editing_prompts.json"

DATASETS = ("faces", "arts", "objects")
CONCEPT_TYPES = {
    "faces": "identity",
    "arts": "style",
    "objects": "object",
}

# Clean images live under Protection/Clean; protection=none in records.
# Adding a protection method = drop its folder under PROTECTION_ROOT/{Name}/{dataset}/.
_STATIC_PROTECTION_DIRS = {
    "none": "Clean",
    "AdvDM": "AdvDM",
    "Mist": "Mist",
    "PhotoGuard": "PhotoGuard",
    "SDS(+)": "SDS(+)",
    "SDS(-)": "SDS(-)",
    "Diff-Protect": "Diff-Protect",
}


def discover_protection_dirs() -> dict[str, str]:
    """Scan PROTECTION_ROOT for protection methods (Clean -> none).

    Falls back to the static map when the data root is unavailable
    (e.g. on a machine without the dataset mounted).
    """
    if not PROTECTION_ROOT.exists():
        return dict(_STATIC_PROTECTION_DIRS)
    dirs: dict[str, str] = {}
    for child in sorted(PROTECTION_ROOT.iterdir()):
        if not child.is_dir() or "\\" in child.name:
            continue
        name = "none" if child.name == "Clean" else child.name
        dirs[name] = child.name
    return dirs or dict(_STATIC_PROTECTION_DIRS)


PROTECTION_DIRS = discover_protection_dirs()
PROTECTIONS = tuple(PROTECTION_DIRS)

# sd21 outputs live under SD21/ (same as ip2p -> ip2p/). sdxl is a separate editor.
SD21_EDITOR = "sd21"
SD21_EDITOR_DIR = "SD21"


def _editor_segment(editor: str | None) -> tuple[str, ...]:
    if editor is None or editor == SD21_EDITOR:
        return (SD21_EDITOR_DIR,)
    return (editor,)


EDIT_TYPE_MAP = {
    "Content": "content_edit",
    "Attribute": "attribute_edit",
    "Style": "style_edit",
    "Spatial": "spatial_edit",
}

LEVEL1_TRANSFORMS = (
    "jpeg_q50",
    "jpeg_q70",
    "gaussian_blur",
    "crop_resize",
    "brightness_contrast",
    "horizontal_flip",
)


def protection_dir(protection: str) -> Path:
    return PROTECTION_ROOT / PROTECTION_DIRS[protection]


def level1_out(protection: str, dataset: str, transform: str) -> Path:
    folder = PROTECTION_DIRS[protection]
    return EVAL_ROOT / "level1" / folder / dataset / transform


def level2_out(protection: str, dataset: str, edit_type: str, editor: str | None = None) -> Path:
    return EVAL_ROOT.joinpath("level2", *_editor_segment(editor), protection, dataset, edit_type)


def level2_output_path(
    protection: str,
    dataset: str,
    edit_type: str,
    image_id: str,
    prompt_index: int,
    editor: str | None = None,
) -> Path:
    """One output file per prompt: {stem}_p{idx}.png under the edit_type folder."""
    return level2_out(protection, dataset, edit_type, editor) / f"{image_stem(image_id)}_p{prompt_index}.png"


def level3_out(protection: str, dataset: str, chain_id: str, depth: int, editor: str | None = None) -> Path:
    return EVAL_ROOT.joinpath("level3", *_editor_segment(editor), protection, dataset, chain_id, f"depth_{depth}")


COMPLEX_CHAIN_IDS = ("CX1", "CX2", "CX3", "CX4")
COMPLEX_CHAIN_DEPTH = 4


def complex_chain_out(
    protection: str, dataset: str, chain_id: str, depth: int, editor: str | None = None
) -> Path:
    return EVAL_ROOT.joinpath(
        "complex_chain", *_editor_segment(editor), protection, dataset, chain_id, f"depth_{depth}"
    )


def image_stem(name: str) -> str:
    return Path(name).stem


def object_concept_label(image_id: str) -> str:
    """Extract concept label from object filename, e.g. backpack_01 -> backpack."""
    stem = image_stem(image_id)
    return re.sub(r"_\d+$", "", stem).replace("_", " ")


def normalize_image_id(image_id: str) -> str:
    return Path(image_id).stem


def resolve_image_path(directory: Path, image_id: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = directory / f"{image_id}{ext}"
        if candidate.exists() or candidate.is_symlink():
            return candidate
        candidate = directory / f"{Path(image_id).stem}{ext}"
        if candidate.exists() or candidate.is_symlink():
            return candidate
    return None
