"""Mixed chains CX1–CX4: sample semantic edits and image transformations into one chain."""

from __future__ import annotations

import random
import zlib
from pathlib import Path

import yaml

from utils.paths import EDIT_TYPE_MAP, LEVEL1_TRANSFORMS, PROJECT_ROOT

SEMANTIC_EDIT_TYPES = tuple(EDIT_TYPE_MAP.values())

PHYSICAL_OPS = tuple(LEVEL1_TRANSFORMS)
OPERATION_POOL = PHYSICAL_OPS + SEMANTIC_EDIT_TYPES


def chain_seed(image_id: str, chain_id: str, global_seed: int) -> int:
    key = f"{image_id}|{chain_id}|{global_seed}"
    return global_seed ^ zlib.crc32(key.encode("utf-8"))


def sample_complex_chain(
    rng: random.Random,
    *,
    depth: int = 4,
    require_semantic: bool = True,
    require_physical: bool = True,
) -> list[tuple[str, str]]:
    """Return [(kind, op), ...] where kind is 'physical' or 'semantic'."""
    if depth < 2:
        raise ValueError("depth must be >= 2 for mixed chains")
    if require_semantic and require_physical and depth < 2:
        raise ValueError("need depth >= 2 when both kinds are required")

    for _ in range(10_000):
        ops = rng.sample(list(OPERATION_POOL), depth)
        n_sem = sum(1 for op in ops if op in SEMANTIC_EDIT_TYPES)
        n_phy = depth - n_sem
        if require_semantic and n_sem < 1:
            continue
        if require_physical and n_phy < 1:
            continue
        return [
            ("semantic" if op in SEMANTIC_EDIT_TYPES else "physical", op)
            for op in ops
        ]
    raise RuntimeError("failed to sample a valid complex chain")


def load_physical_config(path: Path | None = None) -> dict:
    cfg_path = path or PROJECT_ROOT / "config" / "physical_transforms.yaml"
    with cfg_path.open() as f:
        return yaml.safe_load(f)["transforms"]
