"""PREEMPT framework taxonomy and record-path helpers."""

from __future__ import annotations

from pathlib import Path

from utils.paths import CONFIG_DIR, RECORDS_DIR

# --- Framework blocks -------------------------------------------------------

ONE_SHOT_TRANSFORMATION = "one_shot_transformation"
ONE_SHOT_SEMANTIC = "one_shot_semantic"
SEQUENTIAL_SIMPLE = "sequential_simple"
SEQUENTIAL_COMPLEX = "sequential_complex"

SIMPLE_CHAIN_IDS = ("C1", "C2", "C3", "C4")
COMPLEX_CHAIN_IDS = ("CX1", "CX2", "CX3", "CX4")

FRAMEWORK: dict[str, dict] = {
    "one_shot": {
        "transformation": {
            "id": ONE_SHOT_TRANSFORMATION,
            "eval_dir": "level1",
            "chains": None,
        },
        "semantic": {
            "id": ONE_SHOT_SEMANTIC,
            "eval_dir": "level2",
            "chains": None,
        },
    },
    "sequential": {
        "simple": {
            "id": SEQUENTIAL_SIMPLE,
            "eval_dir": "level3",
            "chains": SIMPLE_CHAIN_IDS,
        },
        "complex": {
            "id": SEQUENTIAL_COMPLEX,
            "eval_dir": "complex_chain",
            "chains": COMPLEX_CHAIN_IDS,
        },
    },
}

# --- Record CSV paths (PREEMPT naming) --------------------------------------

RECORD_PATHS: dict[str, Path] = {
    ONE_SHOT_TRANSFORMATION: RECORDS_DIR / "one_shot_transformation_records.csv",
    ONE_SHOT_SEMANTIC: RECORDS_DIR / "one_shot_semantic_records.csv",
    SEQUENTIAL_SIMPLE: RECORDS_DIR / "sequential_simple_records.csv",
    SEQUENTIAL_COMPLEX: RECORDS_DIR / "sequential_complex_records.csv",
}

CHAIN_SPEC_PATHS: dict[str, Path] = {
    SEQUENTIAL_COMPLEX: RECORDS_DIR / "sequential_complex_chains.csv",
}

CONFIG_PATHS: dict[str, Path] = {
    SEQUENTIAL_SIMPLE: CONFIG_DIR / "chains_simple.yaml",
}

def record_path(block_id: str) -> Path:
    """Return the record CSV for a PREEMPT block."""
    return RECORD_PATHS[block_id]


def metrics_dir(block_id: str) -> Path:
    return RECORDS_DIR / "metrics" / block_id
