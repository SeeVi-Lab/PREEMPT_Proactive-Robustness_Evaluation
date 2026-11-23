#!/usr/bin/env python3
"""PREEMPT Sequential Edit — Chain of Simple Edit (C1–C4)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from utils.preempt import CONFIG_PATHS, RECORD_PATHS, SEQUENTIAL_SIMPLE, SIMPLE_CHAIN_IDS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
target = PROJECT_ROOT / "scripts" / "run_level3_chains.py"
defaults = [
    "--chains",
    str(CONFIG_PATHS[SEQUENTIAL_SIMPLE]),
    "--chain_ids",
    *SIMPLE_CHAIN_IDS,
    "--record",
    str(RECORD_PATHS[SEQUENTIAL_SIMPLE]),
]
sys.argv = [str(target), *defaults, *sys.argv[1:]]
runpy.run_path(str(target), run_name="__main__")
