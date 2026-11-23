#!/usr/bin/env python3
"""Sample CX1–CX4 chain specs for PREEMPT Sequential Complex Edit."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from utils.preempt import CHAIN_SPEC_PATHS, SEQUENTIAL_COMPLEX

PROJECT_ROOT = Path(__file__).resolve().parents[1]
target = PROJECT_ROOT / "scripts" / "generate_complex_chains.py"
defaults = ["--out-chains", str(CHAIN_SPEC_PATHS[SEQUENTIAL_COMPLEX])]
sys.argv = [str(target), *defaults, *sys.argv[1:]]
runpy.run_path(str(target), run_name="__main__")
