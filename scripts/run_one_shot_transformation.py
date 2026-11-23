#!/usr/bin/env python3
"""PREEMPT One-Shot Edit — image transformation (global)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from utils.preempt import ONE_SHOT_TRANSFORMATION, RECORD_PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
target = PROJECT_ROOT / "scripts" / "run_level1_physical.py"
defaults = ["--out_record", str(RECORD_PATHS[ONE_SHOT_TRANSFORMATION])]
sys.argv = [str(target), *defaults, *sys.argv[1:]]
runpy.run_path(str(target), run_name="__main__")
