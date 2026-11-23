#!/usr/bin/env python3
"""PREEMPT One-Shot Edit — semantic (local)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from utils.preempt import ONE_SHOT_SEMANTIC, RECORD_PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
target = PROJECT_ROOT / "scripts" / "run_level2_single_edit.py"
defaults = ["--record", str(RECORD_PATHS[ONE_SHOT_SEMANTIC])]
sys.argv = [str(target), *defaults, *sys.argv[1:]]
runpy.run_path(str(target), run_name="__main__")
