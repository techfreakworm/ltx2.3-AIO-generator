"""Pure functions over LTX 2.3 mode workflow JSON templates."""
from __future__ import annotations

import copy
import json
import pathlib
from typing import Any

WORKFLOWS_DIR = pathlib.Path(__file__).parent / "workflows"

VALID_MODES: tuple[str, ...] = ("t2v", "a2v", "i2v", "lipsync", "keyframe", "style")


def load_template(mode: str) -> dict[str, Any]:
    """Load a fresh, independent copy of the named mode's workflow template."""
    if mode not in VALID_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {VALID_MODES}")
    path = WORKFLOWS_DIR / f"{mode}.json"
    return copy.deepcopy(json.loads(path.read_text()))
