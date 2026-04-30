"""MODE_REGISTRY — one Mode entry per generation mode.

Each Mode declares:
- name: short id ("t2v", "i2v", ...)
- label: display name
- icon: single-character or emoji icon for the sidebar
- stage_map: list of (label, expected_share_pct) for the status banner
- parameterize_fn: (Gradio inputs dict) -> list[(node_id, widget_index, value)]

The parameterize_fn is the only mode-specific logic. Everything else (workflow
loading, validation, dispatch) is mode-agnostic and lives in workflow.py /
backend.py.

Tasks 11 (T2V + I2V) and 12 (A2V + Lipsync + Keyframe + Style) populate
MODE_REGISTRY. This task only sets up the dataclass and the empty container.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

Patch = tuple[int, int, Any]
ParameterizeFn = Callable[[dict[str, Any]], list[Patch]]


@dataclass(frozen=True)
class Stage:
    label: str
    share_pct: int  # rough share of total time, sums to ~100 across stages


@dataclass(frozen=True)
class Mode:
    name: str
    label: str
    icon: str
    parameterize_fn: ParameterizeFn
    stage_map: list[Stage] = field(default_factory=list)


# Filled in by tasks 11–12.
MODE_REGISTRY: dict[str, Mode] = {}
