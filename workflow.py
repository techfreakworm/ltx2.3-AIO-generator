"""Pure functions over LTX 2.3 mode API-format workflow templates.

Templates in `workflows/<mode>.json` are saved from ComfyUI's editor via
"Save (API Format)". Shape: `{node_id_str: {"class_type": str, "inputs": dict}}`.
This is what ComfyUI's `PromptExecutor.execute(prompt=...)` expects directly.
"""

from __future__ import annotations

import copy
import json
import pathlib
from typing import Any

WORKFLOWS_DIR = pathlib.Path(__file__).parent / "workflows"

VALID_MODES: tuple[str, ...] = ("t2v", "a2v", "i2v", "lipsync", "keyframe", "style")


def load_template(mode: str) -> dict[str, Any]:
    """Load a fresh, independent copy of the named mode's API workflow template."""
    if mode not in VALID_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {VALID_MODES}")
    path = WORKFLOWS_DIR / f"{mode}.json"
    return copy.deepcopy(json.loads(path.read_text()))


def set_input(workflow: dict[str, Any], node_id: int | str, field: str, value: Any) -> None:
    """Patch a node's input field in place.

    For API-format workflows, each node has an `inputs` dict keyed by field name.
    `node_id` is the dict key (string for top-level, "<inst>:<inner>" for
    subgraph-internal). `field` is an entry name in `inputs`.

    Args:
        workflow: API-format workflow dict (mapping id → {class_type, inputs}).
        node_id: Dict key of the target node.
        field: Name of the input field to set.
        value: New value (literal, or `[src_id, src_slot]` link form).

    Raises:
        KeyError: If the node doesn't exist in the workflow.
    """
    nid = str(node_id)
    if nid not in workflow:
        raise KeyError(f"node id {nid!r} not found in workflow")
    inputs = workflow[nid].setdefault("inputs", {})
    inputs[field] = value
