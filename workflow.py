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


def set_input(workflow: dict[str, Any], node_id: int, widget_index: int, value: Any) -> None:
    """Patch a node's widgets_values in place.

    Args:
        workflow: A workflow dict (must have a "nodes" list).
        node_id: The id of the node to patch.
        widget_index: Position within the node's widgets_values list.
        value: New value.

    Raises:
        KeyError: If no node with the given id exists.
    """
    for node in workflow["nodes"]:
        if node.get("id") == node_id:
            widgets = node.setdefault("widgets_values", [])
            while len(widgets) <= widget_index:
                widgets.append(None)
            widgets[widget_index] = value
            return
    raise KeyError(f"node id {node_id} not found in workflow")


def validate(workflow: dict[str, Any]) -> None:
    """Static schema validation. Raises ValueError on the first problem found."""
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list) or len(nodes) == 0:
        raise ValueError("workflow has no nodes")

    node_ids = {n.get("id") for n in nodes if "id" in n}
    for link in workflow.get("links", []):
        if not isinstance(link, list) or len(link) < 6:
            raise ValueError(f"malformed link {link}")
        _, src, _, dst, _, _ = link
        if src not in node_ids or dst not in node_ids:
            raise ValueError(f"orphan link {link}")
