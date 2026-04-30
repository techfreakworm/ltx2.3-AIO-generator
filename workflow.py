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


def set_input(workflow: dict[str, Any], node_id: int, widget_index: int | str, value: Any) -> None:
    """Patch a node's widgets_values in place.

    Supports both list-style widgets_values (most ComfyUI nodes — patch by integer index,
    auto-extending with None) and dict-style widgets_values (VHS_LoadAudioUpload and
    similar — patch by string key, raising KeyError if the key doesn't exist).

    Args:
        workflow: A workflow dict (must have a "nodes" list).
        node_id: The id of the node to patch.
        widget_index: Integer index (for list widgets) or string key (for dict widgets).
        value: New value.

    Raises:
        KeyError: If no node with the given id exists, or for dict widgets, if the key
            doesn't already exist on the target dict (we don't add new keys).
        TypeError: If widget_index type doesn't match the node's widgets_values type.
    """
    for node in workflow["nodes"]:
        if node.get("id") != node_id:
            continue
        widgets = node.get("widgets_values")
        if isinstance(widgets, dict):
            if not isinstance(widget_index, str):
                raise TypeError(
                    f"node {node_id} has dict widgets_values; widget_index must be str, "
                    f"got {type(widget_index).__name__}"
                )
            if widget_index not in widgets:
                raise KeyError(
                    f"node {node_id} dict widgets_values has no key {widget_index!r}; "
                    f"available keys: {list(widgets.keys())}"
                )
            widgets[widget_index] = value
            return
        # List/None case — preserve existing list-extension behavior.
        if not isinstance(widget_index, int):
            raise TypeError(
                f"node {node_id} has list widgets_values; widget_index must be int, "
                f"got {type(widget_index).__name__}"
            )
        if widgets is None:
            widgets = []
            node["widgets_values"] = widgets
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
