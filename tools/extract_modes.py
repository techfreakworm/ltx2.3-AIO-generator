"""Extract six mode-specific workflow templates from the master LTX 2.3 All-In-One workflow.

Each ComfyUI group whose title starts with a number (e.g. "01 Text to Video") becomes
a mode template containing only that group's nodes plus shared scaffolding (Models,
Lora, Setting, Prompt, Load Audio/Image/Video, Output groups).

Group title -> output filename mapping:
    01 -> t2v.json
    02 -> a2v.json
    03 -> i2v.json
    04 -> lipsync.json
    05 -> keyframe.json
    06 -> style.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Iterable

GROUP_TO_FILENAME: dict[str, str] = {
    "01": "t2v.json",
    "02": "a2v.json",
    "03": "i2v.json",
    "04": "lipsync.json",
    "05": "keyframe.json",
    "06": "style.json",
}

SHARED_GROUP_PREFIXES: tuple[str, ...] = (
    "Models",
    "Lora",
    "Setting",
    "Prompt",
    "Load Audio",
    "Load Image",
    "Load Images",
    "Load Video",
    "Output",
)


def _node_in_group(node: dict, group: dict) -> bool:
    """Test whether a node's position lies inside a group's bounding box."""
    if "pos" not in node or "bounding" not in group:
        return False
    nx, ny = node["pos"][0], node["pos"][1]
    gx, gy, gw, gh = group["bounding"]
    return (gx <= nx <= gx + gw) and (gy <= ny <= gy + gh)


def _select_groups(master: dict, mode_prefix: str) -> list[dict]:
    """Pick the mode group plus all shared groups."""
    selected: list[dict] = []
    for g in master.get("groups", []):
        title = (g.get("title") or "").strip()
        if title.startswith(mode_prefix + " "):
            selected.append(g)
        elif any(title.startswith(p) for p in SHARED_GROUP_PREFIXES):
            selected.append(g)
    return selected


def _collect_nodes(master: dict, groups: Iterable[dict]) -> list[dict]:
    """Return all nodes lying inside any of the given groups."""
    groups_list = list(groups)
    keep: list[dict] = []
    for node in master.get("nodes", []):
        if any(_node_in_group(node, g) for g in groups_list):
            keep.append(node)
    return keep


def _collect_links(master: dict, kept_node_ids: set[int]) -> list[list]:
    """Keep only links where both endpoints are in the surviving node set."""
    return [
        link
        for link in master.get("links", [])
        # ComfyUI link tuple format: [link_id, src_node_id, src_out, dst_node_id, dst_in, type]
        if link[1] in kept_node_ids and link[3] in kept_node_ids
    ]


def extract_mode(master: dict, mode_prefix: str) -> dict:
    """Build a focused workflow JSON for the given mode group prefix."""
    groups = _select_groups(master, mode_prefix)
    nodes = _collect_nodes(master, groups)
    kept_ids = {n["id"] for n in nodes}
    links = _collect_links(master, kept_ids)

    return {
        "id": f"ltx23-aio-{mode_prefix}",
        "revision": 0,
        "last_node_id": max(kept_ids, default=0),
        "last_link_id": max((l[0] for l in links), default=0),
        "nodes": nodes,
        "links": links,
        "groups": groups,
        "definitions": master.get("definitions", {}),
        "config": master.get("config", {}),
        "extra": master.get("extra", {}),
        "version": master.get("version", 0.4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    master = json.loads(args.master.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    for prefix, filename in GROUP_TO_FILENAME.items():
        wf = extract_mode(master, prefix)
        out_path = args.out / filename
        out_path.write_text(json.dumps(wf, indent=2))
        print(f"  -> wrote {out_path} ({len(wf['nodes'])} nodes, {len(wf['links'])} links)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
