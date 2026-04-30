"""Unit tests for workflow.py — pure functions over JSON dicts."""
import pytest

import workflow


def test_load_template_returns_dict_for_valid_mode():
    wf = workflow.load_template("t2v")
    assert isinstance(wf, dict)
    assert "nodes" in wf
    assert len(wf["nodes"]) > 0


def test_load_template_raises_for_unknown_mode():
    with pytest.raises(ValueError, match="unknown mode"):
        workflow.load_template("nonexistent")


def test_load_template_returns_independent_copy():
    """Mutations to one returned dict must not affect later loads."""
    a = workflow.load_template("t2v")
    a["nodes"].append({"id": -999})
    b = workflow.load_template("t2v")
    assert {-999} & {n.get("id") for n in b["nodes"]} == set()


def test_set_input_patches_widgets_values_in_place():
    wf = workflow.load_template("t2v")
    target_node = next(n for n in wf["nodes"] if n["type"] == "CLIPTextEncode")
    workflow.set_input(wf, target_node["id"], 0, "new prompt text")
    refetched = next(n for n in wf["nodes"] if n["id"] == target_node["id"])
    assert refetched["widgets_values"][0] == "new prompt text"


def test_set_input_raises_for_unknown_node():
    wf = workflow.load_template("t2v")
    with pytest.raises(KeyError, match="node id"):
        workflow.set_input(wf, 999_999_999, 0, "x")


def test_validate_accepts_canonical_template():
    wf = workflow.load_template("t2v")
    workflow.validate(wf)  # must not raise


def test_validate_rejects_workflow_with_no_nodes():
    wf = {"nodes": [], "links": []}
    with pytest.raises(ValueError, match="no nodes"):
        workflow.validate(wf)


def test_validate_rejects_orphan_link():
    wf = workflow.load_template("t2v")
    wf["links"].append([99999, 1, 0, 999_999_999, 0, "INT"])  # destination doesn't exist
    with pytest.raises(ValueError, match="orphan link"):
        workflow.validate(wf)
