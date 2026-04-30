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
