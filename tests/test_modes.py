"""Unit tests for modes.py — MODE_REGISTRY and parameterize_fn correctness."""
import pytest

import modes
import workflow


def test_mode_dataclass_has_expected_fields():
    """Mode dataclass exposes the expected attribute set."""
    fields = {"name", "label", "icon", "parameterize_fn", "stage_map"}
    actual = set(modes.Mode.__dataclass_fields__.keys())
    assert fields == actual


def test_mode_registry_is_a_dict():
    """MODE_REGISTRY exists and is a dict (entries added in Tasks 11–12)."""
    assert isinstance(modes.MODE_REGISTRY, dict)


def test_t2v_parameterize_produces_valid_patches(canonical_inputs):
    inputs = canonical_inputs["t2v"]
    mode = modes.MODE_REGISTRY["t2v"]
    patches = mode.parameterize_fn(inputs)

    # All patches must be (node_id: int, widget_index: int, value: Any)
    for node_id, widget_index, value in patches:
        assert isinstance(node_id, int)
        assert isinstance(widget_index, int)

    # Apply patches to a real template; result must validate.
    wf = workflow.load_template("t2v")
    for patch in patches:
        workflow.set_input(wf, *patch)
    workflow.validate(wf)


def test_i2v_parameterize_uses_image_path(canonical_inputs):
    inputs = canonical_inputs["i2v"]
    mode = modes.MODE_REGISTRY["i2v"]
    patches = mode.parameterize_fn(inputs)
    values = [p[2] for p in patches]
    assert inputs["image"] in values


def test_t2v_and_i2v_in_registry():
    """T2V and I2V exist in MODE_REGISTRY (full completeness in Task 12)."""
    assert "t2v" in modes.MODE_REGISTRY
    assert "i2v" in modes.MODE_REGISTRY
