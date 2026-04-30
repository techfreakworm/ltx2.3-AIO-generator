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


@pytest.mark.parametrize("mode_name", ["a2v", "lipsync", "keyframe", "style"])
def test_remaining_modes_parameterize_validates(mode_name, canonical_inputs):
    inputs = canonical_inputs[mode_name]
    mode = modes.MODE_REGISTRY[mode_name]
    patches = mode.parameterize_fn(inputs)
    assert len(patches) > 0

    wf = workflow.load_template(mode_name)
    for patch in patches:
        workflow.set_input(wf, *patch)
    workflow.validate(wf)


def test_a2v_parameterize_passes_audio_path(canonical_inputs):
    patches = modes.MODE_REGISTRY["a2v"].parameterize_fn(canonical_inputs["a2v"])
    assert canonical_inputs["a2v"]["audio"] in [p[2] for p in patches]


def test_lipsync_parameterize_passes_image_and_audio(canonical_inputs):
    patches = modes.MODE_REGISTRY["lipsync"].parameterize_fn(canonical_inputs["lipsync"])
    values = [p[2] for p in patches]
    assert canonical_inputs["lipsync"]["image"] in values
    assert canonical_inputs["lipsync"]["audio"] in values


def test_keyframe_parameterize_passes_two_frames(canonical_inputs):
    patches = modes.MODE_REGISTRY["keyframe"].parameterize_fn(canonical_inputs["keyframe"])
    values = [p[2] for p in patches]
    assert canonical_inputs["keyframe"]["first_frame"] in values
    assert canonical_inputs["keyframe"]["last_frame"] in values


def test_style_parameterize_passes_input_video(canonical_inputs):
    patches = modes.MODE_REGISTRY["style"].parameterize_fn(canonical_inputs["style"])
    assert canonical_inputs["style"]["input_video"] in [p[2] for p in patches]


def test_mode_registry_has_all_six_keys():
    """All six modes are in the registry now."""
    assert set(modes.MODE_REGISTRY.keys()) == {
        "t2v", "a2v", "i2v", "lipsync", "keyframe", "style",
    }


def test_each_mode_has_required_attributes():
    for name, mode in modes.MODE_REGISTRY.items():
        assert mode.name == name
        assert mode.label  # non-empty
        assert mode.icon  # non-empty
        assert callable(mode.parameterize_fn)
        assert isinstance(mode.stage_map, list) and len(mode.stage_map) > 0
