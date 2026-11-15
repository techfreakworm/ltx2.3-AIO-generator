"""Unit tests for modes.py — MODE_REGISTRY and parameterize_fn correctness."""

import json

import pytest

import modes
import workflow


def _is_link(value):
    """API-format link input: `[<source node id str>, <output slot int>]`."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
    )


def _assert_patches_apply_to_template(mode_name, patches):
    """Every patch must hit an existing literal input of a node in the real template."""
    assert len(patches) > 0
    wf = workflow.load_template(mode_name)
    node_count = len(wf)

    for node_id, field, _value in patches:
        assert isinstance(node_id, str), f"node id {node_id!r} is not a str"
        assert isinstance(field, str), f"field {field!r} is not a str"
        assert node_id in wf, f"{mode_name}: patch targets missing node {node_id!r}"
        inputs = wf[node_id]["inputs"]
        class_type = wf[node_id]["class_type"]
        # set_input creates unknown fields silently, so a misspelt field would go unnoticed.
        assert field in inputs, f"{mode_name}: {class_type} {node_id!r} has no input {field!r}"
        assert not _is_link(inputs[field]), (
            f"{mode_name}: patch would cut the graph link at {node_id!r}.{field}"
        )

    for patch in patches:
        workflow.set_input(wf, *patch)

    assert len(wf) == node_count
    expected = {(node_id, field): value for node_id, field, value in patches}
    assert len(expected) == len(patches), f"{mode_name}: two patches write the same input"
    for (node_id, field), value in expected.items():
        assert wf[node_id]["inputs"][field] == value


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

    _assert_patches_apply_to_template("t2v", patches)


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


@pytest.mark.parametrize("mode_name", ["i2v", "a2v", "lipsync", "keyframe", "style"])
def test_remaining_modes_parameterize_validates(mode_name, canonical_inputs):
    inputs = canonical_inputs[mode_name]
    mode = modes.MODE_REGISTRY[mode_name]
    patches = mode.parameterize_fn(inputs)
    _assert_patches_apply_to_template(mode_name, patches)


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
        "t2v",
        "a2v",
        "i2v",
        "lipsync",
        "keyframe",
        "style",
    }


def test_each_mode_has_required_attributes():
    for name, mode in modes.MODE_REGISTRY.items():
        assert mode.name == name
        assert mode.label  # non-empty
        assert mode.icon  # non-empty
        assert callable(mode.parameterize_fn)
        assert isinstance(mode.stage_map, list) and len(mode.stage_map) > 0


_POWER_LORA = "Power Lora Loader (rgthree)"
_HOLD_CAMERA_ROW = {
    "on": True,
    "lora": "ltx-2-19b-lora-camera-control-static.safetensors",
    "strength": 0.5,
}


def _parameterised(mode_name, inputs):
    """The workflow app._on_generate hands to backend.submit."""
    wf = workflow.load_template(mode_name)
    for patch in modes.MODE_REGISTRY[mode_name].parameterize_fn(inputs):
        workflow.set_input(wf, *patch)
    return wf


def _enabled_lora_rows(wf):
    return sorted(
        (
            row
            for node in wf.values()
            if node["class_type"] == _POWER_LORA
            for row in node["inputs"].values()
            if isinstance(row, dict) and "lora" in row and row.get("on")
        ),
        key=json.dumps,
    )


def test_lipsync_submits_the_distilled_row_and_the_static_camera_lora_only(canonical_inputs):
    raw = json.loads((workflow.WORKFLOWS_DIR / "lipsync.json").read_text())
    exported = [row for row in _enabled_lora_rows(raw) if "camera-control" not in row["lora"]]
    assert [row["lora"] for row in exported] == [
        "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors"
    ]

    enabled = _enabled_lora_rows(_parameterised("lipsync", canonical_inputs["lipsync"]))

    assert enabled == sorted(exported + [_HOLD_CAMERA_ROW], key=json.dumps)


@pytest.mark.parametrize("mode_name", ["t2v", "i2v", "a2v", "keyframe", "style"])
def test_other_modes_submit_no_camera_lora(mode_name, canonical_inputs):
    enabled = _enabled_lora_rows(_parameterised(mode_name, canonical_inputs[mode_name]))
    assert enabled, f"{mode_name}: no enabled LoRA rows at all"
    assert [row["lora"] for row in enabled if "camera-control" in row["lora"]] == []
