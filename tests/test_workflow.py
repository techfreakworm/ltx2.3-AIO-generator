"""Unit tests for workflow.py — pure functions over API-format workflow dicts."""

import json

import pytest

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


def _link_inputs(wf):
    """Every `(node_id, field, source_node_id)` wired by a link input."""
    return [
        (node_id, field, value[0])
        for node_id, node in wf.items()
        if isinstance(node, dict) and isinstance(node.get("inputs"), dict)
        for field, value in node["inputs"].items()
        if _is_link(value)
    ]


def _structural_problems(wf):
    """What `workflow.validate` used to catch, restated for the API format."""
    if not wf:
        return ["workflow has no nodes"]
    problems = []
    for node_id, node in wf.items():
        if not isinstance(node, dict):
            problems.append(f"node {node_id!r} is not a dict")
            continue
        if not isinstance(node.get("class_type"), str) or not node["class_type"]:
            problems.append(f"node {node_id!r} has no class_type")
        if not isinstance(node.get("inputs"), dict):
            problems.append(f"node {node_id!r} has no inputs dict")
    for node_id, field, source in _link_inputs(wf):
        if source not in wf:
            problems.append(f"orphan link: {node_id!r}.{field} -> missing node {source!r}")
    outputs = [
        node_id
        for node_id, node in wf.items()
        if isinstance(node, dict) and node.get("class_type") == "VHS_VideoCombine"
    ]
    if len(outputs) != 1:
        problems.append(f"expected exactly one VHS_VideoCombine output, found {outputs}")
    return problems


@pytest.mark.parametrize("mode", workflow.VALID_MODES)
def test_load_template_returns_dict_for_valid_mode(mode):
    wf = workflow.load_template(mode)
    assert isinstance(wf, dict)
    assert len(wf) > 0
    # Editor format has top-level "nodes"/"links"; PromptExecutor needs API format.
    assert "nodes" not in wf and "links" not in wf
    for node_id, node in wf.items():
        assert isinstance(node_id, str)
        assert isinstance(node, dict) and "class_type" in node and "inputs" in node


def test_load_template_raises_for_unknown_mode():
    with pytest.raises(ValueError, match="unknown mode"):
        workflow.load_template("nonexistent")


def test_load_template_returns_independent_copy():
    """Mutations to one returned dict, nested ones included, must not affect later loads."""
    a = workflow.load_template("t2v")
    node_id = next(iter(a))
    a[node_id]["inputs"]["__sentinel__"] = "mutated"
    a["-999"] = {"class_type": "Phantom", "inputs": {}}

    b = workflow.load_template("t2v")
    assert "-999" not in b
    assert "__sentinel__" not in b[node_id]["inputs"]


def test_set_input_patches_inputs_field_in_place():
    wf = workflow.load_template("t2v")
    node_id = next(nid for nid, node in wf.items() if node["class_type"] == "CLIPTextEncode")
    inputs_before = wf[node_id]["inputs"]
    clip_link = inputs_before["clip"]

    workflow.set_input(wf, node_id, "text", "new prompt text")

    assert wf[node_id]["inputs"] is inputs_before
    assert wf[node_id]["inputs"]["text"] == "new prompt text"
    assert wf[node_id]["inputs"]["clip"] == clip_link


def test_set_input_raises_for_unknown_node():
    wf = workflow.load_template("t2v")
    with pytest.raises(KeyError, match="node id"):
        workflow.set_input(wf, "999999999", "text", "x")
    assert "999999999" not in wf


@pytest.mark.parametrize("mode", workflow.VALID_MODES)
def test_template_passes_structural_checks(mode):
    wf = workflow.load_template(mode)
    assert _structural_problems(wf) == []
    # Guards against _is_link silently matching nothing, which would skip the orphan check.
    assert len(_link_inputs(wf)) > 0


def test_structural_checks_reject_workflow_with_no_nodes():
    assert "workflow has no nodes" in _structural_problems({})


def test_structural_checks_reject_orphan_link():
    wf = workflow.load_template("t2v")
    node_id, field, _source = _link_inputs(wf)[0]
    wf[node_id]["inputs"][field] = ["999999999", 0]
    assert any(p.startswith("orphan link") for p in _structural_problems(wf))


def test_load_template_remaps_retired_upscaler_filename():
    """Lightricks deleted the x2 spatial upscaler v1.0; templates still name it."""
    for mode in workflow.VALID_MODES:
        wf = workflow.load_template(mode)
        assert wf["101"]["inputs"]["model_name"] == "ltx-2.3-spatial-upscaler-x2-1.1.safetensors", (
            f"{mode} still loads a retired upscaler"
        )


def test_load_template_leaves_no_retired_filenames_in_any_mode():
    """No retired model filename may reach the workflow handed to PromptExecutor."""
    for mode in workflow.VALID_MODES:
        serialized = json.dumps(workflow.load_template(mode))
        for retired in ("ltx-2.3-spatial-upscaler-x2-1.0.safetensors",):
            assert retired not in serialized, f"{mode} references retired {retired}"


def test_set_input_patches_one_named_field_of_vhs_audio_node():
    """VHS_* loaders have several named inputs; patching one must leave the rest alone."""
    wf = workflow.load_template("a2v")
    node_id = next(nid for nid, node in wf.items() if node["class_type"] == "VHS_LoadAudioUpload")
    before = dict(wf[node_id]["inputs"])
    assert "audio" in before and len(before) > 1

    workflow.set_input(wf, node_id, "audio", "/tmp/new_value.wav")

    after = wf[node_id]["inputs"]
    assert after["audio"] == "/tmp/new_value.wav"
    assert {k: v for k, v in after.items() if k != "audio"} == {
        k: v for k, v in before.items() if k != "audio"
    }
