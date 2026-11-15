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


_POWER_LORA = "Power Lora Loader (rgthree)"


def _lora_rows(wf):
    """`(node_id, input key, row)` for every Power Lora Loader row in *wf*."""
    return [
        (node_id, key, row)
        for node_id, node in wf.items()
        if node.get("class_type") == _POWER_LORA
        for key, row in node["inputs"].items()
        if isinstance(row, dict) and "on" in row and "lora" in row
    ]


def _is_camera_lora(row):
    return "camera-control" in str(row["lora"]).lower()


_STATIC_CAMERA = "ltx-2-19b-lora-camera-control-static.safetensors"
_HOLD_CAMERA_ROW = {"on": True, "lora": _STATIC_CAMERA, "strength": 0.5}
_DISTILLED_ROW = {
    "on": True,
    "lora": "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors",
    "strength": 0.6,
}


@pytest.mark.parametrize("mode", workflow.VALID_MODES)
def test_load_template_enables_no_camera_move_lora(mode):
    """No dolly/jib row is ever on; lipsync alone holds the camera with the static LoRA."""
    rows = _lora_rows(workflow.load_template(mode))
    assert rows, f"{mode}: no Power Lora Loader rows found"
    enabled = [row for _nid, _key, row in rows if _is_camera_lora(row) and row["on"]]
    assert enabled == ([_HOLD_CAMERA_ROW] if mode == "lipsync" else [])


@pytest.mark.parametrize("mode", workflow.VALID_MODES)
def test_load_template_keeps_non_camera_lora_rows_as_exported(mode):
    """Only a camera row's `on` may differ from the export; nothing else is switched off."""
    raw = json.loads((workflow.WORKFLOWS_DIR / f"{mode}.json").read_text())
    raw_rows = {(nid, key): row for nid, key, row in _lora_rows(raw)}
    loaded_rows = {(nid, key): row for nid, key, row in _lora_rows(workflow.load_template(mode))}
    assert raw_rows.keys() <= loaded_rows.keys()
    added = [loaded_rows[where] for where in loaded_rows.keys() - raw_rows.keys()]
    assert added == ([_HOLD_CAMERA_ROW] if mode == "lipsync" else [])

    for where, raw_row in raw_rows.items():
        expected = dict(raw_row)
        expected["lora"] = workflow.RETIRED_MODEL_RENAMES.get(raw_row["lora"], raw_row["lora"])
        actual = dict(loaded_rows[where])
        if _is_camera_lora(raw_row):
            del expected["on"], actual["on"]
        assert actual == expected, f"{mode}: row {where} changed"

    assert any(row["on"] for row in loaded_rows.values() if not _is_camera_lora(row))


def test_load_template_switches_off_only_enabled_power_lora_camera_rows(tmp_path, monkeypatch):
    """Matched by node type, row shape and filename, so a re-export can renumber freely."""
    camera_on = {
        "on": True,
        "lora": "LTX-2-19B-LoRA-Camera-Control-Dolly-In.safetensors",
        "strength": 0.8,
    }
    template = {
        "7": {
            "class_type": _POWER_LORA,
            "inputs": {
                "PowerLoraLoaderHeaderWidget": {"type": "PowerLoraLoaderHeaderWidget"},
                "lora_9": camera_on,
                "lora_2": {
                    "on": False,
                    "lora": "ltx-2-19b-lora-camera-control-jib-up.safetensors",
                    "strength": 1,
                },
                "lora_4": {
                    "on": True,
                    "lora": "ltx-2.3-22b-distilled-lora-384.safetensors",
                    "strength": 0.6,
                },
                "➕ Add Lora": "",
                "model": ["8", 0],
            },
        },
        "8": {
            "class_type": "Some Other Lora Stack",
            "inputs": {
                "lora_1": {
                    "on": True,
                    "lora": "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
                    "strength": 1,
                },
                "lora_name": "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
            },
        },
    }
    (tmp_path / "t2v.json").write_text(json.dumps(template))
    monkeypatch.setattr(workflow, "WORKFLOWS_DIR", tmp_path)

    loaded = workflow.load_template("t2v")

    expected = json.loads(json.dumps(template))
    expected["7"]["inputs"]["lora_9"]["on"] = False
    assert loaded == expected


def _write_lipsync_template(tmp_path, monkeypatch, template):
    (tmp_path / "lipsync.json").write_text(json.dumps(template))
    monkeypatch.setattr(workflow, "WORKFLOWS_DIR", tmp_path)


def test_lipsync_hold_camera_row_never_overwrites_an_exported_row(tmp_path, monkeypatch):
    """A re-export can renumber rows; the distilled row must survive whatever key it lands on."""
    exported = {
        "PowerLoraLoaderHeaderWidget": {"type": "PowerLoraLoaderHeaderWidget"},
        "lora_1": {
            "on": True,
            "lora": "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
            "strength": 1,
        },
        "lora_3": {"on": True, "lora": _STATIC_CAMERA, "strength": 1},
        "lora_5": _DISTILLED_ROW,
        "➕ Add Lora": "",
        "model": ["43", 0],
    }
    template = {
        "42": {"class_type": _POWER_LORA, "inputs": exported},
        "43": {"class_type": "UNETLoader", "inputs": {"unet_name": "x.safetensors"}},
    }
    _write_lipsync_template(tmp_path, monkeypatch, template)

    loaded = workflow.load_template("lipsync")

    inputs = loaded["42"]["inputs"]
    assert inputs["lora_5"] == _DISTILLED_ROW
    assert inputs["lora_1"] == dict(exported["lora_1"], on=False)
    assert inputs["lora_3"] == dict(exported["lora_3"], on=False)
    added = set(inputs) - set(exported)
    assert len(added) == 1 and inputs[added.pop()] == _HOLD_CAMERA_ROW
    assert loaded["43"] == template["43"]
    enabled = sorted((row for _n, _k, row in _lora_rows(loaded) if row["on"]), key=json.dumps)
    assert enabled == sorted([_DISTILLED_ROW, _HOLD_CAMERA_ROW], key=json.dumps)

    for _n, _k, row in _lora_rows(loaded):
        row["strength"] = 9
    reloaded = workflow.load_template("lipsync")
    assert [row for _n, _k, row in _lora_rows(reloaded) if row["on"]].count(_HOLD_CAMERA_ROW) == 1


@pytest.mark.parametrize("loader_count", [0, 2])
def test_lipsync_refuses_to_guess_which_lora_loader_holds_the_camera(
    tmp_path, monkeypatch, loader_count
):
    template = {
        str(i): {"class_type": _POWER_LORA, "inputs": {"lora_1": dict(_DISTILLED_ROW)}}
        for i in range(loader_count)
    }
    template["99"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "x.safetensors"}}
    _write_lipsync_template(tmp_path, monkeypatch, template)

    with pytest.raises(ValueError, match="Power Lora Loader"):
        workflow.load_template("lipsync")
