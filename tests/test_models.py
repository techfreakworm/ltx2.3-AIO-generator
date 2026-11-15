"""Unit tests for models.py — MODEL_REGISTRY and ensure_models_for_mode."""

import pytest

import models
import modes
import workflow


def test_model_registry_resolves_known_files():
    assert (
        models.MODEL_REGISTRY["ltx-2.3-22b-distilled.safetensors"].repo_id == "Lightricks/LTX-2.3"
    )
    assert models.MODEL_REGISTRY["ltx-2.3-22b-distilled.safetensors"].subfolder == ""


def test_model_registry_includes_gemma_shards():
    for i in range(1, 6):
        key = f"model-{i:05d}-of-00005.safetensors"
        assert key in models.MODEL_REGISTRY
        assert "gemma-3-12b-it" in models.MODEL_REGISTRY[key].repo_id


def test_walk_workflow_for_models_finds_t2v_loaders():
    wf = workflow.load_template("t2v")
    needed = models.walk_workflow_for_models(wf)
    # T2V needs at minimum a transformer (distilled, dev fp8, or GGUF Q4) and a gemma encoder
    assert any(
        name.endswith(".gguf") or "distilled.safetensors" in name or "transformer_only" in name
        for name in needed
    )
    assert any("gemma" in name.lower() for name in needed)


def test_model_registry_has_no_retired_filenames():
    """A registry key that 404s on the Hub makes every generation abort mid-graph."""
    assert "ltx-2.3-spatial-upscaler-x2-1.0.safetensors" not in models.MODEL_REGISTRY
    assert "ltx-2.3-spatial-upscaler-x2-1.1.safetensors" in models.MODEL_REGISTRY
    entry = models.MODEL_REGISTRY["ltx-2.3-spatial-upscaler-x2-1.1.safetensors"]
    assert entry.repo_id == "Lightricks/LTX-2.3"
    assert entry.comfy_type == "latent_upscale_models"


def test_walk_workflow_for_models_requests_current_upscaler_for_every_mode():
    """The download pass must fetch the upscaler the workflow actually loads."""
    for mode in workflow.VALID_MODES:
        needed = models.walk_workflow_for_models(workflow.load_template(mode))
        assert "ltx-2.3-spatial-upscaler-x2-1.1.safetensors" in needed
        assert "ltx-2.3-spatial-upscaler-x2-1.0.safetensors" not in needed


def test_ensure_models_creates_symlinks_local(tmp_path, monkeypatch, fake_hf_cache):
    """In local mode, ensure_models creates symlinks from comfy/models -> HF cache."""
    monkeypatch.setenv("HF_HUB_CACHE", str(fake_hf_cache))
    monkeypatch.setattr(models, "_on_spaces", lambda: False)

    # Force the HF Hub call to fail so the fallback path (cache_dir.rglob) is exercised.
    def _raise(*_args, **_kwargs):
        raise RuntimeError("offline test: forcing fallback to cache scan")

    monkeypatch.setattr(models, "hf_hub_download", _raise)

    comfy_models = tmp_path / "comfyui" / "models"
    monkeypatch.setattr(models, "_comfy_models_dir", lambda: comfy_models)

    needed = {
        "ltx-2.3-22b-distilled.safetensors",
        "model-00001-of-00005.safetensors",
    }
    list(models.ensure_models(needed))

    # Staging is flat, comfyui/models/<comfy_type>/<filename>: a registry
    # subfolder is the file's path inside the HF repo, not part of the destination.
    staged = [
        comfy_models / "checkpoints" / "ltx-2.3-22b-distilled.safetensors",
        comfy_models / "text_encoders" / "model-00001-of-00005.safetensors",
    ]
    for link in staged:
        assert link.is_symlink(), f"{link} was not staged"
        assert link.resolve().is_relative_to(fake_hf_cache.resolve())


def test_ensure_models_skips_unregistered_files_with_warning(
    tmp_path, monkeypatch, fake_hf_cache, caplog
):
    """Files not in MODEL_REGISTRY are skipped (with warning), not raised."""
    import logging

    monkeypatch.setenv("HF_HUB_CACHE", str(fake_hf_cache))
    monkeypatch.setattr(models, "_on_spaces", lambda: False)
    monkeypatch.setattr(models, "_comfy_models_dir", lambda: tmp_path / "comfyui" / "models")

    with caplog.at_level(logging.WARNING):
        list(models.ensure_models({"nonexistent_phantom_file.safetensors"}))

    # Should not raise, should log a warning, should yield no events for the missing entry.
    assert any("nonexistent_phantom_file" in record.message for record in caplog.records)


_STATIC_CAMERA = "ltx-2-19b-lora-camera-control-static.safetensors"


def _camera_files(names):
    return {name for name in names if "camera-control" in name.lower()}


@pytest.mark.parametrize("mode", workflow.VALID_MODES)
def test_mode_model_sets_fetch_only_the_camera_lora_the_mode_loads(
    mode, monkeypatch, canonical_inputs
):
    """Lipsync needs the static LoRA and never dolly-in; no other mode needs a camera LoRA."""
    expected = {_STATIC_CAMERA} if mode == "lipsync" else set()

    # backend.submit's pre-flight walks the parameterised workflow.
    wf = workflow.load_template(mode)
    for patch in modes.MODE_REGISTRY[mode].parameterize_fn(canonical_inputs[mode]):
        workflow.set_input(wf, *patch)
    assert _camera_files(models.walk_workflow_for_models(wf)) == expected

    # ensure_models_for_mode walks the bare template, as tools/refresh_models.py does.
    requested: set[str] = set()

    def _record(filenames):
        requested.update(filenames)
        return iter(())

    monkeypatch.setattr(models, "ensure_models", _record)
    list(models.ensure_models_for_mode(mode))
    assert _camera_files(requested) == expected
    assert any("distilled-lora" in name for name in requested)


def test_static_camera_lora_is_a_registered_lora():
    entry = models.MODEL_REGISTRY[_STATIC_CAMERA]
    assert entry.comfy_type == "loras"
    assert entry.repo_id == "Lightricks/LTX-2-19b-LoRA-Camera-Control-Static"
