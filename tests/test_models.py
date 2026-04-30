"""Unit tests for models.py — MODEL_REGISTRY and ensure_models_for_mode."""

import models
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

    # Each requested file should now have a symlink in comfyui/models/<type>/
    assert (comfy_models / "checkpoints" / "ltx-2.3-22b-distilled.safetensors").is_symlink()
    assert (
        comfy_models / "text_encoders" / "gemma-3-12b-it" / "model-00001-of-00005.safetensors"
    ).is_symlink()


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
