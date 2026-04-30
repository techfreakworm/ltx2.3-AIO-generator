"""Unit tests for models.py — MODEL_REGISTRY and ensure_models_for_mode."""
import models
import workflow


def test_model_registry_resolves_known_files():
    assert models.MODEL_REGISTRY["ltx-2.3-22b-distilled.safetensors"].repo_id == "Lightricks/LTX-2.3"
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
        name.endswith(".gguf")
        or "distilled.safetensors" in name
        or "transformer_only" in name
        for name in needed
    )
    assert any("gemma" in name.lower() for name in needed)
