"""Shared pytest fixtures and CLI flags."""
import json
import os
import pathlib
from typing import Any

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

DEFAULT_MASTER_WORKFLOW = pathlib.Path(
    os.environ.get(
        "LTX23_MASTER_WORKFLOW",
        pathlib.Path.home() / "Projects/comfyui/user/default/workflows"
        / "1. LTX 2.3 All-In-One 260406-05.json",
    )
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--gpu", action="store_true", help="Run L4 GPU smoke tests.")
    parser.addoption(
        "--comfy-real",
        action="store_true",
        help="Use bundled ComfyUI for L2 graph validation (slower).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not config.getoption("--gpu"):
        skip_gpu = pytest.mark.skip(reason="GPU smoke tests skipped (use --gpu)")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def master_workflow() -> dict[str, Any]:
    """The full LTX 2.3 All-In-One workflow JSON (loaded from user's ComfyUI)."""
    if not DEFAULT_MASTER_WORKFLOW.exists():
        pytest.skip(
            f"Master workflow not found at {DEFAULT_MASTER_WORKFLOW}. "
            "Set LTX23_MASTER_WORKFLOW env var to its path."
        )
    return json.loads(DEFAULT_MASTER_WORKFLOW.read_text())


@pytest.fixture
def canonical_inputs() -> dict[str, dict[str, Any]]:
    """Known-good Gradio input dicts per mode (used by L1/L2 tests)."""
    return {
        "t2v": {
            "prompt": "a tiger walking through a misty forest at dawn, cinematic",
            "negative_prompt": "",
            "preset": "balanced",
            "width": 512,
            "height": 768,
            "frames": 81,
            "fps": 24,
            "seed": 42,
            "camera_lora": "none",
            "camera_strength": 0.8,
            "detailer_on": False,
            "detailer_strength": 0.5,
        },
        "i2v": {
            "prompt": "the subject turns toward the camera and smiles",
            "image": "/tmp/portrait.png",
            "preset": "balanced",
            "width": 512,
            "height": 768,
            "frames": 81,
            "fps": 24,
            "seed": 42,
            "camera_lora": "none",
            "camera_strength": 0.8,
            "detailer_on": True,
            "detailer_strength": 0.5,
            "ic_lora": "union",
            "ic_strength": 0.5,
            "pose_on": False,
        },
        "a2v": {
            "prompt": "a dancer moves to the beat in a neon-lit studio",
            "audio": "/tmp/track.wav",
            "preset": "balanced",
            "width": 512,
            "height": 768,
            "frames": 81,
            "fps": 24,
            "seed": 42,
            "audio_cfg": 7.0,
        },
        "lipsync": {
            "prompt": "the person speaks the audio with natural mouth movement",
            "image": "/tmp/portrait.png",
            "audio": "/tmp/speech.wav",
            "preset": "balanced",
            "image_strength": 0.7,
            "frames": 81,
            "fps": 24,
            "seed": 42,
        },
        "keyframe": {
            "prompt": "smooth transition between the two frames",
            "first_frame": "/tmp/start.png",
            "last_frame": "/tmp/end.png",
            "preset": "balanced",
            "frames": 81,
            "fps": 24,
            "seed": 42,
        },
        "style": {
            "prompt": "in the style of a renaissance oil painting",
            "input_video": "/tmp/source.mp4",
            "preset": "balanced",
            "frames": 81,
            "fps": 24,
            "seed": 42,
            "ic_lora": "motion-track",
            "ic_strength": 0.5,
        },
    }


@pytest.fixture
def fake_hf_cache(tmp_path: pathlib.Path) -> pathlib.Path:
    """A fake ~/.cache/huggingface/hub layout with placeholder files."""
    hub = tmp_path / "huggingface" / "hub"
    layouts = {
        "models--Lightricks--LTX-2.3": [
            "ltx-2.3-22b-distilled.safetensors",
            "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
            "ltx-2.3-22b-distilled-lora-384.safetensors",
        ],
        "models--google--gemma-3-12b-it-qat-q4_0-unquantized": [
            "model-00001-of-00005.safetensors",
            "model-00002-of-00005.safetensors",
            "model-00003-of-00005.safetensors",
            "model-00004-of-00005.safetensors",
            "model-00005-of-00005.safetensors",
            "model.safetensors.index.json",
            "tokenizer.model",
            "preprocessor_config.json",
        ],
        "models--Kijai--LTX2.3_comfy": [
            "LTX23_video_vae_bf16.safetensors",
            "LTX23_audio_vae_bf16.safetensors",
        ],
    }
    for repo, files in layouts.items():
        snapshot_dir = hub / repo / "snapshots" / "deadbeef"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for filename in files:
            (snapshot_dir / filename).write_text("")  # placeholder
    return hub
