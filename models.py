"""Model file registry: maps filename -> (HuggingFace repo, subfolder).

Lookups are by filename only — the same filename in two different repos is not
supported. If that ever happens we'll qualify by ComfyUI loader-type.
"""

from __future__ import annotations

import logging
import os
import pathlib
from collections.abc import Iterator
from dataclasses import dataclass

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelEntry:
    repo_id: str
    subfolder: str = ""
    comfy_type: str = "checkpoints"  # ComfyUI models/<comfy_type>/ subdirectory


MODEL_REGISTRY: dict[str, ModelEntry] = {
    # Main LTX 2.3 transformer + LoRAs + upscalers
    "ltx-2.3-22b-distilled.safetensors": ModelEntry("Lightricks/LTX-2.3", comfy_type="checkpoints"),
    "ltx-2.3-22b-dev.safetensors": ModelEntry("Lightricks/LTX-2.3", comfy_type="checkpoints"),
    "ltx-2.3-spatial-upscaler-x2-1.0.safetensors": ModelEntry(
        "Lightricks/LTX-2.3", comfy_type="upscale_models"
    ),
    "ltx-2.3-22b-distilled-lora-384.safetensors": ModelEntry(
        "Lightricks/LTX-2.3", comfy_type="loras"
    ),
    # Gemma 3 12B (5 shards + tokenizer/preprocessor)
    **{
        f"model-{i:05d}-of-00005.safetensors": ModelEntry(
            "google/gemma-3-12b-it-qat-q4_0-unquantized",
            comfy_type="text_encoders",
            subfolder="gemma-3-12b-it",
        )
        for i in range(1, 6)
    },
    "model.safetensors.index.json": ModelEntry(
        "google/gemma-3-12b-it-qat-q4_0-unquantized",
        comfy_type="text_encoders",
        subfolder="gemma-3-12b-it",
    ),
    "tokenizer.model": ModelEntry(
        "google/gemma-3-12b-it-qat-q4_0-unquantized",
        comfy_type="text_encoders",
        subfolder="gemma-3-12b-it",
    ),
    "preprocessor_config.json": ModelEntry(
        "google/gemma-3-12b-it-qat-q4_0-unquantized",
        comfy_type="text_encoders",
        subfolder="gemma-3-12b-it",
    ),
    # Kijai's LTX 2.3 ComfyUI assets
    "LTX23_video_vae_bf16.safetensors": ModelEntry("Kijai/LTX2.3_comfy", comfy_type="vae"),
    "LTX23_audio_vae_bf16.safetensors": ModelEntry("Kijai/LTX2.3_comfy", comfy_type="vae"),
    "ltx-2.3_text_projection_bf16.safetensors": ModelEntry(
        "Kijai/LTX2.3_comfy", comfy_type="text_encoders"
    ),
    # IC-LoRAs
    "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors": ModelEntry(
        "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control", comfy_type="loras"
    ),
    "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors": ModelEntry(
        "Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control", comfy_type="loras"
    ),
    "ltx-2-19b-ic-lora-detailer.safetensors": ModelEntry(
        "Lightricks/LTX-2-19b-IC-LoRA-Detailer", comfy_type="loras"
    ),
    "ltx-2-19b-ic-lora-pose-control.safetensors": ModelEntry(
        "Lightricks/LTX-2-19b-IC-LoRA-Pose-Control", comfy_type="loras"
    ),
    # Camera-control LoRAs (one repo each — explicit hyphen-aware capitalization
    # produces "Dolly-In", "Dolly-Out", etc. matching the actual HF org repo names.)
    **{
        f"ltx-2-19b-lora-camera-control-{movement}.safetensors": ModelEntry(
            f"Lightricks/LTX-2-19b-LoRA-Camera-Control-{'-'.join(p.capitalize() for p in movement.split('-'))}",
            comfy_type="loras",
        )
        for movement in (
            "static",
            "dolly-in",
            "dolly-out",
            "dolly-left",
            "dolly-right",
            "jib-up",
            "jib-down",
        )
    },
}


LOADER_NODE_TYPES: tuple[str, ...] = (
    "CheckpointLoaderSimple",
    "UNETLoader",
    "UnetLoaderGGUF",
    "VAELoader",
    "VAELoaderKJ",
    "LoraLoader",
    "Power Lora Loader (rgthree)",
    "LTXVGemmaCLIPModelLoader",
    "LatentUpscaleModelLoader",
    "DualCLIPLoader",
)


def walk_workflow_for_models(workflow: dict) -> set[str]:
    """Return the set of model filenames referenced by loader nodes in the workflow.

    Pulls filenames from nodes whose `type` matches a known loader. Filenames are
    typically in `widgets_values[0]` (CheckpointLoaderSimple) or in nested rows
    (Power Lora Loader). Falls back to scanning all string-valued widget entries
    for `*.safetensors` / `*.gguf`.
    """
    needed: set[str] = set()
    for node in workflow.get("nodes", []):
        if node.get("type") not in LOADER_NODE_TYPES:
            continue
        widgets = node.get("widgets_values") or []
        for value in _flatten_widget_values(widgets):
            if isinstance(value, str) and (
                value.endswith(".safetensors")
                or value.endswith(".gguf")
                or value == "tokenizer.model"
                or value.endswith(".json")
            ):
                needed.add(value)
    return needed


def _flatten_widget_values(values):
    """Walk nested list/dict widget structures, yielding leaf values."""
    if isinstance(values, dict):
        yield from _flatten_widget_values(list(values.values()))
        return
    for v in values:
        if isinstance(v, (list, tuple)):
            yield from _flatten_widget_values(v)
        elif isinstance(v, dict):
            yield from _flatten_widget_values(list(v.values()))
        else:
            yield v


@dataclass
class DownloadEvent:
    filename: str
    mb_done: float
    mb_total: float


def _on_spaces() -> bool:
    return bool(os.environ.get("SPACES_ZERO_GPU"))


def _comfy_models_dir() -> pathlib.Path:
    raw = os.environ.get("COMFY_MODELS_DIR")
    if raw:
        return pathlib.Path(raw)
    if _on_spaces():
        return pathlib.Path("/data/models")
    return pathlib.Path(__file__).parent / "comfyui" / "models"


def ensure_models(filenames: set[str]) -> Iterator[DownloadEvent]:
    """Ensure each requested model is materialized in comfyui/models/<type>/.

    Local mode: hf_hub_download into the user's HF cache; symlink to comfyui/models/.
    Spaces mode: hf_hub_download with cache_dir=/data; comfyui/models/ symlinks
    point into /data.

    Files not in MODEL_REGISTRY are skipped (with a warning) — useful when the
    workflow has been manually customized with non-canonical filenames that the
    user supplies via their own ComfyUI install.

    Yields DownloadEvent on each successfully materialized file (mb_done==mb_total
    when already cached locally).
    """
    comfy_models = _comfy_models_dir()
    cache_dir = pathlib.Path(
        os.environ.get(
            "HF_HUB_CACHE",
            pathlib.Path.home() / ".cache" / "huggingface" / "hub",
        )
    )

    for filename in filenames:
        if filename not in MODEL_REGISTRY:
            logger.warning(
                "model file %r not in MODEL_REGISTRY; skipping. "
                "Add an entry to MODEL_REGISTRY or override the loader in the workflow.",
                filename,
            )
            continue
        entry = MODEL_REGISTRY[filename]

        # Resolve source: hf_hub_download returns the cache path (or downloads).
        try:
            source = pathlib.Path(
                hf_hub_download(
                    repo_id=entry.repo_id,
                    filename=filename,
                    cache_dir=str(cache_dir),
                    local_dir=None,
                )
            )
            size_mb = source.stat().st_size / 1024 / 1024
            yield DownloadEvent(filename, size_mb, size_mb)
        except Exception as exc:
            # Fall back to scanning the cache for a placeholder file (test mode + offline mode).
            candidates = list(cache_dir.rglob(filename))
            if not candidates:
                logger.warning(
                    "could not download or locate %r in HF cache: %s; skipping",
                    filename,
                    exc,
                )
                continue
            source = candidates[0]
            yield DownloadEvent(filename, 0.0, 0.0)

        # Build symlink target inside comfy_models
        dest_dir = comfy_models / entry.comfy_type
        if entry.subfolder:
            dest_dir = dest_dir / entry.subfolder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename

        if dest.is_symlink() or dest.exists():
            dest.unlink()
        dest.symlink_to(source)


def ensure_models_for_mode(mode: str) -> Iterator[DownloadEvent]:
    """Convenience: walk a mode's workflow and ensure all referenced models exist."""
    import workflow as workflow_module  # local import to avoid cycle at import time

    wf = workflow_module.load_template(mode)
    needed = walk_workflow_for_models(wf)
    yield from ensure_models(needed)
