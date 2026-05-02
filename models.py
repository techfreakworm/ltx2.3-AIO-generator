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
    subfolder: str = ""  # path within the HF repo
    comfy_type: str = "checkpoints"  # ComfyUI models/<comfy_type>/ subdirectory
    # If the workflow expects a different filename than what's in the HF repo
    # (e.g. user's local "ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors"
    # is actually `_transformer_only_bf16.safetensors` in Kijai's repo), set
    # source_filename to the actual repo filename. The local symlink/copy uses
    # the registry key as its name.
    source_filename: str | None = None


MODEL_REGISTRY: dict[str, ModelEntry] = {
    # Main LTX 2.3 transformer + LoRAs + upscalers
    "ltx-2.3-22b-distilled.safetensors": ModelEntry("Lightricks/LTX-2.3", comfy_type="checkpoints"),
    "ltx-2.3-22b-dev.safetensors": ModelEntry("Lightricks/LTX-2.3", comfy_type="checkpoints"),
    "ltx-2.3-spatial-upscaler-x2-1.0.safetensors": ModelEntry(
        "Lightricks/LTX-2.3", comfy_type="latent_upscale_models"
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
    # Kijai's LTX 2.3 ComfyUI assets — files live in vae/ and text_encoders/
    # subfolders within the repo, not at root.
    "LTX23_video_vae_bf16.safetensors": ModelEntry(
        "Kijai/LTX2.3_comfy", subfolder="vae", comfy_type="vae"
    ),
    "LTX23_audio_vae_bf16.safetensors": ModelEntry(
        "Kijai/LTX2.3_comfy", subfolder="vae", comfy_type="vae"
    ),
    "ltx-2.3_text_projection_bf16.safetensors": ModelEntry(
        "Kijai/LTX2.3_comfy", subfolder="text_encoders", comfy_type="text_encoders"
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
    # ----- Renamed/aliased filenames the user's master workflow references.
    # The names look like quantized variants (FP4, FP8, GGUF) but the actual
    # bytes behind them are BF16 — the user's local setup uses symlinks to
    # canonical sources. On Spaces we download the same canonical sources via
    # huggingface_hub and place them under the workflow-expected filename.
    # All of these entries set `subfolder` to the path within the repo and
    # rely on hf_hub_download returning the cached snapshot path (which we
    # then symlink to comfy_models/<comfy_type>/<filename>).
    "gemma_3_12B_it_fp4_mixed.safetensors": ModelEntry(
        # Comfy-Org/ltx-2 ships BF16 Gemma packed as `gemma_3_12B_it.safetensors`
        # in split_files/text_encoders/. The workflow expects the FP4-named
        # variant; we serve the same file under that name.
        "Comfy-Org/ltx-2",
        subfolder="split_files/text_encoders",
        comfy_type="text_encoders",
        source_filename="gemma_3_12B_it.safetensors",
    ),
    "gemma_3_12B_it.safetensors": ModelEntry(
        "Comfy-Org/ltx-2",
        subfolder="split_files/text_encoders",
        comfy_type="text_encoders",
    ),
    "ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors": ModelEntry(
        # Kijai's BF16 transformer-only — actual repo filename has `_bf16` suffix.
        "Kijai/LTX2.3_comfy",
        subfolder="diffusion_models",
        comfy_type="diffusion_models",
        source_filename="ltx-2.3-22b-dev_transformer_only_bf16.safetensors",
    ),
    "ltx-2-3-22b-dev-Q4_K_M.gguf": ModelEntry(
        # Unsloth's GGUF in BF16 (named `…-BF16.gguf` in repo).
        "unsloth/LTX-2.3-GGUF",
        comfy_type="diffusion_models",
        source_filename="ltx-2.3-22b-dev-BF16.gguf",
    ),
    "taeltx2_3.safetensors": ModelEntry(
        "Kijai/LTX2.3_comfy",
        subfolder="vae",
        comfy_type="vae",
    ),
    "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors": ModelEntry(
        "Kijai/LTX2.3_comfy",
        subfolder="loras",
        comfy_type="loras",
    ),
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


_USER_INPUT_LOADERS = {"LoadImage", "VHS_LoadVideo", "VHS_LoadAudioUpload"}
_MODEL_EXTS = (".safetensors", ".gguf", ".pt", ".bin", ".ckpt")


def _walk_for_filenames(value, into: set[str]) -> None:
    """Depth-first walk of a node's inputs, picking out model filenames.

    Power Lora Loader stores its rows nested as `inputs.lora_1 = {on, lora,
    strength}` and similar — a flat values() loop misses these. Recurse
    through dicts and lists/tuples so nested filenames are caught.

    Skips Power Lora Loader rows with `on: false` — those LoRAs aren't
    actually loaded at runtime so there's no point downloading them.
    """
    if isinstance(value, str):
        if value.endswith(_MODEL_EXTS) or value == "tokenizer.model":
            into.add(value)
    elif isinstance(value, dict):
        # Power Lora Loader row: {"on": bool, "lora": "...", "strength": ...}
        if "on" in value and "lora" in value and not value.get("on"):
            return
        for v in value.values():
            _walk_for_filenames(v, into)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _walk_for_filenames(v, into)


def walk_workflow_for_models(workflow: dict) -> set[str]:
    """Return the set of model filenames referenced by the API-format workflow.

    Walks `{node_id: {class_type, inputs}}` and recursively scans each node's
    inputs for strings ending in a model extension. Skips loaders that read
    user-supplied files (LoadImage, VHS_LoadVideo, VHS_LoadAudioUpload).
    Unknown filenames are harmless — `ensure_models` log-warns and skips
    anything not in the registry, so being inclusive here costs nothing.
    """
    needed: set[str] = set()
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") in _USER_INPUT_LOADERS:
            continue
        _walk_for_filenames(node.get("inputs") or {}, needed)
    return needed


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
        return pathlib.Path.home() / "comfyui" / "models"
    return pathlib.Path(__file__).parent / "comfyui" / "models"


def ensure_models(filenames: set[str]) -> Iterator[DownloadEvent]:
    """Ensure each requested model is materialized in comfyui/models/<type>/.

    Local mode: hf_hub_download into the user's HF cache; symlink to comfyui/models/.
    Spaces mode: hf_hub_download with cache_dir under $HOME (no /data dependency);
    files staged at ~/comfyui/models/<comfy_type>/<filename>.

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

        # Short-circuit: if the file is already present at its expected location
        # comfyui/models/<comfy_type>/<filename>, skip. Subfolder is part of the
        # HF source path, not the destination, so the dest is always a flat
        # comfyui/models/<comfy_type>/<filename>.
        existing_dest = comfy_models / entry.comfy_type / filename
        if existing_dest.exists() or existing_dest.is_symlink():
            yield DownloadEvent(filename, 0.0, 0.0)
            continue

        # The HF-side filename may differ from the workflow-expected name
        # (e.g. user's `_fp8_scaled.safetensors` is actually `_bf16.safetensors`
        # in the upstream repo). Honor `source_filename` when set.
        hf_filename = entry.source_filename or filename
        hf_path = f"{entry.subfolder}/{hf_filename}" if entry.subfolder else hf_filename

        try:
            source = pathlib.Path(
                hf_hub_download(
                    repo_id=entry.repo_id,
                    filename=hf_path,
                    cache_dir=str(cache_dir),
                    local_dir=None,
                )
            )
            size_mb = source.stat().st_size / 1024 / 1024
            yield DownloadEvent(filename, size_mb, size_mb)
        except Exception as exc:
            # Fall back to scanning the cache for a matching file (test mode +
            # offline mode). Look for either the workflow filename OR the
            # HF-side filename. Skip `.no_exist/` markers and 0-byte stubs —
            # the HF lib leaves those after a 404, and symlinking them past
            # safetensors yields a confusing "header too small" error
            # downstream.
            def _viable(path):
                try:
                    return ".no_exist" not in path.parts and path.stat().st_size > 64
                except OSError:
                    return False

            candidates = [
                p for p in cache_dir.rglob(filename) if _viable(p)
            ] or [
                p for p in cache_dir.rglob(hf_filename) if _viable(p)
            ]
            if not candidates:
                logger.warning(
                    "could not download or locate %r (hf=%r) in HF cache: %s; skipping",
                    filename, hf_filename, exc,
                )
                continue
            source = candidates[0]
            yield DownloadEvent(filename, 0.0, 0.0)

        # Stage at comfy_models/<comfy_type>/<filename> (workflow-expected name).
        dest_dir = comfy_models / entry.comfy_type
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
