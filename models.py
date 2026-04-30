"""Model file registry: maps filename -> (HuggingFace repo, subfolder).

Lookups are by filename only — the same filename in two different repos is not
supported. If that ever happens we'll qualify by ComfyUI loader-type.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEntry:
    repo_id: str
    subfolder: str = ""
    comfy_type: str = "checkpoints"  # ComfyUI models/<comfy_type>/ subdirectory


MODEL_REGISTRY: dict[str, ModelEntry] = {
    # Main LTX 2.3 transformer + LoRAs + upscalers
    "ltx-2.3-22b-distilled.safetensors": ModelEntry(
        "Lightricks/LTX-2.3", comfy_type="checkpoints"
    ),
    "ltx-2.3-22b-dev.safetensors": ModelEntry(
        "Lightricks/LTX-2.3", comfy_type="checkpoints"
    ),
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
    "LTX23_video_vae_bf16.safetensors": ModelEntry(
        "Kijai/LTX2.3_comfy", comfy_type="vae"
    ),
    "LTX23_audio_vae_bf16.safetensors": ModelEntry(
        "Kijai/LTX2.3_comfy", comfy_type="vae"
    ),
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
