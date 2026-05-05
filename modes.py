"""MODE_REGISTRY — one Mode entry per generation mode.

Each Mode declares:
- name: short id ("t2v", "i2v", ...)
- label: display name
- icon: single-character or emoji icon for the sidebar
- stage_map: list of (label, expected_share_pct) for the status banner
- parameterize_fn: (Gradio inputs dict) -> list[(node_id, field_name, value)]

The workflows live in `workflows/<mode>.json` in ComfyUI's API format
(`{node_id_str: {class_type, inputs}}` — produced by the editor's
"Save (API Format)" feature). That format is what `PromptExecutor.execute()`
consumes directly, so parameterize_fns just patch field values by node id;
no graph→API conversion is needed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# (node_id, field_name, value)
Patch = tuple[str, str, Any]
ParameterizeFn = Callable[[dict[str, Any]], list[Patch]]


@dataclass(frozen=True)
class Stage:
    label: str
    share_pct: int  # rough share of total time, sums to ~100 across stages


@dataclass(frozen=True)
class Mode:
    name: str
    label: str
    icon: str
    parameterize_fn: ParameterizeFn
    stage_map: list[Stage] = field(default_factory=list)


MODE_REGISTRY: dict[str, Mode] = {}


# ---------------------------------------------------------------------------
# Shared user-input node IDs across all 6 mode API workflows.
# Captured 2026-05-01 from `/Users/techfreakworm/Downloads/workflows/*_api.json`
# (master workflow exported via "Save API Format" per mode).
# ---------------------------------------------------------------------------

NODE_PROMPT = "5536"  # CLIPTextEncode (positive) — inputs.text
NODE_NEG_PROMPT = "5537"  # CLIPTextEncode (negative) — inputs.text
NODE_WIDTH = "5383"  # INTConstant — inputs.value
NODE_HEIGHT = "5382"  # INTConstant — inputs.value
NODE_FPS = "5445"  # INTConstant — inputs.value
NODE_CLIP_SECONDS = "196"  # mxSlider — inputs.Xi (length in seconds; frames = Xi*fps+1)
NODE_IMAGE_1 = "149"  # LoadImage (first frame / portrait) — inputs.image
NODE_IMAGE_2 = "5437"  # LoadImage (last frame for keyframe mode) — inputs.image
NODE_AUDIO = "5400"  # VHS_LoadAudioUpload — inputs.audio
NODE_VIDEO = "5444"  # VHS_LoadVideo — inputs.video

# Per-mode RandomNoise (subgraph-internal): id format `<subgraph_inst>:<inner>`.
SEED_NODE_BY_MODE: dict[str, str] = {
    "t2v": "5464:5539",
    "a2v": "463:5540",
    "i2v": "209:5541",
    "lipsync": "521:5542",
    "keyframe": "670:5543",
    "style": "5364:5545",
}


def _seconds_for(frames: int, fps: int) -> int:
    """Inverse of `frames = seconds*fps + 1` from the master's MathExpression."""
    return max(1, (max(1, int(frames)) - 1) // max(1, int(fps)))


def _shared_patches(inp: dict[str, Any], mode: str) -> list[Patch]:
    return [
        (NODE_PROMPT, "text", inp.get("prompt", "")),
        (NODE_NEG_PROMPT, "text", inp.get("negative_prompt", "")),
        (NODE_WIDTH, "value", int(inp.get("width", 512))),
        (NODE_HEIGHT, "value", int(inp.get("height", 768))),
        (NODE_FPS, "value", int(inp.get("fps", 24))),
        (
            NODE_CLIP_SECONDS,
            "Xi",
            _seconds_for(int(inp.get("frames", 81)), int(inp.get("fps", 24))),
        ),
        (SEED_NODE_BY_MODE[mode], "noise_seed", int(inp.get("seed", 42))),
    ]


def _t2v_parameterize(inp: dict[str, Any]) -> list[Patch]:
    return _shared_patches(inp, "t2v")


def _i2v_parameterize(inp: dict[str, Any]) -> list[Patch]:
    return _shared_patches(inp, "i2v") + [
        (NODE_IMAGE_1, "image", inp["image"]),
    ]


def _a2v_parameterize(inp: dict[str, Any]) -> list[Patch]:
    return _shared_patches(inp, "a2v") + [
        (NODE_AUDIO, "audio", inp["audio"]),
    ]


def _lipsync_parameterize(inp: dict[str, Any]) -> list[Patch]:
    return _shared_patches(inp, "lipsync") + [
        (NODE_IMAGE_1, "image", inp["image"]),
        (NODE_AUDIO, "audio", inp["audio"]),
    ]


def _keyframe_parameterize(inp: dict[str, Any]) -> list[Patch]:
    return _shared_patches(inp, "keyframe") + [
        (NODE_IMAGE_1, "image", inp["first_frame"]),
        (NODE_IMAGE_2, "image", inp["last_frame"]),
    ]


def _style_parameterize(inp: dict[str, Any]) -> list[Patch]:
    return _shared_patches(inp, "style") + [
        (NODE_IMAGE_1, "image", inp["image"]),
        (NODE_VIDEO, "video", inp["input_video"]),
        (NODE_VIDEO, "skip_first_frames", 0),
    ]


_T2V_STAGES = [
    Stage("Encode prompt", 5),
    Stage("Diffusion (Stage 1)", 60),
    Stage("Spatial upscale", 7),
    Stage("Diffusion (Stage 2)", 18),
    Stage("Decode video", 10),
]

_I2V_STAGES = [
    Stage("Encode prompt", 5),
    Stage("Encode image", 3),
    Stage("Diffusion (Stage 1)", 55),
    Stage("Spatial upscale", 7),
    Stage("Diffusion (Stage 2)", 20),
    Stage("Decode video", 10),
]

_A2V_STAGES = [
    Stage("Encode prompt", 5),
    Stage("Encode audio", 5),
    Stage("Diffusion (Stage 1)", 55),
    Stage("Spatial upscale", 7),
    Stage("Diffusion (Stage 2)", 18),
    Stage("Decode video", 10),
]

_LIPSYNC_STAGES = list(_A2V_STAGES)
_KEYFRAME_STAGES = [
    Stage("Encode prompt", 5),
    Stage("Encode keyframes", 5),
    Stage("Diffusion (Stage 1)", 55),
    Stage("Spatial upscale", 7),
    Stage("Diffusion (Stage 2)", 18),
    Stage("Decode video", 10),
]
_STYLE_STAGES = [
    Stage("Encode prompt", 5),
    Stage("Encode source video", 10),
    Stage("Diffusion", 70),
    Stage("Decode video", 15),
]


MODE_REGISTRY["t2v"] = Mode(
    name="t2v",
    label="Text → Video",
    icon="📝",
    parameterize_fn=_t2v_parameterize,
    stage_map=_T2V_STAGES,
)
MODE_REGISTRY["i2v"] = Mode(
    name="i2v",
    label="Image → Video",
    icon="🖼",
    parameterize_fn=_i2v_parameterize,
    stage_map=_I2V_STAGES,
)
MODE_REGISTRY["a2v"] = Mode(
    name="a2v",
    label="Audio → Video",
    icon="🎵",
    parameterize_fn=_a2v_parameterize,
    stage_map=_A2V_STAGES,
)
MODE_REGISTRY["lipsync"] = Mode(
    name="lipsync",
    label="Lipsync",
    icon="👄",
    parameterize_fn=_lipsync_parameterize,
    stage_map=_LIPSYNC_STAGES,
)
MODE_REGISTRY["keyframe"] = Mode(
    name="keyframe",
    label="Keyframe → Video",
    icon="🎞",
    parameterize_fn=_keyframe_parameterize,
    stage_map=_KEYFRAME_STAGES,
)
MODE_REGISTRY["style"] = Mode(
    name="style",
    label="Style Transfer",
    icon="🎨",
    parameterize_fn=_style_parameterize,
    stage_map=_STYLE_STAGES,
)
