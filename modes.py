"""MODE_REGISTRY — one Mode entry per generation mode.

Each Mode declares:
- name: short id ("t2v", "i2v", ...)
- label: display name
- icon: single-character or emoji icon for the sidebar
- stage_map: list of (label, expected_share_pct) for the status banner
- parameterize_fn: (Gradio inputs dict) -> list[(node_id, widget_index, value)]

The parameterize_fn is the only mode-specific logic. Everything else (workflow
loading, validation, dispatch) is mode-agnostic and lives in workflow.py /
backend.py.

Tasks 11 (T2V + I2V) and 12 (A2V + Lipsync + Keyframe + Style) populate
MODE_REGISTRY. This task only sets up the dataclass and the empty container.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

Patch = tuple[int, int, Any]
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


# Filled in by tasks 11–12.
MODE_REGISTRY: dict[str, Mode] = {}


# ---------------------------------------------------------------------------
# Node-id constants — captured from workflows/{t2v,i2v}.json on 2026-04-30.
#
# The master workflow uses rgthree's GetNode/SetNode for indirection. SetNodes
# named "pos"/"neg" expose the *outputs* of CLIPTextEncode, not the prompt
# strings. So the canonical place to set the prompt text is the CLIPTextEncode
# node itself.
#
# Width/Height/FPS are INTConstant nodes whose values feed downstream Set_*
# variables.  Clip length comes from a mxSlider (in seconds, then multiplied by
# FPS via a MathExpression to compute frames).  No SetNode for "noise"/seed
# survived the extraction, so seed is intentionally NOT patched here — the
# template's hard-coded value is used until we wire RandomNoise injection in
# Task 12+.
#
# LoRA rows live inside a single Power Lora Loader (rgthree) node whose
# widgets_values is a list of dicts. Patching a specific row requires knowing
# the index, and the canonical mapping (camera_lora value -> row index) belongs
# in models.py once camera-LoRA selection lands. Deferred for now.
# ---------------------------------------------------------------------------

T2V_NODE_PROMPT = 5536            # CLIPTextEncode positive — wv[0] = prompt
T2V_NODE_NEG_PROMPT = 5537        # CLIPTextEncode negative — wv[0] = negative prompt
T2V_NODE_WIDTH = 5383             # INTConstant "Width" — wv[0]
T2V_NODE_HEIGHT = 5382            # INTConstant "Height" — wv[0]
T2V_NODE_FPS = 5445               # INTConstant "FPS" — wv[0]
T2V_NODE_CLIP_LENGTH = 196        # mxSlider "Clip Length ( in seconds )" — wv[0]

I2V_NODE_PROMPT = 5536
I2V_NODE_NEG_PROMPT = 5537
I2V_NODE_WIDTH = 5383
I2V_NODE_HEIGHT = 5382
I2V_NODE_FPS = 5445
I2V_NODE_CLIP_LENGTH = 196
I2V_NODE_IMAGE = 149              # LoadImage "Load Image1" — wv[0] = filename

# Mode-specific media nodes — captured from workflows/{a2v,lipsync,keyframe,style}.json
# on 2026-04-30. All four templates contain the same node ids for these inputs (the
# Loaders group is shared across modes); only a subset is wired into each mode's
# pipeline.
#
# VHS_LoadAudioUpload and VHS_LoadVideo carry dict-style widgets_values keyed by
# "audio"/"video". The current set_input helper is list-indexed; passing
# widget_index=0 against a dict adds a numeric "0" key without replacing the
# canonical "audio"/"video" entry. The runtime file-path swap is therefore not
# yet wired — Task 12 only validates the patch tuple set. Real path injection
# lands when backend.py grows file-staging in Task 17.

A2V_NODE_PROMPT = 5536
A2V_NODE_NEG_PROMPT = 5537
A2V_NODE_WIDTH = 5383
A2V_NODE_HEIGHT = 5382
A2V_NODE_FPS = 5445
A2V_NODE_CLIP_LENGTH = 196
A2V_NODE_AUDIO = 5400             # VHS_LoadAudioUpload — dict wv keyed by "audio"

LIPSYNC_NODE_PROMPT = 5536
LIPSYNC_NODE_NEG_PROMPT = 5537
LIPSYNC_NODE_FPS = 5445
LIPSYNC_NODE_CLIP_LENGTH = 196
LIPSYNC_NODE_IMAGE = 149          # LoadImage "Load Image1" — wv[0] = filename
LIPSYNC_NODE_AUDIO = 5400         # VHS_LoadAudioUpload — dict wv keyed by "audio"

KEYFRAME_NODE_PROMPT = 5536
KEYFRAME_NODE_NEG_PROMPT = 5537
KEYFRAME_NODE_FPS = 5445
KEYFRAME_NODE_CLIP_LENGTH = 196
KEYFRAME_NODE_FIRST_FRAME = 149   # LoadImage "Load Image1" — wv[0] = filename
KEYFRAME_NODE_LAST_FRAME = 5437   # LoadImage "Load Image2" — wv[0] = filename

STYLE_NODE_PROMPT = 5536
STYLE_NODE_NEG_PROMPT = 5537
STYLE_NODE_FPS = 5445
STYLE_NODE_CLIP_LENGTH = 196
STYLE_NODE_INPUT_VIDEO = 5444     # VHS_LoadVideo — dict wv keyed by "video"


def _frames_to_seconds(frames: int, fps: int) -> int:
    """Convert (frames, fps) to integer seconds for the mxSlider clip-length widget.

    The downstream MathExpression is `a*b+1` (a=seconds, b=fps -> total frames),
    so for a target frame count F at fps R we need seconds = ceil((F - 1) / R).
    Round up so the slider is never short of the requested frames.
    """
    if fps <= 0:
        return 1
    return max(1, -(-(frames - 1) // fps))


def _t2v_parameterize(inp: dict[str, Any]) -> list[Patch]:
    return [
        (T2V_NODE_PROMPT, 0, inp["prompt"]),
        (T2V_NODE_NEG_PROMPT, 0, inp.get("negative_prompt", "")),
        (T2V_NODE_WIDTH, 0, int(inp["width"])),
        (T2V_NODE_HEIGHT, 0, int(inp["height"])),
        (T2V_NODE_FPS, 0, int(inp["fps"])),
        (T2V_NODE_CLIP_LENGTH, 0, _frames_to_seconds(int(inp["frames"]), int(inp["fps"]))),
    ]


def _i2v_parameterize(inp: dict[str, Any]) -> list[Patch]:
    return [
        (I2V_NODE_PROMPT, 0, inp["prompt"]),
        (I2V_NODE_NEG_PROMPT, 0, inp.get("negative_prompt", "")),
        (I2V_NODE_IMAGE, 0, inp["image"]),
        (I2V_NODE_WIDTH, 0, int(inp["width"])),
        (I2V_NODE_HEIGHT, 0, int(inp["height"])),
        (I2V_NODE_FPS, 0, int(inp["fps"])),
        (I2V_NODE_CLIP_LENGTH, 0, _frames_to_seconds(int(inp["frames"]), int(inp["fps"]))),
    ]


def _a2v_parameterize(inp: dict[str, Any]) -> list[Patch]:
    return [
        (A2V_NODE_PROMPT, 0, inp["prompt"]),
        (A2V_NODE_NEG_PROMPT, 0, inp.get("negative_prompt", "")),
        (A2V_NODE_AUDIO, 0, inp["audio"]),
        (A2V_NODE_WIDTH, 0, int(inp["width"])),
        (A2V_NODE_HEIGHT, 0, int(inp["height"])),
        (A2V_NODE_FPS, 0, int(inp["fps"])),
        (A2V_NODE_CLIP_LENGTH, 0, _frames_to_seconds(int(inp["frames"]), int(inp["fps"]))),
    ]


def _lipsync_parameterize(inp: dict[str, Any]) -> list[Patch]:
    return [
        (LIPSYNC_NODE_PROMPT, 0, inp["prompt"]),
        (LIPSYNC_NODE_NEG_PROMPT, 0, inp.get("negative_prompt", "")),
        (LIPSYNC_NODE_IMAGE, 0, inp["image"]),
        (LIPSYNC_NODE_AUDIO, 0, inp["audio"]),
        (LIPSYNC_NODE_FPS, 0, int(inp["fps"])),
        (LIPSYNC_NODE_CLIP_LENGTH, 0, _frames_to_seconds(int(inp["frames"]), int(inp["fps"]))),
    ]


def _keyframe_parameterize(inp: dict[str, Any]) -> list[Patch]:
    return [
        (KEYFRAME_NODE_PROMPT, 0, inp["prompt"]),
        (KEYFRAME_NODE_NEG_PROMPT, 0, inp.get("negative_prompt", "")),
        (KEYFRAME_NODE_FIRST_FRAME, 0, inp["first_frame"]),
        (KEYFRAME_NODE_LAST_FRAME, 0, inp["last_frame"]),
        (KEYFRAME_NODE_FPS, 0, int(inp["fps"])),
        (KEYFRAME_NODE_CLIP_LENGTH, 0, _frames_to_seconds(int(inp["frames"]), int(inp["fps"]))),
    ]


def _style_parameterize(inp: dict[str, Any]) -> list[Patch]:
    return [
        (STYLE_NODE_PROMPT, 0, inp["prompt"]),
        (STYLE_NODE_NEG_PROMPT, 0, inp.get("negative_prompt", "")),
        (STYLE_NODE_INPUT_VIDEO, 0, inp["input_video"]),
        (STYLE_NODE_FPS, 0, int(inp["fps"])),
        (STYLE_NODE_CLIP_LENGTH, 0, _frames_to_seconds(int(inp["frames"]), int(inp["fps"]))),
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

_LIPSYNC_STAGES = [
    Stage("Encode prompt", 5),
    Stage("Encode image", 3),
    Stage("Encode audio", 5),
    Stage("Diffusion (Stage 1)", 52),
    Stage("Spatial upscale", 7),
    Stage("Diffusion (Stage 2)", 18),
    Stage("Decode video", 10),
]

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
    Stage("Decode source video", 5),
    Stage("Diffusion (Stage 1)", 55),
    Stage("Spatial upscale", 7),
    Stage("Diffusion (Stage 2)", 18),
    Stage("Decode video", 10),
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
