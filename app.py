# app.py
"""LTX 2.3 All-in-One — Gradio entry point."""
from __future__ import annotations

import os
import pathlib
import sys

import gradio as gr

import modes
import ui


# ---------------------------------------------------------------------------
# Bootstrap — runs once on cold start.
# ---------------------------------------------------------------------------

def _on_spaces() -> bool:
    return bool(os.environ.get("SPACES_ZERO_GPU"))


COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
# Pinned to the same commit the local git submodule uses (set in Task 5).
# Override via env var only when intentionally testing a different ComfyUI version.
COMFYUI_COMMIT = os.environ.get(
    "LTX23_AIO_COMFYUI_COMMIT",
    "eb0686bbb60c83e44c3a3e4f7defd0f589cfef10",
)

CUSTOM_NODES_PINNED: list[tuple[str, str]] = [
    ("https://github.com/Lightricks/ComfyUI-LTXVideo.git", "main"),
    ("https://github.com/kijai/ComfyUI-KJNodes.git", "main"),
    ("https://github.com/rgthree/rgthree-comfy.git", "main"),
    ("https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git", "main"),
    ("https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git", "main"),
]


def _git_clone(url: str, dst: pathlib.Path, ref: str) -> None:
    import subprocess
    subprocess.check_call(["git", "clone", "--depth", "1", "--branch", ref, url, str(dst)])


def _bootstrap() -> None:
    on_spaces = _on_spaces()
    comfy_dir = pathlib.Path("/data/comfyui" if on_spaces else "comfyui")

    if on_spaces and not comfy_dir.exists():
        comfy_dir.parent.mkdir(parents=True, exist_ok=True)
        _git_clone(COMFYUI_REPO, comfy_dir, ref=COMFYUI_COMMIT)
        for node_url, node_ref in CUSTOM_NODES_PINNED:
            name = node_url.rstrip(".git").rsplit("/", 1)[-1]
            _git_clone(node_url, comfy_dir / "custom_nodes" / name, ref=node_ref)
        # Install custom node deps
        import subprocess
        for cn in (comfy_dir / "custom_nodes").iterdir():
            req = cn / "requirements.txt"
            if req.exists():
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])

    if str(comfy_dir) not in sys.path:
        sys.path.insert(0, str(comfy_dir))
    os.environ.setdefault(
        "COMFY_MODELS_DIR",
        str(pathlib.Path("/data/models") if on_spaces else (comfy_dir / "models")),
    )


_bootstrap()


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

_CUSTOM_CSS = """
.status-card { padding: 14px 16px; border-radius: 10px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); }
.status-row { display: flex; gap: 14px; align-items: center; margin-bottom: 8px; }
.status-stage { font-weight: 600; }
.status-meta { font-size: 12px; opacity: 0.75; }
.status-bar { height: 6px; background: rgba(255,255,255,0.08); border-radius: 99px; overflow: hidden; }
.status-fill { height: 100%; background: linear-gradient(90deg,#6ea8fe,#8de9fe); transition: width .3s; }
.status-mem { font-size: 11px; opacity: 0.6; margin-top: 6px; font-family: ui-monospace, monospace; }
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(), title="LTX 2.3 All-in-One", css=_CUSTOM_CSS) as app:
        gr.Markdown("# ⚡ LTX 2.3 All-in-One")
        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                _render_sidebar()
            with gr.Column(scale=4):
                handles = _render_mode_panels()

        for name, h in handles.items():
            inputs = _collect_inputs_for_mode(name, h)
            h["generate_btn"].click(
                fn=_make_handler(name, h),
                inputs=inputs,
                outputs=[h["status"], h["video_out"]],
            )
    return app


def _render_sidebar() -> None:
    gr.Markdown("### Modes")
    for name, mode in modes.MODE_REGISTRY.items():
        gr.Markdown(f"- {mode.icon} {mode.label}")
    gr.Markdown("---\n### Models")
    gr.Button("Unload all models", variant="secondary")


def _render_mode_panels() -> dict[str, dict]:
    """Render one form per mode. Returns the component handles keyed by mode."""
    handles: dict[str, dict] = {}
    with gr.Tabs() as tabs:
        for name, mode in modes.MODE_REGISTRY.items():
            with gr.Tab(label=f"{mode.icon} {mode.label}"):
                handles[name] = _render_one_mode(name)
    return handles


def _render_one_mode(name: str) -> dict:
    """Render a per-mode form. Returns component handles for the generate handler."""
    mode = modes.MODE_REGISTRY[name]
    handles: dict = {"mode": name}

    with gr.Row():
        with gr.Column(scale=2):
            handles["prompt"] = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the shot...")

            # Mode-specific media inputs
            if name == "i2v":
                handles["image"] = gr.Image(label="Source image", type="filepath")
            elif name == "a2v":
                handles["audio"] = gr.Audio(label="Source audio", type="filepath")
            elif name == "lipsync":
                handles["image"] = gr.Image(label="Portrait", type="filepath")
                handles["audio"] = gr.Audio(label="Speech audio", type="filepath")
            elif name == "keyframe":
                handles["first_frame"] = gr.Image(label="First frame", type="filepath")
                handles["last_frame"] = gr.Image(label="Last frame", type="filepath")
            elif name == "style":
                handles["input_video"] = gr.Video(label="Source video")

            handles["preset"] = ui.preset_bar()
            with gr.Row():
                handles["width"] = gr.Slider(256, 1280, value=512, step=32, label="Width")
                handles["height"] = gr.Slider(256, 1280, value=768, step=32, label="Height")
            with gr.Row():
                handles["frames"] = gr.Slider(9, 121, value=81, step=8, label="Frames (8k+1)")
                handles["fps"] = gr.Slider(8, 30, value=24, step=1, label="FPS")
            handles["seed"] = gr.Number(label="Seed", value=42, precision=0)

            with gr.Accordion("Advanced ▾", open=False):
                handles["lora"] = ui.lora_chrome(name)
                handles["negative_prompt"] = gr.Textbox(label="Negative prompt", lines=2)

            handles["generate_btn"] = gr.Button("▶ Generate", variant="primary", size="lg")

        with gr.Column(scale=2):
            handles["status"] = ui.status_banner()
            handles["video_out"] = gr.Video(label="Output", autoplay=True)
            handles["history"] = gr.Markdown("")

    return handles


import time
from typing import Any

import workflow as wf_module
import backend as backend_module

_BACKEND: backend_module.ComfyUILibraryBackend | None = None


def _get_backend() -> backend_module.ComfyUILibraryBackend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = backend_module.ComfyUILibraryBackend()
    return _BACKEND


PRESET_DURATION = {"Fast": 60, "Balanced": 120, "Quality": 300}


async def _on_generate(mode_name: str, **inputs: Any):
    """Generate handler — async generator yielding (status_html, video_path)."""
    mode = modes.MODE_REGISTRY[mode_name]

    # Translate UI inputs into the parameterize_fn input dict.
    params: dict[str, Any] = {
        "prompt": inputs.get("prompt", ""),
        "negative_prompt": inputs.get("negative_prompt", ""),
        "preset": inputs.get("preset", "Balanced").lower(),
        "width": int(inputs.get("width", 512)),
        "height": int(inputs.get("height", 768)),
        "frames": int(inputs.get("frames", 81)),
        "fps": int(inputs.get("fps", 24)),
        "seed": int(inputs.get("seed", 42)),
    }
    for k in ("image", "audio", "first_frame", "last_frame", "input_video",
              "camera_lora", "camera_strength",
              "detailer_on", "detailer_strength",
              "ic_lora", "ic_strength", "pose_on", "audio_cfg", "image_strength"):
        if k in inputs:
            params[k] = inputs[k]

    patches = mode.parameterize_fn(params)
    workflow = wf_module.load_template(mode_name)
    for patch in patches:
        wf_module.set_input(workflow, *patch)
    wf_module.validate(workflow)

    backend = _get_backend()
    duration = PRESET_DURATION.get(inputs.get("preset", "Balanced"), 120)

    started = time.time()
    async for event in backend.submit(mode_name, workflow, gpu_duration=duration):
        elapsed = time.time() - started
        if isinstance(event, backend_module.DownloadEvent):
            status = ui.render_status(
                stage_index=0,
                stage_label=f"Downloading {event.filename}",
                step=int(event.mb_done),
                total_steps=int(max(event.mb_total, 1)),
                elapsed_s=elapsed, eta_s=0,
            )
            yield status, gr.update()
        elif isinstance(event, backend_module.ProgressEvent):
            stage = (
                mode.stage_map[event.stage]
                if event.stage < len(mode.stage_map)
                else mode.stage_map[-1]
            )
            eta = (elapsed / max(event.step, 1)) * (event.total_steps - event.step)
            status = ui.render_status(
                stage_index=event.stage + 1,
                stage_label=stage.label,
                step=event.step,
                total_steps=event.total_steps,
                elapsed_s=elapsed, eta_s=eta,
            )
            yield status, gr.update()
        elif isinstance(event, backend_module.OutputEvent):
            yield ui._render_idle(), event.video_path
        elif isinstance(event, backend_module.ErrorEvent):
            error_html = (
                f'<div class="status-card status-error">'
                f'  <div class="status-row"><span class="status-stage">Error · {event.category}</span></div>'
                f'  <div>{event.message}</div>'
                f'</div>'
            )
            yield error_html, gr.update()


def _input_keys_for_mode(mode_name: str, h: dict) -> list[str]:
    base = ["prompt", "preset", "width", "height", "frames", "fps", "seed"]
    if mode_name == "i2v":
        base.append("image")
    elif mode_name == "a2v":
        base.append("audio")
    elif mode_name == "lipsync":
        base.extend(["image", "audio"])
    elif mode_name == "keyframe":
        base.extend(["first_frame", "last_frame"])
    elif mode_name == "style":
        base.append("input_video")
    base.append("negative_prompt")
    base.extend(["camera_lora", "camera_strength", "detailer_on", "detailer_strength"])
    if h["lora"].ic_lora is not None:
        base.extend(["ic_lora", "ic_strength"])
    if h["lora"].pose_on is not None:
        base.append("pose_on")
    return base


def _collect_inputs_for_mode(mode_name: str, h: dict) -> list:
    """Gather the gr.Component handles to pass into _on_generate."""
    base = [h["prompt"], h["preset"], h["width"], h["height"], h["frames"], h["fps"], h["seed"]]
    if mode_name == "i2v":
        base.append(h["image"])
    elif mode_name == "a2v":
        base.append(h["audio"])
    elif mode_name == "lipsync":
        base.extend([h["image"], h["audio"]])
    elif mode_name == "keyframe":
        base.extend([h["first_frame"], h["last_frame"]])
    elif mode_name == "style":
        base.append(h["input_video"])
    base.append(h["negative_prompt"])
    base.extend([
        h["lora"].camera_lora, h["lora"].camera_strength,
        h["lora"].detailer_on, h["lora"].detailer_strength,
    ])
    if h["lora"].ic_lora is not None:
        base.extend([h["lora"].ic_lora, h["lora"].ic_strength])
    if h["lora"].pose_on is not None:
        base.append(h["lora"].pose_on)
    return base


def _make_handler(mode_name: str, h: dict):
    keys = _input_keys_for_mode(mode_name, h)

    async def handler(*values):
        kwargs = dict(zip(keys, values))
        async for output in _on_generate(mode_name, **kwargs):
            yield output

    return handler


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
