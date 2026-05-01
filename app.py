# app.py
"""LTX 2.3 All-in-One — Gradio entry point."""

from __future__ import annotations

import os
import pathlib
import random
import sys
import time
from typing import Any

import gradio as gr

import backend as backend_module
import modes
import ui
import workflow as wf_module

# ---------------------------------------------------------------------------
# Bootstrap — runs once on cold start.
# ---------------------------------------------------------------------------


def _on_spaces() -> bool:
    return bool(os.environ.get("SPACES_ZERO_GPU"))


COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
COMFYUI_COMMIT = os.environ.get(
    "LTX23_AIO_COMFYUI_COMMIT",
    "eb0686bbb60c83e44c3a3e4f7defd0f589cfef10",
)

CUSTOM_NODES_PINNED: list[tuple[str, str]] = [
    ("https://github.com/Lightricks/ComfyUI-LTXVideo.git", "2acf7af8991f33b5cc06ec26753cb6e88e057d04"),
    ("https://github.com/kijai/ComfyUI-KJNodes.git", "01d9fa9c983273532cacdf9532c74a93c7dc86d2"),
    ("https://github.com/rgthree/rgthree-comfy.git", "683836c46e898668936c433502504cc0627482c5"),
    ("https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git", "2984ec4c4b93292421888f38db74a5e8802a8ff8"),
    ("https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git", "609f3afaa74b2f88ef9ce8d939626065e3247469"),
]


def _git_clone(url: str, dst: pathlib.Path, ref: str) -> None:
    """Clone *url* at *ref* into *dst*. *ref* may be a branch, tag, or SHA.

    `git clone --branch` only accepts branch/tag names, so we use init+fetch
    which works for any object GitHub allows fetching (default: reachable
    commits in public repos).
    """
    import subprocess

    dst = pathlib.Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "-C", str(dst), "init", "-q"])
    subprocess.check_call(["git", "-C", str(dst), "remote", "add", "origin", url])
    subprocess.check_call(["git", "-C", str(dst), "fetch", "--depth", "1", "origin", ref])
    subprocess.check_call(["git", "-C", str(dst), "checkout", "-q", "FETCH_HEAD"])


def _bootstrap() -> None:
    on_spaces = _on_spaces()
    # /data requires the paid persistent-storage add-on (separate from Pro).
    # Without it, /data is unwritable. $HOME is writable and — because ZeroGPU
    # containers freeze on sleep rather than tear down — the clone persists
    # across calls within a single deploy.
    comfy_dir = (pathlib.Path.home() / "comfyui") if on_spaces else pathlib.Path("comfyui")

    if on_spaces and not comfy_dir.exists():
        print(f"[bootstrap] cold start on Spaces; cloning ComfyUI to {comfy_dir}", flush=True)
        comfy_dir.parent.mkdir(parents=True, exist_ok=True)
        _git_clone(COMFYUI_REPO, comfy_dir, ref=COMFYUI_COMMIT)
        for node_url, node_ref in CUSTOM_NODES_PINNED:
            name = node_url.rstrip(".git").rsplit("/", 1)[-1]
            _git_clone(node_url, comfy_dir / "custom_nodes" / name, ref=node_ref)
        import subprocess

        # ComfyUI core requirements + each custom node's requirements
        for req_path in [
            comfy_dir / "requirements.txt",
            *(cn / "requirements.txt" for cn in (comfy_dir / "custom_nodes").iterdir()),
        ]:
            if req_path.exists():
                print(f"[bootstrap] pip install -r {req_path}", flush=True)
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "--quiet", "-r", str(req_path)]
                )

    if str(comfy_dir) not in sys.path:
        sys.path.insert(0, str(comfy_dir))
    os.environ.setdefault("COMFY_MODELS_DIR", str(comfy_dir / "models"))

    # Stage placeholder input files so the workflow's hard-referenced loaders
    # (LoadImage/VHS_Load*) don't error at runtime even when the active mode
    # doesn't actually use the file. Real user uploads are placed alongside via
    # `_stage_to_comfy_input` later.
    seed_dir = pathlib.Path(__file__).parent / "assets" / "seed_inputs"
    inputs_dir = comfy_dir / "input"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    if seed_dir.exists():
        import shutil

        for src in seed_dir.iterdir():
            if not src.is_file():
                continue
            dst = inputs_dir / src.name
            if not dst.exists():
                try:
                    shutil.copy2(src, dst)
                except OSError as exc:
                    print(f"[bootstrap] could not seed {src.name}: {exc}", flush=True)


_bootstrap()


# ---------------------------------------------------------------------------
# Styling: hide the default top tab strip (sidebar drives selection),
# add status-card styling, plus responsive breakpoints (≤1024px tablet,
# ≤700px mobile).
# ---------------------------------------------------------------------------

_CUSTOM_CSS = """
/* Hide the top tab strip from view, but keep it in the DOM and clickable so
   the sidebar buttons can drive selection via programmatic click. */
.aio-tabs > .tab-nav,
.aio-tabs > div:first-child[role="tablist"],
.aio-tabs > div:first-child:has([role="tab"]) {
    position: absolute !important;
    left: -99999px !important;
    top: -99999px !important;
    height: 0 !important;
    overflow: hidden !important;
    visibility: visible !important;
    pointer-events: auto !important;
}

/* Sidebar nav buttons */
.aio-mode-btn { width: 100%; text-align: left; margin: 2px 0; }
.aio-mode-btn-active { background: rgba(110,168,254,0.15) !important; border-left: 3px solid #6ea8fe !important; }

/* Sidebar headings */
.aio-sidebar-heading { font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.6; margin-top: 16px !important; margin-bottom: 4px !important; }

/* Status banner */
.status-card { padding: 14px 16px; border-radius: 10px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); }
.status-row { display: flex; gap: 14px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; }
.status-stage { font-weight: 600; }
.status-meta { font-size: 12px; opacity: 0.75; }
.status-bar { height: 6px; background: rgba(255,255,255,0.08); border-radius: 99px; overflow: hidden; }
.status-fill { height: 100%; background: linear-gradient(90deg,#6ea8fe,#8de9fe); transition: width .3s; }
.status-mem { font-size: 11px; opacity: 0.6; margin-top: 6px; font-family: ui-monospace, monospace; }
.status-error { background: rgba(255,90,90,0.08); border-color: rgba(255,90,90,0.25); }

/* Model status badge */
.aio-model-badge { padding: 8px 10px; border-radius: 8px; background: rgba(255,255,255,0.04); font-size: 11.5px; font-family: ui-monospace, monospace; opacity: 0.85; }

/* Responsive: tablet */
@media (max-width: 1024px) {
    .aio-sidebar { min-width: 160px !important; }
    .aio-mode-btn { font-size: 13px !important; padding: 6px 10px !important; }
}

/* Responsive: mobile — sidebar collapses to top, single column body */
@media (max-width: 700px) {
    .aio-shell { flex-direction: column !important; }
    .aio-sidebar { width: 100% !important; min-width: unset !important; padding: 0 !important; }
    .aio-body { width: 100% !important; }
    .aio-mode-btn-row { display: grid !important; grid-template-columns: repeat(2, 1fr) !important; gap: 6px !important; padding: 8px !important; }
    .aio-mode-btn { width: 100% !important; font-size: 12.5px !important; padding: 8px !important; text-align: center !important; margin: 0 !important; }
    .aio-sidebar-heading { font-size: 10px !important; margin: 12px 0 4px !important; padding: 0 8px !important; }
    .aio-model-badge { margin: 0 8px !important; word-break: break-word; white-space: normal !important; }
    /* sliders + side-by-side rows: stack vertically on mobile so each value
       gets its own width budget */
    .aio-body .form > div, .aio-body [class*="row"] > div { flex: 1 1 100% !important; min-width: 0 !important; }
    .aio-body [class*="row"] { flex-wrap: wrap !important; }
}
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(), title="LTX 2.3 All-in-One", css=_CUSTOM_CSS) as app:
        gr.Markdown("# ⚡ LTX 2.3 All-in-One")

        with gr.Row(elem_classes=["aio-shell"]):
            # Sidebar
            with gr.Column(scale=1, min_width=200, elem_classes=["aio-sidebar"]):
                gr.Markdown("**Modes**", elem_classes=["aio-sidebar-heading"])
                with gr.Column(elem_classes=["aio-mode-btn-row"]):
                    mode_buttons = {
                        name: gr.Button(
                            f"{m.icon}  {m.label}",
                            elem_classes=["aio-mode-btn"],
                            variant="secondary",
                        )
                        for name, m in modes.MODE_REGISTRY.items()
                    }
                gr.Markdown("**Models**", elem_classes=["aio-sidebar-heading"])
                model_status = gr.HTML(_render_model_status_idle(), elem_id="aio-model-status")
                refresh_btn = gr.Button("Refresh", size="sm", variant="secondary")
                unload_btn = gr.Button("Unload all models", size="sm", variant="secondary")
                gr.Markdown("**Settings**", elem_classes=["aio-sidebar-heading"])
                gr.Markdown(
                    "Output: `comfyui/output/LTX2.3/`<br>"
                    "Set `LTX23_AIO_VRAM=lowvram|normalvram|highvram` to override the auto-detected VRAM tier.",
                    elem_classes=["aio-model-badge"],
                )

            # Body
            with gr.Column(scale=4, elem_classes=["aio-body"]):
                handles, tabs_component = _render_mode_panels()

        # Wire generate buttons
        for name, h in handles.items():
            inputs = _collect_inputs_for_mode(name, h)
            h["generate_btn"].click(
                fn=_make_handler(name, h),
                inputs=inputs,
                outputs=[h["status"], h["video_out"]],
            )

        # Sidebar mode buttons drive Tabs.selected via Gradio's update.
        for name, btn in mode_buttons.items():
            btn.click(
                fn=lambda mode_id=name: gr.Tabs(selected=mode_id),
                inputs=None,
                outputs=[tabs_component],
            )

        # Sidebar model info wiring
        refresh_btn.click(fn=_render_model_status, inputs=None, outputs=[model_status])
        unload_btn.click(fn=_unload_models, inputs=None, outputs=[model_status])

    return app


def _render_model_status_idle() -> str:
    return (
        '<div class="aio-model-badge">device: detecting…<br>'
        "loaded: —<br>free: —</div>"
    )


def _render_model_status() -> str:
    """Best-effort device + memory readout for the sidebar."""
    try:
        be = _get_backend()  # ensure ComfyUI is loaded
    except Exception as exc:
        return f'<div class="aio-model-badge">backend not ready<br>{exc}</div>'
    try:
        import comfy.model_management as mm
        import torch

        device = mm.get_torch_device()
        free_gb = mm.get_free_memory(device) / (1024**3)
        if torch.backends.mps.is_available():
            # MPS unified memory: total physical = total system RAM. The
            # "recommended max" from torch.mps is a soft cap (~75% of total)
            # used by the allocator, but actual free can exceed it because
            # macOS shares RAM between CPU and GPU.
            try:
                import psutil

                total_gb = psutil.virtual_memory().total / (1024**3)
            except Exception:
                total_gb = torch.mps.recommended_max_memory() / (1024**3)
            cap_gb = torch.mps.recommended_max_memory() / (1024**3)
            label = "MPS (unified)"
            extra = f"<br>mps cap: {cap_gb:.1f} GB"
        elif torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            label = "CUDA"
            extra = ""
        else:
            total_gb = 0.0
            label = "CPU"
            extra = ""
        loaded = len(getattr(mm, "current_loaded_models", []))
        return (
            '<div class="aio-model-badge">'
            f"device: {label}<br>"
            f"loaded: {loaded} model(s)<br>"
            f"free: {free_gb:.1f} GB / {total_gb:.1f} GB total"
            f"{extra}"
            "</div>"
        )
    except Exception as exc:
        return f'<div class="aio-model-badge">memory probe failed: {exc}</div>'


def _unload_models() -> str:
    try:
        import comfy.model_management as mm
        import torch

        mm.unload_all_models()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        return f'<div class="aio-model-badge">unload failed: {exc}</div>'
    return _render_model_status()


def _render_mode_panels() -> tuple[dict[str, dict], gr.Tabs]:
    """Render one (hidden-tab) panel per mode. Returns the component handles + the Tabs component."""
    handles: dict[str, dict] = {}
    with gr.Tabs(elem_classes=["aio-tabs"]) as tabs:
        for name, mode in modes.MODE_REGISTRY.items():
            with gr.Tab(label=f"{mode.icon}  {mode.label}", id=name):
                handles[name] = _render_one_mode(name)
    return handles, tabs


def _render_one_mode(name: str) -> dict:
    """Render a per-mode form. Returns component handles for the generate handler."""
    handles: dict = {"mode": name}

    with gr.Row():
        with gr.Column(scale=2, min_width=280):
            handles["prompt"] = gr.Textbox(
                label="Prompt", lines=4, placeholder="Describe the shot..."
            )

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

            # Resolution — up to 4K, /32 step
            with gr.Row():
                handles["width"] = gr.Slider(
                    256, 4096, value=512, step=32, label="Width"
                )
                handles["height"] = gr.Slider(
                    256, 4096, value=768, step=32, label="Height"
                )

            # Length controlled in seconds (matches the master workflow's mxSlider).
            # Frames are derived: frames = round(seconds * fps / 8) * 8 + 1.
            with gr.Row():
                handles["seconds"] = gr.Slider(
                    minimum=1, maximum=30, value=3, step=1,
                    label="Length (seconds)",
                    info="Frames are computed as 8·round(seconds·fps/8)+1 (LTX requires 8k+1)",
                )
                handles["fps"] = gr.Slider(8, 30, value=24, step=1, label="FPS")

            handles["frames_display"] = gr.Markdown("Frames: 73", elem_classes=["aio-frames-display"])

            with gr.Row():
                handles["seed"] = gr.Number(label="Seed", value=42, precision=0, minimum=0)
                handles["randomize_seed"] = gr.Checkbox(label="Randomize seed each run", value=True)

            with gr.Accordion("Advanced ▾", open=False):
                handles["lora"] = ui.lora_chrome(name)
                handles["negative_prompt"] = gr.Textbox(label="Negative prompt", lines=2)

            handles["generate_btn"] = gr.Button("▶ Generate", variant="primary", size="lg")

            # Live frames-display update when seconds/fps change
            def _update_frames(seconds, fps):
                f = max(9, int(round(float(seconds) * float(fps) / 8) * 8) + 1)
                return f"**Frames:** {f}  (`{seconds}s` × `{fps} fps`)"

            handles["seconds"].change(
                fn=_update_frames,
                inputs=[handles["seconds"], handles["fps"]],
                outputs=[handles["frames_display"]],
            )
            handles["fps"].change(
                fn=_update_frames,
                inputs=[handles["seconds"], handles["fps"]],
                outputs=[handles["frames_display"]],
            )

        with gr.Column(scale=2, min_width=280):
            handles["status"] = ui.status_banner()
            handles["video_out"] = gr.Video(label="Output", autoplay=True)
            handles["history"] = gr.Markdown("")

    return handles


# ---------------------------------------------------------------------------
# Backend wiring
# ---------------------------------------------------------------------------

_BACKEND: backend_module.ComfyUILibraryBackend | None = None


def _get_backend() -> backend_module.ComfyUILibraryBackend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = backend_module.ComfyUILibraryBackend()
    return _BACKEND


_COMFY_INPUT_DIR = pathlib.Path(__file__).parent / "comfyui" / "input"


def _stage_to_comfy_input(file_path) -> str | None:
    """Copy/stage a path into comfyui/input/ so ComfyUI's LoadImage etc. can find it."""
    if not file_path:
        return None
    if not isinstance(file_path, (str, pathlib.Path)):
        file_path = (
            file_path.get("name") or file_path.get("path") or file_path.get("orig_name")
            if isinstance(file_path, dict)
            else None
        )
        if not file_path:
            return None
    src = pathlib.Path(file_path)
    if not src.exists() or not src.is_file():
        print(f"[_stage] skip {file_path!r}", flush=True)
        return None
    _COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if src.resolve().is_relative_to(_COMFY_INPUT_DIR.resolve()):
            return src.name
    except (ValueError, OSError):
        pass
    dst = _COMFY_INPUT_DIR / src.name
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        import shutil

        shutil.copy2(src, dst)
    return src.name


PRESET_DURATION = {"Fast": 60, "Balanced": 120, "Quality": 300}


def _seconds_to_frames(seconds: float, fps: int) -> int:
    return max(9, int(round(float(seconds) * float(fps) / 8) * 8) + 1)


async def _on_generate(mode_name: str, **inputs: Any):
    """Generate handler — async generator yielding (status_html, video_path)."""
    mode = modes.MODE_REGISTRY[mode_name]

    fps = int(inputs.get("fps", 24))
    seconds = float(inputs.get("seconds", 3))
    frames = _seconds_to_frames(seconds, fps)

    # Seed: respect the explicit value unless the "randomize" checkbox is on.
    seed = int(inputs.get("seed", 42))
    if inputs.get("randomize_seed"):
        seed = random.randint(0, 2**31 - 1)

    params: dict[str, Any] = {
        "prompt": inputs.get("prompt", ""),
        "negative_prompt": inputs.get("negative_prompt", ""),
        "preset": str(inputs.get("preset", "Balanced")).lower(),
        "width": int(inputs.get("width", 512)),
        "height": int(inputs.get("height", 768)),
        "frames": frames,
        "fps": fps,
        "seed": seed,
    }
    for k in (
        "image", "audio", "first_frame", "last_frame", "input_video",
        "camera_lora", "camera_strength", "detailer_on", "detailer_strength",
        "ic_lora", "ic_strength", "pose_on", "audio_cfg", "image_strength",
    ):
        if k in inputs:
            params[k] = inputs[k]

    for key in ("image", "audio", "first_frame", "last_frame", "input_video"):
        if key in params and params[key]:
            staged = _stage_to_comfy_input(params[key])
            if staged is None:
                params.pop(key, None)
            else:
                params[key] = staged

    patches = mode.parameterize_fn(params)
    workflow = wf_module.load_template(mode_name)
    for patch in patches:
        wf_module.set_input(workflow, *patch)

    backend = _get_backend()
    duration = PRESET_DURATION.get(str(inputs.get("preset", "Balanced")), 120)

    started = time.time()
    async for event in backend.submit(mode_name, workflow, gpu_duration=duration):
        elapsed = time.time() - started
        if isinstance(event, backend_module.DownloadEvent):
            status = ui.render_status(
                stage_index=0,
                stage_label=f"Downloading {event.filename}",
                step=int(event.mb_done),
                total_steps=int(max(event.mb_total, 1)),
                elapsed_s=elapsed,
                eta_s=0,
            )
            yield status, gr.update()
        elif isinstance(event, backend_module.ProgressEvent):
            # Each sampler in the workflow gets its own stage label "Diffusion (n)".
            # The static `mode.stage_map` describes the full pipeline (encode →
            # diffusion → upscale → diffusion → decode) but our progress hook
            # only fires inside samplers, so we label by sampler index instead.
            label = f"Diffusion (Stage {event.stage})"
            eta = (elapsed / max(event.step, 1)) * (event.total_steps - event.step)
            status = ui.render_status(
                stage_index=event.stage,
                stage_label=label,
                step=event.step,
                total_steps=event.total_steps,
                elapsed_s=elapsed,
                eta_s=eta,
            )
            yield status, gr.update()
        elif isinstance(event, backend_module.OutputEvent):
            video_update = event.video_path if event.video_path else gr.update()
            yield ui._render_idle(), video_update
        elif isinstance(event, backend_module.ErrorEvent):
            error_html = (
                f'<div class="status-card status-error">'
                f'  <div class="status-row"><span class="status-stage">Error · {event.category}</span></div>'
                f"  <div>{event.message}</div>"
                f"</div>"
            )
            yield error_html, gr.update()


def _input_keys_for_mode(mode_name: str, h: dict) -> list[str]:
    base = ["prompt", "preset", "width", "height", "seconds", "fps", "seed", "randomize_seed"]
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
    base = [
        h["prompt"], h["preset"], h["width"], h["height"],
        h["seconds"], h["fps"], h["seed"], h["randomize_seed"],
    ]
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
        kwargs = dict(zip(keys, values, strict=False))
        async for output in _on_generate(mode_name, **kwargs):
            yield output

    return handler


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
