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
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="LTX 2.3 All-in-One",
        css=_CUSTOM_CSS,
    ) as app:
        gr.Markdown("# ⚡ LTX 2.3 All-in-One")
        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                _render_sidebar()
            with gr.Column(scale=4):
                _render_mode_panels()
    return app


def _render_sidebar() -> None:
    gr.Markdown("### Modes")
    for name, mode in modes.MODE_REGISTRY.items():
        gr.Markdown(f"- {mode.icon} {mode.label}")
    gr.Markdown("---\n### Models")
    gr.Button("Unload all models", variant="secondary")


def _render_mode_panels() -> None:
    with gr.Tabs():
        for name, mode in modes.MODE_REGISTRY.items():
            with gr.Tab(label=f"{mode.icon} {mode.label}"):
                gr.Markdown(f"## {mode.label}")
                gr.Markdown(f"_(Mode `{name}` form goes here — built in Task 22.)_")


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
