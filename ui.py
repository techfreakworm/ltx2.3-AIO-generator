# ui.py
"""Reusable Gradio components shared across modes."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr


def preset_bar(label: str = "Preset") -> gr.Radio:
    """Fast / Balanced / Quality radio. Use as a single component."""
    return gr.Radio(
        choices=["Fast", "Balanced", "Quality"],
        value="Balanced",
        label=label,
        container=True,
        info="Fast: distilled 8 steps · Balanced: two-stage 30+4 · Quality: HQ res_2s sampler",
    )


def status_banner() -> gr.HTML:
    """Status banner: stage chips + progress + memory."""
    return gr.HTML(
        value=_render_idle(),
        elem_classes=["status-banner"],
    )


def _render_idle() -> str:
    return (
        '<div class="status-card status-idle">'
        '<div class="status-row"><span class="status-dot"></span>'
        '<span class="status-label">Idle</span></div></div>'
    )


def render_status(
    stage_index: int,
    stage_label: str,
    step: int,
    total_steps: int,
    elapsed_s: float,
    eta_s: float,
    memory_text: str = "",
) -> str:
    """Render a status banner HTML string for the current event."""
    pct = 0 if total_steps <= 0 else int(100 * step / total_steps)
    return (
        f'<div class="status-card">'
        f'  <div class="status-row">'
        f'    <span class="status-stage">Stage {stage_index} · {stage_label}</span>'
        f'    <span class="status-meta">Step {step}/{total_steps} · '
        f"      {_fmt_secs(elapsed_s)} elapsed · ~{_fmt_secs(eta_s)} remaining</span>"
        f"  </div>"
        f'  <div class="status-bar"><div class="status-fill" style="width:{pct}%"></div></div>'
        f'  <div class="status-mem">{memory_text}</div>'
        f"</div>"
    )


def _fmt_secs(secs: float) -> str:
    secs = int(max(0, secs))
    if secs < 60:
        return f"{secs}s"
    return f"{secs // 60}m {secs % 60}s"


CAMERA_LORAS: list[str] = [
    "none",
    "static",
    "dolly-in",
    "dolly-out",
    "dolly-left",
    "dolly-right",
    "jib-up",
    "jib-down",
]

IC_LORAS_BY_MODE: dict[str, list[str]] = {
    "t2v": [],
    "a2v": [],
    "i2v": ["union", "pose-control"],
    "lipsync": ["pose-control"],
    "keyframe": ["union"],
    "style": ["motion-track", "union"],
}


@dataclass
class LoRAComponents:
    camera_lora: gr.Dropdown
    camera_strength: gr.Slider
    detailer_on: gr.Checkbox
    detailer_strength: gr.Slider
    ic_lora: gr.Dropdown | None
    ic_strength: gr.Slider | None
    pose_on: gr.Checkbox | None


def lora_chrome(mode: str) -> LoRAComponents:
    """Categorized LoRA controls for a given mode (camera + detailer + IC + pose).

    Only LoRAs relevant to the mode are surfaced. Distilled LoRA is auto-applied
    by the workflow when the Fast preset is chosen — not exposed here.
    """
    with gr.Group():
        gr.Markdown("**📷 Camera Movement**")
        camera_lora = gr.Dropdown(
            choices=CAMERA_LORAS,
            value="none",
            label="Camera",
            info="Mutually exclusive — pick one camera direction or none.",
        )
        camera_strength = gr.Slider(
            minimum=0.0,
            maximum=1.5,
            value=0.8,
            step=0.05,
            label="Camera strength",
            visible=True,
        )

    with gr.Group():
        gr.Markdown("**✨ Detailer**")
        detailer_on = gr.Checkbox(label="Apply IC-LoRA-Detailer", value=False)
        detailer_strength = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.5,
            step=0.05,
            label="Detailer strength",
        )

    ic_lora = ic_strength = pose_on = None
    ic_options = IC_LORAS_BY_MODE.get(mode, [])
    if ic_options:
        with gr.Group():
            gr.Markdown("**🎯 Image Conditioning**")
            ic_lora = gr.Dropdown(
                choices=["none"] + ic_options,
                value=ic_options[0] if ic_options else "none",
                label="IC-LoRA",
            )
            ic_strength = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.05,
                label="IC strength",
            )

    if mode in ("i2v", "lipsync"):
        with gr.Group():
            gr.Markdown("**🚶 Pose Control**")
            pose_on = gr.Checkbox(label="Apply IC-LoRA-Pose-Control", value=False)

    return LoRAComponents(
        camera_lora=camera_lora,
        camera_strength=camera_strength,
        detailer_on=detailer_on,
        detailer_strength=detailer_strength,
        ic_lora=ic_lora,
        ic_strength=ic_strength,
        pose_on=pose_on,
    )
