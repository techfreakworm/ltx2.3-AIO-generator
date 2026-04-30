# ui.py
"""Reusable Gradio components shared across modes."""
from __future__ import annotations

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
        f'      {_fmt_secs(elapsed_s)} elapsed · ~{_fmt_secs(eta_s)} remaining</span>'
        f'  </div>'
        f'  <div class="status-bar"><div class="status-fill" style="width:{pct}%"></div></div>'
        f'  <div class="status-mem">{memory_text}</div>'
        f'</div>'
    )


def _fmt_secs(secs: float) -> str:
    secs = int(max(0, secs))
    if secs < 60:
        return f"{secs}s"
    return f"{secs // 60}m {secs % 60}s"
