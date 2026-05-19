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
    ("https://github.com/city96/ComfyUI-GGUF.git", "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"),
    ("https://github.com/Fannovel16/comfyui_controlnet_aux.git", "e8b689a513c3e6b63edc44066560ca5919c0576e"),
    ("https://github.com/evanspearman/ComfyMath.git", "c01177221c31b8e5fbc062778fc8254aeb541638"),
    ("https://github.com/Smirnov75/ComfyUI-mxToolkit.git", "7f7a0e584f12078a1c589645d866ae96bad0cc35"),
    ("https://github.com/DoctorDiffusion/ComfyUI-MediaMixer.git", "2bae7b5ea8fc52d8a4d668d62fed76265f4eec2c"),
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


def _mirror_preload_hf_cache() -> None:
    """Mirror the build-populated HF cache into a writable runtime tree.

    HF Spaces' build pipeline runs `preload_from_hub` as a different user
    than the runtime container, so the populated `~/.cache/huggingface/`
    is read-only for us (uid 1000). Any subsequent `hf_hub_download` call
    that needs to write a NEW file (lazy-loaded LoRAs, GGUF, etc.) fails
    with "Permission denied" because the parent dir isn't writable.

    Fix: build a parallel tree at `~/hf-cache-rw/` that we own, with:
    - dirs: created fresh via mkdir
    - blob files (`blobs/<sha>`): hardlinked (shared inode, instant)
    - relative snapshot symlinks: preserved as symlinks
    - `refs/<branch>` files: byte-copied (HF lib overwrites these)
    - everything else: byte-copied (safest default)
    Then set HF_HOME / HF_HUB_CACHE so HF lib reads/writes through the
    mirror. Reads are zero-copy via hardlink/symlink; new downloads land
    in dirs we created.
    """
    import shutil

    src_root = pathlib.Path.home() / ".cache" / "huggingface"
    dst_root = pathlib.Path.home() / "hf-cache-rw"
    dst_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(dst_root)
    os.environ["HF_HUB_CACHE"] = str(dst_root / "hub")

    if not src_root.exists():
        return

    counts = {"dirs": 0, "hardlinks": 0, "symlinks": 0, "copies": 0, "errors": 0}

    def _treat_as_copy(rel_path: pathlib.PurePath) -> bool:
        # Anything under a refs/ dir, anywhere in the tree.
        return any(part == "refs" for part in rel_path.parts)

    def _walk(s: pathlib.Path, d: pathlib.Path) -> None:
        try:
            d.mkdir(parents=True, exist_ok=True)
            counts["dirs"] += 1
        except OSError as exc:
            print(f"[bootstrap] mirror mkdir fail {d}: {exc}", flush=True)
            counts["errors"] += 1
            return

        for entry in s.iterdir():
            de = d / entry.name
            try:
                if entry.is_symlink():
                    if de.exists() or de.is_symlink():
                        continue
                    target = os.readlink(str(entry))
                    de.symlink_to(target)
                    counts["symlinks"] += 1
                elif entry.is_dir():
                    _walk(entry, de)
                elif entry.is_file():
                    if de.exists():
                        continue
                    rel = de.relative_to(dst_root)
                    if _treat_as_copy(rel):
                        shutil.copy2(entry, de)
                        counts["copies"] += 1
                    else:
                        try:
                            os.link(str(entry), str(de))
                            counts["hardlinks"] += 1
                        except OSError:
                            # Cross-device or other — fall back to symlink.
                            de.symlink_to(entry)
                            counts["symlinks"] += 1
            except OSError as exc:
                print(f"[bootstrap] mirror skip {entry}: {exc}", flush=True)
                counts["errors"] += 1

    _walk(src_root, dst_root)
    print(
        f"[bootstrap] hf cache mirrored to {dst_root}: "
        f"{counts['dirs']} dirs, {counts['hardlinks']} hardlinks, "
        f"{counts['symlinks']} symlinks, {counts['copies']} copies, "
        f"{counts['errors']} errors",
        flush=True,
    )


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

    # Mirror the build-time HF cache (populated by preload_from_hub, owned by
    # build user → read-only for runtime user 1000) into a writable parallel
    # tree under $HOME, then point HF_HUB_CACHE / HF_HOME at it. After this:
    # - preloaded blobs are accessible via hardlink (no data copy, instant reads)
    # - relative snapshot symlinks resolve within the mirror
    # - refs/* are byte-copies so HF lib can overwrite when commits advance
    # - new lazy-downloaded files write to dirs we own → no permission errors
    if on_spaces:
        _mirror_preload_hf_cache()

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
# Styling: hide the default top tab strip (drawer nav drives selection),
# add status-card styling, plus single responsive breakpoint at 1023 px
# (drawer slides over body) / 1024 px+ (drawer pinned).
# ---------------------------------------------------------------------------

_CUSTOM_CSS = """
/* Hide Gradio's top tab strip — sidebar drives selection. */
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

/* === Header === */
.aio-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 11px 18px;
    border-bottom: 1px solid #262C35;
    background: #12161B;
    position: relative;
    /* HF injects #huggingface-space-header at fixed z-index 20 (top-right
       like/share widget). Stay below it by default so we don't cover it. */
    z-index: 15;
}
/* When drawer is open, lift header above scrim (z-45) and drawer (z-50) so
   the hamburger flips to × and remains clickable as a close affordance.
   Toggled in lockstep with .aio-shell.drawer-open via the inline JS below. */
.aio-header.drawer-elevated {
    z-index: 60;
}
.aio-ham-label {
    display: none;
    width: 32px; height: 32px;
    border: 1px solid #262C35;
    border-radius: 5px;
    color: #7C8693;
    cursor: pointer;
    align-items: center; justify-content: center;
    font-size: 18px; font-weight: 300;
    user-select: none;
}
.aio-ham-label:hover { color: #E0A458; border-color: #E0A458; }
.aio-title {
    font-size: 15px; font-weight: 600; letter-spacing: -0.01em;
    color: #E6E8EB;
}
.aio-title .accent { color: #E0A458; }
.aio-mode-tag {
    margin-left: auto;
    padding: 4px 9px;
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 11px; font-weight: 500; letter-spacing: 0.04em;
    color: #E0A458;
    border: 1px solid #E0A458;
    border-radius: 4px;
}

.aio-tipbar {
    margin: 0 0 6px 0;
    padding: 6px 14px;
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
    font-size: 12px;
    color: #B5BCC6;
    background: #1A1F26;
    border-bottom: 1px solid #262C35;
    text-align: center;
}
.aio-tipbar strong { color: #E6E8EB; font-weight: 500; }
.aio-tipbar .aio-heart { color: #E55B6E; }

.aio-mode-warning {
    margin: 4px 0 10px 0 !important;
    padding: 10px 14px !important;
    font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
    font-size: 12px !important;
    line-height: 1.55 !important;
    color: #D4C18B !important;
    background: rgba(224, 164, 88, 0.08) !important;
    border-left: 3px solid #E0A458 !important;
    border-radius: 4px !important;
}
.aio-mode-warning strong { color: #E0A458 !important; font-weight: 500 !important; }

.aio-hf-tip {
    margin: 12px 0 8px 0 !important;
    padding: 9px 14px !important;
    font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
    font-size: 11.5px !important;
    line-height: 1.5 !important;
    color: #9CA8B5 !important;
    background: rgba(124, 134, 147, 0.06) !important;
    border-left: 3px solid #5C6671 !important;
    border-radius: 4px !important;
}
.aio-hf-tip strong { color: #C8D0DA !important; font-weight: 500 !important; }

/* === Drawer === */
.aio-shell { position: relative; }
.aio-drawer {
    width: 220px;
    border-right: 1px solid #262C35;
    background: #12161B;
    padding: 14px 10px !important;
    flex-shrink: 0;
    transition: left 0.2s ease;
}
.aio-drawer-heading {
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.07em;
    color: #7C8693;
    padding: 6px 8px 4px !important;
    margin: 0 !important;
}

/* Mode buttons */
.aio-mode-btn { width: 100%; text-align: left; margin: 2px 0 !important; }
.aio-mode-btn-active {
    background: #1A1F26 !important;
    color: #E0A458 !important;
    border-left: 3px solid #E0A458 !important;
}

/* Model status / settings panels */
.aio-model-badge {
    padding: 9px 11px;
    border-radius: 6px;
    background: #1A1F26;
    border: 1px solid #262C35;
    font-size: 11.5px;
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    color: #7C8693;
}

/* Discord callout — drawer-bottom community button. Warm amber-on-slate
   to match the Topaz palette; the arrow nudges right on hover so it
   reads as actionable without screaming. */
.aio-discord-btn {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    margin: 4px 0;
    border-radius: 8px;
    background: linear-gradient(135deg, #1F2630 0%, #1A1F26 100%);
    border: 1px solid #2C3340;
    color: #E0A458 !important;
    font-size: 12.5px;
    font-weight: 500;
    text-decoration: none !important;
    transition: border-color 0.15s ease, background 0.15s ease, transform 0.15s ease;
}
.aio-discord-btn:hover {
    border-color: #E0A458;
    background: linear-gradient(135deg, #242C37 0%, #1F2630 100%);
}
.aio-discord-btn:hover .aio-discord-arrow { transform: translateX(3px); }
.aio-discord-glyph { font-size: 14px; line-height: 1; }
.aio-discord-arrow {
    margin-left: auto;
    color: #7C8693;
    transition: transform 0.15s ease, color 0.15s ease;
}
.aio-discord-btn:hover .aio-discord-arrow { color: #E0A458; }

/* === Status banner === */
.status-card {
    padding: 12px 16px;
    border-radius: 6px;
    background: #1A1F26;
    border: 1px solid #262C35;
}
.status-row { display: flex; gap: 14px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; }
.status-stage { font-weight: 600; color: #E0A458; }
.status-meta { font-size: 12px; color: #7C8693; font-family: 'IBM Plex Mono', ui-monospace, monospace; }
.status-bar { height: 4px; background: #262C35; border-radius: 99px; overflow: hidden; }
.status-fill { height: 100%; background: #E0A458; transition: width .3s; }
.status-mem { font-size: 11px; color: #7C8693; margin-top: 6px; font-family: 'IBM Plex Mono', ui-monospace, monospace; }
.status-error {
    background: #3A1E20 !important;
    border-color: #F4A6A8 !important;
    color: #F4A6A8 !important;
}
.status-error .status-stage { color: #F4A6A8; }

/* === Drawer toggle behavior at the desktop boundary === */
@media (max-width: 1023px) {
    .aio-ham-label { display: flex; }
    .aio-drawer {
        position: fixed;
        top: 0; bottom: 0;
        left: -100%;
        z-index: 50;
        box-shadow: 4px 0 24px rgba(0,0,0,0.6);
        max-width: 80vw;
        overflow-y: auto;
        overflow-x: hidden;
        padding-top: 80px !important;
    }
    /* `.aio-shell.drawer-open` is toggled by the hamburger's inline JS.
       `body:has(:checked)` would be cleaner but Gradio prefixes user CSS
       with `.gradio-container .contain `, breaking ancestor selectors. */
    .aio-shell.drawer-open .aio-drawer { left: 0; }
    .aio-shell.drawer-open::before {
        content: ""; position: fixed; inset: 0;
        background: rgba(0,0,0,0.92); z-index: 45;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    /* Mobile sub-tweaks */
    .aio-mode-btn { font-size: 13px !important; padding: 7px 10px !important; }
    .aio-body [class*="row"] { flex-wrap: wrap !important; }
    .aio-body [class*="row"] > div { flex: 1 1 100% !important; min-width: 0 !important; }
}

@media (min-width: 1024px) {
    .aio-ham-label { display: none; }
}
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


_TOPAZ_THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#FBE5C7", c100="#F5D29C", c200="#EFC174", c300="#E9B05A",
        c400="#E5A75B", c500="#E0A458", c600="#C68D3F", c700="#A6722E",
        c800="#7E5722", c900="#583C18", c950="#3A2810",
    ),
    neutral_hue=gr.themes.Color(
        c50="#E6E8EB", c100="#C9CDD3", c200="#ACB1B9", c300="#9097A0",
        c400="#7C8693", c500="#626972", c600="#4A4F58", c700="#363B43",
        c800="#262C35", c900="#1A1F26", c950="#12161B",
    ),
    font=(gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"),
    font_mono=(gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"),
).set(
    body_background_fill="#12161B",
    background_fill_primary="#12161B",
    background_fill_secondary="#1A1F26",
    block_background_fill="#1A1F26",
    block_label_background_fill="transparent",
    body_text_color="#E6E8EB",
    body_text_color_subdued="#7C8693",
    border_color_primary="#262C35",
    border_color_accent="#E0A458",
    button_primary_background_fill="#E0A458",
    button_primary_background_fill_hover="#F0B870",
    button_primary_text_color="#12161B",
    button_secondary_background_fill="#1A1F26",
    button_secondary_background_fill_hover="#232930",
    button_secondary_text_color="#E6E8EB",
    button_secondary_border_color="#262C35",
    input_background_fill="#12161B",
    input_border_color="#262C35",
    input_border_color_focus="#E0A458",
    error_background_fill="#3A1E20",
    error_text_color="#F4A6A8",
    slider_color="#E0A458",
)


_HEAD_HTML = """
<script>
(function(){
  if (window._aioDismissInstalled) return;
  window._aioDismissInstalled = true;
  document.addEventListener("click", function(e) {
    var s = document.querySelector(".aio-shell");
    if (!s || !s.classList.contains("drawer-open")) return;
    if (e.target.closest(".aio-drawer") || e.target.closest(".aio-ham-label")) return;
    s.classList.remove("drawer-open");
    var h = document.querySelector(".aio-header");
    if (h) h.classList.remove("drawer-elevated");
    var b = document.querySelector(".aio-ham-label");
    if (b) {
      b.textContent = "\\u2261";
      b.setAttribute("aria-expanded", "false");
    }
  });
})();
</script>
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(theme=_TOPAZ_THEME, title="LTX 2.3 Studio", css=_CUSTOM_CSS, head=_HEAD_HTML) as app:
        # Header: hamburger button toggles `.drawer-open` on `.aio-shell`.
        # The click-outside dismisser is registered via gr.Blocks(head=...)
        # below — Gradio strips <script> tags inside gr.HTML so it has to
        # live in <head> to actually run.
        gr.HTML(
            '<div class="aio-header">'
            '  <button type="button" class="aio-ham-label" '
            '          onclick="(function(b){var s=document.querySelector(\'.aio-shell\');'
            'var o=s.classList.toggle(\'drawer-open\');'
            'var h=document.querySelector(\'.aio-header\');'
            'if(h)h.classList.toggle(\'drawer-elevated\',o);'
            'b.textContent=o?\'\\u00d7\':\'\\u2261\';'
            'b.setAttribute(\'aria-expanded\',o?\'true\':\'false\');})(this)" '
            '          aria-expanded="false" aria-label="Toggle navigation">≡</button>'
            '  <span class="aio-title">LTX 2.3 <span class="accent">Studio</span></span>'
            '  <span class="aio-mode-tag" id="aio-mode-tag">T2V</span>'
            '</div>'
        )
        gr.HTML(
            '<div class="aio-tipbar">'
            'Built with care. '
            '<strong>Drop a <span class="aio-heart">♥</span> at the top</strong> to support it '
            '· '
            'Follow <a href="https://huggingface.co/techfreakworm" target="_blank" rel="noopener noreferrer">@techfreakworm</a> '
            'for what\'s next '
            '· '
            '<a href="https://discord.gg/qbn3exeEXa" target="_blank" rel="noopener noreferrer">Chat with the maker on Discord</a>'
            '</div>'
        )

        with gr.Row(elem_classes=["aio-shell"]):
            # Drawer (drawer behaves as fixed sidebar ≥1024 px;
            # absolute-positioned overlay <1024 px — see _CUSTOM_CSS).
            with gr.Column(scale=1, min_width=200, elem_classes=["aio-drawer"]):
                gr.Markdown("Modes", elem_classes=["aio-drawer-heading"])
                mode_buttons = {
                    name: gr.Button(
                        f"{m.icon}  {m.label}",
                        elem_classes=["aio-mode-btn"],
                        variant="secondary",
                    )
                    for name, m in modes.MODE_REGISTRY.items()
                }
                # ZeroGPU per-call cap, placed right under the mode list so
                # it's visible without scrolling. The pre-flight gate in
                # _on_generate refuses calls whose estimate exceeds this.
                gpu_budget_slider = gr.Slider(
                    minimum=60,
                    maximum=1800,
                    value=240,
                    step=30,
                    label="GPU budget (seconds)",
                    info="Max GPU time per generation. Higher = heavy modes fit; uses more of your daily quota per call.",
                    elem_classes=["aio-gpu-budget"],
                )
                gr.Markdown("Models", elem_classes=["aio-drawer-heading"])
                model_status = gr.HTML(_render_model_status_idle(), elem_id="aio-model-status")
                refresh_btn = gr.Button("Refresh", size="sm", variant="secondary")
                unload_btn = gr.Button("Unload all models", size="sm", variant="secondary")
                gr.Markdown("Settings", elem_classes=["aio-drawer-heading"])
                gr.Markdown(
                    "Output: `comfyui/output/LTX2.3/`<br>"
                    "Set `LTX23_AIO_VRAM=lowvram|normalvram|highvram` to override "
                    "the auto-detected VRAM tier.",
                    elem_classes=["aio-model-badge"],
                )
                gr.Markdown("Community", elem_classes=["aio-drawer-heading"])
                gr.HTML(
                    '<a class="aio-discord-btn" href="https://discord.gg/qbn3exeEXa" '
                    'target="_blank" rel="noopener noreferrer">'
                    '<span class="aio-discord-glyph">✨</span>'
                    '<span>Chat with the maker on Discord</span>'
                    '<span class="aio-discord-arrow">→</span>'
                    '</a>'
                )

            # Body — unchanged, still hosts the 6 mode tabs.
            with gr.Column(scale=4, elem_classes=["aio-body"]):
                handles, tabs_component = _render_mode_panels()

        # Wire generate buttons. The GPU-budget slider lives in the drawer and
        # is the same instance for every mode — append it last so the handler
        # receives it as `gpu_budget` (see `_input_keys_for_mode`).
        for name, h in handles.items():
            inputs = _collect_inputs_for_mode(name, h) + [gpu_budget_slider]
            h["generate_btn"].click(
                fn=_make_handler(name, h),
                inputs=inputs,
                outputs=[h["status"], h["video_out"]],
            )

        # JS to update the header mode tag without a server round-trip.
        # Each mode button injects a tiny on-click that rewrites #aio-mode-tag
        # and (on mobile) auto-collapses the drawer.
        _MODE_TAG_BY_NAME = {
            "t2v": "T2V", "a2v": "A2V", "i2v": "I2V",
            "lipsync": "LIPSYNC", "keyframe": "KEY", "style": "STYLE",
        }
        for name, btn in mode_buttons.items():
            tag = _MODE_TAG_BY_NAME.get(name, name.upper())
            btn.click(
                fn=lambda mode_id=name: gr.Tabs(selected=mode_id),
                inputs=None,
                outputs=[tabs_component],
                js=f"() => {{ "
                   f"const el = document.getElementById('aio-mode-tag'); "
                   f"if (el) el.textContent = {tag!r}; "
                   f"if (window.matchMedia('(max-width: 1023px)').matches) {{ "
                   f"  document.querySelector('.aio-shell')?.classList.remove('drawer-open'); "
                   f"  document.querySelector('.aio-header')?.classList.remove('drawer-elevated'); "
                   f"  const hb = document.querySelector('.aio-ham-label'); "
                   f"  if (hb) {{ hb.textContent = '\\u2261'; hb.setAttribute('aria-expanded', 'false'); }} "
                   f"}} return []; }}",
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
                gr.Markdown(
                    "**Heads up — Style Transfer is the heaviest mode.** "
                    "It runs the source video through pose detection AND adds "
                    "every frame as conditioning, so even the Fast preset can "
                    "blow the per-call GPU budget on free/anonymous tier. "
                    "**A failed run still consumes daily quota.** "
                    "For reliable runs: HF Pro account, resolution ≤ 1024×576, "
                    "source video ≤ 8 s.",
                    elem_classes=["aio-mode-warning"],
                )
                handles["image"] = gr.Image(label="Style reference", type="filepath")
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

            gr.Markdown(
                "**Tip for HF Spaces users:** Heavier configurations "
                "(Cinematic preset, high resolution, long videos) target local "
                "hardware and may abort mid-run on Spaces — burning quota with "
                "no output. Stay at Fast/Balanced + ≤ 1024×576 + ≤ 6 s output "
                "for safe Spaces runs.",
                elem_classes=["aio-hf-tip"],
            )

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


# Must match the comfy_dir used in _bootstrap() — on Spaces this is
# ~/comfyui (mirroring backend.py's _comfy_dir), otherwise repo-local.
_COMFY_INPUT_DIR = (
    (pathlib.Path.home() / "comfyui" / "input")
    if _on_spaces()
    else pathlib.Path(__file__).parent / "comfyui" / "input"
)


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


_FRIENDLY_ERRORS: dict[str, tuple[str, str]] = {
    "gpu_timeout": (
        "Hit the GPU time limit",
        "This run took longer than the GPU budget. Raise the GPU-budget "
        "slider (in the sidebar), or try the Fast preset / a shorter video, "
        "then click Generate again.",
    ),
    "expired_token": (
        "Session timed out",
        "Your sign-in session expired. Refresh the page and try again — "
        "you'll keep your spot in the GPU queue.",
    ),
    "illegal_duration": (
        "GPU budget too high for your account",
        "HF rejected the requested duration as exceeding your account's "
        "per-call cap. Lower the GPU-budget slider (sidebar) and try again, "
        "or drop the preset / shorten the video.",
    ),
    "unlogged": (
        "Sign-in not detected",
        "Make sure you're signed into huggingface.co (top-right avatar), "
        "then refresh this page. Pro accounts get 25 min of GPU per day.",
    ),
    "quota_exceeded": (
        "Daily GPU quota used up",
        "You've used today's GPU minutes. Wait for the rolling 24-hour "
        "reset, or upgrade Pro at huggingface.co/subscribe/pro for more.",
    ),
    "oom": (
        "Ran out of GPU memory",
        "Try a smaller resolution, fewer frames, or the Fast preset.",
    ),
    "interrupt": (
        "Cancelled",
        "Generation was cancelled. Click Generate to start a fresh run.",
    ),
    "download": (
        "Model download failed",
        "Couldn't fetch a required model file. Check your internet and try again.",
    ),
}


def _friendly_error(category: str, raw_message: str) -> tuple[str, str]:
    """Translate a backend error category into (title, body) the user can act on."""
    if category in _FRIENDLY_ERRORS:
        return _FRIENDLY_ERRORS[category]
    return (
        "Generation failed",
        "Something went wrong. Click Generate to retry, or check the Space "
        "logs if it keeps happening.",
    )


def _seconds_to_frames(seconds: float, fps: int) -> int:
    return max(9, int(round(float(seconds) * float(fps) / 8) * 8) + 1)


def _prune_old_outputs(output_dir: pathlib.Path, max_age_seconds: int = 4 * 3600) -> int:
    """Delete files under *output_dir* older than *max_age_seconds*; return count.

    HF Spaces ephemeral disk is 150 GB and preload already eats ~111 GB. Without
    this sweep, generations accumulate in `~/comfyui/output/` until the disk
    fills and the replica goes unhealthy (observed: stuck `RUNNING` with
    `replicas.current=0`). Per-file OSError is swallowed so one bad file
    doesn't abort a sweep that would otherwise free space.
    """
    if not output_dir.exists():
        return 0
    cutoff = time.time() - max_age_seconds
    deleted = 0
    for f in output_dir.rglob("*"):
        try:
            if not f.is_file():
                continue
            if f.stat().st_mtime < cutoff:
                f.unlink()
                deleted += 1
        except OSError:
            continue
    return deleted


async def _on_generate(mode_name: str, *, progress: Any = None, **inputs: Any):
    """Generate handler — async generator yielding (status_html, video_path).

    `progress` is a `gr.Progress` instance injected by Gradio. It's the only
    progress channel that survives the @spaces.GPU subprocess boundary on HF
    Spaces; we forward it to the backend so ComfyUI's per-step counter renders
    a real progress bar instead of a generic Gradio spinner.
    """
    _comfy_dir_now = (
        (pathlib.Path.home() / "comfyui")
        if _on_spaces()
        else pathlib.Path(__file__).parent / "comfyui"
    )
    _prune_old_outputs(_comfy_dir_now / "output")

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
    preset = params["preset"]  # already lowercased above

    # Pre-flight gate: refuse to submit if the estimator says this config
    # needs more GPU time than the user has allocated. ZeroGPU charges actual
    # usage, not declared duration, so under-allocating means the call still
    # burns quota before timing out. Refuse here and tell the user to either
    # bump the GPU-budget slider or reduce frames/preset.
    user_budget: int | None = None
    if "gpu_budget" in inputs and inputs["gpu_budget"] is not None:
        user_budget = int(inputs["gpu_budget"])
        estimate = backend_module._estimate_duration_unclamped(
            mode=mode_name, preset=preset, frames=frames,
        )
        if estimate > user_budget:
            yield (
                f'<div class="status-card status-error">'
                f'  <div class="status-row"><span class="status-stage">GPU budget too low</span></div>'
                f"  <div>This config estimates ~{estimate}s of GPU time, but the "
                f"GPU-budget slider is set to {user_budget}s. Raise the slider, drop "
                f"the preset to Fast or Balanced, or reduce the duration / frame count.</div>"
                f"</div>",
                gr.update(),
            )
            return

    async def _translate(event, started_at):
        """Translate one backend event into Gradio (status_html, video) yields.

        Returns the tuple to yield, plus a flag indicating terminal state.
        """
        elapsed = time.time() - started_at
        if isinstance(event, backend_module.DownloadEvent):
            return (
                ui.render_status(
                    stage_index=0,
                    stage_label=f"Downloading {event.filename}",
                    step=int(event.mb_done),
                    total_steps=int(max(event.mb_total, 1)),
                    elapsed_s=elapsed,
                    eta_s=0,
                ),
                gr.update(),
            )
        if isinstance(event, backend_module.ProgressEvent):
            label = f"Diffusion (Stage {event.stage})"
            eta = (elapsed / max(event.step, 1)) * (event.total_steps - event.step)
            return (
                ui.render_status(
                    stage_index=event.stage,
                    stage_label=label,
                    step=event.step,
                    total_steps=event.total_steps,
                    elapsed_s=elapsed,
                    eta_s=eta,
                ),
                gr.update(),
            )
        if isinstance(event, backend_module.OutputEvent):
            video_update = event.video_path if event.video_path else gr.update()
            return (ui._render_idle(), video_update)
        if isinstance(event, backend_module.ErrorEvent):
            title, body = _friendly_error(event.category, event.message)
            return (
                f'<div class="status-card status-error">'
                f'  <div class="status-row"><span class="status-stage">{title}</span></div>'
                f"  <div>{body}</div>"
                f"</div>",
                gr.update(),
            )
        return None

    # Single attempt. ZeroGPU-side abort (duration cap) and 401 expired-token
    # surface as friendly messages via _friendly_error; user clicks Generate
    # again to retry with a fresh request and fresh X-IP-Token.
    started = time.time()
    async for event in backend.submit(
        mode_name, workflow,
        preset=preset, user_budget=user_budget,
        progress=progress,
    ):
        translated = await _translate(event, started)
        if translated is not None:
            yield translated


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
        base.extend(["image", "input_video"])
    base.append("negative_prompt")
    base.extend(["camera_lora", "camera_strength", "detailer_on", "detailer_strength"])
    if h["lora"].ic_lora is not None:
        base.extend(["ic_lora", "ic_strength"])
    if h["lora"].pose_on is not None:
        base.append("pose_on")
    base.append("gpu_budget")  # appended by build_app() from the global slider
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
        base.extend([h["image"], h["input_video"]])
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

    async def handler(*values, progress=gr.Progress()):
        kwargs = dict(zip(keys, values, strict=False))
        async for output in _on_generate(mode_name, progress=progress, **kwargs):
            yield output

    return handler


if __name__ == "__main__":
    # Gradio 5's file-access policy refuses to serve files outside cwd /
    # tempdir / allowed_paths. ComfyUI writes generated videos to
    # `<comfy_dir>/output/...` which is outside our cwd on Spaces, so
    # whitelist that directory tree explicitly.
    _on_spaces_at_launch = bool(os.environ.get("SPACES_ZERO_GPU"))
    _comfy_dir_at_launch = (
        (pathlib.Path.home() / "comfyui") if _on_spaces_at_launch
        else pathlib.Path(__file__).parent / "comfyui"
    )
    _output_dir = _comfy_dir_at_launch / "output"
    _output_dir.mkdir(parents=True, exist_ok=True)

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[str(_output_dir)],
    )
