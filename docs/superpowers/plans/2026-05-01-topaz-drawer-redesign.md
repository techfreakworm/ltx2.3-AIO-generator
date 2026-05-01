# Topaz Cinema Slate + Drawer Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the Topaz Cinema Slate dark palette and the hamburger-drawer layout (open by default ≥1024 px, hidden behind ≡ button below) to the existing Gradio app, with no logic or backend changes.

**Architecture:** All edits land in `app.py`. Three concerns: (1) `gr.themes.Base().set(...)` overrides for the 12 Topaz tokens, (2) full rewrite of `_CUSTOM_CSS` for slate-on-slate styling + drawer mechanics + responsive breakpoint, (3) markup tweak — wrap the existing sidebar `gr.Column` in a `drawer` div with a header row above the shell that holds the ≡ toggle, title, and active-mode tag.

**Tech Stack:** Gradio 5.50, IBM Plex Sans/Mono via Google Fonts, pure-CSS drawer toggle using a hidden `<input type="checkbox">` + `:checked` sibling selectors (no JS framework — but a tiny inline `<script>` block in `head=` syncs the active-mode tag and persists drawer state to `localStorage`).

---

## File Structure

- **Modify:** `app.py`
  - `_CUSTOM_CSS` block — fully replaced
  - `build_app()` — Blocks `theme=` and `head=` parameters added; markup gets a header row + drawer column wrapper

No other files are touched. `backend.py`, `models.py`, `modes.py`, `workflow.py`, `ui.py` are unaffected.

---

### Task 1 — Add Topaz theme tokens to `gr.Blocks`

**Files:** Modify `app.py:191`

- [ ] **Step 1: Define the Topaz `gr.themes.Base()` instance**

Add this just above `def build_app()` near `app.py:189`:

```python
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
```

- [ ] **Step 2: Wire the theme into `build_app`**

Change `app.py:191` from:

```python
with gr.Blocks(theme=gr.themes.Soft(), title="LTX 2.3 All-in-One", css=_CUSTOM_CSS) as app:
```

to:

```python
with gr.Blocks(theme=_TOPAZ_THEME, title="LTX 2.3 Studio", css=_CUSTOM_CSS) as app:
```

- [ ] **Step 3: Smoke-test that imports still work**

Run: `python -c "import app; print('OK')"` in the project root.
Expected: `OK` printed, no traceback.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(ui): apply Topaz Cinema Slate theme tokens"
```

---

### Task 2 — Replace `_CUSTOM_CSS` with Topaz styles + drawer mechanics

**Files:** Modify `app.py:127-182` (the whole `_CUSTOM_CSS = """..."""` block)

- [ ] **Step 1: Replace the entire `_CUSTOM_CSS` block**

Replace `app.py:127-182` with:

```python
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
}
.aio-ham-toggle { display: none; }  /* hidden checkbox drives drawer state */
.aio-ham-label {
    width: 32px; height: 32px;
    border: 1px solid #262C35;
    border-radius: 5px;
    color: #7C8693;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
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

/* === Drawer === */
.aio-shell { position: relative; }
.aio-drawer {
    width: 220px;
    border-right: 1px solid #262C35;
    background: #12161B;
    padding: 14px 10px !important;
    flex-shrink: 0;
    transition: transform 0.2s ease, width 0.2s ease;
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

/* === Status banner === */
.status-card {
    padding: 12px 16px;
    border-radius: 6px;
    background: #1A1F26;
    border: 1px solid #262C35;
}
.status-row { display: flex; gap: 14px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; }
.status-stage { font-weight: 600; color: #E0A458; }
.status-meta { font-size: 12px; color: #7C8693; font-family: 'IBM Plex Mono', monospace; }
.status-bar { height: 4px; background: #262C35; border-radius: 99px; overflow: hidden; }
.status-fill { height: 100%; background: #E0A458; transition: width .3s; }
.status-mem { font-size: 11px; color: #7C8693; margin-top: 6px; font-family: 'IBM Plex Mono', monospace; }
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
        position: absolute;
        top: 0; left: 0; bottom: 0;
        z-index: 10;
        box-shadow: 4px 0 24px rgba(0,0,0,0.6);
        transform: translateX(-100%);
        max-width: 80vw;
    }
    /* checkbox at #aio-ham-toggle is the only sibling pattern Gradio
       lets us reach without JS — when checked, slide drawer in. */
    body:has(.aio-ham-toggle:checked) .aio-drawer { transform: translateX(0); }
    body:has(.aio-ham-toggle:checked) .aio-shell::before {
        content: ""; position: absolute; inset: 0;
        background: rgba(0,0,0,0.55); z-index: 9;
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
```

- [ ] **Step 2: Verify the CSS doesn't break the import**

Run: `python -c "import app; print(len(app._CUSTOM_CSS), 'chars CSS')"`
Expected: a number (around 4000), no traceback.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(ui): rewrite _CUSTOM_CSS for Topaz palette + drawer mechanics"
```

---

### Task 3 — Add header markup + drawer wrapper to `build_app`

**Files:** Modify `app.py:190-243` (the `build_app` function body)

- [ ] **Step 1: Replace the markup section**

Find this block in `app.py` (currently around 190-220):

```python
def build_app() -> gr.Blocks:
    with gr.Blocks(theme=_TOPAZ_THEME, title="LTX 2.3 Studio", css=_CUSTOM_CSS) as app:
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
```

Replace with:

```python
def build_app() -> gr.Blocks:
    with gr.Blocks(theme=_TOPAZ_THEME, title="LTX 2.3 Studio", css=_CUSTOM_CSS) as app:
        # Header: hamburger checkbox (drives drawer via :checked + :has() in CSS),
        # title, current-mode tag.
        gr.HTML(
            '<div class="aio-header">'
            '  <input type="checkbox" id="aio-ham-toggle" class="aio-ham-toggle">'
            '  <label for="aio-ham-toggle" class="aio-ham-label">≡</label>'
            '  <span class="aio-title">LTX 2.3 <span class="accent">Studio</span></span>'
            '  <span class="aio-mode-tag" id="aio-mode-tag">T2V</span>'
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

            # Body — unchanged, still hosts the 6 mode tabs.
            with gr.Column(scale=4, elem_classes=["aio-body"]):
                handles, tabs_component = _render_mode_panels()
```

- [ ] **Step 2: Smoke-test build_app produces a Blocks**

Run:
```bash
python -c "import app; b = app.build_app(); print(type(b).__name__)"
```
Expected: `Blocks`, no traceback.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(ui): drawer markup + header (hamburger / title / mode tag)"
```

---

### Task 4 — Wire active-mode tag updates from sidebar clicks

**Files:** Modify `app.py:232-237` (the existing mode-button click loop)

- [ ] **Step 1: Update the click handler to also push the new mode tag**

Current code at `app.py:232-237`:

```python
        for name, btn in mode_buttons.items():
            btn.click(
                fn=lambda mode_id=name: gr.Tabs(selected=mode_id),
                inputs=None,
                outputs=[tabs_component],
            )
```

Replace with:

```python
        # JS to update the header mode tag without a server round-trip.
        # Each mode button injects a tiny on-click that rewrites #aio-mode-tag.
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
                   f"/* also collapse drawer on mobile after pick */ "
                   f"if (window.matchMedia('(max-width: 1023px)').matches) {{ "
                   f"  const t = document.getElementById('aio-ham-toggle'); "
                   f"  if (t) t.checked = false; "
                   f"}} return []; }}",
            )
```

- [ ] **Step 2: Smoke-test**

Run:
```bash
python -c "import app; b = app.build_app(); print('ok')"
```
Expected: `ok`, no traceback.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(ui): mode tag updates + auto-close drawer on mobile select"
```

---

### Task 5 — Visual smoke test in browser

**Files:** None (test-only).

- [ ] **Step 1: Launch the app**

Run in one terminal:
```bash
cd /Users/techfreakworm/Projects/llm/ltx2.3-AIO-generator
source .venv/bin/activate
python app.py
```
Expected output: `Running on local URL: http://127.0.0.1:7860`

- [ ] **Step 2: Open browser at desktop width**

Open Chrome at `http://127.0.0.1:7860`. Resize window to 1280 px wide.

Verify:
- Header: ≡ button NOT visible (hidden on ≥1024 px), title "LTX 2.3 **Studio**" with "Studio" in amber, mode tag "T2V" in amber border on right
- Drawer (220 px) visible on left, "Modes" heading in IBM Plex Mono uppercase, 6 mode buttons stacked
- Active mode (T2V by default) has amber left border + amber text + slate-2 bg
- Body pane: form fields use slate background, amber Generate button at the bottom
- Click each mode button → mode tag in header updates, body switches to that mode's form

- [ ] **Step 3: Resize to tablet (1023 px)**

Drag Chrome to 900 px wide.

Verify:
- ≡ button NOW visible in header
- Drawer hidden (off-screen left)
- Click ≡ → drawer slides in, dark scrim covers body
- Click a mode button → drawer auto-closes, body switches mode
- Click ≡ again → drawer hides

- [ ] **Step 4: Resize to phone (380 px)**

Use Chrome devtools → device toolbar → iPhone 12.

Verify:
- Same as tablet, but drawer width capped at 80 vw
- Form fields are full-width (sliders, inputs)
- Generate button readable, no horizontal scrollbar

- [ ] **Step 5: Hit Generate (T2V, default settings)**

Type a short prompt, click Generate.

Verify:
- Status banner appears with `Stage 1 · Encode prompt` text in amber
- Progress bar fills with amber
- After ~30s on local MPS (or longer if no model cache), video appears in output
- Banner switches to `Done` or disappears

- [ ] **Step 6: Trigger an error**

Set width to 0 in slider (or click Generate with empty prompt) — anything that produces an error.

Verify:
- Error banner uses `#3A1E20` background + `#F4A6A8` text
- Stage label and meta text both readable

- [ ] **Step 7: Stop the dev server**

Ctrl+C in the terminal running `python app.py`.

- [ ] **Step 8: Commit screenshot/notes (optional)**

If anything didn't match the spec, file a follow-up; otherwise no commit needed.

---

### Task 6 — Push to GitHub + HF Space

**Files:** None — pushing only.

- [ ] **Step 1: Sync to both remotes**

```bash
cd /Users/techfreakworm/Projects/llm/ltx2.3-AIO-generator
git push origin master
HF_TOKEN=$(hf auth token 2>/dev/null) git push "https://techfreakworm:${HF_TOKEN}@huggingface.co/spaces/techfreakworm/LTX2.3-Studio" master:main
```

- [ ] **Step 2: Verify Space accepts the push**

Wait ~30 s, then:
```bash
HF_TOKEN=$(hf auth token 2>/dev/null) curl -s -H "Authorization: Bearer ${HF_TOKEN}" \
  "https://huggingface.co/api/spaces/techfreakworm/LTX2.3-Studio" \
  | python3 -c "import sys, json; d=json.load(sys.stdin); print('stage:', d['runtime']['stage'], 'sha:', d['sha'][:8])"
```
Expected: `stage: BUILDING` or `RUNNING_BUILDING`, sha matches local HEAD.

- [ ] **Step 3: Wait for build, then visual-spot-check on Spaces**

After ~5 min, open `https://techfreakworm-ltx2-3-studio.hf.space` in Chrome at 1280 px. Verify the Topaz palette + drawer rendered correctly. Resize to phone width and verify hamburger toggle.

---

## Out of scope reminder

These are explicitly NOT touched by this plan (per spec):
- Form layout inside each mode tab (typography updates flow via theme cascade only)
- Model status / settings panel content
- Mode set, generate flow, progress events
- Any CUDA / MPS / Spaces logic
- Custom LoRA UI

If any of those appear visually broken after this plan lands, file separately — they're spec-isolated and a different change.
