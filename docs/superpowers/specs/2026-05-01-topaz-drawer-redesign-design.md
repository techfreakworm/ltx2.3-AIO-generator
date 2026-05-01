# Visual redesign — Topaz Cinema Slate + Drawer layout

**Date:** 2026-05-01
**Status:** Draft, awaiting user review
**Related:** `2026-04-30-ltx23-aio-generator-design.md` (original spec)

---

## Goal

Replace the current `gr.themes.Soft()` cream + purple palette with a dark slate-and-amber palette (**Topaz Cinema Slate**), and replace the always-visible left sidebar with a **hamburger drawer** that opens by default on desktop and is hidden by default on tablet/phone. Both changes are surface-level — no logic or backend changes.

## Why

The current palette reads as a hobby AI demo, not a creative-pro tool. Slate-on-slate is gentlest on the eye when judging color-graded video output, and an amber CTA reads "render," not "alert." The drawer pattern gives the form panel full screen real estate on phones (the sidebar currently stacks above the form on `<700px`, eating half the viewport for nav), while still keeping the sidebar always-visible at desktop widths where it costs nothing.

## Theme tokens

Applied via `gr.themes.Base().set(...)` overrides on the Blocks theme:

| Token | Value | Used for |
|---|---|---|
| `body_background_fill` | `#12161B` | App background |
| `background_fill_primary` | `#12161B` | Form/page background |
| `background_fill_secondary` | `#1A1F26` | Card / panel surface |
| `block_background_fill` | `#1A1F26` | Component (input, slider) surface |
| `body_text_color` | `#E6E8EB` | Primary text |
| `body_text_color_subdued` | `#7C8693` | Secondary / hint text |
| `border_color_primary` | `#262C35` | Card / input border |
| `border_color_accent` | `#E0A458` | Focused input ring |
| `button_primary_background_fill` | `#E0A458` | Generate button |
| `button_primary_text_color` | `#12161B` | Generate button label |
| `error_background_fill` | `#3A1E20` | Error banner background |
| `error_text_color` | `#F4A6A8` | Error banner text |

Fonts: `IBM Plex Sans` (UI 14 px) + `IBM Plex Mono` (mono 13 px), loaded from Google Fonts in the page `<head>` (Gradio's `head` parameter on `Blocks`, or via `_CUSTOM_CSS` `@import`).

## Layout: hamburger drawer

### Markup structure (logical, Gradio components)

```
gr.Row()                                     # header
├── HamburgerButton (gr.Button, ≡ icon)      # toggles drawer
├── gr.Markdown("LTX 2.3 Studio")            # title
└── ActiveModeTag (gr.Markdown, amber pill)  # shows current mode

gr.Row(elem_classes="layer")
├── gr.Column(elem_classes="drawer", visible=...) # 220 px wide
│   └── 6 mode buttons (existing)
└── gr.Column(elem_classes="body-pane")
    └── gr.Tabs(elem_classes="hidden-tabs")  # current 6 mode tabs
```

### Open / closed behavior

- **Desktop (≥1024 px):** drawer open by default, occupies the left 220 px of the viewport. Hamburger still works as a toggle but most users leave it open.
- **Tablet (700–1023 px):** drawer closed by default; opening it slides over content with a translucent overlay (`background: rgba(0,0,0,0.5)`). Tapping outside closes.
- **Phone (<700 px):** same as tablet, but drawer takes 80 % of viewport width when open.

State persists in `localStorage` (`ltx-drawer-open` key) so a user who closes the drawer on desktop stays closed across reloads.

### Active mode header tag

A small amber-bordered pill in the header (e.g., `T2V`, `A2V`, `LIPSYNC`) showing the currently selected mode. Updates whenever a mode button is clicked. Uses `IBM Plex Mono` 11 px so it reads as a label, not a button.

### CSS approach

Pure CSS, no JS framework. Use `:has()` and `<input type="checkbox">` hidden control for drawer toggle, OR a tiny inline `<script>` block that toggles a class on the body. Gradio doesn't sandbox custom scripts in `_CUSTOM_CSS`, but it does support the `head` parameter on `gr.Blocks` for inline `<script>`.

Existing media queries (`@max-width: 700px`, `@max-width: 1024px`) collapse to a single `@max-width: 1023px` block since drawer behavior only differs at the desktop boundary.

## Files touched

- `app.py` — Blocks `theme=`, `head=` (fonts + drawer toggle script), `_CUSTOM_CSS` rewrite, header markup, drawer column wrapping the existing mode buttons
- `README.md` — update screenshot if any (defer; we don't have one yet)

No changes to `backend.py`, `models.py`, `modes.py`, `workflow.py`, `ui.py`.

## Out of scope (do not touch)

- Form layout inside each mode tab (prompt input, parameter sliders) — typography updates only via theme token cascade
- Model status / settings panel content — these display the same info, just on the new palette
- Mode set, generate flow, progress events — backend unchanged
- Any CUDA / MPS / Spaces logic
- Custom LoRA UI (still v1.1+)

## Testing plan

1. `python app.py` locally on macOS, browse `http://127.0.0.1:7860`
2. Resize Chrome window: full width → 1024 px → 700 px → 380 px. Drawer should:
   - stay open ≥1024 px
   - hide & become overlay-on-hamburger <1024 px
3. Click each of 6 mode buttons; confirm:
   - active mode tag in header updates
   - drawer auto-closes on phone after click (open-on-tap → click-to-pick → close)
4. Click Generate on T2V (with Balanced preset, 320×480, 5 s). Confirm progress + output render correctly on the new palette.
5. Trigger an error (e.g., empty prompt) and confirm error banner uses `#3A1E20` / `#F4A6A8`.

## Risks

- Gradio's `head` parameter is on `gr.Blocks` since 4.x — confirm it accepts a multi-line string with `<script>`.
- `gr.themes.Base().set(...)` may not cover every component (e.g., `gr.Slider`'s track). If we hit a gap we add an `elem_classes` override and target it in `_CUSTOM_CSS` — incremental, low-risk.
- The hidden-checkbox-and-`:has()` toggle pattern has Safari ≥15.4 compatibility, fine for our audience.
