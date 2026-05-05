# Project Guidelines — ltx2.3-AIO-generator

Working notes for AI assistants and subagents implementing this project.

> Companion: see `SKILLS.md` for process rules — how to investigate, verify,
> commit, and ship changes here. This file is the *what* and *why*; SKILLS.md
> is the *how*.

---

## ⚠ Git authorship — sole author rule

**Mayank Gupta is the sole author on every commit in this repo.** No exceptions.

When committing:

- Do **NOT** append `Co-Authored-By: Claude ...` (or any other agent name).
- Do **NOT** add "Generated with Claude Code" / "🤖 Generated with..." footers.
- Do **NOT** pass `--author=...` — let git use the user's existing config.
- Do **NOT** include attribution in PR descriptions.

If asked to amend, re-commit, or rebase, strip any prior agent attribution from the commit message. Treat any tooling that suggests adding a Claude trailer as a bug to ignore.

---

## Project overview

Gradio app wrapping the existing ComfyUI LTX 2.3 All-In-One workflow into mode-specific UIs. Same code runs locally (Apple Silicon MPS / NVIDIA CUDA) and on Hugging Face Spaces (ZeroGPU, Pro tier).

**Spec:** `docs/superpowers/specs/2026-04-30-ltx23-aio-generator-design.md`
**Plan:** `docs/superpowers/plans/2026-04-30-ltx23-aio-generator.md`
**Future-improvements backlog:** `docs/future_improvements.md`

If you're a subagent picking up a task, the plan file is your assignment.

---

## Modes (six)

`t2v` text→video · `i2v` image→video · `a2v` audio→video · `lipsync` (image+audio) · `keyframe` (first+last frame→video) · `style` (preprocessor + IC-LoRA → restyle).

Each is a separate API-format JSON in `workflows/`. Per-mode parameter patches live in `modes.py` `parameterize_fn`.

---

## Architectural facts (locked — do not relitigate)

1. **Backend is ComfyUI in library mode.** We call `comfy.execution.PromptExecutor` directly with workflow JSONs we parameterize. We do NOT run ComfyUI as a subprocess.
2. **Six mode-specific workflow JSON files** in `workflows/` are user-exported "API format" from the master workflow. Do not hand-edit. Editor-format (with `nodes` array) does NOT work — `walk_workflow_for_models` and `PromptExecutor` both expect API format.
3. **Models live in HF cache.** Local: `~/.cache/huggingface/hub` symlinked into `comfyui/models/<comfy_type>/`. Spaces: same hub cache mirrored into `~/hf-cache-rw/` (see "Spaces deployment" below). Never commit `*.safetensors`, `*.gguf`, `*.bin`, `*.pt`. The `assets/seed_inputs/` exception in `.gitignore` covers the small placeholder files.
4. **One backend, one process.** The `@spaces.GPU` decorator is the only divergence between local and Spaces runtimes.
5. **VRAM is ComfyUI's job.** The only `empty_cache()` calls live in `backend.py`'s `try/finally`. Don't sprinkle them elsewhere.
6. **Bundled ComfyUI, never user's existing.** Local: git submodule. Spaces: runtime clone via `_git_clone()` in `app.py:_bootstrap()`.
7. **comfy_dir resolves per-platform.** `~/comfyui` on Spaces (writable HOME), `<repo>/comfyui` locally. Both `app.py` and `backend.py` have `_comfy_dir()`-style helpers that MUST stay in sync.
8. **Custom nodes are pinned to SHAs**, not branches. See `CUSTOM_NODES_PINNED` in `app.py`. `--branch <SHA>` doesn't work in `git clone`; we use init+fetch+checkout via `_git_clone()`.

---

## Spaces deployment specifics (where the gotchas live)

### Model loading: `preload_from_hub` + runtime cache mirror

HF Spaces' `preload_from_hub` directive in README YAML downloads listed files at build time into `~/.cache/huggingface/hub`. **Limitation: those files are owned by the build user** (root-ish). At runtime we run as uid 1000 and can't write there — any `hf_hub_download` for a non-preloaded file fails with `Permission denied (os error 13)`.

**Fix:** `_mirror_preload_hf_cache()` in `app.py` walks the read-only preload tree once at bootstrap and builds a parallel writable tree at `~/hf-cache-rw/`:
- `blobs/<sha>` files → **hardlinked** (zero-copy, shared inode, instant reads)
- `snapshots/<commit>/...` symlinks → **preserved** (relative paths resolve within the mirror)
- `refs/<branch>` → **byte-copied** (HF lib overwrites these on etag check; hardlinks would fail)
- All dirs → mkdir (we own them)
- Falls back to symlink if `os.link()` returns EXDEV (cross-device)

Then sets `HF_HOME=~/hf-cache-rw` and `HF_HUB_CACHE=~/hf-cache-rw/hub`. After this, preloaded reads are instant cache hits AND lazy downloads write to dirs we own.

The 10-entry cap on `preload_from_hub` is a hard HF limit. Total preload size cap is 150 GB (Spaces ephemeral storage). Current list is ~111 GB; see `docs/future_improvements.md` for what got dropped (84 GB of unused Lightricks transformers, 39 GB GGUF — both lazy-load when actually referenced).

### Per-call ZeroGPU duration: dynamic estimator + auto-retry

`@spaces.GPU(duration=N)` is a per-call timeout, not a billing cap. Shorter declared duration = faster queue priority on the shared pool. Setting a one-size-fits-all 600s caps everything in the slow lane.

**`_duration_for(executor, workflow, output_ids, mode, preset, multiplier=1.0)`** in `backend.py` estimates from:
- `_BASE_DURATION_S[mode]` — t2v 90s, lipsync 240s, style 360s, etc.
- `_PRESET_MULT[preset]` — fast 1×, balanced 1.5×, quality 3×
- `_frames_from_workflow(workflow)` — read from `EmptyLTXVLatentVideo` `length`
- +60s cold-cache buffer, +0.3s/frame VAE decode
- Clamped to `[60s, 900s]`

`@spaces.GPU(duration=_duration_for)` decorates `_execute_workflow` — ZeroGPU calls the estimator with the same args.

**Auto-retry on timeout** in `_on_generate` (app.py): if first attempt raises `gradio.exceptions.Error('GPU task aborted')`, classified as `category='gpu_timeout'`, the handler shows a "Retrying with extended GPU budget" banner and re-submits with `duration_multiplier=2.0`. The estimator clamps the retry at 900s anyway. One retry only.

### Returning the video path through ZeroGPU's subprocess boundary

`executor.history_result` was unreliable across the `@spaces.GPU` boundary — sometimes the parent process saw an empty dict even when the file was on disk. Fix: `_execute_workflow` reads `history_result["outputs"]` INSIDE the GPU context and returns the path string directly (picklable). Plus a filesystem fallback `_newest_recent_video()` that scans `comfyui/output/` for the newest mp4 modified in the last 60s.

### `allowed_paths` for video output

Gradio 5 refuses to expose files outside cwd / temp / `allowed_paths`. ComfyUI writes to `~/comfyui/output/...` which is outside our app's cwd `/home/user/app` on Spaces. `app.launch(..., allowed_paths=[str(_output_dir)])` whitelists the entire ComfyUI output tree. Without this, video generates fine but `gr.Video` shows blank.

### HF Spaces' header widget z-index (DOM-injected)

When a Space is loaded via the bare embed URL (`https://*.hf.space`), HF injects `#huggingface-space-header` at fixed `z-index: 20` in the top-right (the heart/share widget). Our header z-index has to coexist:
- Default: header `z-index: 15` (below HF widget — visible)
- Drawer open: `.drawer-elevated` class bumps to `z-index: 60` (above scrim 45 / drawer 50, hamburger × clickable as close)

JS toggles `.drawer-elevated` on `.aio-header` in lockstep with `.drawer-open` on `.aio-shell`. Three call sites: hamburger onclick, click-outside dismisser (in `gr.Blocks(head=...)` because `<script>` in `gr.HTML` gets stripped), mode-button auto-close.

### Custom nodes the workflow needs

Pinned in `CUSTOM_NODES_PINNED` (`app.py`):

```
Lightricks/ComfyUI-LTXVideo
kijai/ComfyUI-KJNodes
rgthree/rgthree-comfy
Kosinkadink/ComfyUI-VideoHelperSuite
pythongosssss/ComfyUI-Custom-Scripts
city96/ComfyUI-GGUF
Fannovel16/comfyui_controlnet_aux
evanspearman/ComfyMath
Smirnov75/ComfyUI-mxToolkit
DoctorDiffusion/ComfyUI-MediaMixer  (provides FinalFrameSelector)
```

Also `requirements.txt` includes deps the custom nodes need but their own `requirements.txt` files don't list (gguf, imageio_ffmpeg, opencv-python, matplotlib, diffusers, yt-dlp, psutil).

---

## UI design system: Topaz Cinema Slate

Dark slate background + amber accent, IBM Plex typography. Defined as `_TOPAZ_THEME = gr.themes.Base(...).set(...)` in `app.py`. Custom CSS in `_CUSTOM_CSS` for everything Gradio's theme machinery doesn't cover (drawer, header, mode buttons, status banner).

Layout: hamburger drawer. Pinned 220 px sidebar at ≥1024 px; below that, `position: fixed` overlay sliding from `left: -100%` to `left: 0` via `.aio-shell.drawer-open`.

Mode-tag in header (`#aio-mode-tag`) shows current mode (T2V/A2V/I2V/LIPSYNC/KEY/STYLE), updated by JS in mode-button click handlers.

Spec: `docs/superpowers/specs/2026-05-01-topaz-drawer-redesign-design.md`
Plan: `docs/superpowers/plans/2026-05-01-topaz-drawer-redesign.md`

---

## Critical Gradio scoping facts

- **Gradio prefixes user CSS** with `.gradio-container.gradio-container-<version> .contain ` — selectors that need to escape upward (`body:has(...)`, `html.foo .bar`) are rewritten to nonsense and silently break. Toggle classes via JS on elements INSIDE `.contain` (we use `.aio-shell` and `.aio-header`).
- **Gradio strips `<script>` tags inside `gr.HTML`** at sanitization. Inline scripts MUST go in `gr.Blocks(head=...)` to actually run. The `_HEAD_HTML` string in `app.py` is where the global click-outside dismisser lives.
- **Gradio's form labels have `z-index: 40`** built in. Anything we want above them (drawer, scrim) needs `z-index >= 41`. Our hierarchy: header (15 default → 60 elevated) > drawer (50) > scrim (45) > Gradio labels (40) > body.
- **`onclick="..."` attributes on plain HTML buttons DO survive** sanitization. Use them for tiny per-element interactions (hamburger toggle).

---

## Coding conventions

### Language and structure

- **Python 3.11.** No `match` statements (Spaces Python pin compatibility — Spaces base image is 3.10).
- **Flat layout.** No `src/`, no nested packages. Top-level `.py` files only, each with one clear responsibility.
- **No conda.** Always `python3.11 -m venv .venv`. System binaries via `brew`.

### Style

- **No emojis** in code or commit messages unless the user explicitly asks. UI text and stage labels in `modes.py` / `ui.py` are OK because they are user-facing — not code.
- **Comments only for non-obvious WHY.** Never narrate WHAT. Code with a good name doesn't need a comment.
- **Type hints on public functions.** Internal helpers can skip them if obvious.
- **Imports at top of file.** Inline imports only to break circular deps (e.g., `models.ensure_models_for_mode` imports `workflow` lazily — keep this, it's load-bearing).
- **Format with `ruff format`.** Lint with `ruff check`. Both must pass in CI.

### Commits

- **Conventional Commits style:** `<type>(<scope>): <subject>` — types: `feat`, `fix`, `chore`, `docs`, `test`, `refactor`, `ci`, `perf`.
- **Subject is imperative, lowercase, no trailing period.**
- **Body explains WHY when not obvious.** Reference spec/plan section if relevant.
- **Frequent small commits.** One logical change per commit.
- **No agent attribution** (see top of file).
- See `SKILLS.md` for the full process around when to commit vs hold.

### Testing

- **TDD per the plan.** Each implementation task has the failing test first.
- **No mocks for ComfyUI.** Tests run against real workflow JSONs. Stubs only for HTTP boundaries (HF Hub) and filesystem (use `tmp_path` and the `fake_hf_cache` fixture).
- **L1 + L3 in CI** (no GPU). L2 + L4 are local-developer-only.
- **Test naming:** `test_<unit>_<behavior_under_test>`.
- **`pytest --gpu`** enables L4 smoke tests. Default skips them.
- **`pytest --comfy-real`** uses bundled ComfyUI for L2 instead of the static stub validator.

---

## Editing the master workflow

When the user updates `~/Projects/comfyui/user/default/workflows/1. LTX 2.3 All-In-One 260406-05.json`:

```bash
python3.11 tools/extract_modes.py \
  --master ~/Projects/comfyui/user/default/workflows/"1. LTX 2.3 All-In-One 260406-05.json" \
  --out workflows
```

Then run the test suite — L2 graph-validation catches any node that became invalid in any mode.

After templates regenerate, the node-id constants in `modes.py` (e.g., `T2V_NODE_PROMPT = 240`) may need updating if ComfyUI re-numbered nodes. Procedure in plan Task 11 Step 4.

The user has explicitly said **don't change JSON** — when adding capabilities, prefer parameterize_fn patches over hand-edits. The user re-exports from ComfyUI editor when the workflow changes.

---

## Common pitfalls (read before opening a PR)

### ComfyUI / models

- **Loading models eagerly at import time.** Don't. `backend.py` constructs `PromptExecutor` once at instantiation; models load only when nodes execute.
- **Hard-coded `torch.cuda` calls.** Use `comfy.model_management.get_torch_device()` or guard with `if torch.cuda.is_available()`. Never assume CUDA.
- **Forgetting `.deepcopy` on workflow templates.** `workflow.load_template` already does this; if you bypass it for performance, you'll mutate the cached template.
- **Importing `comfy.*` before `sys.path.insert(0, comfy_dir)`.** Will `ModuleNotFoundError`. The order in `backend.py:__init__` is intentional.
- **`walk_workflow_for_models` returning empty.** Check that the workflow is API format (`{node_id: {class_type, inputs}}`), not editor format (`{nodes: [...]}`). The walker recurses into `Power Lora Loader` rows and skips ones with `on: false`.
- **Hardcoded paths in seed inputs.** The workflow's `LoadImage` / `VHS_LoadVideo` nodes have baked-in default filenames (`Screenshot 2026-04-23 023318.jpeg`, `4. Lipsync Music.mp3`, etc.). Our `assets/seed_inputs/` covers the ones that ship with the master, plus `_stage_to_comfy_input` copies user uploads into `comfyui/input/`. If a workflow update adds a new default filename, add a placeholder file.
- **`_COMFY_INPUT_DIR` and `_comfy_dir()` must agree.** Bug we hit: `app.py` had it hardcoded to `<repo>/comfyui/input` but on Spaces ComfyUI runs at `~/comfyui`. User uploads went to a directory ComfyUI never read. Both have to use the same on-Spaces vs local logic.

### Gradio / UI

- **Adding `<script>` to `gr.HTML`.** Gets stripped. Use `gr.Blocks(head=...)`.
- **Selectors that escape `.contain`.** Gradio rewrites them. Use a class on `.aio-shell` or `.aio-header` instead.
- **`gr.Video` paths outside cwd.** Need `allowed_paths=` on launch.
- **Z-index above HF's injected widget.** Header default z-index must be < 20 to not cover the heart/share widget. We use 15, bump to 60 only when drawer is open.

### Spaces

- **`/data` requires the persistent-storage add-on** (separate paid feature, not included in Pro). We use `~/comfyui` and `~/hf-cache-rw` instead.
- **Build user vs runtime user permissions.** preload_from_hub files are read-only for us. Mirror them — see "Spaces deployment specifics" above.
- **`@spaces.GPU` requires module-level decoration.** Runtime-applied decoration isn't detected by ZeroGPU's startup analyzer. Module-level static decorator + dynamic-duration callable is the supported pattern.
- **`history_result` may not survive ZeroGPU's subprocess boundary.** Compute outputs INSIDE the decorated function and return primitive types (str, int, dict of strs).
- **`allowed_paths` on `app.launch()`** must include the ComfyUI output dir or videos won't display.
- **Custom Dockerfile breaks ZeroGPU.** ZeroGPU is exclusively compatible with `sdk: gradio`. Switching to `sdk: docker` loses GPU access.

### Authoring

- **Adding `Co-Authored-By` because tooling suggests it.** See top of file. Strip it.
- **Don't push during HF testing.** When the user is running tests on the live Space, hold local commits until they say push. They'll explicitly tell you when to push.

---

## Out of scope for v1 (do not implement without asking)

These are documented as v1.1+ in spec § 11. Don't pre-build them just because they'd be easy:

- **Lite mode** (`LTX23_AIO_LITE=1`) for free HF Spaces tier
- **Custom LoRA** add/remove rows (Power-Lora-Loader clone)
- **GGUF Q4 transformer** / "Low VRAM" preset (the GGUF is loaded but always BF16-served at the moment)
- **Auto-launch of user's external ComfyUI** (`LTX23_AIO_COMFYUI_URL`)
- **Multi-prompt queueing**
- **Output history persistence** across sessions
- **Visual regression tests** for the Gradio UI
- **Property-based / fuzz testing** of workflow parameters
- **Persistent Storage add-on integration** (see future_improvements.md item 6)
- **Telemetry-driven duration estimator** (see future_improvements.md item, requires persistent storage)

If a task feels like it needs one of these, stop and ask the user.

---

## When in doubt

1. Read the spec and plan. 15 min of reading vs a day of wrong implementation.
2. Read `docs/future_improvements.md` to see if the change you're considering is already on a known list.
3. Check `git log --oneline` for similar changes — most non-obvious decisions have a fix-commit explaining the reasoning.
4. Ask the user before changing architectural shape.
