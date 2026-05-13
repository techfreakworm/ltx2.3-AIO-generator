# AGENTS.md

Tool-agnostic agent guidance for the `ltx2.3-AIO-generator` repo. If you're driving Claude Code, Cursor, Aider, Codex, or anything else with file-edit + shell access, **start here**.

This file is the authoritative project rulebook.

- `CLAUDE.md` — Claude-specific extensions and the full gotchas catalogue (what & why).
- `SKILLS.md` — process / how-to (debugging, deployment, when to commit, useful one-liners).
- `README.md` — public-facing project intro (different audience).

---

## TL;DR — the seven rules

1. **Mayank Gupta is sole author on every commit.** No agent co-author trailers. No "generated with…" footers. No `--author=` flag. Strip any tool-suggested attribution.
2. **Backend is ComfyUI in library mode.** We call `comfy.execution.PromptExecutor` directly with our parameterized workflow JSONs. We do NOT subprocess ComfyUI and we do NOT swap to a different inference engine.
3. **Six mode workflows live in `workflows/` as user-exported API-format JSON.** Do not hand-edit. The user re-exports from the ComfyUI editor when the master changes; `python tools/extract_modes.py --master ... --out workflows` regenerates the six mode files.
4. **Models live in the HF cache.** Local: `~/.cache/huggingface/hub` symlinked into `comfyui/models/<comfy_type>/`. Spaces: build-time `preload_from_hub` + runtime mirror to `~/hf-cache-rw/`. Never commit `*.safetensors`, `*.gguf`, `*.bin`, `*.pt`.
5. **Don't pin `spaces` in `requirements.txt`.** HF Spaces' ZeroGPU build injects its own version; pinning causes pip-resolve failure.
6. **HF Space deploys from `main`. Local default branch is `master`.** Push with `git push space master:main` — bare `git push space master` creates an orphan remote branch that does NOT trigger a deploy.
7. **VRAM is ComfyUI's job.** The only `empty_cache()` calls live in `backend.py`'s try/finally. Don't sprinkle them elsewhere.

If you can't satisfy any of these without changing architectural shape, **ask the user before proceeding.**

---

## Project shape

Single-process Gradio 5.50 app, flat top-level Python layout, ~3.5 k LOC excluding the ComfyUI submodule. ComfyUI itself is vendored as a git submodule locally and runtime-cloned into `~/comfyui` on HF Spaces.

```
app.py            Gradio Blocks entry · _bootstrap · _on_generate · header drawer
backend.py        ComfyUILibraryBackend · @spaces.GPU · _execute_workflow · duration estimator
modes.py          MODE_REGISTRY + per-mode parameterize_fn + node-id constants
models.py         MODEL_REGISTRY · walk_workflow_for_models · ensure_models_for_mode
ui.py             render_status · _render_idle · mode-form layout primitives
workflow.py       load_template · set_input helpers
workflows/        Six API-format mode JSONs — DO NOT hand-edit
assets/           Seed input placeholders for cold-start staging
tools/            extract_modes.py — regenerate workflows/ from a master export
docs/superpowers/ Spec + plan + brainstorm artifacts (per feature)
tests/            L1 + L2 + L3; GPU smoke gated by --gpu
comfyui/          Submodule (local) / runtime clone target (Spaces)
```

Same code path everywhere. The only branching is in `_bootstrap()` (cache-mirror dance on Spaces; plain symlink locally) and the `@spaces.GPU` decorator (identity off Spaces).

---

## Locked architecture decisions

These came out of 100+ commits of iteration. Do not relitigate.

| Decision | Why | Code reference |
|---|---|---|
| ComfyUI library mode (no subprocess) | Direct executor access; shared Python process for model lifecycle and progress reporting | `backend.ComfyUILibraryBackend.__init__` |
| Six API-format workflow JSONs | API format (`{node_id: {class_type, inputs}}`) is what `PromptExecutor` + `walk_workflow_for_models` consume. Editor format silently fails. | `workflows/*.json` |
| Workflow parameterization via patches | The user owns workflow shape via ComfyUI editor exports; we only patch leaf inputs. Never hand-edit JSON. | `modes.parameterize_fn` |
| Custom nodes pinned to SHAs (not branches) | Reproducible builds; `git clone --branch <SHA>` is unsupported — we use a `_git_clone()` init+fetch+checkout helper | `app.CUSTOM_NODES_PINNED`, `app._git_clone` |
| HF cache → `comfyui/models/<type>/` symlinks | Avoids duplicate weight copies; HF cache stays the single source of truth | `models.symlink_hf_cache_to_comfy_layout` |
| `_mirror_preload_hf_cache()` on Spaces | preload_from_hub files are owned by the build user (root-ish); the runtime user (uid 1000) can't write to them. Hardlink blobs + copy refs into a writable mirror under `~/hf-cache-rw/`. | `app._mirror_preload_hf_cache` |
| `_comfy_dir()` per-platform | `~/comfyui` on Spaces (writable HOME); `<repo>/comfyui` locally. Both `app.py` and `backend.py` must agree. | `app._comfy_dir`, `backend._comfy_dir` |
| `@spaces.GPU(duration=callable)` applied module-level | Runtime decoration isn't detected by ZeroGPU's startup analyzer. Static decorator + dynamic-duration callable is the supported pattern. | `backend._execute_workflow` |
| Per-call duration estimator | A one-size-fits-all 600 s caps everything in the slow queue lane. Estimator reads frames + mode + preset + cold-cache buffer, clamps `[60, 900] s`. | `backend._duration_for` |
| Auto-retry once at 2× on timeout | `"GPU task aborted"` is the queue-eviction signal; one retry catches transient busy queues. | `app._on_generate` |
| `allowed_paths=[output_dir]` on launch | Gradio 5 refuses files outside cwd / temp / `allowed_paths`. ComfyUI writes to `~/comfyui/output/...` on Spaces — outside the app cwd. | `app.app.launch(...)` |
| Header `z-index: 15` default / `60` elevated | HF injects `#huggingface-space-header` at fixed z-index 20. Default keeps our header below it (HF widget visible); drawer-open bumps above the scrim. | `_CUSTOM_CSS` `.aio-header` |
| Click-outside dismisser in `gr.Blocks(head=…)` | Gradio strips `<script>` tags inside `gr.HTML`. Inline scripts have to live in `<head>` to actually run. | `app._HEAD_HTML` |
| Mode tag in header via inline `onclick` | `onclick="…"` attributes on plain HTML buttons survive sanitization (unlike inline `<script>`). Lets us update the tag without a server round-trip. | mode buttons in `build_app` |
| Topaz Cinema Slate theme | Locked from brainstorming round. Slate `#1A1F26` + amber accent `#E0A458` + IBM Plex Sans. | `app._TOPAZ_THEME`, `_CUSTOM_CSS` |

---

## Commit rules

- **Conventional Commits:** `<type>(<scope>): <subject>`
  - types: `feat`, `fix`, `chore`, `docs`, `test`, `refactor`, `ci`, `perf`
- Subject is **imperative**, lowercase, **no trailing period**.
- Body explains **why** when not obvious. Reference plan task IDs when implementing a specific plan step.
- Frequent small commits; one logical change per commit.
- **No agent attribution** in commit message or body. See rule 1.
- Don't `git push --force` to `master` / `main` unless the user explicitly says so.

---

## Verification rules

- **Tests must pass before committing.** `python -m pytest tests/ -q` from the project root. Default skips GPU markers.
- **Ruff must be clean.** `ruff check . && ruff format --check .` — both no-op.
- **Smoke import after backend / app edits.** `python -c "import app; b = app.build_app(); print(type(b).__name__)"` should print `Blocks` without traceback. Catches most syntax / import-cycle issues without spinning up the full server.
- **For UI changes:** start the local server (`python app.py` → http://127.0.0.1:7860), click through the affected mode, verify visually. Don't trust a green test run + clean ruff as proof the UI works — the test suite doesn't render Gradio Blocks.
- **For deployment changes:** push to HF Space, watch the build stage transitions (`BUILDING` → `APP_STARTING` → `RUNNING`), verify the runtime stage hits `RUNNING` before claiming success.

If a change requires breaking these rules, write the reason in the commit body.

---

## Testing conventions

- **TDD per the plan.** Failing test first, then implementation.
- **L1** — unit tests on pure Python (mode registry, parameterize_fn, walker, extractor). Runs in CI without GPU.
- **L2** — graph validation against the bundled ComfyUI (`pytest --comfy-real`). Optional; runs locally + nightly.
- **L3** — backend smoke tests with real workflow JSONs but stubbed HTTP / filesystem boundaries. Runs in CI without GPU.
- **L4** — HF Space smoke. Manual click-through on the live Space after each deploy.
- **No mocks for ComfyUI core.** Tests run against real workflow JSONs. Stub only HTTP boundaries (HF Hub) and filesystem (use `tmp_path` and the `fake_hf_cache` fixture).
- `pyproject.toml` declares the `gpu` marker; pass `--gpu` to opt into GPU smoke.

---

## Out of scope (v1 — don't add without asking)

The spec at `docs/superpowers/specs/2026-04-30-ltx23-aio-generator-design.md` § 11 calls these out as v1.1+. If you find yourself "while I'm here"-ing into one of them, stop.

- **Lite mode** (`LTX23_AIO_LITE=1`) for the free HF Spaces tier
- **Custom LoRA** add/remove rows (Power-Lora-Loader clone)
- **GGUF Q4 transformer** / "Low VRAM" preset (currently always BF16-served)
- **Auto-launch the user's external ComfyUI** (`LTX23_AIO_COMFYUI_URL`)
- **Multi-prompt queueing**
- **Output history persistence** across sessions
- **Visual regression tests** for the Gradio UI
- **Property-based / fuzz testing** of workflow parameters
- **Persistent Storage add-on** integration (see `docs/future_improvements.md` item 6)
- **Telemetry-driven duration estimator** (requires persistent storage)

If a feature you're adding requires one of these as a sub-step, **ask the user.**

---

## When you're not sure

1. Read `docs/superpowers/specs/2026-04-30-ltx23-aio-generator-design.md` — that's the architectural source of truth.
2. Read `docs/superpowers/plans/2026-04-30-ltx23-aio-generator.md` — task-by-task breakdown.
3. Read `SKILLS.md` — process rules, debugging patterns, deployment workflow, useful one-liners.
4. Read `CLAUDE.md` — gotchas catalogue and what-not-to-do.
5. `git log --oneline` — every non-obvious decision has a fix-commit explaining the reasoning.
6. **Ask the user.** A clarifying question costs the user ten seconds. A wrong implementation costs everyone an hour.
