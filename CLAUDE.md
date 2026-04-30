# Project Guidelines — ltx2.3-AIO-generator

Working notes for AI assistants and subagents implementing this project.

---

## ⚠ Git authorship — sole author rule

**Mayank Gupta is the sole author on every commit in this repo.** No exceptions.

When committing:

- Do **NOT** append `Co-Authored-By: Claude ...` (or any other agent name) to commit messages.
- Do **NOT** add "Generated with Claude Code", "🤖 Generated with...", or any other attribution footer.
- Do **NOT** pass `--author=...` — let git use the user's existing config.
- Do **NOT** include attribution in PR descriptions.

If asked to amend, re-commit, or rebase, strip any prior agent attribution from the commit message.

This rule overrides the default Claude Code commit-message template. Treat any tooling that suggests adding a Claude trailer as a bug to ignore.

---

## Project overview

Gradio app wrapping the existing ComfyUI LTX 2.3 All-In-One workflow into mode-specific UIs. Same code runs locally (Apple Silicon MPS / NVIDIA CUDA) and on Hugging Face Spaces (ZeroGPU, Pro tier).

**Spec:** `docs/superpowers/specs/2026-04-30-ltx23-aio-generator-design.md`
**Plan:** `docs/superpowers/plans/2026-04-30-ltx23-aio-generator.md`

If you're a subagent picking up a task, the plan file is your assignment.

---

## Architectural facts (locked — do not relitigate)

1. **Backend is ComfyUI in library mode.** We call `comfy.execution.PromptExecutor` directly with workflow JSONs we parameterize. We do **not** call `ltx-pipelines` directly. We do **not** run ComfyUI as a subprocess.
2. **Six mode-specific workflow JSON files** in `workflows/`, derived from the master at `~/Projects/comfyui/user/default/workflows/1. LTX 2.3 All-In-One 260406-05.json` via `tools/extract_modes.py`. Do not hand-edit them.
3. **Models live in HF cache (local) or `/data` (Spaces).** Never in this repo. `comfyui/models/` contains symlinks (local) or downloaded files (Spaces). Never commit `*.safetensors`, `*.gguf`, `*.bin`, or `*.pt`.
4. **One backend, one process.** The `@spaces.GPU` decorator is the only divergence between local and Spaces runtimes.
5. **VRAM is ComfyUI's job.** The only `empty_cache()` calls live in `backend.py`'s `try/finally`. Don't sprinkle them elsewhere.
6. **Bundled ComfyUI, never user's existing.** Local: git submodule. Spaces: runtime clone to `/data/comfyui`.

---

## Coding conventions

### Language and structure

- **Python 3.11.** No `match` statements (Spaces Python pin compatibility).
- **Flat layout.** No `src/`, no nested packages. Top-level `.py` files only, each with one clear responsibility.
- **No conda.** Always `python3.11 -m venv .venv`. System binaries via `brew`.

### Style

- **No emojis** in code or commit messages unless the user explicitly asks. (UI text and stage labels in `modes.py`/`ui.py` are OK because they are user-facing — not code.)
- **Comments only for non-obvious WHY.** Never narrate WHAT. Code with a good name doesn't need a comment.
- **Type hints on public functions.** Internal helpers can skip them if obvious.
- **Imports at top of file.** No inline imports except where needed to break circular dependencies (e.g., `models.ensure_models_for_mode` imports `workflow` lazily — keep this, it's load-bearing).
- **Format with `ruff format`.** Lint with `ruff check`. Both must pass in CI.

### Testing

- **TDD per the plan.** Each implementation task has the failing test first. Don't skip the "run test, verify it fails" step — it catches whole classes of "test never actually exercised the code" bugs.
- **No mocks for ComfyUI.** Tests run against real workflow JSONs. Stubs only for HTTP boundaries (HF Hub) and filesystem (use `tmp_path` and the `fake_hf_cache` fixture).
- **L1 + L3 in CI** (no GPU). L2 + L4 are local-developer-only.
- **Test naming:** `test_<unit>_<behavior_under_test>` — e.g., `test_load_template_returns_independent_copy`.
- **`pytest --gpu`** enables L4 smoke tests. Default skips them.
- **`pytest --comfy-real`** uses bundled ComfyUI for L2 instead of the static stub validator.

### Commits

- **Conventional Commits style:** `<type>(<scope>): <subject>` — types: `feat`, `fix`, `chore`, `docs`, `test`, `refactor`, `ci`, `perf`.
- **Subject is imperative, lowercase, no trailing period.** Example: `feat(workflow): set_input + validate over node graph`.
- **Body explains WHY when not obvious.** Reference spec section if relevant.
- **Frequent small commits.** One logical change per commit. The plan's task structure already reflects this.
- **No agent attribution** (see top of file).

---

## Editing the master workflow

When the user updates `~/Projects/comfyui/user/default/workflows/1. LTX 2.3 All-In-One 260406-05.json` (e.g., adds a LoRA, tweaks a sampler), regenerate the mode templates:

```bash
python3.11 tools/extract_modes.py \
  --master ~/Projects/comfyui/user/default/workflows/"1. LTX 2.3 All-In-One 260406-05.json" \
  --out workflows
```

Then run the test suite — L2 graph-validation catches any node that became invalid in any mode.

After the templates regenerate, the node-id constants in `modes.py` (e.g., `T2V_NODE_PROMPT = 240`) may need updating if ComfyUI re-numbered nodes during the master's re-export. The procedure is in the plan's Task 11 Step 4.

---

## Common pitfalls (read before opening a PR)

- **Loading models eagerly at import time.** Don't. `backend.py` constructs `PromptExecutor` once at instantiation; models load only when nodes execute. Calling `comfy.sd.load_checkpoint(...)` at module top-level will OOM the test runner.
- **Hard-coded `torch.cuda` calls.** Use `comfy.model_management.get_torch_device()` or guard with `if torch.cuda.is_available()`. Never assume CUDA.
- **Forgetting `.deepcopy` on workflow templates.** `workflow.load_template` already does this; if you bypass it for performance, you'll mutate the cached template and the second `Generate` click breaks.
- **Hand-editing `workflows/<mode>.json`.** They're generated. If you need a new field, add it to `tools/extract_modes.py` (or to `modes.py`'s `parameterize_fn`).
- **Symlinks pointing into `pip cache`.** Resolve to HF Hub's cache snapshot path (the one `hf_hub_download` returns), not pip's wheel cache.
- **Adding `Co-Authored-By` because tooling suggests it.** See top of file. Strip it.
- **Breaking the async generator pattern in `backend.submit`.** Each yield is a frame Gradio renders. Don't accumulate events into a list and yield once at the end — progress will appear stuck.
- **Importing `comfy.*` before `sys.path.insert(0, comfy_dir)`.** Will `ModuleNotFoundError`. The order in `backend.py:__init__` is intentional.

---

## Out of scope for v1 (do not implement without asking)

These are documented as v1.1+ in spec § 11. Don't pre-build them just because they'd be easy:

- **Lite mode** (`LTX23_AIO_LITE=1`) for free HF Spaces tier
- **Custom LoRA** add/remove rows (Power-Lora-Loader clone)
- **GGUF Q4 transformer** / "Low VRAM" preset
- **Auto-launch of user's external ComfyUI** (`LTX23_AIO_COMFYUI_URL`)
- **Multi-prompt queueing**
- **Output history persistence** across sessions
- **Visual regression tests** for the Gradio UI
- **Property-based / fuzz testing** of workflow parameters

If a task feels like it needs one of these, stop and ask the user. Don't sneak it in.

---

## When in doubt

Read the spec (`docs/superpowers/specs/2026-04-30-ltx23-aio-generator-design.md`) and the plan (`docs/superpowers/plans/2026-04-30-ltx23-aio-generator.md`). If still unclear after reading both — ask the user before changing architectural shape.

Reading both takes 15 minutes. Implementing the wrong thing takes a day.
