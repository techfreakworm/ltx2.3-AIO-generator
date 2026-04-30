# Claude / Agent Working Notes — ltx2.3-AIO-generator

Project guidelines for AI assistants working in this repo.

## Git authorship

Mayank Gupta is the **sole author** on every commit. Never:
- Append a `Co-Authored-By: Claude ...` trailer.
- Set `--author` to anything other than the user's existing git config.
- Add "Generated with Claude Code", "🤖", or any similar attribution lines to commit messages.
- Add similar attribution to PR descriptions.

If asked to amend or re-commit, strip any prior Claude attribution.

## Project at a glance

Gradio app wrapping the existing ComfyUI LTX 2.3 All-In-One workflow. Same code runs locally (Apple Silicon MPS or NVIDIA CUDA) and deploys to Hugging Face Spaces (ZeroGPU, Pro tier).

**Key architectural facts (do not relitigate):**

1. **Backend is ComfyUI in library mode**, always. We do not call `ltx-pipelines` directly. We call `comfy.execution.PromptExecutor` with workflow JSONs we parameterize. ComfyUI is bundled (git submodule locally, runtime clone on Spaces).
2. **Six mode-specific workflow JSON files** in `workflows/`. They are derived from the master workflow at `~/Projects/comfyui/user/default/workflows/1. LTX 2.3 All-In-One 260406-05.json` via `tools/extract_modes.py`. Do not hand-edit the JSON files unless re-extracting from a new master.
3. **Models live in HF cache (local) or `/data` (Spaces)**, never in this repo. `comfyui/models/` contains symlinks (local) or downloaded files (Spaces). Do not commit any `*.safetensors` / `*.gguf`.
4. **Library mode means single-process.** No subprocess for ComfyUI. The `@spaces.GPU` decorator is the only difference between local and Spaces runtime.
5. **VRAM management is ComfyUI's job.** Don't write `torch.cuda.empty_cache()` calls outside the `try/finally` in `backend.py`. Don't second-guess ComfyUI's offload tiers.

See `docs/superpowers/specs/2026-04-30-ltx23-aio-generator-design.md` for the full design.

## Coding conventions

- **Python 3.11.** No `match` statements (compatibility with the Spaces Python pin).
- **Flat structure.** No `src/` layout, no nested packages. Each top-level `.py` is one module with one job.
- **No conda.** Use `python3.11 -m venv .venv`. Use `brew` for system binaries.
- **HF cache, not project-local.** Use `hf download <repo>` (the `hf` CLI, not deprecated `huggingface-cli`) without `--local-dir`. Symlink resolved snapshot paths.
- **No mocks for ComfyUI.** Tests against real workflow JSONs. Stubs only for HTTP / filesystem boundaries.
- **No emojis** in code or commit messages unless explicitly requested.
- **Comments only when WHY is non-obvious.** Don't narrate WHAT.

## Editing the master workflow

When the user updates `~/Projects/comfyui/user/default/workflows/1. LTX 2.3 All-In-One 260406-05.json` (e.g., new LoRA, tweaked sampler), re-run:

```bash
python tools/extract_modes.py --master ~/Projects/comfyui/user/default/workflows/"1. LTX 2.3 All-In-One 260406-05.json"
```

This regenerates all six `workflows/<mode>.json` files. L2 graph-validation tests will catch any node that became invalid.

## Out of scope (do not implement without asking)

- "Lite mode" for free HF Spaces tier (`LTX23_AIO_LITE=1`).
- Custom LoRA add/remove rows (Power-Lora-Loader clone).
- GGUF Q4 transformer / "Low VRAM" preset.
- Auto-launch of user's external ComfyUI install (`LTX23_AIO_COMFYUI_URL`).
- Multi-prompt queueing.
- Output history persistence across sessions.

These are documented as v1.1+ in the spec. Do not pre-build them.

## When in doubt

Read `docs/superpowers/specs/2026-04-30-ltx23-aio-generator-design.md`. If still unclear, ask before changing architectural shape.
