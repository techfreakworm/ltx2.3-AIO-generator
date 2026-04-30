# LTX 2.3 All-In-One Generator — Design Spec

**Date:** 2026-04-30
**Status:** Design approved, awaiting implementation plan
**Repo:** `~/Projects/llm/ltx2.3-AIO-generator`

## 1. Overview

A Gradio app that wraps the existing ComfyUI LTX 2.3 All-In-One workflow into a polished, mode-specific UI. Same code runs locally on Apple Silicon (MPS) or NVIDIA (CUDA) and deploys to Hugging Face Spaces with ZeroGPU. The Gradio frontend is a thin layer; ComfyUI is the inference engine — bundled and called as a Python library — so all of ComfyUI's smart model management, MPS handling, and node correctness are inherited rather than reimplemented.

Six generation modes ship in v1, mirroring the groups in `1. LTX 2.3 All-In-One 260406-05.json`:

| # | Mode | LTX-2 pipeline class |
|---|---|---|
| 1 | Text → Video (+optional Audio) | `TI2VidTwoStagesPipeline` / `DistilledPipeline` |
| 2 | Audio → Video (Text + Audio → Video + Audio) | `A2VidPipelineTwoStage` |
| 3 | Image → Video (+optional Audio) | `TI2VidTwoStagesPipeline` |
| 4 | Lipsync (Image + Audio → Video + Audio) | `A2VidPipelineTwoStage` |
| 5 | First / Last Frame → Video | `KeyframeInterpolationPipeline` |
| 6 | Style Transfer (Video → Video, motion control) | `ICLoraPipeline` |

## 2. Decisions log (Q1–Q8 + path)

| # | Question | Decision | Rationale |
|---|---|---|---|
| Q1 | Modes scope | All 6 | Marginal cost per mode is small; the differentiator vs other Gradio LTX demos is the unified shell. |
| Q2 | Settings exposure | Preset (Fast/Balanced/Quality) + Advanced accordion | Clean Spaces demo without sacrificing local power-user control. |
| Q3 | Backend | ComfyUI as headless backend (library mode) | ComfyUI is the production path on MPS; pure-Python `ltx-pipelines` has known crashes (TI2Vid OOM, A2Vid stage 2 SIGUSR1). Re-using ComfyUI's path inherits the fixes. |
| Q4 | Workflow templates | Six mode-specific JSON files | Smaller diff surface, easier tests, evolves per mode. `tools/extract_modes.py` regenerates them from the master workflow. |
| Q5 | LoRA UI | Categorized chrome (Camera dropdown · Detailer toggle · IC-LoRA mode-specific) | Mode-aware, no rope to misconfigure. Custom LoRA escape hatch deferred to v1.1. |
| Q6 | Layout shell | Sidebar nav + 2-column body | Six tab labels are too wide horizontally; sidebar gives mode names room and accommodates global panels. |
| Q7 | ComfyUI install | Bundled (git submodule locally, runtime clone on Spaces) | Self-contained, no dependence on user's existing ComfyUI install. |
| Q8 | Model storage | Local: HF cache → symlinks. Spaces: lazy `hf_hub_download` to `/data`. | Honors HF cache preference; no duplicate downloads; lazy strategy keeps Spaces `/data` budget under control. |
| Path | Spaces tier | Path A — Pro tier | ~70 GB minimum model footprint exceeds free tier `/data`; Balanced preset needs longer per-call duration. |

## 3. Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  Gradio UI   (sidebar nav · 2-col body · per-mode inputs)      │
│  ─ Mode tabs: T2V · A2V · I2V · Lipsync · Keyframe · Style    │
│  ─ Categorized LoRA chrome inside each mode's Advanced ▾       │
│  ─ Models / Settings / History panels in sidebar               │
└────────────────────────────────┬───────────────────────────────┘
                                 │  parameterize 1 of 6 templates
                                 ▼
┌────────────────────────────────────────────────────────────────┐
│  Workflow Builder   (workflows/<mode>.json + UI parameters)    │
│  ─ load_template(mode) → patch nodes → return JSON             │
│  ─ Validates inputs against the mode's required nodes          │
└────────────────────────────────┬───────────────────────────────┘
                                 │  workflow JSON dict
                                 ▼
┌────────────────────────────────────────────────────────────────┐
│  Backend (single impl)   ComfyUILibraryBackend                 │
│  ─ comfy.execution.PromptExecutor.execute(workflow)            │
│  ─ Hooks comfy.utils.PROGRESS_BAR_HOOK → yields ProgressEvent  │
│  ─ On Spaces: wrapped in @spaces.GPU(duration=N)               │
│  ─ Locally:   runs in a worker thread, GIL-released by torch   │
└────────────────────────────────┬───────────────────────────────┘
                                 │  progress events + outputs
                                 ▼
┌────────────────────────────────────────────────────────────────┐
│  Bundled ComfyUI   (vendored as a git submodule)               │
│  ─ ComfyUI core + ComfyUI-LTXVideo + KJNodes + rgthree         │
│  ─ models/ symlinks → ~/.cache/huggingface/hub  (local)        │
│  ─ models/ files on /data persistent volume   (Spaces)         │
└────────────────────────────────────────────────────────────────┘
```

### 3.1 Key invariants

1. **One backend interface, single implementation.** Library mode everywhere (`comfy.execution.PromptExecutor`). The `@spaces.GPU` decorator is the only divergence between local and Spaces.
2. **Workflow JSON is the contract.** Six small templates, parameterized at the leaves only. We don't reinvent ComfyUI's node graph.
3. **Models are never owned by the AIO repo.** Always either symlinked from HF cache (local) or downloaded to `/data` (Spaces). The bundled ComfyUI's `models/` is purely a view onto the cache.
4. **Auto MPS/CUDA dispatch.** The bundled ComfyUI handles device selection and dtype casting. The AIO layer writes no device code.

## 4. File structure

```
ltx2.3-AIO-generator/
├── app.py                    # Gradio entry — sidebar nav, mode rendering, generate handler
├── backend.py                # ComfyUI library backend; PromptExecutor wrapper; progress streaming
├── workflow.py               # load + parameterize a workflow JSON template
├── modes.py                  # MODE_REGISTRY: 6 modes × (inputs, defaults, parameterize fn)
├── models.py                 # symlink HF cache (local) / hf_hub_download to /data (Spaces)
├── ui.py                     # reusable Gradio components: LoRA chrome, preset bar, status banner
├── workflows/                # six mode-specific JSON templates (≤50 nodes each)
│   ├── t2v.json
│   ├── a2v.json
│   ├── i2v.json
│   ├── lipsync.json
│   ├── keyframe.json
│   └── style.json
├── tools/
│   ├── extract_modes.py      # rebuild templates from your master workflow
│   └── refresh_models.py     # refresh HF cache symlinks if snapshot SHAs change
├── tests/
│   ├── conftest.py
│   ├── test_workflow.py
│   └── test_modes.py
├── comfyui/                  # git submodule pinned to a known-good ComfyUI commit
├── setup.sh                  # init submodule, venv, install reqs, symlink models
├── requirements.txt          # gradio, spaces, huggingface-hub, torch, comfyui's own reqs
├── README.md                 # incl. HF Space front matter for one-touch deploy
├── CLAUDE.md                 # project guidelines (incl. sole-author commit rule)
└── .gitignore
```

### 4.1 Module responsibilities

| File | Responsibility | LOC est. |
|---|---|---|
| `app.py` | Gradio Blocks; sidebar navigation; per-mode input forms; calls `backend.submit()` | ~400 |
| `backend.py` | One class `ComfyUILibraryBackend`. Constructor adds `comfyui/` to `sys.path`, loads custom nodes, instantiates `PromptExecutor`. `submit(workflow)` is an async generator yielding `ProgressEvent`s. Handles ZeroGPU detection — wraps `_execute()` in `@spaces.GPU` if env var set. | ~200 |
| `workflow.py` | `load_template(mode)`, `set_input(workflow, node_id, field, value)`, `validate(workflow)`. Pure functions over dicts. | ~120 |
| `modes.py` | One `Mode` dataclass (name, icon, input_specs, parameterize_fn). `MODE_REGISTRY = {"t2v": Mode(...), ...}`. The `parameterize_fn` is the only mode-specific code. | ~300 |
| `models.py` | `ensure_models_for_mode(mode)`: walks the mode's workflow, finds loader nodes, identifies HF repo+filename, downloads via `hf_hub_download`, symlinks into `comfyui/models/...`. On Spaces, downloads to `/data`. | ~150 |
| `ui.py` | `lora_chrome(mode)` returns the categorized LoRA component group. `preset_bar()` returns the Fast/Balanced/Quality radio. `status_banner()` returns the `gr.HTML` for progress + stage text. | ~200 |

Total app code (excluding ComfyUI submodule and workflow JSONs): **~1,400 LOC** across 6 modules.

### 4.2 ComfyUI submodule + custom nodes

Pinned at a known-good commit. Custom nodes installed during `setup.sh` (local) or during runtime bootstrap (Spaces):

- `Lightricks/ComfyUI-LTXVideo` (LTX node implementations: `LTXICLoRALoaderModelOnly`, `LTXVChunkFeedForward`, `LTXVGemmaCLIPModelLoader`)
- `kijai/ComfyUI-KJNodes` (`VAELoaderKJ`, `ResizeImageMaskNode`, `INTConstant`, GetNode/SetNode helpers)
- `rgthree/rgthree-comfy` (`Power Lora Loader`, `Any Switch`, `Fast Groups Bypasser`, `Label`)
- `Kosinkadink/ComfyUI-VideoHelperSuite` (`VHS_VideoCombine`, `VHS_LoadVideo`, `VHS_LoadAudioUpload`)
- `pythongosssss/ComfyUI-Custom-Scripts` (`MathExpression|pysssss` — used by the master workflow for derived dimensions)

## 5. Data flow

User clicks **Generate** in the I2V tab. The path:

```
[1]  app.py: on_generate(mode="i2v", **inputs)
            │  Pulls Mode("i2v") from MODE_REGISTRY
            ▼
[2]  modes.i2v.parameterize_fn(inputs) → list[(node_id, field, value)]
            ▼
[3]  workflow.load_template("i2v") → dict
     workflow.set_input(wf, *patch) for each patch
     workflow.validate(wf)
            ▼
[4]  models.ensure_models_for_mode(wf)
     yields DownloadEvent(filename, mb_done, mb_total)
            ▼
[5]  backend.submit(wf) — async generator
     On Spaces: wrapped in @spaces.GPU(duration=preset_budget)
     Calls comfy.execution.PromptExecutor.execute(wf)
            ▼
[6]  PromptExecutor walks node graph
     Per-node: yields ProgressEvent(stage, step, total_steps)
            ▼
[7]  app.py: async for event in backend.submit(...):
     status_banner.html = render(event)
            ▼
[8]  Final node (VHS_VideoCombine) writes /tmp/out_<ts>.mp4
     yields OutputEvent(path)
            ▼
[9]  Gradio video component renders the file
     History panel adds row: timestamp · seed · duration
```

### 5.1 Three event types

```python
@dataclass
class DownloadEvent:    filename: str; mb_done: float; mb_total: float
@dataclass
class ProgressEvent:    stage: int; stage_label: str; step: int; total_steps: int
@dataclass
class OutputEvent:      video_path: str; audio_path: Optional[str]; meta: dict
```

The Gradio handler is one async generator that consumes these and yields `(status_html, video, history)` tuples.

### 5.2 Cancellation

Gradio's `Button.click(..., cancels=[generate_event])` calls `backend.interrupt()` → `comfy.model_management.interrupt_current_processing()`. The async generator's `finally:` block always frees GPU memory before raising.

## 6. Model loading & VRAM management

ComfyUI's `comfy.model_management` handles the heavy lifting — we write zero code for it.

**Inherited from ComfyUI:**
- Smart offload tiers (tracks total/free VRAM continuously; offloads largest non-live model when next load would overflow).
- Per-node load via `ModelPatcher`; LoRA patching applies deltas in-place without double-loading the base model.
- Automatic device dispatch and dtype casting (BF16/FP16/FP8 per `--force-*` args).
- ComfyUI-LTXVideo's existing MPS edge-case handling.

**AIO layer adds:**

| Concern | Implementation |
|---|---|
| Pre-flight download | `models.ensure_models_for_mode(wf)` walks loader nodes, resolves filenames via a `MODEL_REGISTRY` map, downloads via `hf_hub_download`, symlinks into `comfyui/models/<type>/<name>`. |
| VRAM tier hint | `comfy.cli_args.args.lowvram\|normalvram\|highvram` set at backend init based on detected GPU memory. Override via env var `LTX23_AIO_VRAM`. |
| Memory status badge | `ui.status_banner()` polls `comfy.model_management.get_free_memory()` every 2 s while idle. |
| Manual unload | Sidebar button **Unload all models** → `unload_all_models()` + `empty_cache()`. |
| Inter-mode caching | Single in-process ComfyUI keeps loaded models warm across mode switches. Free for us — ComfyUI's cache does it. |

### 6.1 Memory math (BF16)

| Component | Size | Loaded when |
|---|---|---|
| Distilled 22B transformer | ~44 GB | Diffusion stages |
| Gemma 3 12B text encoder | ~24 GB | Prompt encoding |
| Video VAE | ~2 GB | Encode (i2v/keyframe) + final decode |
| Audio VAE | ~0.5 GB | A2V/Lipsync only |
| LoRAs | <1 GB each | Patched into transformer |
| Latents | ~3 GB at 512×768/81f | Diffusion |

Realistic peak resident: ~70 GB on MPS unified memory; ~45 GB GPU + 24 GB system RAM on H200 80 GB ZeroGPU.

### 6.2 Out-of-scope (v1.1)

`UnetLoaderGGUF` for <24 GB consumer NVIDIA GPUs. The workflow templates already accommodate the GGUF node; v1.1 adds a "Low VRAM" preset that swaps the loader.

## 7. Progress reporting

Two surfaces, layered:

```
┌── Status Banner (gr.HTML) ────────────────────────────────────┐
│  ⠋  Stage 4/6 · Diffusion (Stage 1)                            │
│      Step 18/30 · 1m 12s elapsed · ~2m 41s remaining           │
│      MPS · 47 / 128 GB · transformer + gemma resident          │
│  ████████████████░░░░░░░░░░░░░░  60%                           │
└────────────────────────────────────────────────────────────────┘
```

Below: a `gr.Progress(track_tqdm=True)` picks up ComfyUI's sampler tqdm bars natively.

### 7.1 Stage map per mode

For each mode, `modes.py` declares the stage list mapping ComfyUI node ids → human-readable stage labels.

I2V Balanced preset stage map:

| # | Stage | ComfyUI node(s) | Typical share |
|---|---|---|---|
| 1 | Download missing models | (pre-flight) | 0–60s, only on first run |
| 2 | Encode prompt | `LTXVGemmaCLIPModelLoader` + `CLIPTextEncode` | ~5% |
| 3 | Encode image | `LoadImage` + image VAE encode | ~3% |
| 4 | Diffusion (Stage 1, half-res) | `KSampler` × N steps | ~55% |
| 5 | Spatial upscale (×2) | `LatentUpscaleModelLoader` + sampler | ~7% |
| 6 | Diffusion (Stage 2, full-res, 4 distilled steps) | `KSampler` × 4 | ~20% |
| 7 | Decode video | Video VAE decode + `VHS_VideoCombine` | ~10% |

T2V is shorter (no image encode); Lipsync adds audio encode; Style Transfer is single-stage.

### 7.2 Plumbing

ComfyUI's `PromptExecutor` calls a per-node hook before each node runs. The backend translates `node_id → stage_index` via the mode's stage map. Within sampler nodes, `comfy.utils.PROGRESS_BAR_HOOK` fires per step. ETA: `(elapsed / progress) - elapsed` capped to a sensible minimum.

## 8. Error handling

| # | Category | Surface | Recovery |
|---|---|---|---|
| 1 | Setup / install (`comfyui/` missing, custom node import failure, no torch CUDA/MPS) | Startup banner replaces the UI; red card with the failing component and exact `setup.sh` command. App refuses to start. | Local: `bash setup.sh`. Spaces: surfaces in build log. |
| 2 | Model download (network, HF auth, disk full) | Status banner inline error with retry button. Auth errors prompt for `HF_TOKEN`. | Auto-retry once with backoff for transient. Auth/disk are user-actionable. |
| 3 | Workflow validation (input not provided, frame count not 8k+1, resolution not /32, image too large) | Caught client-side; Gradio inline validation; generate button disabled. | Auto-snap where unambiguous (frame count to nearest 8k+1, resolution to nearest /32). |
| 4 | ComfyUI execution (node not found, shape mismatch, file format) | Status banner shows failing stage in red; collapsible `View full traceback ▾`. | Suggests `tools/refresh_models.py` for symlink issues, `bash setup.sh --update-comfy` for node issues. |
| 5 | OOM | Status banner with stage + memory at failure; **Try Fast preset** button. | On catch: `unload_all_models()` + `empty_cache()`. Next click starts clean. |
| 6 | ZeroGPU duration exceeded (Spaces) | Status banner: "Generation exceeded GPU budget"; suggests **Switch to Fast preset**. Partial output (if decoded) still shown. | `@spaces.GPU(duration=N)` raises a specific exception we catch and translate. |

### 8.1 try/finally discipline

```python
async def submit(self, workflow):
    try:
        async for event in self._execute_with_progress(workflow):
            yield event
    except OutOfMemoryError as e:
        yield ErrorEvent(category="oom", stage=self._current_stage, ...)
    except spaces.exceptions.GPUDurationExceededError as e:
        yield ErrorEvent(category="zerogpu_timeout", ...)
    except Exception as e:
        yield ErrorEvent(category="execution", traceback=fmt(e), ...)
    finally:
        comfy.model_management.unload_all_models()
        torch.mps.empty_cache() if mps else torch.cuda.empty_cache()
```

The `finally` block is the single most important line for VRAM hygiene. Cancellation triggers the same path via `interrupt_current_processing()` raising `InterruptedError`.

### 8.2 Logging

- Local: `comfyui/comfyui.log` + `logs/aio.log` (10 MB rotation).
- Spaces: stderr → Space logs panel; no file logging (Space disk is ephemeral except `/data`).
- Status banner's traceback expander reads the last error from `logs/aio.log` (local) or stderr buffer (Spaces).

### 8.3 Deliberate non-goals

No silent retries on ambiguous errors. Surface loudly with a traceback rather than mask real bugs.

## 9. Deployment

### 9.1 Local

```bash
git clone https://github.com/<your-handle>/ltx2.3-AIO-generator
cd ltx2.3-AIO-generator
bash setup.sh
source .venv/bin/activate
python app.py
```

`setup.sh` (idempotent):

```bash
#!/usr/bin/env bash
set -euo pipefail

python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip

git submodule update --init --recursive
pip install -r comfyui/requirements.txt

cd comfyui/custom_nodes
for repo in \
    Lightricks/ComfyUI-LTXVideo \
    kijai/ComfyUI-KJNodes \
    rgthree/rgthree-comfy \
    Kosinkadink/ComfyUI-VideoHelperSuite \
    pythongosssss/ComfyUI-Custom-Scripts ; do
  name="${repo##*/}"
  [[ -d "$name" ]] || git clone "https://github.com/$repo.git" "$name"
  [[ -f "$name/requirements.txt" ]] && pip install -r "$name/requirements.txt"
done
cd ../..

pip install -r requirements.txt
python tools/refresh_models.py

echo "Setup complete. Run: source .venv/bin/activate && python app.py"
```

### 9.2 HF Spaces (ZeroGPU, Pro tier)

`README.md` front matter:

```yaml
---
title: LTX 2.3 All-in-One Video Generator
emoji: 🎬
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.0"
app_file: app.py
python_version: "3.11"
suggested_hardware: zero-gpu
hf_oauth: false
---
```

Bootstrap inside `app.py` runs once on cold start:

```python
def _bootstrap():
    on_spaces = bool(os.environ.get("SPACES_ZERO_GPU"))
    comfy_dir = pathlib.Path("/data/comfyui" if on_spaces else "comfyui")

    if on_spaces and not comfy_dir.exists():
        _git_clone(COMFYUI_REPO, comfy_dir, ref=COMFYUI_COMMIT)
        for node_repo, node_ref in CUSTOM_NODES_PINNED:
            _git_clone(node_repo, comfy_dir / "custom_nodes" / node_repo.split("/")[-1], ref=node_ref)
        _pip_install_custom_node_reqs(comfy_dir)

    sys.path.insert(0, str(comfy_dir))
    os.environ["COMFY_MODELS_DIR"] = str(
        pathlib.Path("/data/models") if on_spaces else (comfy_dir / "models")
    )
```

Storage budget: `/data` ~50 GB on Pro. Lazy per-mode download keeps usage under budget when only some modes are exercised.

Per-call duration: `@spaces.GPU(duration=...)` per preset:

| Preset | Duration |
|---|---|
| Fast | 60 s |
| Balanced | 120 s |
| Quality | 300 s |

UI auto-greys out presets whose duration exceeds the detected `SPACES_GPU_DURATION_LIMIT`.

### 9.3 One-touch deploy (optional)

`.github/workflows/deploy-space.yml`:

```yaml
on: { push: { branches: [main] } }
jobs:
  push-to-space:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { lfs: true }
      - name: Push to HF Space
        env: { HF_TOKEN: ${{ secrets.HF_TOKEN }} }
        run: |
          git remote add space https://user:$HF_TOKEN@huggingface.co/spaces/<you>/ltx2.3-aio
          git push --force space main
```

### 9.4 Local vs Spaces — what's identical, what differs

| Concern | Local | Spaces |
|---|---|---|
| Backend code | `ComfyUILibraryBackend` | `ComfyUILibraryBackend` (same class) |
| GPU decorator | none (worker thread) | `@spaces.GPU(duration=preset_budget)` |
| ComfyUI install | git submodule | runtime git clone to `/data` |
| Models location | symlinks → `~/.cache/huggingface` | direct files in `/data/models` |
| Logging | `logs/aio.log` + `comfyui/comfyui.log` | stderr → Space logs panel |
| First-run latency | seconds (deps installed by setup.sh) | minutes (clone + first-mode download) |
| Custom nodes update | re-run `setup.sh` | push commit; rebuild Space |

## 10. Testing

Layered so most tests run on CPU in seconds; only smoke touches GPU.

| Layer | What it verifies | GPU? | Time |
|---|---|---|---|
| L1 — Unit | `workflow.load_template`, `set_input`, `validate` (pure functions over JSON dicts) | No | < 1 s |
| L1 — Unit | Each mode's `parameterize_fn`: known input → expected patch list | No | < 1 s |
| L1 — Unit | `MODEL_REGISTRY` lookups: every model in every workflow resolves to an HF repo+filename | No | < 1 s |
| L2 — Graph validation | `load_template + parameterize_fn(canonical_inputs)` produces a workflow that ComfyUI's `validate_prompt` accepts | No | < 5 s |
| L3 — Integration (CPU) | `models.ensure_models_for_mode()` against a fake HF cache; symlinks created correctly | No | < 2 s |
| L4 — Smoke (GPU, opt-in) | One end-to-end generation per mode at minimum viable settings (Fast preset, lowest legal resolution, 1 step). `pytest --gpu`. | Yes | ~3 min for all 6 |

### 10.1 Fixtures

- `canonical_inputs(mode)` — known-good Gradio input dict per mode.
- `fake_hf_cache(tmp_path)` — fake `~/.cache/huggingface/hub` with placeholder files.
- `--gpu` flag enables L4. Default skips with a reason.
- `--comfy-real` flag uses bundled ComfyUI for L2; default uses a stubbed validator.

### 10.2 CI

`.github/workflows/ci.yml` runs L1 + L2 + L3 on `ubuntu-latest`, Python 3.11, every push. ~30 s wall time. No GPU runner. Lint: `ruff check` + `ruff format --check`.

### 10.3 Deliberate non-goals

- No mocks for ComfyUI itself.
- No visual regression tests for Gradio UI.
- No property-based / fuzz testing for workflow params.

## 11. Out of scope (v1)

- **Lite mode for free Spaces tier** — `LTX23_AIO_LITE=1` env var that filters MODE_REGISTRY to T2V+I2V, locks Fast preset, swaps GGUF transformer. Designed in but not built in v1.
- **Custom LoRA escape hatch** — Power-Lora-Loader-style add/remove rows. Categorized chrome covers v1; custom is a v1.1 toggle.
- **GGUF Q4 transformer (`UnetLoaderGGUF`)** — for <24 GB consumer NVIDIA GPUs. Workflow templates accommodate the node; v1.1 adds the "Low VRAM" preset.
- **Auto-launch user's existing ComfyUI** — current design uses bundled ComfyUI exclusively. v1.1 could add `LTX23_AIO_COMFYUI_URL` env var to point at an external server.
- **Multi-prompt queueing** — Gradio default single-shot is fine. ComfyUI's queue isn't exposed.
- **History persistence across sessions** — sidebar history is in-memory. Local could read `outputs/` on startup; Spaces session storage is ephemeral.

## 12. Open questions / follow-ups

- **Pinned ComfyUI commit:** select after a manual end-to-end run on the user's `~/Projects/comfyui/` install. Capture the commit SHA in `setup.sh` and the Spaces bootstrap.
- **Spaces secrets:** HF Space front matter doesn't include any secrets; `HF_TOKEN` only needed if a gated repo is used (not currently). Document in README.
- **Output retention on Spaces:** decide whether `/tmp/out_*.mp4` should also copy to `/data/outputs/` for download-after-restart. v1 default: no, ephemeral.
- **`MODEL_REGISTRY` source of truth:** the registry maps filename → HF repo. We populate it once at v1 from Lightricks' README + Kijai's repo and freeze it; updates require a code change + tests.

## 13. Implementation order (preview — full breakdown in implementation plan)

1. **Repo skeleton** — directory layout, `.gitignore`, `CLAUDE.md`, `README.md` stub, `requirements.txt`.
2. **`tools/extract_modes.py`** — extract six mode templates from the master workflow. Validates by re-loading each in ComfyUI's parser.
3. **`workflow.py`** — pure-function library with L1 + L2 tests.
4. **`modes.py`** — MODE_REGISTRY with `parameterize_fn` per mode + L1 tests.
5. **`models.py`** — registry + `ensure_models_for_mode` + L3 tests with fake HF cache.
6. **`backend.py`** — ComfyUILibraryBackend, async submit, progress hook plumbing. Local smoke test (L4) for Fast/T2V.
7. **`ui.py`** — LoRA chrome, preset bar, status banner.
8. **`app.py`** — Gradio Blocks, sidebar nav, mode rendering, generate handler. Manual end-to-end on Mac for all 6 modes.
9. **`setup.sh`** — idempotent local bootstrap.
10. **`README.md` + Spaces front matter** — push to a test Space, verify cold-start and one Fast generation.
11. **CI workflow** — L1 + L2 + L3 on push.
12. **Optional `.github/workflows/deploy-space.yml`** — push-to-Space CI.
