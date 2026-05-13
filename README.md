---
title: LTX 2.3 Studio
emoji: 🎬
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.50.0"
app_file: app.py
python_version: "3.11"
suggested_hardware: zero-a10g
hf_oauth: false
preload_from_hub:
  - Comfy-Org/ltx-2 split_files/text_encoders/gemma_3_12B_it.safetensors
  - Kijai/LTX2.3_comfy diffusion_models/ltx-2.3-22b-dev_transformer_only_bf16.safetensors,loras/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors,text_encoders/ltx-2.3_text_projection_bf16.safetensors,vae/LTX23_audio_vae_bf16.safetensors,vae/LTX23_video_vae_bf16.safetensors,vae/taeltx2_3.safetensors
  - Lightricks/LTX-2-19b-IC-LoRA-Detailer ltx-2-19b-ic-lora-detailer.safetensors
  - Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down ltx-2-19b-lora-camera-control-jib-down.safetensors
  - Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up ltx-2-19b-lora-camera-control-jib-up.safetensors
  - Lightricks/LTX-2-19b-LoRA-Camera-Control-Static ltx-2-19b-lora-camera-control-static.safetensors
  - Lightricks/LTX-2.3 ltx-2.3-22b-distilled-lora-384.safetensors,ltx-2.3-spatial-upscaler-x2-1.0.safetensors
  - Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors
  - google/gemma-3-12b-it-qat-q4_0-unquantized gemma-3-12b-it/model-00001-of-00005.safetensors,gemma-3-12b-it/model-00002-of-00005.safetensors,gemma-3-12b-it/model-00003-of-00005.safetensors,gemma-3-12b-it/model-00004-of-00005.safetensors,gemma-3-12b-it/model-00005-of-00005.safetensors,gemma-3-12b-it/model.safetensors.index.json,gemma-3-12b-it/preprocessor_config.json,gemma-3-12b-it/tokenizer.model
---

# LTX 2.3 Studio

A single-process Gradio app that wraps [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) — Lightricks' open 22B video generation model — under one focused UI. Six modes (text · image · audio · lipsync · keyframe · style) sharing the same ComfyUI All-In-One workflow. Runs locally on Apple Silicon (MPS) or NVIDIA (CUDA), deploys to Hugging Face Spaces (ZeroGPU).

[![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Live-E0A458?style=flat-square)](https://huggingface.co/spaces/techfreakworm/LTX2.3-Studio)
[![GitHub stars](https://img.shields.io/github/stars/techfreakworm/ltx2.3-AIO-generator?style=flat-square&color=E0A458)](https://github.com/techfreakworm/ltx2.3-AIO-generator/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-E0A458?style=flat-square)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-E0A458?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![Powered by ComfyUI](https://img.shields.io/badge/backend-ComfyUI-E0A458?style=flat-square)](https://github.com/comfyanonymous/ComfyUI)
[![Built on LTX-2.3](https://img.shields.io/badge/model-LTX--2.3%2022B-E0A458?style=flat-square)](https://huggingface.co/Lightricks/LTX-2.3)

→ **Live demo:** https://huggingface.co/spaces/techfreakworm/LTX2.3-Studio

---

## What's inside

Six modes wired through the same ComfyUI All-In-One workflow. Each mode exposes only the inputs it actually consumes — the form stays short and focused.

| Mode | Inputs | Output | Notes |
|---|---|---|---|
| **Text → Video** | Prompt (+ optional audio prompt) | mp4 (+ optional wav) | The core mode. Camera-control LoRAs auto-applied by keyword. |
| **Audio → Video** | Prompt + audio track | mp4 with the input audio preserved | Conditions motion on the audio waveform. |
| **Image → Video** | Image + prompt | mp4 (+ optional audio) | Image-conditioned generation. |
| **Lipsync** | Image + audio | mp4 with audio | Viseme-aligned mouth motion. |
| **Keyframe** | First + last frames + prompt | mp4 | Latent interpolation between two anchors. |
| **Style Transfer** | Source video + style image | mp4 | IC-LoRA restyle; motion preserved from source. |

Every mode carries **Fast / Balanced / Quality** presets (steps × 1, × 1.5, × 3). A per-mode ZeroGPU duration estimator adapts the call timeout to the requested workload.

---

## Quick start (local)

Requires **Python 3.11**, ~80 GB free disk for the weight set, and ~24 GB VRAM (CUDA) or ~32 GB unified memory (Apple Silicon).

```bash
git clone --recurse-submodules https://github.com/techfreakworm/ltx2.3-AIO-generator
cd ltx2.3-AIO-generator
bash setup.sh           # creates .venv, installs ComfyUI + pinned custom nodes + app deps
source .venv/bin/activate
python app.py           # http://127.0.0.1:7860
```

The first run resolves model weights into your HF cache (`~/.cache/huggingface/hub/`) and symlinks them into `comfyui/models/<comfy_type>/`. Subsequent starts skip the download. Expect ~70 GB of weights pulled on a cold first run.

**Apple Silicon notes.** `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically so the few MPS-unsupported ops fall back to CPU. ComfyUI's VRAM autodetect picks the right tier; override with `LTX23_AIO_VRAM=lowvram|normalvram|highvram` if you need to force one.

**LAN access** (phone / tablet on the same WiFi): `python app.py` binds `0.0.0.0:7860`. Visit `http://<your-LAN-IP>:7860` from another device. On macOS, allow inbound for `python` in System Settings → Network → Firewall if the connection refuses.

## Quick start (HF Spaces)

This repo is a Gradio Space. The Pro tier provides ZeroGPU (A10G) access and the per-call duration budget needed for the Balanced and Quality presets.

```bash
git remote add space https://huggingface.co/spaces/<your-handle>/LTX2.3-Studio
git push space master:main       # local branch is master; HF Space deploys from main
```

> ⚠ The refspec `master:main` matters. The local default branch is `master` (GitHub convention); the HF Space deploys from `main`. A bare `git push space master` creates an orphan remote branch that does NOT trigger a deploy.

The Space's `preload_from_hub` directive (see the YAML at the top of this file) bakes ~111 GB of weights into the build image. `app.py:_bootstrap()` then:

1. Clones ComfyUI + pinned custom nodes into `~/comfyui` on cold start (ZeroGPU container freezes preserve them across calls)
2. Mirrors the read-only preload cache into `~/hf-cache-rw/` — works around the build-user-vs-runtime-user permissions trap (preloaded files are root-owned; we run as uid 1000 and can't write to them, so any lazy download to the cache would fail with `Permission denied`)
3. Stages seed input files into `comfyui/input/` so workflow loaders don't error before any user upload arrives

Subsequent requests hit warm cache — no network traffic on inference 2+.

**ZeroGPU duration estimator.** Each generate call carries a dynamic `@spaces.GPU(duration=N)` calculated from mode, preset, and frame count. Clamped at `[60, 900] s`. On timeout (`"GPU task aborted"`), the handler auto-retries once at 2× duration.

---

## Architecture

```
                                     ┌──────────────────────────────────┐
                          browser ──▶│   app.py — Gradio Blocks         │
                                     │   header · drawer · 6 mode tabs  │
                                     └──────────────────┬───────────────┘
                                                        │
                                                        ▼
                                     ┌──────────────────────────────────┐
                                     │   backend.py                     │
                                     │   ComfyUILibraryBackend          │
                                     │   @spaces.GPU(duration=callable) │
                                     │   calls PromptExecutor directly  │
                                     └──────────────────┬───────────────┘
                                                        │
       ┌──────────────┬──────────────┬──────────────────┴──────┬──────────────────┐
       ▼              ▼              ▼                         ▼                  ▼
   modes.py       models.py      workflow.py                ui.py              tools/
   per-mode       walk + ensure  load + patch               per-mode form      extract_modes.py
   parameterize   from HF cache  API-format JSON            builders           (regen workflows/)
                                                        │
                                                        ▼
                                     ┌──────────────────────────────────┐
                                     │   comfyui/                       │
                                     │   submodule (local)              │
                                     │   runtime clone at ~/comfyui     │
                                     │   on HF Spaces                   │
                                     │                                  │
                                     │   ├── custom_nodes/ (pinned SHAs)│
                                     │   └── models/ → HF cache symlinks│
                                     └──────────────────────────────────┘
```

**One backend, one process.** The `@spaces.GPU` decorator is the only divergence between local and Spaces runtime. ComfyUI manages VRAM via its tiered presets — no `empty_cache()` sprinkling needed elsewhere.

**Workflow as data.** Each of the six modes is a user-exported API-format JSON in `workflows/`. The mode handler patches a deep-copied template (`modes.parameterize_fn`) and hands it to ComfyUI's `PromptExecutor`. Updating the master workflow is a three-step ritual: edit in the ComfyUI editor → export → `python tools/extract_modes.py --master ... --out workflows`.

---

## Project layout

```
.
├── app.py              # Gradio Blocks entry, _bootstrap, _on_generate, mode tabs
├── backend.py          # ComfyUILibraryBackend, @spaces.GPU, duration estimator
├── modes.py            # MODE_REGISTRY + per-mode parameterize_fn + node-id constants
├── models.py           # MODEL_REGISTRY, walk_workflow_for_models, ensure_models
├── ui.py               # render_status, _render_idle, mode-form layout primitives
├── workflow.py         # load_template, set_input helpers
├── workflows/          # API-format mode JSONs (do not hand-edit)
│   ├── t2v.json
│   ├── i2v.json
│   ├── a2v.json
│   ├── lipsync.json
│   ├── keyframe.json
│   └── style.json
├── assets/seed_inputs/ # placeholder image / audio / video for cold-start staging
├── tools/
│   └── extract_modes.py  # regenerate workflows/ from a master ComfyUI export
├── docs/
│   ├── future_improvements.md
│   └── superpowers/{specs,plans}/  # spec + implementation plans per feature
├── tests/              # L1 + L3 in CI; L2 with --comfy-real; L4 GPU smoke
├── README.md           # this file (HF Space YAML + project intro)
├── CLAUDE.md           # project facts + gotchas (what & why)
├── AGENTS.md           # tool-agnostic agent rulebook
├── SKILLS.md           # process / debugging / deployment (how)
├── requirements.txt    # pinned deps
├── pyproject.toml      # ruff + pytest config (py311)
├── setup.sh            # venv + ComfyUI + custom nodes bootstrap
└── comfyui/            # git submodule (local) / runtime clone target (Spaces)
```

---

## Tech stack

- **[Gradio 5.50](https://gradio.app/)** — UI shell, native components, `gr.Progress(track_tqdm=True)`
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** — library-mode `PromptExecutor` (pinned commit; submodule locally, runtime-cloned on Spaces)
- **[LTX-2.3 22B](https://huggingface.co/Lightricks/LTX-2.3)** by Lightricks — primary diffusion transformer (BF16 weights via [Kijai/LTX2.3_comfy](https://huggingface.co/Kijai/LTX2.3_comfy))
- **[Gemma 3 12B](https://huggingface.co/google/gemma-3-12b-it)** by Google — multimodal text encoder (requires the full 5-shard model — text-only checkpoints crash on meta-tensor allocation in SDPA)
- **Custom nodes** (pinned SHAs in `app.CUSTOM_NODES_PINNED`):
  - [Lightricks/ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) — LTX sampler / decoder nodes
  - [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) — utility nodes
  - [rgthree/rgthree-comfy](https://github.com/rgthree/rgthree-comfy) — Power-Lora-Loader
  - [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) — video I/O
  - [pythongosssss/ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts) — string / dict helpers
  - [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) — GGUF transformer loader
  - [Fannovel16/comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) — DWPose for Lipsync/Style preprocessors
  - [evanspearman/ComfyMath](https://github.com/evanspearman/ComfyMath) — math nodes for the workflow's keyframe path
  - [Smirnov75/ComfyUI-mxToolkit](https://github.com/Smirnov75/ComfyUI-mxToolkit) — utility nodes
  - [DoctorDiffusion/ComfyUI-MediaMixer](https://github.com/DoctorDiffusion/ComfyUI-MediaMixer) — `FinalFrameSelector`
- **[HF Spaces ZeroGPU](https://huggingface.co/zero-gpu)** (A10G) — `@spaces.GPU(duration=…)` for queue-priority signalling and per-call timeout

---

## Design

Theme: **Topaz Cinema Slate** — slate substrate `#1A1F26`, warm amber accent `#E0A458` used sparingly, IBM Plex Sans throughout. Defined as `_TOPAZ_THEME` + `_CUSTOM_CSS` in `app.py`.

Layout: hamburger drawer. Pinned 220 px sidebar at ≥1024 px (mode buttons + model status + settings); below 1024 px it slides in as a fixed overlay via the `.aio-shell.drawer-open` class. The header carries a live mode tag (T2V/A2V/I2V/LIPSYNC/KEY/STYLE) updated by JS without a server round-trip.

Spec, plan, and design rationale live under `docs/superpowers/specs/` and `docs/superpowers/plans/`.

---

## Notes on running

- **First inference is slow.** Cold-start workflow validation + model load on the active node graph takes ~30 – 90 s. Subsequent calls within the same session reuse loaded models.
- **VRAM tier** is auto-detected; override with `LTX23_AIO_VRAM=lowvram|normalvram|highvram`.
- **ZeroGPU duration cap.** The per-call estimator clamps to `[60, 900] s`. If a generation aborts with `"GPU task aborted"`, the handler retries once at 2× duration. The duration field is the queue-priority signal, not a billing cap.
- **Output directory.** Local: `comfyui/output/LTX2.3/`. Spaces: `~/comfyui/output/LTX2.3/`. Both are whitelisted via `allowed_paths=` on launch (Gradio 5 file-access policy).
- **Local LAN testing.** Bound to `0.0.0.0:7860`. macOS firewall: allow inbound for `python` if a connection from your phone refuses.

---

## License

MIT for the AIO app code (see `LICENSE`).

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) is GPL-3.0.
- LTX-2.3 and Lightricks-published LoRAs / auxiliaries retain Lightricks' open-source licensing — see the individual model cards on Hugging Face.
- Gemma 3 weights are subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
- Each pinned custom node retains its own license; see the linked repositories.

## Credits

- **LTX-2.3** by [Lightricks](https://github.com/Lightricks)
- **ComfyUI** by [comfyanonymous](https://github.com/comfyanonymous)
- **Gemma 3** by [Google DeepMind](https://github.com/google-deepmind)
- **All-In-One ComfyUI workflow** that this app wraps — by [Danielle Falco](https://www.youtube.com/@FutuTek) (FutuTek)
- **Workflow nodes** by Lightricks, [kijai](https://github.com/kijai), [rgthree](https://github.com/rgthree), [Kosinkadink](https://github.com/Kosinkadink), [pythongosssss](https://github.com/pythongosssss), [city96](https://github.com/city96), [Fannovel16](https://github.com/Fannovel16), [evanspearman](https://github.com/evanspearman), [Smirnov75](https://github.com/Smirnov75), [DoctorDiffusion](https://github.com/DoctorDiffusion)

Built by [@techfreakworm](https://huggingface.co/techfreakworm) — drop a ♥ on the [Space](https://huggingface.co/spaces/techfreakworm/LTX2.3-Studio) if it's useful, and follow there for what's next.
