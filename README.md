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

# LTX 2.3 All-in-One Video Generator

A Gradio app for [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) wrapping all six modes of the official ComfyUI All-In-One workflow under a single, focused UI. Runs locally on Apple Silicon (MPS) or NVIDIA (CUDA), and deploys to Hugging Face Spaces (ZeroGPU).

## Modes

1. **Text → Video** (+ optional Audio)
2. **Audio → Video** (Text + Audio → Video + Audio)
3. **Image → Video** (+ optional Audio)
4. **Lipsync** (Image + Audio → Video + Audio)
5. **First / Last Frame → Video** (keyframe interpolation)
6. **Style Transfer** (Video → Video, motion control)

## Local quickstart

Requires Python 3.11, ~80 GB free disk for model weights, and ~24 GB+ GPU memory (CUDA) or 32 GB+ unified memory (Apple Silicon).

```bash
git clone --recurse-submodules https://github.com/<your-handle>/ltx2.3-AIO-generator
cd ltx2.3-AIO-generator
bash setup.sh
source .venv/bin/activate
python app.py
```

The first run downloads ~70 GB of models into your existing `~/.cache/huggingface/hub` (no duplicate copies in this repo) and symlinks them into `comfyui/models/`.

## HF Spaces deployment

This repo is a Gradio Space. The required Pro tier provides ~50 GB persistent `/data` storage and longer per-call ZeroGPU budgets needed for Balanced and Quality presets.

```bash
git remote add space https://huggingface.co/spaces/<your-handle>/ltx2.3-aio
git push space main
```

## License

MIT for the AIO app code. ComfyUI and LTX-2.3 retain their respective licenses.
