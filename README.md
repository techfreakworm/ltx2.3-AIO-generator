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
hf_oauth: true
hf_oauth_expiration_minutes: 480
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
