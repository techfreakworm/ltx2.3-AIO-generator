# Future improvements

A backlog of optimizations that aren't blocking but would tighten the deploy.
None of these are required for current functionality. Order is rough priority,
not commitment.

## Spaces / preload

### ~~0. Re-enable `preload_from_hub` via runtime cache mirror~~ — DONE 2026-05-02

Initial preload deployment failed because HF's build pipeline writes
`~/.cache/huggingface/` as the build user, leaving it read-only for runtime
user 1000. Lazy `hf_hub_download` for non-preloaded files (GGUF, camera LoRAs)
failed with `Permission denied (os error 13)`. `chmod` couldn't help — we
don't own the inode.

Fix landed in `_bootstrap()`'s `_mirror_preload_hf_cache()`:
- Walks `~/.cache/huggingface/` to a parallel `~/hf-cache-rw/` we own
- Hardlinks `blobs/<sha>` files (zero-copy, shared inode, instant reads)
- Preserves relative snapshot symlinks (resolve within the mirror tree)
- Byte-copies `refs/<branch>` files (HF lib overwrites these on etag check)
- Sets `HF_HOME` + `HF_HUB_CACHE` to the mirror so HF lib uses our writable copy
- Falls back to symlink if `os.link()` returns EXDEV (cross-device)

Result: preloaded files are instantly available (cache hit on first generate),
non-preloaded files lazy-download into dirs we own (no permission errors).

### ~~1. Stop preloading models that aren't referenced by any workflow~~ — DONE 2026-05-02

Audit on 2026-05-02 showed two `Lightricks/LTX-2.3` files in `preload_from_hub`
that aren't actually referenced by any workflow JSON we ship:

- `ltx-2.3-22b-dev.safetensors` (~42 GB)
- `ltx-2.3-22b-distilled.safetensors` (~42 GB)

The active path uses `Kijai/LTX2.3_comfy ltx-2.3-22b-dev_transformer_only_bf16.safetensors`.
Removed both — ~84 GB saved. Forced by HF eviction with `storage limit
exceeded (150G)` when total preload was ~234 GB. Risk: if a future workflow
update reintroduces the Lightricks-side filenames, lazy download takes over.

### ~~2. Drop `unsloth/LTX-2.3-GGUF` from preload (~39 GB)~~ — DONE 2026-05-02

Removed alongside (1). GGUF transformer is the low-VRAM alternative; ZeroGPU
H200 has 70 GB so the BF16 transformer always fits. Lazy-loads on first use
of any preset that wires the GGUF path.

### 3. Drop the `Lightricks/LTX-2-19b-LoRA-Camera-Control-Static/Jib-Up/Jib-Down` preload

Each is ~2 GB. The Power Lora Loader has them all listed but defaults all to
`on: false`, so they only load when the user picks one. Lazy-load is
appropriate. Currently kept in preload because of the 10-entry cap +
"easier to keep what we had".

### 4. Auto-generate `preload_from_hub` from `MODEL_REGISTRY`

Today the README list and `MODEL_REGISTRY` in `models.py` can drift. Build a
small `tools/sync_preload.py` that:

1. Reads `MODEL_REGISTRY`
2. Walks the workflow JSONs to find which entries are actually referenced
3. Sorts referenced entries by size (using `huggingface_hub` `repo_info`)
4. Picks the top N entries that fit in the 10-cap
5. Writes them back into the README YAML

Run as a pre-commit or CI step.

### 5. Bake custom-node clones into the build via `requirements.txt` git installs

We currently `git clone` 10 custom-node repos in `_bootstrap()` at runtime.
That's ~30 s of cold start. Some custom nodes ship as pip-installable; for
the others, we could write a small `tools/install_custom_nodes.py` that
runs at build time (via `pip install --no-deps` against git URLs) so the
repos land in the image instead of being fetched at boot.

Tradeoff: Spaces' build pipeline runs the gradio SDK Dockerfile which we
don't control directly. The custom-node clone has to happen at runtime
unless we can move it into the standard `requirements.txt` build step.

### 6. Persistent storage add-on as the "$25/mo button"

If iteration speed becomes the binding constraint, the persistent storage
add-on (Spaces > Settings) at $25/mo for 150 GB makes everything just work
— `/data` is writable, models live there forever, no preload dance.
Sketched approach: `HF_HOME=/data/hf-cache` env var + `_bootstrap()` mkdir
fallback. One-line code change.

## Workflow / runtime

### 7. Move ComfyUI custom-node `requirements.txt` install to build time

Bootstrap currently `pip install`s each custom node's requirements at
runtime. Most are no-ops (deps already in our top-level `requirements.txt`)
but the `pip install --quiet` calls still take a few seconds each. Could
audit and just merge them into the top-level `requirements.txt`.

### 8. Clean up `nodes_replacements.py` warning

ComfyUI core at our pinned commit (`eb0686bb`) emits
`'function' object has no attribute 'register'` because the node-replacement
API surface is incomplete at that SHA. Bumping `COMFYUI_COMMIT` to a newer
tag should silence it. Pure cosmetic — no functional impact.

### 9. Auto-close drawer when user navigates away from header

Currently relies on document-level click listener. Works but has a
microsecond race when the click target is between elements. Could use
`pointerleave` on the drawer instead.

## Cost-of-running

### 10. Trim ZeroGPU duration cap

Currently `@spaces.GPU(duration=300)` reserves 5 min per call. For Fast preset
(distilled 8 steps) actual usage is ~30 s. Could shorten to 120 s — improves
queue priority for the user (per HF docs). Use dynamic duration based on
preset.

### 11. Local-perf "low-VRAM" path for style mode (GGUF Q4 transformer)

Style mode on Apple Silicon runs ~37× slower per sampling step than the other
modes (~596 s/step on Mac vs ~16 s/step for lipsync). Root cause is
architectural — `LTXAddVideoICLoRAGuide` concatenates the source video's
DWPose latents into the noisy target latent, doubling the attention sequence
to ~56 k tokens. Combined with MPS having no flash-attn-2 and the 22B BF16
model approaching the working-memory ceiling, perf collapses on Mac.

H200 handles this fine (flash-attn-3 + tensor cores + dedicated VRAM ⇒
~30–60 s end to end on Spaces). So this is fundamentally a Mac/MPS gap, not
a code bug.

A "Low VRAM" preset that swaps the BF16 transformer for the GGUF Q4
quantized one would reduce per-step memory pressure and may bring local
style perf into the workable range (still slow, but maybe ~60–90 s/step
instead of 600). The GGUF file is already declared in `MODEL_REGISTRY`
(`UnetLoaderGGUF` consumer). What's missing:

1. A workflow toggle that swaps `UNETLoader` → `UnetLoaderGGUF` for the main
   transformer in style.json (and other modes that benefit).
2. A UI control on the Advanced accordion: "Low VRAM (GGUF Q4)".
3. Wire-through in `_style_parameterize` (and friends) to flip the loader
   class.
4. Delete the matching BF16 path nodes when GGUF is selected (or set them
   to bypass) so we don't load both.

Risk: GGUF transformers behave slightly differently from BF16 — output
quality drops, especially for IC-LoRA paths where the dynamic range matters.
Should be opt-in only, never default. Probably v1.1+ scope (it's listed in
"Out of scope for v1" in CLAUDE.md as the GGUF Q4 / Low VRAM preset).
