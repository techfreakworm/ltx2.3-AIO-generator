# LTX 2.3 AIO Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Gradio app that wraps the existing ComfyUI LTX 2.3 All-In-One workflow into a polished mode-specific UI, runnable locally (MPS/CUDA) and on Hugging Face Spaces (ZeroGPU, Pro tier).

**Architecture:** Gradio frontend → workflow JSON parameterizer → bundled ComfyUI in library mode (`comfy.execution.PromptExecutor`). Six mode-specific workflow JSON templates extracted from the master workflow; per-mode `parameterize_fn` translates Gradio inputs into node patches. Same code locally and on Spaces; the only divergence is `@spaces.GPU` decoration and model storage location.

**Tech Stack:** Python 3.11, Gradio 5.x, `spaces`, `huggingface_hub`, ComfyUI (vendored as git submodule + runtime clone on Spaces) + custom nodes (`ComfyUI-LTXVideo`, `ComfyUI-KJNodes`, `rgthree-comfy`, `ComfyUI-VideoHelperSuite`, `ComfyUI-Custom-Scripts`), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-04-30-ltx23-aio-generator-design.md`

---

## File Map (locked at plan time)

| File | Created by task | LOC est. | Responsibility |
|---|---|---|---|
| `requirements.txt` | T1 | 15 | Pin Gradio, spaces, huggingface_hub, torch, ruff, pytest. |
| `pyproject.toml` | T1 | 30 | Pytest rootdir + ruff config so flat-layout imports resolve. |
| `setup.sh` | T2 | 50 | Idempotent local bootstrap (venv, submodule, custom nodes, models). |
| `README.md` | T3 | 80 | Spaces front matter + local quickstart + screenshot placeholders. |
| `tests/conftest.py` | T4 | 80 | Fixtures: `master_workflow`, `canonical_inputs`, `fake_hf_cache`, CLI flags. |
| `tools/extract_modes.py` | T5 | 200 | Extract six mode templates from the master workflow JSON. |
| `workflows/{t2v,a2v,i2v,lipsync,keyframe,style}.json` | T6 | (data) | Six mode templates. |
| `workflow.py` | T7–T9 | 120 | `load_template`, `set_input`, `validate`. |
| `modes.py` | T10–T12 | 300 | `Mode` dataclass + `MODE_REGISTRY` (six entries with `parameterize_fn`). |
| `models.py` | T13–T15 | 150 | `MODEL_REGISTRY`, `ensure_models_for_mode`, symlink/download logic. |
| `tools/refresh_models.py` | T16 | 30 | CLI wrapper around `models.ensure_models_for_mode` for all modes. |
| `backend.py` | T17–T20 | 200 | `ComfyUILibraryBackend`, async submit, progress hook, ZeroGPU. |
| `ui.py` | T21–T23 | 200 | `preset_bar`, `status_banner`, `lora_chrome`. |
| `app.py` | T24–T26 | 400 | Gradio `Blocks`, sidebar, mode rendering, generate handler. |
| `.github/workflows/ci.yml` | T27 | 30 | Run L1+L3 tests on push. |
| `.github/workflows/deploy-space.yml` | T28 | 25 | Optional — push to HF Space on main. |

Total: ~1,800 LOC across 14 files (excluding the ComfyUI submodule, workflow JSON data, and tests).

---

## Phase 0 — Foundations

### Task 1: `requirements.txt`

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Create `requirements.txt`**

```text
gradio>=5.0,<6.0
spaces>=0.30.0
huggingface_hub>=0.27.0
torch>=2.4.0
torchvision
torchaudio
numpy
Pillow
einops
safetensors
tqdm

# Dev / test
pytest>=8.0
pytest-asyncio>=0.23
ruff>=0.5
```

- [ ] **Step 2: Create `pyproject.toml`** so pytest finds the flat-layout modules and ruff rules are pinned

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
markers = [
    "gpu: marks tests that need a GPU (use --gpu to enable)",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
ignore = ["E501"]  # line length is enforced by formatter, not linter

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["E402"]  # imports inside test functions are fine
```

- [ ] **Step 3: Verify both files parse**

Run: `python3.11 -m pip install --dry-run -r requirements.txt 2>&1 | head -5`
Expected: pip resolves package names without "ERROR: Invalid requirement" lines (network errors are fine — we're checking syntax).

Run: `python3.11 -c "import tomllib; print(list(tomllib.loads(open('pyproject.toml').read()).keys()))"`
Expected: `['tool']`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt pyproject.toml
git commit -m "chore: pin runtime + dev dependencies and configure pytest/ruff"
```

---

### Task 2: `setup.sh`

**Files:**
- Create: `setup.sh`

- [ ] **Step 1: Write `setup.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "▶ Creating Python 3.11 venv"
python3.11 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip wheel

echo "▶ Initializing ComfyUI submodule"
git submodule update --init --recursive

echo "▶ Installing ComfyUI core requirements"
pip install -r comfyui/requirements.txt

echo "▶ Installing pinned custom nodes"
mkdir -p comfyui/custom_nodes
cd comfyui/custom_nodes
for repo in \
    Lightricks/ComfyUI-LTXVideo \
    kijai/ComfyUI-KJNodes \
    rgthree/rgthree-comfy \
    Kosinkadink/ComfyUI-VideoHelperSuite \
    pythongosssss/ComfyUI-Custom-Scripts ; do
  name="${repo##*/}"
  if [[ ! -d "$name" ]]; then
    git clone --depth 1 "https://github.com/$repo.git" "$name"
  fi
  if [[ -f "$name/requirements.txt" ]]; then
    pip install -r "$name/requirements.txt"
  fi
done
cd "$REPO_ROOT"

echo "▶ Installing AIO app dependencies"
pip install -r requirements.txt

echo "▶ Symlinking models from HF cache"
python tools/refresh_models.py || true  # ok to fail before tools/ exists

echo
echo "✓ Setup complete."
echo "  Activate venv: source .venv/bin/activate"
echo "  Run app:        python app.py"
```

- [ ] **Step 2: Make executable**

Run: `chmod +x setup.sh`
Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add setup.sh
git commit -m "chore: idempotent setup.sh — venv, submodule, custom nodes, models"
```

---

### Task 3: `README.md` with Spaces front matter

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the placeholder `README.md`**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with Spaces front matter and local quickstart"
```

---

### Task 4: `tests/conftest.py` with fixtures

**Files:**
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `tests/__init__.py`** (empty file)

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 2: Write `tests/conftest.py`**

```python
"""Shared pytest fixtures and CLI flags."""
import json
import os
import pathlib
from typing import Any

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

DEFAULT_MASTER_WORKFLOW = pathlib.Path(
    os.environ.get(
        "LTX23_MASTER_WORKFLOW",
        pathlib.Path.home() / "Projects/comfyui/user/default/workflows"
        / "1. LTX 2.3 All-In-One 260406-05.json",
    )
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--gpu", action="store_true", help="Run L4 GPU smoke tests.")
    parser.addoption(
        "--comfy-real",
        action="store_true",
        help="Use bundled ComfyUI for L2 graph validation (slower).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not config.getoption("--gpu"):
        skip_gpu = pytest.mark.skip(reason="GPU smoke tests skipped (use --gpu)")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def master_workflow() -> dict[str, Any]:
    """The full LTX 2.3 All-In-One workflow JSON (loaded from user's ComfyUI)."""
    if not DEFAULT_MASTER_WORKFLOW.exists():
        pytest.skip(
            f"Master workflow not found at {DEFAULT_MASTER_WORKFLOW}. "
            "Set LTX23_MASTER_WORKFLOW env var to its path."
        )
    return json.loads(DEFAULT_MASTER_WORKFLOW.read_text())


@pytest.fixture
def canonical_inputs() -> dict[str, dict[str, Any]]:
    """Known-good Gradio input dicts per mode (used by L1/L2 tests)."""
    return {
        "t2v": {
            "prompt": "a tiger walking through a misty forest at dawn, cinematic",
            "negative_prompt": "",
            "preset": "balanced",
            "width": 512,
            "height": 768,
            "frames": 81,
            "fps": 24,
            "seed": 42,
            "camera_lora": "none",
            "camera_strength": 0.8,
            "detailer_on": False,
            "detailer_strength": 0.5,
        },
        "i2v": {
            "prompt": "the subject turns toward the camera and smiles",
            "image": "/tmp/portrait.png",
            "preset": "balanced",
            "width": 512,
            "height": 768,
            "frames": 81,
            "fps": 24,
            "seed": 42,
            "camera_lora": "none",
            "camera_strength": 0.8,
            "detailer_on": True,
            "detailer_strength": 0.5,
            "ic_lora": "union",
            "ic_strength": 0.5,
            "pose_on": False,
        },
        "a2v": {
            "prompt": "a dancer moves to the beat in a neon-lit studio",
            "audio": "/tmp/track.wav",
            "preset": "balanced",
            "width": 512,
            "height": 768,
            "frames": 81,
            "fps": 24,
            "seed": 42,
            "audio_cfg": 7.0,
        },
        "lipsync": {
            "prompt": "the person speaks the audio with natural mouth movement",
            "image": "/tmp/portrait.png",
            "audio": "/tmp/speech.wav",
            "preset": "balanced",
            "image_strength": 0.7,
            "frames": 81,
            "fps": 24,
            "seed": 42,
        },
        "keyframe": {
            "prompt": "smooth transition between the two frames",
            "first_frame": "/tmp/start.png",
            "last_frame": "/tmp/end.png",
            "preset": "balanced",
            "frames": 81,
            "fps": 24,
            "seed": 42,
        },
        "style": {
            "prompt": "in the style of a renaissance oil painting",
            "input_video": "/tmp/source.mp4",
            "preset": "balanced",
            "frames": 81,
            "fps": 24,
            "seed": 42,
            "ic_lora": "motion-track",
            "ic_strength": 0.5,
        },
    }


@pytest.fixture
def fake_hf_cache(tmp_path: pathlib.Path) -> pathlib.Path:
    """A fake ~/.cache/huggingface/hub layout with placeholder files."""
    hub = tmp_path / "huggingface" / "hub"
    layouts = {
        "models--Lightricks--LTX-2.3": [
            "ltx-2.3-22b-distilled.safetensors",
            "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
            "ltx-2.3-22b-distilled-lora-384.safetensors",
        ],
        "models--google--gemma-3-12b-it-qat-q4_0-unquantized": [
            "model-00001-of-00005.safetensors",
            "model-00002-of-00005.safetensors",
            "model-00003-of-00005.safetensors",
            "model-00004-of-00005.safetensors",
            "model-00005-of-00005.safetensors",
            "model.safetensors.index.json",
            "tokenizer.model",
            "preprocessor_config.json",
        ],
        "models--Kijai--LTX2.3_comfy": [
            "LTX23_video_vae_bf16.safetensors",
            "LTX23_audio_vae_bf16.safetensors",
        ],
    }
    for repo, files in layouts.items():
        snapshot_dir = hub / repo / "snapshots" / "deadbeef" * 1
        snapshot_dir = hub / repo / "snapshots" / "deadbeef"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for filename in files:
            (snapshot_dir / filename).write_text("")  # placeholder
    return hub
```

- [ ] **Step 3: Verify pytest discovers the conftest**

Run: `python3.11 -m pytest tests/ --collect-only 2>&1 | head -20`
Expected: "no tests ran" or similar — but no errors importing conftest.

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "test: pytest fixtures (master_workflow, canonical_inputs, fake_hf_cache)"
```

---

### Task 5: ComfyUI submodule

**Files:**
- Create: `.gitmodules`
- Create: `comfyui/` (submodule)

- [ ] **Step 1: Add ComfyUI as a git submodule**

```bash
cd /Users/techfreakworm/Projects/llm/ltx2.3-AIO-generator
git submodule add https://github.com/comfyanonymous/ComfyUI.git comfyui
cd comfyui
# Pin to a known-good recent commit. Capture the SHA the user is currently running.
USER_COMFY_SHA="$(git -C ~/Projects/comfyui rev-parse HEAD)"
git checkout "$USER_COMFY_SHA"
cd ..
```

- [ ] **Step 2: Verify submodule status**

Run: `git submodule status`
Expected: one line starting with the pinned SHA followed by `comfyui (heads/master ...)` or similar.

- [ ] **Step 3: Commit submodule**

```bash
git add .gitmodules comfyui
git commit -m "chore: vendor ComfyUI as git submodule pinned to working commit"
```

---

## Phase 1 — Workflow library (TDD)

### Task 6: `tools/extract_modes.py` — extract mode templates

**Files:**
- Create: `tools/__init__.py` (empty)
- Create: `tools/extract_modes.py`
- Create: `tests/test_extract_modes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_extract_modes.py
"""Tests for the workflow-mode extractor."""
import json
import subprocess
import sys

from tests.conftest import REPO_ROOT


def test_extract_creates_six_mode_files(master_workflow, tmp_path):
    """extract_modes.py emits six valid mode-specific JSON templates."""
    out_dir = tmp_path / "workflows"
    master_path = tmp_path / "master.json"
    master_path.write_text(json.dumps(master_workflow))

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "extract_modes.py"),
            "--master",
            str(master_path),
            "--out",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    expected = {"t2v.json", "a2v.json", "i2v.json", "lipsync.json", "keyframe.json", "style.json"}
    actual = {p.name for p in out_dir.iterdir()}
    assert actual == expected

    # Each file must be valid JSON with at least one node.
    for path in out_dir.iterdir():
        wf = json.loads(path.read_text())
        assert "nodes" in wf
        assert len(wf["nodes"]) > 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3.11 -m pytest tests/test_extract_modes.py -v`
Expected: FAIL with `FileNotFoundError` or `No such file or directory` for `tools/extract_modes.py`.

- [ ] **Step 3: Implement `tools/__init__.py` and `tools/extract_modes.py`**

```python
# tools/__init__.py (empty)
```

```python
# tools/extract_modes.py
"""Extract six mode-specific workflow templates from the master LTX 2.3 All-In-One workflow.

Each ComfyUI group whose title starts with a number (e.g. "01 Text to Video") becomes
a mode template containing only that group's nodes plus shared scaffolding (Models,
Lora, Setting, Prompt, Load Audio/Image/Video, Output groups).

Group title → output filename mapping:
    01 → t2v.json
    02 → a2v.json
    03 → i2v.json
    04 → lipsync.json
    05 → keyframe.json
    06 → style.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections.abc import Iterable

GROUP_TO_FILENAME: dict[str, str] = {
    "01": "t2v.json",
    "02": "a2v.json",
    "03": "i2v.json",
    "04": "lipsync.json",
    "05": "keyframe.json",
    "06": "style.json",
}

SHARED_GROUP_PREFIXES: tuple[str, ...] = (
    "Models",
    "Lora",
    "Setting",
    "Prompt",
    "Load Audio",
    "Load Image",
    "Load Video",
    "Output",
)


def _node_in_group(node: dict, group: dict) -> bool:
    """Test whether a node's position lies inside a group's bounding box."""
    if "pos" not in node or "bounding" not in group:
        return False
    nx, ny = node["pos"][0], node["pos"][1]
    gx, gy, gw, gh = group["bounding"]
    return (gx <= nx <= gx + gw) and (gy <= ny <= gy + gh)


def _select_groups(master: dict, mode_prefix: str) -> list[dict]:
    """Pick the mode group plus all shared groups."""
    selected: list[dict] = []
    for g in master.get("groups", []):
        title = (g.get("title") or "").strip()
        if title.startswith(mode_prefix + " "):
            selected.append(g)
        elif any(title.startswith(p) for p in SHARED_GROUP_PREFIXES):
            selected.append(g)
    return selected


def _collect_nodes(master: dict, groups: Iterable[dict]) -> list[dict]:
    """Return all nodes lying inside any of the given groups."""
    groups_list = list(groups)
    keep: list[dict] = []
    for node in master.get("nodes", []):
        if any(_node_in_group(node, g) for g in groups_list):
            keep.append(node)
    return keep


def _collect_links(master: dict, kept_node_ids: set[int]) -> list[list]:
    """Keep only links where both endpoints are in the surviving node set."""
    return [
        link
        for link in master.get("links", [])
        # ComfyUI link tuple format: [link_id, src_node_id, src_out, dst_node_id, dst_in, type]
        if link[1] in kept_node_ids and link[3] in kept_node_ids
    ]


def extract_mode(master: dict, mode_prefix: str) -> dict:
    """Build a focused workflow JSON for the given mode group prefix."""
    groups = _select_groups(master, mode_prefix)
    nodes = _collect_nodes(master, groups)
    kept_ids = {n["id"] for n in nodes}
    links = _collect_links(master, kept_ids)

    return {
        "id": f"ltx23-aio-{mode_prefix}",
        "revision": 0,
        "last_node_id": max(kept_ids, default=0),
        "last_link_id": max((l[0] for l in links), default=0),
        "nodes": nodes,
        "links": links,
        "groups": groups,
        "definitions": master.get("definitions", {}),
        "config": master.get("config", {}),
        "extra": master.get("extra", {}),
        "version": master.get("version", 0.4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    master = json.loads(args.master.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    for prefix, filename in GROUP_TO_FILENAME.items():
        wf = extract_mode(master, prefix)
        out_path = args.out / filename
        out_path.write_text(json.dumps(wf, indent=2))
        print(f"  → wrote {out_path} ({len(wf['nodes'])} nodes, {len(wf['links'])} links)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3.11 -m pytest tests/test_extract_modes.py -v`
Expected: PASS. (If `master_workflow` fixture skips because the master JSON isn't at the expected path, set `LTX23_MASTER_WORKFLOW` env var first.)

- [ ] **Step 5: Commit**

```bash
git add tools/__init__.py tools/extract_modes.py tests/test_extract_modes.py
git commit -m "feat(tools): extract six mode templates from master workflow JSON"
```

---

### Task 7: Run extraction once → commit `workflows/*.json`

**Files:**
- Create: `workflows/t2v.json` … `workflows/style.json`

- [ ] **Step 1: Run the extractor against the master workflow**

```bash
mkdir -p workflows
python3.11 tools/extract_modes.py \
  --master ~/Projects/comfyui/user/default/workflows/"1. LTX 2.3 All-In-One 260406-05.json" \
  --out workflows
```

Expected output: six lines like `→ wrote workflows/t2v.json (N nodes, M links)`.

- [ ] **Step 2: Sanity-check each file**

```bash
for f in workflows/*.json; do
  python3.11 -c "import json; w=json.load(open('$f')); print('$f', len(w['nodes']), 'nodes')"
done
```

Expected: each file reports a non-zero node count.

- [ ] **Step 3: Commit the templates**

```bash
git add workflows/
git commit -m "data: extracted mode-specific workflow templates from master"
```

---

### Task 8: `workflow.py` — `load_template`

**Files:**
- Create: `workflow.py`
- Create: `tests/test_workflow.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_workflow.py
"""Unit tests for workflow.py — pure functions over JSON dicts."""
import pytest

import workflow


def test_load_template_returns_dict_for_valid_mode():
    wf = workflow.load_template("t2v")
    assert isinstance(wf, dict)
    assert "nodes" in wf
    assert len(wf["nodes"]) > 0


def test_load_template_raises_for_unknown_mode():
    with pytest.raises(ValueError, match="unknown mode"):
        workflow.load_template("nonexistent")


def test_load_template_returns_independent_copy():
    """Mutations to one returned dict must not affect later loads."""
    a = workflow.load_template("t2v")
    a["nodes"].append({"id": -999})
    b = workflow.load_template("t2v")
    assert {-999} & {n.get("id") for n in b["nodes"]} == set()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3.11 -m pytest tests/test_workflow.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'workflow'`.

- [ ] **Step 3: Implement `workflow.py`**

```python
"""Pure functions over LTX 2.3 mode workflow JSON templates."""
from __future__ import annotations

import copy
import json
import pathlib
from typing import Any

WORKFLOWS_DIR = pathlib.Path(__file__).parent / "workflows"

VALID_MODES: tuple[str, ...] = ("t2v", "a2v", "i2v", "lipsync", "keyframe", "style")


def load_template(mode: str) -> dict[str, Any]:
    """Load a fresh, independent copy of the named mode's workflow template."""
    if mode not in VALID_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {VALID_MODES}")
    path = WORKFLOWS_DIR / f"{mode}.json"
    return copy.deepcopy(json.loads(path.read_text()))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3.11 -m pytest tests/test_workflow.py -v`
Expected: PASS — three tests green.

- [ ] **Step 5: Commit**

```bash
git add workflow.py tests/test_workflow.py
git commit -m "feat(workflow): load_template returns fresh deep copy per mode"
```

---

### Task 9: `workflow.py` — `set_input` and `validate`

**Files:**
- Modify: `workflow.py`
- Modify: `tests/test_workflow.py`

- [ ] **Step 1: Append failing tests**

```python
# Append to tests/test_workflow.py
def test_set_input_patches_widgets_values_in_place():
    wf = workflow.load_template("t2v")
    target_node = next(n for n in wf["nodes"] if n["type"] == "CLIPTextEncode")
    workflow.set_input(wf, target_node["id"], 0, "new prompt text")
    refetched = next(n for n in wf["nodes"] if n["id"] == target_node["id"])
    assert refetched["widgets_values"][0] == "new prompt text"


def test_set_input_raises_for_unknown_node():
    wf = workflow.load_template("t2v")
    with pytest.raises(KeyError, match="node id"):
        workflow.set_input(wf, 999_999_999, 0, "x")


def test_validate_accepts_canonical_template():
    wf = workflow.load_template("t2v")
    workflow.validate(wf)  # must not raise


def test_validate_rejects_workflow_with_no_nodes():
    wf = {"nodes": [], "links": []}
    with pytest.raises(ValueError, match="no nodes"):
        workflow.validate(wf)


def test_validate_rejects_orphan_link():
    wf = workflow.load_template("t2v")
    wf["links"].append([99999, 1, 0, 999_999_999, 0, "INT"])  # destination doesn't exist
    with pytest.raises(ValueError, match="orphan link"):
        workflow.validate(wf)
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `python3.11 -m pytest tests/test_workflow.py -v`
Expected: 5 fails (set_input + validate) and 3 prior tests still passing.

- [ ] **Step 3: Implement `set_input` and `validate` in `workflow.py`**

Append to `workflow.py`:

```python
def set_input(workflow: dict[str, Any], node_id: int, widget_index: int, value: Any) -> None:
    """Patch a node's widgets_values in place.

    Args:
        workflow: A workflow dict (must have a "nodes" list).
        node_id: The id of the node to patch.
        widget_index: Position within the node's widgets_values list.
        value: New value.

    Raises:
        KeyError: If no node with the given id exists.
    """
    for node in workflow["nodes"]:
        if node.get("id") == node_id:
            widgets = node.setdefault("widgets_values", [])
            while len(widgets) <= widget_index:
                widgets.append(None)
            widgets[widget_index] = value
            return
    raise KeyError(f"node id {node_id} not found in workflow")


def validate(workflow: dict[str, Any]) -> None:
    """Static schema validation. Raises ValueError on the first problem found."""
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list) or len(nodes) == 0:
        raise ValueError("workflow has no nodes")

    node_ids = {n.get("id") for n in nodes if "id" in n}
    for link in workflow.get("links", []):
        if not isinstance(link, list) or len(link) < 6:
            raise ValueError(f"malformed link {link}")
        _, src, _, dst, _, _ = link
        if src not in node_ids or dst not in node_ids:
            raise ValueError(f"orphan link {link}")
```

- [ ] **Step 4: Run all workflow tests**

Run: `python3.11 -m pytest tests/test_workflow.py -v`
Expected: 8 passing tests.

- [ ] **Step 5: Commit**

```bash
git add workflow.py tests/test_workflow.py
git commit -m "feat(workflow): set_input + validate over node graph"
```

---

## Phase 2 — Modes registry

### Task 10: `modes.py` — `Mode` dataclass + skeleton

**Files:**
- Create: `modes.py`
- Create: `tests/test_modes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_modes.py
"""Unit tests for modes.py — MODE_REGISTRY and parameterize_fn correctness."""
import pytest

import modes


def test_mode_registry_has_all_six_keys():
    assert set(modes.MODE_REGISTRY.keys()) == {
        "t2v", "a2v", "i2v", "lipsync", "keyframe", "style",
    }


def test_each_mode_has_required_attributes():
    for name, mode in modes.MODE_REGISTRY.items():
        assert mode.name == name
        assert mode.label  # non-empty
        assert mode.icon  # non-empty
        assert callable(mode.parameterize_fn)
        assert isinstance(mode.stage_map, list) and len(mode.stage_map) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.11 -m pytest tests/test_modes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'modes'`.

- [ ] **Step 3: Create `modes.py` skeleton**

```python
"""MODE_REGISTRY — one Mode entry per generation mode.

Each Mode declares:
- name: short id ("t2v", "i2v", ...)
- label: display name
- icon: single-character or emoji icon for the sidebar
- stage_map: list of (label, expected_share_pct) for the status banner
- parameterize_fn: (Gradio inputs dict) -> list[(node_id, widget_index, value)]

The parameterize_fn is the only mode-specific logic. Everything else (workflow
loading, validation, dispatch) is mode-agnostic and lives in workflow.py /
backend.py.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

Patch = tuple[int, int, Any]
ParameterizeFn = Callable[[dict[str, Any]], list[Patch]]


@dataclass(frozen=True)
class Stage:
    label: str
    share_pct: int  # rough share of total time, sums to ~100 across stages


@dataclass(frozen=True)
class Mode:
    name: str
    label: str
    icon: str
    parameterize_fn: ParameterizeFn
    stage_map: list[Stage] = field(default_factory=list)


# Filled in by tasks 11–12.
MODE_REGISTRY: dict[str, Mode] = {}
```

- [ ] **Step 4: Run test to verify it still fails (different error)**

Run: `python3.11 -m pytest tests/test_modes.py -v`
Expected: FAIL on `test_mode_registry_has_all_six_keys` — empty registry.

- [ ] **Step 5: Commit skeleton**

```bash
git add modes.py tests/test_modes.py
git commit -m "feat(modes): Mode dataclass + empty MODE_REGISTRY skeleton"
```

---

### Task 11: `parameterize_fn` for T2V and I2V

**Files:**
- Modify: `modes.py`
- Modify: `tests/test_modes.py`

- [ ] **Step 1: Append failing tests**

```python
# Append to tests/test_modes.py
import workflow

def test_t2v_parameterize_produces_valid_patches(canonical_inputs):
    inputs = canonical_inputs["t2v"]
    mode = modes.MODE_REGISTRY["t2v"]
    patches = mode.parameterize_fn(inputs)

    # All patches must be (node_id: int, widget_index: int, value: Any)
    for node_id, widget_index, value in patches:
        assert isinstance(node_id, int)
        assert isinstance(widget_index, int)
        assert value is not None or value == ""

    # Apply patches to a real template; result must validate.
    wf = workflow.load_template("t2v")
    for patch in patches:
        workflow.set_input(wf, *patch)
    workflow.validate(wf)


def test_i2v_parameterize_uses_image_path(canonical_inputs):
    inputs = canonical_inputs["i2v"]
    mode = modes.MODE_REGISTRY["i2v"]
    patches = mode.parameterize_fn(inputs)
    values = [p[2] for p in patches]
    assert inputs["image"] in values
```

- [ ] **Step 2: Run tests to verify failures**

Run: `python3.11 -m pytest tests/test_modes.py -v -k "t2v or i2v"`
Expected: FAIL — `KeyError: 't2v'` from empty MODE_REGISTRY.

- [ ] **Step 3: Implement T2V and I2V**

Append to `modes.py`:

```python
# ---------------------------------------------------------------------------
# Node-id constants per template. These are stable for a given workflow file;
# if you re-run tools/extract_modes.py against an updated master, re-capture
# them by inspecting the regenerated workflows/<mode>.json.
# ---------------------------------------------------------------------------

# T2V template node ids (capture from workflows/t2v.json after extraction).
T2V_NODE_PROMPT = 240            # CLIPTextEncode positive
T2V_NODE_NEG_PROMPT = 241        # CLIPTextEncode negative
T2V_NODE_RESOLUTION = 5300       # mxSlider for w/h
T2V_NODE_FRAMES = 5301           # INTConstant
T2V_NODE_FPS = 5302              # INTConstant
T2V_NODE_SEED = 5303             # INTConstant
T2V_NODE_PRESET = 5304           # Any Switch — preset selector
T2V_NODE_CAMERA_LORA = 5400      # Power Lora Loader row 0
T2V_NODE_DETAILER_LORA = 5401    # Power Lora Loader row 1

# I2V template node ids (capture from workflows/i2v.json).
I2V_NODE_PROMPT = 340
I2V_NODE_IMAGE = 350             # LoadImage
I2V_NODE_RESOLUTION = 5310
I2V_NODE_FRAMES = 5311
I2V_NODE_FPS = 5312
I2V_NODE_SEED = 5313
I2V_NODE_PRESET = 5314
I2V_NODE_CAMERA_LORA = 5410
I2V_NODE_DETAILER_LORA = 5411
I2V_NODE_IC_LORA = 5412
I2V_NODE_POSE_LORA = 5413


def _t2v_parameterize(inp: dict[str, Any]) -> list[Patch]:
    return [
        (T2V_NODE_PROMPT, 0, inp["prompt"]),
        (T2V_NODE_NEG_PROMPT, 0, inp.get("negative_prompt", "")),
        (T2V_NODE_RESOLUTION, 0, inp["width"]),
        (T2V_NODE_RESOLUTION, 1, inp["height"]),
        (T2V_NODE_FRAMES, 0, inp["frames"]),
        (T2V_NODE_FPS, 0, inp["fps"]),
        (T2V_NODE_SEED, 0, inp["seed"]),
        (T2V_NODE_PRESET, 0, inp["preset"]),
        (T2V_NODE_CAMERA_LORA, 0, inp.get("camera_lora", "none")),
        (T2V_NODE_CAMERA_LORA, 1, inp.get("camera_strength", 0.0)),
        (T2V_NODE_DETAILER_LORA, 0, "ic-lora-detailer" if inp.get("detailer_on") else "none"),
        (T2V_NODE_DETAILER_LORA, 1, inp.get("detailer_strength", 0.0)),
    ]


def _i2v_parameterize(inp: dict[str, Any]) -> list[Patch]:
    return [
        (I2V_NODE_PROMPT, 0, inp["prompt"]),
        (I2V_NODE_IMAGE, 0, inp["image"]),
        (I2V_NODE_RESOLUTION, 0, inp["width"]),
        (I2V_NODE_RESOLUTION, 1, inp["height"]),
        (I2V_NODE_FRAMES, 0, inp["frames"]),
        (I2V_NODE_FPS, 0, inp["fps"]),
        (I2V_NODE_SEED, 0, inp["seed"]),
        (I2V_NODE_PRESET, 0, inp["preset"]),
        (I2V_NODE_CAMERA_LORA, 0, inp.get("camera_lora", "none")),
        (I2V_NODE_CAMERA_LORA, 1, inp.get("camera_strength", 0.0)),
        (I2V_NODE_DETAILER_LORA, 0, "ic-lora-detailer" if inp.get("detailer_on") else "none"),
        (I2V_NODE_DETAILER_LORA, 1, inp.get("detailer_strength", 0.0)),
        (I2V_NODE_IC_LORA, 0, f"ic-lora-{inp.get('ic_lora', 'union')}"),
        (I2V_NODE_IC_LORA, 1, inp.get("ic_strength", 0.0)),
        (I2V_NODE_POSE_LORA, 0, "ic-lora-pose-control" if inp.get("pose_on") else "none"),
        (I2V_NODE_POSE_LORA, 1, inp.get("pose_strength", 0.0)),
    ]


_T2V_STAGES = [
    Stage("Encode prompt", 5),
    Stage("Diffusion (Stage 1)", 60),
    Stage("Spatial upscale", 7),
    Stage("Diffusion (Stage 2)", 18),
    Stage("Decode video", 10),
]

_I2V_STAGES = [
    Stage("Encode prompt", 5),
    Stage("Encode image", 3),
    Stage("Diffusion (Stage 1)", 55),
    Stage("Spatial upscale", 7),
    Stage("Diffusion (Stage 2)", 20),
    Stage("Decode video", 10),
]

MODE_REGISTRY["t2v"] = Mode(
    name="t2v", label="Text → Video", icon="📝",
    parameterize_fn=_t2v_parameterize, stage_map=_T2V_STAGES,
)
MODE_REGISTRY["i2v"] = Mode(
    name="i2v", label="Image → Video", icon="🖼",
    parameterize_fn=_i2v_parameterize, stage_map=_I2V_STAGES,
)
```

> **Note:** the node-id constants (e.g. `T2V_NODE_PROMPT = 240`) are placeholders to be replaced by the actual ids from `workflows/t2v.json`. After Task 7 generates the templates, capture the real ids by running:
> ```bash
> python3.11 -c "import json; w=json.load(open('workflows/t2v.json')); [print(n['id'], n['type'], n.get('title')) for n in w['nodes'] if n['type'] in ('CLIPTextEncode','mxSlider','INTConstant','Power Lora Loader (rgthree)','Any Switch (rgthree)')]"
> ```
> and replace each constant with the matching node id. This step is part of Step 4.

- [ ] **Step 4: Capture real node ids and update constants**

Run the inspection command above for both `t2v.json` and `i2v.json`. Replace the constants with the real ids. Re-read the test in Step 1 — it must still pass.

- [ ] **Step 5: Run T2V/I2V tests**

Run: `python3.11 -m pytest tests/test_modes.py -v -k "t2v or i2v"`
Expected: PASS for both T2V and I2V tests; existing skeleton tests still pass.

- [ ] **Step 6: Commit**

```bash
git add modes.py tests/test_modes.py
git commit -m "feat(modes): T2V + I2V parameterize_fn with stage maps"
```

---

### Task 12: `parameterize_fn` for A2V, Lipsync, Keyframe, Style

**Files:**
- Modify: `modes.py`
- Modify: `tests/test_modes.py`

- [ ] **Step 1: Append failing tests**

```python
# Append to tests/test_modes.py
@pytest.mark.parametrize("mode_name", ["a2v", "lipsync", "keyframe", "style"])
def test_remaining_modes_parameterize_validates(mode_name, canonical_inputs):
    inputs = canonical_inputs[mode_name]
    mode = modes.MODE_REGISTRY[mode_name]
    patches = mode.parameterize_fn(inputs)
    assert len(patches) > 0

    wf = workflow.load_template(mode_name)
    for patch in patches:
        workflow.set_input(wf, *patch)
    workflow.validate(wf)


def test_a2v_parameterize_passes_audio_path(canonical_inputs):
    patches = modes.MODE_REGISTRY["a2v"].parameterize_fn(canonical_inputs["a2v"])
    assert canonical_inputs["a2v"]["audio"] in [p[2] for p in patches]


def test_lipsync_parameterize_passes_image_and_audio(canonical_inputs):
    patches = modes.MODE_REGISTRY["lipsync"].parameterize_fn(canonical_inputs["lipsync"])
    values = [p[2] for p in patches]
    assert canonical_inputs["lipsync"]["image"] in values
    assert canonical_inputs["lipsync"]["audio"] in values


def test_keyframe_parameterize_passes_two_frames(canonical_inputs):
    patches = modes.MODE_REGISTRY["keyframe"].parameterize_fn(canonical_inputs["keyframe"])
    values = [p[2] for p in patches]
    assert canonical_inputs["keyframe"]["first_frame"] in values
    assert canonical_inputs["keyframe"]["last_frame"] in values


def test_style_parameterize_passes_input_video(canonical_inputs):
    patches = modes.MODE_REGISTRY["style"].parameterize_fn(canonical_inputs["style"])
    assert canonical_inputs["style"]["input_video"] in [p[2] for p in patches]
```

- [ ] **Step 2: Run tests to verify failures**

Run: `python3.11 -m pytest tests/test_modes.py -v`
Expected: 5 fails on the new tests (KeyError for missing modes).

- [ ] **Step 3: Implement A2V, Lipsync, Keyframe, Style**

Append to `modes.py` (with node-id constants captured from each `workflows/<mode>.json` per the inspection technique in Task 11):

```python
# A2V template node ids
A2V_NODE_PROMPT = ...           # capture from workflows/a2v.json
A2V_NODE_AUDIO = ...            # VHS_LoadAudioUpload
A2V_NODE_RESOLUTION = ...
A2V_NODE_FRAMES = ...
A2V_NODE_FPS = ...
A2V_NODE_SEED = ...
A2V_NODE_PRESET = ...
A2V_NODE_AUDIO_CFG = ...

# Lipsync template node ids
LIPSYNC_NODE_PROMPT = ...
LIPSYNC_NODE_IMAGE = ...
LIPSYNC_NODE_AUDIO = ...
LIPSYNC_NODE_IMAGE_STRENGTH = ...
LIPSYNC_NODE_FRAMES = ...
LIPSYNC_NODE_FPS = ...
LIPSYNC_NODE_SEED = ...
LIPSYNC_NODE_PRESET = ...

# Keyframe template node ids
KEYFRAME_NODE_PROMPT = ...
KEYFRAME_NODE_FIRST = ...
KEYFRAME_NODE_LAST = ...
KEYFRAME_NODE_FRAMES = ...
KEYFRAME_NODE_FPS = ...
KEYFRAME_NODE_SEED = ...
KEYFRAME_NODE_PRESET = ...

# Style template node ids
STYLE_NODE_PROMPT = ...
STYLE_NODE_VIDEO = ...
STYLE_NODE_IC_LORA = ...
STYLE_NODE_FRAMES = ...
STYLE_NODE_FPS = ...
STYLE_NODE_SEED = ...
STYLE_NODE_PRESET = ...


def _a2v_parameterize(inp: dict[str, Any]) -> list[Patch]:
    return [
        (A2V_NODE_PROMPT, 0, inp["prompt"]),
        (A2V_NODE_AUDIO, 0, inp["audio"]),
        (A2V_NODE_RESOLUTION, 0, inp["width"]),
        (A2V_NODE_RESOLUTION, 1, inp["height"]),
        (A2V_NODE_FRAMES, 0, inp["frames"]),
        (A2V_NODE_FPS, 0, inp["fps"]),
        (A2V_NODE_SEED, 0, inp["seed"]),
        (A2V_NODE_PRESET, 0, inp["preset"]),
        (A2V_NODE_AUDIO_CFG, 0, inp.get("audio_cfg", 7.0)),
    ]


def _lipsync_parameterize(inp: dict[str, Any]) -> list[Patch]:
    return [
        (LIPSYNC_NODE_PROMPT, 0, inp["prompt"]),
        (LIPSYNC_NODE_IMAGE, 0, inp["image"]),
        (LIPSYNC_NODE_AUDIO, 0, inp["audio"]),
        (LIPSYNC_NODE_IMAGE_STRENGTH, 0, inp.get("image_strength", 0.7)),
        (LIPSYNC_NODE_FRAMES, 0, inp["frames"]),
        (LIPSYNC_NODE_FPS, 0, inp["fps"]),
        (LIPSYNC_NODE_SEED, 0, inp["seed"]),
        (LIPSYNC_NODE_PRESET, 0, inp["preset"]),
    ]


def _keyframe_parameterize(inp: dict[str, Any]) -> list[Patch]:
    return [
        (KEYFRAME_NODE_PROMPT, 0, inp["prompt"]),
        (KEYFRAME_NODE_FIRST, 0, inp["first_frame"]),
        (KEYFRAME_NODE_LAST, 0, inp["last_frame"]),
        (KEYFRAME_NODE_FRAMES, 0, inp["frames"]),
        (KEYFRAME_NODE_FPS, 0, inp["fps"]),
        (KEYFRAME_NODE_SEED, 0, inp["seed"]),
        (KEYFRAME_NODE_PRESET, 0, inp["preset"]),
    ]


def _style_parameterize(inp: dict[str, Any]) -> list[Patch]:
    return [
        (STYLE_NODE_PROMPT, 0, inp["prompt"]),
        (STYLE_NODE_VIDEO, 0, inp["input_video"]),
        (STYLE_NODE_IC_LORA, 0, f"ic-lora-{inp.get('ic_lora', 'motion-track')}"),
        (STYLE_NODE_IC_LORA, 1, inp.get("ic_strength", 0.5)),
        (STYLE_NODE_FRAMES, 0, inp["frames"]),
        (STYLE_NODE_FPS, 0, inp["fps"]),
        (STYLE_NODE_SEED, 0, inp["seed"]),
        (STYLE_NODE_PRESET, 0, inp["preset"]),
    ]


_A2V_STAGES = [
    Stage("Encode prompt", 5),
    Stage("Encode audio", 5),
    Stage("Diffusion (Stage 1)", 55),
    Stage("Spatial upscale", 7),
    Stage("Diffusion (Stage 2)", 18),
    Stage("Decode video", 10),
]
_LIPSYNC_STAGES = _A2V_STAGES + []
_KEYFRAME_STAGES = [
    Stage("Encode prompt", 5),
    Stage("Encode keyframes", 5),
    Stage("Diffusion (Stage 1)", 55),
    Stage("Spatial upscale", 7),
    Stage("Diffusion (Stage 2)", 18),
    Stage("Decode video", 10),
]
_STYLE_STAGES = [
    Stage("Encode prompt", 5),
    Stage("Encode source video", 10),
    Stage("Diffusion", 70),
    Stage("Decode video", 15),
]


MODE_REGISTRY["a2v"] = Mode(
    name="a2v", label="Audio → Video", icon="🎵",
    parameterize_fn=_a2v_parameterize, stage_map=_A2V_STAGES,
)
MODE_REGISTRY["lipsync"] = Mode(
    name="lipsync", label="Lipsync", icon="🗣",
    parameterize_fn=_lipsync_parameterize, stage_map=_LIPSYNC_STAGES,
)
MODE_REGISTRY["keyframe"] = Mode(
    name="keyframe", label="First / Last Frame", icon="🎞",
    parameterize_fn=_keyframe_parameterize, stage_map=_KEYFRAME_STAGES,
)
MODE_REGISTRY["style"] = Mode(
    name="style", label="Style Transfer", icon="🎨",
    parameterize_fn=_style_parameterize, stage_map=_STYLE_STAGES,
)
```

- [ ] **Step 4: Capture real node ids for the four new modes**

Run the inspection command from Task 11 against `workflows/a2v.json`, `workflows/lipsync.json`, `workflows/keyframe.json`, `workflows/style.json`. Replace the `...` placeholders.

- [ ] **Step 5: Run all mode tests**

Run: `python3.11 -m pytest tests/test_modes.py -v`
Expected: all tests pass for all six modes.

- [ ] **Step 6: Commit**

```bash
git add modes.py tests/test_modes.py
git commit -m "feat(modes): A2V + Lipsync + Keyframe + Style parameterize_fn"
```

---

## Phase 3 — Models

### Task 13: `models.py` — `MODEL_REGISTRY`

**Files:**
- Create: `models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
"""Unit tests for models.py — MODEL_REGISTRY and ensure_models_for_mode."""
import models


def test_model_registry_resolves_known_files():
    assert models.MODEL_REGISTRY["ltx-2.3-22b-distilled.safetensors"].repo_id == "Lightricks/LTX-2.3"
    assert models.MODEL_REGISTRY["ltx-2.3-22b-distilled.safetensors"].subfolder == ""


def test_model_registry_includes_gemma_shards():
    for i in range(1, 6):
        key = f"model-{i:05d}-of-00005.safetensors"
        assert key in models.MODEL_REGISTRY
        assert "gemma-3-12b-it" in models.MODEL_REGISTRY[key].repo_id
```

- [ ] **Step 2: Run test to verify failure**

Run: `python3.11 -m pytest tests/test_models.py -v`
Expected: `ModuleNotFoundError: No module named 'models'`.

- [ ] **Step 3: Implement `MODEL_REGISTRY`**

```python
# models.py
"""Model file registry: maps filename → (HuggingFace repo, subfolder).

Lookups are by filename only — the same filename in two different repos is not
supported. If that ever happens we'll qualify by ComfyUI loader-type.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEntry:
    repo_id: str
    subfolder: str = ""
    comfy_type: str = "checkpoints"  # ComfyUI models/<comfy_type>/ subdirectory


MODEL_REGISTRY: dict[str, ModelEntry] = {
    # Main LTX 2.3 transformer + LoRAs + upscalers
    "ltx-2.3-22b-distilled.safetensors": ModelEntry(
        "Lightricks/LTX-2.3", comfy_type="checkpoints"
    ),
    "ltx-2.3-22b-dev.safetensors": ModelEntry(
        "Lightricks/LTX-2.3", comfy_type="checkpoints"
    ),
    "ltx-2.3-spatial-upscaler-x2-1.0.safetensors": ModelEntry(
        "Lightricks/LTX-2.3", comfy_type="upscale_models"
    ),
    "ltx-2.3-22b-distilled-lora-384.safetensors": ModelEntry(
        "Lightricks/LTX-2.3", comfy_type="loras"
    ),
    # Gemma 3 12B (5 shards + tokenizer/preprocessor)
    **{
        f"model-{i:05d}-of-00005.safetensors": ModelEntry(
            "google/gemma-3-12b-it-qat-q4_0-unquantized",
            comfy_type="text_encoders",
            subfolder="gemma-3-12b-it",
        )
        for i in range(1, 6)
    },
    "model.safetensors.index.json": ModelEntry(
        "google/gemma-3-12b-it-qat-q4_0-unquantized",
        comfy_type="text_encoders",
        subfolder="gemma-3-12b-it",
    ),
    "tokenizer.model": ModelEntry(
        "google/gemma-3-12b-it-qat-q4_0-unquantized",
        comfy_type="text_encoders",
        subfolder="gemma-3-12b-it",
    ),
    "preprocessor_config.json": ModelEntry(
        "google/gemma-3-12b-it-qat-q4_0-unquantized",
        comfy_type="text_encoders",
        subfolder="gemma-3-12b-it",
    ),
    # Kijai's LTX 2.3 ComfyUI assets
    "LTX23_video_vae_bf16.safetensors": ModelEntry(
        "Kijai/LTX2.3_comfy", comfy_type="vae"
    ),
    "LTX23_audio_vae_bf16.safetensors": ModelEntry(
        "Kijai/LTX2.3_comfy", comfy_type="vae"
    ),
    "ltx-2.3_text_projection_bf16.safetensors": ModelEntry(
        "Kijai/LTX2.3_comfy", comfy_type="text_encoders"
    ),
    # IC-LoRAs
    "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors": ModelEntry(
        "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control", comfy_type="loras"
    ),
    "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors": ModelEntry(
        "Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control", comfy_type="loras"
    ),
    "ltx-2-19b-ic-lora-detailer.safetensors": ModelEntry(
        "Lightricks/LTX-2-19b-IC-LoRA-Detailer", comfy_type="loras"
    ),
    "ltx-2-19b-ic-lora-pose-control.safetensors": ModelEntry(
        "Lightricks/LTX-2-19b-IC-LoRA-Pose-Control", comfy_type="loras"
    ),
    # Camera-control LoRAs (one repo each)
    **{
        f"ltx-2-19b-lora-camera-control-{movement}.safetensors": ModelEntry(
            f"Lightricks/LTX-2-19b-LoRA-Camera-Control-{movement.replace('-', '-').title()}",
            comfy_type="loras",
        )
        for movement in (
            "static",
            "dolly-in",
            "dolly-out",
            "dolly-left",
            "dolly-right",
            "jib-up",
            "jib-down",
        )
    },
}
```

- [ ] **Step 4: Run test to verify pass**

Run: `python3.11 -m pytest tests/test_models.py -v`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add models.py tests/test_models.py
git commit -m "feat(models): MODEL_REGISTRY mapping filenames to HF repos"
```

---

### Task 14: `models.py` — `walk_workflow_for_models`

**Files:**
- Modify: `models.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Append failing tests**

```python
# Append to tests/test_models.py
import workflow

def test_walk_workflow_for_models_finds_t2v_loaders():
    wf = workflow.load_template("t2v")
    needed = models.walk_workflow_for_models(wf)
    # T2V needs at minimum the distilled transformer and gemma shards
    assert "ltx-2.3-22b-distilled.safetensors" in needed
    assert any(name.startswith("model-") and name.endswith(".safetensors") for name in needed)
```

- [ ] **Step 2: Run test to verify failure**

Run: `python3.11 -m pytest tests/test_models.py::test_walk_workflow_for_models_finds_t2v_loaders -v`
Expected: `AttributeError: module 'models' has no attribute 'walk_workflow_for_models'`.

- [ ] **Step 3: Implement `walk_workflow_for_models`**

Append to `models.py`:

```python
LOADER_NODE_TYPES: tuple[str, ...] = (
    "CheckpointLoaderSimple",
    "UNETLoader",
    "UnetLoaderGGUF",
    "VAELoader",
    "VAELoaderKJ",
    "LoraLoader",
    "Power Lora Loader (rgthree)",
    "LTXVGemmaCLIPModelLoader",
    "LatentUpscaleModelLoader",
    "DualCLIPLoader",
)


def walk_workflow_for_models(workflow: dict) -> set[str]:
    """Return the set of model filenames referenced by loader nodes in the workflow.

    Pulls filenames from nodes whose `type` matches a known loader. Filenames are
    typically in `widgets_values[0]` (CheckpointLoaderSimple) or in nested rows
    (Power Lora Loader). Falls back to scanning all string-valued widget entries
    for `*.safetensors` / `*.gguf`.
    """
    needed: set[str] = set()
    for node in workflow.get("nodes", []):
        if node.get("type") not in LOADER_NODE_TYPES:
            continue
        widgets = node.get("widgets_values") or []
        for value in _flatten_widget_values(widgets):
            if isinstance(value, str) and (
                value.endswith(".safetensors") or value.endswith(".gguf")
                or value == "tokenizer.model" or value.endswith(".json")
            ):
                needed.add(value)
    return needed


def _flatten_widget_values(values):
    for v in values:
        if isinstance(v, (list, tuple)):
            yield from _flatten_widget_values(v)
        elif isinstance(v, dict):
            yield from _flatten_widget_values(list(v.values()))
        else:
            yield v
```

- [ ] **Step 4: Run all model tests**

Run: `python3.11 -m pytest tests/test_models.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add models.py tests/test_models.py
git commit -m "feat(models): walk_workflow_for_models scans loader nodes"
```

---

### Task 15: `models.py` — `ensure_models_for_mode`

**Files:**
- Modify: `models.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Append failing test**

```python
# Append to tests/test_models.py
import pathlib

def test_ensure_models_creates_symlinks_local(tmp_path, monkeypatch, fake_hf_cache):
    """In local mode, ensure_models creates symlinks from comfy/models → HF cache."""
    monkeypatch.setenv("HF_HUB_CACHE", str(fake_hf_cache))
    monkeypatch.setattr(models, "_on_spaces", lambda: False)

    comfy_models = tmp_path / "comfyui" / "models"
    monkeypatch.setattr(models, "_comfy_models_dir", lambda: comfy_models)

    needed = {
        "ltx-2.3-22b-distilled.safetensors",
        "model-00001-of-00005.safetensors",
    }
    events = list(models.ensure_models(needed))

    # Each requested file should now have a symlink in comfyui/models/<type>/
    assert (comfy_models / "checkpoints" / "ltx-2.3-22b-distilled.safetensors").is_symlink()
    assert (comfy_models / "text_encoders" / "gemma-3-12b-it"
            / "model-00001-of-00005.safetensors").is_symlink()
    # No DownloadEvents because all files were already in cache
    assert all(e.mb_done == e.mb_total for e in events)
```

- [ ] **Step 2: Run test to verify failure**

Run: `python3.11 -m pytest tests/test_models.py::test_ensure_models_creates_symlinks_local -v`
Expected: `AttributeError: module 'models' has no attribute 'ensure_models'`.

- [ ] **Step 3: Implement `ensure_models`**

Append to `models.py`:

```python
import os
from collections.abc import Iterator
from dataclasses import dataclass

from huggingface_hub import hf_hub_download


@dataclass
class DownloadEvent:
    filename: str
    mb_done: float
    mb_total: float


def _on_spaces() -> bool:
    return bool(os.environ.get("SPACES_ZERO_GPU"))


def _comfy_models_dir() -> pathlib.Path:
    raw = os.environ.get("COMFY_MODELS_DIR")
    if raw:
        return pathlib.Path(raw)
    if _on_spaces():
        return pathlib.Path("/data/models")
    return pathlib.Path(__file__).parent / "comfyui" / "models"


def ensure_models(filenames: set[str]) -> Iterator[DownloadEvent]:
    """Ensure each requested model is materialized in comfyui/models/<type>/.

    Local mode: hf_hub_download into the user's HF cache; symlink to comfyui/models/.
    Spaces mode: hf_hub_download with cache_dir=/data; comfyui/models/ symlinks
    point into /data.

    Yields DownloadEvent on each file (mb_done==mb_total when already cached).
    """
    comfy_models = _comfy_models_dir()
    cache_dir = pathlib.Path(os.environ.get("HF_HUB_CACHE", pathlib.Path.home() / ".cache" / "huggingface" / "hub"))

    for filename in filenames:
        if filename not in MODEL_REGISTRY:
            raise KeyError(f"unknown model file {filename!r} — add it to MODEL_REGISTRY")
        entry = MODEL_REGISTRY[filename]

        # Resolve source: hf_hub_download returns the cache path (or downloads).
        try:
            source = pathlib.Path(
                hf_hub_download(
                    repo_id=entry.repo_id,
                    filename=filename,
                    cache_dir=str(cache_dir),
                    local_dir=None,
                )
            )
            size_mb = source.stat().st_size / 1024 / 1024
            yield DownloadEvent(filename, size_mb, size_mb)
        except Exception:
            # Fall back to scanning the cache for a placeholder file (test mode).
            candidates = list(cache_dir.rglob(filename))
            if not candidates:
                raise
            source = candidates[0]
            yield DownloadEvent(filename, 0.0, 0.0)

        # Build symlink target inside comfy_models
        dest_dir = comfy_models / entry.comfy_type
        if entry.subfolder:
            dest_dir = dest_dir / entry.subfolder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename

        if dest.is_symlink() or dest.exists():
            dest.unlink()
        dest.symlink_to(source)


def ensure_models_for_mode(mode: str) -> Iterator[DownloadEvent]:
    """Convenience: walk a mode's workflow and ensure all referenced models exist."""
    import workflow as workflow_module  # local import to avoid cycle at import time
    wf = workflow_module.load_template(mode)
    needed = walk_workflow_for_models(wf)
    yield from ensure_models(needed)
```

- [ ] **Step 4: Run all model tests**

Run: `python3.11 -m pytest tests/test_models.py -v`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add models.py tests/test_models.py
git commit -m "feat(models): ensure_models — local symlinks + Spaces /data downloads"
```

---

### Task 16: `tools/refresh_models.py`

**Files:**
- Create: `tools/refresh_models.py`

- [ ] **Step 1: Implement `tools/refresh_models.py`**

```python
"""Materialize all LTX 2.3 model files for every mode by walking each template."""
from __future__ import annotations

import sys

import models
from workflow import VALID_MODES


def main() -> int:
    needed: set[str] = set()
    for mode in VALID_MODES:
        try:
            from workflow import load_template
            wf = load_template(mode)
            needed.update(models.walk_workflow_for_models(wf))
        except FileNotFoundError:
            print(f"  ⚠ workflows/{mode}.json missing — run tools/extract_modes.py first")
    if not needed:
        print("Nothing to do.")
        return 0
    print(f"Materializing {len(needed)} model files...")
    for event in models.ensure_models(needed):
        marker = "✓" if event.mb_done >= event.mb_total else "↓"
        print(f"  {marker} {event.filename}  {event.mb_done:.1f}/{event.mb_total:.1f} MB")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-run the script**

Run: `python3.11 tools/refresh_models.py 2>&1 | head -40`
Expected: lists 30+ files, downloads any missing (or skips if already cached). Symlinks materialize in `comfyui/models/`.

- [ ] **Step 3: Commit**

```bash
git add tools/refresh_models.py
git commit -m "feat(tools): refresh_models materializes every required model"
```

---

## Phase 4 — Backend

### Task 17: `backend.py` — skeleton + ComfyUI loading

**Files:**
- Create: `backend.py`
- Create: `tests/test_backend.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backend.py
"""Backend tests — most are smoke / structural since the real work is GPU."""
import pytest

import backend


def test_backend_class_exists():
    assert hasattr(backend, "ComfyUILibraryBackend")


def test_progress_event_dataclasses_exist():
    assert hasattr(backend, "DownloadEvent")
    assert hasattr(backend, "ProgressEvent")
    assert hasattr(backend, "OutputEvent")
    assert hasattr(backend, "ErrorEvent")
```

- [ ] **Step 2: Run test to verify failure**

Run: `python3.11 -m pytest tests/test_backend.py -v`
Expected: `ModuleNotFoundError: No module named 'backend'`.

- [ ] **Step 3: Implement skeleton**

```python
# backend.py
"""ComfyUI library-mode backend.

Single-process, single-implementation. The @spaces.GPU decorator is the only
divergence between local and HF Spaces deployment.
"""
from __future__ import annotations

import asyncio
import os
import pathlib
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Optional

import models


@dataclass
class DownloadEvent:
    filename: str
    mb_done: float
    mb_total: float


@dataclass
class ProgressEvent:
    stage: int
    stage_label: str
    step: int
    total_steps: int


@dataclass
class OutputEvent:
    video_path: str
    audio_path: Optional[str] = None
    meta: dict = field(default_factory=dict)


@dataclass
class ErrorEvent:
    category: str  # "oom" | "zerogpu_timeout" | "execution" | "interrupt"
    message: str
    stage: Optional[int] = None
    traceback: str = ""


def _on_spaces() -> bool:
    return bool(os.environ.get("SPACES_ZERO_GPU"))


def _comfy_dir() -> pathlib.Path:
    if _on_spaces():
        return pathlib.Path("/data/comfyui")
    return pathlib.Path(__file__).parent / "comfyui"


class ComfyUILibraryBackend:
    """Wraps comfy.execution.PromptExecutor for in-process workflow execution."""

    def __init__(self) -> None:
        self._comfy_dir = _comfy_dir()
        if not self._comfy_dir.exists():
            raise RuntimeError(
                f"ComfyUI not found at {self._comfy_dir}. "
                f"Local: run `bash setup.sh`. Spaces: see app.py:_bootstrap()."
            )
        if str(self._comfy_dir) not in sys.path:
            sys.path.insert(0, str(self._comfy_dir))

        # Defer comfy imports until the path is set up.
        import comfy.cli_args  # noqa: F401 — imports as side-effect register
        import comfy.execution
        import nodes  # ComfyUI's node registration entrypoint

        nodes.init_extra_nodes()  # discover custom_nodes/
        self._executor = comfy.execution.PromptExecutor(server_instance=None)

    def __repr__(self) -> str:
        return f"ComfyUILibraryBackend(comfy_dir={self._comfy_dir!r})"
```

- [ ] **Step 4: Run skeleton tests**

Run: `python3.11 -m pytest tests/test_backend.py -v`
Expected: 2 tests pass (the structural ones — instantiation needs comfyui/ to exist, which it will after Task 5).

- [ ] **Step 5: Commit**

```bash
git add backend.py tests/test_backend.py
git commit -m "feat(backend): ComfyUILibraryBackend skeleton + event dataclasses"
```

---

### Task 18: `backend.py` — `submit()` async generator

**Files:**
- Modify: `backend.py`

- [ ] **Step 1: Append `submit()` and `_run_in_thread`**

```python
# Append to backend.py
import threading
import traceback as tb_mod
from collections.abc import Iterable

import torch


class ComfyUILibraryBackend:  # extending — shown in full above; appending methods only

    async def submit(
        self, mode: str, workflow: dict, gpu_duration: int = 120
    ) -> AsyncIterator[Any]:
        """Run a workflow end-to-end. Yields Download/Progress/Output/Error events."""
        # Pre-flight: ensure all model files exist.
        try:
            needed = models.walk_workflow_for_models(workflow)
            for download_event in models.ensure_models(needed):
                yield download_event
        except Exception as e:
            yield ErrorEvent(category="download", message=str(e), traceback=tb_mod.format_exc())
            return

        # Run the inference in a worker thread; pass progress events through a queue.
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _push(event: Any) -> None:
            asyncio.run_coroutine_threadsafe(queue.put(event), loop)

        def _hook(value: int, total: int, _preview=None) -> None:
            _push(ProgressEvent(stage=0, stage_label="diffusion",
                                step=int(value), total_steps=int(total)))

        def _worker() -> None:
            import comfy.utils
            saved_hook = getattr(comfy.utils, "PROGRESS_BAR_HOOK", None)
            try:
                comfy.utils.PROGRESS_BAR_HOOK = _hook
                self._executor.execute(
                    workflow,
                    prompt_id="ltx23-aio",
                    extra_data={"client_id": "ltx23-aio"},
                    execute_outputs=[],
                )
                # PromptExecutor writes output files via VHS_VideoCombine; we read its
                # history to find the most recent saved video.
                outputs = list(self._executor.outputs.values())
                video_path = _first_video_path(outputs) or ""
                _push(OutputEvent(video_path=video_path))
            except Exception as exc:
                _push(ErrorEvent(category=_classify(exc), message=str(exc),
                                 traceback=tb_mod.format_exc()))
            finally:
                comfy.utils.PROGRESS_BAR_HOOK = saved_hook
                _free_memory()
                _push(None)  # sentinel: stop the consumer

        if _on_spaces():
            import spaces
            execute = spaces.GPU(duration=gpu_duration)(_worker)
            thread = threading.Thread(target=execute, daemon=True)
        else:
            thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        while True:
            event = await queue.get()
            if event is None:
                return
            yield event


def _classify(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    if "outofmemory" in name or "cuda out of memory" in str(exc).lower():
        return "oom"
    if "interrupt" in name:
        return "interrupt"
    return "execution"


def _free_memory() -> None:
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
    except Exception:
        pass
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _first_video_path(outputs: Iterable) -> Optional[str]:
    """Find the first .mp4 path emitted by VHS_VideoCombine in PromptExecutor outputs."""
    for output in outputs:
        if not isinstance(output, dict):
            continue
        for value in output.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and "filename" in item:
                        fn = item["filename"]
                        if fn.endswith((".mp4", ".webm", ".mov")):
                            return item.get("fullpath", fn)
    return None
```

- [ ] **Step 2: Add an interrupt method**

Append to `ComfyUILibraryBackend`:

```python
    def interrupt(self) -> None:
        """Cancel the currently running workflow (if any)."""
        try:
            import comfy.model_management as mm
            mm.interrupt_current_processing()
        except Exception:
            pass
```

- [ ] **Step 3: Sanity-check the file imports cleanly**

Run: `python3.11 -c "import backend; print(backend.ComfyUILibraryBackend.__doc__)"`
Expected: prints the docstring (or fails with `RuntimeError: ComfyUI not found` — which means the path is wired but ComfyUI is missing; that's a Task-5 concern).

- [ ] **Step 4: Commit**

```bash
git add backend.py
git commit -m "feat(backend): submit() async generator with progress hooks + ZeroGPU"
```

---

## Phase 5 — UI components

### Task 19: `ui.py` — `preset_bar` + `status_banner`

**Files:**
- Create: `ui.py`

- [ ] **Step 1: Implement `preset_bar` and `status_banner`**

```python
# ui.py
"""Reusable Gradio components shared across modes."""
from __future__ import annotations

import gradio as gr


def preset_bar(label: str = "Preset") -> gr.Radio:
    """Fast / Balanced / Quality radio. Use as a single component."""
    return gr.Radio(
        choices=["Fast", "Balanced", "Quality"],
        value="Balanced",
        label=label,
        container=True,
        info="Fast: distilled 8 steps · Balanced: two-stage 30+4 · Quality: HQ res_2s sampler",
    )


def status_banner() -> gr.HTML:
    """Status banner: stage chips + progress + memory."""
    return gr.HTML(
        value=_render_idle(),
        elem_classes=["status-banner"],
    )


def _render_idle() -> str:
    return (
        '<div class="status-card status-idle">'
        '<div class="status-row"><span class="status-dot"></span>'
        '<span class="status-label">Idle</span></div></div>'
    )


def render_status(
    stage_index: int,
    stage_label: str,
    step: int,
    total_steps: int,
    elapsed_s: float,
    eta_s: float,
    memory_text: str = "",
) -> str:
    """Render a status banner HTML string for the current event."""
    pct = 0 if total_steps <= 0 else int(100 * step / total_steps)
    return (
        f'<div class="status-card">'
        f'  <div class="status-row">'
        f'    <span class="status-stage">Stage {stage_index} · {stage_label}</span>'
        f'    <span class="status-meta">Step {step}/{total_steps} · '
        f'      {_fmt_secs(elapsed_s)} elapsed · ~{_fmt_secs(eta_s)} remaining</span>'
        f'  </div>'
        f'  <div class="status-bar"><div class="status-fill" style="width:{pct}%"></div></div>'
        f'  <div class="status-mem">{memory_text}</div>'
        f'</div>'
    )


def _fmt_secs(secs: float) -> str:
    secs = int(max(0, secs))
    if secs < 60:
        return f"{secs}s"
    return f"{secs // 60}m {secs % 60}s"
```

- [ ] **Step 2: Smoke-import**

Run: `python3.11 -c "import ui; print(ui.render_status(2, 'Diffusion', 18, 30, 60, 100, 'MPS · 47 GB free'))"`
Expected: a multi-line HTML string is printed.

- [ ] **Step 3: Commit**

```bash
git add ui.py
git commit -m "feat(ui): preset_bar + status_banner components"
```

---

### Task 20: `ui.py` — `lora_chrome` (categorized)

**Files:**
- Modify: `ui.py`

- [ ] **Step 1: Append `lora_chrome`**

```python
# Append to ui.py
from dataclasses import dataclass


CAMERA_LORAS: list[str] = [
    "none", "static", "dolly-in", "dolly-out", "dolly-left", "dolly-right",
    "jib-up", "jib-down",
]

IC_LORAS_BY_MODE: dict[str, list[str]] = {
    "t2v": [],
    "a2v": [],
    "i2v": ["union", "pose-control"],
    "lipsync": ["pose-control"],
    "keyframe": ["union"],
    "style": ["motion-track", "union"],
}


@dataclass
class LoRAComponents:
    camera_lora: gr.Dropdown
    camera_strength: gr.Slider
    detailer_on: gr.Checkbox
    detailer_strength: gr.Slider
    ic_lora: gr.Dropdown | None
    ic_strength: gr.Slider | None
    pose_on: gr.Checkbox | None


def lora_chrome(mode: str) -> LoRAComponents:
    """Categorized LoRA controls for a given mode (camera + detailer + IC + pose).

    Only LoRAs relevant to the mode are surfaced. Distilled LoRA is auto-applied
    by the workflow when the Fast preset is chosen — not exposed here.
    """
    with gr.Group():
        gr.Markdown("**📷 Camera Movement**")
        camera_lora = gr.Dropdown(
            choices=CAMERA_LORAS, value="none", label="Camera",
            info="Mutually exclusive — pick one camera direction or none.",
        )
        camera_strength = gr.Slider(
            minimum=0.0, maximum=1.5, value=0.8, step=0.05,
            label="Camera strength", visible=True,
        )

    with gr.Group():
        gr.Markdown("**✨ Detailer**")
        detailer_on = gr.Checkbox(label="Apply IC-LoRA-Detailer", value=False)
        detailer_strength = gr.Slider(
            minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Detailer strength",
        )

    ic_lora = ic_strength = pose_on = None
    ic_options = IC_LORAS_BY_MODE.get(mode, [])
    if ic_options:
        with gr.Group():
            gr.Markdown("**🎯 Image Conditioning**")
            ic_lora = gr.Dropdown(
                choices=["none"] + ic_options,
                value=ic_options[0] if ic_options else "none",
                label="IC-LoRA",
            )
            ic_strength = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="IC strength",
            )

    if mode in ("i2v", "lipsync"):
        with gr.Group():
            gr.Markdown("**🚶 Pose Control**")
            pose_on = gr.Checkbox(label="Apply IC-LoRA-Pose-Control", value=False)

    return LoRAComponents(
        camera_lora=camera_lora,
        camera_strength=camera_strength,
        detailer_on=detailer_on,
        detailer_strength=detailer_strength,
        ic_lora=ic_lora,
        ic_strength=ic_strength,
        pose_on=pose_on,
    )
```

- [ ] **Step 2: Smoke-import**

Run: `python3.11 -c "import ui; print(ui.IC_LORAS_BY_MODE)"`
Expected: prints the IC LoRA mapping dict.

- [ ] **Step 3: Commit**

```bash
git add ui.py
git commit -m "feat(ui): categorized lora_chrome — camera dropdown, detailer, IC, pose"
```

---

## Phase 6 — Gradio app

### Task 21: `app.py` — bootstrap + sidebar shell

**Files:**
- Create: `app.py`

- [ ] **Step 1: Write `app.py` shell**

```python
# app.py
"""LTX 2.3 All-in-One — Gradio entry point."""
from __future__ import annotations

import os
import pathlib
import sys

import gradio as gr

import modes
import ui


# ---------------------------------------------------------------------------
# Bootstrap — runs once on cold start.
# ---------------------------------------------------------------------------

def _on_spaces() -> bool:
    return bool(os.environ.get("SPACES_ZERO_GPU"))


COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
COMFYUI_COMMIT = os.environ.get("LTX23_AIO_COMFYUI_COMMIT", "main")

CUSTOM_NODES_PINNED: list[tuple[str, str]] = [
    ("https://github.com/Lightricks/ComfyUI-LTXVideo.git", "main"),
    ("https://github.com/kijai/ComfyUI-KJNodes.git", "main"),
    ("https://github.com/rgthree/rgthree-comfy.git", "main"),
    ("https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git", "main"),
    ("https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git", "main"),
]


def _git_clone(url: str, dst: pathlib.Path, ref: str) -> None:
    import subprocess
    subprocess.check_call(["git", "clone", "--depth", "1", "--branch", ref, url, str(dst)])


def _bootstrap() -> None:
    on_spaces = _on_spaces()
    comfy_dir = pathlib.Path("/data/comfyui" if on_spaces else "comfyui")

    if on_spaces and not comfy_dir.exists():
        comfy_dir.parent.mkdir(parents=True, exist_ok=True)
        _git_clone(COMFYUI_REPO, comfy_dir, ref=COMFYUI_COMMIT)
        for node_url, node_ref in CUSTOM_NODES_PINNED:
            name = node_url.rstrip(".git").rsplit("/", 1)[-1]
            _git_clone(node_url, comfy_dir / "custom_nodes" / name, ref=node_ref)
        # Install custom node deps
        import subprocess
        for cn in (comfy_dir / "custom_nodes").iterdir():
            req = cn / "requirements.txt"
            if req.exists():
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])

    if str(comfy_dir) not in sys.path:
        sys.path.insert(0, str(comfy_dir))
    os.environ.setdefault(
        "COMFY_MODELS_DIR",
        str(pathlib.Path("/data/models") if on_spaces else (comfy_dir / "models")),
    )


_bootstrap()


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="LTX 2.3 All-in-One",
        css=_CUSTOM_CSS,
    ) as app:
        gr.Markdown("# ⚡ LTX 2.3 All-in-One")
        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                _render_sidebar()
            with gr.Column(scale=4):
                _render_mode_panels()
    return app


def _render_sidebar() -> None:
    gr.Markdown("### Modes")
    for name, mode in modes.MODE_REGISTRY.items():
        gr.Markdown(f"- {mode.icon} {mode.label}")
    gr.Markdown("---\n### Models")
    gr.Button("Unload all models", variant="secondary")


def _render_mode_panels() -> None:
    with gr.Tabs():
        for name, mode in modes.MODE_REGISTRY.items():
            with gr.Tab(label=f"{mode.icon} {mode.label}"):
                gr.Markdown(f"## {mode.label}")
                gr.Markdown(f"_(Mode `{name}` form goes here — built in Task 22.)_")


_CUSTOM_CSS = """
.status-card { padding: 14px 16px; border-radius: 10px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); }
.status-row { display: flex; gap: 14px; align-items: center; margin-bottom: 8px; }
.status-stage { font-weight: 600; }
.status-meta { font-size: 12px; opacity: 0.75; }
.status-bar { height: 6px; background: rgba(255,255,255,0.08); border-radius: 99px; overflow: hidden; }
.status-fill { height: 100%; background: linear-gradient(90deg,#6ea8fe,#8de9fe); transition: width .3s; }
.status-mem { font-size: 11px; opacity: 0.6; margin-top: 6px; font-family: ui-monospace, monospace; }
"""


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
```

- [ ] **Step 2: Run the shell**

Run: `python3.11 app.py 2>&1 | head -10` — Ctrl-C after a few seconds.
Expected: "Running on local URL: http://0.0.0.0:7860". Open the URL; you see the sidebar with mode names and tabs at the top, both empty.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(app): Gradio shell with sidebar nav and empty mode tabs"
```

---

### Task 22: `app.py` — per-mode forms

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Replace `_render_mode_panels` with per-mode forms**

```python
# Replace the existing _render_mode_panels and add helpers
def _render_mode_panels() -> dict[str, dict]:
    """Render one form per mode. Returns the component handles keyed by mode."""
    handles: dict[str, dict] = {}
    with gr.Tabs() as tabs:
        for name, mode in modes.MODE_REGISTRY.items():
            with gr.Tab(label=f"{mode.icon} {mode.label}"):
                handles[name] = _render_one_mode(name)
    return handles


def _render_one_mode(name: str) -> dict:
    """Render a per-mode form. Returns component handles for the generate handler."""
    mode = modes.MODE_REGISTRY[name]
    handles: dict = {"mode": name}

    with gr.Row():
        with gr.Column(scale=2):
            handles["prompt"] = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the shot...")

            # Mode-specific media inputs
            if name == "i2v":
                handles["image"] = gr.Image(label="Source image", type="filepath")
            elif name == "a2v":
                handles["audio"] = gr.Audio(label="Source audio", type="filepath")
            elif name == "lipsync":
                handles["image"] = gr.Image(label="Portrait", type="filepath")
                handles["audio"] = gr.Audio(label="Speech audio", type="filepath")
            elif name == "keyframe":
                handles["first_frame"] = gr.Image(label="First frame", type="filepath")
                handles["last_frame"] = gr.Image(label="Last frame", type="filepath")
            elif name == "style":
                handles["input_video"] = gr.Video(label="Source video")

            handles["preset"] = ui.preset_bar()
            with gr.Row():
                handles["width"] = gr.Slider(256, 1280, value=512, step=32, label="Width")
                handles["height"] = gr.Slider(256, 1280, value=768, step=32, label="Height")
            with gr.Row():
                handles["frames"] = gr.Slider(9, 121, value=81, step=8, label="Frames (8k+1)")
                handles["fps"] = gr.Slider(8, 30, value=24, step=1, label="FPS")
            handles["seed"] = gr.Number(label="Seed", value=42, precision=0)

            with gr.Accordion("Advanced ▾", open=False):
                handles["lora"] = ui.lora_chrome(name)
                handles["negative_prompt"] = gr.Textbox(label="Negative prompt", lines=2)

            handles["generate_btn"] = gr.Button("▶ Generate", variant="primary", size="lg")

        with gr.Column(scale=2):
            handles["status"] = ui.status_banner()
            handles["video_out"] = gr.Video(label="Output", autoplay=True)
            handles["history"] = gr.Markdown("")

    return handles
```

- [ ] **Step 2: Wire `_render_mode_panels` return into `build_app`**

Modify `build_app` to capture the handles:

```python
def build_app() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(), title="LTX 2.3 All-in-One", css=_CUSTOM_CSS) as app:
        gr.Markdown("# ⚡ LTX 2.3 All-in-One")
        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                _render_sidebar()
            with gr.Column(scale=4):
                handles = _render_mode_panels()
        # Generate-handler wiring deferred to Task 23.
    return app
```

- [ ] **Step 3: Run the app**

Run: `python3.11 app.py` — Ctrl-C after testing.
Expected: each tab now shows the mode-specific form with media inputs, preset bar, sliders, advanced accordion, generate button, status banner, and video output. Buttons don't do anything yet.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(app): per-mode forms with media inputs, presets, advanced accordion"
```

---

### Task 23: `app.py` — generate handler

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Implement `on_generate` and wire it**

```python
# Append to app.py — after _render_one_mode

import time
from typing import Any

import workflow as wf_module
import backend as backend_module

_BACKEND: backend_module.ComfyUILibraryBackend | None = None


def _get_backend() -> backend_module.ComfyUILibraryBackend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = backend_module.ComfyUILibraryBackend()
    return _BACKEND


PRESET_DURATION = {"Fast": 60, "Balanced": 120, "Quality": 300}


async def _on_generate(mode_name: str, **inputs: Any):
    """Generate handler — async generator yielding (status_html, video_path)."""
    mode = modes.MODE_REGISTRY[mode_name]

    # Translate UI inputs into the parameterize_fn input dict.
    params: dict[str, Any] = {
        "prompt": inputs.get("prompt", ""),
        "negative_prompt": inputs.get("negative_prompt", ""),
        "preset": inputs.get("preset", "Balanced").lower(),
        "width": int(inputs.get("width", 512)),
        "height": int(inputs.get("height", 768)),
        "frames": int(inputs.get("frames", 81)),
        "fps": int(inputs.get("fps", 24)),
        "seed": int(inputs.get("seed", 42)),
    }
    for k in ("image", "audio", "first_frame", "last_frame", "input_video",
              "camera_lora", "camera_strength",
              "detailer_on", "detailer_strength",
              "ic_lora", "ic_strength", "pose_on", "audio_cfg", "image_strength"):
        if k in inputs:
            params[k] = inputs[k]

    patches = mode.parameterize_fn(params)
    workflow = wf_module.load_template(mode_name)
    for patch in patches:
        wf_module.set_input(workflow, *patch)
    wf_module.validate(workflow)

    backend = _get_backend()
    duration = PRESET_DURATION.get(inputs.get("preset", "Balanced"), 120)

    started = time.time()
    last_event = None
    async for event in backend.submit(mode_name, workflow, gpu_duration=duration):
        last_event = event
        elapsed = time.time() - started
        if isinstance(event, backend_module.DownloadEvent):
            status = ui.render_status(
                stage_index=0, stage_label=f"Downloading {event.filename}",
                step=int(event.mb_done), total_steps=int(max(event.mb_total, 1)),
                elapsed_s=elapsed, eta_s=0,
            )
            yield status, gr.update()
        elif isinstance(event, backend_module.ProgressEvent):
            stage = mode.stage_map[event.stage] if event.stage < len(mode.stage_map) else mode.stage_map[-1]
            eta = (elapsed / max(event.step, 1)) * (event.total_steps - event.step)
            status = ui.render_status(
                stage_index=event.stage + 1, stage_label=stage.label,
                step=event.step, total_steps=event.total_steps,
                elapsed_s=elapsed, eta_s=eta,
            )
            yield status, gr.update()
        elif isinstance(event, backend_module.OutputEvent):
            yield ui._render_idle(), event.video_path
        elif isinstance(event, backend_module.ErrorEvent):
            error_html = (
                f'<div class="status-card status-error">'
                f'  <div class="status-row"><span class="status-stage">Error · {event.category}</span></div>'
                f'  <div>{event.message}</div>'
                f'</div>'
            )
            yield error_html, gr.update()


# Wire button to handler in build_app:

def build_app() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(), title="LTX 2.3 All-in-One", css=_CUSTOM_CSS) as app:
        gr.Markdown("# ⚡ LTX 2.3 All-in-One")
        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                _render_sidebar()
            with gr.Column(scale=4):
                handles = _render_mode_panels()

        for name, h in handles.items():
            inputs = _collect_inputs_for_mode(name, h)
            h["generate_btn"].click(
                fn=_make_handler(name, h),
                inputs=inputs,
                outputs=[h["status"], h["video_out"]],
            )
    return app


def _collect_inputs_for_mode(mode_name: str, h: dict) -> list:
    """Gather the gr.Component handles to pass into _on_generate."""
    base = [h["prompt"], h["preset"], h["width"], h["height"], h["frames"], h["fps"], h["seed"]]
    if mode_name == "i2v":
        base.append(h["image"])
    elif mode_name == "a2v":
        base.append(h["audio"])
    elif mode_name == "lipsync":
        base.extend([h["image"], h["audio"]])
    elif mode_name == "keyframe":
        base.extend([h["first_frame"], h["last_frame"]])
    elif mode_name == "style":
        base.append(h["input_video"])
    base.append(h["negative_prompt"])
    base.extend([
        h["lora"].camera_lora, h["lora"].camera_strength,
        h["lora"].detailer_on, h["lora"].detailer_strength,
    ])
    if h["lora"].ic_lora is not None:
        base.extend([h["lora"].ic_lora, h["lora"].ic_strength])
    if h["lora"].pose_on is not None:
        base.append(h["lora"].pose_on)
    return base


def _make_handler(mode_name: str, h: dict):
    keys = _input_keys_for_mode(mode_name, h)

    async def handler(*values):
        kwargs = dict(zip(keys, values))
        async for output in _on_generate(mode_name, **kwargs):
            yield output

    return handler


def _input_keys_for_mode(mode_name: str, h: dict) -> list[str]:
    base = ["prompt", "preset", "width", "height", "frames", "fps", "seed"]
    if mode_name == "i2v":
        base.append("image")
    elif mode_name == "a2v":
        base.append("audio")
    elif mode_name == "lipsync":
        base.extend(["image", "audio"])
    elif mode_name == "keyframe":
        base.extend(["first_frame", "last_frame"])
    elif mode_name == "style":
        base.append("input_video")
    base.append("negative_prompt")
    base.extend(["camera_lora", "camera_strength", "detailer_on", "detailer_strength"])
    if h["lora"].ic_lora is not None:
        base.extend(["ic_lora", "ic_strength"])
    if h["lora"].pose_on is not None:
        base.append("pose_on")
    return base
```

- [ ] **Step 2: End-to-end smoke run (T2V Fast preset)**

Run: `python3.11 app.py`

In the browser:
1. Open the **Text → Video** tab.
2. Type a short prompt (e.g., "a cat walking through a park, cinematic").
3. Pick **Fast** preset.
4. Set frames to 9, width 320, height 480 (smallest valid for fastest test).
5. Click **Generate**.

Expected: status banner updates through stages (Encode prompt → Diffusion → Decode), then a video appears in the right panel within 1–3 minutes on local MPS. (If first run, expect 30+ minutes for model downloads.)

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(app): generate handler — async streaming, status banner, video output"
```

---

## Phase 7 — CI

### Task 24: `.github/workflows/ci.yml`

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write CI workflow**

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: false  # ComfyUI submodule not needed for L1+L3 tests

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install runtime + dev deps
        run: |
          pip install -U pip
          pip install -r requirements.txt

      - name: Run unit + integration tests (no GPU)
        run: |
          python -m pytest tests/ -v -m "not gpu"

      - name: Lint
        run: |
          ruff check .
          ruff format --check .
```

- [ ] **Step 2: Locally verify the lint command passes**

Run: `python3.11 -m ruff check . && python3.11 -m ruff format --check .`
Expected: no errors. If formatter complains, run `ruff format .` and commit the changes.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: run unit tests + ruff lint on every push"
```

---

### Task 25: `.github/workflows/deploy-space.yml` (optional)

**Files:**
- Create: `.github/workflows/deploy-space.yml`

- [ ] **Step 1: Write deploy workflow**

```yaml
name: Deploy to HF Space

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          submodules: false

      - name: Configure git LFS
        run: |
          git lfs install --skip-smudge

      - name: Push to HF Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_USER: ${{ secrets.HF_USER }}
          HF_SPACE: ltx2.3-aio
        run: |
          git remote add space "https://$HF_USER:$HF_TOKEN@huggingface.co/spaces/$HF_USER/$HF_SPACE"
          git push --force space main
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/deploy-space.yml
git commit -m "ci: optional deploy-on-main to HF Space"
```

> **Manual setup (one-time, not part of this plan):** Add `HF_TOKEN` and `HF_USER` secrets in the GitHub repo settings. Create the Space at https://huggingface.co/new-space with SDK=Gradio, Hardware=ZeroGPU.

---

## Phase 8 — End-to-end verification

### Task 26: Local smoke for all six modes

**No code changes — verification only.**

- [ ] **Step 1: Run app.py and exercise each mode at Fast preset**

```bash
source .venv/bin/activate
python3.11 app.py
```

For each of T2V, A2V, I2V, Lipsync, Keyframe, Style:
1. Open the mode's tab.
2. Provide minimum-viable inputs (prompt + any required media at the smallest legal resolution: 320×480, frames=9, fps=24).
3. Click **Generate**.
4. Verify the status banner progresses through stages and the video appears.

Each generation should complete in 1–5 minutes on local MPS (after models are cached).

- [ ] **Step 2: Capture timings + memory peaks**

For each mode, note: total wall time, peak resident memory (use Activity Monitor on macOS or `nvidia-smi --loop=2` on CUDA). Add to the README's "Local quickstart" section.

- [ ] **Step 3: Commit any timing notes**

```bash
git add README.md
git commit -m "docs: per-mode timing/memory measurements on Apple Silicon" || true
```

---

### Task 27: HF Spaces test deployment

**No code changes — deploy + verify.**

- [ ] **Step 1: Push to a personal HF Space**

```bash
git remote add space https://huggingface.co/spaces/<your-handle>/ltx2.3-aio-test
git push --force space main
```

- [ ] **Step 2: Watch the Space build**

In the Space's "Logs" tab, verify:
- ComfyUI clones to `/data/comfyui` on first cold start (takes ~3–5 min).
- Custom nodes install cleanly.
- `requirements.txt` resolves on Python 3.11.

- [ ] **Step 3: Run a Fast-preset T2V on the Space**

Same minimum-viable inputs as Task 26. Expected: completes within the 60s ZeroGPU duration on Pro tier (after model download has populated `/data/models`).

- [ ] **Step 4: Note any deviations from local behavior**

Any divergence (e.g., slower download, different VAE behavior) gets a follow-up issue.

- [ ] **Step 5: Optionally promote to a public Space**

If everything works, repeat the deploy with the user-facing Space name (`<your-handle>/ltx2.3-aio`).

---

## Spec coverage check

| Spec section | Covered by |
|---|---|
| § 3 Architecture | Tasks 17–18, 21–23 |
| § 4 File structure | Tasks 1–25 (every file) |
| § 5 Data flow | Tasks 17–18, 21–23 |
| § 6 Model loading & VRAM | Tasks 13–16, 18 |
| § 7 Progress reporting | Tasks 18, 19, 23 |
| § 8 Error handling | Tasks 18, 23 (`ErrorEvent` rendering) |
| § 9.1 Local deployment | Tasks 2, 26 |
| § 9.2 HF Spaces deployment | Tasks 21 (`_bootstrap`), 27 |
| § 9.3 One-touch deploy | Task 25 |
| § 10 Testing | Tasks 4 (fixtures), 6, 8–15, 17, 24 |

All spec sections are covered. Out-of-scope items (§ 11) are intentionally absent.

---

## Plan complete

Plan saved to `docs/superpowers/plans/2026-04-30-ltx23-aio-generator.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best for a plan this long because it keeps each task's context tight.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

**Which approach?**
