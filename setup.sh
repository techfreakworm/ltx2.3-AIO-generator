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
