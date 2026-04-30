"""ComfyUI library-mode backend.

Single-process, single-implementation. The @spaces.GPU decorator is the only
divergence between local and HF Spaces deployment.
"""
from __future__ import annotations

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
    category: str  # "oom" | "zerogpu_timeout" | "execution" | "interrupt" | "download"
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
    """Wraps PromptExecutor for in-process workflow execution."""

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
        # NOTE: ComfyUI ships PromptExecutor in the top-level `execution.py`
        # module, NOT under `comfy.execution`. Same for `nodes`. Both must be
        # imported AFTER the sys.path insert above.
        import asyncio

        import comfy.cli_args  # noqa: F401 — side-effect: registers CLI flags
        import execution  # top-level module — provides PromptExecutor
        import nodes  # top-level module — provides init_extra_nodes (async)

        # init_extra_nodes is an async function in modern ComfyUI; run it once.
        asyncio.run(nodes.init_extra_nodes())  # discover custom_nodes/
        self._executor = execution.PromptExecutor(server_instance=None)

    def __repr__(self) -> str:
        return f"ComfyUILibraryBackend(comfy_dir={self._comfy_dir!r})"
